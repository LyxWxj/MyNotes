---
type: Note
status: Active
related_to:
  - "[[00-moe-inference-research-index]]"
  - "[[01-llm-moe-inference-systems]]"
  - "[[DIT-Offloading-技术方案]]"
  - "[[13-LingBot-Video]]"
---

# 固定长度视频生成的 MoE 架构与推理优化

## 1. 什么是“固定长度视频”

本文把一次请求中整段 latent video 同时进入双向/全序列 DiT，并经过若干去噪步得到完整视频的模型称为固定长度视频生成。实际 API 可以允许不同帧数，但每次采样仍先确定一个完整时空网格；它与逐帧/逐 chunk 向未来滚动的 causal streaming 模型不同。

典型执行流：

```text
prompt / image
  -> text/image encoder
  -> 初始化完整 noisy latent [B, C, F, H, W]
  -> for each denoising timestep:
       full/bidirectional video DiT
       scheduler update
  -> VAE decode 整段 latent
  -> video
```

端到端时间粗略写为：

$$
T_{\mathrm{E2E}} = T_{\mathrm{condition}}
+ S\left(T_{\mathrm{attention}}+T_{\mathrm{MoE}}+T_{\mathrm{other}}\right)
+ T_{\mathrm{VAE}},
$$

其中 $S$ 是去噪步数。任何只优化 $T_{\mathrm{MoE}}$ 的方案，其上限都受 Attention、采样步数和 VAE 的占比约束。

## 2. 视频中的两种主要 MoE

### 2.1 Token-choice sparse MoE

LingBot-Video 在 Transformer block 的 FFN 位置使用 shared + routed experts：

$$
m(u_t)=\sum_i E_i^{(s)}(u_t)
+\sum_{j\in\mathcal{R}_k(u_t)}g_{t,j}E_j^{(r)}(u_t).
$$

每个视觉或条件 token 独立选 Top-k。其运行时与 LLM MoE 相似：router、pack、可选 EP dispatch、grouped GEMM、combine、restore；但 token 规模完全不同。

[LingBot-Video](https://arxiv.org/abs/2607.07675) 报告默认 128 experts、Top-8，并在固定 active parameter budget 下比较 64/128/256 experts；论文选择 128 是因为从 128 增到 256 的质量收益已经变小，而通信和存储继续增加。现有更完整的模型结构笔记见 [[13-LingBot-Video]]。

#### 视频 token 数为何改变 kernel 区间

若 VAE 时间压缩为 $c_t$、空间压缩为 $c_h,c_w$，patch size 为 $p_t,p_h,p_w$，视频 token 数近似：

$$
T_v = \left\lceil\frac{F}{c_t p_t}\right\rceil
\left\lceil\frac{H}{c_h p_h}\right\rceil
\left\lceil\frac{W}{c_w p_w}\right\rceil.
$$

以 81 帧、720p、时间压缩约 4、空间压缩约 8、latent patch 为 $1\times2\times2$ 的量级估算，$T_v$ 已可达到约 7.5 万；Top-8 产生约 60 万 expert assignments/层/步。具体值随 VAE 和 patch 配置变化，但数量级足以说明：

- 这不是 LLM batch=1 decode 的“小 MoE”；
- grouped GEMM 通常有足够大的 M 维；
- EP dispatch payload 巨大，带宽和拓扑重要；
- 把所有 experts 当 dense GEMM 计算的 LLM 小批技巧不适用；
- 相邻 patch 的路由相关性会让 assignment 在少数 experts 上成片聚集。

### 2.2 Timestep/stage MoE

Wan2.2 A14B 使用两套约 14B 的 DiT：

- high-noise expert：早期高噪阶段，偏全局布局与运动；
- low-noise expert：后期低噪阶段，偏细节和纹理。

路由只依赖去噪时间步/SNR 边界：

```text
early denoising steps -> high_noise_model
boundary
late denoising steps  -> low_noise_model
```

官方说明为约 27B total、每步约 14B active。[官方仓库](https://github.com/Wan-Video/Wan2.2) 和 [Diffusers WanPipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan.py) 都把它实现成 `transformer` 与 `transformer_2` 两个模块，而不是一个 token router。

#### Active compute 不等于 resident memory

若两套 BF16 权重同时在 GPU：

$$
M_{\mathrm{weights}} \approx M_{\mathrm{high}}+M_{\mathrm{low}},
$$

并不会因为当前只运行一套而自动减半。Wan 官方 `image2video.py` 的 `_prepare_model_for_timestep` 会在 `offload_model=True` 时把非活跃模型移到 CPU，并把当前模型移入 GPU；这才把峰值 GPU 权重接近一套专家，代价是在阶段切换处搬运大块权重。

阶段 MoE 的优势是完全可预测：整个请求只在边界附近切换一次，适合 phase-level offload/prefetch；缺点是 expert 粒度极粗，不能像 token MoE 那样把单层的 experts 均匀分摊到许多 GPU。

## 3. 与 LLM 不同的推理挑战

### 3.1 同一网格重复执行，router 分布随噪声变化

视频 DiT 会让相同 patch 坐标经历多次去噪。早期 token 近似噪声，后期出现结构和语义，因此 token-choice router 的 expert histogram 可能沿 timestep 明显漂移。只按整段请求的平均 expert 负载做 EPLB 会掩盖某些时间段的尖峰。

应记录二维统计：

$$
L[\mathrm{layer},\mathrm{timestep},\mathrm{expert}],
$$

必要时再按 task、resolution、modality 分桶，而不是只记录 `layer × expert`。

### 3.2 空间/时间局部相关导致“成片热点”

语言 batch 中不同请求的 token 往往较分散；视频相邻 patch、相邻帧和同一物体区域高度相关，router 可能把大片 token 送往同一 expert。平均利用率尚可时，单个 rank 仍可能因一个运动区域或模态而过热。

这会同时造成：

- EP rank straggler；
- pack buffer 瞬时增大；
- Ulysses SP 与 EP 都使用 All-to-All 时争抢互联；
- padding-based grouped GEMM 的浪费随路由倾斜增加。

### 3.3 Attention 可能比 MoE 更早成为瓶颈

在全注意力实现中，视频 self-attention 随 token 数近似 $O(T_v^2)$，FFN/MoE 随 token 数近似 $O(T_vk)$。即使 sparse MoE 已把 active FFN FLOPs 降低，长序列 Attention 仍可能支配时间和 activation memory；窗口化、分解式或稀疏 Attention 会改变这一复杂度，但不会消除视频 token 规模带来的压力。

因此优化前必须 profile：

- Attention、MoE、norm/modulation、scheduler、VAE 的占比；
- SP collective 和 EP collective 是否串行或互相干扰；
- CFG 是否把整个 DiT 工作翻倍；
- 去噪步数是否远大于模型质量实际需要。

### 3.4 多种并行轴相互约束

视频常同时需要：

- SP/CP：切长时空序列，解决 Attention activation；
- EP：切 routed experts；
- TP/HSDP/PP：切大模型权重；
- CFG parallel：拆正负条件分支；
- DP：并行不同视频请求；
- VAE parallel：处理高分辨率 decode 峰值。

vLLM-Omni 当前把 diffusion EP size 定义为 `TP × SP × CFG × DP`，并在 pipeline stage 内组成 EP group，详见 [Expert Parallel guide](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/diffusion/parallelism/expert_parallel/)。这能复用所有 ranks，但也意味着 SP/CFG/DP 的扩展会扩大 expert collective domain；若跨节点拓扑不合适，增加并行度可能减速。

### 3.5 量化敏感度依赖 timestep

扩散早/晚阶段的 activation 分布和误差容忍度不同；视频还要求跨帧一致，局部量化误差可能表现为闪烁而非单个 token 的轻微概率变化。[ViDiT-Q](https://arxiv.org/abs/2406.02540) 采用 timestep-aware mixed precision；[6Bit-Diffusion](https://arxiv.org/abs/2603.18742) 在 NVFP4/INT8 间动态选择并结合 temporal delta cache，报告 1.92x 加速和 3.32x memory reduction。

对于 token-MoE，router、norm、timestep modulation 与 expert 权重可以使用不同精度；router 排名翻转会改变离散执行路径，通常比单个 GEMM 的数值误差更敏感。

### 3.6 Batching 不只要求 shape 相同

LLM continuous batching 可以每轮把所有活跃序列各取一个 token。视频请求若要合批，至少需要兼容：

- 分辨率、帧数、latent shape；
- 当前 timestep/scheduler state；
- CFG 模式；
- 当前 stage expert（Wan high/low）；
- condition 类型与长度；
- cache 策略和是否命中。

对 Wan2.2，把处于 high-noise 和 low-noise 的请求直接放入同一个 model batch 不成立；调度器需要按 stage bucket 排队，或让两套专家各自形成服务阶段。

## 4. 降低延迟：先优化乘法器 $S$

### 4.1 第一优先级：few-step distillation 与 CFG distillation

把 40--50 步降到 4--8 步会同时减少 Attention、MoE、通信和调度器开销，通常比单独优化某个 MoE kernel 更有端到端价值。[LightX2V](https://github.com/ModelTC/LightX2V) 为 Wan2.2 提供 4-step distillation/LoRA 路径；[Mamoda2.5](https://arxiv.org/abs/2605.02641) 也把 DiT-MoE 与 few-step distillation 组合。

若能把 CFG 蒸馏进模型，既减少一倍正/负分支计算，也避免两个分支出现不同 router histogram。不能取消 CFG 时，可用两卡 CFG parallel；vLLM-Omni 文档给出的典型加速约为 1.8x，而不是理想 2x，因为仍需同步和合并。

### 4.2 第二优先级：跨 timestep cache

[TeaCache](https://arxiv.org/abs/2411.19108) 用 timestep-modulated input difference 估计输出变化，选择何时重用 block residual；论文在 Open-Sora-Plan 上报告最高 4.41x、VBench 仅轻微变化。Cache-DiT 已提供 Wan2.2 支持和多种 block cache，见 [官方仓库](https://github.com/vipshop/cache-dit)。

MoE 使用 cache 时要注意：

- 缓存整个 block residual 可以连 router + expert 一起跳过；
- 只缓存某个 expert 输出时必须验证当前 router assignment 未变；
- Wan2.2 high/low expert 的阶段边界必须视为 cache invalidation boundary；
- cache 命中使不同请求/时间步的实际计算量不均，分布式 ranks 必须一致决定是否跳过，否则 collective 次序会错位。

### 4.3 第三优先级：并行 Attention 和 CFG

[vLLM-Omni SP](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/diffusion/parallelism/sequence_parallel/) 报告在大图/视频上约 1.5x--3.6x 加速，支持 Ulysses、Ring 或混合。Ulysses 依赖高带宽 All-to-All；Ring 更容易重叠通信，但短序列有循环开销。

对 token-MoE，要避免把 SP All-to-All 与 EP All-to-All 都放在同一慢速跨节点链路上。常见拓扑原则是：

- 节点内 NVLink domain 做 Ulysses/EP 中通信最重的一轴；
- 跨节点选择更可重叠、消息更规则的一轴；
- 让 DP/CFG 使用相对独立的通信组；
- 实测不同 mesh，而不是只按公式把 GPU 均分。

### 4.4 第四优先级：MoE kernel

视频 token-choice MoE 适合：

- fused TopK + pack/unpack；
- contiguous grouped GEMM；
- FP8 expert GEMM 和低精度 dispatch；
- 根据真实 expert histogram 选择 tile；
- shared/routed expert overlap；
- CUDA Graph/static shape buckets。

由于 token 量大，首要目标是减少 pack memory traffic、padding 和跨 rank 字节，而非套用 LLM 小批 dense GEMM。

## 5. 显存受限

### 5.1 Timestep MoE：phase offload 是天然解

Wan2.2 的访问序列完全确定：先连续使用 high expert，再连续使用 low expert。因此最合理的低显存层次是：

```text
GPU: 当前 expert + activation
CPU pinned memory: 下一 expert
disk: checkpoint / 冷启动副本
```

优化要点：

1. high 阶段开始时常驻 high expert；
2. 在最后若干 high steps 中异步预取 low expert；
3. 边界处只做一次同步切换；
4. low expert 使用期间不再反复换入；
5. 请求结束后再决定保留哪套权重，依据队列中的 stage 分布。

LightX2V 明确支持 `phase` 或 `block` offload；Wan 官方实现支持模型级切换。若 PCIe 带宽不足以隐藏整套 14B 权重传输，phase offload 会降低显存但明显增加单请求延迟。

### 5.2 Token-choice MoE：EP/HSDP 优先于逐 token offload

视觉 token 几乎可能覆盖大量 experts，逐层把 Top-k experts 从 CPU 搬入 GPU 往往会激活大部分专家池，失去稀疏 offload 收益。优先级通常是：

1. expert weight FP8/FP4 quantization；
2. EP 分片 routed experts；
3. HSDP/FSDP2 或 pipeline parallel 分片 dense/shared 部分；
4. SP 降 activation memory；
5. 只在 trace 证明 expert reuse 强时做 hot/cold cache/offload。

LingBot-Video 官方实现提供 PyTorch grouped MM、SGLang Triton 和 FP8 expert backend；vLLM-Omni 的 diffusion EP 当前提供基础 AllGather/ReduceScatter 路径，但官方文档注明尚未集成主 vLLM 的多后端 All-to-All 和 EPLB 等高级能力，因此不能假设 LLM 的所有 EP 特性已自动可用。

### 5.3 别忘了非 MoE 显存

- full attention activation/KV 临时张量；
- VAE encode/decode 峰值；
- text/image encoder；
- CFG 双分支 activation；
- MoE pack、expert padding 和 collective buffer；
- cache residual。

可以组合 SP、VAE tiling/patch parallel、module/block offload 和量化。vLLM-Omni 的 [parallelism overview](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/diffusion/parallelism/overview/) 已分别提供 SP、PP、HSDP、EP、CFG 和 VAE parallel 的入口。

## 6. 提高吞吐

### 6.1 请求级 DP 和 shape bucket

吞吐优先时，先按 `(model, shape, timestep, stage, CFG mode)` 分桶，再在桶内 batch。相同 prompt 产生多样本时可复用 condition embedding；不同 prompt 只要 condition padding 和模型支持也可合批。

batch 增大对 token-MoE 有双重效果：

- 专家 GEMM 更大、效率提高；
- 更多请求可能覆盖更多 experts，offload 工作集扩大。

因此 GPU-resident/EP 部署通常受益，offload 部署则存在最佳 batch，不能无限增大。

### 6.2 把 Wan 两个 experts 变成服务流水

**工程推导：** 多请求场景可以把 high-noise 与 low-noise experts 放在两组设备：

```text
request batch A: high expert -> latent handoff -> low expert
request batch B:             high expert -> latent handoff -> low expert
```

优势：

- 两套权重各自常驻，不做 CPU-GPU 大搬运；
- 两阶段可以并行处理不同请求；
- high/low 设备数可按各阶段实际 step 数和耗时配置。

代价：

- 阶段边界需要传完整 latent；
- 单请求延迟增加一次排队与 D2D 交接；
- 两阶段服务时间不平衡会形成 pipeline bubble；
- 需要版本、scheduler state 和随机数状态一致。

公开 Wan/Diffusers 实现已证明 expert 阶段边界是确定的，但上述“独立服务池”是系统设计建议，不代表官方仓库已经提供完整实现。

### 6.3 Token-MoE 的高吞吐 EP

- 用 EP 分片 128 experts；
- 使用 HT All-to-All 和低精度 dispatch；
- 对 `layer × timestep bucket` 做 placement；
- 将热点 expert 复制到多个 ranks，但计入显存；
- batch/DP 提高每 expert token 数；
- 尝试把 SP 与 EP 放到正交或分层拓扑；
- 以最慢 rank 和 end-to-end videos/s 调优，不只看 grouped GEMM TFLOPS。

## 7. 推荐 profiling 项

```text
request metadata:
  resolution, frames, prompt length, steps, CFG, task

per timestep/layer:
  attention_ms, moe_ms, collective_ms, other_ms
  expert_token_histogram, max/avg load, pack_padding_ratio
  cache_hit, active_stage, peak_activation

end-to-end:
  condition_ms, denoise_ms, vae_ms
  latency P50/P99, videos/s, frames/s
  HBM peak, host RAM, PCIe/NVLink/RDMA bytes
```

对于 Wan2.2 额外记录 stage switch time 和两套模型驻留位置；对于 LingBot-Video 额外记录 Top-k route 在 timestep 间的 Jaccard/转移率，以判断 placement 和 cache 是否能利用时间相关性。

## 8. 结论

- 固定长度视频的 token-choice MoE 是“大 token、高带宽、长序列”问题，不是 LLM decode 的小批问题。
- Wan2.2 是阶段选择两套整模型，不需要 token EP，但有明显的权重驻留与阶段换入问题。
- 低延迟首先减少去噪步、CFG 和跨步冗余；MoE kernel 是后续优化。
- 显存受限时，Wan 适合 phase offload，token-MoE 更适合量化 + EP/HSDP + SP。
- 高吞吐时，按 shape/timestep/stage 分桶；Wan 可设计双阶段流水，token-MoE 则依赖 HT EP、负载均衡和拓扑规划。
