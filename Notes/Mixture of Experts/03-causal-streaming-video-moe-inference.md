---
type: Note
status: Active
related_to:
  - "[[00-moe-inference-research-index]]"
  - "[[02-fixed-length-video-moe-inference]]"
  - "[[causal-video-model]]"
  - "[[StreamDiffusionV2]]"
  - "[[SlackServe]]"
---

# 因果与流式视频的 MoE 推理：状态、流水和播放截止时间

## 1. Causal video 不是把 LLM token 换成 frame

因果视频通常按 frame 或 chunk 自回归，但每个新 chunk 内部仍可能执行多步 diffusion/flow denoising：

```text
history chunks + prompt/action
  -> initialize noisy chunk n
  -> denoise chunk n for S steps with causal/block-causal attention
  -> emit/decode chunk n
  -> update rolling KV / memory
  -> generate chunk n+1
```

与 LLM 每轮只生成一个离散 token 不同，一个视频 chunk 包含大量空间 token，并可能反复经过同一 DiT。因果性解决的是未来依赖和无限长度，不自动解决每块的计算量。

## 2. 评价指标改变了

流式系统至少有四个时延指标：

- **TTFF/TTFC**：首帧或首 chunk 可播放时间；
- **chunk latency**：生成一个新块的服务时间；
- **sustained FPS**：稳态产出速率；
- **deadline miss/jitter**：是否在播放缓冲耗尽前完成下一块。

若每 chunk 含 $F_c$ 帧、播放帧率为 $r$，该块提供的播放预算为：

$$
B_{\mathrm{play}}=\frac{F_c}{r}.
$$

稳态必须满足平均服务时间小于该预算，并控制 P99；平均 FPS 足够但某一块因 expert 热点或权重换入超时，用户仍会看到卡顿。

## 3. 当前可验证的因果 MoE 案例

### 3.1 CausalWan2.2 I2V A14B Preview

FastVideo 发布的 [CausalWan2.2 模型卡](https://huggingface.co/FastVideo/CausalWan2.2-I2V-A14B-Preview-Diffusers) 提供 8-step causal inference。模型仓库同时包含 `transformer` 和 `transformer_2`，继承 Wan2.2 的高噪/低噪双 expert；示例显式设置：

```python
use_fsdp_inference=True
dit_cpu_offload=True  # DiT need to be offloaded for MoE
dmd_denoising_steps=[1000, 850, 700, 550, 350, 275, 200, 125]
```

模型卡也注明它是 preview，质量仍有问题，推理速度尚未优化。因此它能证明“阶段 MoE 已进入因果视频 pipeline”，但不应把示例性能当作成熟生产基线。

### 3.2 StreamChar

[StreamChar](https://arxiv.org/abs/2605.25659) 面向长时角色音视频流，使用带 modality-aware MoE FFN 的联合音视频 DiT，并通过两阶段蒸馏、在线 chunk rollout、progress-aware pointer 和 sink-chunk memory 满足实时与长时一致性。它说明流式 MoE 不仅要路由视觉 token，还可能同时路由音频 token；两种模态的 token 数、deadline 和计算量并不对称。

### 3.3 非 MoE 但必须对照的系统

| 系统 | 关键执行技术 | 对 MoE 设计的启示 |
| --- | --- | --- |
| [MAGI-1](https://arxiv.org/abs/2505.13211) | 24 帧/chunk，最多四块并发，固定峰值成本 | chunk pipeline 可提供专家通信/换入的重叠窗口 |
| [StreamDiffusionV2](https://arxiv.org/abs/2511.07399) | SLO-aware batch、rolling KV、step/layer pipeline | serving 目标应是 deadline-aware goodput |
| [Causal Forcing](https://arxiv.org/abs/2602.02214) | 用 AR teacher 做 ODE 初始化 | causal few-step student 不能直接照搬双向 teacher |
| [Causal Forcing++](https://arxiv.org/abs/2605.15141) | frame-wise 1--2 step causal consistency distillation | 降 TTFF 的算法优先级高于单层 kernel 微调 |
| [MiniWorld](https://arxiv.org/abs/2608.01127) | rolling KV + pipelined asynchronous denoising | 长度无关的峰值状态应作为目标 |

> [!NOTE]
> 这些对照系统公开描述并非都使用 MoE。引用它们是为了抽取因果执行和 serving 约束。

## 4. 相比固定长度视频新增的 MoE 挑战

### 4.1 MoE 权重与 rolling KV 同时争夺 HBM

每层 rolling KV 近似随保留的历史时空 token 数增长：

$$
M_{\mathrm{KV}}\propto L\cdot W_{\mathrm{history}}\cdot d_{kv}\cdot b.
$$

总显存不再只是权重 + 当前完整视频 activation，而是：

$$
M_{\mathrm{peak}}=M_{\mathrm{experts}}+M_{\mathrm{dense/shared}}
+M_{\mathrm{rolling\ KV}}+M_{\mathrm{current\ chunk}}
+M_{\mathrm{MoE\ buffers}}+M_{\mathrm{VAE}}.
$$

复制热点 experts 会减少 EP straggler，却可能挤掉历史 KV，迫使 context window 缩短；反过来保留过长 KV 又会降低可常驻的 experts 数，触发 CPU-GPU 换入。

### 4.2 每块要重复多步去噪

LLM token 产生后不再为该 token 重跑整个模型；causal diffusion 的当前 chunk 会经历 $S$ 次 DiT。好处是一个 chunk 的 token 量大，expert GEMM 效率高；坏处是每块的 route、EP collective 和 Attention 都重复 $S$ 次。

这使三类复用同时存在：

- **历史复用**：rolling KV，避免重算旧 chunks；
- **去噪轨迹复用**：cache 相邻 timestep 的 block residual；
- **expert 复用**：根据前几步的 route 预测后续热点/预取。

三者占用同一 HBM，不能独立把 cache 开到最大。

### 4.3 Stage MoE 与 chunk pipeline 的并发冲突

单 chunk 顺序执行 Wan2.2 时，可把 inactive expert offload，只在边界切一次。但若为了 FPS 同时流水多个 chunks：

```text
chunk n:     low-noise expert
chunk n+1:   high-noise expert
```

两套 experts 在稳态可能同时被需要。此时逐 chunk phase offload 会来回抖动，最坏情况下每个调度周期都搬一套 14B 权重。

因此：

- **单流/显存优先**：phase offload，牺牲 latency/FPS；
- **实时/吞吐优先**：两套 experts 常驻不同 device groups，按 stage 做流水；
- **同卡大显存**：两套同时常驻，避免换入但消耗约两套权重空间。

### 4.4 Token router 的时间一致性

相邻 frames/chunks 描述同一物体。若 router 在 chunk 边界突然把同类视觉 token 送入完全不同的 experts，数值上仍合法，但可能放大外观漂移、纹理闪烁或运动不连续。

这是训练与推理共同问题。推理侧至少应监控：

- 相邻去噪步 route overlap；
- 相邻 chunk、相同空间轨迹的 route transition；
- 每个 expert 的 frame/chunk burstiness；
- route change 与视觉 flicker/identity drift 的相关性。

不能为了系统局部性强行改 TopK 而不做质量评估；router 选择属于模型函数的一部分。

### 4.5 模态不对称

StreamChar/MOVA/JavisDiT++ 一类联合音视频模型同时处理音频和视觉 token：

- 视频 token 数通常更大；
- 音频对口型和时序的 deadline 更严格；
- modality-specific experts 可能天然负载不均；
- shared experts 负责跨模态同步，不能简单 offload 为冷路径；
- 仅按总 token 数做 EP placement 可能让视频 expert 占满一个 rank，音频关键路径等待。

需要以“每模态工作量 × deadline”而不是 token 总量做 placement。

### 4.6 错误会沿 rollout 累积

量化、cache、expert skipping、TopK 改动或 dropped token 在固定视频中影响一段输出；在 causal rollout 中，生成结果又成为未来条件，误差可持续放大。因此离线单 clip VBench 几乎不够，应增加长 rollout 的 identity、motion、action controllability 和漂移指标。

## 5. 降低延迟

### 5.1 算法层：先把每 chunk 的 step 数降下来

优先顺序：

1. causal few-step distillation；
2. CFG distillation/单 pass guidance；
3. 缩小首 chunk 或使用 frame-wise 1--2 step 模式；
4. 在保证长时稳定的前提下做 sparse attention/cache；
5. 再优化 MoE kernel 与通信。

Causal Forcing 指出，从双向 teacher 直接对因果 student 做 ODE 初始化会违反其 frame-level injectivity 条件；应使用 AR teacher。Causal Forcing++ 用 causal consistency distillation 把 frame-wise 2-step 的首帧延迟进一步降低。对 MoE student，teacher/student 的路由结构还要一致或有明确蒸馏映射，否则“输出蒸馏正确”不保证 expert 执行稳定。

### 5.2 系统层：首块关键路径与稳态分开

首块没有可用历史 KV，也没有流水填充，优化目标是：

- prompt/image encoder 提前或缓存；
- 首块使用更少 step/较小 window；
- 所需 experts 预热并常驻；
- 预先 capture 常用 shape 的 CUDA Graph；
- VAE 使用低延迟小块 decode，生成后立即发送。

稳态则可：

- rolling KV 避免历史重算；
- 多 chunks 交错 denoise；
- step/layer pipeline 隐藏通信；
- 在播放缓冲允许时合并 microbatch。

### 5.3 Token-MoE 的低延迟路径

- EP 尽量限制在单 NVLink domain；
- 当前 chunk token 数通常足够大，使用 contiguous grouped GEMM；
- dispatch/combine 与 attention/下一个 chunk 重叠；
- router、pack、TopK 融合；
- 使用 timestep/chunk-aware expert placement；
- 不为追求平均负载进行会阻塞播放的同步迁移。

### 5.4 Stage-MoE 的低延迟路径

**推荐实时拓扑：**

```text
GPU group H: high-noise expert stages
       | latent handoff
GPU group L: low-noise expert stages
       | latent handoff
GPU group V: streaming VAE decode / encode
```

可以让不同 chunks 同时占据 H/L/V 三段。设备比例按每段 service time 配置，使：

$$
\frac{T_H}{n_H}\approx\frac{T_L}{n_L}\approx\frac{T_V}{n_V}.
$$

这是由 CausalWan2.2 的双 expert 结构和流式 step pipeline 推导的拓扑；公开模型卡尚未给出这种完整生产实现。

## 6. 显存受限

### 6.1 必须先把历史状态有界化

仅 offload experts 而让 KV 无限增长无法支持长流。常见办法：

- rolling/sliding KV window；
- sink frames/chunks 保留长期身份锚点；
- 压缩或稀疏选择历史 KV；
- KV quantization；
- 对不同层使用不同历史长度；
- 当前块完成后及时释放 diffusion-only activation/cache。

[StreamDiffusionV2](https://arxiv.org/abs/2511.07399) 使用 sink-token-guided rolling KV；[Fast Autoregressive Video Diffusion](https://arxiv.org/abs/2602.01801) 的 TempCache/AnnSA 进一步研究历史 cache 压缩和稀疏选择，报告长 rollout 下接近恒定峰值显存。

### 6.2 Expert 权重策略

| 约束 | 建议 |
| --- | --- |
| 单卡只能放一套 Wan expert | phase offload；停止多 chunk 跨阶段并发，或接受频繁换入 |
| 多卡总 HBM 足够、单卡不足 | high/low 各自 FSDP/HSDP/PP 分片 |
| token-MoE experts 总量过大 | FP8/FP4 + EP；再评估 hot/cold offload |
| KV 占用过高 | 先缩 rolling window/量化 KV，再决定 expert copies |
| CPU RAM 也不足 | disk-CPU-GPU 分层，但只能用于非严格实时或充分预取 |

LightX2V 已支持 Self-Forcing 的 FP8/NVFP4 路径和三层 offload；CausalWan2.2 模型卡示例也使用 FSDP inference + DiT CPU offload。要注意这证明了“能运行”，不表示能满足实时 SLO。

### 6.3 Workspace 也要有界

MoE pack buffer 不应按无限历史 token 分配；只为当前 denoise chunk 和 EP 接收上界分配。若路由超载，可使用可重放/分段处理，而不是按最坏长视频长度永久保留 buffer。

## 7. 提高吞吐与 sustained FPS

### 7.1 Deadline-aware chunk batching

不能只等到 batch 满。调度器应以剩余播放 slack 决定：

```text
slack = buffered_play_time - predicted_remaining_service_time
```

- slack 大的 stream 可等待并参与大 batch；
- slack 小的 stream 立即执行或降低 step/quality；
- batch 内尽量选择相同 shape、timestep、expert stage；
- 预测需包含最慢 EP rank、VAE 和通信，而非只看平均 DiT 时间。

StreamDiffusionV2 的 SLO-aware batching 与 [[SlackServe]] 的 playout-slack 思路可直接作为调度基线，再加入 MoE stage/expert affinity。

### 7.2 多 chunk 异步流水

流水轴包括：

- denoising step pipeline；
- Transformer layer pipeline；
- high/low expert pipeline；
- DiT/VAE pipeline；
- 多 streams/request microbatch pipeline。

目标不是让单 chunk 同时占据所有 GPU，而是让每个阶段持续有可执行块。过度切分会增加 latent/KV 传输和 bubble，需要以 stage service time 配平。

### 7.3 Expert-affinity scheduling

对 token-choice MoE，若历史 trace 表明某些流/场景持续激活相似 expert 集，可优先把它们调度到持有热点副本的 rank group，减少远程 dispatch。必须保持 router 输出语义不变，affinity 只改变请求/physical expert copy 的放置。

对 stage MoE，affinity 更简单：high-stage queue 只进入 high expert pool，low-stage queue 只进入 low pool。队列长度可直接驱动两池弹性扩缩。

### 7.4 背压和降级

实时系统需要显式过载策略：

- 降低后续 chunk 的 denoise steps；
- 缩短生成分辨率/块大小；
- 减少 CFG 或切换蒸馏模型；
- 丢弃非关键 cache refresh，而不是破坏因果顺序；
- 限制新 stream 接入；
- 绝不让某一 stream 无限占用 expert queue。

这些策略会改变质量，必须和 SLO/质量等级绑定。

## 8. 推荐的状态机

```text
NEW
  -> CONDITION_READY
  -> FIRST_CHUNK_DENOISING
  -> FIRST_CHUNK_DECODING
  -> STREAMING
       -> HIGH_STAGE_READY / TOKEN_MOE_READY
       -> LOW_STAGE_READY
       -> VAE_READY
       -> EMITTED
       -> KV_COMMITTED
       -> next chunk
  -> DRAINING
  -> DONE
```

每个状态携带：

- deadline / playback slack；
- current timestep 与 expert stage；
- latent/KV 所在设备；
- required expert residency；
- cache validity/version；
- quality tier 与可降级范围。

这种显式状态比把整个生成封装为一个不可中断 Python 调用更适合流式调度和多阶段 MoE。

## 9. 需要补齐的工程能力

截至调研日期，公开生态已有单项能力，但完整组合仍不成熟：

| 能力 | 公开状态 |
| --- | --- |
| CausalWan2.2 8-step 模型 | preview 可用，模型卡称速度未优化 |
| diffusion SP/CFG/HSDP/offload | FastVideo、LightX2V、vLLM-Omni 等已有 |
| token-MoE diffusion EP | vLLM-Omni 已有基础支持，EPLB/多后端能力弱于主 vLLM |
| rolling KV / step-layer pipeline | 多个 causal 系统已有，但未普遍与 MoE 联合调度 |
| stage expert service pools | 工程推导，尚缺公开端到端标准实现 |
| timestep/chunk-aware EPLB | 工程推导，尚缺成熟开源实现 |
| deadline-aware expert placement | 研究空缺 |

## 10. 结论

- 因果视频的主要新增约束是 rolling KV、逐块 deadline、误差累积和流水稳态。
- CausalWan2.2 把 Wan 的双 stage experts 带入 causal pipeline；单块 offload 与多块流水存在直接冲突。
- 低延迟首先依靠 causal few-step distillation、首块特化、rolling KV 和专家常驻。
- 显存受限必须同时限制 KV 和权重；只解决其中一个不能支持无限流。
- 高吞吐要按 playback slack 做 chunk batching，并把 timestep stage/expert affinity 纳入调度。
