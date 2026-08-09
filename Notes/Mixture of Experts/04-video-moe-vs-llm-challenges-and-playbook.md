---
type: Note
status: Active
related_to:
  - "[[00-moe-inference-research-index]]"
  - "[[01-llm-moe-inference-systems]]"
  - "[[02-fixed-length-video-moe-inference]]"
  - "[[03-causal-streaming-video-moe-inference]]"
---

# 视频 MoE 与 LLM 的不同挑战及三场景部署手册

## 1. 核心差异矩阵

| 维度 | LLM MoE | 固定长度 token-MoE 视频 | 固定长度 stage-MoE 视频 | Causal/streaming MoE 视频 |
| --- | --- | --- | --- | --- |
| 基本迭代单位 | 1 个新 token/活跃序列 | 完整时空 latent 网格 | 完整网格 + 当前噪声阶段 | 新 frame/chunk × 多个去噪步 |
| 一次 MoE token 数 | decode 小、prefill 大 | 通常数万至数十万 | 无 token router，整模型二选一 | 当前 chunk 通常数千至数万 |
| 重复计算 | 每输出 token 一次 | 同一网格重复 $S$ 个 timestep | 同左，边界切一次 expert | 每新 chunk 重复 $S$ 步，并滚动历史 |
| Attention 状态 | 增长的 KV cache | 通常无跨请求 KV，full attention activation 大 | 同左 | rolling KV + 当前 chunk activation |
| 路由相关性 | 语义 token/请求相关 | 空间、时间、模态、噪声级相关 | 完全由 timestep/SNR 决定 | 还与历史、chunk 和播放阶段相关 |
| 主要通信 | EP dispatch/combine | EP + SP/CFG collective 叠加 | latent/权重搬运，无 token EP | EP + KV/latent handoff + 流水通信 |
| 主要 latency 指标 | TTFT、TPOT | 整段生成时间 | 整段生成时间 + stage switch | TTFF、chunk latency、FPS、jitter |
| 负载失衡 | 热 expert / request skew | 相邻 patch 成片路由，timestep 漂移 | high/low 阶段服务时间不平衡 | expert skew + deadline skew + pipeline bubble |
| 误差表现 | token 概率/文本质量变化 | 空间伪影、跨帧闪烁 | 阶段交界质量变化 | 误差进入未来条件并长期漂移 |

## 2. 视频 MoE 特有的八个难点

### 2.1 路由轴不唯一

LLM 主要按 token hidden state 路由；视频 MoE 可以按 token、timestep、模态、任务甚至分辨率选择 experts。系统首先要识别真实路由粒度，不能看到模型名里的 `MoE` 就直接启用 token EP。

### 2.2 Active parameters 与 HBM 不再同义

Wan2.2 每步只执行约 14B 参数，但两套 experts 同时 resident 时仍接近 27B total 的权重。token-MoE 也只计算 Top-k，但所有 experts 必须在集群 HBM、CPU 或存储层次的某处存在。

### 2.3 长序列让 Attention 和 EP 同时昂贵

LLM decode 经常是 expert 权重带宽主导；视频可能先受 full attention 的二次复杂度限制。SP 和 EP 都可能使用 All-to-All，简单叠加会争抢链路和 SM。

### 2.4 空间相关负载不是 i.i.d.

同一对象/运动区域的 patch 会一起变热，某些 experts 收到成片 token。训练时全局均衡不代表单个 timestep 和单个视频均衡。

### 2.5 路由随噪声轨迹变化

早期高噪和后期语义清晰的 hidden distribution 不同。EPLB、kernel 选择和 expert cache 应按 timestep bucket 统计，而不是假设整条轨迹平稳。

### 2.6 视频 batching 有 scheduler-state 约束

请求 shape 一样仍不一定能合批；还要对齐 timestep、CFG、active stage、cache policy。stage MoE 尤其需要 high/low 两个队列。

### 2.7 因果模型有双重长期状态

rolling KV 和 expert weights 同时占 HBM。热点 expert 复制、KV window、MoE buffer 和当前 chunk 之间必须联合分配。

### 2.8 质量回归更难被短 benchmark 捕获

专家量化、cache 或路由扰动可表现为闪烁、身份漂移和物理不连续；causal rollout 还会累积。必须增加长视频和时序质量测试。

## 3. 场景一：降低延迟

### 3.1 优化顺序

```text
减少模型调用次数
  -> few-step / causal distillation
  -> CFG distillation 或 CFG parallel
  -> 跨 timestep cache / sparse attention

缩短单次调用
  -> attention kernel + SP/CP
  -> MoE fused kernel + quantization
  -> EP communication / stage placement

缩短流水关键路径
  -> encoder/VAE overlap
  -> step/layer/stage pipeline
  -> 预热、CUDA Graph、静态 bucket
```

第一层通常收益最大，因为会同时消除 Attention、MoE 和通信；越往后越依赖 profile。

### 3.2 LLM 配方

- prefill/decode 分开配置；
- decode 使用 LL All-to-All 和小批 kernel；
- EP 尽量不跨节点，能放下时比较 TP；
- fused router/TopK/permute/grouped GEMM；
- CUDA Graph + token bucket；
- FP8/FP4 expert weights 和低精度 dispatch；
- shared/routed expert overlap；
- 不为等待大 batch 牺牲 TPOT。

证据入口：[DeepEP](https://github.com/deepseek-ai/DeepEP)、[NCCL EP](https://arxiv.org/abs/2603.13606)、[TensorRT-LLM dense-GEMM MoE](https://nvidia.github.io/TensorRT-LLM/1.3.0rc15/blogs/tech_blog/blog24_MoE_as_Dense_GEMM.html)。

### 3.3 固定长度 token-MoE 视频配方

- 先做 4--8 step distillation/CFG elimination；
- block-level cross-timestep cache；
- 长序列使用 SP，EP 只在 experts 放不下或有实测收益时启用；
- contiguous grouped GEMM，而不是小批 dense-MoE trick；
- FP8 expert + FP8 dispatch，router/norm/timestep modulation 保留敏感精度；
- SP/EP mesh 做拓扑分层；
- VAE decode 并行或 tile，避免 DiT 已快但 VAE 成为尾巴。

### 3.4 固定长度 stage-MoE 视频配方

- 两套 experts 均 resident：单请求最低延迟，显存最高；
- 显存只容一套：high 阶段末异步预取 low，边界只切一次；
- 多请求：high/low 专家池分离，流水化请求；
- 按 stage 做 request bucket；
- cache 在 expert boundary 强制失效/刷新；
- 不逐层交替搬两套模型。

### 3.5 Causal/streaming 配方

- causal 1--4 step distillation；
- 首 chunk 特化并预热所有关键 experts；
- rolling KV/sink chunk，避免历史重算；
- high/low expert、DiT layer、VAE 分段流水；
- 当前 chunk 的 EP 限制在高速互联域；
- 以 P99 chunk deadline 和 TTFF 调参，不以离线平均 FPS 代替；
- 在播放缓冲不足时优先降 step/quality，不等待大 batch。

证据入口：[Causal Forcing++](https://arxiv.org/abs/2605.15141)、[StreamDiffusionV2](https://arxiv.org/abs/2511.07399)、[CausalWan2.2 模型卡](https://huggingface.co/FastVideo/CausalWan2.2-I2V-A14B-Preview-Diffusers)。

### 3.6 延迟反模式

| 反模式 | 原因 |
| --- | --- |
| 把 EP size 直接设为全部 GPU | collective domain 扩大，单层 latency 可能上升 |
| 用吞吐 kernel 服务 decode | 大 buffer/聚合路径增加固定延迟 |
| 单 chunk 每个 timestep 都换 expert 权重 | 权重 I/O 进入关键路径 |
| 只优化 MoE、不测 Attention/VAE | Amdahl 上限很低 |
| 用平均 FPS 宣称实时 | 无法反映首帧和 deadline miss |

## 4. 场景二：显存受限

### 4.1 统一容量公式

部署前至少估算：

$$
M_{\mathrm{peak}} = M_{\mathrm{resident\ weights}}
+ M_{\mathrm{KV/cache}} + M_{\mathrm{activation}}
+ M_{\mathrm{workspace}} + M_{\mathrm{runtime}}.
$$

其中：

- `resident weights` 要按实际 TP/EP/HSDP/PP 和 redundant experts 算；
- `KV/cache` 对 LLM/causal video 随上下文变化，对 full-sequence video 则包括 timestep residual cache；
- `activation` 对视频强依赖分辨率和帧数；
- `workspace` 包括 MoE pack/padding/collective 和 attention buffer；
- `runtime` 包括 CUDA Graph pools、allocator fragmentation 和框架常驻对象。

### 4.2 优先级

```text
1. 量化主要权重（expert-first）
2. 分片：EP / HSDP / TP / PP
3. 限制状态：KV window、cache budget、batch/token budget
4. offload：module/phase -> block -> leaf
5. 必要时 CPU compute / disk tier
```

越细粒度 offload 越省 HBM，但同步和 PCIe 次数越多。

### 4.3 LLM 配方

- expert-only FP8/FP4/W4A8；
- EP 分片 routed experts；
- KV cache 量化/分页并限制并发；
- 关闭不必要的 redundant experts；
- hot experts 常驻，冷 experts 用 MoE-Infinity 式 cache/prefetch；
- PCIe 搬运太慢时考虑 Fiddler 式 CPU expert compute；
- 离线吞吐场景可用 Klotski 多 batch 隐藏 I/O。

### 4.4 固定长度 token-MoE 视频配方

- expert weights FP8/FP4；
- EP 切专家，SP 切长 activation；
- dense/shared 部分用 HSDP/PP；
- VAE tile/parallel；
- cache 设置显式 HBM budget；
- 只有真实 route trace 显示少数 experts 长期热时才做 offload，否则大视频 token 会覆盖大部分 expert 池。

### 4.5 固定长度 stage-MoE 视频配方

阶段路由确定，因此 phase offload 优于通用 expert prediction：

```text
high expert resident -> prefetch low -> one boundary swap -> low resident
```

再叠加：

- FP8/NVFP4/INT8 quant；
- FSDP/HSDP/PP 分片当前 14B expert；
- text encoder module offload；
- VAE block/tile offload；
- pinned CPU buffer 和异步 stream；
- 若 CPU RAM 不足再使用 disk tier。

### 4.6 Causal/streaming 配方

- rolling KV/window 是第一约束；
- expert quant/EP 与 KV budget 联合搜索；
- 单流允许 phase offload，多 chunk 流水则应停止换入或拆 expert pools；
- sink chunk/长期锚点只保留必要层和必要 token；
- MoE workspace 只按当前 chunk 上界分配；
- 对严格实时流，disk offload 通常不可接受。

证据入口：[Diffusers offloading](https://huggingface.co/docs/diffusers/optimization/memory)、[LightX2V](https://github.com/ModelTC/LightX2V)、[MoE-Infinity](https://arxiv.org/abs/2401.14361)、[Fiddler](https://arxiv.org/abs/2402.07033)。

### 4.7 显存反模式

- 用 `active_params × dtype` 当峰值显存；
- 忽略第二套 Wan transformer；
- 为 EPLB 复制 experts 却不扣减 KV/cache budget；
- 开启 cache 后仍按无 cache 的最大 batch 配置；
- block offload 与 CUDA Graph capture 组合但不测额外 memory pool；
- 为省 1--2 GB activation 触发几十 GB 权重的反复 PCIe 传输。

## 5. 场景三：提高吞吐

### 5.1 目标函数必须是 goodput

离线可最大化 videos/s 或 tokens/s；在线应最大化：

$$
\mathrm{Goodput}=\frac{\text{满足 SLO 的有效输出}}{\text{时间或 GPU 成本}}.
$$

大 batch 提高 kernel 利用率，但增加排队；宽 EP 降低每 rank 权重，却增加网络；热点复制平衡负载，却减少可用 KV/activation 空间。吞吐方案必须带 SLO 约束。

### 5.2 LLM 配方

- continuous batching、chunked prefill；
- DP attention + EP experts；
- HT All-to-All、低精度 dispatch；
- EPLB + 少量热点 expert copies；
- Dual Batch Overlap；
- prefill/decode 使用不同实例和 kernel；
- 网络足够强时评估 Attention-FFN disaggregation；
- 以真实 expert histogram 做 placement 和 kernel dispatch。

### 5.3 固定长度 token-MoE 视频配方

- `(shape, timestep bucket, CFG, cache mode)` 请求分桶；
- DP 提高视频并发，EP 汇聚 expert token；
- HT grouped GEMM/All-to-All；
- timestep-aware EPLB/placement；
- SP 解决超长序列，但限制 collective 竞争；
- condition embedding 复用；
- VAE 单独扩容，避免 DiT queue 清空后堵在 decode。

### 5.4 固定长度 stage-MoE 视频配方

推荐 two-stage serving pipeline：

```text
high-noise queue -> high expert pool
                 -> latent handoff
low-noise queue  -> low expert pool
                 -> VAE pool
```

配置原则：

- 按各阶段 `steps × step_time` 分配 GPU；
- 每池内部按 shape batch；
- 阶段间传 latent，不传整模型；
- queue depth 驱动弹性扩缩；
- 对单请求 latency 设置最大排队时间；
- checkpoint/version 在三池原子切换，避免混合模型版本。

### 5.5 Causal/streaming 配方

- 以 playback slack 做 chunk microbatch；
- step/layer/stage/VAE 多级流水，但保持每级 service time 接近；
- token-MoE 用 expert affinity 和异步 EP；
- stage-MoE 用 high/low 独立池；
- rolling KV 随 stream 迁移时考虑数据局部性；
- backpressure：过载时降 step/quality 或拒绝新流；
- 报告 sustained FPS、P99 deadline miss 和并发 streams/GPU。

证据入口：[StreamDiffusionV2](https://arxiv.org/abs/2511.07399)、[MAGI-1](https://arxiv.org/abs/2505.13211)、[vLLM-Omni diffusion features](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/diffusion_features/)。

### 5.6 吞吐反模式

- 把不同 timestep 或 Wan stage 的请求强行合批；
- 过度 microbatch，使 expert GEMM 重新变成 memory-bound；
- 只看平均 expert load，不看最慢 rank；
- pipeline stage 数大于可持续在途 chunks，导致大量 bubble；
- Attention/FFN disaggregation 未做网络 roofline；
- 通过增加播放缓冲掩盖 deadline miss，却仍宣称低延迟。

## 6. 三种推荐部署模板

### 6.1 模板 A：低并发、延迟优先

```text
单 NVLink domain
  fixed token-MoE: SP x EP 的小规模混合，experts resident
  fixed stage-MoE: high + low 都 resident，或两 GPU groups 常驻
  causal: 1--4 step，rolling KV，首块专用优先级
```

启用：distillation、CFG elimination/parallel、quantized kernel、CUDA Graph。

避免：跨节点宽 EP、leaf offload、大等待 batch。

### 6.2 模板 B：消费级 GPU、显存优先

```text
GPU: 当前 blocks/expert + 当前 chunk/latent
CPU pinned: 下一阶段权重 + encoders
disk: checkpoint 冷层
```

固定 stage-MoE 使用 phase/block offload；token-MoE 使用量化，若只有单卡则选择较小模型或证明有效的 expert cache。causal video 限制 rolling window，接受吞吐和 latency 下降。

### 6.3 模板 C：多请求、吞吐优先

```text
front-end scheduler
  -> shape/timestep/stage buckets
  -> DP/SP/EP DiT pools
  -> optional high/low expert stage pools
  -> VAE pools
```

启用：HT communication、request/chunk batching、EPLB、少量热点复制、pipeline、condition cache、独立 VAE 扩容。实时流量加入 slack/deadline scheduler，离线流量填充剩余容量。

## 7. 实验设计与验收

### 7.1 系统指标

| 场景 | 必测指标 |
| --- | --- |
| LLM | TTFT、TPOT P50/P99、tokens/s、SLO goodput、KV HBM |
| 固定视频 | E2E latency、denoise/VAE breakdown、videos/s、frames/s、峰值 HBM |
| 流式视频 | TTFF、chunk P50/P99、sustained FPS、deadline miss、jitter、并发 streams |
| token-MoE | layer/timestep/expert histogram、max/avg load、pack padding、EP bytes |
| stage-MoE | high/low step time、switch/handoff time、两池 queue depth |

### 7.2 质量指标

- 原任务 benchmark：VBench、prompt alignment、I2V consistency；
- 时序指标：flicker、motion smoothness、identity consistency；
- causal 长 rollout：drift、action following、物体持久性；
- 音视频：lip-sync、音画事件同步；
- 对量化/cache/route 调整做逐项 ablation，不能只看少量主观样例。

### 7.3 最小实验矩阵

```text
steps:       base / 8 / 4 / 2
precision:   BF16 / FP8 / FP4-or-INT4
parallel:    1 GPU / SP / EP / SPxEP / PP-or-HSDP
residency:   all-resident / phase-offload / block-offload
batch:       1 / SLO max / throughput max
route trace: uniform / real / hotspot
video:       short fixed / long fixed / 1-min causal rollout
```

## 8. 公开能力与研究空缺

### 已有可复用能力

- LLM：DeepEP/NCCL EP、vLLM EPLB/DBO、Fused MoE、量化和 offload 研究；
- fixed video：SP/CFG/HSDP/PP、cache、distillation、量化、VAE parallel；
- token-MoE diffusion：LingBot-Video kernels、vLLM-Omni 基础 EP；
- stage-MoE：Wan2.2 双 transformer 与确定性阶段切换；
- causal video：rolling KV、few-step causal distillation、step/layer pipeline、SLO scheduler。

### 仍值得做的方向

1. `layer × timestep × modality` 感知的 diffusion EPLB；
2. SP 与 EP 的联合拓扑/collective 调度；
3. route-stable cache：同时利用去噪轨迹和 expert assignment 稳定性；
4. high/low stage experts 的多请求解耦 serving；
5. rolling KV 与 expert residency 的联合 HBM optimizer；
6. playback-slack-aware expert affinity 与 elastic scaling；
7. 面向长 causal rollout 的 MoE 量化和 cache 质量 benchmark；
8. fixed/causal 视频统一的 step-level 可抢占执行接口。

## 9. 最终选型原则

1. 先确认是 token、timestep 还是 modality MoE。
2. 先 profile Attention/MoE/VAE/采样步，再决定是否值得扩 EP。
3. 延迟优先时保持权重常驻和通信局部；显存优先时量化、分片、phase offload；吞吐优先时分桶、HT EP、负载均衡和流水。
4. fixed video 以整段完成时间为核心；causal video 以首块和逐块 deadline 为核心。
5. 所有 `active parameters`、speedup 和 FPS 都必须附带 resident memory、步数、shape、精度、硬件与 SLO。
