---
type: Note
related_to:
  - "[[vllm-omni-diffusion-roadmap-zh]]"
  - "[[varlen-batching-separation-execution-plan]]"
  - "[[pipeline-5-interface-3-stage-problem-analysis]]"
  - "[[TeaCache-StepExecution Intergration]]"
  - "[[streaming-generation-problem-breakdown]]"
status: Active
---

# DitServingSys 调研

[TridentServe](https://my.feishu.cn/wiki/EL9Rwerdoi5HZPkZgFfcEImxnld)

[TetriServe](https://my.feishu.cn/wiki/EZYMwvpZbiM4ZskcVmvcdTRRnrf)

[GenServe](https://my.feishu.cn/wiki/Us6zwxuPNihanoktAxscUMfKnbW)

[DiT\-Serve](https://my.feishu.cn/wiki/EsfLw55bTiTa7Fk0k9nc19Urnzf)

[SwiftFusion](https://my.feishu.cn/wiki/BT5Jww9priIKNJkKETTcVoONnLb)

[StreamDiffusionV2](https://my.feishu.cn/wiki/QA44wrtlJirgBVkb4NYcEZ9gnkc)

[SlackServe](https://my.feishu.cn/wiki/U0mfwOVbsiLZ2ykB1kIcH3ZWnwh)

[MoDM](https://my.feishu.cn/wiki/S9wQwcyndiE7DPkBs1Bcz4H1nbf)

[LegoDiffusion](https://my.feishu.cn/wiki/YaD9wcdiiiDGPckP5ADcJFqenUf)

[Helix](https://my.feishu.cn/wiki/O3wXwJ9Z4in8qWkaPTRc8NUgnoc)

[GF\-DiT](https://my.feishu.cn/wiki/V50Rw41x2irFDWkOzKZcIQLinAb)

[DistriFusion](https://my.feishu.cn/wiki/AbhIwNgVCiFLCakX9Hyci3HqnJb)

[DisagFusion](https://my.feishu.cn/wiki/AFHNwg5wRiXy3QkvMGncgOxhnZf)

[DiLaServe](https://my.feishu.cn/wiki/FZhRw6gTniXYphkGkNIcD7wInVc)

[DDiT](https://my.feishu.cn/wiki/PQWRweZFYixaMLk3nlQc9a2Ynae)

[Chorus](https://my.feishu.cn/wiki/UOtewMXT3i0QoakGHfHcGV1pnJb)

# 面对混合负载的高效 DiT 推理服务系统 —— 业界现状调研报告

## 一、研究背景与问题

扩散模型已广泛应用于图像、视频及多模态内容生成。随着生成质量、输出分辨率和视频时长不断提升，基于 Diffusion Transformer（DiT）的推理对计算资源和显存容量提出了更高要求。在线服务还需要处理动态到达的混合请求，不同请求在分辨率、时序长度、去噪步数和服务时限等方面存在明显差异。因此，DiT 服务不仅要降低单请求的执行开销，还要在异构负载下维持稳定的吞吐和服务质量。

完整的扩散推理通常包含 Encoder、DiT 和 Decoder 三类计算。Encoder 将文本或输入图像转换为模型条件，DiT 在多个去噪步骤中完成主要的生成计算，Decoder 将最终 latent 转换为图像或视频。三类模块的资源特征并不一致：DiT 是计算量和显存占用最大的阶段，且单个请求会在多个去噪步骤中持续占用执行资源；Encoder 的计算通常较短，Decoder 则具有较强的显存和带宽特征。将这些模块作为一个整体部署，资源配置只能按照整条流水线进行，难以匹配各阶段的实际需求。

整体式部署主要面临以下两方面限制。

其一，DiT 的显存占用限制了服务规模。输出分辨率或视频时长增加后，latent token 数量随之增加，批次内请求的激活和中间状态也会相应扩大。为控制显存风险，系统通常需要降低 batch size，甚至退化为单请求执行。这直接限制了 DiT 对大批次和高分辨率请求的支持能力。

其二，DiT 的连续去噪执行容易受到 Encoder 和 Decoder 的资源竞争影响。当三类模块共享实例或 GPU 时，编解码请求的到达、排队和执行会挤占 DiT 的资源，造成批次拆分、阶段等待和执行间隙，降低流水线并行效率。由于 DiT 的主要计算集中在连续的多步去噪过程中，这类干扰会进一步放大端到端延迟和吞吐损失。对于在线服务而言，需要将 DiT 作为具有独立资源和调度边界的核心服务，避免其执行过程受编解码阶段的负载波动影响。

分离式推理将 Encoder、DiT 和 Decoder 拆分为可独立部署、独立排队和独立扩缩容的服务或执行池。Encoder 输出的条件信息以及 DiT 生成的 latent 通过明确的数据通道在阶段之间传递，DiT 服务专注于去噪计算。该架构能够为 DiT 单独配置 GPU 和显存资源，从而提升高分辨率请求的承载能力，并在适合批处理的负载下扩大有效并发规模；同时，Encoder 和 Decoder 的负载变化不会直接抢占 DiT 的执行资源，有利于保持 DiT 执行池的连续运行。此外，TeaCache\[9\]、CacheDiT\[3\] 等缓存方法可以在 DiT 服务内部复用去噪步骤之间的冗余计算，进一步降低实际前向计算量。

基于上述背景，本报告重点研究以下问题：如何通过阶段分离缓解 DiT 的显存约束，提升高分辨率请求的承载能力和混合负载下的有效并发能力；如何在 Encoder、DiT、Decoder 之间建立清晰的资源和调度边界，保证 DiT 执行池持续处理可执行任务；以及如何将 TeaCache、CacheDiT 等缓存机制纳入在线服务流程，在保证生成质量的前提下减少重复去噪计算。步级批处理、动态调度和显存管理均是实现上述目标的关键技术环节，但其收益需要结合具体分辨率、batch size 和并行度进行评估。

## 二、现有工作分析

### 2\.1 批处理优化：DiT\-Serve

DiT\-Serve\[4\] 面向在线图像和视频生成服务，重点解决异构请求在空间和时间两个维度上的资源利用率问题。其核心设计包括 Step\-Level Batching（步级批处理）和 Brick Attention（块注意力）。Step\-Level Batching 在每个去噪 step 完成后更新执行集合：已完成或被调度器暂停的请求可以退出当前 batch，等待队列中的请求再进入执行。该机制不会在一个 step 执行到一半时中断请求，而是在 step 边界暂停和恢复请求，用于减少短请求完成后等待长请求造成的空闲，并缓解队头阻塞。Brick Attention 则针对不同分辨率和时序长度造成的 padding 浪费，将请求分配到大小不同的独立 GPU 环中，降低序列并行中的空间浪费。此外，DiT\-Serve 引入了 SRPTF（最短剩余处理时间优先）调度策略，根据剩余去噪步数和序列规模估算剩余工作量，从而降低平均等待时间和尾部延迟。

实验结果表明，DiT\-Serve 在 Open\-Sora、Mochi 和 CogVideoX 等模型上实现了吞吐量和延迟方面的改进。然而，该方法也引入了相应代价：步级批处理需要在每个去噪 step 结束后进行调度决策，暂停和恢复请求需要维护请求状态，频繁切换还可能带来上下文管理和通信开销；Brick Attention 则需要较复杂的 GPU 环和序列组织机制。因此，Step\-Level Batching 的收益主要体现在存在并发混合负载时，对于严格串行的单请求执行，其收益有限。

### 2\.2 资源分配优化：DDiT

DDiT\[5\] 是一个面向 Text\-to\-Video \(T2V\) 推理服务的资源调度系统，其主要设计包括阶段间解耦（Inter\-phase Decoupling）和阶段内解耦（Intra\-phase Decoupling）。阶段间解耦将模型权重加载与通信组构建分离，实现 DiT 与 VAE 的异构部署：GPU 实例可以预先加载模型权重，通信组则根据当前阶段和并行度需求按需建立。阶段内解耦支持在 DiT 执行过程中动态调整并行度（DoP），并在去噪步骤之间重新分配 GPU 资源，从而减少静态资源配置造成的浪费。该设计使 DiT 和 VAE 可以根据各自的计算特征和运行负载独立配置资源。

DDiT 的实验分析归纳了 T2V 服务中的四点现象：第一，在其评测设置下，DiT 和 VAE 在 batch size=1 时 GPU 利用率已经较高，简单增加 batch size 的收益有限；第二，DiT 与 VAE 对并行度的需求并不一致，VAE 对同一输入进行处理时，盲目增加 GPU 可能带来冗余计算；第三，最优 DoP 随分辨率变化，低分辨率请求采用过高并行度时，通信开销可能抵消并行收益；第四，静态部署难以应对运行时负载变化，请求开始执行后无法及时调整 DoP。基于上述分析，DDiT 采用基于 starvation time 的贪心调度算法，为不同分辨率确定合适的并行度，并在运行时按照单个去噪步骤动态调整资源分配。实验结果表明，DDiT 在 8×H800 配置上实现了 P99 延迟降低 30\.4%、平均延迟降低 30%，资源成本达到理论最优值的 1\.39×。

### 2\.3 阶段级资源调度：TridentServe

TridentServe\[2\] 首次系统性分析了扩散流水线的阶段异构性，揭示了资源需求的不对称性。通过分析 4 个主流模型（Stable Diffusion 3、Flux\.1、CogVideoX、HunyuanVideo），TridentServe 发现：Encode 阶段是 Transformer 编码器，处理长度 ≤500，主要靠 batching 提效；Diffuse 阶段是 DiT 模型，处理长度 100\-120k，占 70%\+ 时间，计算密集，对并行度敏感；Decode 阶段是 VAE 解码器，内存密集，对并行度不敏感，增加 GPU 收益有限。此外，工作负载模式变化时，阶段间的资源比例也需要动态调整。基于这些分析，TridentServe 提出了阶段级解耦的部署方式，其重要优势之一在于允许各阶段根据工作负载独立扩容：当 DiT 阶段成为瓶颈时，系统可以单独增加 DiT 阶段的 GPU 数量，而无需复制整个管线，从而避免了资源浪费。

基于这些分析，TridentServe 提出动态 stage\-level 服务范式，包含两大抽象：Placement Plan（模型部署）和 Dispatch Plan（请求调度）。Placement Plan 决定每个 GPU 上部署哪些阶段的模型副本，支持 6 种 Placement 类型（⟨EDC⟩、⟨DC⟩、⟨ED⟩、⟨D⟩、⟨E⟩、⟨C⟩），并通过 Virtual Replica 概念实现通信开销最小化的请求执行。Dispatch Plan 则为每个请求的每个阶段选择 GPU、并行策略和执行时间，通过 ILP 建模求解最大化 SLO 达标率。TridentServe 还设计了 Adjust\-on\-Dispatch 机制，当 Monitor 检测到阶段间速度失衡时，无停机地动态重部署模型，进一步体现了分离式架构在资源动态调整方面的灵活性。实验结果表明，TridentServe 在 4 个模型和多种工作负载下实现了平均延迟降低最高 2\.5×，P95 延迟降低最高 3\.6×，P99 延迟降低最高 4\.1×。

### 2\.4 步级并行度调整：TetriServe

TetriServe\[6\] 证明了 DiT 步级调度问题是 NP\-hard 的，并提出了 Round\-Based Scheduling（基于轮次的调度）来解决这一问题。TetriServe 的核心洞察包括：第一，DiT 工作负载输入异构但执行可预测，每个分辨率的每步运行时间高度可预测（CV \< 0\.7%），这使得 deadline\-aware 调度成为可能；第二，序列并行的扩展效率是次线性的，且因分辨率而异，小分辨率用高并行度是浪费，大分辨率用低并行度是不够；第三，步级并行度调整可以适应 deadline，高分辨率或紧急请求分配更多 GPU，小分辨率或不紧急请求节省资源。

基于这些洞察，TetriServe 设计了 Deadline\-Aware GPU Allocation 和 DP 求解装箱算法。Deadline\-Aware GPU Allocation 为每个请求找到满足 deadline 的最小 GPU 分配，通过离线画像获取每种分辨率和 GPU 数量组合的执行时间，运行时查表枚举。DP 求解装箱算法则将每轮的调度问题建模为分组背包问题，用动态规划在 O\(RN\) 时间内求解每轮调度。此外，TetriServe 还设计了 GPU Placement Preservation（放置保持）和 Work\-Conserving Elastic Scale\-Up（弹性扩展）机制，减少轮次间开销，充分利用 GPU。实验结果表明，TetriServe 在 FLUX\.1\-dev（8×H100）和 SD3（4×A40）上实现了 SLO 达标率最高提升 32%，与缓存加速（Nirvana）正交兼容，组合使用 SAR 从 0\.42 提升到 0\.88。

### 2\.5 流水线并行优化：DisagFusion

DisagFusion\[7\] 是一个解耦式扩散模型推理系统，通过异步流水线并行和弹性调度，在异构 GPU 上实现高效部署。传统流水线中，各阶段之间存在大量等待时间，DisagFusion 通过将 latent 分成多个 chunk，实现发送 chunk i 与计算 chunk i\+1 重叠，并使用双缓冲（两个缓冲区交替读写）减少等待时间，从而实现通信与计算的重叠。分离式架构的核心优势之一在于允许各阶段根据工作负载独立扩容：DisagFusion 的弹性调度策略能够根据运行时负载动态调整各阶段的实例数量，当 DiT 阶段出现瓶颈时，系统可以从空闲阶段借 GPU 给瓶颈阶段，实现资源的动态优化配置。

DisagFusion 还设计了弹性实例调度策略，通过性能预测和运行时反馈动态调整各阶段的实例数。性能预测模型基于离线 profiling 获取基准数据，在线微调适应实际负载；运行时反馈则监控各阶段队列长度，检测瓶颈阶段，并从空闲阶段借 GPU 给瓶颈阶段。实验结果表明，DisagFusion 在 Stable Video Diffusion 和 OpenSora 上实现了吞吐量提升 3\.4×\-20\.5×，延迟降低 18\.5×，并支持异构 GPU 部署（如 4× A100 \+ 4× V100 混合配置）。

### 2\.6 微服务化架构：LegoDiffusion

LegoDiffusion\[8\] 提出了微服务化（micro\-serving）的扩散工作流推理系统，将 Text\-to\-Image 工作流分解为松耦合的模型执行节点，实现独立管理和调度。LegoDiffusion 的核心设计包括：编程接口与编译（Python\-embedded DSL 用于组合扩散工作流，图编译器将工作流转换为 DAG）、运行时与数据平面（惰性执行、基于 NVSHMEM 的分布式数据引擎、GPU\-direct 零拷贝张量移动）、工作流节点调度（模型粒度扩展、多租户模型共享、自适应并行度）。微服务化架构的核心优势在于允许各个组件根据工作负载独立扩容：当某个模型（如基础扩散模型）成为瓶颈时，系统可以单独扩展该模型的实例数量，而无需复制整个工作流，从而避免了资源浪费。

LegoDiffusion 的实证分析揭示了整体式部署的四个核心问题：L1（低效的全复制扩展），单体服务强制粗粒度复制，加载延迟增加 80%，GPU 内存浪费 75%；L2（无法共享通用模型），工作流实例严格隔离，流行模型在多个工作流中重复部署；L3（运行时低效），黑盒封装导致系统可见性缺失，无法优化数据流和执行逻辑；L4（系统脆弱性和维护开销），紧耦合导致系统脆弱，单点故障级联。基于这些分析，LegoDiffusion 实现了请求率提升 3×，突发流量容忍度提升 8×，GPU 需求减少高达 3×。

## 三、现有工作的局限性

现有研究分别从异构请求批处理、阶段级部署、动态资源分配、步级并行度和流水线并行等角度进行了优化，为 DiT 服务系统的设计提供了重要基础。但从本项目关注的“混合负载 + 阶段分离 + 缓存在线化 + 流式实时”目标来看，需要先逐项审视每类方案的机制代价与适用边界，再归纳跨工作的共性问题，才能准确定位缺口。

### 3.1 逐项审视：每类方案的机制代价与适用边界

**DiT-Serve（步级批处理）**

- 步级批处理需要在每个去噪 step 结束后重新做调度决策，暂停与恢复请求要求维护完整的请求上下文与中间状态，频繁切换会引入上下文管理和通信开销。其“调度开销几乎可忽略”的结论来自并发度有限的视频模型实验，不能直接外推到高并发混合负载。
- Brick Attention 把不同尺寸的请求装箱到大小不同的 GPU 环上，环的组织、重排与跨环通信机制复杂；收益依赖存在并发混合负载，严格串行或低并发时收益有限。
- 它本质上是调度层优化，默认 DiT 已经拥有独立且充足的 GPU 资源，没有回答“显存不够时怎么办”，也没有把 TeaCache 等缓存方法纳入执行模型。

**DDiT（动态资源分配）**

- 运行时动态调整 DoP 需要重建通信组、迁移或重排中间状态，重配置本身有通信与状态迁移成本；论文自己的实验也显示低分辨率请求使用高 DoP 会因通信开销导致性能下降。
- 调度依赖离线 profiling 得到的“分辨率 × GPU 数 → 执行时间”画像，负载分布变化、新增分辨率或模型时都需要重新标定，在线自适应能力有限。
- 静态集群隔离类方案资源无法跨集群共享，动态分区则隐含 NVLink 级高速互联假设；主要面向 T2V 的 DiT/VAE 两阶段，未覆盖图像生成、多模态与缓存加速。

**TridentServe（阶段级调度）**

- 阶段级静态批处理在请求到达时间与配置对齐时收益显著，但真实在线负载下批处理机会高度碎片化，静态策略难以持续生效；placement plan 的周期性/离线特性使其对突发流量响应慢。
- 仍以请求为最小执行单位，去噪阶段不可中断，长请求会阻塞短请求，缺少 step 边界的抢占与换入换出。

**TetriServe（步级序列并行）**

- round-based 离散化调度需要在轮次边界同步，轮间调度与 GPU Placement Preservation 有额外开销；动态 SP 度数变化涉及序列切分重排和通信开销。
- SLO 建模依赖对执行时间的精确估计，估计误差会直接传导到调度质量；实验以图像生成为主（FLUX.1-dev、SD3），视频与流式场景未验证。
- 与缓存加速组合后 SLO 达标率才从 0.42 提升到 0.88，说明缓存与调度必须协同，但 TetriServe 本身不管理缓存生命周期与显存。

**DisagFusion（异步流水线并行）**

- 弹性实例调度依赖“性能预测 + 运行时反馈”，性能预测需要离线标定并在线微调，预测误差会直接导致扩缩容决策失误；chunk 切分与双缓冲使流水线管理复杂化，阶段间通信即使异步重叠仍有成本。
- 弹性调度主要基于队列长度反馈，粒度较粗；验证模型为 SVD 与 OpenSora，未覆盖多模态与流式场景。

**LegoDiffusion（微服务化）**

- 模型间数据传输、多模型池管理与微服务调用引入额外延迟和复杂度；DAG 编译对动态图与运行时变化的灵活性有限。
- 关注的是“工作流分解”层面的独立扩缩容，对单个 DiT 服务内部的 step-level 批处理、显存管理与缓存执行没有贡献，两者互补但未整合。

### 3.2 跨工作的共性问题

第一，多数工作只做单维度优化，缺少系统级协同。DiT-Serve 管批处理、DDiT 管资源分配、TridentServe 管阶段调度、TetriServe 管并行度、DisagFusion 管流水线、LegoDiffusion 管架构形态，彼此之间没有统一的执行模型。调度类工作默认 DiT 已拥有独立且充足的资源，没有真正解决与 Encoder、Decoder 共享实例时的资源竞争；缓存类工作则停留在单请求推理的附加优化。最典型的是“缓存减计算却增显存”的矛盾：缓存状态与模型激活、请求上下文共同占用显存，会压缩可用于批处理的显存预算，这一矛盾没有任何工作统一处理。

第二，阶段解耦并不自动等同于 DiT 服务的高效运行。不同请求在分辨率、帧数和剩余去噪步数上差异显著，显存占用与单步执行时间也随之变化。如果仍以请求为单位做粗粒度显存预留，批次规模会被大请求限制，阶段分离带来的资源独立性难以发挥。连续批处理、请求上下文管理和显存分配必须协同设计。

第三，缺少统一的成本度量与资源抽象。现有工作分别以请求数、GPU 数、token 数等作为度量，难以在异构请求（分辨率、帧数、步数）之间建立统一的资源视角；调度决策因此无法以“真实计算成本”为依据，SLO 与公平性估计都缺乏可靠底座。

第四，显存管理、计算执行、调度控制三个层面脱节。请求上下文需要跨多个去噪 step 存活，step 边界的暂停/恢复要求安全保存上下文；计算层需要在线装箱最大化执行效率；调度层需要动态调整执行顺序。三个层面各自为战，就无法在 step-wise 调度下稳定维持并发规模。

第五，缓存加速与在线调度缺少统一的执行模型。TeaCache、CacheDiT 等面向单请求整段 forward 的优化，在 step-wise 交错执行下存在命中率、缓存占用、失效时机、请求切换等多个新问题；probe 与 commit 没有事务语义，状态没有按 request + CFG branch 键控，缓存命中带来的计算节省未必能稳定转化为端到端性能提升。

第六，评价指标与实验设定差异大，通用性不足。各工作面向特定任务（视频/图像）与特定模型族，多假设同构 GPU 与充足互联；指标口径（有效并发规模、batch size 吞吐、分辨率上限、GPU 利用率、缓存减少的前向计算量、E/D 对 DiT 等待的影响）不一致，难以横向比较，也难以直接指导本项目的工程目标。

## 四、我们要解决的问题

基于上述缺口，结合仓库内已有的设计文档（[[vllm-omni-diffusion-roadmap-zh]]、[[varlen-batching-separation-execution-plan]]、[[pipeline-5-interface-3-stage-problem-analysis]]、[[TeaCache-StepExecution Intergration]]、[[streaming-generation-problem-breakdown]]），本项目要解决的问题不是“再做一个调度器”，而是建立一套以三条所有权边界和四个执行契约为骨架的 DiT 服务执行模型，并逐项验证。

所有权边界固定为：

- **pipeline**：拥有模型字段、CFG 展开/合并、row 语义、cache 有效性、边界状态投影/恢复、attention-call 语义；
- **runner**：拥有 request 生命周期、admission、调度、物理 batch 调用、声明式 mapping 的校验与应用、清理和指标；
- **transport**：只搬运版本化 payload，绝不解释模型字段。

四个执行契约：

1. **Step batch 契约**：本轮有哪些 request、sample row、CFG branch；公共 tensor 如何 gather/scatter；模型私有输入由谁组装。
2. **Packed attention 契约**：一次具体 attention 调用的 Q/KV packing、cu_seqlens、mask 与 position。
3. **Stage transport 契约**：encode、dit、decode 之间传哪些可序列化状态，谁拥有其生命周期。
4. **Cache / adaptive execution 契约**：哪些结果可复用，何时 probe/commit，以及创建、命中、失效和释放规则。

在此基础上，具体要解决八个问题。

**问题 1：执行模型与所有权边界。** 当前 `InputBatch` 与 `StepRequestState` 混入 Qwen 专用字段（`img_shapes / txt_seq_lens / negative_txt_seq_lens`），Wan、Hunyuan、Cosmos 继续接入只会让公共 dataclass 继续膨胀；runner 目前还可能按 `latents.shape[0]` 推断分片，这无法扩展。我们要引入模型拥有的 step collation 与显式、可校验的 `RequestRowLayout`（描述 request、latent、CFG branch、model-input 与 prediction row 的映射），让 runner 只做“校验与应用”，不做模型特判；`StepRequestState` 定位为本地可变状态而非 wire format。

**问题 2：连续批处理与 step-wise 调度的执行契约。** step execution 主链（`prepare_encode / denoise_step / step_scheduler / post_decode`）已存在，但 attention metadata 由各模型在调用点自行构造，缺少外部直接提供两套 packed `cu_seqlens` 的通用契约。我们要定义 `PackedSequenceLayout`，把 row、stream slice、Q/KV `cu_seqlens` 和 opaque position payload 绑定到一次 attention call，而不是先做一个全局 attention plan；并验证 `serial ≈ dense ≈ packed` 的数值等价。调度只决定“这一步谁一起执行”，不承担理解 text/image stream、RoPE 或模型名的职责。

**问题 3：缓存如何成为在线执行的一等公民。** 当前 TeaCache 按 positive/negative branch 维护全局状态、probe 与 commit 未分离、decision 直接修改 accumulator，step runner 还显式拒绝 cache backend。我们要把 TeaCache 重构为 request-level transactional policy：`probe` 无副作用、`commit` 显式；状态按 request + CFG branch 键控；一次 runner 调用中先 probe logical cohort，再 force-compute COMPUTE 子集，合并 prediction 后让每个请求恰好推进一步；`threshold=0` 与 all-COMPUTE 必须等价于 no-cache；覆盖 finish/abort/error 的清理。这一步应在真正的 packed varlen 之前形成 request-level 闭环，因为两者共享 request 状态所有权、physical sub-batch 重组、CFG row mapping 与 merge/scatter 语义。

**问题 4：缓存/KV 与显存的生命周期管理。** step-wise 交错执行下，缓存与 KV 需要跨 step 存活，不能在 step 结束后立即释放；缓存可回收，但回收即失去未来命中收益。缓存、模型激活与请求上下文共同占用 DiT 显存，进一步压缩批处理预算。我们要把显存预算纳入 admission control 与调度决策，统一管理缓存/KV 的创建、命中、失效与释放；对远程显存池等方案用带宽下界（KV 规模 × step 数 / 时间预算）先做可行性判断，再决定是否投入。

**问题 5：阶段边界与 3-stage 分离的正确性。** 分离部署必须先回答“encode 到 DiT、DiT 到 decode 之间到底传什么、谁拥有生命周期”。我们要定义 `StagePayload` envelope 与 boundary codec、字段所有权表，让 scheduler 对象在 DiT role 本地重建而不是跨进程传输；completion 语义区分 step/chunk/denoise/request 完成四种情况；完成定义是 decoder delivery/ack 而非最后一个 DiT step。先做同进程的 local vertical slice 验证正确性，再引入真实 transport。

**问题 6：流式实时场景的 SLO。** 现有工作聚焦在线吞吐与静态 deadline，对实时交互场景（首帧时间、逐 chunk deadline、播放连续性、交互延迟）缺少建模。我们要建立四类指标（TTFC、chunk deadline miss、playout continuity、interaction latency），用 playout slack 与 service_credit 表达“提前一点就够”的松弛度；以 DiT session affinity + E/D 共享池作为推荐基线；同时明确“3 分钟生成”不等于必须保留全量 KV（rolling window/sink），把流式需求与显存容量解耦。

**问题 7：多模型家族的泛化。** 四种模型族对应三种 attention 拓扑：联合自注意力（Qwen、Hunyuan）、视觉 self + 文本 cross（Wan）、UND causal + GEN cross（Cosmos）。不可强行统一成一个全局 `attention_plan`：同一模型的不同 block/phase 可能调用不同算子，metadata 绑定到一次 attention call；KV 是否可缓存还取决于 timestep modulation、CFG 分支、prompt update、层号与并行布局，不能只看 dynamic/static。我们要用 capability flag 显式声明能力，每个模型在具体调用点解释 layout，按 Qwen → Hunyuan → Wan → Cosmos 的顺序分阶段接入。

**问题 8：评估体系。** 评价必须同时覆盖：DiT 有效并发规模、实际 batch size 在不同负载下的吞吐收益、可支持的最高分辨率、GPU 利用率、缓存减少的实际前向计算量、Encoder/Decoder 对 DiT 等待时间的影响，以及生成质量（如 Vbench 类指标）。正确性门槛用轨迹 parity 而不是“最终图片看起来正常”：`serial ≈ dense ≈ packed`；`threshold=0` 等价 no-cache；固定 decision trace 下 dense-TeaCache ≈ packed-TeaCache。

## 五、总结

本次调研确认：DiT 在线服务的主要瓶颈集中在去噪模块的显存与计算压力；整体式部署还会引入 Encoder、Decoder 与 DiT 之间的资源竞争，且以请求为最小执行单位的执行方式无法充分利用混合负载下的批处理机会。分离式推理、step-wise 调度、缓存加速与弹性扩缩容是业界公认的优化主线，但现有工作各自为战：调度类方案默认资源独立充足，缓存类方案停留在单请求优化，显存/上下文生命周期管理缺少与两者的协同。

本项目要形成的完整服务链路是：以所有权边界固定执行模型（pipeline 拥有语义、runner 拥有生命周期、transport 只搬运 payload），用四个执行契约串联显存/缓存生命周期管理、连续批处理与 step-wise 调度（packed varlen）、TeaCache request-level transactional policy、阶段边界与弹性扩缩容，并以流式 SLO 作为在线场景的验收目标。

交付按依赖顺序分阶段推进：冻结基线与字段所有权表 → model-owned collation + `RequestRowLayout` → TeaCache request-level 闭环 → TeaCache 单 logical-step dense MVP → Qwen packed varlen（cache off）→ TeaCache × varlen → atoms 收敛 → 本地 payload vertical slice → transport 与 3-stage 分离 → 分布式与多模型扩展。每个阶段都有明确的正确性门槛（数值轨迹等价、threshold=0 等价 no-cache、decoder delivery/ack 完成语义），确保优化不牺牲生成质量。

评估将以有效并发规模、分辨率上限、缓存收益、吞吐、延迟与生成质量为统一口径，在流式场景补充 TTFC、chunk deadline miss、playout continuity 与交互延迟指标，最终回答“这套执行模型相比现有工作，能在多大的并发与分辨率下稳定保持 SLO”。

## 参考文献

\[1\] Fang, J\., Pan, J\., Sun, X\., Li, A\., \& Wang, J\. \(2024\)\. xDiT: an Inference Engine for Diffusion Transformers \(DiTs\) with Massive Parallelism\. arXiv preprint arXiv:2411\.01738\.

\[2\] Xia, Y\., Fu, F\., Yuan, H\., Zhang, H\., Miao, X\., Liu, Y\., \.\.\. \& Cui, B\. \(2025\)\. TridentServe: A Stage\-level Serving System for Diffusion Pipelines\. arXiv preprint arXiv:2510\.02838\.

\[3\] Cache-DiT: A Unified Cache Acceleration Framework for Diffusion Transformers\. [https://github\.com/vipshop/cache\-dit](https://github.com/vipshop/cache-dit)

\[4\] DiT\-Serve: An Efficient Serving Engine for Diffusion Transformers\. OpenReview\.

\[5\] Huang, H\., Hu, C\., Zhu, J\., et al\. \(2025\)\. DDiT: Dynamic Resource Allocation for Diffusion Transformer Model Serving\. arXiv preprint arXiv:2506\.13497\.

\[6\] Lu, R\., He, S\., \& Chowdhury, M\. \(2026\)\. TetriServe: Efficiently Serving Mixed DiT Workloads\. ASPLOS '26\. arXiv preprint arXiv:2510\.01565\.

\[7\] Zha, H\., Ma, T\., Yong, Y\., et al\. \(2026\)\. DisagFusion: Asynchronous Pipeline Parallelism and Elastic Scheduling for Disaggregated Diffusion Serving\. arXiv preprint arXiv:2605\.25550\.

\[8\] Yang, L\., Li, S\., Feng, T\., et al\. \(2026\)\. LegoDiffusion: Micro\-Serving Text\-to\-Image Diffusion Workflows\. arXiv preprint arXiv:2604\.08123\.

\[9\] Liu, F\., Zhang, S\., Wang, X\., et al\. \(2025\)\. Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model\. CVPR 2025\. [arXiv:2411\.19108](https://arxiv.org/abs/2411.19108)
