---
type: Note
related_to: "[[tensor-and-sequence-parallelism]]"
status: Active
---

# 流水线并行（GPipe / 1F1B / Interleaved / ZeroBubble / DualPipe / Chunk 化 / 动态 PP）

> 把模型**按层切成若干 stage**，每个 stage 放一张（或一组）GPU。数据像流水线一样流经各 stage。核心矛盾是**流水线气泡（bubble）**——设备等数据时无事可做。

## 0. 基本概念

- **Stage**：一段连续的层，分配给一个设备（常与 TP 组合：一个 PP stage = 一组 TP 卡）。
- **Micro-batch**：一个大 batch 切成多个微批次，逐个送进流水线，让各 stage 并行干活。
- **Bubble 率**：理想情况下所有 stage 满负荷；实际因"先灌满再排空"，前后有空洞。

## 1. GPipe（2019，Google）

- 调度：**fill-drain**——先顺序做完全部 micro-batch 的前向（fill），再统一做反向（drain）。
- 气泡率：`(P-1) / (M+P-1) ≈ (P-1)/M`（P=stage 数，M=micro-batch 数）。M 越大气泡越小，但**内存也越大**。
- 致命缺点：要保存**全部 M 个 micro-batch 的激活**，激活内存随 M 线性增长。

```
stage0: F1 F2 F3 F4 █ B4 B3 B2 B1 █
stage1: █ F1 F2 F3 F4 █ B4 B3 B2 B1
stage2: ██ F1 F2 F3 F4 ██ B4 B3 B2 B1
         └─ 气泡 ─┘           └─ 气泡 ─┘
```

## 2. PipeDream / 1F1B（2019/2020）

**1F1B（One-Forward-One-Backward）**：进入稳态后，每个 stage **交替执行"一个前向 + 一个反向"**，气泡大小与 GPipe 相同，但：

- **激活内存降到 O(P)**（每个 stage 只需保存流水线内 P 个 micro-batch 的激活，而非 M 个）——这是它取代 GPipe 的根本原因。
- PipeDream 原版为异步（每个 stage 不等 flush）但需要 **weight stashing**（各 micro-batch 用各自版本权重），实现复杂。
- **PipeDream-Flush / Megatron 1F1B**：定期 flush 保证权重一致，同步语义 + 低内存，成为工业标准。

```
stage0: F1 F2 F3 F4 B1 B2 B3 B4      (稳态时前向反向交错)
stage1:    F1 F2 F3 F4 B1 B2 B3 B4
stage2:       F1 F2 F3 F4 B1 B2 B3 B4
```

## 3. Interleaved / Virtual Pipeline（VPP，Megatron，2021）

把每个 stage 的层再切成 `v` 份**虚拟 stage**，交替分配（stage0 拿 1, 1+v, 1+2v... 层），1F1B 在虚拟 stage 间交错：

- 气泡率降到 `≈ (P-1)/(v·M)`，**v 越大气泡越小**（P=4, v=2 时气泡减半）。
- 代价：stage 间 p2p 通信次数增加 v 倍，层间依赖复杂化。
- 工业界普遍使用（Megatron `--num-layers-per-virtual-pipeline-stage`，DeepSeek、各家预训练都在用）。

## 4. Chunk-Based Pipeline Parallelism（chunk 化流水线调度）

**思想**：把"调度单元"从【整组层 × 整 micro-batch】细化为 **chunk**（更小的计算块），让流水线粒度不再受 micro-batch 数量限制。chunk 化体现在两个维度：

- **层维度 chunk（virtual pipeline）**：VPP 把每个 stage 的层切成 v 份虚拟 stage，本质就是"层 chunk"交替分配；DeepSeek-V3 的 profile 数据显示每个 PP chunk 约含 4 个 MoE 层，chunk 边界同时也是插入通信（all-to-all）的重叠窗口。
- **计算阶段维度 chunk（F/B/W）**：把一次 forward/backward 拆成更小的阶段块分别调度。ZeroBubble 把 backward 拆成 B（backward-data）与 W（backward-weight）；DualPipe 用四个符号描述 chunk：**F**=forward chunk、**B**=完整 backward chunk、**W**=仅权重梯度 backward chunk、**F&B**=互相重叠的一对前向/后向 chunk。

DualPipe README 的气泡公式（PP=stage 数，需为偶数；同 stage 数下比较）：

| 方法 | 气泡 | 每设备参数 | 每设备激活 |
|---|---|---|---|
| 1F1B | (PP-1)(F+B) | 1× | PP |
| ZB1P | (PP-1)(F+B-2W) | 1× | PP |
| DualPipe | (PP/2-1)(F&B+B-3W) | 2× | PP+1 |
| DualPipeV | (PP/2-1)(F&B+B-3W) | 2× | 设备数减半 |

chunk 化的收益：

- **气泡更小**：调度以 chunk 为单位，气泡率由 chunk 数决定，不必靠增大 micro-batch 换吞吐；
- **计算-通信重叠窗口更大**：chunk 边界可插入 p2p / all-to-all，把通信藏进相邻 chunk 的计算里（DualPipe 的关键机制）；
- **变长输入友好**：chunk 大小可按序列长度/显存动态调整（DynaPipe、InfiniPipe 的 token-level PP 都依赖"序列切成 chunk"这一抽象）；
- **与 gradient checkpointing 结合**：可在 chunk 粒度做 checkpoint（如 InfiniPipe 的阶段感知 chunk 级自适应 checkpointing）。

> [!NOTE]
> 推理侧也有"chunked prefill"：把长 prefill 请求切成多个 chunk 与 decode 交织（vLLM 等已标配），避免长 prefill 独占 GPU。与 PP 结合时，朴素做法是把每个 chunk 当作独立请求依次穿过整个流水线——"chunk 作为调度单元"同样适用于推理场景。

## 5. ZeroBubble（2023/2024）

**洞察**：反向传播可拆成 **B（backward-data，算梯度）** 和 **W（backward-weight，算权重梯度）** 两部分；1F1B 把 B/W 捆在一起，导致调度的"形状"受限。ZeroBubble 把 D/B/W 三段**分别调度**，可以做到：

| 变体 | 气泡 | 激活内存（相对 1F1B） |
|---|---|---|
| ZB-H1 | ≈0 | 1× |
| ZB-H2 | ≈0 | 2× |
| **ZB-V（V-shape）** | ≈0 | 1× |

- ZB-V 采用 **V 形（波状）调度**：各 stage 前向/反向的方向交错，形成 V 形波形，零气泡且内存不高，是实践中最受好评的变体。
- 注意：梯度裁剪/NaN 检查的全局 all-reduce 会破坏零气泡，需要把这类同步移到更新后异步做。

## 6. V-Shape 流水线与 DualPipe（DeepSeek-V3，2025）

- **V-Shape 调度**：由 ZB-V 等演进的"波状/双向"调度家族，前向和反向在流水线里同向推进，消除对称气泡。
- **DualPipe（DeepSeek-V3）**：在 V-Shape 基础上做**双向双流**——前向与后向**在时间上重叠**，且把**计算与通信（尤其 MoE 的 all-to-all）完全重叠**：
  - 气泡率比 1F1B 显著下降（约 -78%）；
  - 配合 EP 的 dispatch/combine 通信隐藏，是 671B MoE 在 2048 卡 H800 上实现高 MFU 的关键。
  - 每个 PP 内部还叠加了"双向推进 + FP8"，细节见 DeepSeek-V3 技术报告。

## 7. 异步流水线（Async Pipeline）

- 思路：不做 flush，每个 stage 拿到数据就干活，消除气泡；但权重版本不一致，需要**权重预测/延迟补偿**（如 Megatron-Core 的 `--async-pipeline`、字节 TorchTP 等）。
- 适用：PP 很深、同步 1F1B 气泡无法接受、且能容忍优化语义变化时。

## 8. 静态 PP vs 动态 PP

**静态 PP**：stage 划分（层→设备映射）、micro-batch 大小与数量、调度时序在训练/部署前一次性定死。GPipe、1F1B、VPP、ZeroBubble、DualPipe 都属于这一类——调度图再精巧，运行时执行的也是**固定 schedule**。隐含假设：各 micro-batch 执行时间接近均匀、设备同构、序列长度接近。一旦假设不成立：

- 序列长度异构（多任务、长上下文）→ micro-batch 执行时间抖动，1F1B 因相邻 stage 紧耦合而 **blocking**（设备空等），气泡重新出现；
- 慢节点 / straggler、异构或共享集群 → 固定划分导致负载不均衡；
- 长上下文 → batch-level PP 每个 micro-batch 的激活峰值高，显存成为瓶颈。

**动态 PP**：按数据分布或运行时状态**实时调整**流水线的某一层抽象（micro-batch 形状、chunk 大小、stage 的层分配、token/batch 粒度）：

| 系统 | 场景 | 动态点 | 效果 / 备注 |
|---|---|---|---|
| DynaPipe（EuroSys'24） | 多任务变长序列训练 | 按长度排序样本后，用动态规划切成变长 micro-batch；自适应控制注入时机（基于 safety stock 思想防阻塞）；显存感知 | T5 4.39× / GPT 3.25×（vs packing 基线） |
| InfiniPipe / EPP（2025） | 长上下文训练 | 动态编排 batch-level PP（短序列打包）与 token-level PP（长序列切 token chunk）；阶段感知 chunk 级自适应梯度 checkpointing | 最高 1.69× |
| DynaPipe（NeurIPS'25） | LLM 推理 | 实时预测各 stage 执行延迟，**动态重分配层**（层迁移）平衡计算 | 应对请求长度与负载波动 |
| Adaptra（2025） | 推理/训练 | straggler 感知，运行时弹性重配置流水线拓扑 | 容错 / 异构 |
| Malleus（2024） | 异构集群训练 | 弹性混合并行，运行时调整 PP/DP/TP 各维度资源分配 | 资源异质 |

关键权衡：

| 维度 | 静态 PP | 动态 PP |
|---|---|---|
| 划分时机 | 部署前固定 | 运行时调整（micro-batch / chunk / 层分配） |
| 调度假设 | 执行时间均匀、设备同构 | 显式建模抖动、异构、负载波动 |
| 优势 | 简单可预测、易实现易调试、通信模式确定 | 负载均衡、长上下文/变长场景吞吐高、能容忍慢节点 |
| 代价 | 对变长序列/慢节点/波动敏感，气泡回升 | 调度复杂、内存与通信不确定、易死锁、实现调试难 |
| 代表 | GPipe / 1F1B / VPP / ZeroBubble / DualPipe | 训练：DynaPipe、EPP；推理：层重分配、Adaptra、Malleus |

> [!TIP]
> 生产默认仍是**静态 PP**（可预测、可调试）；当序列长度分布极不均匀（多任务、长上下文）、集群异构或共享干扰明显、需要弹性伸缩时，再考虑动态方案。工程上常见"半动态"过渡：训练侧按数据分布自动 packing/切 chunk，推理侧按请求长度动态切分 chunked prefill。

## 9. 显存与通信账本

- **参数**：每卡只持有自己 stage 的层 → 显存 ≈ 总参数/P。
- **激活**：1F1B 只需 P 个 micro-batch 的激活（vs GPipe 的 M 个）。
- **通信**：stage 间 p2p（每次传激活/梯度），**量小但次数多、延迟敏感**；PP 是最不适合跨慢速网络的维度之一（通常放节点间但要求 IB 低延迟）。
- **吞吐约束**：`global batch = micro-batch × M × DP`；PP 对 global batch 有下限要求（M 太小气泡大）。

## 10. 对比总表

| 方法 | 年份 | 气泡率（近似） | 激活内存 | 备注 |
|---|---|---|---|---|
| GPipe | 2019 | (P-1)/M | 高（M 份） | 开创性，内存硬伤 |
| PipeDream 1F1B | 2019-2020 | (P-1)/M | 低（P 份） | 工业标准 |
| Interleaved VPP | 2021 | (P-1)/(v·M) | 低 | 通信次数 ×v |
| ZeroBubble H1/V | 2023 | ≈0 | 1× | 调度 D/B/W |
| V-Shape / DualPipe | 2024-2025 | 极低 | 低 | 双向 + 通信重叠，MoE 友好 |
| 异步流水 | 2023+ | 0（名义） | 中 | 权重不一致，需预测 |

> 上表均为**静态调度**；动态 PP（DynaPipe、EPP、推理侧层重分配等）没有固定气泡公式，见第 8 节。

> [!TIP]
> 选型：中小模型/快速迭代用 **1F1B + Interleaved**；超大 MoE、追求极致 MFU 用 **DualPipe/V-Shape 类双向流水**；PP 深度大且网络好可试**异步流水**。

## 参考

- GPipe：https://arxiv.org/abs/1811.06965 ；PipeDream：https://arxiv.org/abs/1806.03377
- Megatron-LM（含 Interleaved）：https://arxiv.org/abs/1909.08053
- ZeroBubble：https://arxiv.org/abs/2401.10241
- DualPipe / DeepSeek-V3（含 1F1B/ZB1P/DualPipe/DualPipeV 气泡对比表）：https://github.com/deepseek-ai/DualPipe ；https://arxiv.org/abs/2412.19437
- V-Shape 相关讨论：https://sail.sea.com/blog/articles/63
- Chunked prefill（推理）：https://docs.vllm.ai/en/latest/features/chunked_prefill.html
- DynaPipe（动态 micro-batching，训练）：https://arxiv.org/abs/2311.10418 ；代码 https://github.com/awslabs/optimizing-multitask-training-through-dynamic-pipelines
- InfiniPipe / EPP（token 级 + batch 级弹性 PP）：https://arxiv.org/abs/2509.21275 ；代码 https://github.com/wsjdsg/InfiniPipe-code
- DynaPipe（推理动态层重分配，NeurIPS 2025）：https://neurips.cc/virtual/2025/loc/san-diego/poster/119240
- Adaptra（straggler 弹性重配置）：https://arxiv.org/abs/2504.19232 ；Malleus（异质资源弹性混合并行）：https://arxiv.org/abs/2410.13333
