---
type: Note
related_to: "[[tensor-and-sequence-parallelism]]"
status: Active
---

# 流水线并行（GPipe / 1F1B / Interleaved / ZeroBubble / DualPipe）

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

## 4. ZeroBubble（2023/2024）

**洞察**：反向传播可拆成 **B（backward-data，算梯度）** 和 **W（backward-weight，算权重梯度）** 两部分；1F1B 把 B/W 捆在一起，导致调度的"形状"受限。ZeroBubble 把 D/B/W 三段**分别调度**，可以做到：

| 变体 | 气泡 | 激活内存（相对 1F1B） |
|---|---|---|
| ZB-H1 | ≈0 | 1× |
| ZB-H2 | ≈0 | 2× |
| **ZB-V（V-shape）** | ≈0 | 1× |

- ZB-V 采用 **V 形（波状）调度**：各 stage 前向/反向的方向交错，形成 V 形波形，零气泡且内存不高，是实践中最受好评的变体。
- 注意：梯度裁剪/NaN 检查的全局 all-reduce 会破坏零气泡，需要把这类同步移到更新后异步做。

## 5. V-Shape 流水线与 DualPipe（DeepSeek-V3，2025）

- **V-Shape 调度**：由 ZB-V 等演进的"波状/双向"调度家族，前向和反向在流水线里同向推进，消除对称气泡。
- **DualPipe（DeepSeek-V3）**：在 V-Shape 基础上做**双向双流**——前向与后向**在时间上重叠**，且把**计算与通信（尤其 MoE 的 all-to-all）完全重叠**：
  - 气泡率比 1F1B 显著下降（约 -78%）；
  - 配合 EP 的 dispatch/combine 通信隐藏，是 671B MoE 在 2048 卡 H800 上实现高 MFU 的关键。
  - 每个 PP 内部还叠加了"双向推进 + FP8"，细节见 DeepSeek-V3 技术报告。

## 6. 异步流水线（Async Pipeline）

- 思路：不做 flush，每个 stage 拿到数据就干活，消除气泡；但权重版本不一致，需要**权重预测/延迟补偿**（如 Megatron-Core 的 `--async-pipeline`、字节 TorchTP 等）。
- 适用：PP 很深、同步 1F1B 气泡无法接受、且能容忍优化语义变化时。

## 7. 显存与通信账本

- **参数**：每卡只持有自己 stage 的层 → 显存 ≈ 总参数/P。
- **激活**：1F1B 只需 P 个 micro-batch 的激活（vs GPipe 的 M 个）。
- **通信**：stage 间 p2p（每次传激活/梯度），**量小但次数多、延迟敏感**；PP 是最不适合跨慢速网络的维度之一（通常放节点间但要求 IB 低延迟）。
- **吞吐约束**：`global batch = micro-batch × M × DP`；PP 对 global batch 有下限要求（M 太小气泡大）。

## 8. 对比总表

| 方法 | 年份 | 气泡率（近似） | 激活内存 | 备注 |
|---|---|---|---|---|
| GPipe | 2019 | (P-1)/M | 高（M 份） | 开创性，内存硬伤 |
| PipeDream 1F1B | 2019-2020 | (P-1)/M | 低（P 份） | 工业标准 |
| Interleaved VPP | 2021 | (P-1)/(v·M) | 低 | 通信次数 ×v |
| ZeroBubble H1/V | 2023 | ≈0 | 1× | 调度 D/B/W |
| V-Shape / DualPipe | 2024-2025 | 极低 | 低 | 双向 + 通信重叠，MoE 友好 |
| 异步流水 | 2023+ | 0（名义） | 中 | 权重不一致，需预测 |

> [!TIP]
> 选型：中小模型/快速迭代用 **1F1B + Interleaved**；超大 MoE、追求极致 MFU 用 **DualPipe/V-Shape 类双向流水**；PP 深度大且网络好可试**异步流水**。

## 参考

- GPipe：https://arxiv.org/abs/1811.06965 ；PipeDream：https://arxiv.org/abs/1806.03377
- Megatron-LM（含 Interleaved）：https://arxiv.org/abs/1909.08053
- ZeroBubble：https://arxiv.org/abs/2401.10241
- DualPipe / DeepSeek-V3：https://github.com/deepseek-ai/DualPipe ；https://arxiv.org/abs/2412.19437
- V-Shape 相关讨论：https://sail.sea.com/blog/articles/63
