---
type: Note
related_to: "[[data-parallelism-and-sharding]]"
status: Active
---

# 张量并行与序列并行（TP / SP / CP）

> 数据并行切"数据"，这里切**计算本身**：张量并行把单层权重拆到多卡；序列并行把 token 序列拆到多卡。两者都是**细粒度、高频率通信**的维度，必须放在 NVLink 节点内。

## 1. 张量并行（Tensor Parallelism，Megatron-LM，2019）

### 原理

Transformer 的线性层 `Y = X·W` 可以按矩阵乘法切分：

- **列并行（Column Parallel）**：`W` 按列切成 `[W1, W2]`，每卡算 `Y_i = X·W_i`，然后 **all-reduce** 拼接结果。
- **行并行（Row Parallel）**：`W` 按行切，每卡拿不同的 `X_i`，算完直接拼接（无需 all-reduce）。

Megatron 的经典布局（每个 Transformer 块）：

```
          X
          │
     ┌────┴────┐
   QKV 列并行（3×H/TP）    ← 输出 all-reduce
     └────┬────┘
        attention
     ┌────┴────┐
  MLP 列并行 → GELU → MLP 行并行（输出拼接）
     └────┬────┘
          Y
```

- 每层通信：**2 次 all-reduce**（attention 输出 + MLP 输出），通信量 O(hidden)。
- 关键约束：**hidden size 必须能被 TP 整除**；TP 越大，单卡显存越低、但 all-reduce 次数线性增多。
- 适用：单层参数放不下一张卡、hidden 特别大（4096+）时；**必须配 NVLink**（每层都有集合通信）。

> [!NOTE]
> TP 是所有并行维度里**通信最频繁**的（每层 2 次同步），所以 TP size 通常 ≤ 8（H100 单节点 8 卡），跨节点 TP 是禁忌。

## 2. 序列并行（Megatron Sequence Parallelism，2023）

**动机**：TP 只在 LayerNorm/Dropout 处仍有完整序列激活，长序列时激活显存依旧爆炸。

**做法**：把 **LayerNorm 和 Dropout 的序列维度也切分**（`seq/TP`），只在这些算子上做 `all-gather + reduce-scatter`：

- 每个 Transformer 块增加一次 `all-gather`（喂给注意力/MLP 前恢复序列）+ 一次 `reduce-scatter`（LN/Dropout 前切回）。
- 效果：激活内存从 `O(seq × hidden)` 降到 `O(seq × hidden / TP)`（对 LN/Dropout 部分），通信量几乎与 TP 相同。
- **必须与 TP 一起开**（共用同一进程组），Megatron 系推荐 TP 时总是开 `--sequence-parallel`。

## 3. DeepSpeed-Ulysses（SP-Ulysses，2023）

**思想**：把序列切成 `SP` 份，用 **all-to-all** 把 QKV 重排成"每个 rank 持有全部序列、部分头"：

```
rank i 持有序列块 [i·L/SP, (i+1)·L/SP] 的全部头
   ── all-to-all ──▶  rank i 持有全部序列、头的子集 [i·H/SP, (i+1)·H/SP]
```

- 每层通信：**2 次 all-to-all**（QKV 分发 + 输出回收）。
- 限制：`序列长度 % SP == 0` 且 `注意力头数 % SP == 0`（否则要高级 uaa 模式）。
- 优点：单次大块 all-to-all、实现简单、对注意力实现（FlashAttention）透明；能利用 NVLink 高带宽。
- 缺点：SP 不能超过头数；短序列下 all-to-all 开销占比高。

## 4. Ring-Attention（2023）

**思想**：把序列切成块，每个 rank 持有 `Q` 块 + 初始 `KV` 块，KV 块沿环**逐轮 p2p 传递**，每轮算部分注意力并累加：

- 优点：**序列长度不限**（不需要整除）、p2p 可与计算重叠、SP 规模可很大（跨节点也行）。
- 缺点：块级循环开销（短序列明显）、负载不均（长序列场景各块 KV 长度不同）。
- **Zigzag 优化（2024）**：按之字形重新分配 token 块，让不同长度序列的 KV 负载均衡，成为长上下文 CP 的主流实现。

## 5. USP（Unified Sequence Parallelism，2024）

DeepSeek 提出，**把 Ulysses 和 Ring 合并**：

- 序列维度用 Ring 切分、注意力头维度用 Ulysses（head parallel）切分，构成 2D 网格；
- 综合两者优点：任意序列长度 + 负载均衡 + 更大 SP 规模（`SP = ring × head`）。
- 实际效果：长序列下通信量、扩展性都优于单独 Ulysses/Ring，DeepSeek-V3 训练即采用类似思路（USP + 其他优化）。

## 6. Context Parallelism（CP，Megatron-Core / NeMo / Llama 3）

**CP 是"序列并行"在工业界的通用叫法**，目的都是长序列（8K+）训练/推理：

- Megatron-Core：`--context-parallel-size N`，通信类型可选 `p2p`（ring 风格）或 `allgather`；
- 与 TP 的区别：TP 切 hidden/权重，CP 切序列；CP 的通信频率低于 TP，可以适度跨节点；
- 搭配规则：`TP + CP` 组合时，通常 CP 放外层（跨节点）、TP 放内层（节点内）；
- Llama 3 / NeMo 的长上下文训练标配：先预训练短上下文，再用 CP 扩展到 128K+。

## 7. 对比与选型

| 技术 | 切什么 | 每层通信 | 序列长度约束 | 负载均衡 | 适用 |
|---|---|---|---|---|---|
| Megatron-TP | 权重/计算 | 2×all-reduce | hidden % TP = 0 | 天然均衡 | hidden 大、单层放不下 |
| Megatron-SP | LN/Dropout 激活 | +1 all-gather +1 reduce-scatter | 无 | 均衡 | TP 必配，压激活 |
| Ulysses | 序列+头 | 2×all-to-all | seq % SP = 0、head % SP = 0 | 均衡 | 中等序列、NVLink |
| Ring-Attention | 序列 | 环 p2p ×2 | 无 | 需 zigzag | 任意长序列、跨节点 |
| USP | 序列+头 | all-to-all + p2p 混合 | 宽松 | 均衡 | 超长序列（DeepSeek） |
| CP（通用） | 序列 | p2p / all-gather | 依实现 | 依实现 | 8K+ 长上下文 |

> [!TIP]
> 经验法则：**短序列选 Ulysses（低开销），长序列/任意长度选 Ring（zigzag）**；两头都要就上 USP。TP 始终优先占满节点内 NVLink。

## 参考

- Megatron-LM：https://arxiv.org/abs/1909.08053 ；序列并行：https://arxiv.org/abs/2205.05198
- DeepSpeed-Ulysses：https://arxiv.org/abs/2309.14509
- Ring Attention：https://arxiv.org/abs/2310.01889 ；zigzag：https://arxiv.org/abs/2408.10188
- USP：https://arxiv.org/abs/2405.07719
- Megatron-Core CP 指南：https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html
