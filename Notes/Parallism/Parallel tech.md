---
type: Note
related_to: "[[data-parallelism-and-sharding]]"
status: Active
---

# 并行训练技术全景

> 大模型训练/推理的核心矛盾：**单卡放不下（显存）、算不快（算力）、传不动（带宽）**。并行技术就是围绕这三个瓶颈，把"一个模型"拆到"多张卡"上，同时尽量少付出通信代价。

## 并行维度总览

| 维度 | 切分对象 | 典型技术 | 解决什么 | 通信模式 |
|---|---|---|---|---|
| **数据并行 DP** | batch（数据） | DDP、ZeRO、FSDP、HSDP | 吞吐量、显存冗余 | all-reduce / reduce-scatter + all-gather |
| **张量并行 TP** | 单层权重/计算 | Megatron-TP | 单层放不下（hidden 过大） | 层内 all-reduce（NVLink） |
| **序列并行 SP/CP** | 序列（token） | Megatron-SP、Ulysses、Ring-Attention、USP | 长序列显存（激活） | all-to-all / ring p2p |
| **流水线并行 PP** | 层（深度） | GPipe、1F1B、Interleaved、ZeroBubble、DualPipe | 层数过多、显存分摊 | stage 间 p2p send/recv |
| **专家并行 EP** | MoE 专家 | DeepSpeed-MoE、Megatron-MoE、DeepEP | 专家数多、单卡放不下 | all-to-all（dispatch/combine） |
| **混合并行** | 多个维度叠加 | 3D/4D/5D 并行 | 千卡以上训练 | 复合通信 |
| **显存/通信优化** | 非并行辅助 | 激活重计算、offload、梯度压缩 | 压内存、砍通信 | — |

**GPU 数 = DP × TP × PP × CP × EP**（各维度正交，FSDP 通常看作 DP 轴上的参数分片）。

## 各维度笔记导航

- [[data-parallelism-and-sharding|数据并行与状态分片]] — DDP / ZeRO(0-3) / ZeRO-Offload / ZeRO-Infinity / ZeRO++ / FSDP / FSDP2 / HSDP / Megatron-FSDP
- [[tensor-and-sequence-parallelism|张量并行与序列并行]] — Megatron-TP / Sequence Parallelism / DeepSpeed-Ulysses / Ring-Attention / USP / Context Parallelism
- [[pipeline-parallelism|流水线并行]] — GPipe / PipeDream 1F1B / Interleaved VPP / ZeroBubble / V-Shape / DualPipe / 异步流水
- [[expert-parallelism-and-moe|专家并行与 MoE]] — EP 原理 / DeepSpeed-MoE / Megatron-MoE / DeepEP / Hybrid-EP / 负载均衡
- [[hybrid-parallelism-and-frameworks|混合并行与分布式框架]] — 3D/4D/5D 并行 / DTensor / GSPMD / torchtitan / Megatron-Core / DeepSpeed / 工业案例
- [[memory-and-communication-optimization|显存与通信优化]] — 激活重计算 / offload / 梯度累积 / 通信压缩与重叠

## 硬件与通信基础

> 集合通信是所有并行的底层地基，原语语义、算法与 NCCL 调优单独成册：[[collective-communication-index|集合通信与 NCCL 调优]]（`Notes/CollectiveCommunication/`）。

### 常见互联带宽（量级）

| 互联 | 典型带宽 | 说明 |
|---|---|---|
| 单卡 HBM | ~3-8 TB/s（H100 ~3.35 TB/s，B200 ~8 TB/s） | 片内显存带宽 |
| NVLink（机内） | ~900 GB/s（H100），1.8 TB/s（B200） | 同节点 GPU 间，TP 首选 |
| InfiniBand / RoCE（跨机） | 400-800 Gbps ≈ 50-100 GB/s | 跨节点，比 NVLink 慢一个量级 |
| PCIe | ~64-128 GB/s | 与 CPU 通信（offload 用） |

> [!IMPORTANT]
> 一切并行设计都服从"**通信拓扑匹配**"原则：高频通信的维度（TP、CP）放节点内 NVLink，低频/大块通信的维度（PP、EP、DP）放跨节点 IB。

### 核心集合通信原语

| 原语 | 作用 | 通信量（每 rank，大致） |
|---|---|---|
| `all-reduce` | 各 rank 数据求和并广播给所有人 | 2×数据量 |
| `reduce-scatter` | 各 rank 数据求和后**切分**，每人拿一块 | 1×数据量 |
| `all-gather` | 把各 rank 的分片拼成完整数据 | (N-1)/N×数据量 ≈ 1× |
| `all-to-all` | 每个 rank 给每个 rank 发不同数据 | 每人发 (N-1)/N×自己的数据 |
| `p2p send/recv` | 点对点传输 | 1×数据量 |

**关键洞察**：`all-reduce` 可以分解为 `reduce-scatter + all-gather`——这正是 DDP → ZeRO-2/3 的通信结构变化。

## 选型规则（由简到繁）

1. **先 DP**：数据并行永远是起点，模型放得下就用 DDP。
2. **显存不够 → FSDP/ZeRO-3**：分片参数/梯度/优化器状态；跨节点用 **HSDP**（节点内分片 + 节点间复制）省跨节点通信。
3. **单层放不下 / hidden 太大 → TP**（配合 Megatron-SP）。
4. **层数太深（50+ 层）→ PP**（1F1B + Interleaved），注意气泡与通信延迟。
5. **序列太长（8K+）→ CP/SP**：Ulysses（all-to-all）或 Ring-Attention（p2p，更通用）。
6. **MoE 模型 → EP**：all-to-all 分发 token，配 DeepEP 之类内核。
7. **千卡以上 → 混合并行**：按"TP/CP 进节点、PP/EP 出节点"的拓扑映射排布 3D/4D/5D mesh。
8. **还是放不下 → 显存优化兜底**：激活重计算、CPU/NVMe offload、梯度累积、FP8。

## 一个直觉模型：显存从哪来，往哪去

```
单卡训练一版模型的显存 = 模型状态（参数+梯度+优化器状态）
                      + 激活（前向中间结果，反向要用）
                      + 临时缓冲（通信、kernel scratch）
                      + 碎片
```

- **模型状态** → ZeRO/FSDP 分片、offload
- **激活** → SP/CP 切序列、激活重计算、激活 offload
- **通信** → 拓扑匹配、压缩、重叠（async）

详细拆解见 [[memory-and-communication-optimization|显存与通信优化]]。
