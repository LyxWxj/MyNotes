---
type: Note
related_to: "[[hybrid-parallelism-and-frameworks]]"
status: Active
---

# 显存与通信优化（重计算 / Offload / 梯度累积 / 通信压缩与重叠）

> 并行解决"怎么分摊"，这里的技巧解决"怎么省"：显存实在不够时，用**时间换空间**（重计算、offload）；通信成为瓶颈时，用**压缩/重叠**换带宽。ZeRO 系列本质上也属于这一类（省显存）。

## 1. 显存构成（哪里在吃显存）

| 项目 | 大小（近似） | 对策 |
|---|---|---|
| 模型状态（参数+梯度+优化器） | 16 B/参数（Adam） | ZeRO/FSDP 分片、offload |
| 激活（前向中间结果） | O(层数 × seq × hidden)，attention 还有 O(seq²) | SP/CP 切分、重计算、offload |
| 临时缓冲 | 通信缓冲、kernel scratch | 分桶、复用 |
| 碎片 | 动态分配导致 | 预分配、统一内存池 |

## 2. 激活重计算（Activation / Gradient Checkpointing）

- **做法**：前向时**不保存**部分激活，反向用到时**重算一遍**。
- **收益**：显存从 `O(每层激活 × 层数)` 降到 `O(每层激活 × checkpoint 间隔)`（通常选 Transformer 块边界）。
- **代价**：约 **30-40% 的额外前向计算**（时间换空间）。
- **选择性重计算（Selective）**：Megatron 的实践——只重算 **attention 区域**（显存大户）而不重算 MLP，兼顾显存与速度。
- 常见组合：FSDP + activation checkpointing + 大 batch，是 8B-70B 级预训练的默认配置。

## 3. 激活 Offload（前向卸载，反向预取）

- 前向把激活搬到 **CPU 内存**（或异构存储），反向开始前按需**预取**回 GPU；
- 2026 年起 torch.compile 已支持 **SAC + offload 策略**（静态激活切分 + 图级卸载调度），torchtitan 可直接 `--compile.sac_and_offload` 启用；
- 与重计算的区别：offload 不重算、但吃 PCIe 带宽；两者可混合（重计算"贵"的层、offload"便宜"的层）。

## 4. CPU / NVMe 参数 Offload

- **ZeRO-Offload / ZeRO-Infinity**：优化器状态→CPU/NVMe，GPU 只留参数与计算（详见 [[data-parallelism-and-sharding|数据并行与状态分片]]）。
- **FSDP CPU offload**：`CPUOffloadPolicy()`，参数/梯度/优化器可全放 CPU，配 `torch.compile` 有额外收益。
- 代价：走 PCIe（~64 GB/s）比 HBM（~3 TB/s）慢 1-2 个数量级，只作兜底。

## 5. 梯度累积（Gradient Accumulation）

- 把大 batch 拆成若干小步，**累加梯度后再更新**——显存不变、等效 batch 变大；
- 收益：模拟大 global batch、减少更新频率（顺带降低同步开销）；
- 注意：与 PP 的 micro-batch 是两个概念；梯度累积会延迟更新，需配合正确的 loss 缩放。

## 6. 通信优化

### 6.1 重叠（Overlap）

- **异步 all-reduce**：DDP 默认按梯度分桶（bucket），反向算到哪、reduce 到哪，与计算重叠；
- **FSDP/ZeRO 预取**：前向/反向前提前 all-gather 下一层参数（FSDP2 的 implicit/explicit prefetching）；
- **TP 通信重叠**：Megatron `--tp-comm-overlap`；
- **MoE 专用**：DeepEP 用 SM 分区把 dispatch/combine 与专家 GEMM 重叠；
- **流水线**：DualPipe 把 PP 通信与 EP 通信整体重叠。

### 6.2 压缩（Compression）

| 技术 | 压缩对象 | 效果 | 出处 |
|---|---|---|---|
| **1-bit Adam** | 梯度（误差反馈 EF 补偿） | 通信量降 ~5-16×，收敛不变 | DeepSpeed 2020 |
| **qwZ / qgZ（ZeRO++）** | 参数/梯度 INT8 量化 | 通信总量 ~4× 下降 | DeepSpeed 2023 |
| **Top-K 稀疏化 + EF** | 梯度稀疏化 | 大幅降通信 | TernGrad / EF 系列 |
| FP8 训练 | 计算+通信双降 | DeepSeek-V3 首个超大规模验证 | DeepSeek 2024 |

> [!WARNING]
> 梯度压缩会引入误差，需配合**误差反馈（error feedback）**保证收敛；精度敏感场景（小模型、RL 训练）慎用激进压缩。

### 6.3 拓扑与调优

- 集合通信走 **NCCL**：`NCCL_P2P_LEVEL`、`NCCL_SHM_*`、多 rail IB 绑定等直接影响吞吐；
- **层级化通信**：HSDP 把 reduce-scatter 压到节点内、all-reduce 只在节点间做小张量；
- **通信量最小化设计**：能用 reduce-scatter 不用 all-reduce（省一半）、能用 p2p 环不用 all-to-all（负载可控时）。

## 7. 组合路线图（显存不够时的升级路径）

```
模型放得下 → 直接 DDP + 梯度累积
    ↓ 显存吃紧
FSDP2（bf16 + fp32 reduce）+ activation checkpointing
    ↓ 还吃紧
+ CPU offload（参数/优化器） 或 HSDP（跨节点）
    ↓ 还吃紧
TP/SP 切单层与序列 → PP 切层数 → EP（MoE）
    ↓ 终极方案
ZeRO-Infinity 类 NVMe offload（吞吐换规模）
```

## 8. 度量指标

- **MFU（Model FLOPs Utilization）**：实际算力 / 理论峰值，衡量并行+通信总效率；
- **通信占比**：通信时间 / 迭代时间（profiling 工具：PyTorch profiler、Nsight Systems、NCCL 计时）；
- **每卡显存水位**：`nvidia-smi` / memory snapshot，定位是状态、激活还是缓冲在涨。

## 参考

- Activation Checkpointing：https://arxiv.org/abs/1604.06174 ；Megatron selective：https://arxiv.org/abs/2205.05198
- ZeRO-Offload/Infinity：https://www.deepspeed.ai/tutorials/zero-offload/ ；https://arxiv.org/abs/2104.07857
- 1-bit Adam：https://www.deepspeed.ai/2020/09/08/onebit-adam-blog-post.html
- torch.compile SAC/offload（torchtitan devlog）：https://docs.pytorch.org/devlogs/distributed/2026-06-23-cpu-offloading/
- DeepSeek-V3（FP8）：https://arxiv.org/abs/2412.19437
