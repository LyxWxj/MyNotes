---
type: Note
related_to: "[[Parallel tech]]"
status: Active
---

# 数据并行与状态分片（DDP / ZeRO / FSDP / HSDP）

> 数据并行把 **batch 切分**到多张卡上，每卡一份完整模型副本、各自算梯度、再同步。问题在于**模型状态（参数、梯度、优化器状态）被冗余复制**。ZeRO/FSDP 家族的核心思想：**不复制，切分（partition）**。

## 0. 显存账本：三种"模型状态"

混合精度（FP16/BF16 计算 + FP32 主权重）训练时，每个参数对应：

| 模型状态 | 内容 | 每参数字节数 |
|---|---|---|
| 参数（FP16/BF16） | 前向/反向用的权重 | 2 B |
| 梯度（FP16/BF16） | 反向算出的梯度 | 2 B |
| 优化器状态（FP32） | Adam：主权重 4B + 一阶矩 4B + 二阶矩 4B | 12 B |

**合计 16 B/参数**（Adam 场景）。于是 7.5B 模型在 64 卡 DDP 下，光模型状态就要 120 GB/卡——其中 12/16 是优化器状态、14/16 是冗余的。

## 1. DDP（Distributed Data Parallel）

- 每卡**完整副本** + 各自处理一份数据 → 前向/反向 → **all-reduce 同步梯度** → 各自更新。
- 通信量：每步 **2 × 模型大小**（all-reduce 的收发）。
- 优点：实现简单、扩展性好（8 卡到几百卡都行）。
- 缺点：显存冗余严重；卡数越多，单卡瓶颈不变（只省时间不省显存）。

```
        卡0(全模型)     卡1(全模型)     卡2(全模型)     卡3(全模型)
           │ 前反向        │ 前反向        │ 前反向        │ 前反向
           └──────┬───────┴───────┬───────┴───────┬───────┘
                  └── all-reduce 梯度 ──→  每卡拿到全量梯度
```

## 2. ZeRO（Zero Redundancy Optimizer，DeepSpeed，2019）

ZeRO 不是新并行维度，而是**对 DP 的显存去冗余**：把三种模型状态按 DP 进程数 `N` 切分，谁持有谁负责更新。

| Stage | 切分内容 | 每参数显存（近似） | 通信量/步 |
|---|---|---|---|
| 0（=DDP） | 无 | 16 B | 2×模型 |
| 1 | 优化器状态 | 4 + 12/N B | 2×模型 |
| 2 | 优化器状态 + 梯度 | 2 + (2+12)/N B | 1×模型 |
| 3 | 优化器状态 + 梯度 + **参数** | (2+2+12)/N B | 1.5×模型 |

> N=64 时，ZeRO-3 把 7.5B 模型的模型状态从 120 GB 压到 ~1.9 GB/卡，显存瓶颈转移到激活上。

### 各 stage 的通信结构

- **ZeRO-1/2**：梯度用 `reduce-scatter`（只求自己的分片）→ 每卡更新自己的参数分片。通信量只有 DDP 一半。
- **ZeRO-3**：前向/反向前先 `all-gather` 全量参数，反向后用 `reduce-scatter` 收敛梯度分片。通信量 ≈ 1×（gather）+ 0.5×（scatter）= 1.5× 模型。

> [!NOTE]
> ZeRO-3 的代价：**每层都要一次 all-gather + 一次 reduce-scatter**，通信次数多、粒度细，跨节点时延迟敏感；好处是显存几乎与卡数成反比。

### ZeRO-Offload / ZeRO-Infinity

- **ZeRO-Offload（2020）**：把优化器状态和梯度放到 **CPU 内存**（CPU 算优化器更新），GPU 只留参数；适合单机多卡 + 大 CPU 内存。
- **ZeRO-Infinity（2021，SC'21 最佳论文）**：offload 扩展到 **NVMe SSD**，配合 DeepNVMe 高效 I/O 引擎 + 自动的 CPU/GPU/NVMe 三级调度，声称可训练**万亿参数**模型（只要 CPU/NVMe 够大）。
- 代价：走 PCIe/存储的带宽远低于 HBM，**只适合"显存实在不够"的场景**，训练吞吐会明显下降。

### ZeRO++（2023）

三个独立组件，目标是**砍掉 ZeRO-3 的通信开销**（合计通信量减少约 4×）：

| 组件 | 做法 | 效果 |
|---|---|---|
| **qwZ**（quantized weights） | 参数 all-gather 前做 block-wise INT8 量化 | 参数通信减半（FP16→INT8） |
| **hpZ**（hierarchical partitioning） | 分层切分：节点内 NVLink 全量 all-gather，跨节点只传分片 | 跨节点通信大幅下降 |
| **qgZ**（quantized gradients） | 用 all-to-all 量化梯度聚合替换 reduce-scatter | 梯度通信减半左右 |

## 3. FSDP（Fully Sharded Data Parallel，PyTorch，2022）

FSDP 是 **ZeRO-3 的 PyTorch 原生实现**（Meta 提出，与 ZeRO 殊途同归）：

- 把模型参数**展平（flatten）成 1D 缓冲**后按 rank 切分，每个 FSDP 单元在计算前 all-gather、计算后重新 shard。
- 通过 `auto_wrap_policy`（按层/模块包装）控制通信粒度：包装越细，显存越省、通信越多。
- 关键配置（FSDP1 时代）：
  - `sharding_strategy`：`FULL_SHARD`（=ZeRO-3）、`SHARD_GRAD_OP`（=ZeRO-2）、`NO_SHARD`（=DDP）、`HYBRID_SHARD`（=HSDP）。
  - `backward_prefetch`：反向时预取下一层参数，隐藏 all-gather 延迟。
  - `cpu_offload`：参数/梯度卸载到 CPU。
  - `mixed_precision`：`param_dtype=bf16, reduce_dtype=fp32` 是常见组合（梯度 fp32 归约保精度）。
  - `activation_checkpointing`：配合重计算进一步压显存。

## 4. FSDP2（PyTorch 2.4+，2024）

FSDP1 已被官方标记 deprecated，新代码用 **`fully_shard`**（composable API）：

- **逐参数分片**：不再展平，参数直接变成 **DTensor**（`Shard(dim=0)`），每个参数独立管理。
- **2D device mesh 原生支持 HSDP**：mesh 的第 0 维做分片（shard）、第 1 维做复制（replicate）。
- **免通信的分布式 state_dict**：保存/加载 checkpoint 不需要 all-gather 出全量参数（配合 Distributed Checkpoint, DCP）。
- **更干净的内存管理**：避免 `recordStream` 和 CPU 同步，显存占用更低且确定。
- **meta device 初始化**：`meta` 上建模型 → `fully_shard` → `to_empty` + `reset_parameters`，初始化阶段不占显存。
- **扩展点**：tensor subclass 可自定义 all-gather（float8 all-gather、NF4 QLoRA 等）。
- **与 TP 的组合**：先 `parallelize_module`（TP）再 `fully_shard`（FSDP），FSDP 会自动沿 mesh 未占用的轴切分——顺序必须是 **TP 在前、FSDP 在后**。

```python
from torch.distributed.fsdp import fully_shard

for layer in model.layers:        # 逐层包装，粒度控制显存/通信
    fully_shard(layer)
fully_shard(model)
```

## 5. HSDP（Hybrid Sharded Data Parallel）

**问题**：ZeRO-3/FSDP 的跨节点通信（每层 all-gather + reduce-scatter）在节点间带宽有限时是瓶颈。

**HSDP 思路**：混合两种策略——
- **节点内**：按 FSDP/ZeRO-3 方式**分片**（reduce-scatter 梯度）；
- **节点间**：按 DDP 方式**复制**（all-reduce 汇总各节点的梯度分片）。

```
┌─ 节点 A ─────────────┐   ┌─ 节点 B ─────────────┐
│ 卡0 卡1 卡2 卡3       │   │ 卡4 卡5 卡6 卡7       │
│ 分片: P0 P1 P2 P3     │   │ 分片: P0 P1 P2 P3     │
│ 节点内 reduce-scatter │   │ 节点内 reduce-scatter │
└──────┬───────────────┘   └──────┬───────────────┘
       └────── 跨节点 all-reduce（只有梯度分片）──────┘
```

- 效果：**跨节点通信量 ≈ 模型大小 × 分片因子**（而不是全量参数），NVLink 扛高频分片通信，IB 只做低频聚合。
- 实现：FSDP1 的 `HYBRID_SHARD`；FSDP2 的 2D mesh（`fully_shard(mesh=2d_mesh)`）；Megatron-Core 的 `--outer-dp-sharding-strategy no_shard`。
- 适用：**跨节点多机训练**的默认选择；单机内 FSDP 与 HSDP 等价。

## 6. Megatron-FSDP（Megatron-Core 的 DP 分片）

Megatron-Core 也内置了可配置的分片 DP：

- `--data-parallel-sharding-strategy`：
  - `optim` = ZeRO-1（只切优化器状态）
  - `optim_grads` = ZeRO-2
  - `optim_grads_params` = ZeRO-3
- `--num-distributed-optimizer-instances > 1`：层级式（hierarchical）DP，即节点内分片 + 节点间复制的 HSDP 变体。
- `--outer-dp-sharding-strategy optim`：**Hybrid-FSDP**——外层（跨节点）也切优化器状态，进一步省显存、代价是外层多一次通信。
- 与 TP/CP/EP 天然组合，是 Megatron 生态中"FSDP 轴"的标准入口。

## 7. 横向对比与选型

| 技术 | 分片粒度 | 通信量/步 | 跨节点友好度 | 适用场景 |
|---|---|---|---|---|
| DDP | 无 | 2×模型 | 高（单次大 all-reduce） | 模型单卡放得下 |
| ZeRO-1 | 优化器状态 | 2×模型 | 中 | 显存小紧张 |
| ZeRO-2 / FSDP SHARD_GRAD_OP | +梯度 | 1×模型 | 中 | 常见微调 |
| ZeRO-3 / FSDP FULL_SHARD / FSDP2 | 全部 | 1.5×模型 | 低（细粒度通信多） | 大模型预训练 |
| **HSDP**（HYBRID_SHARD / 2D mesh） | 节点内全分片 | 节点内 1.5× + 节点间 0.5× | **高** | 多机大模型默认 |
| Megatron-FSDP | 可配置 | 同上映射 | 高 | Megatron 生态 |

> [!TIP]
> 2025-2026 年的主流路线：**PyTorch 侧用 FSDP2 + HSDP（2D mesh）**；Megatron 生态用 **Megatron-FSDP + hierarchical DP**；两者都建议配 `bf16 + fp32 梯度归约 + activation checkpointing`。

## 参考

- DeepSpeed ZeRO 论文/文档：https://deepspeed.readthedocs.io/en/latest/zero3.html
- ZeRO++：https://www.deepspeed.ai/tutorials/zeropp/
- PyTorch FSDP2 教程：https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html
- Megatron-Core Parallelism Guide：https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html
