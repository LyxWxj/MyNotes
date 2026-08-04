---
type: Note
related_to: "[[expert-parallelism-and-moe]]"
status: Active
---

# 混合并行与分布式框架（3D/4D/5D、DTensor、torchtitan、Megatron-Core）

> 单卡 8 层、单节点 8 卡时，一种并行就够了；千卡训练 671B 模型时，**所有维度一起上**。混合并行的本质：把 GPU 组织成多维 mesh，每个维度负责一种切分。

## 1. 从 1D 到 5D

| 名称 | 维度组合 | 总卡数 | 典型出处 |
|---|---|---|---|
| 1D | DP（或纯 FSDP） | DP | 入门 |
| 2D | DP × TP / FSDP × TP | DP·TP | FSDP2+TP、Fabric |
| 3D | DP × TP × PP | DP·TP·PP | Megatron-DeepSpeed |
| 4D | FSDP × TP × PP × CP | DP·TP·PP·CP | Llama 3、torchtitan |
| 5D/6D | + EP（MoE）+ 更多轴 | ×EP | DeepSeek-V3、Megatron-Core |

**核心公式**：`总 GPU 数 = TP × PP × CP × EP × DP`（FSDP 不增加卡数，它只是 DP 轴上的分片方式）。

## 2. 拓扑映射原则

```
4D 例子：128 卡 = TP4 × PP4 × CP2 × DP4

卡 0-3   TP 组（NVLink，最内层）
卡 0-7   CP 组（节点内优先）
卡 0-31  PP 组（尽量节点间均匀）
卡 0-127 DP/FSDP 组（跨节点，通信最少）
```

- **频率越高越往里放**：TP/CP（每层通信）→ 节点内 NVLink；PP/EP/DP（低频或大块）→ 跨节点 IB。
- 3D/4D 并行把每维度进程组独立建好（`process_group` / `DeviceMesh`），通信互不干扰。

## 3. 抽象层：DeviceMesh / DTensor / GSPMD

### PyTorch：DeviceMesh + DTensor（2023-2026 主线）

- **DeviceMesh**：把 N 维 GPU 网格（如 `mesh = DeviceMesh("cuda", [[0,1],[2,3]])`）显式建模；
- **DTensor**：张量级切分描述，三种 placement：
  - `Shard(dim)`：沿某维切分；
  - `Replicate()`：复制；
  - `Partial()`：部分求和（梯度归约中间态）。
- **SPMD 语义**：同一份代码跑在所有 rank，算子按 placement 自动生成通信（如 TP 的 all-reduce、FSDP 的 gather/scatter）。
- **组合**：`parallelize_module`（TP）+ `fully_shard`（FSDP2）+ PP schedule + CP，全部构建在 DTensor/DeviceMesh 上——FSDP2 的参数本身就是 DTensor。
- 配套：**Distributed Checkpoint（DCP）** 保存/恢复任意并行布局的 checkpoint，支持并行度变化后重载（universal checkpointing）。

### JAX/XLA：GSPMD

- 编译器级 SPMD：用 `jax.jit` + `shard_map` / `NamedSharding` 标注张量布局，**XLA 自动推导所有中间张量的切分并插入通信**；
- 不需要手写通信代码；缺点是调试黑盒、对自定义 kernel 支持弱。

> [!NOTE]
> PyTorch DTensor 的设计深受 GSPMD 影响，但 PyTorch 走"运行时 API"路线、JAX 走"编译器推导"路线。工业界两条路线并存（Megatron/torchtitan 走 PyTorch 路线；Gemini/TPU 生态走 JAX 路线）。

## 4. 主流框架速览

| 框架 | 组织 | 特点 |
|---|---|---|
| **Megatron-Core** | NVIDIA | TP/PP/CP/EP/FSDP 全支持；分布式优化器；universal checkpoint；工业预训练标准之一 |
| **Megatron-DeepSpeed** | Microsoft 社区 | Megatron 3D 并行 + DeepSpeed ZeRO/offload/MoE |
| **torchtitan** | PyTorch 官方 | Llama 参考实现；FSDP2 + torchtp + PP + CP + float8 + DCP + torch.compile 全可组合 |
| **DeepSpeed** | Microsoft | ZeRO 系列 + 3D 并行 + MoE + DeepSpeed-Inference/offload |
| **ColossalAI** | 开源 | Colossal-Auto 自动并行、Gemini 异构显存 |
| **OneFlow（OneS）** | SiliconFlow | SBP（Split/Broadcast/Partial）抽象 + AutoParallel，一套代码自动并行 |
| **MindSpeed** | 华为昇腾 | Megatron 生态向昇腾的移植加速，配合 MindFormers |
| **Whale / veScale** | 字节跳动 | Whale 大规模训练框架；veScale 基于 DTensor 的 Megatron 兼容实现，优化梯度归约与打包 |
| **JAX/XLA + GSPMD** | Google | TPU/GPU 编译器级自动并行，长上下文（Ring-Attention 等）生态强 |

## 5. 工业案例

### GPT-3 175B（Megatron 经典 3D 配置）

- 128-512 卡：TP=4、PP=8 + DP；每张卡只放一部分层，配合分布式优化器。

### Llama 3 405B（Meta，2024）

- 最多 16K H100（Grand Teton 服务器，700W TDP/卡）；
- **4D 并行**：FSDP（DP 轴全分片）+ TP + PP + CP；先 8K 上下文预训练，再 CP 扩展到长上下文；
- 设计原则：TP/CP 放节点内、PP 跨节点、FSDP 维度负责最外层数据并行。

### DeepSeek-V3（2024-2025）

- 671B MoE（256 专家，37B 激活），2048 张 H800；
- **DualPipe 双向流水**（气泡 -78%）+ **USP 序列并行** + **EP（DeepEP）** + FP8 混合精度；
- Megatron-Core 风格的推荐配置：TP=2、PP=16、EP=64（1024 卡规模），把 all-to-all 通信完全重叠进计算。

## 6. 配置检查清单（千卡场景）

1. hidden 与 TP 整除关系、头数与 CP/SP 整除关系；
2. TP/CP 进程组完整落在节点内（避免跨节点 TP）；
3. PP stage 数 × 每 stage 层数 = 总层数，且每 stage 负载均衡；
4. global batch = micro-batch × PP 内 micro-batch 数 × DP，确认气泡可接受；
5. MoE：EP 度 ≤ 专家数，路由均衡 + all-to-all 与计算重叠；
6. checkpoint 用 DCP/universal 格式，允许并行度变化；
7. 通信 profiling：先看 MFU/通信占比，再决定是否上 ZeroBubble/异步流水/压缩。

## 参考

- Megatron-DeepSpeed 3D 并行：https://github.com/microsoft/Megatron-DeepSpeed
- DTensor/DeviceMesh：https://pytorch.org/tutorials/beginner/dtensor_tutorial.html ；GSPMD：https://arxiv.org/abs/2105.04663
- torchtitan：https://github.com/pytorch/torchtitan
- Llama 3 并行（OSDI'25）：https://www.usenix.org/conference/osdi25/presentation/wang-zheng
- DeepSeek-V3：https://arxiv.org/abs/2412.19437
- Megatron-Core Parallelism Guide：https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html
