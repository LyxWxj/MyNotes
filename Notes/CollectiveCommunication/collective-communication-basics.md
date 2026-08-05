---
type: Note
related_to: "[[collective-communication-index]]"
status: Active
---

# 集合通信基础（原语 / 算法 / 拓扑）

> 集合通信 = **一组进程（rank）之间交换数据的通信模式**。理解它的关键是两个视角：**语义**（谁拿到什么）和**算法**（怎么在物理拓扑上高效实现）。

## 1. 原语语义与通信量

假设 N 个 rank，每个 rank 持有数据量 S（字节）。下表给出"每 rank 发送/接收的总量"（近似）。

| 原语 | 语义 | 通信量/rank | 常见用途 |
|---|---|---|---|
| `broadcast` | 1 个 rank 的数据发给所有人 | ~S | 广播初始化权重、共享随机种子 |
| `reduce` | 所有人数据做归约（sum/min/max），结果在 1 个 rank | ~S | 归约到 0 号 rank |
| `all-reduce` | 归约后广播给所有人 | **2S** | DDP 梯度同步、TP 层输出 |
| `reduce-scatter` | 归约后**切分**，每人拿一块 | **S** | ZeRO-2/3、FSDP 梯度分片 |
| `all-gather` | 把各 rank 的分片拼成完整数据 | ~S×(N-1)/N | ZeRO-3/FSDP 参数恢复 |
| `gather` / `scatter` | 汇集到 1 个 rank / 从 1 个 rank 分发 | S | 检查点、数据分发 |
| `all-to-all` | 每个 rank 给每个 rank 发**不同**的数据 | S×(N-1)/N | MoE dispatch/combine、Ulysses |
| `p2p send/recv` | 点对点传输 | S | PP 相邻 stage 传激活 |

> [!IMPORTANT]
> 关键恒等式：**`all-reduce ≡ reduce-scatter + all-gather`**。DDP → ZeRO/FSDP 的演进，本质上就是把 2S 的 all-reduce 拆成 S 的 reduce-scatter + S 的 all-gather，从而把"每个人都持有全量"变成"每人只持分片"。

## 2. 性能模型：延迟与带宽

一次集合通信耗时近似：

```
T ≈ α（启动/同步开销）+ β × S（传输）
```

- **α（延迟）**：微秒级，与数据量无关，与 hop 数、协议握手相关；
- **β = 1/带宽**：纳秒/字节级，由互联决定；
- 小消息是**延迟受限**（α 主导），大消息是**带宽受限**（β·S 主导）——调优时先判断属于哪种。

## 3. 算法：Ring / Tree / CollNet / NVLS / PAT

### Ring（环）

- 所有 rank 排成环，数据切成 N 段，每段沿环转一圈，边传边归约；
- **带宽最优**：all-reduce 的 busbw 可达 `2×(N-1)/N × 链路带宽`，N 大时趋近 2×；
- 代价：延迟随 N **线性**增长（转 N 圈）；N=1 退化。

### Tree（树）

- 二叉树结构逐层归约/广播，延迟 **O(log N)**；
- 带宽较差（高层链路是瓶颈）；适合**跨节点大消息**或延迟敏感的小消息；
- NCCL 默认在节点间用 Tree、节点内用 Ring 的混合场景很常见。

### CollNet（In-Network Computing，SHARP）

- 借助 **InfiniBand 交换机上的 SHARP 引擎**做网内归约：数据在路上就被相加，**交换机=计算单元**；
- 网络流量从"每 rank 收发 2S"降为"每 rank 收发 S"，有效带宽翻倍；
- 需要支持 SHARP 的网络（Quantum/ConnectX 系列）和 NCCL-SHARP 插件。

### NVLS（NVLink SHARP）

- 在 **NVSwitch** 上做类似 SHARP 的网内归约（NCCL 2.18+ 的 NVLS/NVLS-Tree）；
- 8 卡全互联 NVLink 域内 all-reduce 可接近线性带宽；
- **NVLSTree** = 节点内 NVLS + 节点间 Tree，多节点时的主流混合算法。

### PAT（Parallel Aggregation Tree）

- NCCL 2.23+ 引入，面向**跨数据中心/超大集群**：多条并行树同时工作，兼得带宽与延迟优势。

## 4. busbw：衡量"真正用上的带宽"

nccl-tests 输出两个关键指标：

```
algbw  = 数据量 / 耗时            （算法带宽）
busbw  = 按通信模式折算的总线带宽  （衡量拓扑利用效率）
```

对 all-reduce：`busbw = 2×(N-1)/N × algbw × ...`（不同原语折算系数不同）。

> 判断标准：**busbw 是否接近互联理论值**。例如 8 卡 H100（NVLink ~900 GB/s）单节点 all-reduce 应跑到几百 GB/s 量级；跨节点要看 IB 链路数（400Gbps≈50GB/s/条）。

## 5. 拓扑感知：节点内 / 节点间

```
┌─ 节点 A ──────────────────────────┐   ┌─ 节点 B ──────────────┐
│ 卡0──NVLink──卡1        IB HCA0 ──┼───┼── IB HCA0 ──卡0──卡1   │
│   │         │             │       │   │             │      │   │
│ 卡2──NVLink──卡3        IB HCA1 ──┼───┼── IB HCA1 ──卡2──卡3   │
└───────────────────────────────────┘   └────────────────────────┘
```

- **NVLink**：低延迟、超高带宽，适合高频小消息（TP/CP 的 all-reduce）；
- **InfiniBand**：带宽高但延迟更高，适合大块低频传输；**rail-optimized** 网络下每个 NIC 直连不同交换机，ring 应尽量"同 rail"（`NCCL_CROSS_NIC` 控制）；
- **PCIe / QPI**：兜底路径；PXN 技术让跨节点流量经 NVLink 汇聚到指定 NIC，绕过 CPU/QPI（见 [[nccl-tuning-and-debugging|NCCL 调优与排查]]）。

## 6. 与并行技术的通信量速查（复习）

| 场景 | 原语 | 每步通信量 |
|---|---|---|
| DDP 梯度同步 | all-reduce | 2×模型 |
| ZeRO-2 / FSDP SHARD_GRAD_OP | reduce-scatter | 1×模型 |
| ZeRO-3 / FSDP FULL_SHARD | all-gather + reduce-scatter | 1.5×模型 |
| TP 每层 | all-reduce ×2 | O(hidden) |
| Ulysses 每层 | all-to-all ×2 | O(seq×hidden/SP×N) |
| MoE 每层 | all-to-all ×2 | O(EP×batch×hidden) |

## 参考

- NCCL 文档（Algorithms）：https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html
- NCCL 跨数据中心与拓扑感知（NVIDIA Blog）：https://developer.nvidia.com/blog/nccl-deep-dive-cross-data-center-communication-and-network-topology-awareness/
- nccl-tests README（busbw 定义）：https://github.com/NVIDIA/nccl-tests
