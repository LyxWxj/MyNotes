# DDiT: Dynamic Resource Allocation for Diffusion Transformer Model Serving

**arXiv:** 2506.13497

**Date:** 2025-06-16

**Authors:** Huang, Heyang; Hu, Cunchen; Zhu, Jiaqi; et al.

**Affiliation:** University of Chinese Academy of Sciences, Institute of Computing Technology

---

## 1. 论文概述

DDiT 是一个面向 Text-to-Video (T2V) 推理服务的高效资源调度系统。针对 DiT 和 VAE 的异构计算特性，提出阶段间解耦（Inter-phase Decoupling）和阶段内解耦（Intra-phase Decoupling）机制，结合步骤级贪心调度算法，实现 GPU 资源的动态分配。

**核心贡献**：

- 通过实证分析揭示 T2V 系统的四个关键 Insight
- 设计 DiT-VAE 解耦部署机制，消除资源不均衡
- 提出步骤级调度算法，支持运行时 DoP 动态调整
- 在 8×H800 上实现 P99 延迟降低 30.4%，平均延迟降低 30%

---

## 2. 背景与问题

### 2.1 T2V 系统架构

以 OpenSora 为代表的 T2V 系统包含三个模块：

```
文字输入 → [T5 文本编码器] → [DiT 扩散模型] → [VAE 解码器] → 视频输出
```

| 模块 | 参数量 | 计算特点 |
|------|--------|----------|
| T5 Encoder | 4.8B | 相对轻量，延迟占比小 |
| STDiT3 | 1.1B | O(L²) Attention，计算密集 |
| OpenSoraVAE | 384M | Conv + Upsample，带宽受限 |

### 2.2 现有问题

1. **同构部署**：DiT 和 VAE 使用相同 DoP，忽略两者计算特性的差异
2. **静态配置**：固定 DoP 无法适应不同分辨率的请求
3. **请求级调度**：无法在执行过程中重新分配资源

---

## 3. 实证分析（四个 Insight）

### Insight 1: Batching 对 T2V 无效

DiT 和 VAE 在 batch size=1 时 GPU 利用率已接近饱和。增大 batch size 不能提升吞吐量，反而增加延迟。

**结论**：最优调度策略是逐请求串行处理。

### Insight 2: DiT 与 VAE 的并行度需求不一致

| 模块 | DoP 增加时的表现 |
|------|------------------|
| DiT | 延迟随 DoP 增加而下降（尤其高分辨率） |
| VAE | 延迟基本不受 DoP 影响 |

**原因**：VAE 阶段所有 GPU 处理相同的输入张量，增加 GPU 只是冗余计算。

**数字示例（360p）**：

| DoP | VAE 执行时间 | VAE 占总推理时间比例 | 浪费的 GPU 数 |
|-----|-------------|-------------------|--------------|
| 1 | 0.87s | 4.5% | 0 |
| 4 | 0.87s | 14.3% | 3 |
| 8 | 0.87s | ~20% | 7 |

### Insight 3: 最优 DoP 因分辨率而异

定义 DiT 每步执行时间变化率：

$$z = 1 - \frac{\text{DiT\_step\_time}(\text{DoP}=i, r)}{\text{DiT\_step\_time}(\text{DoP}=i/2, r)}$$

| 分辨率 | DoP 1→2 | DoP 2→4 | DoP 4→8 | 最优 B |
|--------|---------|---------|---------|--------|
| 144p | 0% | -7% | -2% | 1 |
| 240p | 45% | 20% | -7% | 2 |
| 360p | 46% | 47% | 34% | 4 |

低分辨率用高 DoP 反而因通信开销导致性能下降。

### Insight 4: 静态部署导致资源浪费

请求一旦开始执行就无法调整 DoP。当一个 240p 请求占用 2 GPU 时，后续 360p 请求只能用剩余 2 GPU（最优需要 4），且无法在 240p 完成后动态获取释放的 GPU。

---

## 4. DDiT 系统设计

### 4.1 架构概览

```
┌───────────────────────────────────────────────────────┐
│                Centralized Control Plane              │
│  ┌────────────────┐  ┌───────────────┐  ┌────────────┐│
│  │Global Scheduler│  │Cluster Monitor│  │Resource    ││
│  │                │  │               │  │Allocator   ││
│  └────────────────┘  └───────────────┘  └────────────┘│
└───────────────────────────────────────────────────────┘
         │                    │
         ▼                    ▼
┌───────────────────────────────────────────────────────┐
│               Engine Units (弹性单元)                   │
│  ┌─────────────────┐  ┌──────────────┐  ┌────────────┐│
│  │Engine Controller│  │Model Engine  │  │Worker      ││
│  │(步骤级调度)       │  │(DiT/VAE)     │  │(GPU进程)   ││
│  └─────────────────┘  └──────────────┘  └────────────┘│
└───────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  Offline Profiler (离线性能画像)                      │
│  存储：resolution → optimal DoP 映射 (RIB)            │
└─────────────────────────────────────────────────────┘
```

### 4.2 Global Scheduler（全局调度器）

#### 4.2.1 问题建模

**优化目标**：最小化所有 GPU 的累积资源占用时间

$$O = \sum_{GPU_j \in \text{cluster}} \text{occupied\_time}(GPU_j)$$

**约束条件**：

- N 种分辨率类型，每种占比 $x_i$
- M 个 GPU（m 个实例，每实例 n 个 GPU）
- 每种分辨率有对应的最优 DoP

#### 4.2.2 理论最优调度算法（Algorithm 1）

在已知分辨率分布情况下的理论下界，采用动态规划求解。

**问题输入**：

- `M = m × n`：总 GPU 数量（m 个实例，每实例 n 个 GPU）
- `N`：分辨率类型数量
- `x_j`：第 j 种分辨率的请求占比
- `ps = {1, 2, 4, …}`：可选的 DoP 值列表

**算法伪代码**：

```
Algorithm 1: Theoretical Optimal Scheduling Algorithm

Input: M (GPU数量), N (分辨率类型数), x (各类型占比)
Output: dp[M][N] (最小累积资源占用时间)

1: m, n ← 实例数和每实例GPU数, M = m × n
2: N ← 分辨率类型数
3: dp[M+1][N+1] ← 初始化为全0
4: dp[i][0] = 0 (1 ≤ i ≤ M), dp[0][j] = ∞ (1 ≤ j ≤ N)
5: x1:x2:...:xN ← 各分辨率类型的占比
6: ps = {1, 2, 4, ...} ← 可选DoP列表
7: G[m][n] ← GPU分配状态表

8: for i = 1 to M do
9:   for j = 1 to N do
10:    FIND_OPTIMAL_TIME(G, dp, ps, i, j, xj)
11: Return dp[M][N]
```

**状态定义**：

$$dp[i][j] = \text{前 } i \text{ 个 GPU 分配给前 } j \text{ 种分辨率类型的最小累积占用时间}$$

**状态转移方程**：

$$dp[i][j] = \min_{k=1}^{i} \min_{p \in ps} \left\{ dp[i-k][j-1] + k \times \text{Occupy}(x_j, d, \alpha) \right\}$$

其中：

- `k`：分配给第 j 种类型的 GPU 数量（从 1 到 i 枚举）
- `p`：DoP 值（从 ps 列表中枚举）
- `α`：BandwidthAwarePartition 函数计算的模型实例数
- `d`：该分辨率和 DoP 下的执行时间（离线 Profile 获得）

**FIND_OPTIMAL_TIME 函数详解**：

```
function FIND_OPTIMAL_TIME(G, dp, ps, i, j, xj):
    for k = 1 to i do                // 枚举分配给第j类的GPU数量
        for p in ps do               // 枚举DoP值
            // 计算可用模型实例数（考虑网络拓扑）
            α ← BandwidthAwarePartition(G[网络拓扑][i-k+1..i], k, p)
            
            if α == 0 then      // 无法形成有效的并行组
                Continues
                
            // 获取该分辨率和DoP下的执行时间
            d ← EstimateExecutionTime(p, j)
            
            // 计算累积占用时间并更新dp
            dp[i][j] = min(dp[i][j], 
                          dp[i-k][j-1] + k × Occupy(xj, d, α))
```

**BandwidthAwarePartition 函数**：

该函数考虑网络拓扑对并行效率的影响，计算给定 GPU 集合和 DoP 下能创建的模型实例数。

**场景示例**：

```
集群配置：2台机器，每台8 GPU
  机器内：NVLink（高带宽，400GB/s）
  机器间：RDMA（低带宽，200Gbps）

当前状态：
  机器1：GPU 0-7 全部空闲
  机器2：GPU 0 已占用，GPU 1-7 空闲

可用GPU：机器1的GPU 0-7 + 机器2的GPU 1-7 = 15个GPU
```

**不同 DoP 下的实例数计算**：

| DoP | 可创建实例数 | 原因 |
|-----|-------------|------|
| 1 | 15 | 每个 GPU 独立运行一个实例 |
| 2 | 7 | 每 2 个 GPU 一组，需要高速互联 |
| 4 | 3 | 每 4 个 GPU 一组，机器 1 可创建 2 个，机器 2 可创建 1 个 |
| 8 | 1 | 需要 8 个高速互联 GPU，只有机器 1 满足 |

**关键约束**：Sequence Parallelism 要求参与并行的 GPU 之间有高速互联（NVLink），否则通信开销会抵消并行收益。

**Occupy 函数（累积资源占用时间计算）**：

根据排队模型计算单个 GPU 的平均资源占用时间。

**Batch Model（离线批量处理）**：

假设：

- 系统中有 S 个待处理请求
- 无新请求到达
- 模型实例数 α ≥ 1

请求均匀分配到 α 个实例，每个实例处理 $\lceil \frac{S \cdot x_j}{\alpha} \rceil$ 个请求。

$$W_{\text{Batch}}(\text{type } j) = \left\lceil \frac{S \cdot x_j}{\alpha} \right\rceil \times d$$

**示例**：

```
S = 100 个请求，x_j = 0.3（30%是360p），α = 2 个实例，d = 10s

每个实例处理：⌈100 × 0.3 / 2⌉ = 15 个请求
单GPU占用时间：15 × 10 = 150s
k个GPU总占用时间：k × 150s
```

**Queue Model（在线稳态）**：

假设：

- 请求到达服从 Poisson 分布，到达率 λ
- 服务时间固定为 d（M/D 排队模型）
- 利用率 ρ < 1

**M/D/1 队列（α = 1）**：

$$W_{M/D/1}(\text{type } j) = \frac{1}{\mu} + \frac{\rho}{2\mu(1-\rho)}$$

其中 $\mu = 1/d$，$\rho = \lambda \cdot x_j / \mu$

**M/D/c 队列（α > 1）**：

精确计算复杂，采用 M/M/c 近似：

$$W_{M/D/c}(\text{type } j) \approx \frac{W_{M/M/c}}{2} = \frac{1}{2} \left[ \frac{1}{\mu} + \frac{r^\alpha}{\alpha!(\alpha\mu)(1-\rho)^2} p_0 \right]$$

其中：

- $\mu = 1/d$
- $\rho = \lambda \cdot x_j / (\alpha \cdot \mu)$
- $r = \lambda \cdot x_j / \mu$
- $p_0 = \left( \frac{r^\alpha}{\alpha!(1-\rho)} + \sum_{s=0}^{\alpha-1} \frac{r^s}{s!} \right)^{-1}$

使用 Stirling 公式 $n! \approx \sqrt{2\pi n}(n/e)^n$ 优化阶乘计算。

**动态规划求解示例**：

```
场景：8 GPU，3种分辨率（144p:240p:360p = 1:1:1）

初始化：
  dp[0][0] = 0
  dp[i][0] = 0 (i ≥ 0)
  dp[0][j] = ∞ (j ≥ 1)

第1轮：处理144p (j=1)
  dp[1][1] = min over k=1, p∈{1,2,4}:
    dp[0][0] + 1 × Occupy(0.33, d_144p_1, α)
    = 0 + 1 × 3.01 = 3.01  (k=1, p=1, α=1)
    
  dp[2][1] = min over k=1,2, p∈{1,2,4}:
    k=1: dp[1][0] + 1 × 3.01 = 3.01
    k=2: dp[0][0] + 2 × 3.01 = 6.02 (但可创建2个实例)
         = 0 + 2 × ⌈0.33×S/2⌉ × d
    ...继续枚举

第2轮：处理240p (j=2)
  dp[i][2] = min over k, p:
    dp[i-k][1] + k × Occupy(0.33, d_240p_p, α)
    
第3轮：处理360p (j=3)
  dp[i][3] = min over k, p:
    dp[i-k][2] + k × Occupy(0.33, d_360p_p, α)

最终结果：dp[8][3] = 最小累积资源占用时间
```

**算法复杂度**：

- 时间：O(M² × N × |ps|)
- 空间：O(M × N)

该算法作为理论下界，用于评估在线贪心算法的近似比。实验表明 DDiT 的贪心算法达到理论最优的 1.39×，而 baseline 中最好的达到 2.08×。

#### 4.2.3 Greedy Scheduling（贪心调度算法）

实际在线调度算法，不依赖已知分布。

**核心思想**：为每种分辨率确定一个平衡点 B（性能增益最大的 DoP 翻倍点）。

**两阶段贪心策略**：

**阶段一：请求到达时**

```
新请求到达，需要 B 个 GPU
    │
    ├─ 空闲 GPU ≥ B → 分配 B 个 → Running
    │
    └─ 空闲 GPU = G < B → 分配 G 个 → Hungry
```

**阶段二：执行过程中**

```
其他请求完成，释放 GPU
    │
    ├─ 检查 Hungry 请求队列
    │
    ├─ 按饥饿时间排序（优先级）
    │
    └─ 分配给最高优先者，直到 DoP = B
```

**优先级计算（饥饿时间）**：

$$r^*_{v.} = (r_{\text{cur\_step}} - r_{\text{last\_step}}) \times (r_{\text{cur\_step\_time}} - r_{\text{opt\_step\_time}})$$

| 符号 | 含义 |
|------|------|
| $r_{\text{cur\_step}}$ | 当前步骤编号 |
| $r_{\text{last\_step}}$ | 最近一次 GPU 分配事件时的步骤编号 |
| $r_{\text{cur\_step\_time}}$ | 当前 DoP 下的每步执行时间 |
| $r_{\text{opt\_step\_time}}$ | 最优 DoP 下的每步执行时间 |

乘积含义：请求从最近一次 GPU 分配到当前时刻，因 GPU 不足而累积的额外执行时间。

**举例**：

```
请求 A：240p，B=2，当前 DoP=1，已执行 5 步
  每步时间差 = 0.15s - 0.08s = 0.07s
  饥饿时间 = 5 × 0.07 = 0.35s

请求 B：360p，B=4，当前 DoP=2，已执行 3 步
  每步时间差 = 0.35s - 0.20s = 0.15s
  饥饿时间 = 3 × 0.15 = 0.45s

→ 请求 B 优先获得 GPU
```

### 4.3 Efficient Decoupling（高效解耦机制）

#### 4.3.1 Inter-phase Decoupling（阶段间解耦）

**核心设计**：模型权重加载与通信组构建分离

```
初始化阶段（只做一次）：
  所有 GPU 加载完整模型权重
  不建立任何通信组

DiT 阶段开始：
  根据最优 DoP 动态建立通信组
  例如：GPU 0,1,2,3 建立 SP=4 的通信组

DiT 阶段结束：
  保留 GPU 0（master），释放 GPU 1,2,3
  latent tensor 留在 GPU 0 上

VAE 阶段：
  仅 GPU 0 执行
  释放的 GPU 可分配给其他请求
```

**为什么不能预定义分组**：

传统方式：

```
Group 1: GPU 0-3 → DiT
Group 2: GPU 4-7 → VAE
问题：DiT 完成后 GPU 4-7 空闲，但无法被其他请求使用（被 Group 2 锁定）
```

DDiT 方式：

```
所有 GPU 都是"弹性单元"
DiT 完成 → GPU 1,2,3 释放 → 可立即分配给其他请求的 DiT
```

**Scale-down 机制**：

```
Engine Unit R，当前 DoP = d
    │
    ├─ 选择 ID 最小的 GPU 作为 master
    │
    ├─ master 保留 latent tensor
    │
    ├─ 非 master GPU 释放回资源池
    │
    └─ 通信组重建为 DoP = d' (d' < d)
```

#### 4.3.2 Intra-phase Decoupling（阶段内解耦）

**目标**：在 DiT 执行过程中动态调整 DoP。

**实现机制**：

```
Engine Controller 的工作流程：

DiT-Step 1: DoP=2 (GPU 0,1)
    │
    ├─ 执行完成
    ├─ 向 Scheduler 报告：step=1, 当前 DoP=2
    │
    ↓
Scheduler: "GPU 2,3 刚释放，可以给这个请求"
    │
    ├─ 发送控制消息：new_gpu_ids = [0,1,2,3]
    │
    ↓
Engine Controller 收到消息：
    │
    ├─ 更新 DoP = 4
    ├─ 广播 latent tensor 到 GPU 2,3
    ├─ 重建通信组
    │
    ↓
DiT-Step 2: DoP=4 (GPU 0,1,2,3)
```

**通信开销分析**：

| 操作 | 耗时 |
|------|------|
| Transfer（latent tensor 广播） | < 1ms |
| Scale up（通信组重建） | < 1ms |
| DiT 单步执行 | 数百 ms ~ 数 s |

开销占比 < 0.1%，可忽略。

**请求生命周期（对应 Figure 9）**：

```
① 请求到达 → Scheduler 尝试分配 B 个 GPU
② 分配不足 → 标记 Hungry，发送控制消息到 Engine Controller
③ Engine Controller 建立连接，开始 DiT-Step 1
④ 每步执行：Controller 与 Worker 通信，检查是否有新 GPU
⑤ 其他请求完成 → Scheduler 重新分配 GPU（DoP Promotion）
⑥ DiT 完成 → 释放部分 GPU，通知 Scheduler
⑦ VAE 阶段（仅 master GPU）
⑧ VAE 完成 → 请求结束，释放所有 GPU
```

---

## 5. 实验结果

### 5.1 测试环境

- 8× NVIDIA H800 (80GB HBM)，NVLink 400GB/s
- 多节点：8 台服务器，200 Gbps RDMA
- 模型：T5v1.1-xxl (4.8B) + STDiT3 (1.1B) + OpenSoraVAE (384M)

### 5.2 Baseline 详解

#### Static DoP (SDoP)

最简单的部署策略，所有请求使用固定的 DoP。

```
DoP=1: 每个 GPU 独立运行一个模型实例
DoP=2: 每 2 个 GPU 组成一个并行组
DoP=4: 每 4 个 GPU 组成一个并行组
```

**优点**：部署简单，无需动态调度

**缺点**：无法适应不同分辨率的请求，低分辨率用高 DoP 浪费，高分辨率用低 DoP 延迟高

#### Static Partition & Cluster Isolation (SPCI)

根据历史请求分布，将 GPU 静态划分为多个集群，每个集群服务特定分辨率类型。

```
示例（8 GPU，请求分布 144p:240p:360p = 1:1:1）：
  Cluster 1: GPU 0-1 (DoP=1) → 服务 144p
  Cluster 2: GPU 2-3 (DoP=2) → 服务 240p
  Cluster 3: GPU 4-7 (DoP=4) → 服务 360p
```

**优点**：每种分辨率有专门的资源保障

**缺点**：

- 集群隔离导致资源无法共享
- 当某类请求过多时，对应集群过载，其他集群空闲
- 静态划分无法适应请求分布的动态变化

#### Dynamic Partition & Cluster Isolation (DPCI)

与 SPCI 类似，但 DoP 配置基于 B 值（最优并行度）动态确定，而非固定值。

```
示例（8 GPU，B 值：144p→1, 240p→2, 360p→4）：
  Cluster 1: GPU 0 (DoP=1) → 服务 144p
  Cluster 2: GPU 1-2 (DoP=2) → 服务 240p
  Cluster 3: GPU 3-6 (DoP=4) → 服务 360p
  GPU 7: 空闲
```

**优点**：DoP 配置更合理

**缺点**：仍然存在集群隔离问题，资源无法跨集群共享

#### Dynamic Partition (DP)

去除严格的集群隔离约束，允许跨集群资源共享。

```
场景：360p 请求到达，但 GPU 4-7 集群已满
  SPCI/DPCI: 请求等待，无法使用其他集群的空闲 GPU
  DP: 可以将请求降级到 DoP=2 的集群（GPU 2-3）执行
```

**核心机制**：

1. **降级执行**：当目标集群资源不足时，请求可以在更低 DoP 的集群执行
2. **资源共享**：空闲 GPU 可以被任何类型的请求使用
3. **灵活调度**：根据实时负载动态调整资源分配

**示例场景**：

```
时间 T1:
  请求 R1 (240p) 到达 → 分配 GPU 0-1 (DoP=2)
  请求 R2 (360p) 到达 → 分配 GPU 2-5 (DoP=4)

时间 T2:
  请求 R3 (360p) 到达
  SPCI: GPU 6-7 只有 2 个，不足 DoP=4，R3 等待
  DP: R3 可以用 GPU 6-7 (DoP=2) 降级执行，或等待 R2 完成后获得 GPU 2-5

时间 T3:
  R1 完成，释放 GPU 0-1
  DP: 可以将 GPU 0-1 分配给正在以 DoP=2 执行的 R3，提升其 DoP 到 4
```

**与 DDiT 的关键区别**：

| 维度 | DP | DDiT |
|------|-----|------|
| DoP 调整 | 请求级（开始时确定） | 步骤级（每步可调整） |
| DiT-VAE 解耦 | 无 | 有（阶段间解耦） |
| 资源分配 | 基于 B 值的静态最优 | 基于饥饿时间的动态最优 |
| 并行策略 | 固定通信组 | 弹性通信组（权重加载与通信构建分离） |

DP 是 DDiT 的一个简化版本，去除了集群隔离但仍保持请求级调度。DDiT 在此基础上增加了步骤级调度和阶段间解耦，进一步提升了资源利用率。

### 5.3 主要结果

**单节点**：

| 场景 | DDiT 改进 |
|------|-----------|
| 低到达率 (0.25) | P99 降低 12.88% |
| 中到达率 (0.5) | P99 降低 36.6%，Avg 降低 44.4%（vs Static DoP=4） |
| 突发负载 | P99 降低 20.7%，Avg 降低 21.7% |

**多节点 (64 GPU)**：

| 指标 | 改进 |
|------|------|
| P99 延迟 | 降低 30.4% |
| 平均延迟 | 降低 30% |
| 成本 | 1.39× 理论下界（vs baseline 的 2.08×） |

### 5.4 消融实验

| 机制 | P99 改进 |
|------|----------|
| DiT-VAE Decouple | 最多降低 26.1% |
| DoP Promotion | 最多降低 35.2% |

---

## 6. 与相关工作的对比

| 维度 | DistriFusion | DDiT |
|------|--------------|------|
| 并行层次 | Patch Parallelism（空间切块） | Sequence Parallelism + 异构部署 |
| 通信策略 | Displaced Patch（复用上一步特征图） | 步骤级动态调整通信组 |
| 调度粒度 | 单请求内多 GPU 协作 | 跨请求步骤级调度 |
| 适用场景 | 单张图片高分辨率生成 | 多请求在线视频生成服务 |
| 核心贡献 | 异步通信隐藏延迟 | 解耦部署 + 弹性资源分配 |

---

## 7. 关键 Takeaway

1. **T2V 系统的资源管理需要异构思维**：DiT 和 VAE 的计算特性差异巨大，统一配置必然导致浪费
2. **步骤级调度比请求级调度更灵活**：能在运行时根据资源变化调整策略
3. **饥饿时间是很好的优先级指标**：同时考虑了等待时间和性能惩罚
4. **权重加载与通信构建解耦是关键工程技巧**：避免了预定义分组带来的资源锁定

### 核心 Insight

**并行设备数量与性能提升的非线性关系**：

对于固定分辨率的请求，并非并行设备越多速度提升越大。由于 Amdahl's Law 和通信开销的存在，存在一个**性价比最高的并行度 B**（通过变化率 z 确定）：

$$z = 1 - \frac{\text{DiT\_step\_time}(\text{DoP}=i)}{\text{DiT\_step\_time}(\text{DoP}=i/2)}$$

超过 B 后，增加 GPU 带来的边际收益递减，甚至可能因通信开销导致性能下降。

**步骤级调度的代价**：

DDiT 的步骤级调度（每个 denoising step 都可能调整 DoP）带来了以下开销：

| 开销类型 | 具体表现 | 影响程度 |
|----------|----------|----------|
| **调度决策开销** | Scheduler 需要在每步结束时评估是否调整 DoP | O(饥饿请求数) |
| **通信组重建开销** | DoP 变化时需要重建 NCCL 通信组 | < 1ms |
| **数据传输开销** | latent tensor 广播到新加入的 GPU | < 1ms |
| **状态同步开销** | Engine Controller 与 Worker 之间的消息传递 | 可忽略 |

**权衡分析**：

- **收益**：步骤级调度能及时响应资源变化，减少 GPU 空闲时间
- **代价**：调度逻辑复杂，每步都有额外的控制开销
- **适用场景**：高负载、请求到达率变化大的场景收益明显；低负载场景收益有限，调度开销可能抵消收益

**论文中的验证**：

消融实验表明，DoP Promotion（步骤级调度的核心机制）在突发负载下 P99 延迟降低 35.2%，但在过载场景下效果有限，因为 DoP Promotion 会延长后续请求的等待时间。

---

## 8. 引用

```bibtex
@article{huang2025ddit,
  title={DDiT: Dynamic Resource Allocation for Diffusion Transformer Model Serving},
  author={Huang, Heyang and Hu, Cunchen and Zhu, Jiaqi and others},
  journal={arXiv preprint arXiv:2506.13497},
  year={2025}
}
```
