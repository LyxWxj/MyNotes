---
type: Note
related_to: "[[Diffusion-Serving]]"
status: Active
url: https://arxiv.org/abs/2510.01565
---

# TetriServe: Efficiently Serving Mixed DiT Workloads

> University of Michigan（Runyu Lu, Shiqi He, Mosharaf Chowdhury 等）
> ASPLOS '26
> arXiv:2510.01565
> 代码：https://github.com/DiT-Serving/TetriServe
> 论文：https://arxiv.org/abs/2510.01565

---

## 一、问题背景

DiT 模型（如 FLUX.1-dev、SD3）在图像生成中表现出色，但**在线服务面临 SLO 达标困难**：

- 生成一张 2048×2048 的图像在单个 H100 上需要近 1 分钟
- 4096×4096 可能超过 10 分钟
- 请求的分辨率和 deadline 高度异构

### 核心矛盾：固定并行度的 " 一刀切 " 策略无法适应异构工作负载

```
SP=1（数据并行）：小分辨率请求按时完成，大分辨率请求超时
SP=8（高并行度）：大分辨率请求加速了，但小分辨率请求通信开销过大，反而变慢
```

实验证明：在 Uniform 工作负载下，没有任何固定 SP 策略的 SLO 达标率超过 0.6。

### DiT 与 LLM 的关键区别

| 特性 | LLM | DiT |
|------|-----|-----|
| 状态 | 有状态（KV cache） | **无状态** |
| 瓶颈 | 内存受限（memory-bound） | **计算受限**（compute-bound） |
| 模型大小 | 很大，需要模型并行 | 较小（最大 12B），单 GPU 可放下 |
| 并行策略 | Tensor/Pipeline Parallelism | **Sequence Parallelism (SP)** |

---

## 二、三个关键洞察

### Insight 1：DiT 工作负载输入异构但执行可预测

| 图像大小 | Token 数 | TFLOPs | 执行时间 CV |
|---------|---------|--------|-----------|
| 256×256 | 256 | 556 | <0.13% |
| 512×512 | 1,024 | 1,388 | <0.15% |
| 1024×1024 | 4,096 | 5,046 | <0.14% |
| 2048×2048 | 16,384 | 24,965 | <0.28% |

**每个分辨率的每步运行时间高度可预测（CV < 0.7%）**，这使得 deadline-aware 调度成为可能。

### Insight 2：序列并行的扩展效率是次线性的，且因分辨率而异

```
小分辨率（256×256）：SP=8 时通信占比 >30%，扩展效率很差
大分辨率（2048×2048）：SP=8 时通信占比 <1%，扩展效率好
```

**对小分辨率用高并行度是浪费，对大分辨率用低并行度是不够。**

### Insight 3：步级并行度调整可以适应 deadline

- 高分辨率或紧急请求 → 分配更多 GPU
- 小分辨率或不紧急请求 → 节省资源
- **在同一个请求的不同去噪步之间，可以动态改变并行度**

---

## 三、NP-hard 证明

### 问题形式化

给定 N 个 GPU 和 R 个请求，找到一个步级调度方案，最大化按时完成的请求数量。

```
每个请求 req_i 包含 S_i 个依赖的去噪步 {s_i1, s_i2, ..., s_iS_i}
每步 s_ij 可以用 k ∈ {1, 2, 4, ..., N} 个 GPU 执行
执行时间 T_ij(k) 是 k 的函数
完成时间 C_i = Σ_j (Q_ij + T_ij(A_ij))，其中 Q_ij 是排队延迟

目标：max Σ I[C_i ≤ D_i]
```

约束条件：

1. **步依赖**：每步必须等前一步完成才能开始
2. **GPU 容量**：任意时刻分配的 GPU 总数不超过 N

**NP-hard 证明：** 即使在单步情况下（S_i = 1），问题也是 NP-hard 的。通过归约到 0-1 整数线性规划（ZILP）证明。多步情况自然也是 NP-hard。

---

## 四、TetriServe 系统设计

### 4.1 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    TetriServe                             │
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │                Scheduler（调度器）                  │    │
│  │                                                    │    │
│  │  ┌────────────────────┐  ┌────────────────────┐   │    │
│  │  │ Deadline-Aware     │  │ Round-Based        │   │    │
│  │  │ GPU Allocation     │  │ Request Packing    │   │    │
│  │  │（最小化 GPU 小时）   │  │（DP 求解装箱）      │   │    │
│  │  └────────────────────┘  └────────────────────┘   │    │
│  └──────────────────────────────────────────────────┘    │
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │              Execution Engine（执行引擎）           │    │
│  │  GPU Worker Pool + Latent Manager                  │    │
│  └──────────────────────────────────────────────────┘    │
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │              Request Tracker（请求追踪器）          │    │
│  │  记录每个请求的分辨率、deadline、剩余步数            │    │
│  └──────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

### 4.2 请求生命周期

```
1. 请求到达 → Request Tracker 记录元数据（分辨率、deadline、状态）
2. Scheduler 在下一轮调度中决定该请求用几个 GPU、执行几步
3. Execution Engine 的 GPU Worker 执行分配的步数
4. Latent Manager 管理中间潜空间表示
5. 完成后通知 Request Tracker 更新状态
6. 所有步完成后返回结果给用户
```

---

## 五、核心算法

### 5.1 Deadline-Aware GPU Allocation（最小化 GPU 小时）

**核心思想：** 为每个请求找到**满足 deadline 的最小 GPU 分配**，从而节省 GPU 资源给其他请求。

```
目标：min Σ_j (A_ij × T_ij(A_ij))   ← 最小化总 GPU 小时
约束：Σ_j (Q_ij + T_ij(A_ij)) ≤ D_i  ← 满足 deadline
```

**离线画像：** 对每种分辨率和 GPU 数量组合，预先测量执行时间，存入查找表。运行时直接查表枚举。

**示例：**

```
请求 R1（256×256）：SP=1 就够了（高并行度反而效率低）
请求 R2（1024×1024）：需要 SP=2 或 SP=4
请求 R3（2048×2048）：需要 SP=4 或 SP=8
```

每个请求得到一组候选分配方案 `(步数, GPU数)`，作为后续装箱的输入。

### 5.2 Round-Based Scheduling（基于轮次的调度）

**核心思想：** 将连续时间离散化为固定时长的轮次（round），每轮做一次调度决策。

```
时间轴：|---Round 1---|---Round 2---|---Round 3---|---Round 4---|
        ↑ 调度决策     ↑ 调度决策     ↑ 调度决策     ↑ 调度决策
```

**轮次时长 τ 的选择：**

- 短轮次：更细粒度的抢占，更敏捷的调度，但调度频率高
- 长轮次：摊销调度开销，但排队延迟大，可能错过 deadline
- 实践中根据步粒度确定，每轮执行多个去噪步

**每轮调度的目标：** 最小化 "definitely late" 的请求数量——那些如果本轮不推进就**必定**超时的请求。

**判断 definitely late：**

```
如果本轮结束后（t_{r+1} = t_r + τ），即使用最快的并行度（T_i^min），
剩余步数的下界仍然超过 deadline，则该请求 definitely late：

LB_i(o) = Σ_m s̃_i^m(o) × T_i^min    ← 剩余时间下界
sv_i(o) = I[t_{r+1} + LB_i(o) ≤ D_i]  ← 是否能存活到下一轮
```

### 5.3 Dynamic Programming 求解装箱（Algorithm 1）

每轮的调度问题是一个**分组背包问题（Group Knapsack）**：

| 概念 | 对应 |
|------|------|
| 组（Group） | 每个请求 |
| 选项（Option） | 用不同 GPU 数执行不同步数，或不执行 |
| 宽度（Width） | 消耗的 GPU 数 |
| 价值（Value） | 二值 " 存活 "（是否 definitely late） |
| 容量 | N 个 GPU |
| 目标 | 最大化存活请求数 |

**DP 状态转移：**

```
dp[c] = 处理前 i 个请求后，消耗 c 个 GPU 时的最大存活数

对每个请求 i：
  next[0..N] ← dp    （滚动数组）
  对每个容量 c = 0 to N：
    对每个选项 o ∈ O_i：
      if w_i(o) ≤ c:
        next[c] = max(next[c], dp[c - w_i(o)] + sv_i(o))
  dp ← next

c* = argmax_c dp[c]
从 c* 回溯得到调度方案
```

**复杂度：** O(R × N) 时间，O(N) 空间（滚动数组）。比暴力枚举快几个数量级，可在毫秒级完成。

### 5.4 GPU Placement Preservation（放置保持）

请求在连续轮次之间尽量保持在**相同的 GPU 上**执行：

- 避免状态传输延迟
- 避免通信组重建开销
- 消除轮次之间的空闲气泡

### 5.5 Work-Conserving Elastic Scale-Up（弹性扩展）

调度完成后如果有 GPU 空闲，将其分配给能从中受益的请求：

```
如果 T_i(k_i') < T_i(k_i)（增加 GPU 能加速）：
  → 把空闲 GPU 分配给该请求
  → 优先分配给加速效果最大的请求
```

**确保没有 GPU 在轮次内闲置。**

---

## 六、其他优化

### 6.1 Selective Continuous Batching（选择性连续批处理）

- 只对**相同分辨率的小请求**进行批处理
- 只在**不影响 SLO** 的情况下 batch
- 减少 kernel 启动开销，提升吞吐

### 6.2 VAE Decoder 顺序执行

- VAE 解码器在高分辨率下激活内存很大
- 采用顺序解码（不并发），避免 OOM
- 解码不在关键路径上，不影响端到端延迟

### 6.3 Communication Process Groups Warmup

- 预创建常用 GPU 组合的通信组（如 k-C-8 的组合）
- 只预热常用的、重叠的组（如 [0,1,2,3], [0,2,3,4]）
- 其他按需初始化（lazy warmup）
- 平衡启动延迟和内存占用

### 6.4 Latent Transfer

- 中间潜空间张量很小（压缩的潜空间）
- 使用 Future-like 抽象实现异步非阻塞传输
- 传输开销 < 0.05% 的每步延迟，可忽略

### 6.5 与缓存加速（Nirvana）的兼容性

TetriServe 与 Nirvana 正交兼容：

- Nirvana 通过复用先前请求的中间潜空间，跳过前 k 步（N → N-k 步）
- TetriServe 动态调整并行度以适应减少后的步数
- 组合使用效果最佳

```
                  RSSP    TetriServe  RSSP+Nirvana  TetriServe+Nirvana
Uniform           0.32     0.42         0.77          0.88
Skewed            0.04     0.19         0.53          0.75
```

---

## 七、实验结果

在 FLUX.1-dev（8×H100）和 SD3（4×A40）上测试：

### 工作负载

- **Uniform**：四种分辨率（256/512/1024/2048）均匀分布
- **Skewed**：偏向大分辨率的指数分布
- 默认到达率：12 requests/min（Poisson 过程）

### 性能改进

| 指标 | 改进幅度 |
|------|---------|
| SLO 达标率（SAR） | 最高提升 **32%** |
| 平均 SAR（Uniform） | 比最佳固定策略高 **10%** |
| 平均 SAR（Skewed） | 比最佳固定策略高 **15%** |
| 尾部延迟 | 显著优于所有基线 |

### 关键发现

1. TetriServe 在**所有分辨率**上都表现良好，而固定 SP 策略只在特定分辨率上有效
2. 在突发流量下保持稳定的 SAR，而固定策略波动剧烈
3. 与 Nirvana（缓存加速）正交兼容，组合使用 SAR 从 0.42 提升到 0.88
4. 调度开销在毫秒级，满足在线需求

### 基线对比

| 基线 | 描述 |
|------|------|
| xDiT (SP=1/2/4/8) | 固定并行度，每个请求用固定数量 GPU |
| RSSP | 按分辨率选择最佳固定 SP（离线画像），代表 oracle 静态配置 |
| **TetriServe** | 步级动态 SP + deadline-aware 调度 |

---

## 八、核心贡献总结

1. **证明** DiT 步级调度问题是 NP-hard 的
2. 提出 **Round-Based Scheduling**：将连续时间离散化为轮次，使问题可解
3. **Deadline-Aware GPU Allocation**：为每个请求找到满足 deadline 的最小 GPU 分配
4. **DP 求解装箱**：用分组背包的动态规划在 O(RN) 时间内求解每轮调度
5. **GPU Placement Preservation + Elastic Scale-Up**：减少轮次间开销，充分利用 GPU
6. 与缓存加速（Nirvana）正交兼容
7. 代码量：5,033 行 Python + C++

---

## 九、与 DiT-Serve 和 TridentServe 的对比

| 维度 | DiT-Serve | TridentServe | TetriServe |
|------|-----------|--------------|------------|
| **关注点** | 跨请求批处理 | 跨阶段资源分配 | 步级并行度调整 |
| **粒度** | 去噪步级 | 阶段级 | 去噪步级 |
| **核心问题** | 时间/空间低效 | 三阶段不对称 | 固定 SP 的 tradeoff |
| **调度方法** | SRPTF | ILP | DP（分组背包） |
| **并行策略** | Brick Attention | 动态 SP 度数 | 步级动态 SP |
| **时间模型** | 连续 | 连续 | 离散化（round-based） |
| **NP-hard** | 未讨论 | 证明并近似 | 证明并用 DP 求解 |
| **目标** | 吞吐量 + 延迟 | SLO 达标率 | SLO 达标率 |
| **模型类型** | 视频 + 图像 | 视频 + 图像 | 图像 |
| **发表** | OpenReview 投稿 | arXiv | ASPLOS '26 |

### 三者的互补关系

- **DiT-Serve**：解决 " 如何高效批处理异构请求 "（Brick Attention + Step-Level Batching）
- **TridentServe**：解决 " 如何在三个阶段间分配资源 "（stage-level 动态部署）
- **TetriServe**：解决 " 如何根据 deadline 动态调整并行度 "（step-level SP 调整）

三者从不同角度优化扩散模型服务，可以组合使用。
