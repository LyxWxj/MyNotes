---
type: Note
related_to: "[[Diffusion-Serving]]"
status: Active
url: https://arxiv.org/abs/2510.02838
---

# TridentServe: A Stage-level Serving System for Diffusion Pipelines

> 北京大学 Hetu 团队（Yifei Xia, Fangcheng Fu, Bin Cui 等）
> arXiv:2510.02838
> 论文：https://arxiv.org/abs/2510.02838

---

## 一、问题背景

扩散流水线（Diffusion Pipeline）遵循 **Encode → Diffuse → Decode** 三阶段架构：

```
用户输入 → [Encode] → 条件 c → [Diffuse: 多步去噪] → 潜空间结果 → [Decode] → 像素空间图像/视频
```

### 当前服务系统的问题：静态、粗粒度的资源分配

现有系统（如 xDiT、VideoSys）有两种做法：

1. **小模型**：把所有阶段放在每个 GPU 上（co-locate）
2. **大模型**：静态地为每个阶段分配固定数量的 GPU（disaggregated）

但无论哪种，都遵循**流水线级（pipeline-level）**的资源分配——对每个请求的三个阶段分配相同的资源。

---

## 二、作者发现的两个核心低效

### 2.1 Insight 1：阶段之间和请求之间的资源需求差异巨大

通过分析 4 个主流模型（Stable Diffusion 3、Flux.1、CogVideoX、HunyuanVideo），作者发现：

| 阶段 | 特点 | 资源需求 |
|------|------|---------|
| **Encode** | Transformer 编码器，处理长度 ≤500 | 很轻量，主要靠 batching 提效 |
| **Diffuse** | DiT 模型，处理长度 100–120k，占 70%+ 时间 | **计算密集**，对并行度敏感 |
| **Decode** | VAE 解码器，**内存密集** | 对并行度不敏感，增加 GPU 收益有限 |

**关键发现：**

- **同一请求的不同阶段**需要不同的并行度（Diffuse 在高分辨率下需要大并行度，Decode 不需要）
- **不同请求**因分辨率/时长不同，需要的并行度也不同

> 例如：512p 的请求用 SP2 就够了，2048p 的请求需要 SP8
> 但如果对 512p 的请求也分配 SP8，GPU 就浪费了

### 2.2 Insight 2：工作负载模式变化时，阶段间的资源比例也要变

当请求到达率变化时（比如从 Light 变成 Heavy），三个阶段的处理速度变化比例不同，需要**动态调整每个阶段的副本数量**来保持吞吐平衡。

```
Light 负载下：  Encode:Diffuse:Decode = 10:63:27
Heavy 负载下：  Encode:Diffuse:Decode = 2:88:10
```

静态分配无法适应这种变化。

### 2.3 设计原则

基于以上分析，作者提出两个设计原则：

- **Principle 1**：对请求的资源分配应该是**动态的、stage-level**的，而不是静态的、pipeline-level 的
- **Principle 2**：对阶段模型的资源分配应该是**自动的、动态的**，而不是人工的、静态的

---

## 三、TridentServe 系统设计

### 3.1 两大抽象

| 抽象 | 含义 | 负责什么 |
|------|------|---------|
| **Placement Plan (P)** | 每个 GPU 上部署哪些阶段的模型副本 | 模型端资源分配 |
| **Dispatch Plan (Γ)** | 每个请求的每个阶段用哪些 GPU、什么并行策略执行 | 请求端资源分配 |

### 3.2 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      Planning（规划层）                        │
│                                                              │
│  ┌─────────────────┐         ┌─────────────────┐            │
│  │ Dynamic          │         │ Resource-Aware   │            │
│  │ Orchestrator     │         │ Dispatcher       │            │
│  │                  │         │                  │            │
│  │ 生成 Placement   │         │ 生成 Dispatch    │            │
│  │ Plan (P)         │         │ Plan (Γ)         │            │
│  └────────┬─────────┘         └────────┬─────────┘            │
│           │                            │                     │
│  ┌────────▼────────────────────────────▼─────────┐           │
│  │              Runtime Engine（执行层）            │           │
│  │                                                │           │
│  │  Dynamic Reinstance → Stage Prepare → Merge Exec│           │
│  └────────────────────┬───────────────────────────┘           │
│                       │                                      │
│  ┌────────────────────▼───────────────────────────┐           │
│  │    Profiler（离线画像） + Monitor（在线监控）      │           │
│  └─────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 整体工作流程

```
1. Offline Profiling：Profiler 收集每个阶段在各分辨率/并行度下的延迟和内存数据

2. Bootstrap Placement：系统启动时，Orchestrator 根据画像生成初始 Placement Plan P_init

3. Online Serving：运行时，Dispatcher 结合请求元数据和集群信息生成 Dispatch Plan Γ，
   交给 Runtime Engine 执行

4. Adaptive Re-Placement：Monitor 持续监控，检测到工作负载变化导致瓶颈时，
   Orchestrator 生成新的 P_switch，通过 Adjust-on-Dispatch 无停机切换
```

---

## 四、Placement Plan（模型部署）

### 4.1 Placement 类型

每个 GPU 的部署方案 `π_g` 有 6 种类型：

```
⟨EDC⟩  ← 一个 GPU 放三个阶段（小模型）
⟨DC⟩   ← Diffuse + Decode
⟨ED⟩   ← Encode + Diffuse
⟨D⟩    ← 只放 Diffuse
⟨E⟩    ← 只放 Encode
⟨C⟩    ← 只放 Decode
```

其中包含 D 的 4 种称为 **Primary Placement**，不包含 D 的 2 种称为 **Auxiliary Placement**。

### 4.2 Virtual Replica（虚拟副本）

要执行一个请求，需要一组 GPU 其中部署的阶段联合覆盖 {E, D, C}。这样的 GPU 集合称为 **Virtual Replica (VR)**。

| Virtual Replica | Primary Replica | Auxiliary Replica | 通信开销 |
|----------------|----------------|-------------------|---------|
| V0: ⟨EDC⟩ | ⟨EDC⟩ | 无 | 最小（零通信） |
| V1: ⟨DC⟩ + ⟨E⟩ | ⟨DC⟩ | ⟨E⟩ | 较小 |
| V2: ⟨ED⟩ + ⟨C⟩ | ⟨ED⟩ | ⟨C⟩ | 较大 |
| V3: ⟨D⟩ + ⟨E⟩ + ⟨C⟩ | ⟨D⟩ | ⟨E⟩ + ⟨C⟩ | 最大 |

**选择逻辑：** 对每个请求，选择通信开销最小的可行 VR 类型（V0 ≺ V1 ≺ V2 ≺ V3）。

### 4.3 部署比例计算（Algorithm 2）

```python
# 输入：请求集 R，GPU 数量 G，各 Placement 类型的处理速度 {v_π}

# Step 1: 为每个请求选择最优 VR 类型
for r in R:
    OptVR(r) = min{t ∈ T | peakMem(t) ≤ cap(t)}  # 内存可行的最小通信类型

# Step 2: 按比例分配 VR 类型数量
for t in T:
    α_t = |{r ∈ R: OptVR(r)=t}| / |R|  # 该类型请求的比例
    N_t = floor(α_t * G)                  # 该类型分配的 GPU 数

# Step 3: 在每种 VR 类型内，按处理速度比例拆分 Primary 和 Auxiliary
for t in T:
    (N_t_prim, N_t_aux) = Split(N_t, {v_π}, t)

# Step 4: 按机器打包
P = PackPerMachine({(t, N_t_prim, N_t_aux)}, G)
```

**目标：** 让每种 VR 类型的 GPU 数量与请求分布成比例，同时在每种类型内平衡 Primary 和 Auxiliary 的处理速度。

---

## 五、Dispatch Plan（请求调度）

### 5.1 问题形式化

对每个请求 r（有 SLO 截止时间 d_r），为每个阶段 s 选择：

- 用哪些 GPU（G_r^s）
- 用什么并行策略（φ_s）
- 在什么时间执行

目标：最大化 SLO 达标率。

这是一个 **NP-complete 的 Job-Shop 调度问题**，无法实时求解。

### 5.2 两步决策

#### Step 1：先决定 Diffuse 阶段的调度 Γ_D

**关键洞察：** Diffuse 阶段占 70%+ 时间，且对并行度高度敏感；Encode 和 Decode 对并行度几乎不敏感。所以先精心决定 Γ_D，然后 Γ_E 和 Γ_C 直接从 Γ_D 推导。

**ILP 建模：**

```
决策变量：
  x_{r,i,k} ∈ {0,1}：是否在类型 i 的 Primary Replica 上用并行度 k 执行请求 r
  D_r ∈ {0,1}：请求 r 是否按时完成

目标函数：
  max Σ_r Σ_i Σ_k (W_r · D_r - Q_{r,i} · x_{r,i,k})

  其中：
  - W_r：SLO 奖励权重（按时完成给正奖励，延迟给负奖励，带 aging 防饥饿）
  - Q_{r,i}：通信惩罚（鼓励选择通信少的 Placement 类型）

约束：
  (C0) x_{r,i,k} ≤ E_{r,k} · F_{r,i,k}  ← 过滤低效/不可行的组合
  (C1) Σ_i Σ_k x_{r,i,k} ≤ 1            ← 每个请求最多分配一次
  (C2) Σ_r Σ_k x_{r,i,k} ≤ B_i          ← 不超过可用 Primary Replica 数
  (C3) t_{r,i,k} · x_{r,i,k} ≤ d_r + M(1-D_r)  ← 运行时间不超过 SLO
```

**求解效率：** 由于预过滤了变量，且集群一般只有 1-2 种 Primary Placement 类型，ILP 可在百毫秒内求解。

#### Step 2：根据 Γ_D 推导 Γ_E 和 Γ_C

```
给定 Γ_D 和选定的 Primary Replica 类型：

Γ_E：
  - 如果 Primary Replica 包含 E → 复用 G_D（合并执行，零通信）
  - 否则 → 从 E 类型 Auxiliary Replica 中选择空闲或最早完成的 GPU

Γ_C：
  - 如果 Primary Replica 包含 C → 取 G_D 的子集（Decode 需要的资源比 Diffuse 少）
  - 否则 → 从 C 类型 Auxiliary Replica 中选择空闲或最早完成的 GPU
```

---

## 六、Runtime Engine（执行引擎）

### 6.1 Dispatch Plan 的执行三步

#### 1. Dynamic Reinstance（动态重实例化）

- 临时将目标 GPU 编组成执行实例
- 激活通信组
- **热集合 + 懒初始化**：只预初始化常用的 GPU 组合（热集合），不常用的首次使用时初始化（懒初始化）
- 目标：毫秒级重配置，无全局暂停，内存占用可控

#### 2. Stage Preparation（阶段准备）

**模型副本管理：**

- 每个节点维护一份 CPU 共享副本
- GPU 只存放当前 Placement Plan 指定的阶段

**输入传输 - Proactive Push（主动推送）：**

前一个阶段完成后，立即把输出推送到下一个阶段的 handoff buffer (HB)：

```
Encode 计算 → Encode 通信（推送到 Diffuse 的 HB）
              Diffuse 计算（从 HB 读取）→ Diffuse 通信（推送到 Decode 的 HB）
                                              Decode 计算（从 HB 读取）
```

- 每个 GPU 维护一个 device-resident 的 HB
- 如果 HB 满了，buffer 到 pinned host memory
- **通信策略**：
  - 跨节点：GPUDirect RDMA → 目标节点一个 GPU → 节点内广播
  - 节点内：直接通过共享通信器广播

#### 3. Merging Execute（合并执行）

- 如果同一请求的连续阶段在同一个 GPU 上，合并为一次执行
- 减少 CPU 调度开销

### 6.2 Adjust-on-Dispatch（按需调整）

当 Monitor 检测到阶段间速度失衡（最快阶段 ≥ 最慢阶段的 1.5×）时触发：

```
1. Orchestrator 生成新的 Placement Plan P_switch
2. Runtime Engine 立即更新元数据（不实际移动模型）
3. Dispatcher 按 P_switch 生成新的 Dispatch Plan
4. 当新的 Dispatch Plan 发现目标 GPU 上没有所需模型时：
   a. 优先从同节点的其他 GPU 用 GPUDirect P2P 传输
   b. 否则从 CPU 共享副本加载
5. 使用分块流式传输避免 OOM
```

**关键保证：** 由于每个 worker 按 FIFO 执行 plan，旧 Placement 下的 plan 会先完成，新 plan 才开始，确保切换安全。

---

## 七、Profiler 和 Monitor

### 7.1 Profiler（离线画像）

利用 GVT 工作负载的**强可预测性**，离线收集两级信息：

| 级别 | 内容 | 用途 |
|------|------|------|
| **Request Metadata** | 每个阶段在各并行策略下的运行时间和峰值内存 | 生成 Dispatch Plan |
| **Request Statistics** | 所有阶段的处理长度和峰值内存分布 | 指导 Orchestrator 和 Dispatcher |

### 7.2 Monitor（在线监控）

周期性收集：

| 维度 | 内容 | 用途 |
|------|------|------|
| **GPU-worker Status** | 每个 GPU 的空闲状态、当前 Placement、剩余内存、正在执行的 plan | 生成 Dispatch Plan |
| **Stage Throughput** | 每种 Placement 类型的处理速率 v_π | 触发 Adjust-on-Dispatch |

---

## 八、实验结果

在 4 个模型和多种工作负载下测试：

| 模型 | 类型 | 部署方式 |
|------|------|---------|
| Stable Diffusion 3 | 图像 | Co-locate |
| Flux.1 | 图像 | Disaggregated |
| CogVideoX 1.5-5B | 视频 | Co-locate |
| HunyuanVideo | 视频 | Disaggregated |

### 性能改进

| 指标 | 改进幅度 |
|------|---------|
| SLO 达标率 | 持续提升 |
| 平均延迟 | 降低最高 **2.5×** |
| P95 延迟 | 降低最高 **3.6×** |
| P99 延迟 | 降低最高 **4.1×** |

### 关键发现

- B1–B4（pipeline-level 方案）在 Flux 和 HunyuanVideo 上**全部 OOM**
- TridentServe 通过 stage-level 的动态资源分配**避免了 OOM**
- 在动态和专有工作负载下表现尤为突出
- ILP 求解时间在百毫秒级，满足在线调度需求

### 基线对比

| 基线 | 描述 |
|------|------|
| B1: Static Pipeline-level | xDiT 的方式，静态并行度，FIFO |
| B2: Bucketed Pipeline-level | 按并行度分桶，桶内 FIFO |
| B3: Dynamic Pipeline-level (FIFO) | 动态选择并行度，三个阶段相同资源 |
| B4: Dynamic Pipeline-level (SRTF) | 同 B3 + SRTF 调度 |
| B5: Bucketed Stage-level | 手动 disaggregated + 分桶 |
| B6: Dynamic Stage-level (SRTF) | 手动 disaggregated + 动态并行度 + SRTF |
| **TridentServe** | 自动 disaggregated + 动态 stage-level + ILP 调度 |

---

## 九、核心贡献总结

1. **首次系统性分析**扩散流水线的阶段异构性，揭示资源需求的不对称
2. 提出**动态 stage-level 服务范式**：Placement Plan（模型部署）+ Dispatch Plan（请求调度）
3. **Dynamic Orchestrator**：自动优化模型部署，无需人工调参
4. **Resource-Aware Dispatcher**：基于 ILP 的请求调度，最大化 SLO 达标率
5. **Adjust-on-Dispatch**：无停机的动态重部署机制
6. 代码量：核心约 12K 行 Python/Triton（Runtime Engine 10K + Planners 2K）

---

## 十、与 DiT-Serve 的对比

| 维度 | DiT-Serve | TridentServe |
|------|-----------|--------------|
| **关注点** | 跨请求的批处理优化 | 跨阶段的资源分配优化 |
| **粒度** | 步级（step-level） | 阶段级（stage-level） |
| **核心问题** | 异构请求的时间/空间低效 | 三阶段资源需求不对称 |
| **调度单位** | 去噪步 | Encode/Diffuse/Decode 阶段 |
| **并行策略** | Brick Attention（序列并行） | 动态选择 SP 度数 |
| **模型部署** | 固定的 DP 副本 | 动态的 Placement Plan |
| **互补性** | 解决 " 怎么批处理 " | 解决 " 怎么分配资源 " |

**两者是互补的：** TridentServe 解决了 stage-level 的资源分配问题，但在每个 stage 内部的批处理和注意力优化上，可以用 DiT-Serve 的技术。

---

## 十一、实现细节

- **代码量**：核心约 12K 行 Python/Triton
  - Runtime Engine：10K LOC
  - Planners：2K LOC
- **集成模型**：Stable Diffusion、PixArt、Flux、CogVideoX、HunyuanVideo、HunyuanDiT（约 16K LOC）
- **异步执行**：Ray + 协程（coroutines）
- **通信后端**：NCCL + NIXL
- **ILP 求解器**：PuLP
- **支持动态 batching**：多个轻量请求可以 batch 处理
- **兼容 MP**：通过将多个设备视为一个来集成模型并行
