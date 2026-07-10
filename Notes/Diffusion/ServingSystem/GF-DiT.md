# GF-DiT: Scheduling Parallelism for Diffusion Transformer Serving

**arXiv:** 2606.13501
**Date:** 2026-07-02
**Authors:** Qiang, Xinwei; Hu, Yifan; Sun, Shixuan; et al.
**Affiliation:** SJTU Liquid, Shanghai Jiao Tong University
**Code:** https://github.com/SJTU-Liquid/GF-DiT

---

## 1. 论文概述

GF-DiT 是一个策略可编程的弹性 DiT 推理运行时，将 GPU 并行度作为一等可调度资源，动态调整运行中请求的并行配置。

**核心贡献**：
- 提出异步执行抽象，将请求分解为可独立调度的轨迹任务
- 设计 Group-Free Collectives，支持低开销的在线执行组重配置
- 吞吐量提升 **6.01×**，延迟降低 **95%**，SLO 违规率降低 **90%**
- 通信组建立开销从 778ms 降至 **60μs**
- 已集成到 **vLLM-Omni**

---

## 2. 背景与问题

### 2.1 DiT 推理的异构性

**请求间异构**：
- 不同分辨率：144p vs 720p
- 不同序列长度：短视频 vs 长视频
- 不同模型：SDXL vs SVD

**阶段间异构**：
```
去噪步骤:
Step 1-2: 低计算量，可低并行度
Step 3-5: 高计算量，需高并行度
Step 6-8: 中等计算量，可中等并行度
```

**系统间异构**：
- 负载波动
- GPU 可用性变化
- 网络带宽变化

### 2.2 静态并行的问题

**固定并行配置**：

```python
# 静态配置：整个请求生命周期使用相同并行度
request = DiTRequest(resolution="720p", steps=8)
request.parallel_degree = 4  # 固定为 4 GPU

for step in range(8):
    # 所有步骤都使用 4 GPU
    process_step(request, step, gpus=4)
```

**问题**：
1. **资源浪费**：低计算量步骤使用过多 GPU
2. **利用率低**：无法在请求间动态分配 GPU
3. **SLO 违规**：无法适应负载变化

---

## 3. 核心技术

### 3.1 异步执行抽象

**核心思想**：将请求分解为独立可调度的轨迹任务

**轨迹任务（Trajectory Task）**：
```python
class TrajectoryTask:
    def __init__(self, request, step, chunk):
        self.request = request
        self.step = step      # 去噪步骤
        self.chunk = chunk    # 序列分块
        
    def execute(self, gpus):
        # 使用指定 GPU 执行
        return process_chunk(self.request, self.step, self.chunk, gpus)
```

**请求分解**：
```
原始请求: 720p, 8 steps, 16 chunks

分解为:
- Step 1, Chunk 1-4 → Task 1-4
- Step 1, Chunk 5-8 → Task 5-8
- Step 2, Chunk 1-4 → Task 9-12
...

每个任务可独立调度到不同 GPU
```

### 3.2 弹性并行调度

**调度策略**：

```python
def schedule_tasks(tasks, available_gpus):
    # 按优先级排序
    tasks.sort(key=lambda t: t.priority, reverse=True)
    
    for task in tasks:
        # 动态选择并行度
        parallel_degree = select_parallel_degree(task, available_gpus)
        
        # 分配 GPU
        gpus = allocate_gpus(available_gpus, parallel_degree)
        
        # 提交执行
        submit_task(task, gpus)
```

**并行度选择**：
```python
def select_parallel_degree(task, available_gpus):
    # 基于任务特性和可用资源选择
    if task.step < 2:
        # 早期步骤：低并行度
        return min(2, available_gpus)
    elif task.step < 6:
        # 中期步骤：高并行度
        return min(8, available_gpus)
    else:
        # 后期步骤：中等并行度
        return min(4, available_gpus)
```

### 3.3 Group-Free Collectives

**问题**：传统 NCCL 通信组建立开销大（~778ms）

**解决方案**：Group-Free Collectives，无需预建立通信组

**传统方式**：
```python
# 需要预先建立通信组
group = ncclCommInitRank(nranks, rank, ...)
# 通信组固定，无法动态调整
ncclAllReduce(data, group)
```

**Group-Free 方式**：
```python
# 直接使用 GPU ID 通信
ncclAllReduce(data, gpu_ids=[0, 1, 2, 3])
# 可动态调整参与的 GPU
ncclAllReduce(data, gpu_ids=[0, 1, 2])
```

**优势**：
1. **低开销**：通信组建立从 778ms 降至 60μs
2. **灵活性**：可动态调整参与的 GPU
3. **兼容性**：兼容现有 NCCL 接口

---

## 4. 系统实现

### 4.1 架构（vLLM-Omni 集成）

```
┌─────────────────────────────────────────────────────┐
│                  GF-DiT Runtime                     │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐│
│  │  Trajectory  │  │   Elastic    │  │  Group-    ││
│  │  Decomposer  │  │  Scheduler   │  │  Free NCCL ││
│  └──────────────┘  └──────────────┘  └────────────┘│
└─────────────────────────────────────────────────────┘
         ↓                    ↓                  ↓
   ┌──────────────────────────────────────────────────┐
│              GPU Resource Pool                       │
└──────────────────────────────────────────────────┘
         ↓            ↓            ↓            ↓
   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
   │  GPU 0  │  │  GPU 1  │  │  GPU 2  │  │  GPU 3  │
   └─────────┘  └─────────┘  └─────────┘  └─────────┘
```

### 4.2 轨迹任务调度

```python
class TrajectoryScheduler:
    def __init__(self):
        self.task_queue = PriorityQueue()
        self.gpu_pool = GPUPool()
        
    def schedule(self):
        while not self.task_queue.empty():
            # 获取最高优先级任务
            task = self.task_queue.get()
            
            # 选择并行度
            degree = self.select_degree(task)
            
            # 分配 GPU
            gpus = self.gpu_pool.allocate(degree)
            
            # 异步提交
            self.submit_async(task, gpus)
            
    def select_degree(self, task):
        # 基于 SLO 和资源状态选择
        slo_remaining = task.slo_deadline - time.now()
        compute_needed = task.estimate_compute()
        
        # 计算满足 SLO 所需的最小并行度
        for degree in [1, 2, 4, 8]:
            latency = self.estimate_latency(compute_needed, degree)
            if latency < slo_remaining:
                return degree
                
        return 8  # 最大并行度
```

### 4.3 GPU 重分配

```python
def redistribute_gpus(running_tasks, freed_gpus):
    # 找出可从 GPU 中受益的任务
    for task in running_tasks:
        current_degree = task.current_degree
        optimal_degree = task.optimal_degree
        
        if optimal_degree > current_degree:
            # 分配更多 GPU
            additional = min(optimal_degree - current_degree, len(freed_gpus))
            task.add_gpus(freed_gpus[:additional])
            freed_gpus = freed_gpus[additional:]
```

---

## 5. 实验结果

### 5.1 实验设置

- **模型**：Stable Diffusion XL, Stable Video Diffusion
- **硬件**：8× A100
- **负载**：混合分辨率 + 混合序列长度

### 5.2 性能对比

| 方法 | 吞吐量 (req/s) | 平均延迟 (s) | SLO 违规率 |
|------|---------------|-------------|-----------|
| Static (DoP=1) | 2.1 | 12.3 | 45% |
| Static (DoP=4) | 5.3 | 4.2 | 18% |
| Ulysses | 8.7 | 2.8 | 12% |
| **GF-DiT** | **12.6** | **0.6** | **1.8%** |

### 5.3 通信组开销

| 方法 | 通信组建立时间 | 动态调整 |
|------|---------------|----------|
| NCCL Group | 778ms | 不支持 |
| **Group-Free** | **60μs** | **支持** |

### 5.4 弹性调度效果

| 场景 | 静态并行 | 弹性并行 | 提升 |
|------|----------|----------|------|
| 混合分辨率 | 5.3 req/s | 12.6 req/s | 2.38× |
| 负载波动 | 3.8 req/s | 11.2 req/s | 2.95× |
| GPU 故障 | 2.1 req/s | 9.8 req/s | 4.67× |

---

## 6. 优势与局限

### 优势

1. **弹性并行**：动态适应请求和系统变化
2. **低开销**：Group-Free Collectives 减少通信组开销
3. **高利用率**：GPU 可在请求间动态分配
4. **可编程**：支持自定义调度策略

### 局限

1. **实现复杂**：需要修改推理引擎
2. **预测依赖**：需要准确的延迟预测
3. **调试困难**：动态调度增加调试复杂度

---

## 7. 与其他并行方法的对比

| 方法 | 并行类型 | 动态性 | 通信开销 |
|------|----------|--------|----------|
| Data Parallelism | 数据并行 | 静态 | 低 |
| Ulysses | 序列并行 | 静态 | 高 |
| Ring Attention | 序列并行 | 静态 | 中 |
| **GF-DiT** | **弹性并行** | **动态** | **低** |

---

## 8. 关键术语

| 术语 | 解释 |
|------|------|
| Trajectory Task | 轨迹任务，请求分解后的独立调度单元 |
| Elastic Parallelism | 弹性并行，动态调整并行度 |
| Group-Free Collectives | 无组集合通信，低开销的动态通信 |
| Policy-programmable | 策略可编程，支持自定义调度策略 |
| SLO (Service Level Objective) | 服务级别目标 |
