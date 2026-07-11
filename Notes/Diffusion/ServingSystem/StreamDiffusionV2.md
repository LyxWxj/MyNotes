# StreamDiffusionV2: A Streaming System for Dynamic and Interactive Video Generation

**arXiv:** 2511.07399
**Date:** 2026-02-22
**Authors:** Feng, Tianrui; Li, Zhi; Yang, Shuo; et al.
**Affiliation:** UC Berkeley, Stanford, Adobe Research

---

## 1. 论文概述

StreamDiffusionV2 是一个面向 **交互式直播** 的流式视频生成系统，通过 SLO 感知调度、滚动 KV Cache 和可扩展流水线编排，实现实时生成。

**核心贡献**：
- 集成 SLO 感知批调度器和块调度器
- 设计 Sink-Token 引导的滚动 KV Cache
- 提出运动感知噪声控制器
- 实现可扩展流水线编排，近线性 FPS 扩展
- 首帧 0.5s，14B 模型 58.28 FPS（4×H100）

---

## 2. 背景与问题

### 2.1 直播场景的需求

**与离线生成的区别**：

| 需求 | 离线生成 | 直播生成 |
|------|----------|----------|
| 首帧延迟 | 可接受 | 必须最小化 |
| 帧率要求 | 24-30 FPS | 30-60 FPS |
| 交互性 | 无 | 实时响应 |
| 抖动容忍 | 高 | 极低 |

**SLO 定义**：
- **TTFF (Time to First Frame)**：首帧生成时间
- **Frame Deadline**：每帧截止时间
- **Jitter**：帧间延迟抖动

### 2.2 现有方法的局限

**图像基础流式模型**：
- 使用图像扩散模型逐帧生成
- **问题**：时间一致性差

**离线视频模型**：
- 批处理优化吞吐量
- **问题**：无法满足实时 SLO

---

## 3. 核心技术

### 3.1 SLO 感知批调度

**问题**：如何在满足 SLO 的前提下最大化吞吐量？

**批调度策略**：

```python
class SLOBatchScheduler:
    def __init__(self):
        self.pending_requests = []
        self.active_batches = []
        
    def schedule(self):
        # 按 SLO 紧急度排序
        self.pending_requests.sort(key=lambda r: r.slo_deadline)
        
        # 组批
        batch = []
        batch_deadline = float('inf')
        
        for request in self.pending_requests:
            # 检查是否可以加入当前批
            if self.can_add_to_batch(batch, request):
                batch.append(request)
                batch_deadline = min(batch_deadline, request.slo_deadline)
            else:
                # 提交当前批，开始新批
                if batch:
                    self.submit_batch(batch, batch_deadline)
                batch = [request]
                batch_deadline = request.slo_deadline
                
        # 提交最后一批
        if batch:
            self.submit_batch(batch, batch_deadline)
```

### 3.2 块调度器（Block Scheduler）

**问题**：不同去噪步骤的计算量不同

**解决方案**：将去噪过程分为块，独立调度

```
去噪步骤:
[Block 1: Step 1-2] → 低计算量，快速处理
[Block 2: Step 3-5] → 高计算量，优先调度
[Block 3: Step 6-8] → 中等计算量，正常处理
```

**块调度算法**：
```python
def schedule_blocks(blocks, deadline):
    # 计算每个块的紧急度
    for block in blocks:
        block.urgency = deadline - block.estimated_time
        
    # 按紧急度排序
    blocks.sort(key=lambda b: b.urgency)
    
    # 优先处理紧急块
    for block in blocks:
        if block.urgency < threshold:
            # 紧急：立即处理
            process_immediately(block)
        else:
            # 正常：排队处理
            enqueue(block)
```

### 3.3 Sink-Token 引导的滚动 KV Cache

**问题**：视频生成需要长上下文，KV Cache 占用大量显存

**解决方案**：使用 Sink-Token 引导的滚动 Cache

**Sink-Token 机制**：
```python
class RollingKVCache:
    def __init__(self, max_length, sink_tokens=4):
        self.max_length = max_length
        self.sink_tokens = sink_tokens  # 保留前 N 个 token
        self.cache = None
        
    def update(self, new_kv):
        if self.cache is None:
            self.cache = new_kv
        else:
            # 保留 sink tokens + 最近的 tokens
            sink = self.cache[:, :, :self.sink_tokens, :]
            recent = self.cache[:, :, -(self.max_length - self.sink_tokens):, :]
            
            # 拼接新 KV
            self.cache = torch.cat([sink, recent, new_kv], dim=2)
            
            # 截断到最大长度
            if self.cache.size(2) > self.max_length:
                self.cache = self.cache[:, :, :self.max_length, :]
```

**优势**：
1. **显存固定**：Cache 大小固定，不随序列增长
2. **质量保持**：Sink tokens 保持全局信息
3. **效率高**：滚动更新，无需重新计算

### 3.4 运动感知噪声控制器

**问题**：不同区域的运动幅度不同

**解决方案**：根据运动幅度调整噪声

```python
class MotionAwareNoiseController:
    def __init__(self):
        self.motion_estimator = MotionEstimator()
        
    def add_noise(self, latent, motion_map):
        # 运动幅度大的区域：更多噪声（需要更多去噪）
        # 运动幅度小的区域：更少噪声（可以复用）
        
        noise = torch.randn_like(latent)
        
        # 根据运动幅度缩放噪声
        noise_scale = 1.0 + motion_map * 0.5  # 运动大 → 噪声大
        scaled_noise = noise * noise_scale
        
        # 添加噪声
        noisy_latent = latent + scaled_noise
        return noisy_latent
```

### 3.5 可扩展流水线编排

**目标**：跨 GPU 线性扩展 FPS

**流水线设计**：
```
GPU 0: [Step 1] → [Step 2] → [Step 3] → ...
GPU 1: [Step 1] → [Step 2] → [Step 3] → ...
GPU 2: [Step 1] → [Step 2] → [Step 3] → ...
GPU 3: [Step 1] → [Step 2] → [Step 3] → ...

流水线化:
时间 →
GPU 0: [S1-F1] [S2-F1] [S3-F1] [S1-F2] [S2-F2] ...
GPU 1: [S1-F2] [S2-F2] [S3-F2] [S1-F3] [S2-F3] ...
GPU 2: [S1-F3] [S2-F3] [S3-F3] [S1-F4] [S2-F4] ...
GPU 3: [S1-F4] [S2-F4] [S3-F4] [S1-F5] [S2-F5] ...

F = Frame, S = Step
```

**并行维度**：
1. **步骤并行**：不同步骤在不同 GPU
2. **帧并行**：不同帧在不同 GPU
3. **层并行**：不同网络层在不同 GPU

---

## 4. 系统实现

### 4.1 架构

```
┌─────────────────────────────────────────────────────┐
│              StreamDiffusionV2 Pipeline              │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐│
│  │ SLO Batch    │  │   Block      │  │  Pipeline  ││
│  │ Scheduler    │  │  Scheduler   │  │  Orchestr. ││
│  └──────────────┘  └──────────────┘  └────────────┘│
└─────────────────────────────────────────────────────┘
         ↓                    ↓                  ↓
   ┌──────────────────────────────────────────────────┐
│              Diffusion Engine                        │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐│
│  │ Rolling KV   │  │ Motion-aware │  │  Multi-GPU ││
│  │ Cache        │  │ Noise Ctrl   │  │  Pipeline  ││
│  └──────────────┘  └──────────────┘  └────────────┘│
└──────────────────────────────────────────────────┘
```

### 4.2 关键实现

**滚动 Cache 实现**：
```python
class StreamingDiT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rolling_cache = RollingKVCache(
            max_length=config.cache_length,
            sink_tokens=config.sink_tokens
        )
        
    def forward(self, x, timestep):
        # 计算 KV
        kv = self.compute_kv(x)
        
        # 更新滚动 cache
        self.rolling_cache.update(kv)
        
        # 使用缓存的 KV 进行 attention
        cached_kv = self.rolling_cache.get()
        out = self.attention(x, cached_kv)
        
        return out
```

**帧率控制**：
```python
class FrameRateController:
    def __init__(self, target_fps=30):
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        
    def wait_for_next_frame(self):
        current_time = time.time()
        next_frame_time = self.last_frame_time + self.frame_interval
        
        if current_time < next_frame_time:
            # 等待到下一帧时间
            time.sleep(next_frame_time - current_time)
            
        self.last_frame_time = time.time()
```

---

## 5. 实验结果

### 5.1 实验设置

- **模型**：14B 参数视频扩散模型，1.3B 参数模型
- **硬件**：4× H100
- **分辨率**：720p

### 5.2 性能指标

| 指标 | 14B 模型 | 1.3B 模型 |
|------|----------|----------|
| 首帧延迟 | 0.5s | 0.3s |
| FPS | 58.28 | 64.52 |
| 抖动 | < 5ms | < 3ms |

### 5.3 扩展性

| GPU 数量 | 14B FPS | 扩展效率 |
|----------|---------|----------|
| 1 | 16.2 | 1.0× |
| 2 | 31.5 | 0.97× |
| 4 | 58.28 | 0.90× |

### 5.4 与基线对比

| 方法 | FPS | TTFF | 时间一致性 |
|------|-----|------|-----------|
| Image-based Stream | 45 | 0.8s | 差 |
| Offline Video | 30 | 5.2s | 好 |
| **StreamDiffusionV2** | **58** | **0.5s** | **好** |

---

## 6. 优势与局限

### 优势

1. **实时性**：满足直播 SLO
2. **高质量**：视频扩散模型的时间一致性
3. **可扩展**：近线性 FPS 扩展
4. **交互性**：支持实时控制

### 局限

1. **硬件要求**：需要多 GPU 支持
2. **模型限制**：需要适配的视频扩散模型
3. **质量权衡**：极端低延迟下质量下降

---

## 7. 与其他流式系统的对比

| 系统 | 基础模型 | FPS | 交互性 | 适用场景 |
|------|----------|-----|--------|----------|
| StreamDiffusion v1 | 图像模型 | 45 | 低 | 创意直播 |
| LivePortrait | 图像模型 | 60 | 中 | 虚拟人 |
| **StreamDiffusionV2** | **视频模型** | **58** | **高** | **通用直播** |

---

## 8. 关键术语

| 术语 | 解释 |
|------|------|
| SLO-aware Batching | SLO 感知批调度 |
| Block Scheduler | 块调度器，独立调度去噪块 |
| Rolling KV Cache | 滚动 KV Cache，固定显存 |
| Sink-Token | 水槽 token，保留全局信息 |
| Motion-aware Noise | 运动感知噪声控制 |
| Pipeline Orchestration | 流水线编排 |
