# SlackServe: Adaptive Resource Management and Quality Control for Streaming Video Generation

**arXiv:** 2606.15319
**Date:** 2026-06-13
**Authors:** Xia, Yifei; Yuan, Hao; Ling, Suhan; et al.

---

## 1. 论文概述

SlackServe 是一个面向 **流式视频生成** 的播放连续性保障系统，通过播放缓冲（playout slack）驱动的资源调度和质量控制，实现实时视频流生成。

**核心贡献**：
- 提出播放缓冲（playout slack）作为统一调度信号
- 设计三级优先队列和弹性序列并行
- 引入双模态 Pareto 路由进行质量-延迟权衡
- QoE 提升 **1.64×-3.29×**，TTFC 降低 **1.61×-9.65×**

---

## 2. 背景与问题

### 2.1 流式视频生成

**与离线生成的区别**：

| 特性 | 离线生成 | 流式生成 |
|------|----------|----------|
| 输出方式 | 完整视频 | 逐 chunk 输出 |
| SLO | 总延迟 | 播放连续性 |
| 用户体验 | 等待后观看 | 实时观看 |

**流式生成流程**：
```
[Chunk 1] → [生成] → [播放]
[Chunk 2] → [生成] → [播放]
[Chunk 3] → [生成] → [播放]
...
```

### 2.2 播放连续性挑战

**SLO 定义**：
- **播放缓冲（Playout Slack）**：已生成但未播放的 chunk 数量
- **目标**：保持播放缓冲 > 0，避免播放卡顿

**问题场景**：
```
时间线:
生成: [Chunk1] ... [Chunk2] ... [Chunk3] ...
播放: [Chunk1] [Chunk2] [Chunk3] [???] ← 缓冲耗尽，卡顿！
      ↑_________________________↑
            播放缓冲逐渐减少
```

---

## 3. 核心技术

### 3.1 播放缓冲驱动调度

**核心思想**：将播放缓冲作为统一调度信号

**调度优先级**：
$$\text{Priority} = \frac{1}{\text{playout_slack}}$$

**缓冲状态分类**：

| 状态 | 缓冲量 | 策略 |
|------|--------|------|
| 安全 | > 3 chunks | 正常调度 |
| 警告 | 1-3 chunks | 提升优先级 |
| 危险 | < 1 chunk | 紧急调度 |

### 3.2 三级优先队列

**设计**：

```
┌─────────────────────────────────────────┐
│              Scheduler                  │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐│
│  │ High    │  │ Medium  │  │ Low     ││
│  │ Priority│  │ Priority│  │ Priority││
│  │ (危险)  │  │ (警告)  │  │ (安全)  ││
│  └─────────┘  └─────────┘  └─────────┘│
└─────────────────────────────────────────┘
         ↓            ↓            ↓
   ┌─────────┐  ┌─────────┐  ┌─────────┐
   │ 优先    │  │ 普通    │  │ 后台    │
   │ 处理    │  │ 处理    │  │ 处理    │
   └─────────┘  └─────────┘  └─────────┘
```

**调度算法**：
```python
def schedule_stream(stream):
    # 计算优先级
    slack = stream.playout_slack
    
    if slack < 1:
        # 危险：立即处理
        return HIGH_PRIORITY_QUEUE
    elif slack < 3:
        # 警告：提升优先级
        return MEDIUM_PRIORITY_QUEUE
    else:
        # 安全：正常处理
        return LOW_PRIORITY_QUEUE
```

### 3.3 弹性序列并行

**问题**：不同 chunk 可能需要不同的并行度

**解决方案**：根据缓冲状态动态调整

```python
def adjust_parallelism(stream, chunk):
    slack = stream.playout_slack
    
    if slack < 1:
        # 危险：最大并行度
        return max_parallelism
    elif slack < 3:
        # 警告：中等并行度
        return medium_parallelism
    else:
        # 安全：最小并行度（节省资源）
        return min_parallelism
```

### 3.4 双模态 Pareto 路由

**问题**：质量与延迟的权衡

**解决方案**：使用 Pareto 最优选择

```
质量-延迟 Pareto 前沿:
质量 ↑
     │    ★ 高质量高延迟
     │   ★
     │  ★  ← Pareto 前沿
     │ ★
     │★
     └──────────────────→ 延迟
```

**路由策略**：
```python
def pareto_routing(stream, chunk):
    slack = stream.playout_slack
    
    if slack > 3:
        # 安全：选择高质量配置
        return high_quality_config
    elif slack > 1:
        # 警告：选择平衡配置
        return balanced_config
    else:
        # 危险：选择低延迟配置
        return low_latency_config
```

---

## 4. 系统实现

### 4.1 架构

```
┌─────────────────────────────────────────────────────┐
│                  SlackServe Controller               │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐│
│  │  Playout     │  │   Priority   │  │  Pareto    ││
│  │  Monitor     │  │   Scheduler  │  │  Router    ││
│  └──────────────┘  └──────────────┘  └────────────┘│
└─────────────────────────────────────────────────────┘
         ↓                    ↓                  ↓
   ┌──────────────────────────────────────────────────┐
│              Resource Pool (N GPUs)                  │
└──────────────────────────────────────────────────┘
         ↓                    ↓                  ↓
   ┌──────────┐        ┌──────────┐        ┌──────────┐
   │ Stream 1 │        │ Stream 2 │        │ Stream 3 │
   │ slack=5  │        │ slack=2  │        │ slack=0  │
   │ (低优先) │        │ (中优先) │        │ (高优先) │
   └──────────┘        └──────────┘        └──────────┘
```

### 4.2 缓冲监控

```python
class PlayoutMonitor:
    def __init__(self):
        self.streams = {}
        
    def update(self, stream_id, chunk_generated, chunk_played):
        # 更新缓冲量
        self.streams[stream_id].generated += chunk_generated
        self.streams[stream_id].played += chunk_played
        
        # 计算 slack
        slack = self.streams[stream_id].generated - self.streams[stream_id].played
        self.streams[stream_id].playout_slack = slack
        
    def get_slack(self, stream_id):
        return self.streams[stream_id].playout_slack
```

### 4.3 质量控制

```python
class QualityController:
    def select_config(self, stream, chunk):
        slack = stream.playout_slack
        
        # 从 Pareto 前沿选择配置
        configs = self.get_pareto_configs()
        
        # 根据 slack 选择
        if slack > 3:
            # 高质量
            return configs[-1]  # 最高质量
        elif slack > 1:
            # 平衡
            return configs[len(configs) // 2]
        else:
            # 低延迟
            return configs[0]  # 最低延迟
```

---

## 5. 实验结果

### 5.1 实验设置

- **模型**：AR-DiT 流式视频生成模型
- **硬件**：16× H100
- **指标**：CPR (Continuous Play Ratio), TTFC (Time to First Chunk)

### 5.2 性能对比

| 方法 | CPR | TTFC (s) | 质量分数 |
|------|-----|----------|----------|
| FIFO | 0.45 | 2.5 | 0.92 |
| 静态优先级 | 0.62 | 1.8 | 0.90 |
| **SlackServe** | **0.95** | **0.3** | **0.91** |

### 5.3 不同负载下的表现

| 负载强度 | CPR 提升 | TTFC 降低 |
|----------|----------|-----------|
| 低负载 | 1.64× | 1.61× |
| 中负载 | 2.15× | 3.42× |
| 高负载 | 3.29× | 9.65× |

---

## 6. 优势与局限

### 优势

1. **播放连续性**：显式保障用户体验
2. **动态调度**：根据缓冲状态自适应
3. **质量可控**：Pareto 路由平衡质量与延迟

### 局限

1. **预测依赖**：需要准确的播放预测
2. **调度开销**：频繁调整有额外成本
3. **模型特化**：主要针对流式生成

---

## 7. 与其他流式系统的对比

| 系统 | 调度信号 | 质量控制 | 适用场景 |
|------|----------|----------|----------|
| FIFO | 无 | 无 | 通用 |
| 延迟驱动 | 延迟 | 固定 | 离线生成 |
| **SlackServe** | **播放缓冲** | **Pareto** | **流式生成** |

---

## 8. 关键术语

| 术语 | 解释 |
|------|------|
| Playout Slack | 播放缓冲，已生成但未播放的 chunk 数量 |
| Continuous Play Ratio (CPR) | 连续播放比率，衡量播放连续性 |
| Time to First Chunk (TTFC) | 首个 chunk 生成时间 |
| Pareto Routing | Pareto 最优路由，平衡质量与延迟 |
| Streaming Video Generation | 流式视频生成 |
