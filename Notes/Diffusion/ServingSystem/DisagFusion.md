# DisagFusion: Asynchronous Pipeline Parallelism and Elastic Scheduling for Disaggregated Diffusion Serving

**arXiv:** 2605.25550
**Date:** 2026-05-25
**Authors:** Zha, Hantian; Ma, Teng; Yong, Yang; et al.

---

## 1. 论文概述

DisagFusion 是一个解耦式扩散模型推理系统，通过异步流水线并行和弹性调度，在异构 GPU 上实现高效部署。

**核心贡献**：
- 提出异步流水线并行，重叠计算与阶段间通信
- 设计混合实例调度策略，结合性能预测与运行时反馈
- 实现 **3.4×-20.5× 吞吐量提升**，延迟降低 **18.5×**

---

## 2. 背景与问题

### 2.1 扩散模型的阶段特性

典型扩散推理流程：
```
[Text Encoder] → [Diffusion Transformer (DiT)] → [VAE Decoder]
     轻量              计算密集                   带宽受限
```

**各阶段特点**：

| 阶段 | 参数量 | 计算特点 | 显存需求 |
|------|--------|----------|----------|
| Text Encoder | 1-5B | 向量计算 | 中等 |
| DiT | 2-14B | Attention O(n²) | 高 |
| VAE Decoder | 300-500M | Conv + Upsample | 中等 |

### 2.2 整体部署的问题

**Monolithic 部署**：所有阶段放在同一 GPU 上

**问题**：
1. **资源不均衡**：DiT 阶段计算密集，VAE 阶段带宽受限
2. **显存浪费**：需要为最大阶段（DiT）预留全部显存
3. **无法异构**：无法针对阶段特性选择合适 GPU

### 2.3 解耦部署的挑战

**Decoupled 部署**：各阶段独立部署在不同 GPU

**新问题**：
1. **阶段间通信**：数据需要在 GPU 间传输
2. **负载不均衡**：不同阶段处理速度不同
3. **调度复杂**：需要动态调整实例比例

---

## 3. 核心技术

### 3.1 异步流水线并行

**传统流水线**：
```
阶段 A: [计算] → [发送] → [等待] → [计算] → ...
阶段 B: [等待] → [接收] → [计算] → [发送] → ...
        ↑ 大量等待时间
```

**异步流水线**：
```
阶段 A: [计算 chunk1] → [发送 chunk1 + 计算 chunk2] → ...
阶段 B: [接收 chunk1 + 计算 chunk0] → [发送 chunk1 + 计算 chunk2] → ...
        ↑ 通信与计算重叠
```

**关键设计**：
1. **分块处理**：将 latent 分成多个 chunk
2. **异步发送**：发送 chunk $i$ 与计算 chunk $i+1$ 重叠
3. **双缓冲**：使用两个缓冲区交替读写

### 3.2 弹性实例调度

**问题**：负载变化时，固定实例比例导致资源浪费

**解决方案**：动态调整各阶段的实例数

```
初始配置: Text Encoder: 1 GPU, DiT: 4 GPU, VAE: 1 GPU
负载变化: DiT 请求增多
新配置: Text Encoder: 1 GPU, DiT: 5 GPU, VAE: 0 GPU (共享)
```

**调度策略**：

1. **性能预测**：
   $$\text{Throughput}_i = f(\text{GPU}_i, \text{batch_size}, \text{resolution})$$
   
2. **运行时反馈**：
   - 监控各阶段队列长度
   - 检测瓶颈阶段

3. **动态调整**：
   - 瓶颈阶段增加实例
   - 空闲阶段减少实例

### 3.3 轻量级性能预测

**预测模型**：
$$\text{Latency} = \alpha \cdot \text{compute} + \beta \cdot \text{memory} + \gamma \cdot \text{comm}$$

**参数标定**：
- 离线 profiling 获取基准数据
- 在线微调适应实际负载

---

## 4. 系统实现

### 4.1 架构

```
┌─────────────────────────────────────────────────────┐
│                  DisagFusion Controller              │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐│
│  │ Performance  │  │   Elastic    │  │  Pipeline  ││
│  │  Predictor   │  │  Scheduler   │  │  Manager   ││
│  └──────────────┘  └──────────────┘  └────────────┘│
└─────────────────────────────────────────────────────┘
         ↓                    ↓                  ↓
   ┌──────────┐        ┌──────────┐        ┌──────────┐
   │  Text    │  ───→  │   DiT    │  ───→  │   VAE    │
   │ Encoder  │        │  Cluster │        │ Decoder  │
   │ (1 GPU)  │        │ (N GPUs) │        │ (M GPUs) │
   └──────────┘        └──────────┘        └──────────┘
```

### 4.2 异步通信实现

```python
class AsyncPipeline:
    def __init__(self, stages):
        self.stages = stages
        self.buffers = [DoubleBuffer() for _ in stages]
        
    def process(self, input_data):
        # 启动所有阶段的异步处理
        futures = []
        for i, stage in enumerate(self.stages):
            # 异步接收上一阶段的输出
            recv_future = self.buffers[i].async_recv()
            # 异步计算
            compute_future = stage.async_compute(recv_future)
            # 异步发送到下一阶段
            send_future = self.buffers[i+1].async_send(compute_future)
            futures.append(send_future)
            
        # 等待所有完成
        return wait_all(futures)
```

### 4.3 弹性调度算法

```python
def elastic_schedule(current_config, workload):
    # 预测各阶段吞吐量
    throughput = {}
    for stage, gpus in current_config.items():
        throughput[stage] = predict_throughput(stage, gpus, workload)
    
    # 找出瓶颈阶段
    bottleneck = min(throughput, key=throughput.get)
    
    # 调整：从空闲阶段借 GPU
    idle_stage = max(throughput, key=throughput.get)
    if throughput[idle_stage] > throughput[bottleneck] * 1.5:
        current_config[bottleneck] += 1
        current_config[idle_stage] -= 1
        
    return current_config
```

---

## 5. 实验结果

### 5.1 实验设置

- **模型**：Stable Video Diffusion, OpenSora
- **硬件**：8× A100, 8× V100 混合集群
- **负载**：真实 trace + 合成负载

### 5.2 性能对比

| 方法 | 吞吐量 (req/s) | P99 延迟 (s) |
|------|---------------|-------------|
| Monolithic | 2.1 | 18.5 |
| Static Decoupled | 5.3 | 7.2 |
| **DisagFusion** | **43.1** | **1.0** |

### 5.3 异构优势

| 配置 | 成本 | 吞吐量 |
|------|------|--------|
| 8× A100 | $32/h | 43.1 req/s |
| 4× A100 + 4× V100 | $20/h | 38.7 req/s |
| **混合配置** | **$20/h** | **最优性价比** |

---

## 6. 优势与局限

### 优势

1. **资源高效**：异步流水线减少等待时间
2. **弹性伸缩**：动态适应负载变化
3. **异构友好**：可利用不同 GPU 特性

### 局限

1. **实现复杂**：需要精细的流水线管理
2. **预测依赖**：性能预测需要准确标定
3. **通信开销**：阶段间通信仍有成本

---

## 7. 与其他解耦系统的对比

| 系统 | 流水线方式 | 调度策略 | 适用场景 |
|------|------------|----------|----------|
| Triton | 同步 | 固定 | 通用模型 |
| **DisagFusion** | **异步** | **弹性** | **扩散模型** |
| Splitwise | 同步 | 静态 | LLM |

---

## 8. 关键术语

| 术语 | 解释 |
|------|------|
| Disaggregated Serving | 解耦部署，各阶段独立服务 |
| Asynchronous Pipeline | 异步流水线，通信与计算重叠 |
| Elastic Scheduling | 弹性调度，动态调整实例比例 |
| Double Buffering | 双缓冲，交替读写减少等待 |
| Performance Prediction | 性能预测，预估阶段吞吐量 |
