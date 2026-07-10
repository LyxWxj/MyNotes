# DiLaServe: High SLO Attainment Serving for Diffusion Language Models

**arXiv:** 2606.29094
**Date:** 2026-06-27
**Authors:** Chang, Tzu-Tao; Hong, Benjamin Yuanyang; Pham, Kiet; et al.
**Affiliation:** Microsoft Research, Stanford

---

## 1. 论文概述

DiLaServe 是一个面向 **扩散语言模型（DLM）** 的集群级推理系统，通过置信度阈值调整、质量感知优化和近似 KV Cache 建模，实现高 SLO 达成率。

**核心贡献**：
- 提出置信度阈值自适应调整，平衡速度与质量
- 设计质量感知集群重配置优化
- 显式建模近似 KV Cache 的步骤级异构性
- SLO 达成率提升 **56.6 个百分点**，延迟降低 **46%**

---

## 2. 背景与问题

### 2.1 扩散语言模型（DLM）

**与自回归模型的区别**：

| 特性 | 自回归模型 (AR) | 扩散语言模型 (DLM) |
|------|----------------|-------------------|
| 生成方式 | 逐 token 生成 | 多 token 并行去噪 |
| 推理步骤 | N 步（序列长度） | K 步（去噪步数） |
| 并行度 | 低 | 高 |
| 质量 | 高 | 竞争力 |

**DLM 推理流程**：
```
初始噪声 → [去噪步骤 1] → [去噪步骤 2] → ... → [去噪步骤 K] → 输出
           ↓ 并行生成多个 token
```

### 2.2 DLM 推理的挑战

1. **速度-质量权衡**：
   - 置信度阈值高 → 质量好但慢
   - 置信度阈值低 → 快但质量差

2. **并行度选择**：
   - 不同请求需要不同的并行度
   - 负载变化时需要动态调整

3. **近似 KV Cache**：
   - DLM 使用近似 KV Cache 加速
   - 不同步骤的 cache 命中率不同，导致成本异构

---

## 3. 核心技术

### 3.1 置信度阈值自适应

**DLM 去噪机制**：
```python
for step in range(K):
    # 生成候选 token
    candidates = model.denoise(latent, step)
    
    # 置信度过滤
    confident_tokens = candidates[candidates.confidence > threshold]
    
    # 已确认 token 不再参与后续去噪
    latent = remove_confirmed(latent, confident_tokens)
```

**阈值调整策略**：
- **SLO 压力大**：降低阈值，快速生成
- **SLO 宽松**：提高阈值，保证质量

**自适应算法**：
$$\text{threshold}_t = \text{threshold}_{t-1} + \alpha \cdot (\text{SLO}_\text{target} - \text{SLO}_\text{actual})$$

### 3.2 质量感知集群重配置

**问题**：负载变化时，如何调整集群配置？

**优化目标**：
$$\max \text{Quality} \quad \text{s.t.} \quad \text{SLO} \leq \text{target}$$

**决策变量**：
- 每个实例的并行度
- 请求分配策略
- 置信度阈值

**求解方法**：
1. 离线 profiling 建立质量-延迟模型
2. 在线求解线性规划
3. 运行时微调

### 3.3 近似 KV Cache 建模

**问题**：近似 KV Cache 导致不同步骤成本不同

**观察**：
```
步骤 1: Cache 命中率 90% → 低成本
步骤 2: Cache 命中率 70% → 中等成本
步骤 3: Cache 命中率 50% → 高成本
```

**建模**：
$$\text{Cost}_k = \text{base_cost} + (1 - \text{hit_rate}_k) \times \text{miss_penalty}$$

**调度优化**：
- 将高成本步骤分配到计算能力强的 GPU
- 将低成本步骤分配到普通 GPU

---

## 4. 系统实现

### 4.1 架构

```
┌─────────────────────────────────────────────────────┐
│                  DiLaServe Controller                │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐│
│  │  Threshold   │  │   Cluster    │  │   Cache    ││
│  │  Adapter     │  │  Reconfig    │  │  Modeler   ││
│  └──────────────┘  └──────────────┘  └────────────┘│
└─────────────────────────────────────────────────────┘
         ↓                    ↓                  ↓
   ┌──────────────────────────────────────────────────┐
   │              Request Scheduler                    │
   └──────────────────────────────────────────────────┘
         ↓                    ↓                  ↓
   ┌──────────┐        ┌──────────┐        ┌──────────┐
   │  DLM     │        │  DLM     │        │  DLM     │
   │Instance 1│        │Instance 2│        │Instance 3│
   └──────────┘        └──────────┘        └──────────┘
```

### 4.2 调度算法

```python
def schedule_request(request, instances):
    # 估计请求的 SLO 需求
    slo_budget = estimate_slo(request)
    
    # 选择最佳实例
    best_instance = None
    best_score = float('inf')
    
    for instance in instances:
        # 预测延迟和质量
        latency = predict_latency(instance, request)
        quality = predict_quality(instance, request)
        
        # 计算综合得分
        score = latency / slo_budget + quality_weight * (1 - quality)
        
        if score < best_score:
            best_score = score
            best_instance = instance
            
    return best_instance
```

### 4.3 置信度调整

```python
def adjust_threshold(current_threshold, slo_actual, slo_target):
    # PID 控制器
    error = slo_target - slo_actual
    adjustment = kp * error + ki * integral(error) + kd * derivative(error)
    
    # 更新阈值
    new_threshold = current_threshold + adjustment
    
    # 限制范围
    return clamp(new_threshold, min_threshold, max_threshold)
```

---

## 5. 实验结果

### 5.1 实验设置

- **模型**：MDLM, SEDD 等 DLM 模型
- **硬件**：32× A100 集群
- **负载**：真实 trace + 合成负载

### 5.2 性能对比

| 方法 | SLO 达成率 | 平均延迟 (s) | 质量损失 |
|------|-----------|-------------|----------|
| FIFO | 42.3% | 8.2 | - |
| 静态阈值 | 58.7% | 6.1 | < 1% |
| **DiLaServe** | **99.3%** | **4.4** | **< 1%** |

### 5.3 不同负载下的表现

| 负载强度 | SLO 达成率 | 延迟降低 |
|----------|-----------|----------|
| 低负载 | 99.9% | 35% |
| 中负载 | 99.3% | 46% |
| 高负载 | 95.2% | 38% |

---

## 6. 优势与局限

### 优势

1. **SLO 感知**：显式优化 SLO 达成率
2. **质量可控**：置信度阈值平衡速度与质量
3. **异构建模**：考虑 KV Cache 的步骤级差异

### 局限

1. **模型特化**：主要针对 DLM，通用性有限
2. **预测依赖**：需要准确的延迟/质量预测
3. **调参复杂**：多个超参数需要调优

---

## 7. 与其他 DLM 推理系统的对比

| 系统 | 优化目标 | 适用模型 | 特点 |
|------|----------|----------|------|
| **DiLaServe** | SLO 达成率 | DLM | 置信度自适应 |
| DistriFusion | 延迟 | DiT | 空间并行 |
| GF-DiT | 吞吐量 | DiT | 弹性并行 |

---

## 8. 关键术语

| 术语 | 解释 |
|------|------|
| Diffusion Language Model (DLM) | 扩散语言模型 |
| Confidence Threshold | 置信度阈值，决定何时确认 token |
| Approximate KV Cache | 近似 KV Cache，加速 DLM 推理 |
| SLO (Service Level Objective) | 服务级别目标 |
| Quality-aware Optimization | 质量感知优化 |
