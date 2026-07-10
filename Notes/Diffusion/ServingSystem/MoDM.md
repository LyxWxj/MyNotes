# MoDM: Efficient Serving for Image Generation via Mixture-of-Diffusion Models

**arXiv:** 2503.11972
**Date:** 2025-08-02
**Authors:** Xia, Yuchen; Sharma, Divyam; Yuan, Yichao; et al.
**Venue:** ASPLOS 2026

---

## 1. 论文概述

MoDM 是一个基于缓存的扩散模型推理系统，通过混合不同规模的扩散模型，动态平衡延迟与质量。

**核心贡献**：
- 缓存最终图像（而非中间特征），实现跨模型家族复用
- 小模型处理缓存命中请求，大模型处理缓存未命中请求
- 全局监控器优化 GPU 资源分配
- 平均服务时间降低 **2.5×**，质量无损

---

## 2. 背景与问题

### 2.1 模型规模与质量的权衡

| 模型 | 参数量 | 延迟 | 质量 |
|------|--------|------|------|
| SD-Turbo | 1B | 0.5s | 中等 |
| SDXL | 2.6B | 2.0s | 高 |
| SD3 | 8B | 8.0s | 极高 |

**问题**：如何在满足质量要求的前提下最小化延迟？

### 2.2 现有缓存方法的局限

**内部特征缓存**：
- 缓存模型中间层特征
- **问题**：模型特异性强，无法跨模型复用

**语义缓存**：
- 缓存 prompt embedding
- **问题**：相同 prompt 可能生成不同图像

---

## 3. 核心技术

### 3.1 图像级缓存

**核心思想**：缓存最终生成的图像，而非中间特征

```
缓存结构:
{
  "prompt_hash": "abc123",
  "image": <PIL Image>,
  "model": "SDXL",
  "timestamp": "2025-01-01 12:00:00",
  "quality_score": 0.92
}
```

**优势**：
1. **跨模型复用**：任何模型生成的图像都可复用
2. **质量可控**：只缓存高质量图像
3. **简单高效**：无需存储中间状态

### 3.2 混合模型调度

**调度策略**：

```
请求到达 → 查找缓存
            ↓
    ┌───────┴───────┐
    │ 缓存命中？     │
    └───────┬───────┘
        ↓ 是    ↓ 否
    ┌───────┐ ┌───────┐
    │小模型  │ │大模型  │
    │处理   │ │处理   │
    └───────┘ └───────┘
        ↓        ↓
    ┌───────┐ ┌───────┐
    │缓存图  │ │生成新  │
    │+小模型 │ │图像   │
    │增强   │ │       │
    └───────┘ └───────┘
```

**小模型增强**：
```python
def enhance_with_cache(small_model_output, cached_image):
    # 使用缓存图像增强小模型输出
    # 方法 1: 风格迁移
    enhanced = style_transfer(small_model_output, cached_image)
    # 方法 2: 质量增强
    enhanced = quality_enhance(small_model_output, cached_image)
    return enhanced
```

### 3.3 全局监控器

**资源分配优化**：

$$\min \sum_{i} \text{Latency}_i \quad \text{s.t.} \quad \sum_{j} \text{GPU}_j \leq \text{Total GPU}$$

**决策变量**：
- 分配给小模型的 GPU 数量
- 分配给大模型的 GPU 数量
- 缓存替换策略

**动态调整**：
```python
def adjust_resources(cache_hit_rate, request_rate):
    if cache_hit_rate > 0.7:
        # 缓存命中率高，增加小模型资源
        allocate_more_gpus("small_model")
    else:
        # 缓存命中率低，增加大模型资源
        allocate_more_gpus("large_model")
```

---

## 4. 系统实现

### 4.1 架构

```
┌─────────────────────────────────────────────────────┐
│                   MoDM Controller                   │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐│
│  │ Cache Manager│  │   Global     │  │  Request   ││
│  │              │  │   Monitor    │  │  Router    ││
│  └──────────────┘  └──────────────┘  └────────────┘│
└─────────────────────────────────────────────────────┘
         ↓                    ↓                  ↓
   ┌──────────┐        ┌──────────┐        ┌──────────┐
   │  Cache   │        │ Small    │        │  Large   │
   │  Store   │        │ Models   │        │  Models  │
   │          │        │ (N GPUs) │        │ (M GPUs) │
   └──────────┘        └──────────┘        └──────────┘
```

### 4.2 缓存查找

```python
def lookup_cache(prompt, similarity_threshold=0.95):
    # 计算 prompt embedding
    embedding = encode_prompt(prompt)
    
    # 在缓存中查找相似 prompt
    for cached_item in cache_store:
        sim = cosine_similarity(embedding, cached_item.embedding)
        if sim > similarity_threshold:
            return cached_item
            
    return None
```

### 4.3 缓存替换策略

**LRU + 质量权重**：
```python
def replace_cache(new_item):
    # 计算替换得分
    for cached_item in cache_store:
        age_score = time_since_last_access(cached_item)
        quality_score = cached_item.quality_score
        size_score = cached_item.image_size
        
        # 综合得分（越低越优先替换）
        score = quality_score / (age_score * size_score)
        cached_item.replace_score = score
        
    # 替换得分最低的
    to_replace = min(cache_store, key=lambda x: x.replace_score)
    cache_store.remove(to_replace)
    cache_store.append(new_item)
```

---

## 5. 实验结果

### 5.1 实验设置

- **模型**：SD-Turbo (1B), SDXL (2.6B), SD3 (8B)
- **硬件**：8× A100
- **负载**：真实 prompt 数据集

### 5.2 性能对比

| 方法 | 平均延迟 (s) | 吞吐量 (req/s) | 质量分数 |
|------|-------------|---------------|----------|
| Always Large | 8.0 | 1.2 | 0.95 |
| Always Small | 0.5 | 12.0 | 0.78 |
| Static Cache | 2.1 | 4.8 | 0.88 |
| **MoDM** | **0.8** | **10.5** | **0.93** |

### 5.3 缓存效果

| 缓存命中率 | 平均延迟 | 质量影响 |
|------------|----------|----------|
| 30% | 1.5s | < 1% |
| 50% | 1.0s | < 1% |
| 70% | 0.7s | < 2% |

---

## 6. 优势与局限

### 优势

1. **通用性强**：可跨模型家族复用缓存
2. **质量可控**：只缓存高质量图像
3. **资源高效**：动态调整模型资源

### 局限

1. **缓存空间**：图像占用存储空间大
2. **相似度阈值**：需要仔细调优
3. **新鲜度**：缓存图像可能过时

---

## 7. 与其他缓存方法的对比

| 方法 | 缓存内容 | 跨模型 | 加速比 |
|------|----------|--------|--------|
| 内部特征缓存 | 中间特征 | 否 | 1.5× |
| 语义缓存 | Prompt embedding | 是 | 1.2× |
| **MoDM** | **最终图像** | **是** | **2.5×** |

---

## 8. 关键术语

| 术语 | 解释 |
|------|------|
| Mixture-of-Diffusion Models | 混合不同规模扩散模型 |
| Image-level Caching | 图像级缓存，缓存最终生成结果 |
| Cache Hit/Miss | 缓存命中/未命中 |
| Global Monitor | 全局监控器，优化资源分配 |
| Quality-aware Scheduling | 质量感知调度 |
