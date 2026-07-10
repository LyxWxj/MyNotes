# Chorus: Beyond Few-Step Inference - Accelerating Video DiT Serving with Inter-Request Caching

**arXiv:** 2604.04451
**Date:** 2026-04-06
**Authors:** Liu, Hao; Huang, Ye; Huang, Chenghuan; et al.

---

## 1. 论文概述

Chorus 是一个跨请求缓存加速方法，利用不同请求之间的相似性来加速 Video DiT 推理。针对工业级 4 步蒸馏模型，实现 **45% 加速**。

**核心贡献**：
- 首次提出跨请求（inter-request）缓存复用
- 设计三阶段缓存策略，适配蒸馏模型
- 引入 Token-Guided Attention Amplification 改善语义对齐

---

## 2. 背景与问题

### 2.1 现有缓存方法的局限

**Intra-request caching（请求内缓存）**：
- 利用单个请求内扩散步骤间的相似性
- 跳过冗余去噪步骤
- **问题**：在 4 步蒸馏模型上效果有限（步骤太少）

### 2.2 Video DiT 的新机会

> **关键观察**：相似 prompt 生成的视频在早期去噪步骤中具有高度相似的 latent 特征。

```
请求 A: "一只猫在沙发上睡觉"
请求 B: "一只狗在沙发上休息"

步骤 1-2: latent 特征高度相似（沙发、室内场景）
步骤 3-4: 开始分化（猫 vs 狗）
```

---

## 3. 核心技术：三阶段缓存策略

### 3.1 阶段划分

| 阶段 | 去噪步骤 | 缓存策略 | 理由 |
|------|----------|----------|------|
| Stage 1 | Step 1-2 | **完全复用** | 高度相似，差异小 |
| Stage 2 | Step 3 | **区域复用** | 部分区域相似，部分分化 |
| Stage 3 | Step 4 | **无复用** | 完全分化，需要独立生成 |

### 3.2 Stage 1: 完全复用

```python
# 查找相似请求的缓存
similar_request = find_similar(prompt_embedding, cache_db)
if similar_request:
    # 完全复用前两步的 latent
    latent = similar_request.cached_latent
else:
    # 正常生成
    latent = generate_initial_noise()
```

### 3.3 Stage 2: 区域复用

**核心思想**：视频的不同区域可能有不同的复用策略

```
视频帧布局:
┌─────────────────────┐
│   背景区域 (可复用)   │
│  ┌─────────────────┐ │
│  │   主体区域       │ │
│  │  (需要重新生成)  │ │
│  └─────────────────┘ │
│   前景区域 (可复用)   │
└─────────────────────┘
```

**区域选择**：
- 通过 attention map 识别主体区域
- 背景/前景区域复用缓存
- 主体区域独立生成

### 3.4 Token-Guided Attention Amplification

**问题**：完全复用可能导致语义对齐偏差

**解决方案**：在 attention 计算中放大 prompt token 的权重

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}} + \alpha \cdot \text{prompt\_bias}\right)V$$

其中 $\alpha$ 是放大系数，prompt_bias 强调 prompt 相关 token。

---

## 4. 系统实现

### 4.1 缓存管理

```python
class CacheManager:
    def __init__(self):
        self.cache_db = {}  # prompt_embedding -> latent
        
    def store(self, prompt_embedding, latent, step):
        """存储缓存"""
        key = self.compute_key(prompt_embedding)
        self.cache_db[key] = {
            'latent': latent,
            'step': step,
            'timestamp': time.time()
        }
        
    def retrieve(self, prompt_embedding, threshold=0.8):
        """检索相似缓存"""
        key = self.compute_key(prompt_embedding)
        # 使用 embedding 相似度查找
        similar = self.find_similar(key, threshold)
        return similar
```

### 4.2 相似度计算

使用 CLIP embedding 的余弦相似度：

$$\text{sim}(A, B) = \frac{E_A \cdot E_B}{\|E_A\| \|E_B\|}$$

阈值：$\theta = 0.8$（实验确定）

---

## 5. 实验结果

### 5.1 实验设置

- **模型**：工业级 4 步蒸馏 Video DiT
- **数据集**：WebVid-10M 子集
- **硬件**：8× A100

### 5.2 性能对比

| 方法 | 延迟 (s) | 加速比 | 质量损失 |
|------|----------|--------|----------|
| Baseline (无缓存) | 4.2 | 1.0× | - |
| Intra-request caching | 3.8 | 1.1× | < 1% |
| **Chorus** | **2.9** | **1.45×** | < 2% |

### 5.3 不同阶段的效果

| 阶段 | 复用率 | 贡献加速 |
|------|--------|----------|
| Stage 1 | 85% | 25% |
| Stage 2 | 60% | 15% |
| Stage 3 | 0% | 0% |

---

## 6. 优势与局限

### 优势

1. **对蒸馏模型有效**：4 步模型也能获得显著加速
2. **质量可控**：通过区域选择和 attention 放大控制质量
3. **通用性**：可应用于任何 Video DiT

### 局限

1. **相似请求依赖**：需要有足够的相似 prompt
2. **缓存开销**：需要存储大量 latent，占用显存
3. **阈值敏感**：相似度阈值需要仔细调优

---

## 7. 与其他方法的对比

| 方法 | 缓存类型 | 适用模型 | 加速比 |
|------|----------|----------|--------|
| DistriFusion | 空间分片 | 任何 DiT | 6.1× |
| **Chorus** | 跨请求 | 蒸馏 DiT | 1.45× |
| GF-DiT | 弹性并行 | 任何 DiT | 6.0× |

---

## 8. 关键术语

| 术语 | 解释 |
|------|------|
| Inter-request Caching | 跨请求缓存复用 |
| Three-stage Strategy | 三阶段缓存策略 |
| Token-Guided Attention Amplification | 放大 prompt token 权重的机制 |
| Distilled Model | 蒸馏模型（少步推理） |
| Latent Similarity | Latent 空间的相似度 |
