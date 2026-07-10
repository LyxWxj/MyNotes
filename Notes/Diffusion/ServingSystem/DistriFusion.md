# DistriFusion: Distributed Parallel Inference for High-Resolution Diffusion Models

**arXiv:** 2402.19481
**Date:** 2024-02
**Authors:** Li, Muyang; Cai, Tianle; Cao, Jiaxin; et al.
**Affiliation:** MIT, NVIDIA

---

## 1. 论文概述

DistriFusion 是一个分布式并行推理框架，通过跨多 GPU 分片（patch）实现高分辨率图像生成。核心创新是 **Displaced Patch Parallelism**：利用扩散过程相邻步骤输入的高相似性，复用上一步的特征图提供上下文，从而支持异步通信。

**核心贡献**：
- 提出 Displaced Patch Parallelism，解决 patch 间交互的通信开销问题
- 支持异步通信，实现计算与通信的流水线重叠
- 在 Stable Diffusion XL 上实现 **6.1× 加速**（8×A100 vs 1×A100）
- CVPR 2024 Highlight

---

## 2. 背景与问题

### 2.1 高分辨率生成的挑战

高分辨率图像生成面临：
- **计算成本巨大**：DiT 的 Attention 计算复杂度 O(n²)
- **显存限制**：单 GPU 无法容纳完整计算图
- **延迟过高**：交互式应用无法接受长时间等待

### 2.2 现有并行方法的局限

1. **朴素 Patch Parallelism**：
   - 将输入分为多个 patch，分配到不同 GPU
   - **问题**：破坏 patch 间交互，损失保真度

2. **同步通信**：
   - 在每步计算中交换 patch 边界信息
   - **问题**：通信开销巨大，抵消并行收益

---

## 3. 核心技术：Displaced Patch Parallelism

### 3.1 关键观察

> **扩散过程的连续性**：相邻扩散步骤的输入高度相似。

数学表示：
$$\|x_t - x_{t-1}\| \ll \|x_t\|$$

其中 $x_t$ 是步骤 $t$ 的输入，$x_{t-1}$ 是上一步的输入。

### 3.2 设计思路

利用上一步的特征图作为当前步的上下文：

```
步骤 t-1: [GPU1 计算 patch1] [GPU2 计算 patch2]
                    ↓ 保存特征图
步骤 t:   [GPU1 计算 patch1] [GPU2 计算 patch2]
          ↑ 复用 t-1 的特征图作为上下文
```

### 3.3 异步通信机制

```
时间线:
GPU1: [计算 patch1] → [发送特征图] → [计算 patch1] → ...
GPU2: [计算 patch2] → [接收特征图] → [计算 patch2] → ...

关键: 发送/接收与下一块计算重叠
```

---

## 4. 系统实现

### 4.1 分片策略

- 将输入 latent 在空间维度上均匀分片
- 每个 GPU 处理一个 patch
- 边界区域保留重叠以减少接缝

### 4.2 特征图缓存

```python
# 每步计算后保存特征图
if step > 0:
    cached_features = current_features.detach()
    
# 下一步复用
if step > 0:
    context = cached_features[overlapping_region]
```

### 4.3 通信优化

- **异步发送/接收**：使用 NCCL 的异步通信原语
- **计算重叠**：在等待通信完成时继续计算非边界区域
- **流水线化**：将通信分解为小块，与计算交替执行

---

## 5. 实验结果

### 5.1 实验设置

- **模型**：Stable Diffusion XL
- **硬件**：8× NVIDIA A100
- **分辨率**：1024×1024 及以上

### 5.2 性能对比

| 方法 | A100 数量 | 延迟 (s) | 加速比 |
|------|----------|----------|--------|
| 朴素实现 | 1 | 12.3 | 1.0× |
| 朴素 Patch Parallelism | 8 | 2.1 | 5.9× |
| **DistriFusion** | 8 | **2.0** | **6.1×** |

### 5.3 质量对比

- **FID 分数**：与单 GPU 实现几乎无差异
- **视觉质量**：无明显接缝或伪影
- **PSNR**：> 40 dB（与原始输出对比）

---

## 6. 优势与局限

### 优势

1. **线性加速**：接近理想的并行效率
2. **质量无损**：利用相邻步骤相似性，保持生成质量
3. **通用性**：可应用于任何扩散模型架构

### 局限

1. **适用场景**：主要针对高分辨率，低分辨率收益有限
2. **通信依赖**：需要高带宽 GPU 互联（NVLink）
3. **模型限制**：需要扩散过程具有连续性假设

---

## 7. 启示与意义

DistriFusion 的核心思想——**利用扩散过程的时间局部性**——启发了后续许多工作：

1. **时间冗余利用**：相邻步骤可复用计算结果
2. **异步并行**：通信与计算重叠是关键
3. **Patch 级并行**：为空间维度并行提供了可行方案

这一思想在后续的 **Sequence Parallelism**（如 Ulysses, Ring Attention）中得到进一步发展。

---

## 8. 关键术语

| 术语 | 解释 |
|------|------|
| Displaced Patch Parallelism | 利用相邻步骤相似性的并行策略 |
| Feature Map Reuse | 复用上一步的特征图作为上下文 |
| Asynchronous Communication | 通信与计算重叠执行 |
| Spatial Sharding | 在空间维度上分片输入 |
