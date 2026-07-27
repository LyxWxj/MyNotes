# I-JEPA: Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (2023)

> **论文**: I-JEPA: Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture
> **作者**: Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, Nicolas Ballas
> **来源**: arXiv 2301.08243 | ICCV 2023
> **分类**: cs.CV, cs.AI, cs.LG

---

## 一、核心思想

I-JEPA 是一种**非生成式**的图像自监督学习方法。给定单个上下文块（context block），预测同图像中多个目标块（target block）的**表示**（而非像素）。

**关键洞察**：预测在表示空间中进行，不需要手工数据增强。

**📖 什么是 Joint Embedding（联合嵌入）？**

**一句话**：两个编码器处理同一事物的不同"视角"，产生相似的表示。

**直觉**：你和朋友看同一张照片的不同部分，你们看到的像素完全不同，但脑中形成的"这是一只猫"的概念相同——在**表示空间**中对齐，而非像素空间。

```
输入 x → [视图 1] → x₁ → [编码器 f_θ] → z₁ ─┐
                                               ├→ 损失：让 z₁ ≈ z₂
输入 x → [视图 2] → x₂ → [编码器 f_ξ] → z₂ ─┘
```

**传统方法**（SimCLR、BYOL、DINO）依赖**手工数据增强**（裁剪、颜色抖动）生成两个视图，引入人为偏见。

**JEPA 的突破**：用**掩码**代替增强——可见部分和掩码部分就是两个"视图"，无需手工设计，学到的是自然语义表示。

| | 生成式 (MAE) | Joint Embedding |
|---|---|---|
| 预测目标 | 像素值 | 嵌入向量 |
| 学到什么 | 低级纹理 | 高级语义 |
| 数据增强 | 通常不需要 | 传统方法需要 |

---

## 二、方法论详解

### 架构组件

I-JEPA 使用三个 Vision Transformer 组件：

| 组件 | 功能 | 更新方式 |
|------|------|---------|
| **Context Encoder** | 编码可见块 | 梯度更新 |
| **Target Encoder** | 计算目标表示 | EMA 更新 |
| **Predictor** | 从上下文预测目标 | 梯度更新 |

### 目标构造

- 随机采样 M=4 个目标块
- **尺度**：0.15-0.2（较大，捕获语义信息）
- **宽高比**：0.75-1.5
- **关键**：目标块在**编码器输出**上掩码，而非输入
  → 确保高语义级别的目标

### 上下文构造

- 单个上下文块
- **尺度**：0.85-1.0（几乎整图）
- **宽高比**：1:1（正方形）
- 移除与目标块的重叠区域

### 预测过程

```
上下文编码器输出 + 位置掩码 token → 预测器 → 目标块的预测表示
```

- 预测器对每个目标块应用一次（M 次）
- 位置掩码 token = 可学习向量 + 位置嵌入

### 损失函数

$$\mathcal{L} = \frac{1}{M} \sum_{i=1}^{M} \sum_{j \in B_i} \| \hat{s}_{y_j} - s_{y_j} \|^2$$

- 预测和目标 patch 级表示之间的 L2 距离
- 目标编码器通过 EMA 更新（非梯度）

---
