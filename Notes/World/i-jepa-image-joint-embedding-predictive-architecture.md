---
type: Note
related_to: "[[world-model]]"
status: Active
tags:
  - self-supervised
  - JEPA
  - representation-learning
  - image
  - vision-transformer
  - ICCV-2023
---

# I-JEPA: Image-based Joint Embedding Predictive Architecture

## 元信息

| 字段 | 内容 |
|------|------|
| **作者** | Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, Nicolas Ballas |
| **日期** | 2023-01 |
| **arXiv** | [2301.08243](https://arxiv.org/abs/2301.08243) |
| **URL** | https://arxiv.org/abs/2301.08243 |
| **Zotero Key** | 5LZPEG5F |
| **会议** | ICCV 2023 |
| **代码** | https://github.com/facebookresearch/ijepa |

## 摘要 (Abstract)

I-JEPA 是一种**非生成式的图像自监督学习方法**。核心思想：从单个上下文块，预测同一图像中各种目标块的**表示**。关键设计选择是 masking 策略：需要以足够大的尺度采样目标块（语义级），并使用足够有信息量的（空间分布的）上下文块。

## 核心贡献 (Key Contributions)

1. **JEPA 的图像实现** — 首次将 LeCun 提出的 JEPA 框架应用于图像领域
2. **无需数据增强** — 学习高度语义化的图像表示，不依赖手工设计的数据增强
3. **高度可扩展** — 与 Vision Transformer 结合时表现出良好的可扩展性
4. **训练效率高** — ViT-Huge/14 在 ImageNet 上使用 16 块 A100 GPU 在 72 小时内完成训练

## 方法 (Method)

### 整体架构

```
输入图像
    |
    v
[Masking] --> Context blocks + Target blocks
    |                    |
    v                    v
[Context Encoder]   [Target Encoder] (EMA)
    |                    |
    v                    v
[Predictor] -----> Predicted representations
                         |
                         v
                    Loss: predicted vs target representations
```

### 关键组件

1. **Context Encoder (ViT)**: 编码可见的 context blocks
2. **Target Encoder (EMA)**: 编码 masked target blocks，使用 EMA (Exponential Moving Average) 更新
3. **Predictor**: 从 context 表示预测 target 表示
4. **Loss**: 在表示空间中计算预测与目标的 MSE loss

### Masking 策略

- **Target blocks**: 足够大尺度，捕捉语义级信息
- **Context block**: 空间分布，提供足够的上下文信息
- **关键**: 不使用像素级重建，完全在表示空间中进行预测

### 与对比学习的区别

| 特性 | 对比学习 | I-JEPA |
|------|---------|--------|
| 数据增强 | 需要手工设计 | 不需要 |
| 负样本 | 需要 | 不需要 |
| 预测目标 | 不变性 | 表示预测 |
| 语义级别 | 视图不变性 | 语义块预测 |

## 实验结果 (Results)

### ImageNet 性能

- **线性探测 (Linear Probing)**: 强性能
- **半监督分类**: 优于之前的自监督方法
- **迁移任务**: 在多种下游任务上表现良好

### 消融研究

- Masking 策略至关重要：target block 大小和 context 分布都需要仔细设计
- EMA target encoder 是稳定训练的关键
- 不使用数据增强是 I-JEPA 的重要优势

## 与其他工作的关系 (Related Work)

- **JEPA (LeCun 2022)**: 本文是 JEPA 框架的首次实践验证
- **V-JEPA**: I-JEPA 的视频版本 (Bardes et al., 2024)
- **MAE (Masked Autoencoder)**: 同样使用 masking，但 MAE 在像素空间重建
- **DINO/DINOv2**: 同为自监督视觉表示学习，但使用对比学习
- **BYOL/SimSiam**: 无需负样本的对比学习方法
- **BEiT**: 使用离散 tokens 的 masked image modeling

## 个人笔记 (Notes)

- **I-JEPA 是 JEPA 理论的首次成功实践**，证明了在表示空间中预测的可行性
- **不使用数据增强**是一个重大优势：避免了对比学习中 augmentation 设计的繁琐
- **训练效率高**: 72 小时在 16 块 A100 上训练 ViT-Huge，效率很高
- **EMA target encoder** 的设计借鉴了 BYOL 等工作，但在 JEPA 框架下有新的意义
- 从 world model 的角度看，I-JEPA 学习的是图像的**内部表示模型**，能够预测图像中未见部分的表示
- 与 MAE 的对比最有意义：两者都使用 masking，但预测空间完全不同
- 这个工作为后续 V-JEPA 以及更广泛的 JEPA 应用奠定了基础
