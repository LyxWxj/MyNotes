---
type: Note
related_to: "[[world-model]]"
status: Active
tags:
  - self-supervised
  - JEPA
  - representation-learning
  - video
  - vision-transformer
  - temporal-reasoning
---

# V-JEPA: Video Joint Embedding Predictive Architecture

## 元信息

| 字段 | 内容 |
|------|------|
| **作者** | Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mahmoud Assran, Nicolas Ballas |
| **日期** | 2024-02 |
| **arXiv** | [2404.08471](https://arxiv.org/abs/2404.08471) |
| **URL** | https://arxiv.org/abs/2404.08471 |
| **Zotero Key** | AGGIF4MQ |
| **注** | Zotero 中记录的 arXiv ID 2404.16103 有误，正确为 2404.08471 |

## 摘要 (Abstract)

V-JEPA 将 JEPA 框架扩展到视频理解领域，通过在潜在空间中预测被遮盖的时空区域来学习时空表示。模型仅使用**特征预测目标**训练，不需要预训练的图像编码器、文本、负样本、重建或其他监督信号。在 200 万公开视频上训练，评估了下游图像和视频基准。

## 核心贡献 (Key Contributions)

1. **JEPA 的视频扩展** — 将 JEPA 框架从图像扩展到视频的时空域
2. **纯特征预测** — 仅使用特征预测作为唯一训练目标，无需任何外部监督
3. **从零训练** — 不使用预训练图像编码器，完全从视频数据从零开始训练
4. **通用视觉表示** — 学到的表示同时适用于运动和外观任务

## 方法 (Method)

### 整体架构

```
输入视频
    |
    v
[Spatiotemporal Masking] --> Context patches + Target patches
    |                              |
    v                              v
[Context Encoder (ViT)]     [Target Encoder (EMA)]
    |                              |
    v                              v
[Predictor] ------------> Predicted representations
                                |
                                v
                          Loss: predicted vs target representations
```

### 关键设计

1. **时空 Masking**: 在视频的时空维度上进行 masking，遮盖连续的时空区域
2. **Context Encoder (ViT)**: 编码可见的时空 patches
3. **Target Encoder (EMA)**: 编码被遮盖的目标区域，使用 EMA 更新
4. **Predictor**: 从 context 表示预测 target 表示
5. **纯特征预测**: Loss 仅在表示空间中计算，不涉及像素重建

### 训练细节

- **训练数据**: 200 万公开视频
- **最大模型**: ViT-H/16
- **从零训练**: 不使用任何预训练权重
- **无外部监督**: 不使用文本、标签、预训练编码器等

### 与 I-JEPA 的对比

| 特性 | I-JEPA | V-JEPA |
|------|--------|--------|
| 输入 | 图像 | 视频 |
| Masking | 空间 | 时空 |
| 预测维度 | 空间 | 时空 |
| 下游任务 | 图像分类 | 视频理解 + 图像理解 |

## 实验结果 (Results)

### 冻结骨干网络性能 (Frozen Backbone)

| 基准 | 性能 |
|------|------|
| Kinetics-400 | 81.9% |
| Something-Something-v2 | 72.2% |
| ImageNet-1K | 77.9% |

### 关键发现

- **运动和外观**: 特征预测从视频中产生**通用的视觉表示**，在运动和外观任务上都表现良好
- **时间推理**: 展示了视频理解中的时间推理能力
- **无需微调**: 使用冻结的骨干网络就能获得强性能

## 与其他工作的关系 (Related Work)

- **I-JEPA**: V-JEPA 的图像版本，本文将其扩展到视频 (Assran et al., 2023)
- **JEPA (LeCun 2022)**: 理论框架来源
- **VideoMAE**: 同样使用 masking 进行视频自监督，但在像素空间重建
- **VIVIT/Video Swin Transformer**: 视频 Transformer 架构
- **CLIP**: 使用文本监督的视觉表示学习
- **DINOv2**: 图像自监督学习

## 个人笔记 (Notes)

- **V-JEPA 是 JEPA 系列的重要里程碑**，将表示空间预测从图像扩展到视频
- **从零训练**是一个重要声明：不需要预训练的图像编码器，纯视频就能学到好的表示
- **冻结骨干网络的强性能**表明学到的表示具有很好的泛化性
- Something-Something-v2 上的 72.2% 特别有意义：这个数据集侧重时间推理
- 从 world model 的角度看，V-JEPA 学习的是**视频的时空世界模型**，能够预测被遮盖区域的表示
- 与 VideoMAE 的对比最有意义：两者都使用 masking，但 V-JEPA 在表示空间预测
- 这个工作验证了 LeCun 在 2022 年白皮书中的 JEPA 理论预言
- 对于理解 world model 的表示学习范式非常有帮助
