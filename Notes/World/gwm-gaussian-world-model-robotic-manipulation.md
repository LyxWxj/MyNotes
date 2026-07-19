---
type: Note
related_to: "[[world-model]]"
status: Active
url: https://arxiv.org/abs/2508.17600
tags:
  - world-model
  - gaussian-splatting
  - robotics
  - 3d-vae
  - dit
---

# GWM: Towards Scalable Gaussian World Models for Robotic Manipulation

## 元信息

| 字段 | 内容 |
|------|------|
| **作者** | Guanxing Lu, Baoxiong Jia, Puhao Li, Yixin Chen, Ziwei Wang, Yansong Tang, Siyuan Huang |
| **会议** | ICCV 2025 |
| **日期** | 2025-08-25 (v1), 2025-09-17 (v2) |
| **arXiv ID** | 2508.17600 |
| **URL** | https://arxiv.org/abs/2508.17600 |
| **项目主页** | https://gaussian-world-model.github.io/ |

## 摘要

GWM 提出了一种用于机器人操作的新型 3D 世界模型——Gaussian World Model。与现有图像级世界模型缺乏几何信息不同，GWM 通过推断 Gaussian primitives 在机器人动作作用下的传播来重建未来状态。其核心是 latent Diffusion Transformer (DiT) 结合 3D 变分自编码器（VAE），实现基于 Gaussian Splatting 的细粒度场景级未来状态重建。GWM 既可以通过自监督未来预测训练增强模仿学习的视觉表征，也可以作为神经模拟器支持基于模型的强化学习。

## 核心贡献

1. **提出 Gaussian World Model (GWM)**：首个将 Gaussian Splatting 用于机器人操作世界模型的工作
2. **Latent DiT + 3D VAE 架构**：在 latent 空间中进行扩散，结合 3D VAE 实现高效重建
3. **双重用途**：既可增强模仿学习表征，又可作为 MBRL 的神经模拟器
4. **数据扩展潜力**：展示了 3D 世界模型的初步数据扩展能力

## 方法

- **Gaussian Splatting 表示**：用 3D Gaussian primitives 表示场景状态，每个 Gaussian 包含位置、协方差、颜色等参数
- **3D VAE**：将 Gaussian 参数编码到 latent 空间，实现压缩和高效处理
- **Latent DiT**：在 latent 空间中用 Diffusion Transformer 预测未来状态的 Gaussian 参数
- **动作条件**：以机器人动作作为条件，预测动作导致的场景变化
- **两种训练模式**：
  - 自监督未来预测：增强视觉表征（用于模仿学习）
  - 神经模拟器：支持 model-based RL

## 实验结果

- 仿真和真实世界实验均验证 GWM 能精确预测不同机器人动作下的未来场景
- 训练出的策略在多个基准上显著超越 SOTA
- 展示了 3D 世界模型的数据扩展潜力

## 与其他工作的关系

- **与 UniSim/Genie 等 2D 世界模型的区别**：GWM 在 3D 空间中操作，具有显式几何理解，而非仅在图像空间
- **与 3D Gaussian Splatting 的关系**：将 3DGS 从静态场景重建扩展到动态未来预测
- **与 Cosmos 的关系**：Cosmos 是通用世界基础模型，GWM 是机器人操作专用的 3D 世界模型
- **与 DiT 架构的复用**：使用标准 DiT 架构在 latent 空间做扩散

## 笔记

- 核心 insight：图像级世界模型缺乏 3D 几何理解，而机器人操作需要精确的空间和物理理解
- Gaussian Splatting 作为 3D 场景表示非常适合世界模型——高效渲染、可微分、紧凑
- "神经模拟器"的概念很有吸引力——用学到的世界模型代替真实环境进行 RL 训练
- 3D VAE + Latent DiT 的组合是处理高维 3D 数据的有效方案
- ICCV 2025 发表，说明 3D 世界模型方向得到认可
- 关注数据扩展潜力：随着数据量增加，3D 世界模型的能力是否持续提升？
