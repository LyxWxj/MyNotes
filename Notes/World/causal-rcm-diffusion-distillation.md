---
type: Note
related_to: "[[world-model]]"
status: Active
url: https://arxiv.org/abs/2606.25473
tags:
  - world-model
  - diffusion-distillation
  - causal-training
  - cosmos
  - streaming-video
---

# Causal-rCM: A Unified Teacher-Forcing and Self-Forcing Open Recipe for Autoregressive Diffusion Distillation

## 元信息

| 字段 | 内容 |
|------|------|
| **作者** | Kaiwen Zheng, Guande He, Min Zhao, Jintao Zhang, Huayu Chen, Jianfei Chen, Chen-Hsuan Lin, Ming-Yu Liu, Jun Zhu, Qianli Ma |
| **机构** | NVIDIA, Tsinghua University |
| **日期** | 2026-06-24 |
| **arXiv ID** | 2606.25473 |
| **URL** | https://arxiv.org/abs/2606.25473 |

## 摘要

Causal-rCM 将 rCM（consistency model + distribution matching distillation）框架扩展到自回归视频扩散领域。核心发现是 forward divergence（CM）和 reverse divergence（DMD）的互补性：teacher-forcing 提供离线的、forward-divergence 因果训练范式，self-forcing 提供 on-policy 的、reverse-divergence 精炼。Causal-rCM 统一了两者，提出了一个领先的、统一的、可扩展的算法-基础设施开源方案。

## 核心贡献

1. **Teacher-forcing 作为最优初始化**：实验表明 teacher-forcing CM 是 self-forcing DMD 的最佳互补初始化策略
2. **首个自回归连续时间 CM 实现**：通过自定义 mask FlashAttention-2 JVP kernel 实现，比离散时间 CM (dCM) 快 10 倍收敛
3. **Causal-rCM 算法**：统一的、可扩展的扩散蒸馏和因果训练方案
4. **SOTA 流式视频生成**：在帧级和 chunk 级流式生成中均达到 SOTA，仅使用合成数据训练

## 方法

- **Teacher-forcing (TF)**：离线训练范式，使用真实前序帧作为条件，对应 forward divergence
- **Self-forcing (SF)**：on-policy 训练范式，使用模型自身生成的前序帧，对应 reverse divergence
- **Consistency Model (CM)**：forward divergence 的代表方法
- **Distribution Matching Distillation (DMD)**：reverse divergence 的代表方法
- **Causal-rCM**：统一 TF-CM 初始化 + SF-DMD 精炼的两阶段训练
- **自定义 JVP Kernel**：FlashAttention-2 的 JVP（Jacobian-Vector Product）自定义 mask，实现连续时间 CM

## 实验结果

- 蒸馏后的 2-step 因果 Wan2.1-1.3B 模型在仅 1-2 个采样步下达到 VBench-T2V 84.63 分
- 仅使用合成数据训练即达到 SOTA
- 成功应用到 Cosmos 3——NVIDIA 的先进全模态世界基础模型

## 与其他工作的关系

- **与 rCM 的关系**：Causal-rCM 是 rCM 在自回归视频扩散中的扩展
- **与 Cosmos 3 的关系**：被应用于 Cosmos 3 的视频生成骨干，启用交互式世界模型
- **与 Self Forcing 的关系**：Self Forcing 是 reverse divergence 方法，Causal-rCM 统一了 TF 和 SF
- **与 CausVid 的关系**：CausVid 也是因果视频生成方法，Causal-rCM 提供了更统一的框架
- **与 Inferix 的关系**：Inferix 支持 Self Forcing 和 CausVid 等模型的推理

## 笔记

- 核心 insight：forward divergence (CM) 和 reverse divergence (DMD) 在自回归扩散蒸馏中互补
- Teacher-forcing 作为初始化 + Self-forcing 作为精炼，两阶段训练效果最佳
- 连续时间 CM 比离散时间 CM 快 10 倍收敛，得益于自定义 JVP kernel
- 仅用合成数据训练即达到 SOTA，说明蒸馏方法的数据效率很高
- 应用到 Cosmos 3 说明该方法有实际部署价值
- 与 Inferix 形成上下游关系：Causal-rCM 负责蒸馏加速（减少去噪步数），Inferix 负责推理加速
