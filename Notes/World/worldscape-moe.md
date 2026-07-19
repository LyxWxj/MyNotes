---
type: Note
related_to: "[[world-model]]"
status: Active
url: https://arxiv.org/abs/2607.03964
tags:
  - DiT
  - MoE
  - world-model
  - action-control
  - heterogeneous
---

# Worldscape-MoE: A Unified Mixture-of-Experts World Model for Scalable Heterogeneous Action Control

## Meta Information

- **Authors**: Jianjie Fang, Yongyan Xu, Ziyou Wang, Chen Gao, Yuchao Huang, Zhaolu Wang
- **Date**: 2026-07
- **arXiv ID**: 2607.03964
- **DOI**: 10.48550/arXiv.2607.03964
- **URL**: https://arxiv.org/abs/2607.03964
- **Zotero Key**: QPIK8TLN

## Abstract

A Mixture-of-Experts world model built on Diffusion Transformers for scalable heterogeneous action control. Supports locomotion, robotic manipulation, and egocentric hand control. Shows heterogeneous supervision improves rather than interferes with individual control capabilities.

## Key Contributions

1. **MoE世界模型**: 基于Diffusion Transformers的Mixture-of-Experts世界模型
2. **异构动作控制**: 支持多种异构动作控制
3. **多任务支持**: 支持运动(locomotion)、机器人操作、第一人称手部控制
4. **协同学习**: 异构监督提升而非干扰各控制能力

## Method

- Mixture-of-Experts (MoE) 架构
- Diffusion Transformers (DiT)
- 异构动作控制
- 多任务统一训练

## Results

- 支持三种控制任务：locomotion, robotic manipulation, egocentric hand control
- 异构监督提升各控制能力
- 可扩展的统一世界模型

## Related Work

- Mixture-of-Experts架构
- Diffusion Transformers
- 多任务学习
- 机器人控制

## Notes

- MoE架构在世界模型中的应用
- 异构任务的统一建模
- 多任务学习的正向迁移效应

---

*Last updated: 2026-07-18*
