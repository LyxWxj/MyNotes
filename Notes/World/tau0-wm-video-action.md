---
type: Note
related_to: "[[world-model]]"
status: Active
url: https://arxiv.org/abs/2606.01027
zotero_key: KJXM3J3J
tags:
  - world-model
  - robotic-manipulation
  - policy-learning
  - video-action
---

# tau0-WM: A Unified Video-Action World Model for Robotic Manipulation

## Meta

- **Authors**: Pengfei Zhou, Shengcong Chen, Di Chen, Jiaxu Wang, Rongjun Jin, Bingwen Zhu
- **Date**: 2026-06
- **arXiv**: [2606.01027](https://arxiv.org/abs/2606.01027)
- **DOI**: 10.48550/arXiv.2606.01027

## Abstract

A unified video-action world model that integrates policy learning, video prediction, and action evaluation within a single future-predictive framework. Trained on approximately 27,300 hours of real-robot teleoperation data.

## Key Contributions

- 提出统一的视频-动作世界模型框架，同时处理策略学习、视频预测和动作评估
- 在单一的未来预测框架内整合三个关键组件
- 使用约 27,300 小时的真实机器人遥操作数据进行训练
- 展示了大规模真实数据对世界模型性能的重要性

## Method

- **统一框架**: 将 policy learning、video prediction、action evaluation 统一在一个 future-predictive 框架中
- **视频-动作联合建模**: 同时预测未来视频帧和对应的机器人动作
- **大规模数据训练**: 基于 27,300 小时真实机器人遥操作数据
- **未来预测范式**: 通过预测未来状态来指导当前决策

## Results

- 在机器人操作任务上取得优异表现
- 验证了统一框架相比分离模块的优势
- 大规模真实数据训练显著提升了泛化能力

## Related Work

- 与 RT-2 等视觉-语言-动作模型 (VLA) 的区别在于显式的视频预测能力
- 与 UniSim 等仿真世界模型的对比，本文使用真实数据
- 与 diffusion policy 等方法在动作生成上的互补

## Notes

- 统一视频-动作建模是机器人领域的重要趋势，本文提供了大规模验证
- 27,300 小时的真实数据规模值得关注，体现了数据驱动方法的潜力
- 关注其在不同操作任务上的泛化表现
- "tau0" 命名暗示了与物理时间常数的关联，值得深入理解
