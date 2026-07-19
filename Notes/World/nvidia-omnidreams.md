---
type: Note
related_to: "[[world-model]]"
status: Active
url: https://arxiv.org/abs/2606.03159
tags:
  - world-model
  - autonomous-driving
  - simulation
  - NVIDIA
  - real-time
---

# NVIDIA OmniDreams: Real-Time Generative World Model for Closed-Loop Autonomous Vehicle Simulation

## Meta Information

- **Authors**: NVIDIA, Aarti Basant, Amlan Kar, Despoina Paschalidou, Sanja Fidler
- **Date**: 2026-06
- **arXiv ID**: 2606.03159
- **DOI**: 10.48550/arXiv.2606.03159
- **URL**: https://arxiv.org/abs/2606.03159
- **Zotero Key**: 33ZQ749V

## Abstract

A foundation generative world model mid- and post-trained from the Cosmos diffusion model to autoregressively generate action-conditioned videos in real time. Trained on 21k hours of driving scenarios. A WAM post-trained from OmniDreams surpasses the VLA-based Alpamayo 1.5 research policy model while using only 1/5 the total parameters.

## Key Contributions

1. **Foundation World Model**: 基于Cosmos扩散模型进行中期和后期训练的基础生成式世界模型
2. **Real-Time Generation**: 能够实时自回归生成动作条件视频
3. **大规模驾驶数据训练**: 使用21k小时驾驶场景数据进行训练
4. **高效策略学习**: 从OmniDreams后期训练的WAM仅使用1/5参数就超越了基于VLA的Alpamayo 1.5策略模型

## Method

- 基于Cosmos扩散模型架构
- 中期训练(mid-training)和后期训练(post-training)策略
- 自回归视频生成
- 动作条件化(action-conditioned)生成

## Results

- 在21k小时驾驶场景数据上训练
- WAM后期训练版本超越VLA-based Alpamayo 1.5
- 参数效率：仅使用1/5的总参数
- 实时生成能力

## Related Work

- Cosmos扩散模型
- Alpamayo 1.5 (VLA-based policy model)
- 自动驾驶仿真领域相关工作

## Notes

- 这是NVIDIA在自动驾驶世界模型领域的重要工作
- 强调了实时性和参数效率
- 适用于闭环自动驾驶仿真

---

*Last updated: 2026-07-18*
