---
type: Note
related_to: "[[world-model]]"
status: Active
url: https://arxiv.org/abs/2606.04130
zotero_key: 39CF3T6N
tags:
  - world-model
  - diffusion
  - adversarial
  - latent-action
  - self-supervised
---

# CLAW: Learning Continuous Latent Action World Models via Adversarial Latent Regularization

## Meta

- **Authors**: Tewodros Ayalew, Matthew Jeung, Samuel Wheeler, Xiao Zhang, Andre de la Cruz Arce, Kaylene Stocking, Michael Maire, Matthew R. Walter
- **Date**: 2026-06
- **arXiv**: [2606.04130](https://arxiv.org/abs/2606.04130)
- **DOI**: 10.48550/arXiv.2606.04130

## Abstract

A fully end-to-end self-supervised framework for learning a world model jointly with continuous latent action representations directly from action-free videos. Uses adversarial latent regularization and diffusion-based video generation.

## Key Contributions

- 提出端到端自监督框架，从无动作视频中学习世界模型和连续隐式动作表示
- 引入对抗性隐式正则化 (Adversarial Latent Regularization) 技术
- 结合扩散模型进行视频生成
- 无需动作标签即可学习可控的世界模型

## Method

- **连续隐式动作**: 在潜在空间中学习连续的动作表示，无需显式动作标签
- **对抗性正则化**: 使用对抗训练确保隐式动作空间的结构化和可控性
- **扩散模型骨干**: 基于扩散模型的视频生成架构
- **端到端自监督**: 从原始视频数据直接训练，无需额外标注

## Results

- 成功从无动作视频中学习到有意义的隐式动作表示
- 生成的视频具有良好的可控性和一致性
- 在多个基准上展示了与有监督方法可比的性能

## Related Work

- 与 LAPO 等隐式动作学习方法的区别在于使用对抗正则化
- 与 Video Diffusion Models 的结合是本文的特色
- 与 World Models (Ha & Schmidhuber 2018) 在隐式空间建模上的传承关系

## Notes

- 从无动作视频学习是重要的自监督方向，降低了数据收集成本
- 对抗性正则化是保证隐式空间质量的关键技术
- 关注隐式动作与真实动作之间的对应关系
- 扩散模型在世界模型中的应用越来越普遍，本文提供了新的范式
