---
type: Note
related_to: "[[world-model]]"
status: Active
url: https://arxiv.org/abs/2606.12072
zotero_key: JG9QIZWY
tags:
  - world-model
  - video-generation
  - reinforcement-learning
  - self-distillation
---

# World Model Self-Distillation: Training World Models to Solve General Tasks

## Meta

- **Authors**: Sebastian Stapf, Pablo Acuaviva Huertos, Aram Davtyan, Paolo Favaro
- **Date**: 2026-06
- **arXiv**: [2606.12072](https://arxiv.org/abs/2606.12072)
- **DOI**: 10.48550/arXiv.2606.12072

## Abstract

Combines self-distillation with reinforcement learning to elicit task-solving ability in pretrained video generators. A VLM generates tasks and solutions; a Demonstrator video diffusion model is distilled into an Executor conditioned only on image and task prompt.

## Key Contributions

- 提出自蒸馏方法，将预训练视频生成器转变为任务求解器
- 结合强化学习和自蒸馏，无需额外标注数据
- 设计 VLM 生成任务和解决方案的自动化流程
- Demonstrator-Executor 架构实现从条件生成到无条件任务求解的转变

## Method

- **自蒸馏框架**: Demonstrator 模型（条件生成）蒸馏为 Executor 模型（仅基于图像和任务提示）
- **VLM 任务生成**: 使用视觉语言模型自动生成训练任务和对应解决方案
- **强化学习优化**: 通过 RL 进一步优化 Executor 的任务求解能力
- **视频扩散模型**: 基于扩散模型的视频生成骨干网络

## Results

- 成功将预训练视频生成器转化为通用任务求解器
- Executor 模型无需视频条件即可完成任务
- 在多种任务上展示了良好的泛化能力

## Related Work

- 与 UniSim 等视频生成世界模型的区别在于引入了任务求解能力
- 与 RL-based 世界模型 (Dreamer 系列) 的对比，本文使用扩散模型
- 自蒸馏方法借鉴了 LLM 领域的 self-play 和 self-improvement 思路

## Notes

- 自蒸馏是将视频生成模型"激活"为智能体的有效途径
- VLM 生成任务的思路降低了对人工标注的依赖
- 关注 Demonstrator 和 Executor 之间的能力差距
- 与 Decision Diffuser 等方法的异同值得深入分析
