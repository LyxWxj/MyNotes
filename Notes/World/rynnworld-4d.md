---
type: Note
related_to: "[[world-model]]"
status: Active
url: https://arxiv.org/abs/2607.06559
tags:
  - world-model
  - embodied-ai
  - 4D
  - diffusion
  - robotic-manipulation
---

# RynnWorld-4D: 4D Embodied World Models for Robotic Manipulation

## Meta Information

- **Authors**: Haoyu Zhao, Xingyue Zhao, Siteng Huang, Xin Li, Deli Zhao, Zhongyu Li
- **Date**: 2026-07
- **arXiv ID**: 2607.06559
- **DOI**: 10.48550/arXiv.2607.06559
- **URL**: https://arxiv.org/abs/2607.06559
- **Zotero Key**: PYDWH7VM

## Abstract

Introduces a generative model co-producing future RGB frames, depth maps, and optical flow within one unified diffusion process. Uses a tri-branch architecture with cross-modal attention. Trained on 254.4 million frames across egocentric human and robotic manipulation videos.

## Key Contributions

1. **4D世界模型**: 同时生成RGB帧、深度图和光流的统一生成模型
2. **三分支架构**: 使用交叉模态注意力的tri-branch架构
3. **统扩散过程**: 在单一扩散过程中联合生成多种模态
4. **大规模训练**: 在2.544亿帧的第一人称人类和机器人操作视频上训练

## Method

- 统一扩散过程(Unified Diffusion Process)
- 三分支架构(Tri-branch Architecture)
- 交叉模态注意力(Cross-modal Attention)
- 联合生成RGB、深度、光流

## Results

- 训练数据：254.4 million frames
- 数据来源：第一人称人类视频和机器人操作视频
- 同时生成RGB帧、深度图和光流

## Related Work

- 4D场景理解
- 机器人操作世界模型
- 扩散模型在视频生成中的应用

## Notes

- 4D表示(3D空间+时间)对于机器人操作非常重要
- 深度和光流信息有助于更好地理解3D场景
- 适用于具身AI和机器人操作任务

---

*Last updated: 2026-07-18*
