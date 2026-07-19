---
type: Note
related_to: "[[world-model]]"
status: Active
url: https://arxiv.org/abs/2606.10135
zotero_key: 6DEBN5EP
tags:
  - world-model
  - open-source
  - autoregressive
  - bidirectional
  - DMD
---

# BiWM: Advancing Open-Source Interactive Video World Models with Bidirectional Autoregression

## Meta

- **Authors**: Shaohao Rui, Xiaofeng Mao, Zhanyu Zhang, Peijia Lin, Yansong Zhu, Yibo Zhang, Haibin Wan, Weijie Ma
- **Date**: 2026-06
- **arXiv**: [2606.10135](https://arxiv.org/abs/2606.10135)
- **DOI**: 10.48550/arXiv.2606.10135

## Abstract

First full-stack framework for interactive video world models under the bidirectional autoregressive paradigm. Uses few-step Distribution Matching Distillation (DMD) stage. Supports Wan2.1-1.3B through LTX-2.3-22B.

## Key Contributions

- 提出首个基于双向自回归范式的全栈交互式视频世界模型框架
- 引入少步 Distribution Matching Distillation (DMD) 阶段，提升生成效率
- 支持从 Wan2.1-1.3B 到 LTX-2.3-22B 的多种模型规模，具备良好的可扩展性
- 开源实现，推动开放社区的交互式视频世界模型发展

## Method

- **双向自回归范式 (Bidirectional Autoregression)**: 结合前向和后向自回归生成，使模型能够从两个方向理解视频序列的因果关系
- **Distribution Matching Distillation (DMD)**: 通过少步蒸馏技术加速推理，减少生成所需的采样步骤
- **全栈框架设计**: 从数据处理、模型训练到推理部署的完整流水线

## Results

- 支持多种规模的视频生成模型 (1.3B - 22B 参数)
- DMD 蒸馏显著减少推理步数，提升交互式应用的实时性
- 在交互式视频生成质量上取得竞争性表现

## Related Work

- 与单向自回归视频生成模型 (如 VideoGPT) 的区别在于引入双向信息流
- DMD 蒸馏方法借鉴了图像生成领域的知识蒸馏技术
- 开源定位使其与闭源商业世界模型形成互补

## Notes

- 双向自回归是一个有趣的方向，结合了自回归模型的序列建模能力和双向上下文的理解能力
- DMD 蒸馏对于交互式应用至关重要，关注其在实际部署中的延迟表现
- 开源策略有助于学术界复现和改进，值得跟踪后续社区反馈
