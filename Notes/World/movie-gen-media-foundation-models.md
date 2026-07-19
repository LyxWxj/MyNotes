---
type: Note
related_to: "[[world-model]]"
status: Active
url: https://arxiv.org/abs/2410.13720
tags:
  - world-model
  - video-generation
  - audio-generation
  - media-foundation-model
  - meta
---

# Movie Gen: A Cast of Media Foundation Models

## 元信息

| 字段 | 内容 |
|------|------|
| **作者** | Adam Polyak, Amit Zohar, Andrew Brown, ... (88 位研究者，Meta) |
| **机构** | Meta / FAIR |
| **日期** | 2024-10-17 (v1), 2025-02-26 (v2) |
| **arXiv ID** | 2410.13720 |
| **URL** | https://arxiv.org/abs/2410.13720 |
| **视频展示** | https://go.fb.me/MovieGenResearchVideos |

## 摘要

Movie Gen 是 Meta 推出的一系列媒体基础模型，能够生成高质量 1080p HD 视频（不同宽高比）和同步音频。额外能力包括精确的指令式视频编辑和基于用户图像的个性化视频生成。在 text-to-video、video personalization、video editing、video-to-audio、text-to-audio 五个任务上均达到 SOTA。最大视频生成模型为 30B 参数 Transformer，最大上下文长度 73K video tokens，生成 16 秒 16fps 视频。

## 核心贡献

1. **多任务媒体基础模型家族**：统一覆盖视频生成、音频生成、视频编辑、视频个性化等多个任务
2. **30B 参数规模**：迄今报告的最大视频生成 Transformer 模型之一
3. **架构创新**：latent 空间、训练目标、数据策划、评估协议、并行技术和推理优化的多项简化
4. **Scaling 见证**：展示了预训练数据、模型规模和训练计算量扩展带来的收益
5. **同步音视频生成**：同时生成高质量视频和匹配的音频

## 方法

- **模型架构**：基于 Transformer 的视频/音频生成模型
- **Latent 空间**：在压缩的 latent 表示中进行生成
- **30B 参数视频模型**：73K video tokens 上下文，16 秒 16fps
- **多模态**：同时处理视频和音频模态
- **指令式编辑**：支持自然语言指令驱动的精确视频编辑
- **个性化**：基于单张用户图像生成个性化视频
- **并行技术**：大规模训练的并行化策略

## 实验结果

- 在 text-to-video synthesis、video personalization、video editing、video-to-audio、text-to-audio 五个任务上均达到 SOTA
- 生成 1080p HD 视频，支持多种宽高比
- 16 秒 16fps 的视频生成能力

## 与其他工作的关系

- **与 Sora 的竞争**：Movie Gen 是 Meta 对标 OpenAI Sora 的工作
- **与 Cosmos 的区别**：Movie Gen 专注媒体生成（视频+音频），Cosmos 专注物理 AI 的世界模拟
- **与 Causal-rCM 的关系**：Causal-rCM 论文提到将方法应用到 Cosmos 3 的视频生成骨干
- **Scaling 视频生成**：展示了将 LLM scaling 思想迁移到视频生成的可行性

## 笔记

- 30B 参数的视频生成模型，规模惊人——视频生成正在走 LLM 的 scaling 路线
- 多任务统一（视频生成+音频生成+编辑+个性化）是未来趋势
- "cast of models" 的表述很有趣——一个模型家族而非单一模型
- 73K video tokens 的上下文长度意味着对高效注意力机制的需求
- Meta 的开放态度值得肯定——公开了大量技术细节
- 关键问题：如此大模型的推理成本如何？是否适合实时应用？
