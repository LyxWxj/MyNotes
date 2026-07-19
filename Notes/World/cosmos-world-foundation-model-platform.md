---
type: Note
related_to: "[[world-model]]"
status: Active
url: https://arxiv.org/abs/2501.03575
tags:
  - world-model
  - cosmos
  - nvidia
  - physical-ai
  - world-foundation-model
---

# Cosmos World Foundation Model Platform for Physical AI

## 元信息

| 字段 | 内容 |
|------|------|
| **作者** | NVIDIA (78 位研究者) |
| **机构** | NVIDIA |
| **日期** | 2025-01-07 |
| **arXiv ID** | 2501.03575 |
| **URL** | https://arxiv.org/abs/2501.03575 |
| **代码** | https://github.com/nvidia-cosmos/cosmos-predict1 |
| **许可证** | 宽松许可 |

## 摘要

Physical AI 需要先在数字世界中训练——需要自身的数字孪生（策略模型）和世界的数字孪生（世界模型）。Cosmos 是一个世界基础模型平台，帮助开发者为其 Physical AI 设置构建定制化世界模型。将世界基础模型定位为通用世界模型，可微调为下游应用的定制世界模型。平台涵盖视频策划流水线、预训练世界基础模型、后训练示例和视频分词器。

## 核心贡献

1. **世界基础模型定位**：提出"世界基础模型"概念——通用世界模型可微调为下游定制模型
2. **完整平台**：涵盖数据策划、预训练、后训练、分词器的完整 pipeline
3. **视频策划流水线**：大规模视频数据的筛选和处理
4. **预训练世界基础模型**：可直接使用或微调的基础模型
5. **视频分词器**：将视频编码为离散 token 的工具
6. **开源开放**：模型权重和代码均开放，宽松许可

## 方法

- **世界基础模型 (WFM)**：在大规模视频数据上预训练的通用世界模型
- **后训练 (Post-training)**：将 WFM 微调为特定应用的定制世界模型
- **视频分词器**：将连续视频信号转换为离散 token 序列
- **视频策划流水线**：自动化的大规模视频数据筛选
- **Physical AI 范式**：策略模型（数字孪生自身）+ 世界模型（数字孪生世界）

## 实验结果

- 提供了多个后训练的示例，展示如何将基础模型定制为下游应用
- 世界基础模型在多种场景下展示了良好的泛化能力

## 与其他工作的关系

- **与 Cosmos 3 的关系**：Cosmos v1 是 Cosmos 系列的第一代，Cosmos 3 在此基础上扩展到全模态
- **与 Sora 的区别**：Sora 是视频生成工具，Cosmos 定位为 Physical AI 的世界模型平台
- **与 Movie Gen 的区别**：Movie Gen 专注媒体生成，Cosmos 面向机器人和自动驾驶等 Physical AI
- **与 Genie/UniSim 的关系**：同属世界模型方向，但 Cosmos 强调平台化和可定制性

## 笔记

- "世界基础模型"的定位很清晰——像 LLM 是语言任务的基础模型一样，WFM 是世界模拟的基础模型
- 完整的平台思维：不只是模型，还有数据策划、分词器、后训练工具链
- "Physical AI 需要数字孪生"的论述很有说服力——策略模型 + 世界模型的双孪生架构
- Cosmos v1 到 Cosmos 3 的演进展示了 NVIDIA 在世界模型方向的持续投入
- 开源策略（宽松许可）有助于社区采用和生态建设
- 关键观察：NVIDIA 从芯片公司转型为 AI 平台公司，Cosmos 是其 Physical AI 战略的核心
