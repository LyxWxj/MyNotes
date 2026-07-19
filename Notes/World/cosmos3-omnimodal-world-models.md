---
type: Note
related_to: "[[world-model]]"
status: Active
url: https://arxiv.org/abs/2606.02800
tags:
  - world-model
  - cosmos
  - nvidia
  - omnimodal
  - physical-ai
---

# Cosmos 3: Omnimodal World Models for Physical AI

## 元信息

| 字段 | 内容 |
|------|------|
| **作者** | NVIDIA (294 位研究者) |
| **机构** | NVIDIA |
| **日期** | 2026-06-01 (v1), 2026-06-23 (v4) |
| **arXiv ID** | 2606.02800 |
| **URL** | https://arxiv.org/abs/2606.02800 |
| **代码** | https://github.com/nvidia/cosmos |
| **HuggingFace** | https://huggingface.co/collections/nvidia/cosmos3 |
| **项目主页** | https://research.nvidia.com/labs/cosmos-lab/cosmos3 |
| **许可证** | Linux Foundation OpenMDW-1.1 |

## 摘要

Cosmos 3 是 NVIDIA 推出的全模态（omnimodal）世界模型家族，旨在统一处理和生成语言、图像、视频、音频和动作序列。基于统一的 Mixture-of-Transformers (MoT) 架构，支持高度灵活的输入-输出配置，将视觉-语言模型、视频生成器、世界模拟器和世界-动作模型整合到单一框架中。在理解与生成任务上建立新 SOTA，后训练模型被 Artificial Analysis 评为最佳开源 T2I 和 I2V 模型，被 RoboArena 评为最佳策略模型。

## 核心贡献

1. **统一全模态架构**：单一 MoT 架构处理 5 种模态（语言、图像、视频、音频、动作）
2. **灵活输入输出配置**：将 VLM、视频生成器、世界模拟器、世界-动作模型统一到一个框架
3. **SOTA 性能**：在多种理解和生成任务上建立新基准
4. **Physical AI 专用**：专为具身智能体设计，桥接感知、生成和动作
5. **全面开源**：代码、模型检查点、合成数据集、评估基准均开放

## 方法

- **Mixture-of-Transformers (MoT)**：统一架构，不同模态共享或专用的 Transformer 模块
- **全模态处理**：语言、图像、视频、音频、动作序列的联合处理
- **灵活 I/O 配置**：支持任意模态组合的输入输出
- **后训练**：针对特定任务（T2I、I2V、策略学习）进行后训练优化
- **合成数据集**：策划大规模合成数据用于训练
- **评估基准**：配套专门的评估基准

## 实验结果

- Artificial Analysis 评为最佳开源 T2I 和 I2V 模型
- RoboArena 评为最佳策略模型
- 在多种理解和生成任务上达到 SOTA
- 作为通用骨干适用于具身智能体的多种任务

## 与其他工作的关系

- **与 Cosmos v1 的关系**：Cosmos 3 是 Cosmos 系列的第三代，从视频生成扩展到全模态
- **与 Movie Gen 的区别**：Movie Gen 专注媒体生成，Cosmos 3 统一了感知-生成-动作
- **与 Causal-rCM 的关系**：Causal-rCM 被应用于 Cosmos 3 的视频生成骨干
- **与 Inferix 的关系**：Inferix 支持 Cosmos 系列模型的推理
- **与 GPT-4o/Gemini 的区别**：Cosmos 3 显式包含动作模态，面向 Physical AI 而非纯语言/视觉

## 笔记

- 全模态（omnimodal）是世界模型的终极形态——语言+视觉+音频+动作的统一
- Mixture-of-Transformers 是处理多模态的关键架构创新
- 显式包含动作模态是 Cosmos 3 与一般 VLM 的关键区别——真正的"世界-动作模型"
- 294 位作者的团队规模反映了 NVIDIA 在 Physical AI 上的巨大投入
- 全面开源（OpenMDW-1.1 许可）对社区非常友好
- 关键问题：如此大的统一模型，推理效率如何？是否能实时运行？
