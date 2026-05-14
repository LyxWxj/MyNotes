---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm
---

# Speculators

## 概述

Speculators 是用于加速 LLM 推理的推测解码库，提供高效的 draft 模型训练，与 vLLM 无缝集成。

## 核心特性

- **离线训练数据生成**：使用 vLLM 生成 hidden states，保存到磁盘用于训练
- **Draft 模型训练**：支持单层和多层 draft 模型的端到端训练，支持 MoE 模型
- **标准化格式**：HuggingFace 兼容格式，支持从外部研究仓库转换
- **vLLM 集成**：直接部署到 vLLM，低延迟生产级推理

## 为什么使用推测解码？

- **降低延迟**：交互式应用中 token 生成速度提升 2-3 倍
- **更好的 GPU 利用**：将延迟/内存瓶颈的解码转换为计算瓶颈的并行验证
- **无质量损失**：接受的 token 与目标模型完全一致
- **成本效率**：单 GPU 服务更多请求

## 资源

- [GitHub](https://github.com/vllm-project/speculators)
- [Examples](https://github.com/vllm-project/speculators/tree/main/examples)
