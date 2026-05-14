---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm
---

# Batch Invariance

## 概述

Batch Invariance（批不变性）确保模型输出不受 batch size 或 batch 内请求顺序的影响，即相同的 prompt 在不同批次配置下产生完全相同的输出。

## 关键特性

- **确定性输出**：无论 batch 中有多少其他请求，相同 prompt 的输出结果一致
- **硬件要求**：需要 **H100 或 B100** 及以上 GPU（利用硬件级别的确定性特性）
- **精度保证**：在硬件数值精度范围内实现完全确定性

## 重要性

### 调试与可复现性

- 消除因 batch size 变化导致的输出不确定性
- 便于调试和回归测试

### 公平性

- 保证单个请求不受其他并发请求影响
- 在生产环境中提供一致的服务质量

## 技术实现

- 利用 H100/B100 的硬件确定性特性
- 通过数值稳定的归约操作消除浮点累加顺序的影响
- 在 attention 计算中使用确定性的 softmax 和归约

## 使用方式

- 默认在支持的硬件上自动启用
- 无需额外配置

## 限制

- 仅支持 H100/B100 及以上 GPU
- 较早的 GPU 架构无法保证完全确定性
