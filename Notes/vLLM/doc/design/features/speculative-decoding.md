---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm
_organized: true
---

# Speculative Decoding

## 概述

Speculative Decoding（推测解码）通过让小型 draft 模型快速生成候选 token，再由目标模型并行验证，减少中低 QPS 场景下的 inter-token 延迟。

## 支持的方法

| 方法 | 低 QPS | 高 QPS | 说明 |
| --- | --- | --- | --- |
| EAGLE | 高 | 中高 | 强大的通用 model-based 方法 |
| MTP | 高 | 中高 | 目标模型原生支持时最佳 |
| Draft Model | 高 | 中 | 需要单独的 draft 模型 |
| PARD | 高 | 中高 | 低 draft 延迟 |
| MLP | 中高 | 中 | 兼容 MLP speculator 时可用 |
| N-Gram | 低中 | 中 | 轻量，易启用 |
| Suffix | 低中 | 中 | 无额外 draft 模型，动态推测深度 |

## 无损保证

### 理论无损

推测解码在硬件数值精度范围内理论上无损，浮点误差可能导致轻微分布差异。

### 算法无损

vLLM 的实现经过算法验证：
- **Rejection Sampler 收敛**：确保采样结果与目标分布一致
- **Greedy 采样相等**：验证推测解码与非推测解码的贪心输出一致

### Logprob 稳定性

vLLM 不保证 logprob 跨运行稳定，相同 prompt 可能产生不同输出。

## 已知不兼容

- Pipeline Parallelism 不兼容（vllm≤0.15.0）
- Draft model 不支持（vllm≤0.10.0）

## 使用方式

```bash
# 离线推理
python examples/offline_inference/spec_decode.py

# 基准测试
# 参考 benchmarking/cli.md
```

## 各方法详情

### EAGLE

基于 EAGLE 模型的推测解码，延迟减少效果最佳。

### MTP (Multi-Token Prediction)

目标模型原生支持多 token 预测时使用。

### Draft Model

使用独立的小型 draft 模型生成候选 token。

### PARD (Parallel Draft Model)

并行 draft 模型，降低 draft 延迟。

### MLP

使用 MLP speculator，兼容时效果好。

### N-Gram

基于 n-gram 的轻量推测，无需额外模型。

### Suffix Decoding

动态推测深度，无需额外 draft 模型。

## 训练自定义 Draft 模型

参考 [vllm-project/speculators](https://github.com/vllm-project/speculators) 训练和集成自定义 draft 模型。
