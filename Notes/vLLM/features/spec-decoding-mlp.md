---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm
---

# MLP Draft Models

## 概述

MLP Speculator 是基于多层感知器的 draft 模型，基于上下文向量和采样 token 进行预测。

## 使用方式

```python
llm = LLM(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    speculative_config={
        "model": "ibm-ai-platform/llama3-8b-accelerator",
        "draft_tensor_parallel_size": 1,
        "method": "mlp_speculator",
    },
)
```

## 预训练模型

| 模型 | 链接 |
| --- | --- |
| llama3-8b-accelerator | ibm-ai-platform/llama3-8b-accelerator |
| llama3-70b-accelerator | ibm-ai-platform/llama3-70b-accelerator |
| granite-8b-code-instruct-accelerator | ibm-granite/granite-8b-code-instruct-accelerator |

## 关键参数

- `method`: `"mlp_speculator"`
- `draft_tensor_parallel_size`: draft 模型 TP 大小

## 已知问题

- `llama3-70b-accelerator` 可能报错 `AttributeError: 'MLPSpeculatorConfig' object has no attribute 'num_attention_heads'`
