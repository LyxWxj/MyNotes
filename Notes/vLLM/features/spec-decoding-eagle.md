---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm
---

# EAGLE Draft Models

## 概述

EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency) 是基于 draft 模型的推测解码方法，延迟减少效果最佳。

## 使用方式

### EAGLE

```python
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    tensor_parallel_size=4,
    speculative_config={
        "model": "yuhuili/EAGLE-LLaMA3-Instruct-8B",
        "draft_tensor_parallel_size": 1,
        "num_speculative_tokens": 2,
        "method": "eagle",
    },
)
```

### EAGLE3

```python
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    tensor_parallel_size=2,
    speculative_config={
        "model": "RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3",
        "draft_tensor_parallel_size": 2,
        "num_speculative_tokens": 2,
        "method": "eagle3",
    },
)
```

## 预训练 Draft 模型

- [RedHatAI/speculator-models](https://huggingface.co/collections/RedHatAI/speculator-models)
- [yuhuili/models](https://huggingface.co/yuhuili/models?search=eagle)

## 关键参数

- `method`: `"eagle"` 或 `"eagle3"`
- `num_speculative_tokens`: 推测 token 数
- `draft_tensor_parallel_size`: draft 模型的 TP 大小
