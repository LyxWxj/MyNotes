---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm
---

# PARD (Parallel Draft Model)

## 概述

PARD 是并行 draft 模型推测解码方法，通过并行生成多个候选 token 降低 draft 延迟。

## 使用方式

### 离线推理

```python
llm = LLM(
    model="Qwen/Qwen3-8B",
    speculative_config={
        "model": "amd/PARD-Qwen3-0.6B",
        "num_speculative_tokens": 12,
        "method": "draft_model",
        "parallel_drafting": True,
    },
)
```

### 在线服务

```bash
vllm serve Qwen/Qwen3-4B \
    --speculative_config '{"model": "amd/PARD-Qwen3-0.6B", "num_speculative_tokens": 12, "method": "draft_model", "parallel_drafting": true}'
```

## 关键参数

- `parallel_drafting: True`：启用并行 draft
- `num_speculative_tokens`: 推测 token 数（可设较高值如 12）

## 预训练模型

- [amd/pard](https://huggingface.co/collections/amd/pard)
