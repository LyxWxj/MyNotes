---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm
_organized: true
---

# N-Gram Speculation

## 概述

N-Gram 推测解码通过在 prompt 中匹配 n-gram 生成候选 token，无需额外 draft 模型，是最轻量的推测方法。

## 使用方式

```python
llm = LLM(
    model="Qwen/Qwen3-8B",
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 5,
        "prompt_lookup_max": 4,
    },
)
```

## 关键参数

- `method`: `"ngram"`
- `num_speculative_tokens`: 推测 token 数
- `prompt_lookup_max`: n-gram 最大匹配长度

## 特点

- 无需额外模型，内存开销最小
- 适用于有重复模式的输入（如代码补全、模板填充）
- 延迟提升相对有限
