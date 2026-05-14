---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm
---

# Suffix Decoding

## 概述

Suffix Decoding 通过模式匹配生成候选 token，与 n-gram 类似但更强大：可匹配 prompt 和先前生成内容，使用频率计数选择最可能的延续，自适应推测深度。

## 适用场景

- 代码编辑
- Agent 循环（自我反思、自一致性）
- RL rollouts
- 高重复性任务

## 安装

```bash
pip install arctic-inference
```

## 使用方式

```python
llm = LLM(
    model="Qwen/Qwen3-8B",
    speculative_config={
        "method": "suffix",
        "num_speculative_tokens": 32,  # 最大推测 token 数
    },
)
```

## 关键特性

- 动态推测深度：每个请求每步自适应调整
- `num_speculative_tokens` 指定最大值（建议 16 或 32）
- 无需额外 draft 模型
- 同时匹配 prompt 和先前生成内容
