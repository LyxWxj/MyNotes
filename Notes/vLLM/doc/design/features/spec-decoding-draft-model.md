---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm
_organized: true
---

# Draft Model

## 概述

Draft Model 是经典的推测解码方法，使用独立的小型 draft 模型快速生成候选 token，由目标模型并行验证。

## 使用方式

### 离线推理

```python
llm = LLM(
    model="Qwen/Qwen3-8B",
    speculative_config={
        "model": "Qwen/Qwen3-0.6B",
        "num_speculative_tokens": 5,
        "method": "draft_model",
    },
)
```

### 在线服务

```bash
vllm serve Qwen/Qwen3-4B-Thinking-2507 \
    --speculative_config '{"model": "Qwen/Qwen3-0.6B", "num_speculative_tokens": 5, "method": "draft_model"}'
```

## 关键参数

- `model`: draft 模型路径
- `num_speculative_tokens`: 推测 token 数
- `method`: `"draft_model"`

## 注意事项

- 使用 `--speculative_config` 设置所有推测解码配置
- 旧的 `--speculative_model` 和单独参数已弃用
- 客户端请求代码无需改变
