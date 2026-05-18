---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm
_organized: true
---

# MTP (Multi-Token Prediction)

## 概述

MTP 利用目标模型原生的多 token 预测能力进行推测解码，无需额外 draft 模型。

## 使用方式

### 离线推理

```python
llm = LLM(
    model="XiaomiMiMo/MiMo-7B-Base",
    speculative_config={"method": "mtp", "num_speculative_tokens": 1},
)
```

### 在线服务

```bash
vllm serve XiaomiMiMo/MiMo-7B-Base \
    --speculative_config '{"method":"mtp","num_speculative_tokens":1}'
```

## 注意事项

- 仅支持原生 MTP 的模型系列
- `num_speculative_tokens` 控制推测深度，建议从 1 开始
- 不支持 MTP 的模型需使用 EAGLE 或 draft model
