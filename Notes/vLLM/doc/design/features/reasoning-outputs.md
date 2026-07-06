---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm
_organized: true
---

# Reasoning Outputs

## 概述

vLLM 支持推理模型（如 DeepSeek R1、Qwen3 等），这些模型在输出中包含推理步骤（reasoning）和最终结论（content）两个字段。

## 支持的模型

| 模型系列 | Parser | 结构化输出 | Tool Calling |
| --- | --- | --- | --- |
| DeepSeek R1 | `deepseek_r1` | json, regex | ❌ |
| DeepSeek-V3.1 | `deepseek_v3` | json, regex | ❌ |
| ERNIE-4.5-VL | `ernie45` | json, regex | ❌ |
| ERNIE-4.5-Thinking | `ernie45` | json, regex | ✅ |
| GLM-4.5 | `glm45` | json, regex | ✅ |
| Qwen3 | `qwen3` | json, regex | ✅ |
| QwQ-32B | `deepseek_r1` | json, regex | ✅ |

## 快速开始

```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --reasoning-parser deepseek_r1
```

```python
response = client.chat.completions.create(model=model, messages=messages)
reasoning = response.choices[0].message.reasoning  # 推理步骤
content = response.choices[0].message.content        # 最终结论
```

## 流式输出

- `reasoning` 字段在 `delta` 中可用
- 需用 `hasattr` 检查属性是否存在

## Tool Calling

推理内容在 tool calling 启用时也可用。Tool calling 仅从 `content` 字段解析函数调用，不从 `reasoning` 中解析。

## 服务级默认配置

```bash
vllm serve Qwen/Qwen3-8B \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": false}'
```

请求级 `chat_template_kwargs` 优先于服务端默认值。

## Thinking Budget Control

限制推理 token 数量：

```python
response = client.chat.completions.create(
    model=model, messages=messages,
    extra_body={"thinking_token_budget": 10}
)
```

- `--reasoning-config` 定义推理边界 token（`reasoning_start_str`、`reasoning_end_str`）
- 支持自定义结束短语使推理终止更自然

## 限制

- `reasoning` 字段仅在 chat completion 端点可用
