---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm
_organized: true
---

# Interleaved Thinking

## 概述

Interleaved Thinking（交错思考）允许模型在工具调用之间进行推理，在收到工具结果后进行更精细的决策。模型可以在多个工具调用之间链式推理，基于中间结果做出更准确的判断。

## 核心能力

- **工具结果推理**：在工具调用后推理结果，再决定下一步
- **链式工具调用**：多个工具调用间穿插推理步骤
- **精细决策**：基于中间结果做出更 nuanced 的判断
- **透明推理**：展示工具选择的推理过程

## 支持的模型

| 模型系列 | Reasoning Parser |
| --- | --- |
| moonshotai/Kimi-K2-Thinking | `kimi_k2` |
| MiniMaxAI/MiniMax-M2 | `minimax_m2` |

## 使用方式

需要同时启用 tool calling 和 reasoning parser：

```bash
vllm serve MiniMaxAI/MiniMax-M2 \
  --tensor-parallel-size 4 \
  --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2 \
  --enable-auto-tool-choice
```

### 请求流程

1. 发送带 tools 的请求
2. 模型返回 tool_calls 和 reasoning
3. 执行工具，将结果和 reasoning 追加到 messages
4. 再次请求，模型基于推理和工具结果生成最终响应

```python
messages.append({
    "role": "assistant",
    "tool_calls": response.choices[0].message.tool_calls,
    "reasoning": response.choices[0].message.reasoning,  # 追加推理
})
```

## 注意事项

- **增加 token 消耗**：交错思考会增加 token 使用量
- **增加延迟**：推理步骤会增加响应时间
- 需要根据预算和性能需求权衡使用
