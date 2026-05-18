---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm
_organized: true
---

# Tool Calling

## 概述

vLLM 支持函数调用（Tool Calling），包括 named、`auto`、`required`、`none` 四种 `tool_choice` 模式。

## 快速开始

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json \
    --chat-template examples/tool_chat_template_llama3.1_json.jinja
```

## Tool Choice 模式

| 模式 | Schema 约束 | 说明 |
| --- | --- | --- |
| Named function | ✅ | 参数保证符合 JSON Schema |
| `required` | ✅ | 保证生成至少一个 tool call |
| `auto` | ❌ | 模型自由生成，parser 提取 |
| `none` | N/A | 不生成 tool call |

## 支持的模型 Parser

| Parser | 模型 |
| --- | --- |
| `hermes` | Hermes 2 Pro/3, Qwen2.5 |
| `mistral` | Mistral 7B Instruct |
| `llama3_json` | Llama 3.1/3.2/4 |
| `llama4_pythonic` | Llama 4 |
| `pythonic` | Llama 3.2, ToolACE |
| `granite4` / `granite` | IBM Granite |
| `internlm` | InternLM 2.5 |
| `jamba` | Jamba 1.5 |
| `xlam` | Salesforce xLAM |
| `deepseek_v3` | DeepSeek-V3 |
| `deepseek_v31` | DeepSeek-V3.1 |
| `minimax_m1` | MiniMax-M1 |
| `kimi_k2` | Kimi-K2 |
| `hunyuan_a13b` | Hunyuan A13B |
| `glm45` / `glm47` | GLM-4.5/4.7 |
| `qwen3_xml` | Qwen3-Coder |
| `functiongemma` | FunctionGemma 270M |
| `olmo3` | OLMo 3 |
| `openai` | GPT-OSS |
| `gigachat3` | GigaChat 3 |
| `longcat` | LongCat-Flash-Chat |

## 自定义 Parser

实现 `ToolParser` 基类，注册到 `ToolParserManager`：

```python
class MyParser(ToolParser):
    def extract_tool_calls(self, model_output, request):
        return ExtractedToolCallInformation(...)
    def extract_tool_calls_streaming(self, ...):
        return delta
```

通过 `--tool-parser-plugin <path>` 和 `--tool-call-parser <name>` 使用。

## 注意事项

- `auto` 模式下参数可能不符合 Schema
- vLLM 不实现 OpenAI 的 `strict` 模式
- Named function 首次使用有 FSM 编译延迟
- 需要配合合适的 chat template 使用
