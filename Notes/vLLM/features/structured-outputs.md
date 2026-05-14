---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm
---

# Structured Outputs

## 概述

vLLM 支持结构化输出生成，使用 xgrammar 或 guidance 作为后端，确保模型输出符合指定的格式约束（JSON Schema、正则表达式、CFG 语法、选项列表）。

## 支持的约束类型

| 类型 | 说明 |
| --- | --- |
| `choice` | 输出必须是指定选项之一 |
| `regex` | 输出匹配正则表达式 |
| `json` | 输出符合 JSON Schema |
| `grammar` | 输出符合上下文无关文法（EBNF） |
| `structural_tag` | 在指定标签内按 JSON Schema 生成 |

## 在线服务

通过 `extra_body` 传入 `structured_outputs` 参数：

```python
# Choice
extra_body={"structured_outputs": {"choice": ["positive", "negative"]}}

# Regex
extra_body={"structured_outputs": {"regex": r"\w+@\w+\.com\n"}}

# JSON Schema (Pydantic)
response_format={"type": "json_schema", "json_schema": {"name": "...", "schema": schema}}

# Grammar (EBNF)
extra_body={"structured_outputs": {"grammar": ebnf_string}}
```

## 推理输出结合

结构化输出可与推理模型的 `reasoning` 字段结合使用。Qwen3 需要 `--structured-outputs-config.enable_in_reasoning=True`。

## 实验性自动解析

使用 `client.beta.chat.completions.parse()` 传入 Pydantic 模型，自动解析输出：

```python
completion = client.beta.chat.completions.parse(
    model=model, messages=messages, response_format=MyModel,
)
message.parsed  # 自动解析的 Pydantic 对象
```

## 离线推理

使用 `StructuredOutputsParams`：

```python
from vllm.sampling_params import StructuredOutputsParams
params = SamplingParams(structured_outputs=StructuredOutputsParams(choice=["Positive", "Negative"]))
```

## 后端配置

```bash
vllm serve <model> --structured-outputs-config.backend auto
```

默认 `auto` 自动选择后端，也可指定 `xgrammar` 或 `guidance`。
