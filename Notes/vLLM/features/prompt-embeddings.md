---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm
_organized: true
---

# Prompt Embedding Inputs

## 概述

Prompt Embeddings 允许直接向模型传入预计算的 embedding tensor，绕过 tokenizer 的文本→token→embedding 流程。

## 工作原理

传统流程：**文本 → Token IDs → Embedding（查表）**

Prompt Embeddings 跳过前两步，直接提供 embedding 向量，模型可以处理超出词表范围的自定义 embedding。

## 离线推理

```python
# prompt_embeds shape: (sequence_length, hidden_size)
outputs = llm.generate({
    "prompt_embeds": embedding_tensor,
})
```

支持从 HuggingFace Transformers 模型获取 embedding。

## 在线服务

- 启用方式：`vllm serve <model> --enable-prompt-embeds`
- 通过 Completions API 的 `prompt_embeds` 字段传入
- Embedding 需 base64 编码
- 可混合 `prompt_embeds` 和 `prompt`，embeds 排在前面

## 注意事项

- 错误的 embedding 形状会导致引擎崩溃
- 仅对可信用户启用此功能
