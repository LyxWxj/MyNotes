---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm
---

# GGUF

## 概述

GGUF 是通用量化格式，支持多种精度，在 vLLM 中仍为实验性支持。

> ⚠️ 实验性功能，可能与其他特性不兼容。仅支持单文件 GGUF 模型。

## 使用方式

### 从 HuggingFace 加载

```bash
vllm serve unsloth/Qwen3-0.6B-GGUF:Q4_K_M --tokenizer Qwen/Qwen3-0.6B
```

格式：`repo_id:quant_type`

### 本地文件加载

```bash
vllm serve ./Qwen3-0.6B-Q4_K_M.gguf --tokenizer Qwen/Qwen3-0.6B
```

### Tensor Parallelism

```bash
vllm serve unsloth/Qwen3-0.6B-GGUF:Q4_K_M --tokenizer Qwen/Qwen3-0.6B --tensor-parallel-size 2
```

### 离线推理

```python
llm = LLM(model="unsloth/Qwen3-0.6B-GGUF:Q4_K_M", tokenizer="Qwen/Qwen3-0.6B")
```

## 注意事项

- 推荐使用基础模型的 tokenizer，GGUF 的 tokenizer 转换不稳定
- 多文件 GGUF 需用 `gguf-split` 工具合并
- HuggingFace 不支持的模型需手动提供 `--hf-config-path`
