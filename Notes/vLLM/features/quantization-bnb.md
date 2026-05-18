---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm
_organized: true
---

# BitsAndBytes

## 概述

BitsAndBytes 支持 4-bit 和 8-bit 量化，无需校准数据即可在飞行中量化模型。

## 特点

- 无需校准数据
- 支持预量化 checkpoint 和飞行中量化
- 安装简单：`pip install bitsandbytes>=0.49.2`

## 使用方式

### 预量化模型

vLLM 自动从 config 推断量化方法：

```python
llm = LLM(model="unsloth/tinyllama-bnb-4bit", dtype=torch.bfloat16)
```

### 飞行中 4-bit 量化

```python
llm = LLM(model="huggyllama/llama-7b", quantization="bitsandbytes", dtype=torch.bfloat16)
```

### 在线服务

```bash
vllm serve <model> --quantization bitsandbytes
```
