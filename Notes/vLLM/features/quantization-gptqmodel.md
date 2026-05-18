---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm
_organized: true
---

# GPTQModel

## 概述

GPTQModel 是 ModelCloud.AI 开发的 GPTQ 量化工具，支持 4-bit 和 8-bit 量化，支持动态 per-module 量化。

## 特点

- 支持 Marlin 和 Machete 自定义 kernel，优化 Ampere/Hopper 性能
- 支持动态 per-module 量化，不同层可使用不同量化参数
- HuggingFace 上有 5000+ 预量化模型

## 量化流程

```python
from gptqmodel import GPTQModel, QuantizeConfig

quant_config = QuantizeConfig(bits=4, group_size=128)
model = GPTQModel.load(model_id, quant_config)
model.quantize(calibration_dataset, batch_size=2)
model.save(quant_path)
```

## vLLM 使用

```python
llm = LLM(model="ModelCloud/DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2")
```

## 关键参数

- `bits=4` 或 `bits=8`：量化位数
- `group_size=128`：分组大小
- `batch_size`：校准时批大小
