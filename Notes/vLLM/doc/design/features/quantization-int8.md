---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm
_organized: true
---

# INT8 W8A8

## 概述

INT8 W8A8 量化将权重和激活均压缩到 8-bit 整数，兼顾内存节省和推理性能。

## 硬件要求

- NVIDIA GPU compute capability > 7.5（Turing、Ampere、Ada、Hopper）
- ⚠️ Blackwell (SM ≥ 10.0) 不支持，需使用 FP8

## 量化流程

### 1. 加载模型

```python
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", dtype="auto")
```

### 2. 准备校准数据

使用 `ultrachat` 等数据集，512 样本，序列长度 2048。

### 3. 应用量化

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
]
oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=2048, num_calibration_samples=512)
model.save_pretrained(save_dir, save_compressed=True)
```

### 4. 评估

```bash
lm_eval --model vllm --model_args pretrained=<path>,add_bos_token=true --tasks gsm8k
```

## 最佳实践

- 校准数据 512 样本起步
- 使用 SmoothQuant + GPTQ 组合提升精度
- 量化模型对 bos token 敏感，评估时需 `add_bos_token=True`
