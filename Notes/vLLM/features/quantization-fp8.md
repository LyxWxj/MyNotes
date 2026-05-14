---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm
---

# FP8 W8A8

## 概述

FP8 量化使用 8-bit 浮点数（E4M3/E5M2）进行权重和激活量化，实现 2x 内存减少和最高 1.6x 吞吐提升。

## 硬件要求

- **W8A8**：Hopper (SM 9.0)、Ada Lovelace (SM 8.9)
- **W8A16**：Turing (SM 7.5) 及以上（使用 Marlin kernel）

## FP8 格式

- **E4M3**：4 位指数 + 3 位尾数，范围 ±448
- **E5M2**：5 位指数 + 2 位尾数，范围 ±57344（更高动态范围，更低精度）

## 量化流程

使用 LLM Compressor：

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])
oneshot(model=model, recipe=recipe)
model.save_pretrained(save_dir)
```

**FP8_DYNAMIC 方案**：权重静态 per-channel 量化，激活动态 per-token 量化，无需校准数据。

## 在线动态量化

无需预量化，直接运行时量化：

```python
llm = LLM("facebook/opt-125m", quantization="fp8")
```

所有 Linear 层（lm_head 除外）量化为 FP8_E4M3，激活动态 per-tensor 缩放。

## 精度评估

```bash
lm_eval --model vllm --model_args pretrained=<path>,add_bos_token=True --tasks gsm8k --num_fewshot 5
```
