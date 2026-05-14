---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm
---

# Quantized KV Cache

## 概述

FP8 KV Cache 量化将 KV Cache 压缩到 FP8 格式，显著减少内存占用，支持更长上下文和更高吞吐。

## 量化策略

| 策略 | 说明 |
| --- | --- |
| Per-tensor | 每个 Q/K/V tensor 一个缩放因子 |
| Per-attention-head | 每个注意力头一个缩放因子（仅 Flash Attention） |

## 缩放因子校准

### 1. 无校准（默认）

所有缩放因子为 1.0：

```python
llm = LLM(model="...", kv_cache_dtype="fp8", calculate_kv_scales=False)
```

### 2. 随机 token 校准

预热时自动估计：

```python
llm = LLM(model="...", kv_cache_dtype="fp8", calculate_kv_scales=True)
```

### 3. 数据集校准（推荐）

使用 LLM Compressor 进行高质量校准：

```python
recipe = QuantizationModifier(
    config_groups={"attention": QuantizationScheme(targets=["LlamaAttention"], input_activations=fp8_args)},
    kv_cache_scheme=fp8_args,
)
oneshot(model=model, dataset=ds, recipe=recipe)
```

## dtype 选项

- `kv_cache_dtype="auto"`：使用模型默认类型
- `kv_cache_dtype="fp8_e4m3"`：CUDA 11.8+ 和 ROCm
- `kv_cache_dtype="fp8_e5m2"`：CUDA 11.8+

## 注意事项

- Flash Attention 3 + FP8 KV Cache 时，attention 也在 FP8 域计算
- Per-attention-head 量化仅支持 Flash Attention 后端
