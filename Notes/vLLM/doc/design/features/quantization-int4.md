---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm
_organized: true
---

# INT4 W4A16

## 概述

INT4 W4A16 量化将权重压缩到 4-bit 整数，激活保持 FP16，适用于低 QPS 场景下的内存节省和延迟优化。

## 硬件要求

- NVIDIA GPU compute capability > 8.0（Ampere、Ada、Hopper、Blackwell）

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

recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=2048, num_calibration_samples=512)
model.save_pretrained(save_dir, save_compressed=True)
```

### 4. 评估

```bash
lm_eval --model vllm --model_args pretrained=<path>,add_bos_token=true --tasks gsm8k
```

## 最佳实践

- 校准数据 512 样本起步，精度下降时增加
- 序列长度 2048 起步
- 关键超参数：`dampening_frac`（GPTQ 影响力）、`actorder="weight"`（激活排序，可提升精度）
