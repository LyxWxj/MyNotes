---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm
---

# AutoAWQ

## 概述

AutoAWQ 是 INT4 权重量化工具，将 BF16/FP16 精度降低到 INT4，显著减少模型内存占用。

> ⚠️ AutoAWQ 已弃用，功能已迁移到 [LLM Compressor](https://github.com/vllm-project/llm-compressor/tree/main/examples/awq)。

## 量化流程

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}
model.quantize(tokenizer, quant_config=quant_config)
model.save_quantized(quant_path)
```

## vLLM 使用

```bash
python examples/offline_inference/llm_engine_example.py \
    --model TheBloke/Llama-2-7b-Chat-AWQ --quantization awq
```

```python
llm = LLM(model="TheBloke/Llama-2-7b-Chat-AWQ", quantization="AWQ")
```

## 关键参数

- `w_bit=4`：4-bit 权重
- `q_group_size=128`：分组大小
- `zero_point=True`：零点量化
- `version="GEMM"`：优化版本
