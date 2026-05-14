---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm
---

# AMD Quark

## 概述

Quark 是 AMD 的量化工具包，支持权重、激活和 KV Cache 量化，以及 AWQ、GPTQ、Rotation、SmoothQuant 等算法。

## 支持的格式

- FP8 per-tensor
- MXFP4、MXFP6（OCP 规范）
- 混合精度：MXFP4 + FP8

## 量化流程

### 1. 安装

```bash
pip install amd-quark
```

### 2. 量化配置

```python
from quark.torch.quantization import Config, QuantizationConfig, FP8E4M3PerTensorSpec

FP8_SPEC = FP8E4M3PerTensorSpec(observer_method="min_max", is_dynamic=False).to_quantization_spec()
global_config = QuantizationConfig(input_tensors=FP8_SPEC, weight=FP8_SPEC)
```

### 3. 量化与导出

```python
from quark.torch import ModelQuantizer, ModelExporter

quantizer = ModelQuantizer(quant_config)
quant_model = quantizer.quantize_model(model, calib_dataloader)
freezed_model = quantizer.freeze(model)
exporter = ModelExporter(config=export_config, export_dir=EXPORT_DIR)
exporter.export_safetensors_model(freezed_model, quant_config=quant_config, tokenizer=tokenizer)
```

### 4. vLLM 使用

```python
llm = LLM(model="<path>", kv_cache_dtype="fp8", quantization="quark")
```

## CLI 量化脚本

```bash
python3 quantize_quark.py --model_dir meta-llama/Llama-2-70b-chat-hf \
    --output_dir /path/to/output --quant_scheme w_fp8_a_fp8 \
    --kv_cache_dtype fp8 --quant_algo autosmoothquant --model_export hf_format
```

## MXFP4/MXFP6

支持 OCP Microscaling 格式，在不支持原生 MX 操作的设备上使用模拟 kernel。

## 混合精度

支持 per-layer 混合精度量化（MXFP4 + FP8），平衡精度和吞吐。
