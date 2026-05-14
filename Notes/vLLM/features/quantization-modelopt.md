---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm
---

# NVIDIA Model Optimizer

## 概述

NVIDIA Model Optimizer 是针对 NVIDIA GPU 的模型优化库，支持 LLM、VLM 和扩散模型的 PTQ 和 QAT。

## 支持的 Checkpoint 格式

| 格式 | 说明 |
| --- | --- |
| FP8 | per-tensor 权重缩放 |
| FP8_PER_CHANNEL_PER_TOKEN | per-channel 权重 + 动态 per-token 激活 |
| FP8_PB_WO | 块级 FP8 权重-only (128×128) |
| NVFP4 | NVFP4 格式 |
| MXFP8 | MXFP8 格式 |

## 量化流程

```python
import modelopt.torch.quantization as mtq

config = mtq.FP8_DEFAULT_CFG
model = mtq.quantize(model, config, forward_loop)
```

导出：

```python
from modelopt.torch.export import export_hf_checkpoint
export_hf_checkpoint(model, export_dir)
```

## vLLM 使用

```python
llm = LLM(model="nvidia/Llama-3.1-8B-Instruct-FP8", quantization="modelopt")
```

```bash
vllm serve <path> --quantization modelopt
```

## 检测机制

vLLM 通过 `hf_quant_config.json` 检测 ModelOpt checkpoint。
