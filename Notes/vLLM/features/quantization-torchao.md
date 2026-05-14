---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm
---

# TorchAO

## 概述

TorchAO 是 PyTorch 的架构优化库，提供高性能数据类型、优化技术和 kernel，支持 torch.compile、FSDP 等 PyTorch 原生特性。

## 安装

```bash
pip install --pre torchao>=10.0.0 --index-url https://download.pytorch.org/whl/nightly/cu126
```

## 量化 HuggingFace 模型

```python
from transformers import TorchAoConfig, AutoModelForCausalLM
from torchao.quantization import Int8WeightOnlyConfig

quantization_config = TorchAoConfig(Int8WeightOnlyConfig())
model = AutoModelForCausalLM.from_pretrained(
    model_name, dtype="auto", device_map="auto", quantization_config=quantization_config
)
```

支持推送到 HuggingFace Hub 保存。

## 资源

- [TorchAO Quantization Space](https://huggingface.co/spaces/medekk/TorchAO_Quantization)：UI 量化工具
- [Benchmarks](https://github.com/pytorch/ao/tree/main/torchao/quantization#benchmarks)
