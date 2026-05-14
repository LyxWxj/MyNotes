---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm
---

# Intel AutoRound

## 概述

AutoRound 是 Intel 的高级量化算法，支持 INT2、INT3、INT4、INT8、MXFP8、MXFP4、NVFP4 和 GGUF 格式。

## 特点

- 2-3 bit 超低精度量化仍有良好精度
- 快速混合 bits/dtypes 方案生成
- 支持导出 AutoRound、AutoAWQ、AutoGPTQ、GGUF 格式
- 支持 10+ 视觉语言模型
- Per-layer 混合位量化
- RTN 模式快速量化

## 支持的方案（Intel 平台）

- `W4A16`：权重 4-bit，激活 16-bit
- `W8A16`：权重 8-bit，激活 16-bit

## 量化流程

### CLI

```bash
auto-round --model Qwen/Qwen3-0.6B --scheme W4A16 --format auto_round --output_dir ./tmp_autoround
```

### Python API

```python
from auto_round import AutoRound
autoround = AutoRound(model_name, scheme="W4A16")
autoround.quantize_and_save(output_dir, format="auto_round")
```

## vLLM 部署

```bash
vllm serve Intel/DeepSeek-R1-0528-Qwen3-8B-int4-AutoRound --gpu-memory-utilization 0.8 --max-model-len 4096
```

> Intel GPU/CPU 部署 `wNa16` 模型需添加 `--enforce-eager`。
