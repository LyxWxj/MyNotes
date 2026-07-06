---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm
_organized: true
---

# LLM Compressor

## 概述

LLM Compressor 是 vLLM 官方推荐的模型优化库，支持 FP4、FP8、INT8、INT4 等多种量化格式。

## 核心优势

- **减少内存占用**：在更小的 GPU 上运行更大模型
- **降低推理成本**：单 GPU 服务更多并发用户
- **加速推理**：更小的数据类型减少内存带宽消耗

## 支持的量化算法

- AWQ、GPTQ、AutoRound、Round-to-Nearest
- QuIP、SpinQuant 风格变换
- KV Cache 和 Attention 量化

## 支持的量化方法

- FP8、INT8、INT4、NVFP4、MXFP4
- 混合精度量化

## 关键特性

- **One-Shot 量化**：最少校准数据快速量化
- **vLLM 集成**：使用 compressed-tensors 格式无缝部署
- **HuggingFace 兼容**

## 资源

- [GitHub](https://github.com/vllm-project/llm-compressor)
- [Examples](https://github.com/vllm-project/llm-compressor/tree/main/examples)
