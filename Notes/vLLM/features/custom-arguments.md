---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm
---

# Custom Arguments

## 概述

vLLM 允许用户通过命令行自定义模型和服务参数，灵活配置推理引擎的行为。

## 主要配置类别

### 模型参数

- `--model`：模型名称或路径
- `--tokenizer`：分词器路径（可选）
- `--dtype`：数据类型（auto, half, float16, bfloat16, float）
- `--max-model-len`：最大序列长度

### 服务参数

- `--host` / `--port`：服务地址和端口
- `--api-key`：API 密钥
- `--served-model-name`：对外暴露的模型名称

### 性能参数

- `--tensor-parallel-size`：张量并行度
- `--pipeline-parallel-size`：流水线并行度
- `--gpu-memory-utilization`：GPU 显存利用率
- `--max-num-seqs`：最大并发序列数
- `--max-num-batched-tokens`：最大批处理 token 数

### 采样参数

- `--temperature`：采样温度
- `--top-p` / `--top-k`：采样策略
- `--max-num-seqs`：最大并发请求

## 自定义参数传递

### 通过 API 请求

```python
extra_body={"custom_param": "value"}
```

### 通过环境变量

```bash
export VLLM_CUSTOM_PARAM=value
```

## 扩展机制

- 支持通过插件扩展自定义参数
- 可通过 entrypoints 注册自定义处理器
