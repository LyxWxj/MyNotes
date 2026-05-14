---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm
---

# NixlConnector

## 概述

NixlConnector 是 vLLM 分离式预填充的高性能 KV Cache 传输 Connector，使用 NIXL 库实现完全异步的 send/recv 操作。

## 前置条件

```bash
# NVIDIA 平台
uv pip install nixl

# ROCm 平台：使用 RIXL，已包含在 ROCm Docker 中
# 非 CUDA 平台：从源码构建
python tools/install_nixl_from_source_ubuntu.py
```

### 传输配置

NIXL 使用 UCX 作为默认传输库：

```bash
export UCX_TLS=all
export UCX_NET_DEVICES=all
```

支持选择后端：`UCX`、`LIBFABRIC`、`GDS` 等。

## 基本用法

### Producer (Prefiller)

```bash
CUDA_VISIBLE_DEVICES=0 \
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
vllm serve Qwen/Qwen3-0.6B --port 8100 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
```

### Consumer (Decoder)

```bash
CUDA_VISIBLE_DEVICES=1 \
VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
vllm serve Qwen/Qwen3-0.6B --port 8200 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
```

### Proxy Server

```bash
python tests/v1/kv_connector/nixl_integration/toy_proxy_server.py \
  --port 8192 --prefiller-hosts localhost --prefiller-ports 8100 \
  --decoder-hosts localhost --decoder-ports 8200
```

## 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `VLLM_NIXL_SIDE_CHANNEL_PORT` | 5600 | 握手通信端口 |
| `VLLM_NIXL_SIDE_CHANNEL_HOST` | localhost | 跨机器部署时设置 |
| `VLLM_NIXL_ABORT_REQUEST_TIMEOUT` | 480 | KV Cache 释放超时（秒） |

## KV Role

- `kv_producer`：Prefiller 实例
- `kv_consumer`：Decoder 实例
- `kv_both`：对称模式，可同时作为 producer 和 consumer

## KV Load Failure Policy

- `fail`（默认）：KV 加载失败立即报错
- `recompute`：本地重算，可能导致性能抖动

## 实验特性

- **异构 KV Layout**：`enable_permute_local_kv=True`（Prefill HND + Decode NHD）
- **Cross layers blocks**：`enable_cross_layers_blocks=True`，减少传输缓冲区

## 多机部署

支持多 Prefiller + 多 Decoder 的跨机器部署，通过 `VLLM_NIXL_SIDE_CHANNEL_HOST` 指定各节点 IP。

## 兼容性矩阵

### 通用支持特性

所有模型架构均支持：Chunked Prefill、APC、Data Parallel、CUDA graph、Logprobs、Prompt Embeds、多 NIXL 后端。

### 模型架构兼容性

| 模型类型 | Basic PD | Spec Decode | Hetero TP | Cross-layer | SWA | Host buffer | Hetero block |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dense Transformer | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟠 |
| MLA (DeepSeek-V2/V3) | ✅ | ✅ | 🟠 | ✅ | ✅ | ✅ | 🟠 |
| Hybrid SSM/Mamba | ✅ | ❔ | 🚧 | ❌ | ✅ | ✅ | ❌ |
| MoE | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟠 |
| Encoder-Decoder | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

### P/D 必须一致

- vLLM 版本和 NIXL connector 版本
- 模型架构、dtype、KV heads 数量、head size、hidden layers
- Attention backend
- KV cache dtype

### 可以不同的配置

- tensor-parallel-size（异构 TP）
- block-size（异构 block size）
- KV cache block 数量

### KV Cache Layout

- 默认 HND 布局（非 MLA 模型）
- NHD 支持但不支持异构 TP head splitting
- 实验性 HND ↔ NHD permute：`enable_permute_local_kv=true`

### 量化 KV Cache

- 静态量化：✅ 支持
- 动态量化：❌ 不支持（per-block scales 不传输）
- Packed-layout scales：✅ 支持
