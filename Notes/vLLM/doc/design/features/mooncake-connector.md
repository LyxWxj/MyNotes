---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm
---

# MooncakeConnector

## 概述

MooncakeConnector 使用 Mooncake 的 RDMA 技术实现零拷贝 KV Cache 传输，适用于分离式预填充场景。Mooncake 构建多级缓存池（DRAM/SSD），最大化利用多 NIC 资源。

## 前置条件

```bash
uv pip install mooncake-transfer-engine
```

## 使用方式

### Prefiller 节点

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8010 \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer"}'
```

### Decoder 节点

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8020 \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer"}'
```

### Proxy

```bash
python examples/online_serving/disaggregated_serving/mooncake_connector/mooncake_connector_proxy.py \
  --prefill http://192.168.0.2:8010 --decode http://192.168.0.3:8020
```

## 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `VLLM_MOONCAKE_BOOTSTRAP_PORT` | 8998 | Bootstrap 服务端口（仅 prefiller） |
| `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT` | 480 | KV Cache 释放超时（秒） |

## KV Role

- `kv_producer`：Prefiller 实例
- `kv_consumer`：Decoder 实例
- `kv_both`：对称模式

## 额外配置

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `num_workers` | 10 | 传输线程池大小 |
| `mooncake_protocol` | rdma | 传输协议 |
