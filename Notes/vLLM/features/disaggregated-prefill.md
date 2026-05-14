---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm
---

# Disaggregated Prefill

## 概述

Disaggregated Prefill（分离式预填充）将 LLM 推理的预填充（prefill）和解码（decode）阶段分配到不同的 vLLM 实例中运行。

## 核心优势

1. **独立调优 TTFT 和 ITL**：可为 prefill 和 decode 设置不同的并行策略（tp、pp），独立优化首 token 延迟和每 token 延迟
2. **控制尾部 ITL**：避免在解码过程中插入 prefill 任务，消除尾部延迟尖峰

!!! note
    Disaggregated Prefill **不会**提升吞吐量，仅用于延迟优化。

## Connector 类型

vLLM 支持多种 Connector 实现：

| Connector | 特点 |
| --- | --- |
| **ExampleConnector** | 参考实现 |
| **LMCacheConnectorV1** | 使用 NIXL 作为底层 KV 传输 |
| **NixlConnector** | 完全异步 send/recv，高性能 |
| **P2pNcclConnector** | 基于 NCCL 的点对点传输 |
| **MooncakeConnector** | 基于 RDMA 的传输 |
| **MultiConnector** | 组合多个 Connector |
| **OffloadingConnector** | KV 数据卸载到 CPU 内存 |
| **FlexKVConnectorV1** | 分布式 KV Store 和多级缓存管理 |

## 关键抽象

### Connector

允许 **kv consumer** 从 **kv producer** 检索 KV Cache。

### LookupBuffer

提供类 SQL 的 API：
- `insert`：非阻塞，插入 KV Cache
- `drop_select`：阻塞，检索并删除匹配的 KV Cache

### Pipe

单向 FIFO 管道，支持 `send_tensor` 和 `recv_tensor`。

## 架构设计

- **Scheduler Connector**：与调度器同进程，调度 KV Cache 传输操作
- **Worker Connector**：在 worker 进程中，执行 KV Cache 传输操作
- 支持**逐层** KV Cache 存取，与注意力模块深度集成

## 扩展方式

三种实现第三方 Connector 的方式：

1. **Fully-customized Connector**：完全自定义，控制力最强
2. **Database-like Connector**：实现 LookupBuffer 的 `insert`/`drop_select` API
3. **Distributed P2P Connector**：实现 Pipe 的 `send_tensor`/`recv_tensor` API

## 实现位置

所有代码位于 `vllm/distributed/kv_transfer`
