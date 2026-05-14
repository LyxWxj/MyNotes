---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm
---

# Sleep Mode

## 概述

Sleep Mode 允许临时释放 GPU 显存（模型权重和 KV Cache），无需停止服务器或卸载 Docker 容器。适用于 RLHF、训练或节省成本的场景。

## 核心优势

- **释放 GPU 显存**：卸载权重到 CPU 并丢弃 KV Cache，释放 90%+ 显存
- **快速恢复**：无需完整模型重载即可唤醒
- **API 控制**：通过 HTTP 端点或 Python API 控制 sleep/wake_up
- **支持分布式**：兼容 tensor parallelism、pipeline parallelism
- **细粒度控制**：可选择性恢复权重或 KV Cache

## Sleep 级别

### Level 1

- 卸载模型权重到 CPU RAM，丢弃 KV Cache
- 权重保留在 CPU 内存中
- 适用于休眠后恢复同一模型

### Level 2

- 丢弃模型权重和 KV Cache
- 适用于更换模型或 RLHF 权重更新

## 使用方式

### 离线推理

```python
llm = LLM("Qwen/Qwen3-0.6B", enable_sleep_mode=True)

# Level 1 sleep
llm.sleep(level=1)
llm.wake_up()

# Level 2 sleep (RLHF 场景)
llm.sleep(level=2)
llm.wake_up(tags=["weights"])        # 仅恢复权重
llm.collective_rpc("reload_weights") # 原地加载新权重
llm.wake_up(tags=["kv_cache"])       # 恢复 KV Cache
```

### 在线服务

需要开发模式：`VLLM_SERVER_DEV_MODE=1`

```bash
VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3-0.6B --enable-sleep-mode
```

HTTP 端点：
- `POST /sleep?level=1` — 休眠
- `POST /wake_up` — 唤醒（支持 `?tags=weights`）
- `POST /collective_rpc` — RPC 调用
- `GET /is_sleeping` — 检查状态

## RLHF 权重更新流程

```python
llm.sleep(level=2)
# ... 获取新权重
llm.wake_up(tags=["weights"])  # 避免 OOM
# ... 更新权重
llm.wake_up(tags=["kv_cache"])
```

## ROCm 限制

通过 `VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE`（MB）控制虚拟内存分配块大小，默认 256MB。
