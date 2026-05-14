---
type: Note
related_to: "[[vLLM-omni]]"
status: Active
url: https://github.com/vllm-project/vllm-omni/blob/main/docs/design/module/entrypoint_module.md
---

# Entrypoint Module

> **注意**：此文档即将更新。

入口点模块定义vLLM-Omni的API接口，用于离线和在线服务。

## 主要入口点

### Omni（离线推理）

```python
from vllm_omni.entrypoints.omni import Omni

omni = Omni(model="Qwen/Qwen3-Omni-30B-A3B-Instruct")

outputs = omni.generate(om_inputs, sampling_params_list)
```

### AsyncOmni（异步推理）

```python
from vllm_omni.entrypoints.async_omni import AsyncOmni

async_omni = AsyncOmni(model="Qwen/Qwen3-Omni-30B-A3B-Instruct")

# 异步生成
outputs = await async_omni.generate_async(om_inputs, sampling_params_list)
```

### API Server（在线服务）

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091
```

## 相关文档

- [Architecture Overview](../architecture_overview.md) - 接口设计示例
- [AsyncOmni Architecture](async_omni_architecture.md) - 异步架构详细设计
