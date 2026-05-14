---
type: Note
related_to: "[[vLLM-omni]]"
status: Active
url: https://github.com/vllm-project/vllm-omni/blob/main/docs/design/feature/disaggregated_inference.md
---

# Disaggregated Inference（解聚推理）

本指南说明如何在vllm-omni中配置和使用分布式连接器（`vllm_omni/distributed/omni_connectors`）进行多阶段管道。

## 概述

连接器支持管道阶段之间的数据传输（如Thinker → Talker），当前连接器以D2H2D（设备到主机到设备）模式运行。

## 连接器选择

| 用例 | 推荐连接器 | 备注 |
|------|-----------|------|
| 单节点 | SharedMemoryConnector | 未指定连接器时自动配置 |
| 多节点（Mooncake Store） | MooncakeStoreConnector | 基于TCP，需要Mooncake Master + 元数据服务器 |
| 多节点（Mooncake RDMA） | MooncakeTransferEngineConnector | RDMA/TCP直接传输，最快 |
| 多节点（Yuanrong） | YuanrongConnector | 需要Yuanrong Datasystem + etcd |

## 核心API

```python
class OmniConnectorBase(ABC):
    @abstractmethod
    def put(self, from_stage: str, to_stage: str, put_key: str, data: Any) -> tuple[bool, int, Optional[dict]]:
        """存储数据，返回：(success, serialized_size, metadata)"""

    @abstractmethod
    def get(self, from_stage: str, to_stage: str, get_key: str, metadata: Optional[dict] = None) -> Optional[tuple[Any, int]]:
        """检索数据，返回：(object, serialized_size)"""
```

### 元数据传递

某些连接器（如SharedMemoryConnector）在`put()`期间生成临时资源，此`metadata`必须通过控制平面传递，以便`get()`可以定位数据。

## 配置模型

### 连接器定义

```yaml
runtime:
  connectors:
    connector_of_shared_memory:
      name: SharedMemoryConnector
      extra:
        shm_threshold_bytes: 65536
```

### 阶段连接

```yaml
stage_args:
  - stage_id: 0
    output_connectors:
      to_stage_1: connector_of_shared_memory
  - stage_id: 1
    input_connectors:
      from_stage_0: connector_of_shared_memory
```

如果管道边没有显式连接器，系统会自动为该边创建SharedMemoryConnector。

## 与vLLM的关系

vLLM提供特定的分布式机制：
- KV Transfer：针对KV缓存优化
- EC Transfer：针对编码器嵌入优化
- Device Communicators：低级原语（NCCL、SHM）

vllm-omni通过通用连接器抽象补充：
1. 通过单个`put`/`get`API统一传输
2. 支持跨进程或节点的DAG式管道
3. 可包装vLLM特定传输以保持一致接口

## 操作注意事项

- 快速失败配置验证：缺少预期边会导致启动失败
- 缺少有效负载会停止阶段：验证连接器连接和元数据传播

## 未来路线图：D2D传输

当前连接器使用D2H2D路径，未来版本将引入直接设备到设备连接器（NCCL、UCX、IPC）以减少大张量有效负载的延迟。

## 后端特定设置

- [SharedMemoryConnector](omni_connectors/shared_memory_connector.md)
- [MooncakeStoreConnector](omni_connectors/mooncake_store_connector.md)
- [MooncakeTransferEngineConnector](omni_connectors/mooncake_transfer_engine_connector.md)
- [YuanrongConnector](omni_connectors/yuanrong_connector.md)
