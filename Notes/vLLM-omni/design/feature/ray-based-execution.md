---
type: Note
related_to: "[[vLLM-omni]]"
status: Active
url: https://github.com/vllm-project/vllm-omni/blob/main/docs/design/feature/ray_based_execution.md
---

# Ray-based Execution（基于Ray的执行）

## 概述

`ray_utils`目录包含vllm-omni中分布式执行的实用程序，支持**Ray**和**Multiprocessing**后端。

## 安装

```bash
pip install "ray[default]"
```

## Ray Utils

`ray_utils`模块提供管理Ray集群和actor的辅助函数，用于：
- **多节点部署**：在不同物理机器上运行管道阶段
- **资源管理**：高效的GPU/CPU分配

### 基本用法

在初始化引擎时指定`worker_backend="ray"`：

```bash
vllm serve Qwen/Qwen2.5-Omni-7B \
  --omni \
  --port 8091 \
  --worker-backend ray \
  --ray-address auto
```

### 集群设置

**步骤1：启动头节点**
```bash
ray start --head --port=6399
```

**步骤2：连接工作节点**
```bash
ray start --address=<HEAD_NODE_IP>:6399
```

> **提示**：完整的集群设置脚本参考vLLM示例：[run_cluster.sh](https://github.com/vllm-project/vllm/blob/main/examples/online_serving/run_cluster.sh)

### 分布式连接器支持

在Ray上运行时，系统自动调整通信策略：

| 场景 | 推荐连接器 | 备注 |
|------|-----------|------|
| 跨节点 | MooncakeTransferEngineConnector | RDMA，最快 |
| 跨节点（备选） | MooncakeStoreConnector | TCP回退 |
| 同节点 | SharedMemoryConnector | 高效 |
| 同节点（备选） | Ray原生对象存储（plasma） | |

**SHM阈值差异**：当`worker_backend="ray"`时，SharedMemoryConnector默认阈值设为`sys.maxsize`，强制有效负载内联（不使用SHM）。如需在Ray运行中使用SHM，请在连接器配置中覆盖`shm_threshold_bytes`。

### 内部辅助函数

- **`initialize_ray_cluster`**：连接到现有Ray集群或启动本地集群

## 故障排除

- **连接问题**：确保Ray头节点可访问，端口（默认6399）已开放
- **版本不匹配**：确保所有节点运行相同版本的Ray和Python
