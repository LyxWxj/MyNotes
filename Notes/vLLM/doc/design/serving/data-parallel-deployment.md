---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm/blob/main/docs/serving/data_parallel_deployment.md
---

# Data Parallel Deployment（数据并行部署）

vLLM支持数据并行部署，其中模型权重在单独的实例/GPU上复制以处理独立的请求批次。

适用于密集和MoE模型。

## MoE模型的特殊考虑

对于MoE模型（特别是像DeepSeek这样采用MLA的模型），对注意力层使用数据并行，对专家层使用专家或张量并行（EP或TP）可能更有利。

在这些情况下，数据并行rank不是完全独立的：
- 前向传播必须对齐
- 所有rank的专家层在每个前向传播期间都需要同步，即使当前调度的请求数少于DP rank

默认情况下，专家层形成大小为`DP × TP`的张量并行组。要使用专家并行，包含`--enable-expert-parallel` CLI参数。

## 架构

在vLLM中，每个DP rank部署为单独的"核心引擎"进程，通过ZMQ套接字与前端进程通信。数据并行注意力可以与张量并行注意力结合使用，此时每个DP引擎拥有等于配置TP大小的每GPU工作进程数。

对于MoE模型，当任何rank有请求进行中时，必须确保在当前没有请求调度的所有rank中执行空的"虚拟"前向传播。这通过单独的DP协调器进程处理，该进程与所有rank通信，并每N步执行一次集体操作以确定所有rank何时变为空闲并可以暂停。

## 负载均衡

在所有情况下，在DP rank之间进行负载均衡是有益的。对于在线部署，可以通过考虑每个DP引擎的状态（特别是其当前调度和等待的请求以及KV缓存状态）来优化此平衡。

## 在线部署模式

vLLM支持两种在线部署模式：

### 1. 内部负载均衡

vLLM支持"自包含"数据并行部署，暴露单个API端点。

**配置**：在vllm serve命令行参数中包含`--data-parallel-size=4`。

**单节点示例**：
```bash
vllm serve $MODEL --data-parallel-size 4 --tensor-parallel-size 2
```

**多节点示例**：
```bash
# 节点0（IP地址10.99.48.128）
vllm serve $MODEL --data-parallel-size 4 --data-parallel-size-local 2 \
                  --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345

# 节点1
vllm serve $MODEL --headless --data-parallel-size 4 --data-parallel-size-local 2 \
                  --data-parallel-start-rank 2 \
                  --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345
```

**使用Ray**：
```bash
vllm serve $MODEL --data-parallel-size 4 --data-parallel-size-local 2 \
                  --data-parallel-backend=ray
```

**Ray的优势**：
- 单个启动命令（在任何节点上）即可启动所有本地和远程DP rank
- 无需指定`--data-parallel-address`
- 无需指定`--data-parallel-rpc-port`

**API服务器扩展**：当部署大型DP大小时，API服务器进程可能成为瓶颈。使用`--api-server-count`命令行选项进行扩展（例如`--api-server-count=4`）。

### 2. 混合负载均衡

混合负载均衡介于内部和外部方法之间。每个节点运行自己的API服务器，仅将请求排队到该节点上共置的数据并行引擎。上游负载均衡器（例如入口控制器或流量路由器）将用户请求分散到这些每节点端点。

**启用**：`--data-parallel-hybrid-lb`

**关键区别**：
- 必须提供`--data-parallel-size-local`和`--data-parallel-start-rank`
- 不兼容`--headless`
- 根据本地rank数量扩展`--api-server-count`

### 3. 外部负载均衡

对于大规模部署，在外部处理数据并行rank的编排和负载均衡可能更有意义。

在这种情况下，将每个DP rank视为单独的vLLM部署，具有自己的端点，并让外部路由器在它们之间平衡HTTP请求。

**单节点配置**：
```bash
# Rank 0
CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL --data-parallel-size 2 --data-parallel-rank 0 \
                                         --port 8000
# Rank 1
CUDA_VISIBLE_DEVICES=1 vllm serve $MODEL --data-parallel-size 2 --data-parallel-rank 1 \
                                         --port 8001
```

**多节点配置**：
```bash
# Rank 0（IP地址10.99.48.128）
vllm serve $MODEL --data-parallel-size 2 --data-parallel-rank 0 \
                  --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345
# Rank 1
vllm serve $MODEL --data-parallel-size 2 --data-parallel-rank 1 \
                  --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345
```

协调器进程也在此场景中运行，与DP rank 0引擎共置。
