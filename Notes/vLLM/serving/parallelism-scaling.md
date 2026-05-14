---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm/blob/main/docs/serving/parallelism_scaling.md
---

# Parallelism and Scaling（并行与扩展）

## 单模型副本的分布式推理策略

选择分布式推理策略的指南：

| 场景 | 策略 | 示例配置 |
|------|------|----------|
| **单GPU** | 无需分布式推理 | - |
| **单节点多GPU** | 张量并行 | `tensor_parallel_size=4` |
| **多节点多GPU** | 张量并行 + 管道并行 | `tensor_parallel_size=8, pipeline_parallel_size=2` |

增加GPU和节点数量直到有足够的GPU内存容纳模型。设置`tensor_parallel_size`为每个节点的GPU数量，`pipeline_parallel_size`为节点数量。

### 日志检查

运行`vllm`后，查找以下日志消息：

```text
INFO 07-23 13:56:04 [kv_cache_utils.py:775] GPU KV cache size: 643,232 tokens
INFO 07-23 13:56:04 [kv_cache_utils.py:779] Maximum concurrency for 40,960 tokens per request: 15.70x
```

- `GPU KV cache size`：GPU KV缓存可存储的总token数
- `Maximum concurrency`：如果每个请求需要指定token数，可并发服务的请求数估计

### 边缘情况：不均匀GPU分割

如果模型适合单节点但GPU数量不能均匀分割模型大小，启用管道并行（沿层分割，支持不均匀分割）。设置`tensor_parallel_size=1`，`pipeline_parallel_size`为GPU数量。

如果节点上的GPU没有NVLINK互连（如L40S），使用管道并行代替张量并行以获得更高吞吐量和更低通信开销。

### MoE模型的分布式服务

对于MoE模型，通常可以利用专家的固有并行性，使用单独的并行策略处理专家层。vLLM支持大规模部署，结合数据并行注意力和专家或张量并行MoE层。

## 单节点部署

vLLM支持分布式张量并行和管道并行推理和服务。实现包括[Megatron-LM的张量并行算法](https://arxiv.org/pdf/1909.08053.pdf)。

默认分布式运行时：
- **多节点推理**：[Ray](https://github.com/ray-project/ray)
- **单节点推理**：原生Python `multiprocessing`

可通过`distributed_executor_backend`覆盖默认值：`mp`（multiprocessing）或`ray`（Ray）。

### 多GPU推理

```python
from vllm import LLM

llm = LLM("facebook/opt-13b", tensor_parallel_size=4)
output = llm.generate("San Francisco is a")
```

### 多GPU服务

```bash
vllm serve facebook/opt-13b \
     --tensor-parallel-size 4
```

### 启用管道并行

```bash
# 8个GPU，4个张量并行，2个管道并行
vllm serve gpt2 \
     --tensor-parallel-size 4 \
     --pipeline-parallel-size 2
```

## 多节点部署

如果单节点没有足够的GPU容纳模型，将vLLM部署到多个节点。确保每个节点提供相同的执行环境，包括模型路径和Python包。建议使用容器镜像。

### Ray是什么？

Ray是一个用于扩展Python程序的分布式计算框架。多节点vLLM部署可以使用Ray作为运行时引擎。

vLLM使用Ray管理跨多个节点的任务分布式执行，并控制执行位置。

Ray是可选依赖项。使用前显式安装：

```bash
pip install "ray[cgraph]"
```

### 使用容器的Ray集群设置

辅助脚本[examples/online_serving/run_cluster.sh](../../examples/online_serving/run_cluster.sh)跨节点启动容器并初始化Ray。

**头节点**：
```bash
bash run_cluster.sh \
                vllm/vllm-openai \
                <HEAD_NODE_IP> \
                --head \
                /path/to/the/huggingface/home/in/this/node \
                -e VLLM_HOST_IP=<HEAD_NODE_IP>
```

**工作节点**：
```bash
bash run_cluster.sh \
                vllm/vllm-openai \
                <HEAD_NODE_IP> \
                --worker \
                /path/to/the/huggingface/home/in/this/node \
                -e VLLM_HOST_IP=<WORKER_NODE_IP>
```

> **注意**：`VLLM_HOST_IP`对每个工作节点是唯一的。保持运行这些命令的shell打开；关闭任何shell都会终止集群。

> **网络安全**：为安全起见，将`VLLM_HOST_IP`设置为私网段的地址。通过此网络发送的流量是未加密的。

### 在Ray集群上运行vLLM

Ray集群运行后，像在单节点设置中一样使用vLLM。跨Ray集群的所有资源对vLLM可见，因此在单个节点上运行单个`vllm`命令就足够了。

常见做法是将张量并行大小设置为每个节点的GPU数量，管道并行大小设置为节点数量：

```bash
vllm serve /path/to/the/model/in/the/container \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 2 \
    --distributed-executor-backend ray
```

### 使用MultiProcessing运行vLLM

除Ray外，多节点vLLM部署也可以使用`multiprocessing`作为运行时引擎：

**头节点**：
```bash
vllm serve /path/to/the/model/in/the/container \
  --tensor-parallel-size 8 --pipeline-parallel-size 2 \
  --nnodes 2 --node-rank 0 \
  --master-addr <HEAD_NODE_IP>
```

**工作节点**：
```bash
vllm serve /path/to/the/model/in/the/container \
  --tensor-parallel-size 8 --pipeline-parallel-size 2 \
  --nnodes 2 --node-rank 1 \
  --master-addr <HEAD_NODE_IP> --headless
```

## 优化张量并行的网络通信

高效的张量并行需要快速的节点间通信，最好通过高速网络适配器（如InfiniBand）。

要设置集群使用InfiniBand，向辅助脚本追加参数如`--privileged -e NCCL_IB_HCA=mlx5`。

## 启用GPUDirect RDMA

GPUDirect RDMA（远程直接内存访问）是NVIDIA技术，允许网络适配器直接访问GPU内存，绕过CPU和系统内存。

### Docker设置

```bash
docker run --gpus all \
    --ipc=host \
    --shm-size=16G \
    -v /dev/shm:/dev/shm \
    vllm/vllm-openai
```

### Kubernetes设置

```yaml
spec:
  containers:
    - name: vllm
      image: vllm/vllm-openai
      securityContext:
        capabilities:
          add: ["IPC_LOCK"]
      volumeMounts:
        - mountPath: /dev/shm
          name: dshm
      resources:
        limits:
          nvidia.com/gpu: 8
        requests:
          nvidia.com/gpu: 8
  volumes:
    - name: dshm
      emptyDir:
        medium: Memory
```

> **确认GPUDirect RDMA操作**：使用详细NCCL日志运行vLLM：`NCCL_DEBUG=TRACE vllm serve ...`
> - 如果找到`[send] via NET/IB/GDRDMA`，NCCL正在使用InfiniBand与GPUDirect RDMA（高效）
> - 如果找到`[send] via NET/Socket`，NCCL使用原始TCP套接字（低效）

> **预下载HuggingFace模型**：建议在启动vLLM前下载模型。在每个节点上下载模型到相同路径，或存储在所有节点可访问的分布式文件系统上。
