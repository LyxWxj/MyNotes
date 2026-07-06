---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm/blob/main/docs/serving/expert_parallel_deployment.md
---

# Expert Parallel Deployment（专家并行部署）

vLLM支持专家并行（EP），允许混合专家（MoE）模型中的专家部署在单独的GPU上，提高局部性、效率和整体吞吐量。

EP通常与数据并行（DP）耦合。虽然DP可以独立于EP使用，但EP与DP结合使用时更高效。

## 前置条件

使用EP前，需要安装必要的依赖：

1. **安装DeepEP**：按照vLLM的EP内核指南设置主机环境
2. **安装DeepGEMM库**：按照[官方说明](https://github.com/deepseek-ai/DeepGEMM#installation)
3. **对于解聚服务**：运行[`install_gdrcopy.sh`](../../tools/install_gdrcopy.sh)脚本安装`gdrcopy`

### 后端选择指南

vLLM提供多个EP通信后端，使用`--all2all-backend`选择：

| 后端 | 用例 | 特性 | 最佳适用 |
|------|------|------|----------|
| `allgather_reducescatter` | 默认后端 | 使用allgather/reducescatter原语的标准all2all | 通用，适用于任何EP+DP配置 |
| `deepep_high_throughput` | 多节点预填充 | 分组GEMM，连续布局，优化预填充 | 预填充主导的工作负载 |
| `deepep_low_latency` | 多节点解码 | CUDA图支持，掩码布局，优化解码 | 解码主导的工作负载 |
| `flashinfer_nvlink_one_sided` | MNNVL系统 | FlashInfer的单侧A2A策略 | 高吞吐量工作负载 |
| `flashinfer_nvlink_two_sided` | MNNVL系统 | FlashInfer的双侧A2A策略 | 跨节点NVLink系统 |

## 单节点部署

### 配置

通过设置`--enable-expert-parallel`标志启用EP。EP大小自动计算：

```
EP_SIZE = TP_SIZE × DP_SIZE
```

### 启用EP时的层行为

| 层类型 | 行为 | 使用的并行 |
|--------|------|-----------|
| **专家（MoE）层** | 跨所有EP rank分片 | 专家并行（EP），大小为`TP × DP` |
| **注意力层** | 取决于TP大小 | 见下文 |

**注意力层并行**：
- **当`TP = 1`**：注意力权重在所有DP rank上**复制**（数据并行）
- **当`TP > 1`**：注意力权重使用张量并行在每个DP组内的TP rank上**分片**

**示例**：`TP=2, DP=4`（共8个GPU）
- 专家层形成大小为8的EP组，专家分布在所有GPU上
- 注意力层在4个DP组中的每个组内使用TP=2

### 示例命令

```bash
# 单节点EP部署
vllm serve deepseek-ai/DeepSeek-V3-0324 \
    --tensor-parallel-size 1 \       # 1个GPU上的张量并行
    --data-parallel-size 8 \         # 8个进程上的数据并行
    --enable-expert-parallel         # 启用专家并行
```

## 多节点部署

对于多节点部署，使用DeepEP通信内核的两种模式之一。

### 部署步骤

1. **每个节点运行一个命令** - 每个节点需要自己的启动命令
2. **配置网络** - 确保正确的IP地址和端口配置
3. **设置节点角色** - 第一个节点处理请求，其他节点以无头模式运行

### 2节点部署示例

```bash
# 节点1（主节点 - 处理传入请求）
vllm serve deepseek-ai/DeepSeek-V3-0324 \
    --all2all-backend deepep_low_latency \
    --tensor-parallel-size 1 \               # 每节点TP大小
    --enable-expert-parallel \               # 启用EP
    --data-parallel-size 16 \                # 所有节点的总DP大小
    --data-parallel-size-local 8 \           # 此节点的本地DP大小（每节点8个GPU）
    --data-parallel-address 192.168.1.100 \  # 替换为节点1的实际IP
    --data-parallel-rpc-port 13345 \         # RPC通信端口
    --api-server-count=8                     # API服务器数量

# 节点2（次节点 - 无头模式，无API服务器）
vllm serve deepseek-ai/DeepSeek-V3-0324 \
    --all2all-backend deepep_low_latency \
    --tensor-parallel-size 1 \               # 每节点TP大小
    --enable-expert-parallel \               # 启用EP
    --data-parallel-size 16 \                # 所有节点的总DP大小
    --data-parallel-size-local 8 \           # 此节点的本地DP大小
    --data-parallel-start-rank 8 \           # 此节点的起始rank偏移
    --data-parallel-address 192.168.1.100 \  # 主节点（节点1）的IP
    --data-parallel-rpc-port 13345 \         # 与主节点相同的RPC端口
    --headless                               # 无API服务器，仅工作进程
```

### 关键配置说明

- **无头模式**：次节点使用`--headless`标志运行，所有客户端请求由主节点处理
- **Rank计算**：`--data-parallel-start-rank`应等于之前节点的累积本地DP大小
- **负载扩展**：在主节点上调整`--api-server-count`以处理更高的请求负载

### 网络配置

> **InfiniBand集群**：设置此环境变量以防止初始化挂起：
> ```bash
> export GLOO_SOCKET_IFNAME=eth0
> ```

## 专家并行负载均衡器（EPLB）

虽然MoE模型通常训练为每个专家接收相似数量的token，但在实践中，token在专家之间的分布可能高度倾斜。vLLM提供专家并行负载均衡器（EPLB）以在EP rank之间重新分配专家映射，平衡专家之间的负载。

### 配置

使用`--enable-eplb`标志启用EPLB。

### EPLB参数

使用`--eplb-config`参数配置EPLB，接受JSON字符串：

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `window_size` | 跟踪用于重新平衡决策的引擎步数 | 1000 |
| `step_interval` | 重新平衡频率（每N个引擎步） | 3000 |
| `log_balancedness` | 记录平衡度指标 | `false` |
| `num_redundant_experts` | 每个EP rank超出均匀分布的额外全局专家 | `0` |
| `use_async` | 使用非阻塞EPLB以减少延迟开销 | `false` |
| `policy` | 专家并行负载均衡的策略类型 | `"default"` |

**示例**：
```bash
vllm serve Qwen/Qwen3-30B-A3B \
  --enable-eplb \
  --eplb-config '{"window_size":1000,"step_interval":3000,"num_redundant_experts":2,"log_balancedness":true}'
```

### 专家分布公式

- **默认**：每个EP rank有`NUM_TOTAL_EXPERTS ÷ NUM_EP_RANKS`个专家
- **带冗余**：每个EP rank有`(NUM_TOTAL_EXPERTS + NUM_REDUNDANT_EXPERTS) ÷ NUM_EP_RANKS`个专家

### 内存占用开销

EPLB使用需要放入GPU内存的冗余专家。这意味着EPLB可能不适合内存受限的环境或KV缓存空间紧张的情况。

此开销等于`NUM_MOE_LAYERS * BYTES_PER_EXPERT * (NUM_TOTAL_EXPERTS + NUM_REDUNDANT_EXPERTS) ÷ NUM_EP_RANKS`。对于DeepSeekV3，每个EP rank一个冗余专家约为`2.4 GB`。

## 性能优化

- **DeepEP内核**：`high_throughput`和`low_latency`内核针对解聚服务优化，混合工作负载可能表现不佳
- **双批次重叠**：使用`--enable-dbo`重叠all-to-all通信与计算
- **异步调度（实验性）**：尝试`--async-scheduling`重叠调度与模型执行

## 故障排除

| 错误 | 解决方案 |
|------|----------|
| `non-zero status: 7 cannot register cq buf` | 使用Infiniband/RoCE时，确保主机VM和pod显示`ulimit -l` "unlimited" |
| `init failed for transport: IBGDA` | InfiniBand GDA内核模块缺失。在每个GPU节点上运行`tools/ep_kernels/configure_system_drivers.sh`并重启 |
| NVSHMEM peer disconnect | 通常是网络配置错误。通过Kubernetes部署时，验证每个pod以`hostNetwork: true`、`securityContext.privileged: true`运行 |

## 解聚服务（预填充/解码分离）

对于需要严格SLA保证首token时间和token间延迟的生产部署，解聚服务允许独立扩展预填充和解码操作。

### 架构概述

- **预填充实例**：使用`deepep_high_throughput`后端以获得最佳预填充性能
- **解码实例**：使用`deepep_low_latency`后端以获得最小解码延迟
- **KV缓存传输**：通过NIXL或其他KV连接器连接实例

### 设置步骤

1. **安装gdrcopy/ucx/nixl**：运行[install_gdrcopy.sh](../../tools/install_gdrcopy.sh)脚本
2. **配置两个实例**：向预填充和解码实例添加`--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'`
3. **客户端编排**：使用客户端脚本协调预填充/解码操作

### 基准测试

- 要模拟解聚服务的解码部署，传递`--kv-transfer-config '{"kv_connector":"DecodeBenchConnector","kv_role":"kv_both"}'`
- **CUDAGraph捕获**：使用`--compilation_config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'`仅为解码启用CUDA图捕获
