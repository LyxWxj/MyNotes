---
type: Note
related_to: "[[vllm]]"
status: Active
---

# vllm/distributed 分布式基础设施模块

`vllm/vllm/distributed/` 是 vLLM 原生的分布式基础设施模块，覆盖分布式通信、并行状态管理、KV cache 跨节点传输、专家负载均衡、权重传输等多个子系统。

## 1. parallel_state.py — 分布式并行状态管理（核心）

整个分布式模块的**中枢**，接管 PyTorch 的分布式环境控制：

- 初始化/销毁分布式环境（`init_distributed_environment` / `destroy_distributed_environment`）
- 管理 Tensor Parallel（TP）、Pipeline Parallel（PP）、Data Parallel（DP）进程组的创建与销毁（`initialize_model_parallel` / `destroy_model_parallel`）
- 提供 `GroupCoordinator` 封装进程组内的 all-reduce、broadcast、all-gather、reduce-scatter 等集合通信操作
- 维护 world size、rank、device 等全局分布式状态
- 支持 symmetric memory、multimodal 并行等高级特性

典型工作流：`init_distributed_environment` → `initialize_model_parallel` → 业务逻辑 → `destroy_model_parallel` → `destroy_distributed_environment`

## 2. device_communicators/ — 设备级通信后端

GPU、RDMA、共享内存等不同层级的通信实现：

- **custom_all_reduce.py / quick_all_reduce.py** — vLLM 自定义 all-reduce，针对小张量优化，减少 GPU 间同步开销，支持 P2P 直接内存访问
- **flashinfer_all_reduce.py** — 基于 FlashInfer 内核的 all-reduce
- **pynccl.py / pynccl_wrapper.py / pynccl_allocator.py** — NCCL 的 Python 封装，专用于 GPU 间的高效集合通信
- **cuda_communicator.py / cpu_communicator.py / xpu_communicator.py** — 不同设备类型（CUDA/CPU/XPU）的通信器适配
- **shm_broadcast.py / shm_object_storage.py** — 基于 Python `multiprocessing.shared_memory` 的同机进程间广播和对象存储
- **symm_mem.py** — CUDA symmetric memory（跨 GPU 直接内存访问）封装
- **ray_communicator.py** — Ray 分布式框架的通信适配
- **mnnvl_compat.py** — MN-NVL（Multi-Node NVLink）兼容层
- **all2all.py** — all-to-all 通信操作
- **base_device_communicator.py** — 设备通信器的抽象基类规范

## 3. kv_transfer/ — KV Cache 跨节点传输

Prefill-Decode **分离部署**（disaggregated serving）的核心通信层：

- **kv_connector/base.py** — KV 连接器抽象基类，定义统一的 `put`/`get` 接口
- **kv_connector/factory.py** — 连接器工厂，根据配置动态创建连接器实例
- **kv_connector/v1/** — 丰富的连接器实现（V1 引擎）：
  - `mooncake/` — Mooncake 分布式 KV 存储连接器
  - `nixl_connector.py` — NVIDIA NIXL 多节点 RDMA 传输
  - `lmcache_connector.py` / `lmcache_mp_connector.py` — LM-Cache 集成（支持多进程）
  - `hf3fs/` — 基于 3FS 分布式文件系统的连接器（含 client、metadata server、gather/scatter 工具）
  - `p2p/` — 基于 NCCL 的 GPU 点对点传输 + tensor memory pool
  - `moriio/` — Moriio 连接器
  - `flexkv_connector.py` — FlexKV 连接器
  - `offloading_connector.py` / `offloading/` — KV Cache 异步 offload 到 CPU/磁盘（含 scheduler、worker）
  - `simple_cpu_offload_connector.py` — 轻量级 CPU offload
  - `multi_connector.py` — 多连接器组合调度
  - `decode_bench_connector.py` — 基准测试用 decode 连接器
  - `example_connector.py` / `example_hidden_states_connector.py` — 示例/参考实现
  - `ssm_conv_transfer_utils.py` — SSM（State Space Model）卷积转移工具
- **kv_transfer_state.py** — KV transfer 组的生命周期管理（初始化、关闭、状态查询）
- **kv_connector/utils.py** — 连接器公共工具

## 4. kv_events.py — KV Cache 事件系统

定义和传输 KV cache 相关事件（基于 ZMQ PUB/SUB + msgspec 高性能序列化），用于分布式节点间同步 KV block 的状态变更：

- `KVCacheEvent` — 事件体（free / allocate / copy 等 block 操作）
- `EventBatch` — 批量事件容器
- `KVEventBatchPublisher` — 事件发布端（PUB socket）
- `KVEventBatchSubscriber` — 事件订阅端（SUB socket）
- `KVEventAggregator` — 多节点事件聚合器

## 5. eplb/ — Expert Parallelism Load Balancer（专家并行负载均衡器）

针对 **MoE（Mixture of Experts）模型**的专家并行负载均衡：

- **eplb_state.py** — 全局负载均衡状态管理
- **eplb_communicator.py** — 节点间负载均衡通信
- **rebalance_execute.py** — 制定和执行专家重分配计划
- **async_worker.py** — 异步重平衡 worker，不阻塞推理主循环
- **eplb_utils.py** — EPLB 工具函数
- **policy/abstract.py** — 负载均衡策略抽象基类
- **policy/default.py** — 默认负载均衡策略

## 6. elastic_ep/ — 弹性专家并行

支持运行时**动态调整**专家并行度的弹性伸缩机制：

- **elastic_state.py** — 弹性 EP 状态管理
- **elastic_execute.py** — 弹性执行逻辑（权重重分配、通信重组）
- **standby_state.py** — 备用节点状态管理

## 7. weight_transfer/ — 模型权重传输引擎

模型权重从 trainer 节点同步到 inference worker 的传输系统：

- **base.py** — 权重传输引擎抽象基类
- **factory.py** — 引擎工厂（根据配置选择实现）
- **nccl_engine.py** — 基于 NCCL broadcast 的跨节点权重同步
- **ipc_engine.py** — 基于共享内存 IPC 的同机权重同步
- **packed_tensor.py** — 打包张量编码/解码（多 tensor 合并传输）

## 8. ec_transfer/ — Embedding Cache 传输连接器

与 KV cache transfer 平行的一套传输机制，专门针对 **embedding 级别**的缓存传输（区别于 attention KV cache）：

- **ec_connector/base.py** — EC 连接器抽象基类
- **ec_connector/factory.py** — EC 连接器工厂
- **ec_connector/example_connector.py** — 示例 EC 连接器实现
- **ec_transfer_state.py** — EC transfer 状态管理

## 9. 辅助模块

- **communication_op.py** — 集合通信操作（all-reduce、all-gather）的高层封装
- **stateless_coordinator.py** — 无状态进程组协调器：创建**临时一次性进程组**用于特定通信操作（如模型加载参数广播），用完即销毁，通过 TCPStore 协调 rank 间握手
- **utils.py** — 大量工具函数：`StatelessProcessGroup`（无状态进程组）、网络地址解析、port 分配、张量分割/合并、`sched_yield` 等

## 与 vllm_omni/distributed 的对比

| 维度 | `vllm/distributed` | `vllm_omni/distributed` |
|---|---|---|
| 定位 | vLLM **原生**分布式基础层 | vLLM Omni **多模态扩展**的分布式层 |
| 并行策略 | TP/PP/DP/EP 全支持 | 复用 vLLM 并行，不自行管理 |
| KV 传输 | 通用 KV connector 框架 + 十多种后端 | 四个专用 connector（Mooncake/SHM/Yuanrong/MooncakeTransfer） |
| 额外能力 | EPLB、弹性 EP、权重传输、EC 传输、KV 事件系统 | Stage 协调器（ZMQ ROUTER/PUB）、PD 分离 monkey patch |
| 通信层 | 自研 custom all-reduce + PyNCCL + SHM 多层 | 依赖 vLLM 通信层，不重复实现 |
