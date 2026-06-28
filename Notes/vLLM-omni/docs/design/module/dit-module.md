---
type: Note
related_to: "[[vllm-omni]]"
status: Active
url: https://github.com/vllm-project/vllm-omni/blob/main/docs/design/module/dit_module.md
---

# Diffusion Module Architecture Design

vLLM-Omni扩散模块（`vllm_omni/diffusion`）是一个用于扩散模型的高性能推理引擎，采用模块化架构，将关注点分离到多个组件中。它为非自回归生成任务（如图像和视频生成）提供高效执行。

## 架构概述

扩散模块遵循**多进程、分布式架构**，具有清晰的关注点分离：

```
┌─────────────────────────────────────────────────────────────────┐
│                        DiffusionEngine                          │
│                        (Orchestrator)                           │
├─────────────────────────────────────────────────────────────────┤
│                          Scheduler                              │
│                   (Request State Manager)                       │
├─────────────────────────────────────────────────────────────────┤
│                           Worker                                │
│                    (Model Execution)                            │
├─────────────────────────────────────────────────────────────────┤
│                     Diffusion Pipeline                          │
│                   (Model-Specific Logic)                        │
├─────────────────────────────────────────────────────────────────┤
│                  Acceleration Components                        │
│     (Attention Backends, Cache, Parallel Strategies)           │
└─────────────────────────────────────────────────────────────────┘
```

## 1. Diffusion Engine

**位置**：`vllm_omni/diffusion/diffusion_engine.py`

### 职责

`DiffusionEngine`是扩散推理系统的**编排器**，管理工作进程的生命周期并协调执行流程。

### 关键组件

#### 1.1 初始化

```python
class DiffusionEngine:
    def __init__(self, od_config: OmniDiffusionConfig):
        self.od_config = od_config
        self.post_process_func = get_diffusion_post_process_func(od_config)
        self.pre_process_func = get_diffusion_pre_process_func(od_config)
        self._processes: list[mp.Process] = []
        self._make_client()
```

**关键特性**：
- **预/后处理**：通过注册模式注册模型特定的预处理和后处理函数
- **工作进程管理**：启动和管理多个工作进程（每个GPU一个）
- **进程隔离**：使用多进程实现真正的并行

#### 1.2 工作进程启动流程

```python
def _launch_workers(self, broadcast_handle):
    # 每个GPU创建一个进程
    for i in range(num_gpus):
        process = mp.Process(
            target=worker_proc.worker_main,
            args=(i, od_config, writer, broadcast_handle),
            name=f"DiffusionWorker-{i}",
        )
        process.start()
```

**设计决策**：
- **Spawn方法**：确保每个工作进程的干净状态
- **Pipe通信**：使用`mp.Pipe`进行初始化握手
- **设备选择**：每个工作进程分配特定GPU（`cuda:{rank}`）

#### 1.3 请求处理流程

```python
def step(self, requests: list[OmniDiffusionRequest]):
    # 1. 预处理请求
    requests = self.pre_process_func(requests)

    # 2. 发送到调度器并等待响应
    output = self.add_req_and_wait_for_response(requests)

    # 3. 后处理结果
    result = self.post_process_func(output.output)
    return result
```

**流程**：
1. **预处理**：应用模型特定变换
2. **调度**：委托给调度器进行分发
3. **后处理**：将原始输出转换为最终格式（如PIL图像）

## 2. Scheduler

**位置**：`vllm_omni/diffusion/sched/`

### 架构

调度器是**请求状态调度器**，拥有请求生命周期管理和调度决策权，而执行保持在`DiffusionEngine`和执行器中。

### 关键组件

#### 2.1 调度器接口

```python
class SchedulerInterface(ABC):
    def add_request(self, request: OmniDiffusionRequest) -> str: ...
    def schedule(self) -> DiffusionSchedulerOutput: ...
    def update_from_output(
        self,
        sched_output: DiffusionSchedulerOutput,
        output: DiffusionOutput,
    ) -> set[str]: ...
```

**职责**：
- **生命周期契约**：定义引擎如何添加请求、触发一个调度周期并反馈执行器结果
- **稳定边界**：`DiffusionSchedulerOutput`是`DiffusionEngine`消费的唯一调度结果
- **可插拔性**：不同的调度策略可以重用相同的引擎集成路径

#### 2.2 请求状态模型

```python
class DiffusionRequestStatus(enum.IntEnum):
    WAITING = ...
    RUNNING = ...
    PREEMPTED = ...
    FINISHED_COMPLETED = ...
    FINISHED_ABORTED = ...
    FINISHED_ERROR = ...

@dataclass
class DiffusionRequestState:
    sched_req_id: str
    req: OmniDiffusionRequest
    status: DiffusionRequestStatus = DiffusionRequestStatus.WAITING
```

**设计特性**：
- **调度器拥有ID**：每个`OmniDiffusionRequest`由内部`sched_req_id`跟踪，与公共`request_id`值分离
- **显式生命周期**：请求经历等待、运行、可选抢占和终止状态
- **集中式错误处理**：完成、中止和错误状态都在调度器层规范化

#### 2.3 `_BaseScheduler`中的共享簿记

```python
class _BaseScheduler(SchedulerInterface):
    def __init__(self) -> None:
        self._request_states = {}
        self._request_id_to_sched_req_id = {}
        self._waiting = deque()
        self._running = []
        self._finished_req_ids = set()
        self.max_num_running_reqs = 1
```

**设计特性**：
- **通用状态存储**：共享请求映射和等待/运行集位于基类中
- **共享清理逻辑**：请求ID注册、完成处理和状态移除集中而不是在每个策略中重复
- **当前约束**：`max_num_running_reqs`保持为1，因为当前引擎路径仍然是同步请求模式执行

#### 2.4 当前`RequestScheduler`策略

```python
class RequestScheduler(_BaseScheduler):
    def schedule(self) -> DiffusionSchedulerOutput:
        # 1. 在调度结果中保留现有的RUNNING请求
        # 2. 在容量允许时拉取WAITING请求
        # 3. 将新接纳的请求移入RUNNING
```

**行为**：
- **FIFO请求调度**：等待请求按队列顺序提升
- **单请求接纳**：当前策略一次只接纳一个活动请求
- **执行器结果反馈**：`update_from_output()`将执行器输出转换为`FINISHED_COMPLETED`或`FINISHED_ERROR`

#### 2.5 引擎驱动的执行循环

```python
sched_req_id = scheduler.add_request(request)
while True:
    sched_output = scheduler.schedule()
    output = executor.add_req(req)
    finished_req_ids = scheduler.update_from_output(sched_output, output)
```

**设计决策**：
- **关注点分离**：调度器管理状态和策略；执行器处理运行时执行
- **无调度器拥有的IPC**：调度器不再直接与工作进程通信
- **保守并发**：当前请求模式实现仍然只允许一次一个活动请求

## 3. Worker

**位置**：`vllm_omni/diffusion/worker/gpu_worker.py`

### 架构

工作进程是**独立进程**，执行实际的模型推理。每个工作进程在专用GPU上运行，并参与分布式推理。

### 关键组件

#### 3.1 工作进程结构

```python
class WorkerProc:
    def __init__(self, od_config, gpu_id, broadcast_handle):
        # 初始化ZMQ上下文用于IPC
        self.context = zmq.Context(io_threads=2)

        # 连接到广播队列（接收请求）
        self.mq = MessageQueue.create_from_handle(broadcast_handle, gpu_id)

        # 创建结果队列（仅rank 0）
        if gpu_id == 0:
            self.result_mq = MessageQueue(n_reader=1, ...)

        # 初始化GPU工作进程
        self.worker = GPUWorker(local_rank=gpu_id, rank=gpu_id, od_config=od_config)
```

**初始化步骤**：
1. **IPC设置**：创建ZMQ上下文和消息队列
2. **分布式环境设置**：初始化PyTorch分布式通信
   - CUDA GPU：使用NCCL（快速GPU通信）
   - NPU：使用HCCL（华为集体通信库）
3. **模型加载**：在分配的GPU上加载扩散管道
4. **缓存设置**：如果配置了缓存后端则启用

#### 3.2 GPU工作进程

```python
class GPUWorker:
    def init_device_and_model(self):
        # 设置分布式环境变量
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        # 初始化PyTorch分布式
        init_distributed_environment(world_size, rank)
        parallel_config = self.od_config.parallel_config
        initialize_model_parallel(
            data_parallel_size=parallel_config.data_parallel_size,
            cfg_parallel_size=parallel_config.cfg_parallel_size,
            sequence_parallel_size=parallel_config.sequence_parallel_size,
            tensor_parallel_size=parallel_config.tensor_parallel_size,
            pipeline_parallel_size=parallel_config.pipeline_parallel_size,
        )

        # 加载模型
        model_loader = DiffusersPipelineLoader(load_config)
        self.pipeline = model_loader.load_model(od_config, load_device=f"cuda:{rank}")

        # 设置缓存后端
        from vllm_omni.diffusion.cache.selector import get_cache_backend
        self.cache_backend = get_cache_backend(od_config.cache_backend, od_config.cache_config)

        if self.cache_backend is not None:
            self.cache_backend.enable(self.pipeline)
```

**关键特性**：
- **张量并行**：通过PyTorch分布式支持多GPU张量并行
- **模型加载**：使用`DiffusersPipelineLoader`高效加载权重
- **缓存集成**：透明启用缓存后端（TeaCache、cache-dit等）

#### 3.3 工作进程忙循环

```python
def worker_busy_loop(self):
    while self._running:
        # 1. 接收统一消息（生成请求、RPC请求或关闭）
        msg = self.recv_message()

        # 2. 根据类型路由消息
        if isinstance(msg, dict) and msg.get("type") == "rpc":
            # 处理RPC请求
            result, should_reply = self.execute_rpc(msg)
            if should_reply:
                self.return_result(result)

        elif isinstance(msg, dict) and msg.get("type") == "shutdown":
            # 处理关闭消息
            self._running = False

        else:
            # 处理生成请求（OmniDiffusionRequest列表）
            output = self.worker.execute_model(msg, self.od_config)
            self.return_result(output)
```

**执行流程**：
1. **接收**：从共享内存队列出队统一消息
2. **路由**：处理不同消息类型（生成、RPC、关闭）
3. **执行**：对生成请求运行前向传播通过管道
4. **响应**：发送结果（生成为rank 0，RPC为指定rank）

#### 3.4 模型执行

```python
@torch.inference_mode()
def execute_model(self, reqs: list[OmniDiffusionRequest], od_config):
    req = reqs[0]  # TODO: 支持批处理

    # 如果启用则刷新缓存后端
    if self.cache_backend is not None and self.cache_backend.is_enabled():
        self.cache_backend.refresh(self.pipeline, req.num_inference_steps)

    # 设置并行的前向上下文
    with set_forward_context(
        vllm_config=self.vllm_config,
        omni_diffusion_config=self.od_config
    ):
        output = self.pipeline.forward(req)
    return output
```

模型执行利用多个并行策略，这些策略在前向传播期间透明应用。`set_forward_context()`上下文管理器使并行组信息在整个前向传播期间可用：

```python
# 在transformer层内，并行组通过以下方式访问：
from vllm_omni.diffusion.distributed.parallel_state import (
    get_sp_group, get_dp_group, get_cfg_group, get_pp_group
)
```

**优化**：
- **缓存刷新**：在每次生成前清除缓存状态以获得干净状态
- **上下文管理**：前向上下文确保并行组在执行期间可用
- **单请求**：当前一次处理一个请求（批处理TODO）

## 4. Diffusion Pipeline

**位置**：`vllm_omni/diffusion/models/*/pipeline_*.py`

管道是编排扩散过程的**模型特定实现**。不同模型（QwenImage、Wan2.2、Z-Image）有自己的管道实现。

大多数管道实现参考自`diffusers`。多步骤扩散循环通常是整个推理过程中最耗时的部分，由管道类中的`diffuse`函数定义：

```python
def diffuse(self, ...):
    for i, t in enumerate(timesteps):
        # 正向提示的前向传播
        transformer_kwargs = {
            "hidden_states": latents,
            "timestep": timestep / 1000,
            "encoder_hidden_states": prompt_embeds,
        }
        noise_pred = self.transformer(**transformer_kwargs)[0]

        # 负向提示的前向传播（CFG）
        if do_true_cfg:
            neg_transformer_kwargs = {...}
            neg_transformer_kwargs["cache_branch"] = "negative"
            neg_noise_pred = self.transformer(**neg_transformer_kwargs)[0]

            # 合并预测
            comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
            noise_pred = comb_pred * (cond_norm / noise_norm)

        # 调度器步骤
        latents = self.scheduler.step(noise_pred, t, latents)[0]

    return latents
```

**关键特性**：
- **CFG支持**：处理带有单独前向传播的无分类器引导
- **缓存分支**：使用`cache_branch`参数进行缓存感知执行
- **True CFG**：实现带有规范保留的高级CFG

## 5. 加速组件

### 5.1 注意力后端

**位置**：`vllm_omni/diffusion/attention/`

#### 架构

注意力系统使用**后端选择器模式**，根据硬件和模型配置自动选择最佳注意力实现。

#### 后端选择

```python
class Attention(nn.Module):
    def __init__(self, num_heads, head_size, causal, softmax_scale, ...):
        # 自动选择后端
        self.attn_backend = get_attn_backend(-1)
        self.attn_impl_cls = self.attn_backend.get_impl_cls()
        self.attention = self.attn_impl_cls(...)
```

**可用后端**：

| 后端 | 描述 | 可用性 |
|------|------|--------|
| **FlashAttention** | 优化的CUDA内核（FA2/FA3） | 需要`flash-attn`包 |
| **SDPA** | PyTorch的缩放点积注意力 | 始终可用（PyTorch内置） |
| **SageAttention** | 来自SageAttention库的稀疏注意力实现 | 需要`sage-attention`包 |
| **AscendAttention** | 针对Ascend硬件的NPU优化注意力 | 仅在Ascend NPU硬件上可用 |

#### 后端选择机制

```python
def get_attn_backend(head_size: int) -> type[AttentionBackend]:
    # 检查环境变量
    backend_name = os.environ.get("DIFFUSION_ATTENTION_BACKEND")

    if backend_name:
        return load_backend(backend_name.upper())

    # 默认使用SDPA
    return SDPABackend
```

**选择优先级**：
1. **环境变量**：`DIFFUSION_ATTENTION_BACKEND`用于手动覆盖
   - 有效值：`FLASH_ATTN`、`TORCH_SDPA`、`SAGE_ATTN`、`ASCEND`
   - 示例：`export DIFFUSION_ATTENTION_BACKEND=SAGE_ATTN`
2. **自动回退**：如果所选后端不可用则回退到SDPA
3. **硬件检测**：可以根据设备类型选择

### 5.2 并行注意力

**位置**：`vllm_omni/diffusion/attention/parallel/`

#### 架构

并行注意力策略在**注意力层级别实现序列并行（SP）**。这些策略通过分割序列维度将注意力计算分布到多个GPU上，使用不同的通信模式。它们工作在AttentionBackend实现之上，处理并行化/通信，而后端处理实际的注意力计算。

**关键区别**：与AttentionBackend（提供内核实现）不同，ParallelAttentionStrategy为多GPU注意力并行提供通信模式。

#### Ulysses序列并行（USP）

USP是一种序列并行注意力策略，通过分布序列维度和注意力头将注意力计算分割到多个GPU上。它使用**all-to-all通信**来高效并行化非常长序列的注意力。

Ulysses在两个维度上分割注意力计算：
1. **序列维度**：跨GPU分割序列长度
2. **头维度**：跨GPU分割注意力头

**配置**：`ulysses_degree`贡献于`sequence_parallel_size`

#### Ring序列并行

Ring注意力是一种**并行注意力策略**，使用基于环的点对点（P2P）通信实现序列并行。它通过`ParallelAttentionStrategy`接口实现，而不是`AttentionBackend`。

P2P环通信应用于在GPU之间循环Key/Value块。内部根据可用后端使用`ring_flash_attn_func`或`ring_pytorch_attn_func`。

**配置**：`ring_degree`贡献于`sequence_parallel_size`

#### 与AttentionBackend的关系

并行注意力策略（Ring、Ulysses）工作在AttentionBackend实现之上：
- 它们使用AttentionBackend进行实际的注意力计算
- 它们处理多GPU通信/并行化层
- 它们实现`ParallelAttentionStrategy`接口

### 5.3 缓存后端

**位置**：`vllm_omni/diffusion/cache/`

#### 架构

缓存后端提供**统一接口**用于应用不同的缓存策略来加速扩散推理。系统支持多个后端（TeaCache、cache-dit），具有一致的API用于启用和刷新缓存状态。

#### 缓存后端接口

```python
class CacheBackend(ABC):
    def __init__(self, config: DiffusionCacheConfig):
        self.config = config
        self.enabled = False

    @abstractmethod
    def enable(self, pipeline: Any) -> None:
        """在管道上启用缓存。"""

    @abstractmethod
    def refresh(self, pipeline: Any, num_inference_steps: int, verbose: bool = True) -> None:
        """为新生成刷新缓存状态。"""

    def is_enabled(self) -> bool:
        """检查缓存是否启用。"""
        return self.enabled
```

**设计模式**：
- **抽象基类**：定义所有缓存后端的契约
- **基于管道**：与管道实例一起工作
- **状态管理**：为生成之间的干净状态提供刷新机制

#### 可用后端

**1. TeaCache后端**

```python
class TeaCacheBackend(CacheBackend):
    def enable(self, pipeline: Any):
        transformer = pipeline.transformer
        transformer_type = transformer.__class__.__name__

        teacache_config = TeaCacheConfig(
            transformer_type=transformer_type,
            rel_l1_thresh=self.config.rel_l1_thresh,
            coefficients=self.config.coefficients,
        )

        apply_teacache_hook(transformer, teacache_config)
        self.enabled = True

    def refresh(self, pipeline: Any, num_inference_steps: int, verbose: bool = True):
        transformer = pipeline.transformer
        if hasattr(transformer, "_hook_registry"):
            transformer._hook_registry.reset_hook(TeaCacheHook._HOOK_NAME)
```

**TeaCache特性**：
- **时间步感知**：基于时间步嵌入相似性缓存
- **自适应**：动态决定何时重用缓存计算
- **CFG感知**：分别处理正/负分支
- **自定义钩子系统**：使用自定义前向拦截机制（通过`HookRegistry`）

**2. Cache-DiT后端**

```python
class CacheDiTBackend(CacheBackend):
    def enable(self, pipeline: Any):
        # 使用cache-dit库进行加速
        # 支持DBCache、SCM（步骤计算掩码）、TaylorSeer
        # 与单和双transformer架构一起工作
        self.enabled = True

    def refresh(self, pipeline: Any, num_inference_steps: int, verbose: bool = True):
        # 使用新的num_inference_steps更新缓存上下文
```

**Cache-DiT特性**：
- **DBCache**：具有可配置计算块的动态块缓存
- **SCM**：步骤计算掩码以获得额外加速
- **TaylorSeer**：用于缓存准确性的高级校准
- **双transformer支持**：处理像Wan2.2这样有两个transformer的模型

#### 缓存后端选择器

```python
def get_cache_backend(
    cache_backend: str | None,
    cache_config: dict | DiffusionCacheConfig
) -> CacheBackend | None:
    if cache_backend is None or cache_backend == "none":
        return None

    if isinstance(cache_config, dict):
        cache_config = DiffusionCacheConfig.from_dict(cache_config)

    if cache_backend == "cache_dit":
        return CacheDiTBackend(cache_config)
    elif cache_backend == "tea_cache":
        return TeaCacheBackend(cache_config)
    else:
        raise ValueError(f"Unsupported cache backend: {cache_backend}")
```

**使用流程**：
1. **选择**：`get_cache_backend()`返回适当的后端实例
2. **启用**：在工作进程初始化期间调用`backend.enable(pipeline)`
3. **刷新**：在每次生成前调用`backend.refresh(pipeline, num_inference_steps)`
4. **检查**：`backend.is_enabled()`验证缓存是否活动

### 5.4 并行策略

**位置**：`vllm_omni/diffusion/distributed/parallel_state.py`

#### 并行类型

系统支持多个正交并行策略：

**序列并行（SP）**
- **目的**：跨GPU分割序列维度
- **注意力层SP**：Ring注意力和Ulysses在注意力层级别实现SP
- **用例**：非常长的序列（如高分辨率图像）

**数据并行（DP）**
- **目的**：跨GPU复制模型，分割批次
- **用例**：批处理，吞吐量优化

**张量并行（TP）**（实验性）
- **目的**：跨GPU分割模型权重
- **实现**：使用vLLM的张量并行组
- **用例**：不适合单个GPU的大型模型

**CFG并行**（开发中）
- **目的**：并行化无分类器引导（正/负提示）
- **基础设施**：CFG并行组已初始化，可通过`get_cfg_group()`获取

#### 并行组管理

```python
def initialize_model_parallel(
    data_parallel_size: int = 1,
    cfg_parallel_size: int = 1,
    sequence_parallel_size: int | None = None,
    ulysses_degree: int = 1,
    ring_degree: int = 1,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    vae_parallel_size: int = 0,
):
    # 生成正交并行组
    rank_generator = RankGenerator(
        tensor_parallel_size,
        sequence_parallel_size,
        pipeline_parallel_size,
        cfg_parallel_size,
        data_parallel_size,
        "tp-sp-pp-cfg-dp",
    )

    # 初始化每个并行组
    _DP = init_model_parallel_group(rank_generator.get_ranks("dp"), ...)
    _CFG = init_model_parallel_group(rank_generator.get_ranks("cfg"), ...)
    _SP = init_model_parallel_group(rank_generator.get_ranks("sp"), ...)
    _PP = init_model_parallel_group(rank_generator.get_ranks("pp"), ...)
    _TP = init_model_parallel_group(rank_generator.get_ranks("tp"), ...)
```

**Rank顺序**：`tp-sp-pp-cfg-dp`（张量 → 序列 → 管道 → cfg → 数据）

## 6. 数据流

### 完整请求流

```
1. 用户请求
   └─> OmniDiffusion.generate(prompt)
       └─> 准备OmniDiffusionRequest
           └─> DiffusionEngine.step(requests)

2. 预处理
   └─> pre_process_func(requests)
       └─> 模型特定变换

3. 调度
   └─> scheduler.add_request(request)
       └─> scheduler.schedule()
           └─> DiffusionEngine将调度的请求提交给executor.add_req(req)

4. 工作进程执行
   └─> WorkerProc.worker_busy_loop()
       └─> GPUWorker.execute_model(reqs)
           └─> Pipeline.forward(req)
               ├─> encode_prompt()
               ├─> prepare_latents()
               ├─> diffuse() [循环]
               │   ├─> transformer.forward() [带缓存后端钩子]
               │   └─> scheduler.step()
               └─> vae.decode()

5. 结果收集
   └─> Executor返回DiffusionOutput
       └─> scheduler.update_from_output(...)
           └─> DiffusionEngine弹出完成的请求状态

6. 后处理
   └─> post_process_func(output)
       └─> 转换为PIL图像/最终格式
```
