## Initialization

当前运行时已经分成两个边界：主进程中的 `StageClient`，以及设备资源所属的 worker 进程。`StageEngine` 在 worker 内部创建，真正持有模型、KV cache 和执行资源。

In `vllm_rlt/entrypoints/llm.py`

```text
LLM(model, ...)
  -> model 是路径/仓库 ID：resolve_model_source + resolve_model_config
  -> StageClient(model, engine_options...)
  -> StageClient.spawn StageWorker
  -> StageWorker 加载模型并创建 StageEngine
  -> 等待 ready 消息
```

`LLM` 只负责公开 Python API、tokenizer 和外部 request ID。文本 prompt 的 tokenizer 可以在第一次 `generate()` 时延迟加载；字符串模型路径的权重由 worker 进程加载，显式 `Module` 则先复制到 CPU 后交给 worker，不在主进程创建第二个 runner。

### Model

In `vllm_rlt/models/__init__.py`

```text
AutoModelForCausalLM.from_pretrained(path_or_repo)
  -> resolve_model_source(...)
  -> resolve_model_config(...) 读取 config.json
  -> MODEL_MAPPING[model_type]
  -> model_cls.from_pretrained(resolved_path, revision, device, dtype)
```

当前映射包括 `ouro`、`nanbeige` 和 `huginn_raven`。模型执行按三种 operation 组织：

```text
PRELUDE   token IDs -> hidden
RECURRENT hidden + KV metadata -> hidden (+ optional gate)
CODA      hidden (+ boundary KV) -> logits
```

Ouro/Nanbeige 使用普通 prelude/recurrent/coda；Huginn 声明 `requires_boundary_kv = True`，使用 prepared 接口。Huginn 还要求 `last_exited` KV、atomic prefill、同步调度，并暂不支持 speculative decoding。

### LLMEngine/StageEngine

In `vllm_rlt/engine/llm_engine.py`

```text
LLMEngine = StageEngine  # 兼容名称
StageClient              # 主进程中的控制/生命周期 owner
StageWorker              # worker 进程中的资源 owner
```

`StageClient` 通过 `multiprocessing.Pipe` 发送 control command，维护：

|字段|作用|
|---|---|
|`requests`|外部 request ID -> 当前 execution generation|
|`generations`|generation -> 外部 request ID，防止旧结果写入新请求|
|`outputs`|worker 返回但尚未被 `step()` 取走的输出|
|`pending_controls` / `replies`|异步 control 调用及其 reply|
|`process` / `channel`|StageWorker 进程和控制通道|

In `vllm_rlt/engine/worker.py`

```text
StageWorker(...)
  -> 设置 device、dtype、torch seed
  -> 在 worker 内加载 model
  -> StageEngine(model, stage_config, engine_options)
  -> 创建可选 TransferConnector
  -> engine.connector = connector
  -> 发送 ready(model/cache/stage 信息)
  -> command loop + connector.progress() + engine.step()
```

`StageConfig.role` 当前为 `prefill`、`decode` 或 `prefill_decode`。它派生 `accepts_prompt` 与 `produces_tokens`，用于入口检查和 Scheduler policy 选择；它不表示一次 model operation。

`StageEngine.__init__()` 的主要顺序：

```text
StageEngine(model, configs..., stage_config, connector)
  -> Profiler(device)
  -> defaults + compatibility checks
  -> create_backend(...)
  -> plan_cache(...) -> KVCacheManager(...)
  -> Scheduler(...)
  -> connector readiness 接入 Scheduler
  -> RngRegistry -> ModelRunner(...)
  -> 可选 SpeculativeRunner / AsyncSpeculativeRunner
  -> Executor(...)
  -> PreemptionManager(...)
```

|配置|初始化约束|
|---|---|
|CUDA Graph|需要 CUDA 和 backend 的 `cuda_graphs` 能力|
|Packed prefill|`prefill_uva=True` 时需要 CUDA、`last_exited` 和 `packed_prefill`|
|Split prefill|`atomic_prefill=False` 时要求 `last_exited`|
|Async scheduling|CUDA backend 需要 `async_scheduling` 能力|
|Async speculative|CUDA、`device_context_lengths` backend 能力、无 preemption、refill policy、固定 target depth|
|Speculative request|请求的 `max_loops` 等于 target depth，`exit_threshold=1`；async 版本还要求 greedy sampling|

**Executor (`vllm_rlt/worker/executor.py`)**

|成员|关系|
|---|---|
|`runner`|引用 `ModelRunner`|
|`speculation`|引用可选的 speculative runner|
|`submit(SchedulerOutput)`|普通 batch 转成 `ExecutionBatch`；speculative batch 直接交给 speculative runner|

```text
SchedulerOutput
  ├─ is_speculative=True
  │    -> SpeculativeRunner.submit(SchedulerOutput)
  └─ is_speculative=False
       -> build_execution_batch(...)
       -> ModelRunner.submit(SchedulerOutput, ExecutionBatch)
```

`Operation`、`StateSpan`、`ExecutionBatch` 当前统一定义在 `vllm_rlt/request.py`。

|`build_execution_batch()`|内容|
|---|---|
|phase -> operation|`PREFILL/PREFILLING_PRELUDE/PRELUDE` -> PRELUDE；`PREFILLING_RECURRENT/RECURRENT` -> RECURRENT；`CODA` -> CODA|
|位置快照|`token_start`、`token_count`、depth、chunk 起点、prompt completion|
|loop boundary|`PREFILL` 使用完整 depth；RECURRENT 使用一轮|
|gate|`RECURRENT` span 可设置 `emit_gate`|
|校验|同一 batch 必须共享 operation 和 execution boundary|

#### Profiler

In `vllm_rlt/profiling.py`

```text
Profiler(device)
  -> 保存 device
  -> current = None
  -> 第一次 start() 时创建 session
```

`StageClient`/`LLM` 通过 `start_profile()`、`stop_profile()`、`profile_status()` 和 `wait_for_profile_artifacts()` 调用 worker 内的 profiler。

#### Attention Backend

In `vllm_rlt/attention/backend.py`

```text
create_backend(backend, device, dtype, head_dim, block_size,
               required_capabilities)
  -> validate_backend_name()
  -> 创建候选 AttentionBackend
  -> 检查 packed_prefill / cuda_graphs /
     async_scheduling / device_context_lengths
  -> 返回 kernel + capabilities + selection info
```

|选择方式|行为|
|---|---|
|显式 backend|初始化或能力检查失败直接报错，不 fallback|
|`auto` + CUDA|FA4 -> FA3 -> FA2 -> Triton -> Torch|
|`auto` + CPU|直接使用 Torch reference attention|
|运行期 kernel 错误|不触发 fallback|

`AttentionBackend.__call__()` 服务普通 paged attention，`prefill()` 服务 packed prefill。当前 FA4 提供 `packed_prefill`；Triton/FA2/FA3 也声明 async scheduling 和 device context length 能力。

| |Packed prefill|普通 paged 路径|
|---|---|---|
|Query|连续 token rows 按 sequence 拼成 ragged batch|每个 query row 使用自己的 context length/page table|
|Metadata|`cu_seqlens_q`、sequence key、packed table|逐 row block table/context length|
|KV|仍由 KVCacheManager 提供 request/depth page|仍由 KVCacheManager 提供 request/depth page|
|适用|需要 backend packed capability；要求 sequence rows 连续且每个 request/depth 只形成一个 sequence|实现更通用，支持普通单 query rows|

#### KVCache

In `vllm_rlt/core/memory.py:plan_cache(...)`

```python
bytes_per_block = (
    2                         # K 和 V
    * config.num_hidden_layers
    * cache.block_size
    * config.num_key_value_heads
    * config.head_dim
    * parameter.element_size()
)
```

```text
block 数来源：
  1. cache.num_blocks
  2. cache.kv_cache_memory_bytes / bytes_per_block
  3. CPU 默认 256
  4. CUDA profile：最大计划 batch 的 prelude + recurrent + coda
     - peak/concurrent execution
     - execution_buffer_bytes()
     - metadata / context scratch
     - memory reserve / graph reserve
```

`execution_buffer_bytes()` 仍是启动时的 KV budget 估算，不是 allocator trace。当前估算包含：

|类型|内容|
|---|---|
|static buffers|Workspace、hidden、page-table metadata|
|async buffers|routing buffers、device state、typed sparse-write descriptors|
|split prefill|`prefill_states`、`prefill_hidden`、`BatchInputBuffers`|
|submission reuse|`MAX_CONCURRENT_SUBMISSIONS` 深度的 Uva/metadata staging|

#### KVCacheManager

In `vllm_rlt/core/kv_cache_manager.py`

```python
shape = (num_blocks, num_layers, block_size, num_kv_heads, head_dim)
key_cache = torch.empty(shape)
value_cache = torch.empty_like(key_cache)
metadata_builder = MetadataBuilder(self)
```

|配置|含义|
|---|---|
|`last_exited`|每个 token block 按 recurrence depth 保存 plane；`storage_depths=max_loops`|
|`shared`|所有 depth 共用一个 plane；`storage_depths=1`|
|incremental allocation|先分配当前 frontier，后续由 Scheduler/Manager 增长 page|
|prefix caching|只允许 `last_exited`；通过完整 page 共享 KV，不共享最终 CODA hidden|

`KVCacheManager` 负责 allocation/free/growth、page tables、written-prefix 校验、prefix cache、transfer lease，以及 `KVMetadata`/`DeviceKVMetadata` 所需的 allocation 事实。`MetadataBuilder` 只构造 metadata，不接管 KV ownership。

#### Scheduler

In `vllm_rlt/core/scheduler.py`

```text
Scheduler(config, cache_manager, speculative_config, ...)
  -> requests: request_id -> Request
  -> queues: RequestPhase -> deque[request_id]
  -> SchedulerOutput(phase, ScheduledRequest[])
  -> scheduling policy
```

当前调度单位是 `RequestPhase`，定义在 `vllm_rlt/request.py`：

|Phase|含义|
|---|---|
|`WAITING`|尚未完成 KV admission|
|`RECEIVING` / `TRANSFERRING`|跨 stage import/export 生命周期|
|`PREFILL`|atomic prompt prefill|
|`PREFILLING_PRELUDE`|split prompt chunk 的 token -> hidden|
|`PREFILLING_RECURRENT`|split prompt chunk 的单轮 recurrent|
|`PRELUDE`|decode token 的 prelude|
|`RECURRENT`|decode 单轮 recurrent|
|`SPECULATIVE`|本地 draft/verify round|
|`CODA`|hidden -> sampled token|
|`FINISHED`|逻辑完成；最终资源由 finish/retirement 路径释放|

`SchedulerOutput.phase=None` 表示 mixed batch；每个 `ScheduledRequest` 自带 phase。`ScheduledRequest` 同时快照 token range、depth、chunk 起点和 completion 标记，执行层不再从可变 Request 状态反推已提交位置。

|配置|Policy|
|---|---|
|prefill-only stage|`PromptOnlyPolicy`|
|atomic + refill/no-refill|`RefillPolicy` / `NoRefillPolicy`|
|split prefill|`SplitPrefillPolicy`|
|mixed|`MixedPolicy`，只在非 atomic、非 speculative 时允许|
|speculative|`SpeculativePolicy`|

> [!info] Atomic Prefill vs Split Prefill
> - `atomic_prefill=True`：队列进入 `PREFILL`，一次 `ExecutionBatch` 执行 prompt prelude 和完整 recurrent depth。
> - `atomic_prefill=False`：依次进入 `PREFILLING_PRELUDE` 与 `PREFILLING_RECURRENT`；Runner 跨调度边界保留 staged hidden。

> [!info] RequestPhase 与 Operation
> `RequestPhase` 描述 Scheduler 的请求工作；`Operation` 描述 ModelRuntime 的一次模型调用。Executor 负责把前者绑定为 `ExecutionBatch`，两者不再混用。

#### SpeculativeRunner

In `vllm_rlt/worker/speculative.py`

`SpeculativeRunner` 与 `ModelRunner` 共享 model、KV cache、`ModelRuntime` 和 Engine 的 `RngRegistry`；普通版本复用 `core_stream`，异步版本由 `AsyncSpeculativeRunner` 扩展。

```text
RequestPhase.SPECULATIVE
  -> Executor.submit()
  -> draft: PRELUDE -> draft_loops * RECURRENT -> CODA
  -> verify: remaining target loops -> CODA
  -> acceptance
  -> SpeculativeResult
  -> StageEngine commit token / stop check / KV frontier update
```

|版本|自有状态/资源|
|---|---|
|普通|`SpeculativeSubmission`、draft proposals、target logits、可选 graph|
|异步|2 个 `SpeculativeRoundBuffer`、`SpeculativeRequestState`、compute/copy events、device token/position|
|异步并发|最多两个未退休 round；`copy_stream` 完成 compact result readback 后 ticket 才可 collect|
|异步限制|CUDA、Triton/device-context-length backend、无 CUDA Graph；request 需 fixed target depth、`exit_threshold=1` 和 greedy sampling|

#### ModelRunner

In `vllm_rlt/worker/model_runner.py`

```text
ModelRunner(model, cache_manager, configs...)
  -> model.eval() + cache_manager.metadata_builder
  -> ModelRuntime + Sampler + shared RngRegistry
  -> optional RecurrentGraphs / CodaGraphs
  -> staged input buffers + packed metadata buffers
  -> SubmissionQueue + request state/readback slots
  -> CUDA streams + optional AsyncState
  -> optional split-prefill hidden/input buffers
```

**ModelRuntime (`vllm_rlt/worker/executor.py`)**

```text
ForwardBatch
  -> prepare(): 选择 graph 或 eager fallback，并准备 graph calls
  -> forward(): 执行 PRELUDE / RECURRENT / CODA
  -> ForwardResult(hidden, gate_logits, logits)
```

|Operation|处理|
|---|---|
|PRELUDE|从 tokens 创建 hidden；boundary-KV 模型使用 `prelude_coda_kv_metadata`|
|RECURRENT|遍历 metadata；graph path 逐 call replay，eager path 调 `model.recurrent_prepared`|
|CODA|调用 `model.coda` 或 `coda_prepared`；可使用 `CodaGraphs`|

`ModelRunner.submit()` 分为两段准备：

```text
prepare_on_cpu(ExecutionBatch)
  -> MetadataBuilder.build()
  -> StagedInputTensor / BatchInputBuffers / PackedMetadataBuffer 写入 host staging

prepare_on_gpu(PreparedExecution)
  -> 上传 input buffers / metadata
  -> 绑定 AsyncState routing 或 Workspace
  -> ModelRuntime.prepare()
  -> 在 core/boundary stream 执行
```

**Streams / buffers**

|资源|作用|
|---|---|
|`core_stream`|带 recurrent loop 的计算|
|`boundary_stream`|无 loop 的 PRELU/CODA 和 KV finalize；`multi_stream=False` 时与 core 复用|
|`copy_stream`|AsyncState routing、host/device metadata 和 async speculative result transfer|
|`readback_stream`|将 sampling/gate result 复制到 pinned host slot|
|`StagedInputTensor`|按 dtype 组织可增长 CPU staging，并用 UVA 或 async copy 上传|
|`PackedMetadataBuffer`|把 packed sequence IDs、lengths、cumulative lengths 和 page tables 压缩后上传|
|`SubmissionQueue`|最多保留两个 in-flight submissions；统一决定何时可复用输入/metadata pool|
|`AsyncState`|设备侧 hidden/token slots、page tables、routing buffers、finalize buffers|
|split-prefill buffers|`prefill_states`、`prefill_hidden`、`BatchInputBuffers`、prompt token buffers|

request 的最新 compute/finalize event 写入 `ModelRunner.events`；复用 state slot、workspace、staged tensor 或 readback slot 前，必须先由 submission/event 完成其 GPU reader。

### PreemptionManager

In `vllm_rlt/engine/preemption.py`

|成员|作用|
|---|---|
|`engine`|访问 StageEngine、Scheduler、ModelRunner、KVCacheManager|
|`snapshots`|按 request ID 保存 phase、KV snapshot、CPU hidden、CPU input token|
|`suspend_callback`|Scheduler 选 victim 后调用；队列迁移仍由 Scheduler 完成|
|`resume_callback`|Scheduler admission 时尝试恢复 snapshot|

**preempt/suspend：可运行 phase -> WAITING**

```text
Scheduler.preempt(requesting_request)
  -> 选择安全 victim
  -> PreemptionManager.suspend(victim)
       -> synchronize ModelRunner
       -> rewind split-prefill phase
       -> snapshot(phase, KV, hidden, input token)
       -> ModelRunner.suspend()  # 保留 RNG
       -> free KV / 清空 device refs
  -> Scheduler.enqueue(victim, WAITING)
```

|victim 检查|条件|
|---|---|
|phase|不能是 `WAITING` 或 `RECEIVING`|
|输出|`num_output_placeholders == 0`|
|保护集合|不能是当前 requesting request 或本批 `excluded` request|
|transfer|不能持有 transfer lease|
|priority-only|只允许抢占 priority 数值更大的 request|
|排序|priority 优先，再按 `required_blocks(position + 1)` 选择 footprint 更大的 request|

**resume：WAITING -> snapshot 中的原 phase**

|返回值|含义|
|---|---|
|`None`|没有 snapshot，继续普通 admission|
|`False`|snapshot 存在但 KV 暂不可分配，保留 WAITING|
|`RequestPhase`|KV/state 恢复成功；Scheduler 将 request 重新 enqueue 到该 phase|

```text
resume(request)
  -> allocate KV frontier
  -> restore KVSnapshot
  -> restore hidden/input token to cache device
  -> CUDA current stream synchronize
  -> return saved RequestPhase
```

挂起只释放 Runner slot，保留 request RNG；终止路径调用 `ModelRunner.release()` 并删除 RNG generator。`discard_snapshot()` 用于取消不再恢复的 request。
