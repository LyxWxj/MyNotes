---
type: Note
---

# Travel Through vLLM-rlt

> **代码基准 Commit：** `ce200f0d2085fb041c66691245e29aef2a9ed714`（提交时间：`2026-11-20T18:26:24+08:00`）
>
> **笔记核对时间：** `2026-11-23 13:40:20 +0800`（Asia/Shanghai）
>
> **范围：** 当前本地检出；已跟踪文件相对上述 commit 无修改。本地还有未跟踪文件，见目录树下方说明。

## 仓库目录结构

以下按当前 commit 的已跟踪内容整理；目录注释描述主要用途，文件名只列阅读入口。

```text
vllm-rlt/
├── vllm_rlt/                  # 引擎 Python 包；__init__.py 导出公共 API
│   ├── attention/             # attention backend 选择、能力声明与元数据
│   ├── core/                  # Scheduler/policy、KVCacheManager 和 KV 容量规划
│   ├── engine/                # StageClient、StageWorker、StageEngine、connector、抢占
│   ├── entrypoints/           # 离线 LLM/CLI、单 stage/PD 服务入口、公共参数和 chat 模板
│   ├── kernels/               # Torch/Triton/Flash/Ascend attention、KV、routing 等算子
│   ├── layers/                # 模型共用的 normalization 和 rotary embedding
│   ├── models/                # Ouro、Nanbeige、Huginn、Dory 与 ModelConfig/加载分发
│   ├── pd/                    # prefill/decode coordinator、NIXL connector/transport 和配置
│   ├── serving/               # HTTP 协议、aiohttp server、请求 channel 和 engine 线程
│   ├── worker/                # Executor/ModelRuntime、runner、sampler、graphs、buffers、自推测
│   ├── config.py              # cache、scheduler、exit、execution、speculative 配置
│   ├── request.py             # Request/phase、执行描述和 RequestOutput
│   ├── sampling_params.py     # 每请求采样、循环深度及优先级参数
│   ├── device.py              # CPU/CUDA/NPU 设备解析
│   └── profiling*.py          # profiler 会话及 trace/artifact 处理
├── tests/                     # 引擎、scheduler、服务、模型、后端、设备等回归测试
│   └── reference/             # 独立的小模型和 Dory/Huginn 参考实现
├── benchmarks/                # 运行时、抢占、GSM8K 精度评估及相关后端
│   ├── fixtures/              # 固定 GSM8K 子集等 benchmark 输入
│   └── kernels/               # 独立 kernel benchmark
├── csrc/                      # 可选 PyTorch C++/CUDA 扩展与绑定
│   └── activation/            # SiLU-and-Mul CUDA 实现
├── docs/                      # 架构、scheduler、KV、服务、精度和 profiling 文档
│   ├── assets/                # 文档图片、性能图与其他静态资源
│   │   ├── inference/         # Ouro 推理示意图和动画
│   │   ├── performance/       # 已记录的 workload 性能图
│   │   └── runtime/           # runtime 图及生成脚本
│   ├── diagrams/              # stage/scheduler 等 PNG、SVG 和索引
│   └── recipes/               # Ascend、Ouro-Thinking 等特定运行方案
├── requirements/              # 可选后端的额外依赖约束
├── .github/                   # GitHub 贡献模板与 CI
│   ├── ISSUE_TEMPLATE/        # issue 分类模板
│   └── workflows/             # CPU 测试工作流
├── .agents/skills/            # 随仓库维护的专项 agent 工作流
│   ├── rlt-perf-opt/          # 性能分析与优化流程
│   ├── rlt-refactor/          # 推理运行时重构规则
│   └── vllm-rlt-review-pr/    # PR 审查规则
│       ├── agents/            # agent 配置
│       └── references/        # 审查证据、接口和测试规则
├── README.md                  # 项目概览、安装及模型支持
├── pyproject.toml             # Python 依赖、可选项、CLI、pytest/Ruff 配置
├── setup.py / MANIFEST.in     # 本地扩展构建和打包文件
└── CHANGELOG.md / CONTRIBUTING.md / LICENSE / NOTICE
```

## Initialization

当前运行时有三种初始化入口。它们最终都汇聚到 worker 内的

`StageEngine`，但外层 owner 不同。

**1. CLI 启动单 stage 在线服务：`vllm_rlt.entrypoints.serve`**

```text
serve.main()
  -> 解析 CLI + runtime_configs()
  -> create_app(partial(load_engine, args))
  -> EngineWorker.start()
  -> load_engine()
       -> resolve_model_config() + load_tokenizer()
       -> StageClient(...)
       -> start_stage_worker()
       -> StageWorker -> StageEngine
       -> 等待 ready
  -> EngineWorker 持有 tokenizer/engine，单线程驱动 engine.step()
  -> HTTP RequestChannel 接收请求并输出 SSE/JSON
```

**2. CLI 启动 PD 在线服务：`vllm_rlt.entrypoints.pd_serve`**

```text
pd_serve.main()
  -> create_app(partial(load_engine, args))
  -> load_engine()
       -> resolve_model_config() + load_tokenizer()
       -> PDEngine(...)
            -> prepare_model() + PDConfig
            -> 为每个 prefill/decode device 创建 StageClient
            -> 每个 StageClient 启动 StageWorker -> StageEngine
            -> connector_factory -> PDStageConnector -> NixlConnector
            -> 等待所有 ready
            -> 校验 model/KV/attention fingerprint
            -> 发送 connect，建立 P/D peer
  -> EngineWorker 持有 PDEngine/tokenizer，驱动 PD coordinator
```

**3. 离线推理：`LLM` / `vllm_rlt.entrypoints.cli`**

```text
LLM(model, ...)
  -> prepare_model(model, revision)
       -> resolved source / revision / read-only ModelConfig
  -> StageClient(model, model_config, engine_options...)
  -> start_stage_worker(...)
	  -> StageWorker 加载或构造 model，并创建 StageEngine
	  -> 等待 ready
	  -> generate()
	       -> tokenizer.encode()（需要时延迟加载）
	       -> StageClient.add_request()
	       -> LLM.generate() 调用 StageClient.step() 轮询 worker 输出
```

| 边界          | 初始化 owner                            | 主要资源                                                    |
| ----------- | ------------------------------------ | ------------------------------------------------------- |
| HTTP 在线层    | `EngineWorker` + event loop          | `RequestChannel`、tokenizer、流式输出、请求并发限制                  |
| PD 部署层      | `PDEngine`                           | P/D `Peer`、`Pending`、transfer 状态、多个 `StageClient`       |
| stage 控制层   | `StageClient`                        | worker process、`Pipe`、外部 ID/generation、control reply    |
| stage 资源层   | `StageWorker` / `StageEngine`        | model、KV cache、Scheduler、Runner、streams、connector       |
| 跨 stage 传输层 | `PDStageConnector` / `NixlConnector` | registered KV/hidden memory、transfer lease、NIXL handles |

字符串模型路径的权重在 worker 中加载；显式 `Module` 会先复制到 CPU 后交给 worker。`StageEngine` 始终在资源所属 worker 内创建，主进程或 HTTP 线程不持有第二个 model runner。

#### Model

代码入口：`vllm_rlt/models/config.py`、`vllm_rlt/models/__init__.py` 和各模型实现。

```text
prepare_model(model, revision, model_config)
  -> 路径/仓库 ID：resolve_model_config(...)
  -> config.json -> ModelConfig.from_dict(...)
  -> Module/具体 config：包装为 ModelConfig
  -> 返回 source、revision、ModelConfig

AutoModelForCausalLM.from_pretrained(source, model_config=...)
  -> ModelConfig.model_type -> MODEL_MAPPING[model_type]
  -> model_cls.from_pretrained(..., config=model_specific_config, device, dtype)
```

当前映射包括 `ouro`、`nanbeige`、`huginn_raven` 和 `dory`。`ModelConfig` 持有具体模型配置及其派生限制，在 worker 之间按只读约定传递。Hub 模型优先使用完整的本地 revision 快照；缺少文件时会要求交互式确认下载。

模型执行按三种 operation 组织；一个 submission 可以只执行其中一种，也可以把 PRELUDE 和若干次 RECURRENT 合并：

```text
PRELUDE   token IDs -> hidden（某些模型还读写 boundary KV）
RECURRENT hidden + depth/KV metadata -> hidden (+ optional gate)
CODA      hidden (+ boundary KV) -> logits -> sampling
```

|模型|与执行和 KV 有关的特点|
|---|---|
|Ouro|普通 prelude/recurrent/coda，具有训练过的 early-exit gate；prompt prefill 始终跑完整深度。|
|Nanbeige|固定循环模型，当前实现没有 early-exit gate；配置中的 `total_ut_steps` 决定完整 prompt 深度。|
|Huginn|`requires_boundary_kv=True`，PRELUDE/CODA 也有注意力和 KV；要求 `last_exited`、同步调度，不支持 mixed/WCPB、prefix caching、自推测或 PD；支持部分 decode CUDA Graph。|
|Dory|`requires_boundary_kv=True`，输入、循环、输出阶段均有注意力 KV；只有注意力子层占 KV，使用 `num_kv_layers` 紧凑编号，循环状态还携带 input injection。需要 sliding-window 能力。|

Dory 的 `recurrent_kv_cache_mode=per_loop` 对应 `last_exited`，`last_loop` 对应 `shared`；boundary KV 使用 depth 0。Dory 没有训练过的退出 gate，标记 `fixed_loop_count=True`：普通请求的 `target_loops` 默认为模型完整深度，当前低于完整深度会因为尚未支持逐请求 prompt 深度而被拒绝；`max_loops` 若提供必须与 target 匹配，`min_loops` 接受默认值或 target。`trace` 模式有单独的重放规则。

Dory 的普通同步/异步执行和 decode CUDA Graph 已有实现；同步 eager 自推测需要 `per_loop`、refill 和固定完整深度，block 支持 greedy/采样，pipeline 支持 greedy。boundary-KV 自推测仍拒绝 async、CUDA Graph 和 preemption；mixed/WCPB、prefix caching、PD 不在支持范围。普通 Dory preemption/snapshot 的 boundary 状态尚未完成专项验证。见 `docs/dory.md`、`docs/runtime.md`、`tests/test_dory*.py`。

> [!note] 文档新旧
> 当前 `README.md` 与 `docs/README.md` 的部分 Dory 概述仍保留更早的限制描述；`docs/runtime.md` 也仍把异步自推测写成仅支持 greedy，而 `docs/async-speculative.md`、实现和测试已包含 async block 随机采样。阅读时以本提交的实现和对应专项测试为准。实现存在或某个测试通过，不等于所有 checkpoint、dtype、batch shape 和硬件组合都已完成精度与性能验证。

### StageClient

In `vllm_rlt/engine/llm_engine.py`

```text
StageClient():
	...
	-> self.process, self.channel = start_stage_worker()
		-> process = context.Process(
				target=worker_main,
				args=(...)
				kwargs=dict(...)
				name=f"vllm-rlt-{stage_config.stage_id}"
			)
			-> worker_main(...)
				owner=StageWorker(...)
				owner.run() # 处理控制消息、推进 connector 和本地引擎；空闲时短暂休眠
			返回worker_main的入口进程和管道
	-> wait_ready()  # wait_for_ready 是构造参数；实际等待方法是 wait_ready
```

`StageClient` 通过 `multiprocessing.Pipe` 发送 control command，维护：

|字段|作用|
|---|---|
|`model_config`|入口阶段解析出的只读具体模型配置|
|`requests`|外部 request ID -> 当前 execution generation|
|`generations`|generation -> 外部 request ID，隔离旧结果|
|`outputs`|worker 返回但尚未被 `step()` 取走的输出|
|`pending_controls` / `replies`|异步 control 调用及其 reply|
|`process` / `channel`|StageWorker 进程和控制通道|

### StageWorker

In `vllm_rlt/engine/worker.py`

```text
StageWorker(...)
  -> 设置 device、dtype、torch seed
  -> Module：移动到 worker device
     具体模型 config（dataclass）：构造对应模型
     source：AutoModelForCausalLM.from_pretrained(..., model_config)
  -> StageEngine(model, stage_config, engine_options)
  -> 创建可选 TransferConnector，并接入 engine
  -> 发送 ready(cache/stage/connector 容量信息)
  -> command loop + connector.progress() + engine.step()
```

`ready` 消息只报告运行时容量和 transport 信息；具体模型配置由

`StageClient` 保留，不再从 worker 消息反序列化。

`StageConfig.role` 当前为 `prefill`、`decode` 或 `prefill_decode`。它派生

`accepts_prompt` 与 `produces_tokens`，用于入口检查和 Scheduler policy 选择；

它不表示一次 model operation。

### StageEngine

`StageEngine.__init__()` 的主要顺序：

```text
StageEngine(model, configs..., stage_config, connector)
  -> Profiler(device)
  -> defaults + resolve_cuda_graph_rows(...)
  -> requires_staged_prefill(scheduler_config, stage role)
  -> model/layout/scheduler compatibility checks
  -> create_backend(...)
  -> plan_cache(..., speculative, staged_prefill)
  -> KVCacheManager(...)
  -> Scheduler(..., total_ut_steps, eos_token_id, exit_mode)
  -> connector readiness/export predicate 接入 Scheduler
  -> shared RngRegistry -> ModelRunner / AscendModelRunner(..., staged_prefill)
  -> SpeculativeRunner 或 AsyncSpeculativeRunner
  -> Executor(...)
  -> PreemptionManager(...)
```

|配置/条件|初始化约束|
|---|---|
|staged prefill|`wavefront_chunked_prefill=True`，或 token-producing stage 使用 `mode="mixed"`；要求 `last_exited` KV。boundary-KV 模型当前拒绝 staged prefill。|
|CUDA Graph|需要 CUDA 和 backend 的 `cuda_graphs` 能力；显式过小的 row limit 使超出范围的 batch 走 eager。|
|自动 graph row limit|`cuda_graph_max_batch_size=None` 时普通路径为 128；异步 speculative 为 `max(128, min(B, C*(K+1)))`，`C=max_num_seqs`、`B=max_num_batched_tokens`、`K=num_speculative_tokens`。|
|packed prefill|`packed_prefill=None` 自动选择，`True` 表示必须支持，否则初始化失败，`False` 禁用；与是否使用 UVA 的 `use_uva` 独立。CUDA 自动路径通常要求 `last_exited` 和普通非 staged prefill；Ascend 有自己的 host metadata 规则。|
|async scheduling|CUDA/NPU backend 需要相应 `async_scheduling` 能力；非 speculative 且不标记 `fixed_loop_count` 的模型需要 `ouro_delayed`、`random_lookahead` 或 `trace`。|
|speculative|要求 `last_exited`、`exit_mode="ouro"`、`target_loops == model.total_ut_steps` 和 `mode="refill"`；请求必须固定 target depth 且 `exit_threshold=1`。boundary-KV 模型还必须显式支持自推测。|
|async block speculative|要求 CUDA、backend 的 `device_context_lengths`，且禁止 preemption；支持 greedy 和 temperature/top-k/top-p 采样，并可在同一 batch 混合。|
|pipeline speculative|`target_loops % draft_loops == 0`，请求必须 greedy；异步 pipeline 可使用 CUDA Graph 和 preemption，仍受模型特定限制。|
|Huginn / Dory|二者都不支持 staged prefill、prefix caching 或 PD；Huginn 拒绝 async 和自推测，Dory 的普通 async/Graph 与同步 eager 自推测分别有不同约束。|
|Ascend NPU|当前仅 Ouro 四层循环、BF16 权重和 KV，单 stage、本地调度；要求显式 KV 容量，支持固定深度 greedy 和可选普通 async；不支持 graph、speculative、PD、preemption 或 prefix caching。|

#### Executor (`vllm_rlt/worker/executor.py`)

|成员|关系|
|---|---|
|`runner`|引用普通 `ModelRunner`|
|`speculation`|引用可选的 speculative runner|
|`submit(SchedulerOutput)`|speculative batch 交给 round controller；普通 batch 先绑定 `ExecutionBatch`|

```text
SchedulerOutput
  ├─ phase == SPECULATIVE
  │    -> SpeculativeRunner.submit(SchedulerOutput)
  └─ ordinary / mixed
       -> build_execution_batch(...)
       -> ModelRunner.submit(SchedulerOutput, batch=ExecutionBatch)
```

`build_execution_batch()` 的绑定：

|内容|来源|
|---|---|
|phase -> operation|`PREFILL/PREFILLING_PRELUDE/PRELUDE` -> PRELUDE；`PREFILLING_RECURRENT/RECURRENT` -> RECURRENT；`CODA` -> CODA|
|位置和深度|`token_start`、`token_count`、`depth`、`chunk_start`、`prefill_hidden_row_offset`|
|loop boundary|优先使用 `SchedulerOutput.loop_count`；直接调用时 PREFILL 为完整 depth、RECURRENT 为一轮|
|hidden 保留|`retain_chunk`、`retain_last` 描述 staged prefill 的 gather/scatter 生命周期|
|校验|同一 batch 必须共享 operation 和 execution boundary|

`Operation`、`StateSpan`、`ExecutionBatch` 当前统一定义在

`vllm_rlt/request.py`。Executor 只绑定 immutable execution descriptor，不更新

Request 进度、不创建 KV metadata。

#### Profiler

In `vllm_rlt/profiling.py`

```text
Profiler(device)
  -> 保存 device
  -> current = None
  -> 第一次 start() 时创建 session
```

`StageClient`/`LLM` 通过 `start_profile()`、`stop_profile()`、`profile_status()`

和 `wait_for_profile_artifacts()` 调用 worker 内的 profiler。

#### Attention Backend

In `vllm_rlt/attention/backend.py`：

```text
create_backend(backend, device, dtype, head_dim, block_size,
               required_capabilities, prefer_packed_prefill, ...)
  -> validate_backend_name()
  -> 创建候选 AttentionBackend
  -> 检查 packed、graph、async、device context、FP8、sliding window 等能力
  -> 返回 kernel + capabilities + selection info
```

|选择方式|行为|
|---|---|
|显式 backend|初始化或能力检查失败直接报错，不 fallback。|
|`auto` + CUDA|FA4 -> FA3 -> FA2 -> Triton -> Torch，选择第一个满足设备、dtype、形状与所需能力的候选。|
|`auto` + CPU|选择 Torch reference attention。|
|Ascend NPU|必须显式选择 `ascend_eager` 或 `torch`；`ascend_eager` 要求 NPU + BF16，使用 Torch SDPA。|
|运行期 kernel 错误|不会触发 backend fallback。|

|能力|当前实现|
|---|---|
|`packed_prefill`|FA4 原生支持；普通 Triton 可按 eligibility 自动启用，滑窗 packed 还需要 tile probe；Ascend eager 可使用共享 packed metadata。`use_uva` 只选择输入传输方式，并非 packed 的开关。|
|`packed_matches_decode`|FA4 的普通路径声明支持；静态 scale FP8 路径不再声明逐位一致。同步 speculative 只有同时满足该能力和模型约束时才使用 packed verification。|
|`async_scheduling` / `device_context_lengths`|Triton、FA2、FA3、FA4 支持两者；`ascend_eager` 声明 async，不声明 device-owned context lengths。|
|`sliding_window`|Torch、Triton 和满足版本/shape 条件的 FA4 支持 masking；masking 不等于回收窗口外的 KV pages。|
|`fp8_kv`|Torch/Triton 支持动态 per-vector scale；兼容的 FA4 支持启动时校准的固定 scale，属于实验路径。|

`AttentionBackend.__call__()` 服务普通 paged attention，`prefill()` 服务 packed prefill。异步 speculative 使用 device-owned single-query metadata，不依赖 packed FlashAttention verification。

| |Packed prefill|普通 paged 路径|
|---|---|---|
|Query|连续 token rows 按 sequence 拼成 ragged batch|每个 query row 使用自己的 context length/page table|
|Metadata|`cu_seqlens_q`、sequence IDs、packed table|逐 row block table/context length|
|KV|仍由 KVCacheManager 提供 request/depth page|仍由 KVCacheManager 提供 request/depth page|
|适用|需要 `packed_prefill`；若要与 decode 保持逐位一致，还需要 `packed_matches_decode`|通用的逐 query 路径|

滑窗 packed prefill 还要求受支持的 packed-window kernel 和普通、同步、eager、单 stage 执行，不能同时启用 speculation/preemption。对这条路径，batch 中每条 prompt 都长于 64 tokens 时才采用 packed，否则使用 per-query；不要把它推广为所有普通 packed prefill 的阈值。

#### Ascend NPU 路径

`vllm_rlt/device.py` 在显式请求 NPU 时导入 `torch_npu`；`StageEngine` 在 NPU 上选择继承自 `ModelRunner` 的 `AscendModelRunner`。`ascend_eager` 使用 Torch SDPA，按 page table 收集可见 KV；共享 paged KV writer 和 packed prefill metadata，固定深度 async 下可使用设备侧状态、流和 event。

当前准入范围是单 stage Ouro 四循环、BF16 权重/KV、greedy、`exit_threshold=1`，必须显式设置 `num_blocks` 或 `kv_cache_memory_bytes`。参考 `docs/recipes/ouro_ascend.md` 的固定深度、完整 prompt prefill、单流配置；该方案不使用 graph、speculation、PD、preemption、prefix caching 或 incremental KV。任意 chunked prefill、混合长度及 adaptive exit 的精度仍未完成全面验证；不能由单次 throughput 结果推断等质量。

#### KVCache

In `vllm_rlt/core/memory.py:plan_cache(…)` 与 `vllm_rlt/core/kv_cache_manager.py:kv_bytes_per_block(…)`：

```python
kv_layers = getattr(config, "num_kv_layers", config.num_hidden_layers)
vector_bytes = (
    config.head_dim * parameter.element_size()
    if cache.kv_cache_dtype == "auto"
    else config.head_dim + 4  # FP8 code + 每向量一个 FP32 scale
)
bytes_per_block = (
    2 * kv_layers * cache.block_size * config.num_key_value_heads * vector_bytes
)  # K 和 V 均计算；num_blocks 计所有物理 depth plane
```

```text
block 数来源：
  1. 显式 cache.num_blocks
  2. 显式 cache.kv_cache_memory_bytes / bytes_per_block
  3. CPU 默认 256；NPU 必须显式指定容量
  4. CUDA profile：最大计划 batch 的 prelude + recurrent + coda
     - 按 async/multi-stream 条件预留 concurrent peak
     - execution_buffer_bytes(..., staged_prefill)
     - metadata / context scratch
     - cache.memory_reserve_bytes
     - graph_memory_reserve(...)
     - 不超过并发请求在完整上下文下可使用的 block 上限
```

`execution_buffer_bytes()` 是启动时的 runtime buffer 估算，不是 allocator trace。Dory 只按有 KV 的注意力子层计数，不能直接用全部物理子层数。

|预留项|内容|
|---|---|
|static buffers|Workspace、hidden、page-table metadata|
|async buffers|routing banks、device state、exit/finalize descriptors|
|staged prefill|按 `prefill_stage_rows()` 预留每个 request 的 hidden ring、gather/scatter 和 prompt buffers|
|submission reuse|`MAX_CONCURRENT_SUBMISSIONS=2` 深度的输入/metadata staging|
|graph reserve|每个 graph pool 的 headroom；speculative 额外计算 captured coda logits 的保留空间|

##### FP8 KV

`CacheConfig(kv_cache_dtype="fp8_e4m3")` 可选 FP8 存储；默认 `"auto"` 跟随模型 dtype。Torch/Triton 对每个 `(block, layer, token, KV head)` 的 K/V 向量分别存 FP32 scale，量化时 scale 约为 `amax / 448` 并对零值作下界处理；读 attention 时反量化。以 `head_dim=128` 的 BF16 为例，K/V 主数据和 scale 总计约为 BF16 的 51.6%，但量化可能改变 logits、greedy token 和退出深度，需单独验证。

实验性 FA4 FP8 使用启动时合成输入校准得到的全池固定 scale，会同时影响 Q 的计算精度；它要求兼容的 FA4、SM10x、BF16 和指定 head dimension，不能套用 Torch/Triton 的动态 scale 语义。snapshot/finalize 保留 scale；PD 的动态 scale 随 KV 传输，固定 scale 在连接时校验一致。实现和限制见 `docs/runtime.md` 的 FP8 KV 小节及 `tests/test_fp8_kv_cache.py`、`tests/test_fa4_fp8.py`。

#### KVCacheManager

In `vllm_rlt/core/kv_cache_manager.py`

```python
shape = (num_blocks, num_layers, block_size, num_kv_heads, head_dim)
key_cache = torch.empty(shape)
value_cache = torch.empty_like(key_cache)
metadata_builder = cache_manager.metadata_builder  # property 返回 MetadataBuilder(cache_manager)
```

|配置/状态|含义|
|---|---|
|`last_exited`|每个 token block 按 recurrence depth 保存 plane；`storage_depths=max_loops`。早退出后 finalize 将最后有效 recurrent KV 补到后续需要的 plane。|
|`shared`|所有 depth 共用一个 plane；`storage_depths=1`|
|`_WrittenPositions`|同时记录连续 written prefix 和 out-of-order pending positions|
|prefix caching|只允许 `last_exited`；共享完整 written prompt page，不共享最终 CODA hidden|
|`KVSnapshot`|保存所有 plane 的 page 数据、written 状态以及可选 FP8 scale，供 preemption restore|

`KVCacheManager` 负责 allocation/free/growth、page tables、written-prefix 校验、

prefix cache、transfer lease 和 snapshot/restore。`MetadataBuilder` 只构造

`KVMetadata`/`DeviceKVMetadata`，不接管 KV ownership。

#### Scheduler

In `vllm_rlt/core/scheduler.py`

```text
Scheduler(config, cache_manager, speculative_config, ...)
  -> requests: request_id -> Request
  -> queues: RequestPhase -> deque[request_id]
  -> staged_prefill / wavefront_chunked_prefill
  -> SchedulingPolicy
  -> SchedulerOutput(phase, ScheduledRequest[], loop_count)
```

当前调度单位是 `RequestPhase`，定义在 `vllm_rlt/request.py`：

|Phase|含义|
|---|---|
|`WAITING`|尚未完成 KV admission|
|`RECEIVING` / `TRANSFERRING`|跨 stage import/export 生命周期|
|`PREFILL`|普通 atomic prompt prefill|
|`PREFILLING_PRELUDE`|staged/wavefront prompt chunk 的 token -> hidden|
|`PREFILLING_RECURRENT`|staged/wavefront prompt chunk 的 recurrent depth|
|`PRELUDE`|decode token 的 prelude|
|`RECURRENT`|decode 单轮或 mixed batch 中的 recurrent work|
|`SPECULATIVE`|本地 block/pipeline round|
|`CODA`|hidden -> sampled token|
|`FINISHED`|终止标记；Scheduler finish 后移除 request|

`ScheduledRequest` 快照 token range、depth、chunk 起点、prompt completion 和

hidden-row offset；`SchedulerOutput.phase=None` 表示 mixed batch，每个条目自带

phase。`loop_count` 描述这次提交实际包含的 recurrent stages。

|配置|Policy|
|---|---|
|prefill-only stage|`PromptOnlyPolicy`；是否 staged 取决于 wavefront 配置|
|普通 atomic + refill/no-refill|`RefillPolicy` / `NoRefillPolicy`|
|staged prefill|`SplitPrefillPolicy`，当前只实现 refill 顺序|
|`mode="mixed"`|`MixedPolicy`，可混合兼容的 prompt/recurrent work，但拒绝 speculative|
|speculative config|`SpeculativePolicy`，prompt 阶段是否 staged 仍由 wavefront 配置决定|

> [!info] Atomic Prefill vs Staged/Wavefront Prefill
> - 默认普通路径：进入 `PREFILL`，每个 prompt chunk 的一次 `ExecutionBatch` 执行 prelude 和完整 recurrent depth；长 prompt 仍可分多个 chunk。
> - staged 路径：依次进入 `PREFILLING_PRELUDE` 与 `PREFILLING_RECURRENT`；Runner 保存 chunk hidden，wavefront 可以让不同 chunk/depth 在窗口内重新组合。
> - mixed 路径：可把兼容的 `PREFILLING_RECURRENT` 与 `RECURRENT` 放进同一批；配合 wavefront 还可合并兼容的 prompt/decode PRELUDE；speculative round 不进入 mixed batch。

> [!info] RequestPhase 与 Operation
> `RequestPhase` 描述 Scheduler 的请求工作；`Operation` 描述 ModelRuntime 的一次模型调用。Executor 负责把前者绑定为 `ExecutionBatch`，两者不混用。

#### SpeculativeRunner

In `vllm_rlt/worker/speculative.py` 和 `vllm_rlt/worker/async_speculative.py`

`SpeculativeConfig` 有 `mode="block"` 和 `mode="pipeline"`。两种模式都共享 model、KV cache、`ModelRuntime` 和 Engine 的 `RngRegistry`；同步版本使用 `ModelRunner.core_stream`，异步版本由 `AsyncSpeculativeRunner` 扩展。block 的随机采样通过 rejection sampling 维护 target 分布的语义；数值误差、执行顺序和 RNG 消耗方式不同，并不保证与普通解码同 seed 生成相同文本。

**Block mode**

```text
RequestPhase.SPECULATIVE
  -> draft: 逐候选执行 PRELUDE -> draft_loops * RECURRENT -> CODA
  -> verify: 候选位置继续执行 remaining target loops -> CODA
  -> greedy acceptance / sampling rejection
  -> ModelRunnerOutput(speculative, frontiers)
  -> StageEngine/Scheduler commit token、KV frontier、stop decision
```

|对象|作用|
|---|---|
|`SpeculativeSubmission`|保留 logits、draft IDs、proposal distributions 和 completion event|
|`SpeculativeResult`|accepted token IDs、accepted count、可选 `kv_frontier`|
|`SubmissionQueue`|保护输入 staging，收集前不得复用 submission-owned buffer|
|普通 graph|同步 runner 可用 `RecurrentGraphs` / `CodaGraphs`；符合限制的新 shape 可延迟 capture，超过 row/entry 限制则 eager fallback|

**Pipeline mode**

```text
PipelineState(request)
  -> PipelineToken(position, token, depth, hidden, candidate)
  -> 每次提交推进 draft_loops
  -> depth == target_loops 的 token 产生 target decision
  -> accepted 时保留仍有效的后继 token；rejected 时丢弃依赖它的后继
  -> 返回 kv_frontier
```

`PipelineState` 是 speculative runner 私有状态；Scheduler 仍拥有 admission、

已提交输出、EOS、KV allocation 和生命周期。pipeline 要求 greedy，并要求

`target_loops` 能被 `draft_loops` 整除；preemption 通过同一状态快照恢复。

**AsyncSpeculativeRunner**

|资源/状态|内容|
|---|---|
|round buffers|2 个 `SpeculativeRoundBuffer`，保存 host/device page tables、compact result、lease 和 events|
|block state|GPU 上的 token/position slot table；CPU `token_start` 只用于容量上界|
|pipeline state|GPU token/depth/hidden/candidate/active 数组；不依赖 CPU 每轮提交 token|
|streams|compute stream 顺序执行；copy stream 复制 compact result 并产生 `copy_done`|
|ticket|`collect()` 等待 copy event，按 block/pipeline layout 返回 `ModelRunnerOutput`；`retire()` 释放 buffer lease|
|并发上限|最多两个未退休 round；后继 round 可在前一 round 交付前提交|

**Async speculative CUDA Graph**

```text
speculative_capture_sizes(execution, scheduler, config)
  -> 为 drafts=0..K 计算可达 verification rows
  -> graph_capture_sizes() 生成有限 bucket
  -> StageEngine 初始化时准备 DeviceRecurrentGraphs / bucketed CodaGraphs
```

|机制|行为|
|---|---|
|`DeviceRecurrentGraphs`|GPU 保留实际 position/KV frontier；capture entry 不保存 Request 或 CPU allocation|
|bucket replay|选择最小可容纳 bucket，复制 active rows；超过最大 bucket eager fallback|
|padding|inactive rows 的 hidden/context length 置零，`active_rows` 防止读取或写入 KV|
|coda graphs|async bucketed CodaGraphs 只用于 row-independent Ouro head；cache-backed boundary CodaGraphs 另需 matching decode KV metadata|
|entry 上限|`cuda_graph_max_graphs` 控制 capture 数；新 shape 超限时保留已有 graph、其余 eager|
|capture 生命周期|启动阶段预热和 capture；runtime lazy capture 需要设备同步，并保护 capture 期间的 Python GC/graph pool|

异步 speculative 不使用 packed FlashAttention verification。async block 支持 greedy 和随机采样；pipeline 仍只支持 greedy。它要求 CUDA、

`device_context_lengths` backend 和 fixed target depth。`cuda_graphs=True` 时

允许 bucketed graph，而不是强制关闭 graph；显式过小的 row limit 只影响覆盖范围。

#### ModelRunner

In `vllm_rlt/worker/model_runner.py`

```text
ModelRunner(model, cache_manager, configs..., staged_prefill)
  -> model.eval() + MetadataBuilder
  -> ModelRuntime + Sampler + shared RngRegistry
  -> optional RecurrentGraphs / boundary CodaGraphs / model-specific PreludeGraphs
  -> StagedInputTensor + PackedMetadataBuffer
  -> SubmissionQueue + request state/readback slots
  -> CUDA streams + optional AsyncState
  -> staged_prefill 时创建 hidden ring/gather/prompt buffers
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
|RECURRENT|遍历 metadata；普通 graph 逐 call replay，eager 调 `recurrent_prepared`|
|CODA|调用 `coda`/`coda_prepared`；boundary-KV 可使用 `CodaGraphs`|

`ModelRunner.submit()` 分为两段准备：

```text
prepare_on_cpu(ExecutionBatch)
  -> MetadataBuilder.build()
  -> staged input / packed metadata / prefill segment descriptors

ModelRunner.submit() 按 loop_count 选择 core/boundary stream
  -> prepare_on_gpu(PreparedExecution)
       -> 上传 input、metadata 和 prompt/hidden routing
       -> gather staged hidden 或绑定 AsyncState/Workspace
       -> ModelRuntime.prepare()
  -> _execute() / ModelRuntime.forward()，记录 completion event
  -> 返回 submission ticket；需要 host 结果时 collect() 得到 ModelRunnerOutput
```

**Streams / buffers**

|资源|作用|
|---|---|
|`core_stream`|recurrent loop 和包含 recurrent 的 PRELUDE；async speculative 也使用它|
|`boundary_stream`|无 loop 的 PRELUDE/CODA、KV finalize；单流模式时与 core 复用|
|`copy_stream`|AsyncState routing、host/device metadata 和 async speculative result transfer|
|`readback_stream`|sampling/gate/result 复制到 pinned host slot|
|`capture_stream`|`RecurrentGraphs`/`DeviceRecurrentGraphs` 的 warmup 和 capture|
|`CodaGraphs.stream`|Coda graph 的 capture stream；replay 仍由调用方当前 stream 排序|
|`StagedInputTensor`|按 dtype 管理可增长 CPU staging，并用 UVA 或 async copy 上传|
|`PackedMetadataBuffer`|压缩 packed sequence IDs、lengths、cumulative lengths 和 page tables|
|`SubmissionQueue`|最多保留两个 in-flight submissions；统一决定何时可复用输入/metadata pool|
|`AsyncState`|设备侧 hidden/token slots、page tables、routing/finalize buffers|
|staged-prefill buffers|`prefill_states`、`prefill_hidden`、`BatchInputBuffers`、prompt token buffers；wavefront 按 depth 扩大 hidden ring|

`ModelRunner.events[request_id]` 保存最新 compute/finalize event；复用 state slot、

workspace、staged tensor、readback slot 或 request-owned allocation 前，必须先

完成其 GPU reader。`ModelRunnerOutput` 统一承载 gate scores、token IDs、depths、

completion、speculative results 和 frontiers。

#### PreemptionManager

In `vllm_rlt/engine/preemption.py`

|成员|作用|
|---|---|
|`engine`|访问 StageEngine、Scheduler、ModelRunner、KVCacheManager 和 speculative runner|
|`snapshots`|按 request ID 保存 phase、hidden/token、可选 KVSnapshot、可选 PipelineState|
|`suspend_callback`|Scheduler 选 victim 后调用；队列迁移仍由 Scheduler 完成|
|`resume_callback`|Scheduler admission 时尝试恢复 snapshot|

**suspend：可运行 phase -> WAITING**

```text
Scheduler.preempt(requesting_request)
  -> 选择安全 victim
  -> PreemptionManager.suspend(victim, offload_kv=not priority_only)
       -> 首次 suspend 时 synchronize ModelRunner
       -> rewind staged prefill phase
       -> 保存 phase + hidden/token + 可选 pipeline state
       -> runner.release() / ModelRunner.suspend()  # 保留 RNG
       -> capacity pressure 时将 KVSnapshot 放到 CPU 并 free KV allocation；priority-only 时保留 resident KV
  -> Scheduler.enqueue(victim, WAITING)
```

|模式|snapshot 中的 `kv`|后续行为|
|---|---|---|
|`offload_kv=True`|CPU `KVSnapshot`|释放 device pages；resume 时重新 allocate + restore|
|`offload_kv=False`|`None`，KV allocation 保留|只恢复 hidden/token/pipeline state，不重复分配 KV|
|已有 resident snapshot 再遇压力|先前状态仍在 snapshots|可再次 suspend 转成 CPU KV snapshot，不重复逻辑挂起；admission/恢复容量不足时可以依次回收此类 WAITING resident allocation|

|victim 检查|条件|
|---|---|
|phase|`RECEIVING` 不可抢占；`WAITING` 只有在仍持有 resident KV 且发生 capacity pressure 时才可被 offload|
|输出|`num_output_placeholders == 0`|
|speculation|`num_speculative_rounds == 0`|
|保护集合|不能是当前 requesting request 或本批 `excluded` request|
|transfer|不能持有 transfer lease|
|priority-only|只允许抢占 priority 数值更大的 request|
|排序|priority 优先，再按 `required_blocks(position + 1)` 选择 footprint 更大的 request|

**resume：WAITING -> snapshot 中的原 phase**

|返回值|含义|
|---|---|
|`None`|没有 snapshot，继续普通 admission|
|`False`|CPU KV snapshot 存在但 KV 暂不可分配，保留 WAITING|
|`RequestPhase`|KV/state/pipeline 恢复成功；Scheduler 重新 enqueue 到该 phase|

```text
resume(request)
  -> 若 kv != None：allocate frontier + restore KVSnapshot
  -> restore hidden/input token 到 cache device
  -> speculative runner.restore_pipeline(...)
  -> 如果曾 offload KV 且设备为 CUDA：current stream synchronize，保证恢复数据可见
  -> 删除 snapshot，返回 saved RequestPhase
```

挂起释放 runner slot，但保留 request RNG；终止路径才调用 `release()` 并删除

RNG generator。`discard_snapshot()` 只丢弃保存记录；正常 finish/abort 路径仍由

Scheduler/Engine 释放 request 的剩余 KV 和生命周期资源。

### PDEngine/PDStageConnector

In `vllm_rlt/pd/engine.py`

`PDEngine` 是 PD 部署的 CPU coordinator，不执行 model forward；它拥有多个

`StageClient`、P/D peer、跨 stage request 状态和 transfer 生命周期。

```text
PDEngine(model, ...)
  -> prepare_model() -> read-only ModelConfig
  -> PDConfig + P/D CacheConfig + P/D SchedulerConfig
  -> 校验 model family、KV layout/block size、attention backend
  -> peers / requests / transfers / metrics
  -> 对每个 (role, device)：
       StageClient(..., StageConfig(role), connector_factory=PDStageConnector.create)
         -> StageWorker -> StageEngine
         -> PDStageConnector.create()
  -> 等待所有 StageClient.ready
  -> 比较各 worker fingerprint
  -> 向异角色 peer 发送 connect
  -> 等待所有 StageClient.connected
```

|对象|初始化 owner|持有内容|
|---|---|---|
|`PDEngine`|主进程/HTTP worker|`Peer`、`Pending`、request 映射、transfer 状态、计时和 profile 状态|
|`StageClient`|PDEngine 创建|单个 stage 的进程、Pipe、generation 映射和 control 生命周期|
|`StageEngine`|对应 StageWorker|该 GPU 的 model、KV cache、Scheduler、Runner 和本地 request|
|`PDStageConnector`|对应 StageWorker|hidden staging、`RequestTransfer`、slot、KV import/export 协议|
|`NixlConnector`|PDStageConnector 创建|NIXL agent、registered KV/hidden memory、peer、in-flight transfer handle|

**`PDStageConnector.create()`**

```text
engine.scheduler.config.max_num_seqs
  + max_receiving_requests（仅 produces_tokens 的 stage）
  + max_draining_requests（仅 accepts_prompt 的 stage）
  -> 分配 device hidden staging
  -> NixlConnector(stage_id, cache, hidden, PDConfig)
  -> 注册 key_cache / value_cache / hidden memory
  -> 返回 PDStageConnector(engine, hidden, transport, config)
```

**启动约束**

|检查|作用|
|---|---|
|P/D model fingerprint|比较模型权重、dtype、KV layout、depth、attention 信息|
|KV geometry|P/D 必须使用相同 layout、block size、KV dtype、depth 和 block geometry；固定 scale FP8 还要校验校准 scale|
|角色配置|prefill worker 不启用 speculative；decode worker 才接收 speculative config|
|transport backend|NIXL 注册两侧 KV/hidden memory，建立 remote agent|

**传输所有权**

```text
PDEngine
  -> control channel: reserve / prefill / commit / ack / cancel 元数据
PDStageConnector
  -> reserve 本地 slot、维护 RequestTransfer、确认 compute/transfer 边界
NixlConnector
  -> 用 registered VRAM 地址执行 KV/hidden WRITE
  -> poll 完成状态并释放 transfer handle
StageEngine / KVCacheManager
  -> 保持 request phase、KV allocation、lease 和 compute 生命周期
```

控制通道不承载 KV tensor。目标侧先 reserve 本地 Request/KV，此时处于 `RECEIVING`；收到 commit **且**所有预期 chunk 到达后才能 `accept_import()`，再经过共享 admission 进入 `CODA` 生成首 token。源侧的 compute 完成、目标确认和远端访问结束共同决定何时安全释放，commit 单独不足以释放源 KV。PDEngine 负责跨 stage 顺序，单个 StageEngine 仍负责本地调度和执行。

## Runtime

### 请求如何被加入

在线 `/v1/completions` 请求先由 `CompletionRequest.parse()` 解析。HTTP event loop 调用 `EngineWorker.submit()`，为它创建 `RequestChannel` 并进入 `pending` 队列；channel 有独立的输出缓冲和取消状态。

```text
serving/server.py: complete()
  -> CompletionRequest.parse(body, model)
  -> EngineWorker.submit(spec)
       -> RequestChannel(spec, output_buffer)
       -> channels[request_id] = channel; pending.append(channel)
  -> EngineWorker._run() 在单线程 executor 中调用 _tick(...)
       -> tokenizer.encode(channel.spec.prompt)
       -> channel.spec.resolve_params(...)  # 按实际 prompt 长度解析限制
       -> engine.add_request(request_id, tokens, params, ...)
       -> engine.step() -> 增量解码 -> TokenEvent -> channel -> SSE/JSON
```

单 stage 时这里的 `engine` 是 `StageClient`，PD 服务则是 `PDEngine`；HTTP owner thread 驱动的是 client/coordinator，不直接执行 worker 内的模型。离线 `LLM.generate()` 复用单 stage client：文本输入按需加载 tokenizer，token ID 输入无需 tokenizer，然后循环 `add_request()` / `step()`，收集完成结果。

```text
StageClient.add_request(external_id, prompt_token_ids, params)
  -> generation = uuid4().hex
  -> requests[external_id] = generation
  -> generations[generation] = external_id
  -> _call("add_request", generation, prompt_token_ids, params, ...)
       -> submit_control(...) -> self.send("control", ...)
       -> 等待对应 control reply
  -> StageWorker.command(): methods["add_request"]
       -> StageEngine.add_request() -> _add_request()
```

`StageClient` 不创建运行时 `Request`。generation 隔离相同外部 ID 的历次执行，输出返回时再映射回外部 ID；提交失败会撤销映射。`_call()` 等到 control reply 只代表注册已完成，不代表推理结束；`StageClient.step()` 轮询已有输出，实际 `StageEngine.step()` 由 worker 的 `run()` 循环主动驱动。

`StageEngine._add_request()` 校验 request ID、非空 prompt、token 范围、循环参数和 speculative/NPU 限制，并检查 `len(prompt) + max_tokens - 1` 是否超过模型上下文或整个 KV 池容量。之后构造 `Request`，`Scheduler.add_request()` 将它注册并加入 `WAITING`；真正的请求准入和 KV 分配发生在后续调度中。`-1` 是因为最后采样出的 token 不需要再送回模型。

```text
StageEngine._add_request():
	-> 校验request_id, sampling_params，loop_counts，speculative, trace等
	-> 计算最大的sequence长度`capacity=len(prompt) + max_tokens - 1`
	-> 需要的块数`required = cache.required_blocks(capacity)`并校验
		-> `((max_tokens + self.block_size - 1) // self.block_size) * self.storage_depths`意思是`max_tokens//self.block_size`向上取整再重复`storage_depths`(每个深度一个平面)
	-> self.scheduler.add_request():
		-> 建立request_id->request的映射
		-> 将request_id插入WAITING队列
```

代码：`vllm_rlt/serving/{server,protocol,worker}.py`、`vllm_rlt/entrypoints/llm.py`、`vllm_rlt/engine/{llm_engine,worker,stage_engine}.py`。

### 请求调度与计算

单 stage 的普通路径可以概括为：

```text
WAITING --admission / allocate--> PREFILL（可重复多个 prompt chunk）
       或 WAITING -> PREFILLING_PRELUDE <-> PREFILLING_RECURRENT（staged）
  -> 完整 prompt 达到 full depth -> CODA -> 采样首个 token
  -> PRELUDE -> RECURRENT（可循环多次） -> CODA -> 采样下个 token
  -> 继续 PRELUDE，或遇到 EOS / max_tokens 后 FINISHED
```

prefill-only 且要导出的请求在完整 prompt 后转入 `TRANSFERRING`；PD decode 的本地请求先 `RECEIVING`，接收完整并通过 admission 后从 `CODA` 开始。启用自推测时首个 CODA 之后转入 `SPECULATIVE` round，而不是每个输出都走普通 decode 循环。

一次本地调度/执行周期：

In `vllm_rlt/engine/worker.py`:

```text
StageWorker.run()
  -> connector.progress()（在PD分离时PDStageConnector）
  -> for output in StageEngine.step()
	  -> self.output(output)
```

PDStageConnector.progress() 是一个非阻塞的传输推进轮询，即使当前没有可运行的模型请求，也会继续推进尚未结束的传输。

PDStageConnector.progress() 会轮询 NIXL 完成状态与接收通知，尝试在带宽额度允许时提交待发送的数据块；发送端满足条件后通知 prefill 完成或 commit,接收端确认 commit 和所有预期数据都达到后调用 accept_import()。发送端只有在本地 staging 已安全读取，远程端确认且传输结束后才释放资源。

In `vllm_rlt/pd/connector.py`:

```text
PDStageConnector.progress()：
	-> self.cache.poll_prefixes():
	-> for output in self.engine.step():
		-> self.output(output)
			-> self.output_finished(output.request_id)
```

In `vllm_rlt/core/kv_cache_manager.py`:

```text
KVCacheManager.poll_prefixes():
	轮询处理KV已经写入GPU,但Prefix Cache索引还不能立即建立的情况。
	-> pending = self._pending_prefixes and clear the pending_prefixes
	-> for request_id, allocation, tokens, length, event in pending:
		-> 若event,event.query非空则将(request_i,allocation,tokens, length, event)压入_pending_prefixes, continue
		-> self.publish_prefix(request_id, tokens, length) # 发布前缀
```

(此处可以优化对于列表的重复读写修改)

`PDStageConnector._pending_prefixes` 记录了若干五元组 `(request_id, allocation, tokens, length, event)`

- `allocation` 是一个 `_Allocation` 对象：

> In `vllm_rlt/core/kv_cache_manager.py`:

```python
@dataclass
class _Allocation:
    max_tokens: int
    block_tables: tuple[tuple[int, ...], ...]
    written: list[list[_WrittenPositions]]
    transfer_leases: set[str] = field(default_factory=set)
    release_requested: bool = False
```

> - `max_tokens`: 该请求逻辑上最多允许的 tokens 数量
> - `block_tables`: 逻辑 `KV page`到物理`KV block`的映射表。`block_tables[plane][page]` 得到某个存储平面上的物理`block ID`.
> - `written`: `KV`写入覆盖范围的记录，结构是 `[storage_depths][num_layers]`。每个元素是`_WrittenPositions`,其中`prefix`表示连续写完的前缀长度,`pending`保存前面有空洞时已经写入的离散位置。它用于防止`attention`读取未初始化的`KV`,也用于判断`Prefix Cache` 是否可以发布。
> - `transfer_leases`: 当前持有该 `KV allocation` 的传输租约集合。PD 传输期间，源端或接收端会把 `transfer ID` 放进这里，防止 `KV block` 被提前释放。
> - `release_requested`: 已经请求释放 `allocation`，但由于仍存在 `transfer_leases` 暂时不能释放。等所有租约解除后，才真正回收物理 `blocks`。

> [!Notice] `_WrittenPositions`:
> In `vllm_rlt/core/kv_cache_manager.py`:
> ```python
> @dataclass
> class _WrittenPositions:
> 	prefix: int = 0
> 	pending: set[int]
> 	def add(self, position: int)
> 	def add_range(self, start: int, end: int)

- tokens: 请求的完整 prompt token ID 序列
- length: 当前已经 Prefill 完成的 token 数，也就是有效 KV 前缀长度，通常来自 request.num_prefilled_tokens
- event: GPU 写入完成事件，通常是 CUDA event。它表示对应的 KV 写入操作是否已经完成,`query()` 方法是无阻塞方法，用于检查事件是否完成。

In `vllm_rlt/core/kv_cache_manager.py: KVCacheManager.publish_prefix()`:

```text
KVCacheManager.publish_prefix(): # 仅在开启prefix_caching时有作用
	-> allocation = self._get_allocation(request_id)
		-> allocation = self._allocations[request_id]
	-> if event is not None: # 对应Scheduler中的调用路径
		->  self._pending_prefixes.append((request_id, allocation, tuple(tokens), length, event))
	-> 否则
	-> 遍历self._prefix_keys(tokens[:publishable_tokens]): (index, key)
		-> end = (index + 1) * block_size
		-> 遍历每个平面的第index层的blocks然后添加引用计数并在self._prefixes[key] = blocks添加索引
```

### StageEngine.step()

In `vllm_rlt/engine/stage_engine.py: StageEngine.step()`:

```text
StageEngine.step():
	-> StageEngine._step():
		# 异步投机解码路径
		StageEngine._step_async_speculative()
	or  # 异步路径
		StageEngine._step_async()
	or  # 同步路径
		batch = self._schedule() # 调度一批请求
		ticket = self.executor.submit(batch) # 提交这批请求
		result = ticket.collect() # 获取结果
		return self._update(batch, result) # 更新执行情况
```

ticket 是 executor.submit(batch) 返回的提交句柄: 它关联这个 batch 的执行结果，完成事件和必要的读回缓冲区。

同步路径下 ticket 是 `Submission` 类型的 (见`vllm_rlt/worker/model_runner.py`)

```python
@dataclass
class Submission:
	batch: SchedulerOutput | None
	values: torch.Tensor | None
	event: object = None
	cached: ModelRunnerOutput | None = None
	slot: ReadbackSlot | None = None
	device_values: torch.Tensor | None = None
	depths: tuple[int, ...] = ()
	positions: tuple[int, ...] = ()
	output_indices: tuple[int, ...] = ()
	operation: Operation | None = None
```

Submission.collect() 会主动同步 `CudaEvent`

#### `StageEngine._step_async()`

- `self._inflight` 存储已经提交给 GPU 但是其完成状态或结果生命周期尚未完全结束的 `submission ticket`
- `self._pending_coda` 保存已经提交但尚未交付的 `CODA` ticket.

```text
StageEngine._step_async():
	-> 保留 `self._inflight`中`not ready`的`ticket`
	-> outputs = self._collect_coda():
			# 从已经异步提交的`CODA batch`中收集采样结果，并把`token`交回`Scheduler/Engine`处理
			-> 遍历self._pending_coda: (ticket)
				-> 若ticket就绪或者等待其就绪：
					outputs.extend(self._deliver_coda(ticket))
						-> self._deliver_coda(ticket):
							return self._update(ticket.batch, ticket.collect, delivered=True)
				-> 否则加入pending
			self._pending_coda = pending
			return outputs
	-> batch = self._schedule(...)
	-> if batch.phase == CODA:
			-> 新的 CODA batch 若包含仍有未决输出的请求，就反复调用 _collect_coda(wait=True)，直到这些请求先前的 token 都已交付、停止条件已判定。
			-> 交付可能因 EOS 或长度上限而结束请求，所以之后用
  replace() 从 batch 中过滤掉已不在 Scheduler 中的请求；若过滤后为空，就返回刚收集到的 outputs，不提交空 batch。
	-> ticket = self.executor.submit(batch)
	-> self._inflight.append(ticket)
	-> if batch.phase == CODA:
			self._pending_coda.append(ticket)
	-> 遍历batch.requests: (row, scheduled_request):
		-> PRELUDE: 删除该请求之前保留的 exit signal
		-> RECURRENT: 若需要前一轮 gate score，则 collect 对应 ticket 并构造 ExitSignal
	-> update = self.scheduler.update_after_submission(...): 按已提交的 batch 推进 Scheduler 状态
	-> self.model_runner.bind_sampled_tokens(...): 将设备上的采样 token 绑定为下一次 PRELUDE 输入
	-> 将 update.retain_signal 中的 gate score 记录到 self._pending_exit_signals
	-> outputs.extend(self._apply_update(update))
	-> return outputs
```

异步路径允许同一请求的后续 `PRELUDE` 和 `recurrent` 计算先运行；但这些计算可能会把请求再次推进到 `CODA`。`CODA` 会采样新 `token`，因此必须先知道上一个`token` 是否触发 `EOS` 或达到长度上限，否则可能多采样一个 `token`。

#### `StageEngine._step_async_speculative()`

```text
StageEngine._step_async_speculative():
	-> self._retire_speculative():
		# 非阻塞地回收已经完成的请求及其 speculative state
	-> outputs = []
	-> if pipeline_speculative and enable_preemption and WAITING队列非空:
		-> while self._pending_speculative:
				outputs.extend(self._collect_speculative())
				# preemption 选择 victim 前，先等待已有 round 到达 quiescent point
	-> if len(self._pending_speculative) == 2:
		-> outputs.extend(self._collect_speculative())
			# 最多保留两个未收集 round，先收集最老的 round 以释放 buffer
	-> batch = self._schedule()
	-> if batch is not None:
		-> if batch.phase == SPECULATIVE:
			-> ticket = self.executor.submit(batch)
			-> self._pending_speculative.append(ticket)
			-> self._apply_update(self.scheduler.update_after_submission(batch))
				# speculative round 在 device 上提交后，先推进 host 侧提交状态
		-> else:
			-> ticket = self.executor.submit(batch)
			-> result = ticket.collect() if ticket.has_result else None
			-> outputs.extend(self._update(batch, result))
				# 非 speculative batch 用于 bootstrap 等路径，按普通方式更新
	-> elif self._pending_speculative:
		-> outputs.extend(self._collect_speculative())
	-> elif self._retiring_speculative:
		-> self._retire_speculative(wait=True)
	-> elif self.scheduler.has_unfinished_requests and not self._waiting_for_connector():
		-> raise RuntimeError("speculative scheduler made no progress")
	-> return outputs
```

`_collect_speculative()` 总是处理 `self._pending_speculative[0]`，保证 round 按提交顺序交付：

```text
_collect_speculative():
	-> ticket = self._pending_speculative[0]
	-> 若 ticket 未就绪，记录一次 readback wait
	-> result = ticket.collect()
		# 等待 copy stream 完成，并读取 speculative token、accepted count 和 frontier
	-> 遍历 ticket 中的 request/state/round:
		-> 跳过已经取消、结束或 generation 已过期的 request
		-> 校验 state.delivered == round_id，禁止 round 乱序交付
		-> 检查 numerical_error，并推进 state.delivered
	-> outputs = self._update(ticket.batch, result)
	-> ticket.retire()
	-> self._pending_speculative.pop(0)
	-> self._retire_speculative()
	-> return outputs
```

#### `StageEngine._schedule()`

主要行为发生在 SchedulerPolicy.schedule() 中

In `vllm_rlt/core/scheduling_policy.py`

不同策略有不同的行为

```text
StageEngine._schedule(prefer_recurrent=False):
	-> self._connector_blocked.clear() # 清空上轮 connector 阻塞记录
	-> self.scheduler.schedule(prefer_recurrent=prefer_recurrent)
		-> 若没有已注册的 request，返回 None
		-> self.policy.schedule(self, prefer_recurrent=prefer_recurrent)
			-> 策略决定何时 _admit()，以及从哪些 phase 队列取请求
			-> take_phases(...) 按就绪状态、batch 预算和 KV 容量选取请求
			-> 组出 SchedulerOutput，或在没有可选请求时返回 None
```

各策略的主要顺序：

- `RefillPolicy`：先看就绪的 `PRELUDE`、满足条件的 `CODA` 和到期的 `RECURRENT`，再尝试准入和 prefill。`prefer_recurrent=True` 时先尝试就绪的 `RECURRENT`。
- `NoRefillPolicy`：按 `FILL -> CORE -> CODA` 推进。进入 `CORE` 后先处理就绪的 recurrent，再处理 CODA，然后回到 `FILL`。
- `SplitPrefillPolicy`：沿用 refill 顺序，把 `PREFILL` 拆成 `PREFILLING_PRELUDE` 和 `PREFILLING_RECURRENT`。
- `MixedPolicy`：优先尝试将就绪的 decode 和 prefill recurrent 拼入同一 batch，其余情况按 split prefill 路径处理。
- `SpeculativePolicy`：先处理 `CODA`，再根据 `decode_due`、准入和 prefill 的情况选择 `SPECULATIVE`。
- `PromptOnlyPolicy`：准入后只从 prefill 队列选取请求。

#### `Executor.submit()`

```text
Executor.submit(scheduled_batch):
	-> 若 scheduled_batch.is_speculative，交给 speculation.submit()
		-> 未配置 speculation 时抛出 ValueError
		-> 返回对应的 round ticket，由 Engine 在后续 collect
	-> 否则 batch = build_execution_batch(scheduled_batch, model.config.total_ut_steps)
		-> 对每条 scheduled_request，按 phase 选择 PRELUDE、RECURRENT 或 CODA
		-> 将 request、token_start、token_count 和 depth 绑定到 StateSpan
		-> 确定本次执行的 loop_count，整段 PREFILL 可从 PRELUDE 连跑全部 recurrent loops
		-> 检查所有 span 的 operation 和 loop_count 一致，再组出 ExecutionBatch
	-> return self.runner.submit(scheduled_batch, batch=batch)
		-> submissions.begin() 取得可复用的 submission slot
		-> prepare_on_cpu(batch) 准备输入和 KV metadata
		-> _wait_for_requests(batch, stream) 建立前序设备依赖
		-> prepare_on_gpu(prepared) 绑定设备输入和 forward
		-> _execute(scheduled_batch, prepared) 执行 forward，得到结果 tensor（若有）
		-> 记录 compute_done，按需安排 device -> host 读回
		-> return Submission(...)
```

`scheduled_batch` 保留本次调度选中的请求和 phase，供结果交付时按行对应。`ExecutionBatch` 描述要跑的模型操作，Runner 再据此准备输入和 KV metadata。

**`ModelRunner.submit()` 的准备与收尾**

```text
ModelRunner.submit():
	-> batch.loop_count > 0 走 core_stream，否则走 boundary_stream
	-> SubmissionQueue.begin(): 清理已完成的计算 event，队列满时等最老的一次计算
		-> 轮转到安全的 slot，供本次输入和 metadata 缓冲使用
	-> prepare_on_cpu(batch)
	-> 若 routing 有需要导入的 hidden/token，先在 copy_stream 上等待来源 stream 和请求 event
	-> 在选中的执行 stream 上建立依赖，再 prepare_on_gpu() 和 _execute()
		-> 有 routing 时，在执行收尾记录 routing buffer 的使用完成 event
	-> 异步 recurrent 没有后续 gate 消费者时，丢弃 result，省去读回
	-> 保留 device_values。有设备 stream 时，在执行 stream 上记录 compute_done
		-> 若有 result，取得 readback slot，让读回 stream 在 compute_done 后复制到 pinned host buffer
		-> 有读回时记录读回完成 event，作为 Submission.event，否则使用 compute_done
		-> 将 compute_done 写入每个 request 的 model_runner.events，再调用 submissions.complete(compute_done)
	-> 返回 Submission，带上 batch、event、结果和行对应的 depths/positions/output_indices
	-> 出错时同步本次可能使用的 stream，排空 submission 队列并解除已借用的读回 slot
	-> 最后在执行 stream 上为本轮 workspace 记录完成 event（若有）
```

`_readback_slot()` 按 dtype 找未租用的 slot，优先取上一次拷贝已完成的 slot。若只有未租用但仍在拷贝的 slot，会等一个可用。读回 event 供 `Submission.collect()` 等待，后继计算使用每个请求的 `compute_done` 建立依赖。

**`prepare_on_cpu()`**

```text
prepare_on_cpu(batch):
	-> reset staged input buffers
	-> _prepare_cpu_inputs(batch) 生成 PreparedExecution
		-> 校验深度，保存每个 span 的 position、depth 和 output index
		-> CODA 保存实际完成的 loop 数和 sampling params，执行用的 boundary KV depth 仍是 0
		-> recurrent 一个 span 有多行时，取各 span 的末行作为 gate 结果行
		-> PRELUDE + loops: _prepare_loop_inputs() 直接准备多个连续操作的 forwards
		-> split/mixed prefill: 用本次 slot 的 BatchInputBuffers，准备 span 分段和 token 来源
		-> 普通单段 PRELUDE: 记录显式 token、设备采样 token 或 CPU token ID
		-> 其余准备按配置使用 AsyncState.prepare() 或 Workspace
	-> PRELUDE + loops 已准备好 forwards，直接返回
	-> RECURRENT: 为各执行深度准备 metadata，已有首个 KV view 时从后续深度开始
	-> boundary-KV 模型的 PRELUDE/CODA: 准备 boundary metadata
	-> 放入 PreparedForward，返回 prepared
```

- `_prepare_loop_inputs()` 把各请求的 token span 展成模型行，为每一轮构造 KV 视图，并记录每条请求需要保留的末行 hidden。支持 packed 的路径可以先存紧凑描述，等设备侧展开。shared KV 布局由 `_prepare_shared_loop_inputs()` 按 token offset 分组，依次推进每条请求的位置。
- `_prefill_segments()` 记录本地 request slot、chunk hidden 偏移、行数和 batch 行起点。`_prepare_prefill_tokens()` 把显式或缓存的 prompt token、设备采样 token 等输入登记成拷贝描述，设备准备时再汇入同一个 token buffer。
- `AsyncState.prepare()` 取得每个请求的状态 slot，记录 slot、depth、position，并准备待更新的 block table 和待导入的状态。`_stage_metadata()` 则由 `MetadataBuilder.build()` 生成各行的 block table、写入位置等布局，再暂存为待上传的视图。

**`_wait_for_requests()`**

```text
_wait_for_requests(batch, stream):
	-> 遍历 batch.spans，取每个请求最近的 self.events[request_id]
	-> 每个不同的 event 只调用一次 stream.wait_event(event)
	-> 对已有 hidden_state 调用 record_stream(stream)，覆盖该 stream 的使用期
```

`wait_event()` 把依赖排进目标 stream。后继 work 可以先提交，等前一段计算或 KV finalize 完成后才执行。routing 有导入时，copy stream 也要等请求 event，因为它可能先于执行 stream 读取已有的 hidden/token。

**`prepare_on_gpu()`**

```text
prepare_on_gpu(prepared):
	-> CODA 为非贪心采样准备每请求的 generator
	-> 有 routing: 在 copy_stream 调用 routing.transfer()
		-> 提交 block table 更新和 hidden/token 导入，生成设备侧 slot 或 KV metadata
		-> 记录 ready_event，CUDA 执行 stream 等它后再消费
	-> 若已有 Workspace KV metadata，上传对应的输入视图
	-> split/mixed batch: 绑定分段描述或准备 mixed hidden 的来源和写回位置
		-> PRELUDE: 提交 token_transfers，得到 token_ids
		-> RECURRENT: _gather_prefill() 将各请求的隐藏状态聚合成 batch hidden
	-> 普通 PRELUDE: _prepare_tokens() 从 routing slot 或已记录的 token 来源组成输入
	-> _prepare_forward(prepared) 绑定当前执行所需的 ForwardBatch
```

`_prepare_forward()` 上传并绑定 KV metadata 与 gate 行号。PRELUDE 绑定 token，RECURRENT/CODA 通过 `_load_hidden()` 取 chunk、routing 或普通 request hidden，再交给 `ModelRuntime.prepare()` 选择可用的 graph/fallback。PRELUDE 连跑多轮时走 `_prepare_loop_forwards()`，把 CPU 准备好的多段 forward 绑定到设备视图。需要 boundary KV 的模型还可能经 `_prepare_prefill_coda()` 为非末尾的 prompt token 补一段计算，只更新 boundary KV。

**`_execute()`**

```text
_execute(scheduled_batch, prepared):
	-> _execute_batch(prepared.batch, prepared)
		-> 遍历 prepared.forwards，必要时从上一段 hidden 选择本段要用的行
		-> ModelRuntime.forward(forward)
			-> PRELUDE 根据 token 得到 hidden，fused 路径随后按 metadata 遍历 recurrent depth
			-> RECURRENT 使用输入 hidden 和各 depth 的 KV metadata，逐轮 replay graph 或调用 recurrent_prepared()
			-> CODA 从 hidden 得到 logits，特殊 prefill CODA 只补写 boundary KV
		-> PRELUDE/RECURRENT: _store_hidden() 写回需要保留的 hidden，或按 entry.outputs 保存指定末行
			-> split/mixed 将中间行写回 chunk 槽，并按需保存请求的最后一行
			-> 普通路径写回 request hidden 或 scatter 到 AsyncState slot
		-> RECURRENT 需要 gate 时，取模型 gate logits 或 lookahead_head 输出，选中对应行后做 sigmoid
		-> CODA 有 logits 输出时，按请求采样 token，必要时写入设备侧 token slot
		-> return values # gate scores、采样 token 或 None
```

SPECULATIVE 分支走独立的 round runner。异步模式在 compute stream 上做 draft/verify 或 pipeline step，推进设备侧请求状态，再由 copy stream 读回 compact result 并返回 `SpeculativeRoundTicket`。提交和回收顺序见上面的 `StageEngine._step_async_speculative()`。

#### `StageEngine._update()`

```text
StageEngine._update(batch, result, delivered=False):
	-> 将本次 batch、ModelRunnerOutput 和 model_runner.events 传给 scheduler.update_from_output()
		-> result 可以为 None，例如只推进 prefill 的 batch
		-> request_events 为 prefill 更新提供每个请求的设备完成 event
		-> delivered=True 用于异步 CODA 结果交付，Scheduler 会消费之前留下的输出占位
	-> 得到 SchedulerUpdate，包含需要通知、收尾或输出的 request
	-> return self._apply_update(update)
```

#### `StageEngine._apply_update`

```text
StageEngine._apply_update(update):
	-> 遍历 update.prefill: (scheduled_request, phase, progress)
		-> _notify_connector_prefill(scheduled_request, phase, progress)
			-> 对需要导出 KV 的请求，调用 connector.prefill_submitted()
	-> 对 update.exited 调用 model_runner.finalize_many()
		-> 例如 last_exited 布局下提前退出时，用退出深度的 KV 补齐跳过的 recurrent depth
	-> 若启用 speculative runner，累计 committed_tokens 和 accepted_tokens
	-> 对 update.finished 中的请求调用 self._finish(request, request.finish_reason)
		-> 异步 speculative 路径先标记结束，等设备不再访问该请求后再回收资源
	-> 对 update.outputs 中的请求调用 RequestOutput.from_request() 并返回列表
```

**`StageEngine._notify_connector_prefill()`**

```text
_notify_connector_prefill(scheduled_request, phase, progress):
	-> 取 scheduled_request.request
	-> 存在 connector 且 connector.exports(request) 时继续
		-> 普通 PREFILL 的 start = scheduled_request.token_start
		-> split prefill 优先取 progress.transfer_start，否则取 scheduled_request.chunk_start
		-> 传入 request、phase、start、progress、runner 的 per-request event 和 core_stream
		-> connector.prefill_submitted(...)
```

wavefront 同一轮可能完成同一请求的多个 span。`transfer_start` 取这一轮推进前的 `prefill_reclaimable_tokens`，通知携带的是本轮新完成的连续范围。

**`PDStageConnector.prefill_submitted()`**

```text
prefill_submitted(request, phase, start, progress, event, stream):
	-> 查找 transfer，校验 transfer.request 仍是当前对象
	-> staged PRELUDE 或 wavefront RECURRENT 时，记录与已有传输的重叠
	-> 若 chunk_complete=False，只更新 last_event 并返回
	-> 在所给 stream 上等待 producer event
		-> prompt_complete 时将最后的 hidden_state 拷到 transfer 的 staging slot
		-> 记录 ready event，覆盖这一段 KV 和最后 hidden 的准备
	-> queue_chunk(transfer, start, request.num_prefilled_tokens, ready, prompt_complete)
	-> 将 progress.prompt_complete 记为 transfer.compute_done
```

`queue_chunk()` 先用目标端已命中的 `target_cached_tokens` 裁掉不必发送的前缀，再由 `kv_segments()` 按源、目标各自的 block table 映射 KV 范围。完整 block 可以整体描述，部分 block 按 layer 取有效 token，最终段再附上 hidden。 `partition_segments()` 合并两端都连续的范围，并按字节数和描述符数量拆分，依次写入带 ready event 和 sequence 的 `transfer.chunks`。后续 `progress()` 确认 event 就绪且 transport 有余量才提交传输。

#### `ModelRunner.bind_sampled_tokens()`

```text
ModelRunner.bind_sampled_tokens(batch, values, rows):
	-> values 来自当前 CODA ticket 的 device_values
	-> 遍历 rows（update.next_prelude 中的 batch 行号）
		-> batch.requests[row].request.input_token_tensor = values[row]
			# 只给还有输出预算、已排入下一次 PRELUDE 的请求绑定 token
```

下一次 `PRELUDE` 可以直接读取设备上的 token，Runner 用 stream/event 保证它在采样完成后才读取。CPU 侧随后通过 `_collect_coda()` 提交这个 token，并检查 EOS 和长度限制。

#### `Scheduler.update_after_submission()`

```text
Scheduler.update_after_submission(batch, completion=..., request_events=..., signals=...):
	-> 创建 SchedulerUpdate
	-> wavefront_groups = _wavefront_prefill_groups(batch)
	-> loop_count = _batch_loop_count(batch)
	-> 按 batch.phase_for(scheduled_request) 逐行更新
		-> 跳过已结束或对象身份与 self.requests 中记录不一致的 request
		-> 先调用 _apply_prefill_phase_progress(..., fallback_event=completion)
			-> 若已处理该 prefill 行，继续下一行
	-> PRELUDE: 重置 loops_done、remaining_probability 和 pending_exit_depth，进入 RECURRENT
	-> RECURRENT:
		-> loops_done += 1，按 trace 或 loop 上限判断退出
		-> 若需要上一轮 ExitSignal，校验 request、position 和 depth，再应用延迟退出规则
		-> 退出时记录 pending_exit_depth，加入 update.exited 并进入 CODA
		-> 继续时回到 RECURRENT，延迟退出策略按需把当前行号加入 update.retain_signal
	-> CODA:
		-> num_output_placeholders += 1，表示该输出已提交采样、尚未交付
		-> 已交付数量 + 占位数量仍小于 max_tokens 时，排入 PRELUDE，行号加入 update.next_prelude
	-> SPECULATIVE:
		-> num_speculative_rounds += 1，pipeline 模式预留 1 个输出，block 模式预留本次 token_count
		-> 重新排入 SPECULATIVE，后续调度仍受在途 round 和输出预算限制
	-> return update
```

**`_wavefront_prefill_groups()` 和 `_batch_loop_count()`**

```text
_wavefront_prefill_groups(batch):
	-> 未启用 wavefront 时返回空字典
	-> 按 batch.phase_for(item) 选出 split prefill 的 PRELUDE/RECURRENT span
	-> 排除已结束或对象身份不匹配的请求
	-> 按 (request_id, phase) 将同一请求的 span 放到一组

_batch_loop_count(batch):
	-> 优先使用 batch.loop_count，包括显式的 0
	-> 没有显式值时，从本次 batch 的 phase 推断
		-> 有普通 PREFILL: total_ut_steps
		-> 有 PRELUDE、CODA 或 SPECULATIVE: 0
		-> 有 PREFILLING_RECURRENT 或 RECURRENT: 1
		-> 其余: 0
```

**`_apply_prefill_phase_progress()`**

```text
_apply_prefill_phase_progress(scheduled_request, phase, wavefront_groups, update, ...):
	-> 处理 prefill 时，若有 request_events 字典，取当前 request 的 event
		-> 调用方未传字典时，使用 fallback_event
	-> wavefront + split prefill:
		-> 从 wavefront_groups pop 掉该 (request_id, phase) 整组
		-> 第一次遇到这一组时调用 _apply_wavefront_prefill_batch_progress()
		-> 将 (representative, phase, progress) 加入 update.prefill
		-> 同组后续的行已合并处理，仍返回 True
	-> 其他 prefill phase: 调用 _apply_prefill_progress()
		-> 将 (scheduled_request, phase, progress) 加入 update.prefill，返回 True
	-> 非 prefill phase 返回 False，交给调用方处理
```

**`_apply_wavefront_prefill_batch_progress()`**

```text
_apply_wavefront_prefill_batch_progress(phase, scheduled_requests, loop_count, event):
	-> span 按 token_start 排序，校验属于同一请求，PRELUDE 需要 0 loops，RECURRENT 需要 1 loop
	-> transfer_start = request.prefill_reclaimable_tokens # 先记下推进前的可回收边界
	-> PREFILLING_PRELUDE:
		-> 从 num_preluded_tokens 起逐段校验并延伸连续范围
		-> 更新 num_preluded_tokens，本轮没有完成任何全深度 chunk
	-> PREFILLING_RECURRENT:
		-> 对每个 span，校验 start 位于该 depth 的 frontier
		-> 调用 _advance_prefill_depth_frontier()，将这一段推进一个 depth
		-> 到达最终 depth 且 end 正好是 chunk 末尾时，核对 chunk 按顺序完成
			-> 更新 prefill_reclaimable_tokens，以该 span 作为 representative
		-> 有完整 chunk 时调用 publish_prefix(..., event)
	-> prompt 全部完成时设置 loops_done，由 _prompt_destination() 选下一 phase
	-> 否则窗口有空间时排入 PREFILLING_PRELUDE，窗口满时排入 PREFILLING_RECURRENT
	-> 返回 representative 和 PrefillProgress(chunk_complete, prompt_complete, transfer_start)
```

`_prelude_window_open()` 用 `prefill_reclaimable_tokens + prefill_chunk_size * depth 数` 作为 wavefront 窗口上限，并裁到 prompt 长度。只有连续完成最终 depth 的整个 chunk 才推进可回收边界，给后续 PRELUDE 腾出槽位。

**`_apply_prefill_progress()` 和 `_apply_split_prefill_progress()`**

```text
_apply_prefill_progress(scheduled_request, phase, loop_count, event):
	-> 普通 PREFILL:
		-> num_prefilled_tokens += token_count，调用 publish_prefix(..., event)
		-> prompt 完成则设置 loops_done，排入 _prompt_destination(request)
		-> 否则继续 PREFILL，返回 PrefillProgress(True, prompt_complete)
	-> split phase: 转交 _apply_split_prefill_progress()

_apply_split_prefill_progress(scheduled_request, phase, loop_count, event):
	-> 取本次 [start, end)
	-> PREFILLING_PRELUDE: 更新 num_preluded_tokens 和 chunk 起点，从 depth 0 开始
		-> 若本轮只跑 PRELUDE，排入 PREFILLING_RECURRENT，返回未完成
	-> PREFILLING_RECURRENT: 从 scheduled_request.depth 开始
	-> 遍历本轮执行的各 depth，校验 recurrent span 处于连续范围内并调用 _advance_prefill_depth_frontier()
	-> 若没有推进到最终 depth，或还没完成本 chunk，继续 PREFILLING_RECURRENT
	-> 完成 chunk 时 publish_prefix(..., event)，将 prefill_reclaimable_tokens 推进到 end
	-> prompt 完成则设置 loops_done 并走 _prompt_destination()，否则排入 PREFILLING_PRELUDE
	-> 返回 chunk/prompt 的完成状态

_advance_prefill_depth_frontier(request, depth, token_end):
	-> prefill_depth_ranges[depth][0] = token_end，表示当前 depth 已消费到这里
	-> 若有下一 depth，将它的可执行上界 [1] 延伸到 token_end，返回 False
	-> 若当前已是最后 depth，更新 num_prefilled_tokens，返回 True

_prompt_destination(request):
	-> exports_request(request) 为 True 时返回 TRANSFERRING
	-> 其他请求进入 CODA，准备采样第一个输出 token
```

三个进度量分别对应不同的边界。`num_preluded_tokens` 指 PRELUDE 已推进的位置，`num_prefilled_tokens` 指完整 recurrent 深度已推进的位置，`prefill_reclaimable_tokens` 指已完成、可回收槽位的整段 chunk 边界。它们记录 host 侧进度，prefix cache 通过 event 确认 KV 写入完成后才发布。

#### `Scheduler.update_from_output()`

```text
Scheduler.update_from_output(batch, result, request_events=..., delivered=False):
	-> 若 result 为 None，使用空的 ModelRunnerOutput
	-> 创建 SchedulerUpdate
	-> wavefront_groups = _wavefront_prefill_groups(batch)
	-> loop_count = _batch_loop_count(batch)
	-> 按本次 batch 保存的 phase 逐行处理，跳过已结束或被替换的 request
		-> 先调用 _apply_prefill_phase_progress(..., fallback_event=result.completion)
			-> wavefront 进 _apply_wavefront_prefill_batch_progress()，其他 prefill 进 _apply_prefill_progress()
			-> 已处理的 prefill 行跳过后面的 phase 分支
	-> PRELUDE: 重置 loop 和退出状态，进入 RECURRENT
	-> RECURRENT: loops_done += 1，根据 trace、固定深度或当前 gate score 判断退出
		-> Ouro 用 score 更新 remaining_probability，延迟退出模式可记录下一轮退出深度
		-> 退出时加入 update.exited 并进入 CODA，否则继续 RECURRENT
	-> CODA:
		-> delivered=True 时，校验 output_indices[row] 等于已交付 token 数，且恰有一个 placeholder，然后扣掉它
		-> 从 result 记录 token，异步交付用 result.depths[row]，同步用 request.loops_done
		-> 遇 EOS（且未忽略）或达到 max_tokens 时结束
		-> 若继续且 delivered=False，排入 PRELUDE 或 SPECULATIVE
	-> SPECULATIVE:
		-> 有 result.frontiers 时，扣除占位和 round 计数
			-> 校验 GPU frontier = prompt 长度 + 已交付 token 数 - 1 + 本轮返回 token 数
		-> 逐个提交本轮 token，遇到停止条件即停，只统计实际提交的 committed/accepted 数量
		-> 若请求继续，同步路径截断未提交的 KV 后缀，异步路径保留可能已在写入的下一轮 KV
		-> 若请求继续且不在 SPECULATIVE 队列，重新入队
	-> CODA 和 SPECULATIVE 将 request 加入 update.outputs，结束的也加入 update.finished
	-> return update
```

异步 CODA 的下一次 `PRELUDE` 已在提交阶段按预算安排，因此 `delivered=True` 只交付结果和检查停止条件。同步 CODA 在这里读到 token 后，才决定该请求的下一阶段。

两种更新共用上面展开的 prefill 子流程。`update_after_submission()` 在提交时推进调度状态，未提供 per-request event 字典时使用 `completion`。`update_from_output()` 在同步更新和异步结果交付时使用，未提供该字典时使用 `result.completion`。 `_record_token()` 将 token 和对应退出深度追加到请求，再检查 EOS 或最大生成长度，命中时设置 `finish_reason` 并让调用方加入 `update.finished`。

#### 同步、异步和退出

同步 `_step()` 对选中的 batch submit；只有 ticket 需要 host 结果时才 collect，否则可在 event 约束下推进逻辑状态。普通异步 `_step_async()` 可以先提交下一段工作，再回收前一段必要的 exit signal / CODA 结果；`num_output_placeholders` 防止未确认 EOS 前重复采样。自推测异步路径最多保留两个未退休 round，并对过期 request/round 结果进行检查。

普通 Ouro `exit_mode="ouro"` 用 gate hazard 更新累计退出概率；`ouro_delayed` 将这个决定推迟一轮，`random_lookahead` 使用独立的随机 lookahead head，`trace` 使用预先给定的 depth。`exit_threshold=1` 禁用自适应退出，`max_loops` 是上限。完整 prompt 始终跑模型完整深度；上述退出选择主要影响后续 decode。Dory 的固定 target 规则见 Model 小节。

同步更新由 `Scheduler.update_from_output()` 处理；异步提交时 `update_after_submission()` 可先更新 host 侧逻辑进度，event 仍负责设备依赖，后续 collect 再交付需要 host 确认的 token/score。请求队列中的 phase 因此不是 GPU 已完成的证明。

#### 输出、取消和资源回收

`ModelRunnerOutput` 的 token IDs、depth 和结束信息经 Scheduler 转成 `RequestOutput`。worker 输出携带内部 generation，`StageClient` 映射回外部 ID；HTTP worker 只增量解码新 token，再通过 `RequestChannel` 交付 text、usage 和 finish reason。一次 speculative round 可以产生多个已提交 token，不能假设一次事件恰好一个 token。

结束、取消或失败会停止后续调度并清理 runner/RNG 和 request/KV。逻辑结束不等于物理内存立即可复用：异步 round、readback、GPU reader 和 PD transfer lease 仍可能持有资源，必须等相关 event / transport 安全后才能退休。客户端过慢导致 channel 缓冲满时，HTTP 层会报错并安排取消。

### 阅读与验证入口

|问题|对应文件|
|---|---|
|架构、stage owner 与执行边界|`docs/unified_execution.md`、`vllm_rlt/engine/stage_engine.py`|
|admission、phase 和 mixed/wavefront 选择|`docs/scheduler.md`、`docs/split-prefill-wavefront.md`、`vllm_rlt/core/{scheduler,scheduling_policy}.py`|
|KV 布局、prefix、transfer pin、FP8|`docs/kv_cache_walkthrough.md`、`docs/kv_layout_computation.md`、`docs/runtime.md`、`vllm_rlt/core/kv_cache_manager.py`|
|自推测与异步生命周期|`docs/async-speculative.md`、`vllm_rlt/worker/{speculative,async_speculative}.py`|
|HTTP/PD、设备限制|`docs/serving.md`、`docs/recipes/ouro_ascend.md`、`vllm_rlt/pd/`|
|回归测试|`tests/test_scheduler_selection.py`、`tests/test_stage_engine.py`、`tests/test_serving.py`、`tests/test_async_speculative_random.py`、`tests/test_fp8_kv_cache.py`、`tests/test_dory*.py`、`tests/test_ascend*.py`|

上述测试文件是阅读入口；本笔记的静态代码核对不代表已在 GPU/NPU 上重新运行这些测试，也不构成精度或性能结论。
