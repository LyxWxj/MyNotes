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

