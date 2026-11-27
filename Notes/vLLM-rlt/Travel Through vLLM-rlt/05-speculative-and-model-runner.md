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


##### SpeculativeRunner 的 round 执行

```text
同步 block:
	-> _prepare_inputs() 按每个 span 的 token_count 准备 draft steps 和 target metadata
	-> _submit_round(): 每个候选先跑 PRELUDE + draft_loops
		-> 用 draft CODA logits 提出下一 token，保持在设备上供下一 draft step 使用
		-> 按请求整理 draft hidden，续跑到 target_loops，再用 CODA 产出 target logits
	-> _collect_round(): 读回 draft IDs，按请求处理接受、修正或 bonus
	-> 返回 SpeculativeResult，由 Scheduler 提交 token，继续生成时按 frontier 裁剪未提交的 KV

AsyncSpeculativeRunner.submit():
	-> _prepare_round() 租用一个 round buffer，准备本轮 page tables
	-> _state() 首轮以已交付的 token/position 初始化设备状态，后续直接复用
	-> block: _draft() -> _verify() -> _accept_and_update_state()
		-> draft/verify 的实际位置来自设备 state，CPU token_start 只用于上界规划
		-> greedy 在 GPU 找首个不匹配的位置，随机行做 batched rejection sampling
		-> 把下一 token 和实际 position 写回设备状态，同时产出 compact result
	-> pipeline: _pipeline_step() 推进设备上的 token、depth、hidden、candidate 和 active 状态
	-> 记录 compute_done，copy stream 等待后读回 compact result，再记录 copy_done
	-> 返回 SpeculativeRoundTicket，CPU 按提交顺序 collect、校验、更新并 retire
```

Block 模式最多提出 `K` 个 draft，对应最多 `K + 1` 个验证行。Greedy 从头匹配 target argmax，首次不匹配时返回修正 token，全匹配时返回 bonus。随机采样用 draft 分布 `q` 提出候选，再用 target 分布 `p` 验证。候选以 `min(1, p(x) / q(x))` 的概率接受。首次拒绝时从归一化的 `max(p - q, 0)` 采样修正 token，全部接受时从最后一个 target 分布采样。异步实现批量处理不同 draft 长度，只考虑连续的接受前缀，并将实际停止位置的非法分布标记带回 Engine。

Pipeline 每个 step 推进 `draft_loops`。到达 draft 深度的行提供后继候选，到达 target 深度的队首行最多提交一个 token。接受后保留仍有效的后继 hidden/depth，拒绝后丢弃依赖错误候选的后继，从 target token 重新开始。同步模式在 collect 中处理这些决策，异步模式在同一 compute stream 上更新设备状态。后一轮可以在前一轮 CPU 交付前提交，计算仍按 stream 顺序读取前一轮更新后的状态。

代码：`vllm_rlt/worker/{speculative,async_speculative,speculative_sampling}.py`。

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


##### Runner metadata 与缓冲区复用

```text
MetadataBuilder.build(request_ids, depth_sets, positions):
	-> 校验每行的 allocation、depth 和 token position
	-> 按 plane 取 block table，算物理 block 和页内 offset，拒绝重复写地址
	-> plan_attention_metadata() 生成可见上下文、页表以及 packed query 边界
	-> 返回每组 depth 的 KVLayout，并保留 allocation 身份

stage_metadata(layouts) -> StagedInputTensor.upload() -> MetadataBuilder.bind():
	-> int64 保存 positions 和写入地址，int32 保存页表和长度信息
	-> CPU 准备先 append，再把每种 dtype 已用的区间整体快照和上传
	-> bind() 在设备缓冲上按原 shape、stride、offset 重建 KVMetadata 视图
```

`StagedInputTensor.append()` 返回描述本轮布局的 CPU slice。扩容会保留原数据和逻辑偏移。`upload()` 轮转取 pinned snapshot 和设备目标 buffer，通过 `copy_tensor()` 选择适用的 UVA 路径或普通拷贝。CPU 可以开始准备下一轮，而设备继续读上一轮的 snapshot。

|路径|准备内容|复用条件|
|---|---|---|
|普通 staging / `Workspace`|准备位置、写入地址、context lengths 与页表，Workspace 可复用固定布局的输入和 hidden|`SubmissionQueue.begin()` 在输入池轮转前确认旧计算结束，Workspace 也有自己的完成 event|
|packed prefill / `PackedMetadataBuffer`|按 sequence 存 token、position、sequence ID、长度、累计 query 边界和页表，上传后在设备上展开逐 token 写地址|整套 compact storage 归一个 submission slot 所有|
|async routing / `AsyncState`|按请求保存设备侧 hidden/token 和页表，copy stream 生成本批 slot 和 KV metadata，执行 stream 等 ready event，按 slot gather/scatter|routing buffer 独立用 done event 控制轮转，请求 slot 在最后一次使用结束后释放|
|split/mixed prefill / `BatchInputBuffers`|描述各 span 在 batch 与 chunk 槽中的行范围，`TensorTransfer` 把分散的 token/hidden 汇入 batch，并把需要保留的 hidden 写回|临时 buffer 随提交轮转，hidden ring 槽等连续 chunk 到达最终 depth 后才能接收新 chunk|
|readback slot|读回采样 token 或 gate score 到固定 host buffer|上次 DMA 完成且租约已释放，通常由 `collect()` 解除租约|

`KVMetadata` 保存 host 已知行和对应 allocation。`DeviceKVMetadata` 用在位置、block table 等由设备侧状态决定的路径，如异步 speculative。它仍携带 allocation 身份供校验，CPU 无需先读回每轮的位置。metadata 描述地址和视图，KV 页的分配和引用计数仍由 `KVCacheManager` 管理。

代码：`vllm_rlt/worker/{metadata,buffers,tensors,async_state,submission_queue}.py`。

