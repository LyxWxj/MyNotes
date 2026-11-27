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


##### 普通采样与 `RngRegistry`

```text
CODA logits -> Sampler.sample(logits, params, generator):
	-> temperature == 0: 直接 argmax
	-> 否则要求该请求的 generator，调用 probabilities()
		-> logits 转 FP32，再除以 temperature
		-> top-k 按索引保留最多 k 项，其余设为 -inf
		-> top-p 按分数排序，保留累积概率跨过阈值的那一项，裁掉后续项
		-> 对保留项重新 softmax
	-> multinomial 抽一个 token，返回设备上的标量 tensor
```

`RngRegistry.acquire()` 首次按请求的 `seed` 和采样设备创建 generator。普通和 speculative runner 共用 StageEngine 持有的 registry，因此 CODA bootstrap 之后的随机推测会延续该请求的 RNG 状态。抢占挂起时保留 generator，结束或取消释放时才移除。greedy 采样不消耗 RNG，模型自身的随机状态初始化仍可借用同一个 registry。

代码：`vllm_rlt/worker/sampler.py`。

