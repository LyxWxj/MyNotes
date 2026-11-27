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


##### `Scheduler._admit()`

```text
_admit():
	-> 从 WAITING 取候选，按 FIFO 或 priority 数值从小到大排列
	-> 按 admission_scan_limit 扫描，先检查 active slot
		-> RECEIVING 和已释放计算状态的请求不占 active slot
		-> priority 模式且允许挂起时，slot 不足可尝试抢占较低优先级请求
	-> 已收到完整导入的 RECEIVING 请求: 解除导入 lease，进入 CODA
	-> 已有 KV allocation 和 transfer lease 的源端请求: 直接进入 prefill
	-> 其余请求先 _try_resume()，有 snapshot 则尝试恢复到原 phase
	-> 首次准入: _plan_admission() -> _commit_admission()
		-> 预算不够时可回收仍驻留 KV 的 WAITING victim，再重算计划
		-> 分配成功后，将 prefill 进度和各 depth 边界初始化到 prefix 命中位置
		-> _enter_prefill(): 进入 PREFILL 或 PREFILLING_PRELUDE
	-> 暂时无法恢复或分配的请求放入 deferred
		-> 后续请求每成功准入一次，前面 deferred 的 bypass 计数加一
		-> 达到 max_admission_bypasses 后结束扫描，让在跑的请求有机会释放资源
	-> 按原扫描顺序把 deferred 放回 WAITING 队首
```

`_plan_admission()` 把逻辑上限、当前分配量和准入预算分开。逻辑上限是 `prompt 长度 + max_tokens - 1`。增量分配先映射命中前缀和首个 prefill chunk，其余页按需增长。关闭增量分配时一开始就分配到逻辑上限。

```text
AdmissionPlan.total_budget_blocks:
	-> 本请求预算内的页数 - 已命中的 prefix 页数（都乘 storage_depths）
	-> 加上现有请求尚未兑现的增长预算
	-> 加上 active_count 非零时的 watermark
	-> 加上本次命中后将失去可淘汰资格的 prefix-only block
	-> 总数不超过 cache.num_free_blocks 才实际 allocate()
```

开启增量分配时，无挂起回调会把输出增长也算进预算。有挂起回调时先覆盖 prompt，decode 增长不足再走抢占。prefix-only 页在命中前计入可释放容量，命中后多了请求引用，所以必须把这部分补回预算。`reserve_transfer()` 使用同一套预算，提前映射整个 prompt 并 pin transfer，之后仍要参与计算准入。

##### `Scheduler.take_phases()`

```text
take_phases(phases):
	-> 在 policy 给出的 phase 顺序里扫描队列，共用 token budget 和 max_num_seqs
	-> 检查 execution_ready（含 connector 背压）、prefill 窗口、speculative round/输出上限
	-> _make_scheduled_request(): 按 phase 快照 token_start、token_count 和 depth
		-> 普通 prefill 从 num_prefilled_tokens 开始，受 chunk size 和剩余预算限制
		-> split recurrent 只取当前 depth 边界内已完成 PRELUDE 的连续范围
		-> 普通 decode 执行单个位置，speculative 受 K + 1、剩余输出和 token budget 限制
	-> 对需要 KV 容量的 phase，_ensure_execution_capacity() 检查本轮的 token frontier
		-> 不够时尝试抢占，已选入本 batch 的请求受 selected 集合保护
	-> 选中后扣 token budget，记录 phase；条件暂不满足的请求留给后续轮次
	-> 按实际选中的工作更新公平性计数，再用 _planned_loop_count() 决定执行边界
	-> 返回 SchedulerOutput，mixed batch 用每个 ScheduledRequest 自带的 phase
```

`_take_wavefront_phase()` 先收集可运行的候选请求，再按 depth 轮询这些请求，让一个请求在有余量时贡献多个 span。PRELUDE 使用本轮临时 cursor 和窗口上界。RECURRENT 从各 depth 的 frontier 出发，范围受该 depth 的上界、`num_preluded_tokens` 和 chunk 边界共同限制。`_make_wavefront_task()` 以 `prefill_window_origin` 为基准算 chunk 编号，按 depth 数量取模得到 hidden ring slot。token budget 按行扣减，sequence budget 按请求计数。选完后按候选顺序和 token 位置整理 span，随后由提交更新推进实际 frontier。

`_planned_loop_count()` 决定这次提交能连跑几轮。普通 PREFILL 跑完整深度，独立 PRELUDE/CODA 为 0，普通 RECURRENT 为 1。wavefront 会在每个 recurrent depth 后交回调度。非 wavefront 的 split prefill 可以连跑同一段已满足依赖的深度，取全 batch 的共同上限。mixed 模式有可合批的 decode recurrent 时按轮交回调度，独立的 staged PRELUDE 也可先产出 hidden，等下一批合并执行。

代码：`vllm_rlt/core/scheduler.py` 中的 admission、selection 与 loop planning。

