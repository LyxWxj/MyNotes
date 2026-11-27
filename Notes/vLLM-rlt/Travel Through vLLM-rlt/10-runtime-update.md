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

