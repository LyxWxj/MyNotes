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
	-> cache.poll_prefixes() 推进本地 prefix 发布
	-> transport.poll() 回收已完成的 WRITE handle 和字节额度，接收对端的 sequence 通知
	-> 轮转遍历 transfer，尝试发送 ready event 已完成且额度允许的 chunk
		-> 源端全部 chunk 已提交后通知 commit(count)
		-> 目标端收到 commit 且拿齐全部 sequence 后 accept_import()，通知 received
	-> 源端完整 prompt 的 staging event 完成后释放计算状态，已 commit、收到 ack 且无在途 WRITE 后释放传输资源
```

随后由 `StageWorker.run()` 驱动 `engine.step()`。`StageWorker.output()` 发送结果，若结果标记请求结束，再通知 `connector.output_finished()`。

In `vllm_rlt/core/kv_cache_manager.py`:

```text
KVCacheManager.poll_prefixes():
	轮询处理KV已经写入GPU,但Prefix Cache索引还不能立即建立的情况。
	-> pending = self._pending_prefixes and clear the pending_prefixes
	-> for request_id, allocation, tokens, length, event in pending:
		-> 若 request 当前的 allocation 已不是记录中的对象，跳过这条记录
		-> 若 event 存在且 event.query() 为 False，放回 _pending_prefixes 等下一轮
		-> self.publish_prefix(request_id, tokens, length) # 发布前缀
```

(此处可以优化对于列表的重复读写修改)

`KVCacheManager._pending_prefixes` 记录了若干五元组 `(request_id, allocation, tokens, length, event)`

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
		-> 若有 plane/layer 的连续写入前缀不足 end，到此停止发布
		-> 取每个 plane 在逻辑页 index 对应的物理 block，加引用并登记 self._prefixes[key] = blocks
```

