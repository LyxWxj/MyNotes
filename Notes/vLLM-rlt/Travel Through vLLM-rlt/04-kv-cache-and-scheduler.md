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


##### KV 页分配、增长与回收

```text
lookup_prefix(tokens):
	-> poll_prefixes() 先发布 event 已完成的候选
	-> 逐页计算链式 token hash，遇到第一处 miss 停止
	-> 命中的 key 移到 LRU 末尾，返回每个逻辑页跨 plane 的物理 block

allocate(request_id, max_tokens, initial_tokens, prefix):
	-> 先给共享的 prefix blocks 加引用，避免后续 _claim() 将它们当成可回收页
	-> _claim() 为每个 plane 分配剩余物理页，失败时撤销刚加的引用
	-> 各 plane 的 block table = 命中的页 + 新页，written.prefix 从已命中长度开始

ensure_capacity(request_id, tokens):
	-> 校验 tokens 在逻辑上限内，计算还缺几页
	-> 为所有 plane 补页，追加到原 block tables
	-> 新页的有效范围由后续 KV 写入推进

free(request_id):
	-> 有 transfer lease 时设置 release_requested，保留当前 allocation
	-> 没有 lease 时移除 allocation，逐 block 减引用，引用归零才回到空闲池
	-> unpin_transfer() 移除最后一个 lease 时处理延期的 free()
```

`_claim()` 先用空闲页，短缺时才检查可淘汰的 prefix 页。若加上可淘汰页仍不够，直接返回失败。否则从 LRU 最旧的条目开始减引用，直到凑够页数。prefix 索引自己也持有引用，因此一次请求结束后，其完整 prompt 页仍可服务后续命中。命中和发布都为最后一个 prompt token 留出重算空间，CODA 需要它的 hidden。

`truncate_suffix()` 用于同步推测在拒绝后缩回逻辑有效范围。它保留物理页，只移除 frontier 之后的 written coverage。调用前需要确保设备访问已结束，且待裁剪范围没有共享页或传输 lease。

代码：`vllm_rlt/core/kv_cache_manager.py`。

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

