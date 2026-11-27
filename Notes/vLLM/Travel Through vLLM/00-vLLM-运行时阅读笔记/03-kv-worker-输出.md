## 6. KV cache：prefix hit、分配、写入和回收

### 6.1 cache manager 与 block pool 的分层

`KVCacheManager` 位于 `vllm/v1/core/kv_cache_manager.py:L130-L226`。它通过 coordinator 统一多个 KV group，持有 `BlockPool`、watermark、KV event metadata 和 null block。`get_computed_blocks()` 在 `L278-L340` 查询最长 prefix hit；prefix cache 关闭时直接返回空，命中时会按最后 token recompute 边界、shared prefix 和 connector 外部 KV 约束返回 block wrapper。

`BlockPool` 位于 `vllm/v1/core/block_pool.py:L107-L174`：

- `blocks` 是物理 block 对象；
- `free_block_queue` 按 eviction 顺序保存可分配/可淘汰 block；
- `cached_block_hash_to_block` 负责 hash → cached block 查找；
- `null_block` 表示没有 backing storage 的逻辑占位。

`cache_full_blocks()`（`block_pool.py:L203-L329`）在 full block 的 hash metadata 完整后写入 prefix-cache map，并可产生 KV events。因而“缓存命中”是 block metadata/residency 的属性，不代表本步一定要执行 forward。

### 6.2 allocate 与 worker 的零填充/CoW 协议

`KVCacheManager.allocate_slots()` 从 `kv_cache_manager.py:L391` 起协调各 KV group 的可用 slots；单类型 manager 的具体分支在 `single_type_kv_cache_manager.py:L379-L535`：

- local prefix hit 的 block 会先 `touch`，让 ref count 跟随 request；
- external computed tokens 会分配承接远端 KV 的 blocks；
- partial prefix hit 会申请私有 CoW block，把共享 tail 复制后再继续写；
- 新 block ID 会记录到 `new_block_ids`，交给 worker 在 forward 前清零。

`SchedulerOutput` 对这两个操作有明确顺序说明（`sched/output.py:L312-L318`）：先清零 `new_block_ids_to_zero`，再应用 `kv_cache_block_copies`。`GPUModelRunner.update_requests()`（`vllm/v1/worker/gpu/model_runner.py:L1376-L1410`）也按这个顺序写 block table；所以 scheduler 不能把“分配”理解成“内容已经可读”。

**不变量（源码事实 + 运行时归纳）：**

```text
block 被分配 ≠ KV 已写入
block 被 prefix hit ≠ 本步无需检查 request frontier
CoW copy 被排队 ≠ copy 已完成
request finished ≠ block 可以立即回 free queue
```

### 6.3 延迟释放与物理安全边界

scheduler 的 `_free_request_blocks()` 在 `scheduler.py:L3267-L3284` 检查 `request.last_sched_seq <= processed_step_seq`。如果一个可能仍在 GPU step 或异步 KV connector 中使用的 block 尚未越过 fence，scheduler 会从 request 中摘出 block，放进 `deferred_frees`；等后续 output completion 推进 `processed_step_seq` 后才真正交给 `KVCacheManager.free()`。`KVCacheManager.free()` 位于 `kv_cache_manager.py:L641-L665`，最终落到 block pool 的 free 逻辑。

connector 可能进一步延长这个生命周期：`_free_request()`（`scheduler.py:L3220-L3252`）先调用 KV/EC connector 的 `request_finished()`，再释放 encoder cache，最后根据 connector 是否仍要写 block 决定是否延迟 free。这正是“请求逻辑上结束”与“物理 KV storage 可复用”之间的边界。

## 7. Worker 与 GPU：提交、stream、采样和输出可见性

### 7.1 GPU worker 的职责

`GPUWorker.execute_model()` 位于 `vllm/v1/worker/gpu_worker.py:L1501-L1564`。在每次 forward 前，它会等待上一轮 pipeline send handle，防止新 forward 覆盖仍被通信读取的 buffer；非首个 PP rank 接收 intermediate tensors；调用 `model_runner.execute_model()`；非末端 PP rank 以 non-blocking send 发送 intermediate tensors，并保存 handles 到下一步等待。

这条等待链说明 pipeline parallelism 中有两个独立 completion：GPU forward 的结果可供下一阶段读取，通信 buffer 的 send completion 才允许本 rank 重用同一 storage。

### 7.2 ModelRunner 的单步生命周期

`GPUModelRunner.execute_model()` 在 `vllm/v1/worker/gpu/model_runner.py:L1865-L2020`，可拆成：

```text
begin draft step
→ finish/free 旧 worker request states
→ add/update 本步 request states、block tables、LoRA、encoder cache
→ 清零新 block、应用 staged CoW
→ connector.begin_step
→ DP sync / CUDAGraph dispatch
→ prepare_inputs + prepare_attn
→ model forward
→ last PP rank sampler
→ AsyncOutput 记录 D2H copy event
→ postprocess request state
→ speculative proposer 生成下一轮 drafts
→ connector.post_forward / finalize_step
```

`finish_requests()`（`model_runner.py:L1286-L1300`）在 worker 侧删除 finished/preempted request state；`add_requests()`（`L1315-L1375`）建立新 request state；`update_requests()`（`L1376-L1410`）更新 computed token 和 block table。worker 的 request state 是执行缓存，不是 scheduler 的唯一事实来源；每一步由 `SchedulerOutput` 增量同步。

### 7.3 AsyncOutput：GPU stream 与 Python 输出的可见性

`AsyncOutput` 位于 `vllm/v1/worker/gpu/async_utils.py:L117-L248`。它保留 GPU tensor 引用，在 `output_copy_stream` 上等待 `main_stream`，把 sampled IDs、logprobs、nan count、draft/verified count 等异步拷到 CPU，并记录 blocking `copy_event`。`get_output()` 在 `L192-L248` 先 synchronize 这个 event，再把 NumPy 数据转换为 `ModelRunnerOutput`。

因此：

- `AsyncOutput` 被构造：D2H copy 已提交，但 Python 还不能读取结果；
- `AsyncOutput.get_output()` 返回：CPU 侧采样结果可读取；
- scheduler 收到 output：逻辑状态开始推进；
- deferred KV/PP/connector work 结束：物理资源才达到可复用条件。

这也是为什么 `EngineCore.step()` 必须等待 future，或者在 batch queue 中把 future 保留到最老 batch 完成；仅看“worker 方法已返回”不能推出“所有输出和通信都完成”。

## 8. 输出路径：从 token ID 到用户可见文本

`OutputProcessor` 定义在 `vllm/v1/engine/output_processor.py:L554-L665`。`add_request()`（`L666-L730`）建立 `RequestState`、外部 ID 到内部 ID 的映射、detokenizer 和可选的 async queue。`process_outputs()`（`L731` 起）是唯一遍历完整 `EngineCoreOutputs` batch 的 frontend 函数：

1. 找到对应 `RequestState`；已 abort 的输出直接忽略；
2. 更新统计、KV/EC transfer metadata、routed experts 等辅助字段；
3. 把新增 token ID 交给增量 detokenizer；
4. 生成 `RequestOutput`/`PoolingRequestOutput`，同步模式返回列表，异步模式写入该 request 的 queue；
5. 遇到 stop string 时返回 `reqs_to_abort`，让 EngineCore 真正从 scheduler 删除。

`BaseIncrementalDetokenizer.update()` 与 `get_next_output_text()` 在 `vllm/v1/engine/detokenizer.py:L115-L225`：detokenizer 会保留可能属于 stop string 的尾部字符，避免先把 stop string 的前缀流给客户端。故 `finish_reason`/stop 判断在 scheduler 与 frontend 之间有两层：scheduler 以 token-level 条件结束，OutputProcessor 以文本-level stop string 再请求 abort/截断输出。

同步输出由 `LLMEngine.step()` 返回；异步输出由 `_run_output_handler()`（`async_llm.py:L870-L965`）完成：

```text
await engine_core.get_output_async()
→ 按 VLLM_V1_OUTPUT_PROC_CHUNK_SIZE 切片
→ invalidate multimodal cache miss
→ output_processor.process_outputs()
→ 把 RequestOutput 放入各自 queue
→ abort stop-string 完成的 request
→ 更新 scheduler stats / logger
```

`AsyncLLM.generate()` 只消费自己的 queue（`async_llm.py:L803-L816`），所以一个大 batch 的 output processing 被切片后可以在 chunk 之间让出 event loop。

