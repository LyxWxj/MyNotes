## 4. EngineCore：输入队列、调度一步、输出一步

### 4.1 client 选择与 transport

`EngineCoreClient.make_client()` 在 `vllm/v1/engine/core_client.py:L111-L183` 中按两个布尔量选择：

- `multiprocess=False`：`InprocClient`，直接持有 `EngineCore`；
- `multiprocess=True, asyncio=False`：`SyncMPClient`，同步阻塞读取 output queue；
- `multiprocess=True, asyncio=True`：`AsyncMPClient`，异步读取 output socket。

`InprocClient.get_output()`（`core_client.py` 的 in-process 实现）直接调用 `engine_core.step_fn()`，再执行 `post_step()`。多进程 client 的 `_send_input()` 把 ADD/ABORT/UTILITY 等消息送到 EngineCore 进程，output socket 线程负责把 `EngineCoreOutputs` 或异常转回 frontend。

`EngineCoreProc.run_engine_core()` 和 socket 线程位于 `vllm/v1/engine/core.py:L1328` 之后、`L2185-L2330` 附近：输入线程解码 ZMQ frame、完成 EngineCore ready handshake、把 request 放入 `input_queue`；输出线程把 output queue 中的结果编码发回 client。这里的队列是跨线程/跨进程缓冲，不等于模型已经完成。

### 4.2 busy loop 与普通 step

EngineCore 的主循环在 `core.py:L1801-L1887`：

```text
_process_input_queue()
    ↓
_process_engine_step()
    ├── scheduler.schedule()
    ├── model_executor.execute_model(non_block=True)
    ├── 等待/轮询结果
    ├── scheduler.update_from_output()
    └── output_queue.put(EngineCoreOutputs)
```

普通 `EngineCore.step()` 位于 `core.py:L701-L735`：

1. scheduler 没有 request 时返回空结果；
2. `scheduler.schedule()` 生成 `SchedulerOutput`；
3. `model_executor.execute_model(..., non_block=True)` 提交 worker；
4. `_wait_for_output()` 等待 future；若 worker 返回 `None`，再调用 `sample_tokens()`；
5. 先处理执行期间到达的 abort，再调用 `scheduler.update_from_output()`；
6. 附加 iteration details 和 weight version，交给上层 output processor。

当 `max_concurrent_batches > 1` 时使用 `step_with_batch_queue()`（`core.py:L746-L871`）：先填充 future 队列，再阻塞等待最老的 batch；structured-output sampling 可能放进 `deferred_step`，避免等待 grammar output 时破坏 forward overlap。这个队列的语义是“已提交但尚未完全消费的 batch”，不是已完成 batch 列表。

### 4.3 GPU future 等待期间仍可处理控制消息

多进程 EngineCore 的 `_wait_for_output()` 在 `core.py:L1942-L1969` 对支持 `poll()`/`wait_briefly()` 的 future 做短暂 spin；在 future 完成前，仍会处理 ADD、ABORT、FAIL、WAKEUP 等消息，其他消息暂存到 deferred input。这个设计把“模型正在算”与“控制面必须响应”分开：abort 可以先标记 scheduler/request 状态，最终 output 到达时由 scheduler 的 stale-output 分支丢弃或回收。

## 5. Scheduler：把请求变成一张可执行的 batch

### 5.1 Request 的逻辑状态

`vllm/v1/request.py:L63-L270` 的 `Request` 保存 prompt/generated token、`num_computed_tokens`、`num_in_flight_tokens`、`num_output_placeholders`、`last_sched_seq`、speculative token、multimodal 输入、KV transfer 状态和 streaming 状态。`num_tokens_with_spec` 与 `get_max_cache_hit_length()` 位于 `L342-L382`：前者把当前逻辑 token 加上 draft token，后者在 prefix hit 计算时保留最后一个 token 重新计算的边界。

`RequestStatus` 在 `request.py:L427-L475` 定义了 `WAITING`、`WAITING_FOR_REMOTE_KVS`、`WAITING_FOR_STREAMING_REQ`、`RUNNING`、`PREEMPTED` 以及 STOPPED/LENGTH/ABORTED/ERROR 等 finished 状态。状态名表达的是调度生命周期，不代表 GPU stream 已经停止。

### 5.2 admission 队列与统一的 token frontier

`Scheduler.__init__()` 在 `vllm/v1/core/sched/scheduler.py:L96-L260` 创建：

- `waiting`：新请求或被抢占后重新等待；
- `kv_holding_waiting`：本地/远端 KV 已有 ownership 但还不能运行；
- `async_kv_waiting`：等待异步 connector load；
- `deferred_waiting`：被结构化输出、pause 或其他条件延迟；
- `running`：已经有 worker state、继续消耗 token budget 的请求。

scheduler 的核心抽象不是一个固定的“prefill phase / decode phase”枚举。`schedule()` 的注释和实现（`scheduler.py:L702-L713` 起）让每个 request 追赶 `num_computed_tokens` 到 `num_tokens_with_spec`；因此 chunked prefill、普通 decode、prefix cache、speculative decoding、jump decoding 都可以由同一套 token budget 表达。`CachedRequestData.is_context_phase()` 只提供统计/worker contract 的上下文信息，并不是 scheduler 的全局模式切换。

### 5.3 schedule() 的资源决策顺序

`Scheduler.schedule()`（`scheduler.py:L702-L1816`）可按以下顺序阅读：

```text
计算本步 token budget / encoder budget / lookahead budget
        ↓
先处理 running：为已运行请求追加 token
        ↓
allocate_slots：命中本地/远端 KV，必要时申请新 blocks
        ↓
空间不足：按 policy 选择低优先级/后进入的 request preempt
        ↓
再处理 waiting：LoRA、prefix hit、KV connector、grammar、encoder
        ↓
生成 SchedulerOutput，并推进 sched_step_seq
```

- running 请求的新增 token 受 `max_num_scheduled_tokens`、模型长度、encoder budget 和 speculative lookahead 限制；
- `kv_cache_manager.allocate_slots()` 失败时，scheduler 会调用 `_preempt_request()` 释放候选请求的 KV blocks，再尝试当前 request；
- waiting request 会先查本地 prefix cache，再处理 connector 的 remote match/async load；远端 KV 尚未到达时进入 `WAITING_FOR_REMOTE_KVS`，不会把未完成 load 当作可读的 GPU block；
- `SchedulerOutput` 包含新请求完整 payload、旧请求 diff、每个 request 的 token 数、scheduled spec IDs、finished/preempted IDs、KV connector metadata、`new_block_ids_to_zero` 和 CoW copies（`vllm/v1/core/sched/output.py:L40-L95、L138-L191、L241-L325`）。

### 5.4 preemption 与 stale output

`_preempt_request()` 在 `scheduler.py:L1831-L1877` 只处理 RUNNING request：释放或延迟释放 KV/encoder state，把 request 标为 `PREEMPTED`，回滚 computed/spec placeholder bookkeeping，然后插回 waiting。若同一个 request 在旧 batch 仍在 GPU 或 pipeline 中，scheduler 会记录 stale output token 数；`update_from_output()` 在 `L2404-L2427` 看到 stale output 时按策略丢弃，避免旧 batch 覆盖新状态。

`_update_after_schedule()`（`scheduler.py:L1879-L1916`）会在提交后乐观增加 `num_computed_tokens` 和 `num_in_flight_tokens`，记录 `last_sched_seq`，并为 prefill chunk/encoder input 做 bookkeeping。这个“乐观推进”是调度层状态；实际 token 是否被接受、是否需要回滚，要等 worker output 回来。

### 5.5 output 如何推进逻辑进度

`update_from_output()` 在 `scheduler.py:L2334-L2660`：

1. 用 worker 输出减少 `num_in_flight_tokens`；
2. 跳过已 abort/finished 的 request，处理 KV load failure；
3. 对 speculative request 计算 accepted/rejected 数；
4. 把 rejected draft 从 `num_computed_tokens` 和 placeholders 中减回去；
5. 追加 sampled token，判断 stop、length、repetition、structured-output 结束条件；
6. 对继续运行的 request 重新入队，对完成的 request 生成 `EngineCoreOutput`。

因此一个 `EngineCoreOutput` 是“这个调度步对 frontend 可见的逻辑结果”，不是“所有相关 GPU memory 都已可回收”的证明。

