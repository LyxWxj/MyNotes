## 9. Speculative decoding：验证、接受、回滚

本项目的 speculative 运行时不是 frontend 特殊分支，而是 scheduler/worker contract 的扩展：

1. scheduler 在 `SchedulerOutput.scheduled_spec_decode_tokens` 中为 request 发送本步要验证的 draft IDs（`sched/output.py:L251-L260`）；
2. model runner 在 target forward 中验证 draft，sampler 返回 sampled IDs，同时统计 drafted/verified token 数；
3. `AsyncOutput` 在 D2H copy 中保存这两个计数（`model_runner.py:L2361-L2387`）；
4. target output 可见后，worker 在同一个 execute path 中调用 `speculator.propose()` 产生下一轮 draft，并写入 request state（`model_runner.py:L2426-L2479`）；
5. scheduler 在 `update_from_output()` 中计算 accepted/rejected，并把 rejected draft 从逻辑 `num_computed_tokens` frontier 和 output placeholders 中回滚（`scheduler.py:L2434-L2470`）。这是请求 bookkeeping 的回滚，不表示物理 KV bytes 立即被擦除或回滚。

可以用这个时间线理解：

```text
step t schedule: target frontier + K draft slots
        ↓
step t forward: target 验证旧 drafts，sampler 给 accepted + bonus token
        ↓
step t worker: proposer 计算 step t+1 drafts
        ↓
step t scheduler update: accepted 留下，rejected 回滚逻辑 frontier
        ↓
step t+1 schedule: 根据新 frontier 再分配 slots
```

`Request.num_tokens_with_spec` 会把 draft 纳入本步可调度 token；但 draft 不是永久 token。`update_draft_token_ids()`（`scheduler.py:L3049-L3095`）在 prefill chunk 时会忽略或裁剪 draft，避免把尚未稳定的 speculative state 当作 prompt cache。preemption 也会清理 draft 与 placeholder，因此 stale output 不能再次接受旧 draft。

建议继续阅读的 spec 实现是 `vllm/v1/worker/gpu/spec_decode/speculator.py:L74-L290`（`BaseSpeculator`、`DraftModelSpeculator`）和 `vllm/v1/worker/gpu/sample/rejection_sampler.py`；它们负责 proposal/rejection 的设备侧细节，而 acceptance 的 request-level bookkeeping 仍在 scheduler。

## 10. 异步调度、pipeline/data parallel 与 disaggregated KV

### 10.1 async scheduling 与 batch queue

scheduler 的 `async_scheduling` 配置会让 EngineCore 尽量不在每一步同步等待所有控制面输入；batch queue 则把多个 `SchedulerOutput`/future 保留在 EngineCore 中。`step_with_batch_queue()` 的约束是：先尽量提交新 batch，队列满或无法继续调度时，才等待最老 future；structured-output grammar 可能把 sampling 延后到 `deferred_step`。这提升吞吐，但也让 `num_in_flight_tokens`、stale output 和 deferred free 成为必要 bookkeeping。

### 10.2 pipeline parallel（PP）

PP 的 ownership 分为两层：

- scheduler 在 `SchedulerOutput` 中给每个 request 发送新 request data 与 block/token diff；
- GPUWorker 在 stage 间发送 intermediate tensors，并保存 send handles；下一步开始前等待 handle，防止 buffer reuse。

末端 PP rank 负责 sampling；`GPUModelRunner` 会把 sampled IDs、draft/verified counts 广播到其他 rank（`model_runner.py:L2334-L2341`），使各 stage 的 request state 继续一致。

### 10.3 data parallel（DP）

`LLMEngine` 初始化时先建立 DP group（`llm_engine.py:L79-L96`）。`has_unfinished_requests_dp()` 会跨 rank 聚合是否还有请求；某个 rank 没有本地请求、但其他 rank 仍有请求时，设置 dummy batch 标记（`llm_engine.py:L202-L217`）。worker 的 `dispatch_cg_and_sync_dp()` 也会让各 rank 的 batch descriptor 对齐。dummy run 只为通信/graph shape 对齐，不能当作真实请求输出。

### 10.4 disaggregated prefill、KV offload 与远端 KV

scheduler 在初始化时以 `SCHEDULER` 角色创建 KV connector，并根据 connector 能力设置 `defer_block_free`、remote match、async load 和 KV delivery requirement（`scheduler.py:L96-L260`）。请求可以带 `kv_transfer_params`/`ec_transfer_params`；EngineCore admission 会在没有 connector 时警告并禁用对应 transfer（`core.py:L576-L597`）。

waiting request 的 remote KV 路径大致是：

```text
本地 prefix lookup
→ connector 查询 remote match
→ 同步 load：当前 step 消费
→ 异步 load：WAITING_FOR_REMOTE_KVS / async_kv_waiting
→ load 成功：把 external blocks 纳入 request
→ load 失败：rollback / reschedule，跳过失败输出
```

`SchedulerOutput.kv_connector_metadata` 和 `KVConnectorBlockState`（`sched/output.py:L300-L321`）是 scheduler 到 worker/connector 的显式 contract。`_free_request()` 先调用 connector 的 finished hook，再判断 block 是否能立即释放，因此 P/D 分离、offload 或跨节点传输不会破坏 block ownership。

相关文档入口：`docs/features/disagg_prefill.md`、`docs/features/kv_offloading_usage.md`；实现入口：`vllm/v1/kv_offload/`、`vllm/distributed/kv_transfer/`、`vllm/v1/core/sched/scheduler.py` 中的 connector 分支。

