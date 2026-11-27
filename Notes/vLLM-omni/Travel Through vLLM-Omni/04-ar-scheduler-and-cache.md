#### OmniARScheduler、KV 与 prefix cache

AR stage 的调度核心是 **OmniARScheduler**，在 vLLM Scheduler 上叠加 Omni 输入协调、流式输入、connector 数据到达、部分 request admission 与 cleanup 语义。每次 schedule 前会清扫已 abort 的排队请求、消费 ready 的 Omni inputs，再委托上游 scheduler 选取本轮 token 工作；之后包装 Omni scheduler output，并附带跨 stage transfer 等元数据。

Prefill / decode、KV block 分配和模型执行仍分属 vLLM scheduler、KV cache manager 与 worker/model runner。Omni 的 scheduler mixin 处理多模态输入、异步 chunk/full-payload 的等待或就绪、超时与终态同步。Prefix cache 相关的 manager/controller/block pool/runner adapter 在 **core/prefix_cache**；它解决可复用 KV 前缀状态，不等于 stage 间 payload connector。

- `vllm_omni/core/sched/omni_ar_scheduler.py#L203`、`vllm_omni/core/sched/omni_ar_scheduler.py#L1297`
- `vllm_omni/core/sched/omni_scheduler_mixin.py#L739`
- `vllm_omni/core/prefix_cache/`
- `docs/design/module/ar_runtime.md`、`docs/design/module/cache_management.md`

### 逻辑完成不总等于设备资源可复用

当 request 结束或被取消，scheduler 可以立即把它从逻辑队列移除；但对应 KV block 若仍被先前 GPU work 读取，就不能马上交给另一个 request 覆写。当前实现存在 deferred-free fence：完成一个非空 scheduled step 后，update 路径推进已处理 step 序号并 drain deferred frees；空 step 不推进 fence。阅读缓存代码时要把“request terminal”“scheduler bookkeeping 清掉”“设备上的写/读完成”“block 可再次分配”当成四个不同时间点。

**OmniARScheduler.update_from_output** 中记录了对 GPU write 完成的假设，并在启用延迟释放时处理 fence。它是 scheduler/设备生命周期的交界处，涉及缓存复用时值得优先追读。

- `vllm_omni/core/sched/omni_ar_scheduler.py#L1363`、`vllm_omni/core/sched/omni_ar_scheduler.py#L2445`
- `tests/core/sched/test_omni_sched_deferred_free_fence.py`
- `tests/core/test_prefix_cache.py`

### Speculative decode 的边界

Omni 对 AR scheduler 做 draft/k-step 限制、排队和输出适配；真正的模型执行、验证和 token 接受语义还横跨 vLLM worker/model runner 及上游 scheduler。不要仅凭 Omni 的 draft 状态字段，推断整个 speculative acceptance/rollback 都在 OmniARScheduler.update_from_output 中完成。排查时沿着 scheduler output → model runner → sampled/accepted ids → request 状态更新这条线走，并查看对应测试。

- `tests/core/sched/test_omni_ar_scheduler_spec_decode.py`
- `tests/core/sched/test_omni_ar_scheduler_kstep_guards.py`
- `vllm_omni/worker_v2/omni_ar_model_runner.py`

### 流式输出与输入

AR 可以增量返回 token，也能接收 streaming input/update。流式请求的 request state、输入到达计数、队列清理和最终输出必须一起理解；取消可能在 request 仍排队、正在等多模态输入、正在运行或已经有部分输出时到来。对应入口包括 **AsyncOmni** 的 input update / abort 和 **OmniSchedulerMixin** 的 wait/input cleanup。

- `tests/core/sched/test_omni_ar_scheduler_streaming.py`
- `tests/core/sched/test_omni_scheduler_ready_inbox.py`
- `tests/core/sched/test_omni_ar_scheduler_free_request_cleanup.py`
