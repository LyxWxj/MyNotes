### 验证入口与阅读路线

| 观察点 | 需要区分的状态 |
| --- | --- |
| 请求路由 | logical stage 与 physical replica；入口 id 与 orchestrator id |
| 异步提交 | submit/enqueue 已完成，与 worker 执行/下游消费完成 |
| 设备输出 | producer stream 已写，与 consumer 已 wait event、storage 可安全复用 |
| request 终态 | caller 已收到终态，与各 stage 清理、绑定释放、设备缓存可复用 |
| 缓存 | logical block 已 free，与先前 GPU work 对 block 的访问已越过 fence |
| connector | connector 做 payload transport/sync，与 orchestrator 做 stage routing |
| batch | diffusion batch slot，与稳定 request id 及 per-request progress |
| retry/preempt | 同一 request 的续跑与新 request 的重试、full-payload position state |

遇到卡住或数据丢失，优先问：当前状态由哪个 owner 持有？它等待哪个 event/消息/ack？超时由谁触发？terminal cleanup 是否覆盖了队列、cache、connector pending state 和 replica binding？

### 验证入口地图

以下是依据 design index 与相关页面整理的测试导航，本次没有执行这些测试。

| 要验证的行为 | 建议入口 |
| --- | --- |
| engine 输入/输出/abort 与 message correlation | `tests/engine/test_async_omni_engine_input.py`、`tests/engine/test_async_omni_engine_outputs.py`、`tests/engine/test_async_omni_engine_abort.py`、`tests/engine/test_rpc_result_router.py` |
| 多 stage routing、错误传播、事件循环 | `tests/engine/test_orchestrator.py`、`tests/engine/test_orchestrator_error_handling.py`、`tests/engine/test_orchestrator_event_driven.py` |
| runtime startup、replica/session binding | `tests/engine/test_async_omni_engine_stage_init.py`、`tests/engine/test_stage_runtime_launch_concurrency.py`、`tests/engine/test_stage_pool_session_affinity.py` |
| AR prefill、streaming、spec decode、preemption、KV transfer | `tests/core/sched/`，尤其 `tests/core/sched/test_omni_ar_scheduler_prefill_admission.py`、`tests/core/sched/test_omni_ar_scheduler_streaming.py`、`tests/core/sched/test_omni_ar_scheduler_spec_decode.py`、`tests/core/sched/test_omni_recompute_preemption.py` |
| Prefix-cache manager / adapter | `tests/core/test_prefix_cache.py`、`tests/core/test_prefix_cache_adapter.py`、`tests/core/test_prefix_cache_mrv2.py` |
| Diffusion scheduler / cancel / worker result | `tests/diffusion/test_diffusion_scheduler.py`、`tests/diffusion/test_request_cancellation.py`、`tests/diffusion/test_result_pump.py`、`tests/diffusion/test_diffusion_worker.py` |
| Diffusion batching 与 request lifecycle | `tests/diffusion/batching/test_diffusion_batching.py`、`tests/diffusion/test_step_request_lifecycle.py` |
| connector / full payload / output ownership | `tests/worker/test_full_payload_lifecycle.py`、`tests/worker_v2/test_omni_ar_model_runner.py`、`tests/worker_v2/test_omni_data_plane.py` |

### 按问题继续深入

- 想知道某个 model_type 到底有哪些 stage：从 **OMNI_PIPELINES** 查起，再看对应 deploy yaml、StagePipelineConfig 和 StageRuntime plan。
- 想追一个请求为何卡在 stage N：先定位 orchestrator request state 与 StagePool binding，再检查该 stage 的 client 输出 pump 和 scheduler waiting state。
- 想追多模态 payload 丢失：区分 output snapshot、native data plane enqueue、connector delivery、receiver readiness 和 stage routing。
- 想追 KV/cache 误复用：确认 scheduler free 状态、deferred-free fence、prefix-cache identity/invalidation 和 GPU stream completion。
- 想追取消泄漏：沿入口 abort → Orchestrator cleanup → 每个 StagePool.abort_requests → stage scheduler/worker cleanup → connector pending payload 清理。
- 想追 diffusion batch 不兼容：从 RequestScheduler compatibility key 或 StepScheduler 的 per-request progress 入手，不从 pipeline model class 开始猜。

### 本次核验状态

- 已读取贡献指南、CI review/issue lessons、design index，以及与 engine orchestration、stage runtime、AR、cache、diffusion、connector 和 async output 相关的设计页；关键描述与源码位置交叉核对。
- 已定位上述测试文件并把它们作为后续行为验证入口。
- **本次未执行 pytest、GPU/e2e 或模型加载。** 本笔记回答的是“从源码和当前设计页如何追运行时”，不声称设备级行为已在本环境实测。
