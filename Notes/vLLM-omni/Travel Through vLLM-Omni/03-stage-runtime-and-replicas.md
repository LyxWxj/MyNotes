#### StageRuntime、StagePool 和 replica

逻辑 stage 可以配置多个 replicas。StagePool 在提交初始请求时选择副本并保留绑定；同一 request 的更新、abort 和跨 stage session 路径要回到对应副本。分布式 stage runtime 通过 coordinator/master 与 membership 控制器管理远端 replica 可见性；选择器只应把 live/attached/serviceable 的 replica 作为候选。

某些 pipeline 还有显式 replica pin 或 session holder。pin 不只是调度提示，它影响 session state 和 KV locality 的归属；释放绑定必须跟 request terminal/abort 生命周期一致。PD disaggregation 则由 pipeline/deploy 配置与 stage role 协同表达，prefill 和 decode stage 间传递 KV / hidden-state 等状态。读 PD 时要分别看 orchestrator routing、connector transport 与接收侧 cache admission。

- `vllm_omni/engine/stage_runtime.py#L1543`
- `vllm_omni/engine/stage_pool.py#L608`、`vllm_omni/engine/stage_pool.py#L770`
- `docs/design/feature/disaggregated_inference.md`
- `tests/engine/test_stage_pool_session_affinity.py`、`tests/engine/test_membership_controller.py`
