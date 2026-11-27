#### Abort、错误、shutdown 与资源生命周期

### Abort 是一条跨 owner 清理链

调用方取消后，入口向 engine 发 abort。fire-and-forget abort 只保证投递控制消息；带 RPC id 的 **abort_async** 会等待 correlated result。Orchestrator 将 abort 传给各 StagePool/client，收集最终 AR stage 可返回的部分输出，再释放 request-to-replica binding、删除 orchestration state 和相关 companion state。Diffusion abort 走 stage client 到 DiffusionEngine，再由 scheduler/executor/worker 清理对应请求。

因此要区分三种完成：abort 控制消息成功入队、stage owner 已执行取消、orchestrator 收到 ack 并清理完共享状态。

### 错误传播

Stage 内 request error 通过 ErrorMessage 或 stage output error 反馈；Orchestrator 区分 request error 与 replica/process 故障。前者通常终止当前 request，后者可能需要标记 dead replica 并让请求失败。跨 stage input processor/connector 失败应有终态错误；不能留下已登记但无人再推进的 request。

- `vllm_omni/engine/omni_engine_base.py#L842`、`vllm_omni/engine/omni_engine_base.py#L859`
- `vllm_omni/engine/orchestrator.py#L907`、`vllm_omni/engine/orchestrator.py#L2188`
- `vllm_omni/engine/orchestrator.py#L1822`、`vllm_omni/engine/orchestrator.py#L3207`

### Shutdown

Shutdown 的顺序应沿 ownership 反向理解：停止接新请求与 orchestrator loop，通知 stage pool/client 结束，停止 replica/worker，关闭共享队列、transfer/runtime 资源。若只在入口 task 结束时看见请求已取消，不能推断远端 worker 和 connector buffer 已释放。

- `vllm_omni/engine/omni_engine_base.py#L977`
- `vllm_omni/engine/stage_runtime.py#L705`
- `docs/design/module/stage_runtime.md`
