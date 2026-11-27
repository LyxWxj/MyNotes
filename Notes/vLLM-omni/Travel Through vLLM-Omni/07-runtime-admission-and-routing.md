## Runtime

### 请求准入与跨 stage 路由

### Orchestrator 与 StagePool

Orchestrator 收到初始消息后创建 orchestration request state 并提交 stage 0。每个 stage 的原始输出被处理成 Omni 输出；未终止的输出可以继续发给调用方或留在本 stage，完成一个中间 stage 的输出则进入跨 stage route。下一 stage 的提交由 Orchestrator 发起，replica 由对应 StagePool 选择。

StagePool 的选择可结合显式 replica pin、session affinity、分布式服务可用性和负载策略；请求进入后保留 request 到 replica 的绑定，后续更新和 abort 才能落到实际持有状态的副本。不要把“逻辑 stage id”与“物理 replica id”混为一谈。

- `vllm_omni/engine/orchestrator.py#L4295`、`vllm_omni/engine/orchestrator.py#L2501`
- `vllm_omni/engine/orchestrator.py#L3207`、`vllm_omni/engine/orchestrator.py#L3639`
- `vllm_omni/engine/stage_pool.py#L538`、`vllm_omni/engine/stage_pool.py#L1603`
