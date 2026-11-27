#### Output update 与 materialization

### 输出回到调用方

StagePool client 的输出由 orchestrator loop 消费。LLM 输出先经过 stage output processor；diffusion 输出有其 stream/result pump。Orchestrator 根据 request state 和输出是否终态执行路由、附加指标、产出 OutputMessage 或转发到下游。入口侧只有一个 output pump 从 engine 拉取输出，再按 request id 分派到请求级 queue，最终恢复外部 id 并 yield。

默认 orchestrator 是 event-driven：LLM replica 使用 reader task 等待输出，diffusion 有对应 poller，再由串行 dispatch queue 保留统一路由语义。环境变量 **VLLM_OMNI_EVENT_DRIVEN_ORCH=0** 选择 legacy poll loop。两种 loop 复用处理和路由逻辑。

- `vllm_omni/engine/orchestrator.py#L1135`、`vllm_omni/engine/orchestrator.py#L1471`
- `vllm_omni/engine/orchestrator.py#L1761`
- `vllm_omni/entrypoints/async_omni_base.py#L408`、`vllm_omni/entrypoints/async_omni_base.py#L328`
