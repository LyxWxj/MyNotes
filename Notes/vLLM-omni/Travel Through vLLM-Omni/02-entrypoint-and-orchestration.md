#### AsyncOmni 与 Orchestrator

### 4.1 入口与消息

以 AsyncOmni 为例，**generate()** 建立内部 request id 和前端请求状态，处理 admission、sampling 参数和输入；输出 handler 在后台持续从 engine 取消息，再路由到请求自己的 asyncio queue。引擎用消息对象跨越入口线程与 orchestrator loop：初始请求通常是 **StageSubmissionMessage**，stage 输出是 **OutputMessage**，控制请求包括 abort / interaction / streaming update。

跨 stage 时，**NextStageInputMessage** 带来源 stage、接收 stage、请求列表、stage 输出和下游提交参数。其校验检查 stage 顺序、请求类型与参数形态，降低把 diffusion 请求误交给 LLM client 等错误。

- `vllm_omni/entrypoints/async_omni.py#L150`
- `vllm_omni/entrypoints/async_omni_base.py#L328`、`vllm_omni/entrypoints/async_omni_base.py#L408`
- `vllm_omni/engine/messages.py`、`vllm_omni/engine/async_omni_engine.py#L539`
