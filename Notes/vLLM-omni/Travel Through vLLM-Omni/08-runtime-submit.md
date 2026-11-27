#### Stage submit

In `vllm_omni/engine/stage_pool.py`、`vllm_omni/engine/orchestrator.py` 和 `vllm_omni/diffusion/diffusion_engine.py`

```text
StagePool.submit_initial()
  -> 校验 explicit pin、session holder、stage input
  -> pick/bind physical replica
  -> LLM: output_processor.add_request()
       -> EngineCoreClient.add_request_async()
  -> diffusion: client.add_request_async()
       -> DiffusionEngine.add_request()
  -> 返回提交结果或 rollback/error
```

`submit_initial()` 是“把工作交给 stage owner”的边界，不是设备完成边界。LLM request registration、diffusion scheduler admission、executor submit、GPU output 可读和下游消费必须分开记录。

|提交动作|状态变化|不能据此推出|
|---|---|---|
|request queue 入队|orchestrator 收到请求|stage 已执行|
|EngineCore request registration|LLM scheduler 拥有 request|首个 token 已生成|
|DiffusionEngine.add_request|diffusion scheduler 拥有 request|denoise 已执行|
|executor submit/ticket|worker/device 已排队|host output 或 tensor 可复用|
|NextStageInputMessage 入队|下游提交已登记|下游已经消费 payload|

下游提交由 `Orchestrator._submit_to_stage()` 统一完成；消息中的 source/receiver stage、stage output、submit kwargs、sampling params 和 requested replica 共同构成下一阶段输入。stage client 负责通信，Orchestrator 仍负责路由。

代码：`vllm_omni/engine/stage_pool.py: submit_initial / submit_update`、`vllm_omni/engine/orchestrator.py: _submit_to_stage`。

