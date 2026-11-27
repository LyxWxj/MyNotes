## Initialization

### Pipeline 配置解析

启动先根据模型配置/model_type 找 pipeline。**OMNI_PIPELINES** 注册的条目可以是静态 **PipelineConfig**，也可以是 resolver；解析后每个 **StagePipelineConfig** 描述逻辑 stage 的类型、输入来源、最终输出行为、执行方式等。deploy 配置再补充 replica 数、设备布局、并行方式和运行参数。

入口主要在 **OmniEngineBase.__init__ → _resolve_stage_configs**；stage 配置结构在 **PipelineConfig / StagePipelineConfig**。因此某个模型看到的执行拓扑，不能只从模型类推断，必须同时看注册表和 deploy 配置。

- `vllm_omni/config/pipeline_registry.py#L174`
- `vllm_omni/config/stage_config.py#L395`、`vllm_omni/config/stage_config.py#L491`
- `vllm_omni/engine/omni_engine_base.py#L303`、`vllm_omni/engine/omni_engine_base.py#L725`

### Engine loop 与 stage runtime 启动

OmniEngineBase 创建后台线程和 asyncio loop，在 bootstrap 中依次初始化 runtime、建立队列/控制通道、构造 Orchestrator，然后运行其主协程。StageRuntime 按逻辑 stage 生成初始化计划，对资源做 admission，启动本地或远端 replica，等待 ready，再组合成 StagePool。

LLM replica 的启动会准备 EngineCore client；diffusion replica 走 diffusion stage launcher，可以使用 inline client 或 process-backed client。StagePool 是面向 orchestrator 的统一 stage 门面，但它保留 AR 与 diffusion client 的不同调用协议。

- `vllm_omni/engine/omni_engine_base.py#L440`、`vllm_omni/engine/omni_engine_base.py#L510`
- `vllm_omni/engine/stage_runtime.py#L331`、`vllm_omni/engine/stage_runtime.py#L1199`、`vllm_omni/engine/stage_runtime.py#L1420`、`vllm_omni/engine/stage_runtime.py#L1496`
- `docs/design/module/stage_runtime.md`

### 启动失败和清理

初始化不是“启动到一半就可服务”。StageRuntime 在失败时收集已创建 client、停止启动中的 replica 并回收 runtime 资源；OmniEngineBase 将启动错误交给 startup future/错误消息，并在退出时取消 loop 中残留任务、关闭 queue 与共享资源。排查启动卡住时要分别检查 stage admission、worker ready 握手、各 replica 初始化和 orchestrator startup future。
