## Initialization

当前运行时有三种初始化入口。它们最终都汇聚到 worker 内的

`StageEngine`，但外层 owner 不同。

**1. CLI 启动单 stage 在线服务：`vllm_rlt.entrypoints.serve`**

```text
serve.main()
  -> 解析 CLI + runtime_configs()
  -> create_app(partial(load_engine, args))
  -> EngineWorker.start()
  -> load_engine()
       -> resolve_model_config() + load_tokenizer()
       -> StageClient(...)
       -> start_stage_worker()
       -> StageWorker -> StageEngine
       -> 等待 ready
  -> EngineWorker 持有 tokenizer/engine，单线程驱动 engine.step()
  -> HTTP RequestChannel 接收请求并输出 SSE/JSON
```

**2. CLI 启动 PD 在线服务：`vllm_rlt.entrypoints.pd_serve`**

```text
pd_serve.main()
  -> create_app(partial(load_engine, args))
  -> load_engine()
       -> resolve_model_config() + load_tokenizer()
       -> PDEngine(...)
            -> prepare_model() + PDConfig
            -> 为每个 prefill/decode device 创建 StageClient
            -> 每个 StageClient 启动 StageWorker -> StageEngine
            -> connector_factory -> PDStageConnector -> NixlConnector
            -> 等待所有 ready
            -> 校验 model/KV/attention fingerprint
            -> 发送 connect，建立 P/D peer
  -> EngineWorker 持有 PDEngine/tokenizer，驱动 PD coordinator
```

**3. 离线推理：`LLM` / `vllm_rlt.entrypoints.cli`**

```text
LLM(model, ...)
  -> prepare_model(model, revision)
       -> resolved source / revision / read-only ModelConfig
  -> StageClient(model, model_config, engine_options...)
  -> start_stage_worker(...)
	  -> StageWorker 加载或构造 model，并创建 StageEngine
	  -> 等待 ready
	  -> generate()
	       -> tokenizer.encode()（需要时延迟加载）
	       -> StageClient.add_request()
	       -> LLM.generate() 调用 StageClient.step() 轮询 worker 输出
```

| 边界          | 初始化 owner                            | 主要资源                                                    |
| ----------- | ------------------------------------ | ------------------------------------------------------- |
| HTTP 在线层    | `EngineWorker` + event loop          | `RequestChannel`、tokenizer、流式输出、请求并发限制                  |
| PD 部署层      | `PDEngine`                           | P/D `Peer`、`Pending`、transfer 状态、多个 `StageClient`       |
| stage 控制层   | `StageClient`                        | worker process、`Pipe`、外部 ID/generation、control reply    |
| stage 资源层   | `StageWorker` / `StageEngine`        | model、KV cache、Scheduler、Runner、streams、connector       |
| 跨 stage 传输层 | `PDStageConnector` / `NixlConnector` | registered KV/hidden memory、transfer lease、NIXL handles |

字符串模型路径的权重在 worker 中加载；显式 `Module` 会先复制到 CPU 后交给 worker。`StageEngine` 始终在资源所属 worker 内创建，主进程或 HTTP 线程不持有第二个 model runner。

