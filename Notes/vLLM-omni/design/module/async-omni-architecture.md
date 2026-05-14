---
type: Note
related_to: "[[vLLM-omni]]"
status: Active
url: https://github.com/vllm-project/vllm-omni/blob/main/docs/design/module/async_omni_architecture.md
---

# AsyncOmni Architecture

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                    API Layer                                    │
│  ┌─────────────────────────────────────┐  ┌──────────────────────────────────┐  │
│  │ AsyncOmni (EngineClient)            │  │ Omni                             │  │
│  │ • generate() / abort() / shutdown() │  │ • generate()                     │  │
│  │ • _final_output_handler()           │  │                                  │  │
│  └─────────────────────────────────────┘  └──────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                              Engine Layer (Proxy)                               │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │ AsyncOmniEngine                                                           │  │
│  │ • _bootstrap_orchestrator() & _initialize_stages()                        │  │
│  │ • add_request() / add_request_async() -> input_processor.process_inputs() │  │
│  │ • try_get_output() / try_get_output_async()                               │  │
│  └───────────────────┬─────────────────────────────────▲─────────────────────┘  │
│         request_queue (janus.Queue)        output_queue (janus.Queue)           │
├──────────────────────┼─────────────────────────────────┼────────────────────────┤
│                      ▼        Orchestration Layer      │                        │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │ Orchestrator [background thread]                                          │  │
│  │ • _request_handler()                                                      │  │
│  │     -  stage_client.add_request_async() & _prewarm_async_chunk_stages()   │  │
│  │ • _orchestration_output_handler()                                         │  │
│  │     -  _process_stage_outputs() -> output_processors[i].process_outputs() │  │
│  │     -  _route_output() & _forward_to_next_stage()                         │  │
│  └──────────┬─────────────────────────┬────────────────────────┬─────────────┘  │
├─────────────┼─────────────────────────┼────────────────────────┼────────────────┤
│             │                 Communication Layer              │                │
│  ┌───────────────────────┐ ┌───────────────────────┐ ┌───────────────────────┐  │
│  │ StageEngineCoreClient │ │ StageEngineCoreClient │ │ StageDiffusionClient  │  │
│  │ • ZMQ ROUTER / PULL   │ │ • ZMQ ROUTER / PULL   │ │ • ZMQ ROUTER / PULL   │  │
│  │ • Msgpack codec       │ │ • Msgpack codec       │ │ • Msgpack codec       │  │
│  └──────────┬────────────┘ └──────────┬────────────┘ └──────────┬────────────┘  │
│             ▼ ZMQ IPC                 ▼ ZMQ IPC                 ▼ ZMQ IPC       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                 Execution Layer                                 │
│  ┌───────────────────────┐ ┌───────────────────────┐ ┌───────────────────────┐  │
│  │ StageCoreProc         │ │ StageCoreProc         │ │ DiffusionEngine       │  │
│  │ [background process]  │ │ [background process]  │ │ [background process]  │  │
│  └───────────────────────┘ └───────────────────────┘ └───────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 执行流程（单个generate请求）

```
[1] App
    -> AsyncOmni.generate(prompt, request_id)

[2] AsyncOmni
    -> _final_output_handler()   (首次请求时启动)
    -> AsyncOmniEngine.add_request(stage_id=0, ...)

[3] AsyncOmniEngine.add_request
    -> (如果stage-0是llm且输入不是EngineCoreRequest)
       InputProcessor.process_inputs()
       OutputProcessor[0].add_request()
    -> request_queue.put(add_request_msg)

[4] Orchestrator._request_handler
    -> _handle_add_request(msg)
    -> stage_clients[0].add_request_async(...)

[5] Orchestrator._orchestration_loop (循环)
    -> 轮询阶段输出
       - llm阶段：await get_output_async()
       - diffusion阶段：get_diffusion_output_nowait()
    -> (llm阶段) output_processors[i].process_outputs(...)
    -> _route_output(...)
    -> 如果完成且不是最终阶段且非async-chunk：
         _forward_to_next_stage(...)
         -> next_stage.add_request_async(...)
    -> output_queue.put(output)

[6] AsyncOmni._final_output_loop (后台协程)
    -> AsyncOmniEngine.try_get_output_async()
    -> 按request_id路由到ClientRequestState.queue

[7] AsyncOmni._process_orchestrator_results
    -> 从ClientRequestState.queue读取
    -> _process_single_result(...)
    -> yield OmniRequestOutput

[8] 退出条件
    -> 收到result["finished"] == True
    -> generate()结束
```

## 运行时序列（单个generate请求）

```
App → AsyncOmni
    → start output_handler once
    → AsyncOmniEngine.add_request(stage_id=0, ...)
    → InputProcessor.process_inputs()
    → request_queue.put(add_request)

Orchestrator
    → _handle_add_request
    → stage_clients[0].add_request_async

循环：poll route forward
    → get_output_async / get_diffusion_output_nowait
    → _route_output
    → 如果需要转发到下一阶段：
        next_stage.add_request_async
    → output_queue.put

AsyncOmni
    → try_get_output_async
    → 按request_id路由
    → yield OmniRequestOutput
```

## 拓扑对比

### 之前（参考）

```
┌────────────────────────────────────────────────────────────────────────────┐
│ Main Process                                                               │
│  ┌──────────────────────┐   ┌────────────────────────────────────────────┐ │
│  │ generate()           │   │ final_output_handler()                     │ │
│  └──────────────────────┘   └────────────────────────────────────────────┘ │
└──────────┬─────────────────────────┬─────────────────────────┬─────────────┘
  mp.Queue (in_q/out_q)    mp.Queue (in_q/out_q)    mp.Queue (in_q/out_q)
           ▼▲                        ▼▲                        ▼▲
┌───────────────────────┐  ┌───────────────────────┐  ┌──────────────────────┐
│ Worker Proc-0         │  │ Worker Proc-1         │  │ Worker Proc-2        │
│ (Thinker LLM)         │  │ (Talker LLM)          │  │ (Vocoder)            │
│  ┌────────────────┐   │  │  ┌────────────────┐   │  │  ┌────────────────┐  │
│  │_stage_worker   │   │  │  │_stage_worker   │   │  │  │_stage_worker   │  │
│  │_async()        │   │  │  │_async()        │   │  │  │_async()        │  │
│  └────────────────┘   │  │  └────────────────┘   │  │  └────────────────┘  │
│  ┌────────────────┐   │  │  ┌────────────────┐   │  │  ┌────────────────┐  │
│  │output_handler()│   │  │  │output_handler()│   │  │  │output_handler()│  │
│  └────────────────┘   │  │  └────────────────┘   │  │  └────────────────┘  │
└──────────┬────────────┘  └──────────┬────────────┘  └──────────┬───────────┘
       ZMQ ▼ ▲ ZMQ               ZMQ ▼ ▲ ZMQ               ZMQ ▼ ▲ ZMQ
┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
│ EngineCore Proc-0    │   │ EngineCore Proc-1    │   │ EngineCore Proc-2    │
│ (Thinker)            │   │ (Talker)             │   │ (Vocoder)            │
└──────────────────────┘   └──────────────────────┘   └──────────────────────┘
```

### 当前

```
┌────────────────────────────────────────────────────────────────────────────┐
│ Main Process                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ Main Thread                                                          │  │
│  │  ┌──────────────────────┐   ┌─────────────────────────────────────┐  │  │
│  │  │ generate()           │   │ final_output_handler()              │  │  │
│  │  └──────────────────────┘   └─────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│         janus.Queue (request_queue) ▼  ▲ janus.Queue (output_queue)        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ Orchestrator Thread                                                  │  │
│  │  ┌──────────────────────┐  ┌──────────────────────────────────────┐  │  │
│  │  │ _request_handler()   │  │ _orchestration_output_handler()      │  │  │
│  │  └──────────────────────┘  └──────────────────────────────────────┘  │  │
│  │  ┌────────────────────────────────────────────────────────────────┐  │  │
│  │  │ _orchestration_loop(): 轮询/处理/路由所有阶段的输出              │  │  │
│  │  └────────────────────────────────────────────────────────────────┘  │  │
│  └───────┬─────────────────────────┬─────────────────────────┬──────────┘  │
└──────────┬─────────────────────────┬─────────────────────────┬─────────────┘
       ZMQ ▼ ▲ ZMQ               ZMQ ▼ ▲ ZMQ               ZMQ ▼ ▲ ZMQ
  ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
  │ EngineCore Proc-0    │  │ EngineCore Proc-1    │  │ EngineCore Proc-2    │
  │ (Thinker)            │  │ (Talker)             │  │ (Vocoder)            │
  └──────────────────────┘  └──────────────────────┘  └──────────────────────┘
```

## 测试脚本

```bash
# Qwen2.5-Omni
cd examples/offline_inference/qwen2_5_omni
python end2end.py --output-dir output_audio --query-type use_mixed_modalities

# Qwen3-Omni
cd ../qwen3_omni
python end2end.py --output-dir output_audio --query-type text --async-chunk --enable-stats

# BAGEL
cd ../bagel
python end2end.py --prompts "A cute cat"

# Text-to-Image
cd ../text_to_image
python text_to_image.py --prompt "a cup of coffee on the table" --output output.png
```

## 关键组件

### API层
- **AsyncOmni**：异步引擎客户端，提供`generate()`、`abort()`、`shutdown()`等方法
- **Omni**：同步接口

### 引擎层（代理）
- **AsyncOmniEngine**：引导编排器、初始化阶段、处理请求和输出

### 编排层
- **Orchestrator**：后台线程，处理请求路由和输出编排

### 通信层
- **StageEngineCoreClient**：LLM阶段客户端
- **StageDiffusionClient**：Diffusion阶段客户端

### 执行层
- **StageCoreProc**：LLM阶段后台进程
- **DiffusionEngine**：Diffusion后台进程
