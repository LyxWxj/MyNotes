---
title: Travel Through vLLM：从入口到 GPU 执行的运行时阅读笔记
tags:
  - vLLM
  - Travel-Through-X
  - LLM-serving
repository: vllm-project/vllm
commit: b26e0840134ea4b4c87c34bd5a0e0aa61397c638
commit_time: 2026-11-23 17:05:49 +08:00
verified_at: 2026-11-27 12:22:52 +08:00
scope: v1 runtime；静态源码阅读；未执行推理、GPU 测试或 benchmark
---

# Travel Through vLLM：从入口到 GPU 执行

> 这份笔记以仓库 commit `b26e0840134ea4b4c87c34bd5a0e0aa61397c638` 为基线，记录 vLLM V1 从请求进入前端，到调度、KV cache、worker 执行、异步输出和资源回收的运行时路径。
>
> 约定：带“源码事实”的句子可在后面的路径和行号中复核；带“推断”的句子是根据多个模块的控制流归纳出来的运行模型。阅读范围是当前 checkout 的 V1 引擎、调度、worker、executor、KV connector 与相关测试/文档。当前工作树另有未跟踪的 `csrc/libtorch_stable/`、`examples/generate/qwen3_06B.py`、`models/`，本笔记没有阅读或修改它们。

## 0. 先看结论：一条请求如何穿过系统

```text
LLM / AsyncLLM / OpenAI server
        │  Renderer + InputProcessor
        ▼
EngineCoreRequest ── OutputProcessor 建立前端状态
        │  EngineCoreClient.add_request
        ├── InprocClient：同进程调用 EngineCore
        └── SyncMPClient / AsyncMPClient：ZMQ 送入 EngineCore 进程
                                      │
                                      ▼
                         EngineCore.add_request
                                      │
                                      ▼
                         Scheduler.add_request
                 WAITING → RUNNING / WAITING_FOR_REMOTE_KVS
                                      │ schedule()
                                      ▼
 SchedulerOutput（token budget、block table diff、spec tokens、connector metadata）
                                      │
                                      ▼
 Executor.execute_model(non_block=True)
                                      │
                                      ▼
 Worker / GPUModelRunner
   更新 worker request state → 清零新 block → CoW → forward → sample/speculate
                                      │
                          Future / AsyncOutput.copy_event
                                      ▼
 Scheduler.update_from_output
   推进逻辑 token frontier、接受/回滚 speculative、完成/重排/释放请求
                                      │
                                      ▼
 EngineCoreOutput
                                      │
                                      ▼
 OutputProcessor.process_outputs
   增量 detokenize → stop 字符串判断 → 同步返回或写入 per-request async queue
```

最容易混淆的三个边界是：

1. **提交与完成不是一回事。** `execute_model(non_block=True)` 只提交执行；GPU forward、D2H copy、pipeline send 和 KV connector 的写入可能仍在进行。`Future.result()` 或 `AsyncOutput.get_output()` 才把“可以读到输出”的时刻显式化。
2. **逻辑进度与物理 KV 写入不是一回事。** scheduler 用 `num_computed_tokens`、`num_in_flight_tokens` 和 output placeholders 表示请求的逻辑进度；block pool 还要等待对应的 GPU/connector 操作安全结束，才能把 block 放回 free queue。
3. **前端状态与 engine-core 状态分离。** tokenizer、detokenizer、stream queue、外部 request ID 在 frontend；请求状态、调度队列和 KV block ownership 在 engine-core；worker 只保留执行下一步所需的 GPU/CPU batch state。

## 1. 仓库地图与阅读入口

项目根目录中，和一次生成请求最相关的路径如下：

```text
vllm/
├── entrypoints/
│   ├── llm.py                    # 离线 LLM.generate/enqueue/wait_for_completion
│   └── openai/                   # OpenAI-compatible HTTP/Rust 前端
├── config/                       # EngineArgs 与各类并行、cache、spec 配置
├── v1/
│   ├── engine/
│   │   ├── llm_engine.py         # 同步/兼容入口，组装 client 与 processors
│   │   ├── async_llm.py          # AsyncLLM、per-request stream、output handler
│   │   ├── input_processor.py    # EngineInput/原始 prompt → EngineCoreRequest
│   │   ├── output_processor.py   # EngineCoreOutput → RequestOutput
│   │   ├── core_client.py        # in-process 或多进程 ZMQ client
│   │   ├── core.py               # EngineCore busy loop、step、输入/输出 socket
│   │   └── detokenizer.py        # 增量解码与 stop string
│   ├── core/
│   │   ├── sched/scheduler.py    # admission、budget、preemption、更新请求
│   │   ├── sched/output.py       # SchedulerOutput 的 worker-side contract
│   │   ├── sched/               # RequestQueue、调度策略、结构化输出
│   │   ├── kv_cache_manager.py   # 多 KV group 的协调器
│   │   ├── single_type_kv_cache_manager.py
│   │   └── block_pool.py         # GPU block 分配、缓存、eviction、free
│   ├── executor/
│   │   ├── abstract.py           # backend 选择与 collective_rpc
│   │   └── uniproc_executor.py   # 单进程 worker 与 AsyncOutputFuture
│   ├── worker/
│   │   ├── gpu_worker.py         # memory profile、PP send/recv、worker 生命周期
│   │   └── gpu/
│   │       ├── model_runner.py   # batch、attention、forward、sampling/spec
│   │       └── async_utils.py    # GPU copy stream、copy_event、CPU output
│   ├── request.py                 # 请求逻辑状态、token frontier、状态枚举
│   ├── outputs.py                 # ModelRunnerOutput、KV connector output
│   └── kv_offload/                # KV offload/connector 相关实现
├── distributed/                   # TP/PP/DP、KV transfer 等通信组件
├── tests/v1/                      # V1 scheduler/worker/spec/streaming/kv 测试
└── docs/features/                 # speculative、disaggregated prefill、offload
```

建议按下面几个入口阅读，而不要从某个 attention kernel 开始：

- **同步路径：** `vllm/entrypoints/llm.py` → `vllm/v1/engine/llm_engine.py` → `vllm/v1/engine/core_client.py` → `vllm/v1/engine/core.py`。
- **异步服务路径：** `vllm/v1/engine/async_llm.py` 的 `generate()` 和 `_run_output_handler()`。
- **调度与缓存：** `vllm/v1/core/sched/scheduler.py`、`vllm/v1/request.py`、`vllm/v1/core/kv_cache_manager.py`、`vllm/v1/core/block_pool.py`。
- **设备执行：** `vllm/v1/executor/abstract.py`、`uniproc_executor.py`、`vllm/v1/worker/gpu_worker.py`、`vllm/v1/worker/gpu/model_runner.py`、`async_utils.py`。

## 阅读导航

主体按概念拆分到相邻页面，下面顺序也对应一次请求的运行时生命周期：

1. [初始化与请求准入](00-vLLM-运行时阅读笔记/01-初始化-请求准入.md)
2. [EngineCore 与调度](00-vLLM-运行时阅读笔记/02-enginecore-调度.md)
3. [KV、Worker 与输出](00-vLLM-运行时阅读笔记/03-kv-worker-输出.md)
4. [Speculative 与分布式路径](00-vLLM-运行时阅读笔记/04-speculative-分布式.md)
5. [生命周期与验证](00-vLLM-运行时阅读笔记/05-生命周期-验证.md)
