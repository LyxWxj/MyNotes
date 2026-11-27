---
title: Travel Through vLLM-Omni
aliases:
  - vLLM-Omni 阅读笔记
  - Travel Through vLLM-Omni
tags:
  - vLLM-Omni
  - runtime
  - multimodal
repository: vllm-omni-batching
revision: e5576969e12c878a9e4c0e9f5ea0a5b4884ce37c
commit_time: 2026-11-26T17:41:07Z
verified_at: 2026-11-27T03:40:12+08:00
scope: 当前 checkout 的引擎主链、AR 与 diffusion 执行路径、缓存/传输边界及对应设计和测试入口；静态阅读
---

# Travel Through vLLM-Omni

> 阅读路线：先看外部请求如何进入，再沿着初始化、stage 执行、跨 stage 传输和终态清理往下追。重点是“谁拥有什么状态、谁负责推进、何时资源真的可复用”，而不是模型算子细节。

## 0. 范围与读法

- 本笔记以提交 **e5576969e12c878a9e4c0e9f5ea0a5b4884ce37c** 为源码基线，提交时间为 **2026-11-26 17:41:07 UTC**。源码、设计文档索引及其列出的测试入口在该 revision 上交叉核对。
- 当前工作树另有未跟踪的 **AGENT.md、artifacts/、docs/design/pipeline_parallelism/、feat/、refactor/**。它们在阅读期间保持原样；本文的运行时主链以提交中的实现为准，不把未跟踪内容提升为稳定接口。
- 本文是静态阅读笔记。下文列出的测试是验证入口，不代表本次执行过这些测试；没有运行 GPU、模型加载或端到端推理。
- 设计页中带有 draft 状态的 invariant 是候选契约。用它们导航代码时，仍以同一 revision 的实现和测试为证据。

## 1. 先建立整体图景

vLLM-Omni 把异构多模态推理组织成一条由配置描述的 stage pipeline。外部 API 不直接管理模型 worker；engine 建立 stage runtime 与 replica，orchestrator 持有请求状态并负责跨 stage 路由。每个 stage 内部再由 AR 的 vLLM EngineCore/Scheduler 或 diffusion runtime 推进请求。

    OpenAI / AsyncOmni / 离线入口
        → AsyncOmniEngine：线程边界、请求队列、输出队列、RPC
        → Orchestrator：请求状态、stage 路由、终态、取消
        → StagePool：replica 选择、绑定、stage client
        → LLM stage：EngineCore → OmniARScheduler → Worker V2 / ModelRunner
          或
          Diffusion stage：DiffusionEngine → Request/StepScheduler → Executor → Worker / Runner
        → OutputMessage / NextStageInputMessage
        → Orchestrator 路由、清理
        → AsyncOmni 输出 pump → 请求级队列 → 调用方

> **源码事实**：OmniEngineBase 在专用线程中启动 orchestrator 的 asyncio loop；stage client、stage input/output 和 transfer 操作在该 loop 中协作。线程之间通过 queue 交接。stage 本身可在本地进程、独立进程或分布式 worker 中执行。  
> **源码位置**：[omni_engine_base.py:132](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/engine/omni_engine_base.py#L132>)、[omni_engine_base.py:510](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/engine/omni_engine_base.py#L510>)、[engine_orchestration.md](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/docs/design/module/engine_orchestration.md>)。

一个有效的心智模型是：**配置决定有哪些 stage 和如何启动；orchestrator 决定请求下一步去哪；stage scheduler 决定本 stage 何时执行；worker/model runner 才触碰设备和模型状态。** OmniConnector 负责数据传输与同步契约，不替 orchestrator 决定路由。

## 2. 目录树与阅读入口

| 目录 | 主要职责 | 建议入口 |
| --- | --- | --- |
| vllm_omni/config | model_type 到 pipeline、stage 拓扑和 deploy 配置解析 | [pipeline_registry.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/config/pipeline_registry.py>)、[stage_config.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/config/stage_config.py>)、[omni_config.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/config/omni_config.py>) |
| vllm_omni/entrypoints | AsyncOmni、OpenAI、CLI、duplex 等用户入口 | [async_omni.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/entrypoints/async_omni.py>)、[async_omni_base.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/entrypoints/async_omni_base.py>) |
| vllm_omni/engine | engine 生命周期、stage 初始化、orchestrator、StagePool、消息协议 | [omni_engine_base.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/engine/omni_engine_base.py>)、[stage_runtime.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/engine/stage_runtime.py>)、[stage_pool.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/engine/stage_pool.py>)、[orchestrator.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/engine/orchestrator.py>)、[messages.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/engine/messages.py>) |
| vllm_omni/core/sched | AR scheduler、Omni scheduler mixin、跨 stage 输入协调 | [omni_ar_scheduler.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/core/sched/omni_ar_scheduler.py>)、[omni_scheduler_mixin.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/core/sched/omni_scheduler_mixin.py>) |
| vllm_omni/core/prefix_cache | AR prefix cache 的管理、controller、block pool 和 runner adapter | [prefix_cache/](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/core/prefix_cache/>) |
| vllm_omni/worker_v2 | AR / generation worker、data plane、output snapshot、prefix-cache 集成 | [omni_data_plane.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/worker_v2/omni_data_plane.py>)、[omni_ar_model_runner.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/worker_v2/omni_ar_model_runner.py>) |
| vllm_omni/diffusion | DiffusionEngine、request/step scheduler、executor、worker、model runner 和缓存 | [diffusion_engine.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/diffusion/diffusion_engine.py>)、[sched/](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/diffusion/sched/>)、[executor/](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/diffusion/executor/>) |
| vllm_omni/distributed | stage 分布式控制、OmniConnector、KV transfer | [omni_connectors/](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/distributed/omni_connectors/>)、[stage_runtime.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/engine/stage_runtime.py>) |
| docs/design | 当前架构与功能契约索引 | [design/index.md](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/docs/design/index.md>) |
| tests | engine、scheduler、worker、connector、diffusion 的回归入口 | [tests/engine/](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/tests/engine/>)、[tests/core/sched/](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/tests/core/sched/>)、[tests/diffusion/](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/tests/diffusion/>) |

## 3. 初始化：从模型配置到可接请求的 stage pool

### 3.1 Pipeline 配置解析

启动先根据模型配置/model_type 找 pipeline。**OMNI_PIPELINES** 注册的条目可以是静态 **PipelineConfig**，也可以是 resolver；解析后每个 **StagePipelineConfig** 描述逻辑 stage 的类型、输入来源、最终输出行为、执行方式等。deploy 配置再补充 replica 数、设备布局、并行方式和运行参数。

入口主要在 **OmniEngineBase.__init__ → _resolve_stage_configs**；stage 配置结构在 **PipelineConfig / StagePipelineConfig**。因此某个模型看到的执行拓扑，不能只从模型类推断，必须同时看注册表和 deploy 配置。

- [pipeline_registry.py:174](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/config/pipeline_registry.py#L174>)
- [stage_config.py:395](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/config/stage_config.py#L395>)、[stage_config.py:491](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/config/stage_config.py#L491>)
- [omni_engine_base.py:303](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/engine/omni_engine_base.py#L303>)、[omni_engine_base.py:725](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/engine/omni_engine_base.py#L725>)

### 3.2 Engine loop 与 stage runtime 启动

OmniEngineBase 创建后台线程和 asyncio loop，在 bootstrap 中依次初始化 runtime、建立队列/控制通道、构造 Orchestrator，然后运行其主协程。StageRuntime 按逻辑 stage 生成初始化计划，对资源做 admission，启动本地或远端 replica，等待 ready，再组合成 StagePool。

LLM replica 的启动会准备 EngineCore client；diffusion replica 走 diffusion stage launcher，可以使用 inline client 或 process-backed client。StagePool 是面向 orchestrator 的统一 stage 门面，但它保留 AR 与 diffusion client 的不同调用协议。

- [omni_engine_base.py:440](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/engine/omni_engine_base.py#L440>)、[omni_engine_base.py:510](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/engine/omni_engine_base.py#L510>)
- [stage_runtime.py:331](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/engine/stage_runtime.py#L331>)、[stage_runtime.py:1199](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/engine/stage_runtime.py#L1199>)、[stage_runtime.py:1420](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/engine/stage_runtime.py#L1420>)、[stage_runtime.py:1496](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/engine/stage_runtime.py#L1496>)
- [stage_runtime.md](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/docs/design/module/stage_runtime.md>)

### 3.3 启动失败和清理

初始化不是“启动到一半就可服务”。StageRuntime 在失败时收集已创建 client、停止启动中的 replica 并回收 runtime 资源；OmniEngineBase 将启动错误交给 startup future/错误消息，并在退出时取消 loop 中残留任务、关闭 queue 与共享资源。排查启动卡住时要分别检查 stage admission、worker ready 握手、各 replica 初始化和 orchestrator startup future。

## 4. 单请求主链：谁收、谁派、谁返回

### 4.1 入口与消息

以 AsyncOmni 为例，**generate()** 建立内部 request id 和前端请求状态，处理 admission、sampling 参数和输入；输出 handler 在后台持续从 engine 取消息，再路由到请求自己的 asyncio queue。引擎用消息对象跨越入口线程与 orchestrator loop：初始请求通常是 **StageSubmissionMessage**，stage 输出是 **OutputMessage**，控制请求包括 abort / interaction / streaming update。

跨 stage 时，**NextStageInputMessage** 带来源 stage、接收 stage、请求列表、stage 输出和下游提交参数。其校验检查 stage 顺序、请求类型与参数形态，降低把 diffusion 请求误交给 LLM client 等错误。

- [async_omni.py:150](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/entrypoints/async_omni.py#L150>)
- [async_omni_base.py:328](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/entrypoints/async_omni_base.py#L328>)、[async_omni_base.py:408](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/entrypoints/async_omni_base.py#L408>)
- [messages.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/engine/messages.py>)、[async_omni_engine.py:539](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/engine/async_omni_engine.py#L539>)

### 4.2 Orchestrator 与 StagePool

Orchestrator 收到初始消息后创建 orchestration request state 并提交 stage 0。每个 stage 的原始输出被处理成 Omni 输出；未终止的输出可以继续发给调用方或留在本 stage，完成一个中间 stage 的输出则进入跨 stage route。下一 stage 的提交由 Orchestrator 发起，replica 由对应 StagePool 选择。

StagePool 的选择可结合显式 replica pin、session affinity、分布式服务可用性和负载策略；请求进入后保留 request 到 replica 的绑定，后续更新和 abort 才能落到实际持有状态的副本。不要把“逻辑 stage id”与“物理 replica id”混为一谈。

- [orchestrator.py:4295](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/engine/orchestrator.py#L4295>)、[orchestrator.py:2501](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/engine/orchestrator.py#L2501>)
- [orchestrator.py:3207](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/engine/orchestrator.py#L3207>)、[orchestrator.py:3639](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/engine/orchestrator.py#L3639>)
- [stage_pool.py:538](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/engine/stage_pool.py#L538>)、[stage_pool.py:1603](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/engine/stage_pool.py#L1603>)

### 4.3 输出回到调用方

StagePool client 的输出由 orchestrator loop 消费。LLM 输出先经过 stage output processor；diffusion 输出有其 stream/result pump。Orchestrator 根据 request state 和输出是否终态执行路由、附加指标、产出 OutputMessage 或转发到下游。入口侧只有一个 output pump 从 engine 拉取输出，再按 request id 分派到请求级 queue，最终恢复外部 id 并 yield。

默认 orchestrator 是 event-driven：LLM replica 使用 reader task 等待输出，diffusion 有对应 poller，再由串行 dispatch queue 保留统一路由语义。环境变量 **VLLM_OMNI_EVENT_DRIVEN_ORCH=0** 选择 legacy poll loop。两种 loop 复用处理和路由逻辑。

- [orchestrator.py:1135](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/engine/orchestrator.py#L1135>)、[orchestrator.py:1471](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/engine/orchestrator.py#L1471>)
- [orchestrator.py:1761](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/engine/orchestrator.py#L1761>)
- [async_omni_base.py:408](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/entrypoints/async_omni_base.py#L408>)、[async_omni_base.py:328](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/entrypoints/async_omni_base.py#L328>)

## 5. AR 路径：EngineCore 调度、KV 生命周期与生成输出

AR stage 的调度核心是 **OmniARScheduler**，在 vLLM Scheduler 上叠加 Omni 输入协调、流式输入、connector 数据到达、部分 request admission 与 cleanup 语义。每次 schedule 前会清扫已 abort 的排队请求、消费 ready 的 Omni inputs，再委托上游 scheduler 选取本轮 token 工作；之后包装 Omni scheduler output，并附带跨 stage transfer 等元数据。

Prefill / decode、KV block 分配和模型执行仍分属 vLLM scheduler、KV cache manager 与 worker/model runner。Omni 的 scheduler mixin 处理多模态输入、异步 chunk/full-payload 的等待或就绪、超时与终态同步。Prefix cache 相关的 manager/controller/block pool/runner adapter 在 **core/prefix_cache**；它解决可复用 KV 前缀状态，不等于 stage 间 payload connector。

- [omni_ar_scheduler.py:203](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/core/sched/omni_ar_scheduler.py#L203>)、[omni_ar_scheduler.py:1297](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/core/sched/omni_ar_scheduler.py#L1297>)
- [omni_scheduler_mixin.py:739](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/core/sched/omni_scheduler_mixin.py#L739>)
- [core/prefix_cache/](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/core/prefix_cache/>)
- [ar_runtime.md](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/docs/design/module/ar_runtime.md>)、[cache_management.md](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/docs/design/module/cache_management.md>)

### 5.1 逻辑完成不总等于设备资源可复用

当 request 结束或被取消，scheduler 可以立即把它从逻辑队列移除；但对应 KV block 若仍被先前 GPU work 读取，就不能马上交给另一个 request 覆写。当前实现存在 deferred-free fence：完成一个非空 scheduled step 后，update 路径推进已处理 step 序号并 drain deferred frees；空 step 不推进 fence。阅读缓存代码时要把“request terminal”“scheduler bookkeeping 清掉”“设备上的写/读完成”“block 可再次分配”当成四个不同时间点。

**OmniARScheduler.update_from_output** 中记录了对 GPU write 完成的假设，并在启用延迟释放时处理 fence。它是 scheduler/设备生命周期的交界处，涉及缓存复用时值得优先追读。

- [omni_ar_scheduler.py:1363](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/core/sched/omni_ar_scheduler.py#L1363>)、[omni_ar_scheduler.py:2445](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/core/sched/omni_ar_scheduler.py#L2445>)
- [test_omni_sched_deferred_free_fence.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/tests/core/sched/test_omni_sched_deferred_free_fence.py>)
- [test_prefix_cache.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/tests/core/test_prefix_cache.py>)

### 5.2 Speculative decode 的边界

Omni 对 AR scheduler 做 draft/k-step 限制、排队和输出适配；真正的模型执行、验证和 token 接受语义还横跨 vLLM worker/model runner 及上游 scheduler。不要仅凭 Omni 的 draft 状态字段，推断整个 speculative acceptance/rollback 都在 OmniARScheduler.update_from_output 中完成。排查时沿着 scheduler output → model runner → sampled/accepted ids → request 状态更新这条线走，并查看对应测试。

- [test_omni_ar_scheduler_spec_decode.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/tests/core/sched/test_omni_ar_scheduler_spec_decode.py>)
- [test_omni_ar_scheduler_kstep_guards.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/tests/core/sched/test_omni_ar_scheduler_kstep_guards.py>)
- [worker_v2/omni_ar_model_runner.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/worker_v2/omni_ar_model_runner.py>)

### 5.3 流式输出与输入

AR 可以增量返回 token，也能接收 streaming input/update。流式请求的 request state、输入到达计数、队列清理和最终输出必须一起理解；取消可能在 request 仍排队、正在等多模态输入、正在运行或已经有部分输出时到来。对应入口包括 **AsyncOmni** 的 input update / abort 和 **OmniSchedulerMixin** 的 wait/input cleanup。

- [test_omni_ar_scheduler_streaming.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/tests/core/sched/test_omni_ar_scheduler_streaming.py>)
- [test_omni_ar_scheduler_ready_inbox.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/tests/core/sched/test_omni_scheduler_ready_inbox.py>)
- [test_omni_ar_scheduler_free_request_cleanup.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/tests/core/sched/test_omni_ar_scheduler_free_request_cleanup.py>)

## 6. Diffusion 路径：RequestScheduler 与 StepScheduler

DiffusionEngine 是 stage 内的请求入口和执行循环 owner。它建立 scheduler、executor、后台 busy loop、输出流和 worker 控制面；scheduler 决定容量、可兼容请求、进度和终态；executor 把批次提交到 worker；worker/model runner 持有设备、pipeline 和模型状态。

执行 backend 可为 **uni**（engine 进程内 worker）、**mp**（每个设备一个 worker process）或 **ray**（跨节点 GPU actor）。这些选择改变 IPC、资源边界和故障表现，但上层 request lifecycle 由同一类 engine/scheduler 契约驱动。

- [diffusion_engine.py:275](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/diffusion/diffusion_engine.py#L275>)、[diffusion_engine.py:431](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/diffusion/diffusion_engine.py#L431>)
- [diffusion_engine.py:537](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/diffusion/diffusion_engine.py#L537>)、[diffusion_engine.py:709](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/diffusion/diffusion_engine.py#L709>)
- [diffusion_runtime.md](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/docs/design/module/diffusion/diffusion_runtime.md>)

### 6.1 两种 scheduler

- **RequestScheduler** 更偏向整次 request / request wave；等待队列按配置容量和兼容 key admission，只有兼容的请求可共享执行 wave。
- **StepScheduler** 维护每个请求的 current step、total steps、交互代数与等待期限。每个 tick 由 engine 将当前 batch 发往 executor，随后 scheduler 根据结果推进、完成、报错，或进入 wait-for-interaction 状态。
- 稳定 request id 是输出归属依据，不能用某一 tick 的 batch 下标作为身份。兼容性显式比较；这有意限制了能够一起运行的 request 组合。

执行路径可读为：add request → scheduler schedule → executor execute → worker/model runner → scheduler update → 输出 / 下一 tick。Request 级失败应尽量局限于单 request；worker group 死亡则会让 executor/stage 进入故障处理路径。

- [request_scheduler.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/diffusion/sched/request_scheduler.py>)、[step_scheduler.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/diffusion/sched/step_scheduler.py>)
- [continuous_batching.md](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/docs/design/module/diffusion/continuous_batching.md>)
- [test_step_request_lifecycle.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/tests/diffusion/test_step_request_lifecycle.py>)、[test_step_interaction_wait.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/tests/diffusion/test_step_interaction_wait.py>)、[test_diffusion_batching.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/tests/diffusion/batching/test_diffusion_batching.py>)

### 6.2 设备输出与同步

输出 tensor 交给异步消费者前，producer stream 上的写必须对 consumer 可见。Diffusion uni executor 的输出流对已 materialize 的 device output 记录/等待 CUDA event，并在 consumer stream 上登记 tensor storage；没有 device event 的 CPU 输出无需该等待。没有 event 能力的平台在返回前完成 producer。没有输出的 denoise step 不应被误当成每步都要全局同步。

这说明**提交 execute 调用**只表示工作已排入执行路径，不表示输出 tensor 已可安全读取。实际数据可见性由 event/stream 或平台同步保证。

- [diffusion_runtime.md#uniproc-backend](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/docs/design/module/diffusion/diffusion_runtime.md#uniproc-backend>)
- [test_result_pump.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/tests/diffusion/test_result_pump.py>)
- [test_uniproc_executor.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/tests/diffusion/test_uniproc_executor.py>)

## 7. Stage 间传输：Connector、async_chunk 与 full payload

### 7.1 OmniConnector 只管传输

Connector 抽象跨 stage/process/device/node 的 payload transport 与同步，不拥有 next-stage 路由，也不应承载某个模型的执行策略。producer 和 consumer 必须约定 payload identity、shape、dtype、placement、ownership 和完成语义；正常完成、取消、错误和 shutdown 都要有确定的资源清理路径。

### 7.2 Async chunk

async_chunk 允许上游产生一个可消费 chunk 后就交给下游，不必等待整个 stage 完成。Orchestrator 可以先向 downstream chunk stage 提交 placeholder/prewarm；随后 scheduler/adapter 管理 chunk readiness，真正数据通过 connector 到达。Generation scheduler 必须以完整 ready chunk 为原子单位 admission，不能把一个 chunk 切成任意子片段；超大 chunk 应形成单一终态错误，避免部分提交后状态失配。

有些 native MRv2 接收路径由 **OmniSchedulingCoordinator** 追踪 runner connector 事件；legacy adapter 路径由 **OmniChunkTransferAdapter** 管理加载和保存循环。二者暴露给调度器的 readiness 语义应一致，但 transport owner 不同。

- [async_chunk.md](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/docs/design/feature/async_chunk.md>)
- [chunk_transfer_adapter.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.py>)
- [omni_ar_scheduler.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/core/sched/omni_ar_scheduler.py>)
- [test_generation_scheduler_atomic_chunk.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/tests/core/sched/test_generation_scheduler_atomic_chunk.py>)、[test_orchestrator_async_chunk_prewarm.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/tests/engine/test_orchestrator_async_chunk_prewarm.py>)

### 7.3 Full payload 累积与丢弃

MRv2 full-payload 路径中，worker data plane 按内部 request id 持有尚未发布的 payload；scheduler 给出 terminal status 后再 flush 或丢弃。正常 stop/length-limit 完成应只 flush 一次；abort/error/ignored 通过 discarded request id 清掉待发 payload。已经提交给 connector 的写不再由 pending accumulator 控制。

如果 AR 请求被 preempt 后继续执行，累计输出和 token boundary 属于同一个逻辑 request：重算的 prefix 不能重复追加；对齐 token 边界的 tensor 可只追加新 suffix；无法按 token 切分的部分重叠应显式失败。retry 则必须建立新的 payload/position state。最近的 full-payload 传输失败修复也提醒：stage transfer 失败不能让请求留在等待态或静默成功。

- [omni_data_plane.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/worker_v2/omni_data_plane.py>)、[omni_scheduler_mixin.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/core/sched/omni_scheduler_mixin.py>)
- [omni_connector.md](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/docs/design/module/omni_connector.md>)、[test_full_payload_lifecycle.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/tests/worker/test_full_payload_lifecycle.py>)
- 当前基线提交主题：full-payload stage transfer 不能成功时令请求失败（commit e5576969）。

### 7.4 排队发送不代表下游已消费

Output materialization / native output worker 将 payload 引用入队，enqueue 成功不等于 downstream 已收到或处理完成。source 的 model output 字段之后被清空，也不意味着队列中的对象或底层 storage 被释放。要定位 tensor 被过早复用，分别追 producer snapshot、队列持有时间、consumer event/wait、request terminal cleanup，不能只看 enqueue 返回。

- [omni_async_output_materialization.md](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/docs/design/feature/omni_async_output_materialization.md>)
- [omni_data_plane.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/worker_v2/omni_data_plane.py>)
- [test_omni_ar_model_runner.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/tests/worker_v2/test_omni_ar_model_runner.py>)

## 8. Replica、分布式与 session affinity

逻辑 stage 可以配置多个 replicas。StagePool 在提交初始请求时选择副本并保留绑定；同一 request 的更新、abort 和跨 stage session 路径要回到对应副本。分布式 stage runtime 通过 coordinator/master 与 membership 控制器管理远端 replica 可见性；选择器只应把 live/attached/serviceable 的 replica 作为候选。

某些 pipeline 还有显式 replica pin 或 session holder。pin 不只是调度提示，它影响 session state 和 KV locality 的归属；释放绑定必须跟 request terminal/abort 生命周期一致。PD disaggregation 则由 pipeline/deploy 配置与 stage role 协同表达，prefill 和 decode stage 间传递 KV / hidden-state 等状态。读 PD 时要分别看 orchestrator routing、connector transport 与接收侧 cache admission。

- [stage_runtime.py:1543](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/engine/stage_runtime.py#L1543>)
- [stage_pool.py:608](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/engine/stage_pool.py#L608>)、[stage_pool.py:770](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/engine/stage_pool.py#L770>)
- [disaggregated_inference.md](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/docs/design/feature/disaggregated_inference.md>)
- [test_stage_pool_session_affinity.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/tests/engine/test_stage_pool_session_affinity.py>)、[test_membership_controller.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/tests/engine/test_membership_controller.py>)

## 9. 取消、错误、正常结束与 shutdown

### 9.1 Abort 是一条跨 owner 清理链

调用方取消后，入口向 engine 发 abort。fire-and-forget abort 只保证投递控制消息；带 RPC id 的 **abort_async** 会等待 correlated result。Orchestrator 将 abort 传给各 StagePool/client，收集最终 AR stage 可返回的部分输出，再释放 request-to-replica binding、删除 orchestration state 和相关 companion state。Diffusion abort 走 stage client 到 DiffusionEngine，再由 scheduler/executor/worker 清理对应请求。

因此要区分三种完成：abort 控制消息成功入队、stage owner 已执行取消、orchestrator 收到 ack 并清理完共享状态。

### 9.2 错误传播

Stage 内 request error 通过 ErrorMessage 或 stage output error 反馈；Orchestrator 区分 request error 与 replica/process 故障。前者通常终止当前 request，后者可能需要标记 dead replica 并让请求失败。跨 stage input processor/connector 失败应有终态错误；不能留下已登记但无人再推进的 request。

- [omni_engine_base.py:842](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/engine/omni_engine_base.py#L842>)、[omni_engine_base.py:859](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/engine/omni_engine_base.py#L859>)
- [orchestrator.py:907](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/engine/orchestrator.py#L907>)、[orchestrator.py:2188](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/engine/orchestrator.py#L2188>)
- [orchestrator.py:1822](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/engine/orchestrator.py#L1822>)、[orchestrator.py:3207](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/engine/orchestrator.py#L3207>)

### 9.3 Shutdown

Shutdown 的顺序应沿 ownership 反向理解：停止接新请求与 orchestrator loop，通知 stage pool/client 结束，停止 replica/worker，关闭共享队列、transfer/runtime 资源。若只在入口 task 结束时看见请求已取消，不能推断远端 worker 和 connector buffer 已释放。

- [omni_engine_base.py:977](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/engine/omni_engine_base.py#L977>)
- [stage_runtime.py:705](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/vllm_omni/engine/stage_runtime.py#L705>)
- [stage_runtime.md](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/docs/design/module/stage_runtime.md>)

## 10. 阅读时最值得守住的几个不变量

| 观察点 | 需要区分的状态 |
| --- | --- |
| 请求路由 | logical stage 与 physical replica；入口 id 与 orchestrator id |
| 异步提交 | submit/enqueue 已完成，与 worker 执行/下游消费完成 |
| 设备输出 | producer stream 已写，与 consumer 已 wait event、storage 可安全复用 |
| request 终态 | caller 已收到终态，与各 stage 清理、绑定释放、设备缓存可复用 |
| 缓存 | logical block 已 free，与先前 GPU work 对 block 的访问已越过 fence |
| connector | connector 做 payload transport/sync，与 orchestrator 做 stage routing |
| batch | diffusion batch slot，与稳定 request id 及 per-request progress |
| retry/preempt | 同一 request 的续跑与新 request 的重试、full-payload position state |

遇到卡住或数据丢失，优先问：当前状态由哪个 owner 持有？它等待哪个 event/消息/ack？超时由谁触发？terminal cleanup 是否覆盖了队列、cache、connector pending state 和 replica binding？

## 11. 验证入口地图

以下是依据 design index 与相关页面整理的测试导航，本次没有执行这些测试。

| 要验证的行为 | 建议入口 |
| --- | --- |
| engine 输入/输出/abort 与 message correlation | [test_async_omni_engine_input.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/tests/engine/test_async_omni_engine_input.py>)、[test_async_omni_engine_outputs.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/tests/engine/test_async_omni_engine_outputs.py>)、[test_async_omni_engine_abort.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/tests/engine/test_async_omni_engine_abort.py>)、[test_rpc_result_router.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/tests/engine/test_rpc_result_router.py>) |
| 多 stage routing、错误传播、事件循环 | [test_orchestrator.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/tests/engine/test_orchestrator.py>)、[test_orchestrator_error_handling.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/tests/engine/test_orchestrator_error_handling.py>)、[test_orchestrator_event_driven.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/tests/engine/test_orchestrator_event_driven.py>) |
| runtime startup、replica/session binding | [test_async_omni_engine_stage_init.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/tests/engine/test_async_omni_engine_stage_init.py>)、[test_stage_runtime_launch_concurrency.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/tests/engine/test_stage_runtime_launch_concurrency.py>)、[test_stage_pool_session_affinity.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/tests/engine/test_stage_pool_session_affinity.py>) |
| AR prefill、streaming、spec decode、preemption、KV transfer | [tests/core/sched/](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/tests/core/sched/>)，尤其 [prefill admission](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/tests/core/sched/test_omni_ar_scheduler_prefill_admission.py>)、[streaming](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/tests/core/sched/test_omni_ar_scheduler_streaming.py>)、[spec decode](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/tests/core/sched/test_omni_ar_scheduler_spec_decode.py>)、[recompute preemption](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/tests/core/sched/test_omni_recompute_preemption.py>) |
| Prefix-cache manager / adapter | [test_prefix_cache.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/tests/core/test_prefix_cache.py>)、[test_prefix_cache_adapter.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/tests/core/test_prefix_cache_adapter.py>)、[test_prefix_cache_mrv2.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/tests/core/test_prefix_cache_mrv2.py>) |
| Diffusion scheduler / cancel / worker result | [test_diffusion_scheduler.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/tests/diffusion/test_diffusion_scheduler.py>)、[test_request_cancellation.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/tests/diffusion/test_request_cancellation.py>)、[test_result_pump.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/tests/diffusion/test_result_pump.py>)、[test_diffusion_worker.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/tests/diffusion/test_diffusion_worker.py>) |
| Diffusion batching 与 request lifecycle | [test_diffusion_batching.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/tests/diffusion/batching/test_diffusion_batching.py>)、[test_step_request_lifecycle.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/tests/diffusion/test_step_request_lifecycle.py>) |
| connector / full payload / output ownership | [test_full_payload_lifecycle.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/tests/worker/test_full_payload_lifecycle.py>)、[test_omni_ar_model_runner.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/tests/worker_v2/test_omni_ar_model_runner.py>)、[test_omni_data_plane.py](</media/lyxwxj/Data/common/Workspace/Omni-infra/Engine/vllm-omni-batching/tests/worker_v2/test_omni_data_plane.py>) |

## 12. 按问题继续深入

- 想知道某个 model_type 到底有哪些 stage：从 **OMNI_PIPELINES** 查起，再看对应 deploy yaml、StagePipelineConfig 和 StageRuntime plan。
- 想追一个请求为何卡在 stage N：先定位 orchestrator request state 与 StagePool binding，再检查该 stage 的 client 输出 pump 和 scheduler waiting state。
- 想追多模态 payload 丢失：区分 output snapshot、native data plane enqueue、connector delivery、receiver readiness 和 stage routing。
- 想追 KV/cache 误复用：确认 scheduler free 状态、deferred-free fence、prefix-cache identity/invalidation 和 GPU stream completion。
- 想追取消泄漏：沿入口 abort → Orchestrator cleanup → 每个 StagePool.abort_requests → stage scheduler/worker cleanup → connector pending payload 清理。
- 想追 diffusion batch 不兼容：从 RequestScheduler compatibility key 或 StepScheduler 的 per-request progress 入手，不从 pipeline model class 开始猜。

## 13. 本次核验状态

- 已读取贡献指南、CI review/issue lessons、design index，以及与 engine orchestration、stage runtime、AR、cache、diffusion、connector 和 async output 相关的设计页；关键描述与源码位置交叉核对。
- 已定位上述测试文件并把它们作为后续行为验证入口。
- **本次未执行 pytest、GPU/e2e 或模型加载。** 本笔记回答的是“从源码和当前设计页如何追运行时”，不声称设备级行为已在本环境实测。
