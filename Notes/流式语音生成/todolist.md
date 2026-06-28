---
type: Note
related_to: "[[vllm-omni]]"
status: Active
---

# 流式语音生成 — 工作规划

> 对应项目计划书「研究内容一：面向实时多模态分块推理的执行建模、连续组批与调度优化」中语音交互相关部分，以及研究内容二（异步流水线）和研究内容三（Session-aware KVCache）中与语音生成相关的模块。
>
> **最新任务**：Duplex 双工语音系统研究（MiniCPM-o 4.5 + Omni 社区 RFC）

---

## 📌 当前任务优先级

> [!important] 本周重点
> 1. **Duplex 双工系统架构理解** - 理解 s2t + tts 双工模型设计
> 2. **MiniCPM-o 4.5 Duplex 实现分析** - 代码阅读与架构梳理
> 3. **Omni 社区 RFC #3745** - 参与社区讨论，理解设计意图
> 4. **Audio Chunk 调度机制** - context-chunk vs request-chunk

---

## 🔬 核心研究问题

> [!question] 需要回答的关键问题
> 1. Duplex 系统如何实现 s2t 和 tts 的并行处理？
> 2. context-chunk (10s) 和 request-chunk (200ms~1s) 的调度策略是什么？
> 3. t token（文本）和 z token（audio latent）如何合并调度？
> 4. Session 级别的上下文管理如何影响 KVCache 复用？
> 5. Async CPU zero-overhead 如何优化语音模型的 CPU 开销？

---

## 阶段零（第 0–1 个月）：Duplex 双工系统调研

> [!note] 新增阶段
> 基于最新分配的任务，优先理解 Duplex 双工语音系统架构

### 资源收集与阅读

- [ ] **论文阅读**
  - [ ] [Duplex TTS 论文](https://arxiv.org/pdf/2602.02204) - 双工语音合成的理论基础
  - [ ] [Google Slides 演示](https://docs.google.com/presentation/d/111-L8zF7A1j_YI_cR8JsblofdScdRr2f/edit#slide=id.p88) - Think Machine Lab 的 Duplex 架构设计
  - [ ] 相关背景论文：StreamSpeech、SpeechGPT、VALL-E X

- [ ] **社区 RFC 分析**
  - [ ] [Omni RFC #3745](https://github.com/vllm-project/vllm-omni/issues/3745) - 仔细阅读 RFC 内容
  - [ ] 分析社区对 Duplex 系统的设计讨论
  - [ ] 梳理 RFC 中的关键技术点和争议问题

### Duplex 架构理解

- [ ] **双工模型组成**
  - [ ] s2t (Speech-to-Text) 模块：实时语音识别
  - [ ] tts (Text-to-Speech) 模块：实时语音合成
  - [ ] 双工并行机制：如何实现同时收听和说话？

- [ ] **Agent Streaming vs Omni Duplex**
  - [ ] Agent Streaming：传统串行流程（function calling → s2t → tts）
  - [ ] Omni Duplex：原子性服务能力，支持并行处理
  - [ ] 对比分析：延迟、吞吐、复杂度

### MiniCPM-o 4.5 Duplex 实现

> [!tip] 代码阅读重点
> 关注 `vllm_omni/model_executor/models/minicpmo_4_5/` 中的 Duplex 相关实现

- [ ] **Thinker-Talker 并行机制**
  - [ ] 分析 Thinker 和 Talker 如何并行执行
  - [ ] 理解 hidden states 的流式传递
  - [ ] 关注 `minicpmo_4_5_omni.py` 中的状态管理

- [ ] **Session 状态管理**
  - [ ] 会话级别的上下文维护
  - [ ] 多阶段状态流转（Listening → Thinking → Speaking）
  - [ ] 状态切换的触发条件和时机

- [ ] **接口转换 Realtime**
  - [ ] 理解如何将非实时模型转换为实时接口
  - [ ] 分析音频 buffer 和 embedding cache 的作用
  - [ ] 关注 prequery 等音频特殊逻辑

### Audio Chunk 调度机制

> [!important] 核心概念
> - **context-chunk (10s)**: CPU worker 积攒的音频块，包含上下文信息
> - **request-chunk (200ms~1s)**: 每次发送的增量语音包

- [ ] **context-chunk 处理**
  - [ ] 滑动窗口机制：如何处理长序列音频
  - [ ] 上下文信息：上一个窗口的 ASR/ST 文本、TTFF
  - [ ] context 信息的传递和复用

- [ ] **request-chunk 调度**
  - [ ] 小请求的调度逻辑
  - [ ] Session 绑定策略：同一 session 的请求路由
  - [ ] 目标函数优化：延迟 vs 吞吐的平衡

### 系统侧优化

- [ ] **Async CPU Zero-overhead**
  - [ ] 分析语音模型的 CPU 开销来源
  - [ ] 理解异步 CPU 处理的实现方式
  - [ ] 识别优化机会：预处理、后处理、编码解码

- [ ] **双 Token 调度**
  - [ ] t token（文本 token）：LLM 输出
  - [ ] z token（audio latent token）：语音 codec token
  - [ ] mt1z 调度特性：混合 token 的调度策略
  - [ ] tz 合并方案：如何统一调度两种 token

### 阶段零输出

1. **Duplex 双工系统架构文档** - 完整的架构图和设计说明
2. **MiniCPM-o 4.5 Duplex 代码分析报告** - 关键模块的实现细节
3. **Audio Chunk 调度策略分析** - context-chunk vs request-chunk 的调度设计
4. **双 Token 调度方案草图** - t token 和 z token 的合并调度设计

---

## 阶段一（第 1–3 个月）：负载分析与性能建模

### 代码阅读

- [ ] **vLLM-omni 异步分块机制**
  - `vllm_omni/model_executor/stage_input_processors/qwen3_omni.py` — Thinker→Talker、Talker→Code2Wav 的 chunk 传递逻辑
  - `vllm_omni/distributed/omni_connectors/transfer_adapter/` — OmniConnector 数据传输适配层
  - `vllm_omni/core/sched/omni_ar_scheduler.py` — AR 阶段调度器
  - `vllm_omni/worker/gpu_model_runner.py` — GPU 模型执行器
  - 目标：理解 `async_chunk` 的调度流程、chunk 生命周期、`WAITING_FOR_CHUNK` 状态机

- [ ] **Qwen3-Omni 模型代码**
  - `vllm_omni/model_executor/models/qwen3_omni/` — 完整模型定义
  - 重点关注：Talker 模型结构、MTP（Multi-Token Prediction）模块、Codec token 生成逻辑
  - 目标：理解 Talker 如何从 Thinker 输出流式生成 codec token

- [ ] **Code2Wav 模块**
  - `vllm_omni/model_executor/models/qwen3_omni/` 中 Code2Wav 相关文件
  - 目标：理解 codec token → 波形的解码流程、流式解码的 chunk 粒度（默认 25 frames）、批处理配置

- [ ] **AsyncOmni 架构**
  - `vllm_omni/engine/async_omni.py` — 异步引擎客户端
  - `vllm_omni/engine/async_omni_engine.py` — 引擎代理层
  - `vllm_omni/engine/orchestrator.py` — 编排层，请求路由与输出编排
  - 目标：理解多 stage 异步执行的完整数据流

- [ ] **Stage 分离与 Disaggregation**
  - `docs/design/feature/disaggregated_inference.md`
  - `vllm_omni/distributed/omni_connectors/` — MooncakeConnector 等跨节点传输
  - 目标：理解 Thinker(prefill) → Thinker(decode) → Talker → Code2Wav 的四级分离如何实现

### 性能分析

- [ ] **端到端延迟拆解**
  - 使用 `torch profiler` / `Nsight Systems` 对 Qwen3-Omni 语音生成链路做算子级 profiling
  - 拆解各阶段耗时：Thinker decode、Talker、Code2Wav、KVCache 访问、stage 间数据传输
  - 输出：各阶段 stage ratio（随 batch size、session 数变化）

- [ ] **chunk 粒度对延迟的影响**
  - 测试不同 `codec_chunk_frames`（如 10/25/50）对 TTFP（首音频时间）、RTF（实时因子）、端到端延迟的影响
  - 分析 chunk 粒度与 GPU 利用率的 trade-off

- [ ] **并发场景下的瓶颈定位**
  - 1/5/10/20 session 并发下，各阶段 queue wait、计算、通信、KVCache 访问的占比
  - 定位高并发下语音生成链路的瓶颈（是 Talker decode 慢、Code2Wav 慢、还是 stage 传输慢？）

- [ ] **Duplex 系统特殊分析**
  - [ ] 双工并行带来的额外开销分析
  - [ ] s2t 和 tts 同时执行时的 GPU 争用
  - [ ] Session 切换的延迟开销

### 论文阅读

- [ ] **Qwen3-Omni 技术报告** — 理解 Thinker-Talker 架构、MTP 机制、流式 codec 生成设计
- [ ] **SoundStream** (Zeghidour et al., 2021) — 端到端神经音频 codec，理解 codec token 的生成方式
- [ ] **EnCodec** (Défossez et al., 2022) — 高保真神经音频压缩，理解残差向量量化（RVQ）
- [ ] **VALL-E** (Wang et al., 2023) — 基于 codec token 的语音合成范式，理解 AR + NAR 混合生成
- [ ] **CosyVoice** (Du et al., 2024) — 阿里流式语音合成系统，理解 chunk-wise 流式生成策略
- [ ] **Duplex TTS 相关论文** — 理解双工语音合成的最新进展

### 阶段一输出

1. 长期会话语音生成负载与瓶颈分析报告
2. 各阶段（Talker / Code2Wav / KVCache / 传输）耗时占比数据
3. chunk 粒度敏感性分析结果
4. **Duplex 系统性能特征分析**

---

## 阶段二（第 4–6 个月）：多阶段异步流水线设计

### 代码阅读

- [ ] **vLLM-omni 调度器深入**
  - `omni_ar_scheduler.py` 中 chunk 调度策略、deadline 感知逻辑
  - `OmniGenerationScheduler` — 生成阶段调度器（Code2Wav）
  - 目标：理解当前调度器如何处理语音生成的连续 chunk 到达

- [ ] **Stage 间通信与流水线控制**
  - `OmniChunkTransferAdapter` — chunk 生命周期管理
  - stage_input_processors 中的 chunk 拆分与合并逻辑
  - 目标：理解 stage 间如何实现异步重叠，当前有哪些同步阻塞点

- [ ] **请求状态机**
  - `RequestStatus` 枚举及相关状态转换
  - `WAITING_FOR_CHUNK` 状态的触发条件与恢复逻辑
  - 目标：理解 chunk 级别的状态管理

- [ ] **Duplex 状态机**
  - [ ] Listening → Thinking → Speaking 状态流转
  - [ ] 状态切换的触发条件（VAD、静音检测、用户打断）
  - [ ] 并发状态处理：同时收听和说话

### 设计与实现

- [ ] **语音生成 stage 分离流水线优化**
  - 分析当前 Thinker→Talker→Code2Wav 流水线中的 bubble（空泡）
  - 设计 Talker 与 Code2Wav 的异步重叠方案：Talker 生成 codec token 的同时 Code2Wav 解码前一批
  - 评估是否需要将 Code2Wav 进一步拆分为多 chunk 并行解码

- [ ] **阶段内并行策略选择**
  - 分析 Talker（3B MoE）在不同 TP/EP 下的 MFU 与延迟
  - 分析 Code2Wav（200M ConvNet）是否适合与 Talker 共享 GPU 或独立部署
  - 输出：不同并发场景下的推荐并行配置

- [ ] **Duplex 双工并行优化**
  - [ ] s2t 和 tts 的流水线重叠方案
  - [ ] 双工场景下的 GPU 资源分配策略
  - [ ] Session 级别的流水线调度

- [ ] **双 Token 调度实现**
  - [ ] t token 和 z token 的统一调度器设计
  - [ ] mt1z 调度算法实现
  - [ ] tz 合并的性能优化

### 论文阅读

- [ ] **CosyVoice 2** (Du et al., 2025) — 流式语音合成的最新进展，chunk-aware 块调度
- [ ] **F5-TTS** (Chen et al., 2024) — 基于 DiT 的流式 TTS，理解非自回归流式生成
- [ ] **MELLE** (Zhang et al., 2024) — 连续语音生成，理解 continuous codec 生成范式
- [ ] **Streaming LLM inference with pipeline parallelism** — 理解多 stage 流水线并行的通用设计模式
- [ ] **Orca** (Yu et al., 2022) — 连续批处理（continuous batching）的开创性工作，理解 chunk-level 调度思想
- [ ] **Duplex Dialogue Systems** — 双工对话系统的最新研究

### 阶段二输出

1. 多阶段分离式运行时原型（语音生成链路）
2. Talker 与 Code2Wav 异步重叠方案与实现
3. 阶段空泡与利用率分析结果
4. **Duplex 双工并行优化方案**
5. **双 Token 调度器原型**

---

## 阶段三（第 7–9 个月）：Session-aware KVCache 管理

### 代码阅读

- [ ] **vLLM KVCache 管理**
  - vLLM 核心的 block manager、prefix matching、KVCache 分配与回收
  - vLLM-omni 中 session 状态的维护方式（如果已实现）
  - 目标：理解当前 KVCache 管理在长 session 语音交互场景下的开销

- [ ] **跨模态条件缓存**
  - Thinker 输出的 vision/audio hidden states 在 Talker 中的缓存与复用
  - 长 session 下条件编码缓存的生命周期管理
  - 目标：理解哪些中间状态可以 session 级复用

- [ ] **Duplex Session 管理**
  - [ ] 双工 session 的 KVCache 特殊需求
  - [ ] s2t 和 tts 共享的上下文信息
  - [ ] Session 切换时的缓存迁移策略

### 设计与实现

- [ ] **Session 级 KVCache 持久化**
  - 设计语音交互 session 的 KVCache 生命周期：创建、chunk 间复用、session 结束清理
  - 避免每个 chunk 重复执行 prefix matching 和 block 分配
  - 关键路径只保留轻量级索引和 cache read

- [ ] **异步缓存管理**
  - 缓存压缩、量化、迁移、淘汰放在异步路径
  - 关键路径（chunk 间延迟）不被缓存管理操作阻塞
  - 输出：缓存索引开销 P99 < 1ms 的实现方案

- [ ] **Duplex 特化缓存**
  - [ ] 双工 session 的专用缓存池
  - [ ] s2t 和 tts 的缓存隔离与共享
  - [ ] 预测性缓存预热（基于对话模式）

### 论文阅读

- [ ] **PagedAttention** (Kwon et al., 2023) — vLLM 的核心 KVCache 管理机制
- [ ] **Prefix Caching / Prompt Caching** — 理解 prefix matching 和 KVCache 复用
- [ ] **ChunkAttention** (Ye et al., 2024) — 前缀树结构的 KVCache 管理，适合多 session 场景
- [ ] **RadixAttention** (Zheng et al., 2023) — SGLang 的 KVCache 复用方案
- [ ] **Attention Sink** (Xiao et al., 2023) — 理解流式场景下 KVCache 的截断与保留策略

### 阶段三输出

1. 会话级 KVCache 管理模块
2. 缓存复用与索引开销分析结果
3. 长 session 语音交互场景下重复计算降低 ≥30% 的实现
4. **Duplex 特化缓存方案**

---

## 阶段四（第 10–12 个月）：端到端调度与系统整合

### 系统整合

- [ ] **端到端调度器**
  - 整合 chunk 级调度、stage 分离流水线、session-aware KVCache
  - 支持多 session 并发、用户打断、连续组批
  - 目标：块间延迟达标率 >99%，P99 延迟 <200ms

- [ ] **多模态输入下的语音生成调度**
  - 视频通话场景：视觉/语音编码器持续输入 + 语音实时生成
  - 分析 encoder 输出到达的随机性对语音生成流水线的影响

- [ ] **Duplex 系统整合**
  - [ ] 双工调度器的端到端集成
  - [ ] s2t 和 tts 的联合优化
  - [ ] 真实对话场景的端到端测试

- [ ] **双 Token 调度系统整合**
  - [ ] t token 和 z token 的统一调度框架
  - [ ] mt1z 调度策略的性能验证
  - [ ] tz 合并的端到端优化效果

### 论文阅读

- [ ] **StreamSpeech** (Ma et al., 2024) — 流式语音到语音翻译，理解 chunk-level 同声传译
- [ ] **SpeechGPT** (Zhang et al., 2023) — 原生语音 LLM，理解 codec token 作为 LLM 输出的设计
- [ ] **VALL-E X** (Zhang et al., 2023) — 跨语言 codec 语音合成
- [ ] **实时推理系统相关**：Sarathi-Serve、DistServe、Splitwise 等论文，理解 disaggregated serving 的系统设计
- [ ] **Duplex Dialogue 最新进展** — 双工对话系统的最新研究成果

### 阶段四输出

1. 端到端原型系统（语音生成链路）
2. 完整实验报告与复现脚本
3. 论文/专利材料
4. **Duplex 双工系统完整实现**
5. **双 Token 调度系统文档**

---

## 📚 参考资源

> [!link] 重要链接

### 论文
- [Duplex TTS 论文](https://arxiv.org/pdf/2602.02204) - 双工语音合成理论基础
- [Think Machine Lab 演示](https://docs.google.com/presentation/d/111-L8zF7A1j_YI_cR8JsblofdScdRr2f/edit#slide=id.p88) - Duplex 架构设计

### 社区
- [Omni RFC #3745](https://github.com/vllm-project/vllm-omni/issues/3745) - Duplex 系统设计讨论

### 代码
- `vllm_omni/model_executor/models/minicpmo_4_5/` - MiniCPM-o 4.5 实现
- `vllm_omni/core/sched/omni_ar_scheduler.py` - AR 调度器
- `vllm_omni/engine/orchestrator.py` - 编排层

---

## 🎯 关键概念

> [!abstract] 核心术语

| 术语 | 定义 |
|------|------|
| **Duplex** | 双工系统，支持同时进行语音识别和语音合成 |
| **s2t** | Speech-to-Text，语音转文本 |
| **tts** | Text-to-Speech，文本转语音 |
| **context-chunk** | 10s 音频块，包含上下文信息，用于滑动窗口处理 |
| **request-chunk** | 200ms~1s 增量语音包，每次实际发送的数据 |
| **t token** | 文本 token，LLM 输出的文本表示 |
| **z token** | Audio latent token，语音 codec token |
| **mt1z** | 混合 token 调度策略，统一处理 t 和 z token |
| **TTFF** | Time To First Audio，首音频时间 |
| **RTF** | Real-Time Factor，实时因子 |

---

## 📊 进度跟踪

> [!tip] 周报模板

### 本周完成
- [ ] 任务 1
- [ ] 任务 2

### 下周计划
- [ ] 任务 1
- [ ] 任务 2

### 遇到的问题
- 问题 1：描述 + 解决方案
- 问题 2：描述 + 解决方案

---

## 持续跟踪

- [ ] vLLM-omni 仓库更新（关注 async_chunk、stage separation、Duplex 相关 PR）
- [ ] Qwen 系列模型更新（Qwen3-Omni 后续版本）
- [ ] MiniCPM 系列更新（MiniCPM-o 4.5+ Duplex 增强）
- [ ] 流式语音生成领域新论文（Codec-based TTS、Streaming LLM Speech、Duplex Dialogue）
- [ ] NVIDIA GTC / NeurIPS / ICML 中实时推理相关的 workshop 和论文
- [ ] Omni 社区 RFC 讨论（特别是 #3745 及相关 issue）

---

*最后更新: 2026-06-02*
*标签: #vLLM-omni #Duplex #双工语音 #流式生成 #调度优化*
