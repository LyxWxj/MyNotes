# 工作日记

## 2026-06-11（周三）

### Wan2.2 A14B T2V 性能 Profile

在 vllm-omni 上对 Wan2.2 A14B T2V 模型进行性能 profiling，分析其在不同配置下的推理延迟和吞吐。

主要关注：
- 各阶段（encode / denoise / decode）的耗时分布
- 不同分辨率下的性能差异
- GPU 显存占用分析

**初步发现**：denoise 阶段占比 70%+，encode/decode 相对轻量。这与 ISSUE #4590（Disaggregated DiT Inference RFC）中提到的 "stage 间资源需求差异大" 一致。

---

## 2026-06-12（周四）

### Wan2.2 5B TI2V 性能 Profile

继续 profiling 工作，对 Wan2.2 5B TI2V（Text+Image-to-Video）模型进行测试。

对比发现：
- 5B 模型相比 A14B 显著更快，但生成质量有差异
- TI2I 场景下 image encoder 的耗时占比不可忽略
- denoise 阶段仍然是主要瓶颈（占总延迟 70%+）

**关联 ISSUE #4590**：RFC 提到 "Audio/video models may have multiple encoders and decoders"，TI2V 的 image encoder 就是一个典型例子。如果要把 image encoder 拆分为独立 stage，就需要 DAG 拓扑支持。

---

## 2026-06-13（周五）

### Profiling 总结与优化方向

整理 Wan2.2 的 profiling 数据，初步识别优化方向：
- denoise 阶段的 step-level batching 可以提升吞吐
- encode/decode 阶段与 denoise 阶段的资源需求差异大，适合分离部署
- 开始关注 Diffusion Serving 领域的系统论文

**关联 ISSUE #4590**：RFC 的核心假设就是 "stage 间资源需求差异大"，profiling 数据验证了这一点。开始阅读 TridentServe 等论文，寻找理论支撑。

---

## 2026-06-14（周六）

### 论文阅读：TridentServe

阅读 **TridentServe: A Stage-level Serving System for Diffusion Pipelines**（北大 Hetu 团队，arXiv:2510.02838）。

核心思路：
- 将 Diffusion Pipeline 拆分为 Encode → Diffuse → Decode 三个 stage
- 每个 stage 独立分配资源（GPU 数量、并行度）
- stage 之间通过流水线并行重叠执行

关键数据：2.5× 平均延迟降低，3.6×/4.1× P95 延迟降低。

**启发**：TridentServe 的 stage-level 资源分配思想与 vllm-omni 正在做的 disaggregated DiT inference 高度相关。vllm-omni 的 PR #3208（disaggregated VAE）本质上就是在实现类似 TridentServe 的 stage 分离。回头看 ISSUE #4590 的 RFC，其核心思路与 TridentServe 一致：把 Encode → Diffuse → Decode 拆开独立调度。

---

## 2026-06-15（周日）

### 论文阅读：GenServe

阅读 **GenServe: Efficient Co-Serving of Heterogeneous Diffusion Model Workloads**（arXiv:2604.04335）。

核心思路：
- 解决 T2I 和 T2V 异构工作负载的 co-serving 问题
- 关键机制：step-level 资源适配、视频请求抢占、弹性序列并行动态批处理、SLO-aware 调度
- SLO 达成率提升 44%

**启发**：GenServe 的 step-level 调度思想可以借鉴到 vllm-omni 的 denoise 阶段——不同请求可以在不同 timestep 上交错执行，而不是等一个请求完成 18 步再去处理下一个。这与 ISSUE #4590 提到的 "Performance optimization: reduce bubbles and scheduler state" 直接相关。

---

## 2026-06-16（周一）

### 论文阅读：DiT-Serve + TetriServe

阅读 **DiT-Serve**（OpenReview）和 **TetriServe**（ASPLOS '26，University of Michigan）。

DiT-Serve 核心：
- Step-level batching + Brick Attention（binpack 不同 context length）
- 2-3× 吞吐提升，3-4× 延迟降低

TetriServe 核心：
- 弹性序列并行（dynamic sequence parallelism）
- 根据请求的分辨率和 deadline 动态调整并行度

**启发**：Brick Attention 的 binpack 思想可以用于 vllm-omni 的 denoise 阶段；TetriServe 的动态并行度可以作为未来的优化方向。

---

## 2026-06-17（周二）

### vllm-omni 社区动态

系统性地跟踪 vllm-omni 社区的几个关键 issue 和 PR：

- **ISSUE #4590**：Disaggregated DiT Inference RFC — 提出将 diffusion pipeline 拆分为独立 stage，支持异构部署
- **ISSUE #4021**：Unified VllmOmniConfig — 配置系统重构，解决 OmniDiffusionConfig 60+ 字段的 "god object" 问题
- **PR #3208**：feat(diffusion): support Qwen-Image disaggregated VAE — 具体实现 stage 分离，新增 `_STAGE_COMPONENTS` 和 `execute_encode`/`execute_decode`
- **PR #3076**：Migrate Existing Pipelines to SupportsComponentDiscovery — Nick Cao 引入通用组件发现协议

**关键发现**：PR #3076 和 PR #3208 解决的是不同问题——#3076 是 CPU offload 的组件发现，#3208 是 stage 分离的组件映射。但两者可以桥接。

---

## 2026-06-18（周三）

### 阅读 vllm-omni 源码：Stage 抽象架构

系统性阅读 vllm-omni 源码，理解 stage 抽象的四层架构：

1. **PipelineConfig** — 模型拓扑声明（编译时确定，不可变）
2. **StagePipelineConfig** — 单 stage 的拓扑定义（execution_type, input_sources）
3. **StageConfig** — 合并 deploy YAML 后的运行时配置
4. **StagePool** — 运行时，管理 replicas + 路由

关键文件：
- `vllm_omni/config/stage_config.py` — Stage 配置系统
- `vllm_omni/config/pipeline_registry.py` — Pipeline 注册表
- `vllm_omni/engine/stage_pool.py` — StagePool（replica 管理、请求路由）

**关联 ISSUE #4021**：当前的配置系统有四种表示（PipelineConfig → StageConfig → OmegaConf → dict），每层都重新序列化。ISSUE #4021 提出的 Unified VllmOmniConfig 就是要解决这个问题。

---

## 2026-06-19（周四）

### 阅读 vllm-omni 源码：两条执行路径

发现 vllm-omni 为 diffusion 推理设计了两条并行的执行路径：

**完整路径**（denoise stage）：
```
StageDiffusionClient → StageDiffusionProc → DiffusionEngine → DiffusionWorker → DiffusionModelRunner
```
支持：TeaCache、torch.compile、LoRA、CPU offload、step-wise 执行、KV transfer。

**轻量路径**（encode/decode stage）：
```
StageSubModuleClient → StageSubModuleProc → _SubModuleEngine → SubModuleWorker → DiffusionSubmoduleRunner
```
只有基础的 forward 能力，无 cache、无 compile、无 step-wise。

关键设计决策：encode/decode 是单次 forward（~1s），不需要 scheduler 和多步迭代，所以用轻量路径省掉了 ~1700 行代码。

**关联 PR #3208**：这两条路径正是 PR #3208 引入的——完整路径用于 denoise stage，轻量路径用于 encode/decode stage。

---

## 2026-06-20（周五）

### 阅读 PR #3208：Qwen-Image disaggregated VAE

详细阅读 PR #3208 的代码实现，理解 stage 分离的具体做法：

- 新增 `_STAGE_COMPONENTS` 字典，定义每个 stage 加载哪些组件
- 新增 `execute_encode()` 和 `execute_decode()` 方法
- 新增 `StageSubModuleClient` / `StageSubModuleProc` / `SubModuleWorker` / `DiffusionSubmoduleRunner`
- 新增 `encode_to_denoise()` 和 `denoise_to_decode()` 作为 stage 间数据转换函数

**关键发现**：stage 分离依赖每个 Pipeline 自己定义 `_STAGE_COMPONENTS`，没有通用化机制。

**关联 ISSUE #4590**：RFC 提到 "Easy Model Integration" 是一个挑战，提出用 mixin-style 设计。但 PR #3208 的实现是模型特定的（每个 Pipeline 自己定义 `_STAGE_COMPONENTS`），没有做到通用化。

**关联 PR #3076**：Nick Cao 的 `SupportsComponentDiscovery` 协议提供了另一种组件发现方式，可以桥接到 stage 分离。

---

## 2026-06-21（周六）

### 阅读 ISSUE #4590：Disaggregated DiT Inference RFC

深入阅读 RFC 内容，理解设计目标：

- 将 diffusion pipeline 拆分为独立 stage，每个 stage 可以运行在不同设备上
- 支持异构部署（encode/decode 用便宜 GPU，denoise 用高端 GPU）
- 吞吐上界从 `1 / (T_encode + T_dit + T_decode)` 提升到 `1 / max(T_encode, T_dit, T_decode)`
- 预期 1.2x-2.2x 吞吐提升

**与论文的关联**：RFC 的 stage 分离思想与 TridentServe 高度一致，都是把 Encode → Diffuse → Decode 拆开独立调度。

**与 PR #3208 的关联**：PR #3208 是 ISSUE #4590 的具体实现，但只实现了 QwenImage 一个模型。RFC 提到要支持 "fused encode/decode across multiple models"，这需要通用化的组件发现机制（PR #3076 的 `SupportsComponentDiscovery`）。

**与 ISSUE #4021 的关联**：RFC 提到 "Performance optimization: caching and transfer for small stages"，这需要配置系统支持 per-stage 的 cache 配置，正是 ISSUE #4021 要解决的。

---

## 2026-06-22（周日）

### 阅读 ISSUE #4021：Unified VllmOmniConfig

阅读配置系统重构的 RFC，理解当前配置系统的问题：

- 四种表示：`StagePipelineConfig` → `StageConfig` → `OmegaConf` → `dict`，每层都重新序列化
- `OmniDiffusionConfig` 有 60+ 个字段，是 "god object"
- Diffusion 和 AR 的配置创建路径完全不同，没有复用

**与 stage 分离的关联**：SubModuleWorker 只需要 12 个字段，但传入了 60+ 字段的 `OmniDiffusionConfig`。配置系统重构是 stage 分离通用化的前提。

**与 PR #3208 的关联**：PR #3208 直接复用了 `OmniDiffusionConfig`，没有做拆分。如果 ISSUE #4021 的配置重构完成，SubModuleWorker 可以用轻量的 `SubModuleConfig` 替代。

**与 PR #3076 的关联**：`SupportsComponentDiscovery` 协议目前只用于 CPU offload，但如果扩展它来支持 stage 分离，就可以避免每个模型手写 `_STAGE_COMPONENTS`。

---

## 2026-06-23（周一）

### 阅读 vllm-omni 源码：Orchestrator 调度机制

深入阅读 Orchestrator 的调度逻辑：

- `_orchestration_loop()` 是主循环，轮询所有 stage 的输出
- `_forward_to_next_stage()` 是 stage 间路由的核心，但硬编码 `next_logical = src_stage_id + 1`
- 当前所有 23 个模型都使用线性链，没有任何模型使用多父节点（DAG）

**关键发现**：Orchestrator 不支持 DAG 拓扑，这是多 Encoder/Decoder 场景的核心障碍。

**关联 ISSUE #4590**：RFC 提到 "DAG-style stage abstraction" 来处理多 encoder/decoder，但当前 Orchestrator 的 `_forward_to_next_stage` 硬编码 `next_logical = src_stage_id + 1`，无法支持 fan-in/fan-out。

**关联 PR #3208**：PR #3208 的 3-stage 线性链（encode→denoise→decode）恰好不需要 DAG，但如果要支持 TI2V（text_encoder + image_encoder → denoise），就必须改 Orchestrator。

---

## 2026-06-24（周二）

### 阅读 vllm-omni 源码：DiffusionEngine 架构

阅读 DiffusionEngine 的内部架构，理解它如何包装底层模型：

```
DiffusionEngine (调度层)
  ├─ SchedulerInterface (调度策略)
  └─ DiffusionExecutor (执行层)
       └─ MultiprocDiffusionExecutor → WorkerProcess → DiffusionWorker → DiffusionModelRunner → pipeline
```

关键设计：DiffusionEngine 不直接持有模型，通过 Scheduler + Executor 分离实现灵活调度。

**关联 ISSUE #4021**：DiffusionEngine 的 Scheduler + Executor 分离设计很好，但 `OmniDiffusionConfig` 的 60+ 字段让 SubModuleWorker 也必须传入完整配置，这是 ISSUE #4021 要解决的问题。

同时阅读 DyninOmni 的 YAML 配置和 stage input processor，理解多模态生成模型（text + image + audio）的数据流。

---

## 2026-06-25（周三）

### 阅读 vllm-omni 源码：QwenImagePipeline 组件加载

阅读 QwenImagePipeline 的 `_STAGE_COMPONENTS` 和 `_init_model` 方法，理解按需加载组件的机制。

同时对比新版（vllm-omni 主仓库）和旧版（vllm-omni-batching 分支）的区别：
- 新版用 `SupportsComponentDiscovery` 协议，按组件角色分类
- 旧版用 `_STAGE_COMPONENTS` 字典，按 stage 需求分类
- 新版没有 `execute_encode`/`execute_decode`，不支持 stage 分离

**发现的问题**：
- `DiffusionSubmoduleRunner` 硬编码 `stage == "decode"` 来决定输出
- `OmniDiffusionConfig` 被滥用（60+ 字段，submodule 只需 12 个）

**关联 PR #3208 vs PR #3076**：
- PR #3208（vllm-omni-batching 分支）用 `_STAGE_COMPONENTS` 做 stage 分离，但没有合入主仓库
- PR #3076（主仓库）用 `SupportsComponentDiscovery` 做组件发现，但不支持 stage 分离
- 两者解决的是不同问题，但可以桥接

**关联 ISSUE #4021**：`OmniDiffusionConfig` 的 60+ 字段问题在 ISSUE #4021 中被明确提出，配置重构是 stage 分离通用化的前提。

---

## 2026-06-26（周四）

### 通用化 Stage 分离方案设计

基于 `SupportsComponentDiscovery` 协议，设计通用化的 stage 分离方案：

扩展协议：
```python
class SupportsComponentDiscovery(Protocol):
    _dit_modules = ["transformer"]
    _encoder_modules = ["text_encoder"]
    _vae_modules = ["vae"]
    _scheduler_modules = ["scheduler"]      # 新增
    _tokenizer_modules = ["tokenizer"]      # 新增
    
    @classmethod
    def get_stage_components(cls, stage: str) -> set[str]:  # 新增
        mapping = {
            "encode":  set(cls._encoder_modules + cls._tokenizer_modules + cls._scheduler_modules),
            "denoise": set(cls._dit_modules + cls._scheduler_modules),
            "decode":  set(cls._vae_modules),
        }
        return mapping.get(stage, set())
```

这样 `_STAGE_COMPONENTS` 就完全不需要了，任何实现协议的模型自动支持 stage 分离。

**关联 PR #3076**：这个方案是在 Nick Cao 的 `SupportsComponentDiscovery` 协议上扩展，而不是重新发明轮子。只需要加 2 个字段（`_scheduler_modules`、`_tokenizer_modules`）和 1 个方法（`get_stage_components()`）。

**关联 PR #3208**：PR #3208 的 `_STAGE_COMPONENTS` 可以被这个通用方案替代，不再需要每个模型手写。

**关联 ISSUE #4590**：RFC 提到 "Easy Model Integration" 是一个挑战，这个通用方案正好解决了这个问题。

**关联 ISSUE #4021**：如果 ISSUE #4021 的配置重构完成，`get_stage_components()` 可以直接返回 `SubModuleConfig`，不需要传入完整的 `OmniDiffusionConfig`。

---

## 2026-06-27（周五）

### 关注 JoyAI（JoyVL）全双工交互模型

发现 vllm-omni 新增了 JoyAI（JoyVL）模型，位于 `experimental/fullduplex/joyvl/`，是全双工交互式视频 AI。

**PR #4575**（2026-06-21，SYLAR）：JoyAI-VL-Interaction streaming interaction serving layer
**PR #4623**（2026-06-24，SYLAR）：JoyVL serving 修复（bounded long-term memory, timestamps, max_pixels）

#### 架构

```
experimental/fullduplex/joyvl/
├── adapter.py           ← JoyVLDuplexAdapter（全双工适配器）
├── bridges/
│   ├── backend.py       ← ModelBackend（模型后端接口）
│   └── delegation.py    ← DelegationBridge（任务委派桥接）
├── decision/
│   ├── output_parser.py ← Action 解析（SILENCE / RESPONSE / DELEGATE）
│   ├── policy.py        ← JoyVLPolicy（决策策略）
│   └── prompts.py       ← 系统 prompt 模板
├── memory/
│   ├── brain.py         ← InteractionBrain（交互大脑）
│   └── memory.py        ← SessionMemory, Summarizer（记忆系统）
└── serving/
    ├── config.py        ← InteractionConfig
    ├── server.py        ← 服务端
    └── session.py       ← InteractionSession（会话管理）
```

#### 核心设计：三种动作

```python
class Action(enum.Enum):
    SILENCE = "silence"    # 保持沉默（画面没什么值得说的）
    RESPONSE = "response"  # 主动说话（描述画面或回答问题）
    DELEGATE = "delegate"  # 委派任务（把问题交给其他模型处理）
```

#### 决策流程

```
视频帧 → sample_frames()（采样 4 帧）
  │
  ▼
JoyVLPolicy.build_messages()（构建 prompt）
  │  system: "你是一个友好的 AI 助手..."
  │  user: [时间戳 + 图片帧 + 用户提问]
  │
  ▼
LLM.generate()（调用 LLM 生成回复）
  │
  ▼
parse_action()（解析动作）
  ├─ </silence> → 沉默
  ├─ </response> 文本 → 说话
  └─ </response> 文本 </delegation> 问题 → 委派
```

#### 记忆系统

```
InteractionBrain
├─ SessionMemory
│   ├─ long_term_memory: str        ← 长期记忆（跨 chunk 汇总）
│   ├─ mid_term_summaries: list     ← 中期记忆（每个 chunk 的摘要）
│   └─ qa_history: list[QAEntry]    ← 问答历史
├─ WorkingChunk                     ← 当前 chunk 的帧和消息
└─ _chunk_frame_count               ← 帧计数器
```

每 `chunk_frames` 帧触发一次记忆整合（consolidation），生成中期摘要，定期更新长期记忆。

#### 关键实现细节

**JoyVLDuplexAdapter**（adapter.py）：
- 实现 `DuplexAdapter` 协议，声明能力：输入 {video, text}，输出 {text}，proactive=True
- `on_input()`：接收视频帧或文字提问，视频帧追加到 `_frames`，文字存为 `_pending_query`
- `should_respond()`：有视频帧就响应
- `respond()`：采样帧 → 构建 prompt → 调用 LLM → 解析动作 → 输出文字或委派

**InteractionSession**（session.py）：
- `step()` 是核心方法：接收帧列表和可选提问，返回 `StepResult`
- 流程：fold_delegations → check flush → set_query → observe frames → infer → commit → submit delegation
- 如果 `force_silence_before_query=True`，没有用户提问时跳过推理直接返回沉默
- 推理调用 `backend.generate()`，通过 OpenAI 兼容 API 与 LLM 交互

**parse_action()**（output_parser.py）：
- 解析 LLM 输出的特殊 token：
  - `</silence>` → 沉默
  - `</response> 文本` → 说话
  - `</response> 文本 </delegation> 问题` → 委派
- 如果没有标记，第一行作为回复文本

**JoyVLPolicy**（policy.py）：
- `build_messages()`：构建 system + user 消息，user 包含时间戳、图片帧、用户提问
- `commit()`：解析 LLM 输出，去重（避免重复回复），记录到记忆
- `sample_frames()`：从 N 帧中均匀采样 num_frames 帧，减少 LLM 输入量
- `_is_repeat()`：用 SequenceMatcher 检测重复回复，阈值可配置

#### 与 Diffusion Pipeline 的区别

| | Diffusion Pipeline | JoyAI |
|---|---|---|
| 输入 | 文本 prompt | 实时视频流 + 文字提问 |
| 输出 | 图片/视频/音频 | 文字回复 / 沉默 / 委派 |
| 模型 | DiT Transformer | LLM（视觉语言模型） |
| 推理方式 | 多步去噪迭代 | 单次自回归生成 |
| 调度 | Orchestrator + StagePool | 全双工 session 管理 |
| 位置 | `diffusion/models/` | `experimental/fullduplex/` |

JoyAI 在 `experimental/` 目录下，说明还是实验性功能。但它展示了 vllm-omni 的一个新方向——不只是 diffusion serving，也可以做交互式多模态 AI。

---

### DAG Parallel 设计 + 创建新分支

整理 DAG parallel 的设计思路，写入 Notes/vLLM-omni/Thoughts.md。

核心问题：
1. Orchestrator 硬编码线性链路由，不支持 DAG
2. 多 Encoder 场景（TI2I）需要 fan-in
3. 多 Decoder 场景（视频+音频）需要 fan-out
4. 条件执行：T2I 任务跳过 image_encoder

需要的架构改动：
- `_forward_to_next_stage` 改为从 `input_sources` 构建 successor 映射
- `OrchestratorRequestState` 增加 `pending_outputs` 缓存多父节点输出
- `custom_process_input_func` 支持多输入列表
- `final_stage_id` 改为 `terminal_stage_ids: set[int]`

**关联 ISSUE #4590**：DAG parallel 是 RFC 中 "DAG-style stage abstraction" 的核心实现。RFC 提到 "Audio/video models may have multiple encoders and decoders"，DAG 拓扑是解决这个问题的前提。

**关联 PR #3208**：PR #3208 的 3-stage 线性链是 DAG 的特例（0→1→2）。DAG 改动不会破坏现有功能，只是扩展了 Orchestrator 的能力。

**关联 PR #3076**：`SupportsComponentDiscovery` 的 `_encoder_modules` 可以用于推导 DAG 拓扑——多个 encoder 自然对应 DAG 的多个前驱节点。

**关联 ISSUE #4021**：DAG 拓扑需要 per-stage 的配置（不同 encoder 可以部署到不同设备），这正是 ISSUE #4021 的 Unified VllmOmniConfig 要支持的。

从 vllm-omni 主仓库最新 main 创建新分支 `feat-disaggregated-dit-inference`，准备开始实现。

---

## 工作总结

### 两周工作内容

| 时间 | 工作 | 产出 | 关联 ISSUE/PR |
|---|---|---|---|
| 6/11-13 | Wan2.2 性能 profiling | 识别 denoise 阶段为瓶颈，确定优化方向 | ISSUE #4590 |
| 6/14-16 | 论文阅读（TridentServe, GenServe, DiT-Serve, TetriServe） | Diffusion Serving 领域认知 | ISSUE #4590 |
| 6/17 | 关注 vllm-omni 社区动态 | 系统性跟踪 ISSUE #4590, #4021, PR #3208, #3076 | 全部 |
| 6/18-19 | 阅读 vllm-omni 源码（Stage 抽象、两条执行路径） | 理解系统架构 | PR #3208, ISSUE #4021 |
| 6/20-22 | 阅读 PR #3208, ISSUE #4590, #4021 | 理解 disaggregated inference 设计 | 全部 |
| 6/23-25 | 深入阅读 Orchestrator, DiffusionEngine, QwenImagePipeline | 发现设计问题 | PR #3208, #3076, ISSUE #4021 |
| 6/26 | 通用化 Stage 分离方案设计 | 基于 SupportsComponentDiscovery 扩展 | PR #3076, #3208, ISSUE #4590, #4021 |
| 6/27 | 关注 JoyAI + DAG Parallel 设计 + 创建新分支 | 全双工交互模型分析 + 准备实现 | PR #4575, #4623, ISSUE #4590, PR #3208, #3076, ISSUE #4021 |

### 论文与社区的关联

| 论文 | 核心思想 | vllm-omni 对应 |
|---|---|---|
| TridentServe | stage-level 资源分配 | PR #3208 disaggregated VAE |
| GenServe | step-level 调度、SLO-aware | TeaCache + step execution |
| DiT-Serve | step-level batching、Brick Attention | denoise 阶段的 batching |
| TetriServe | 弹性序列并行 | sequence_parallel_size 动态调整 |
| Splitwise | prefill/decode 分离 | encode/denoise 分离 |
| — | 全双工交互式 AI | PR #4575, #4623 JoyAI (JoyVL) |

### 关键发现

1. **TridentServe 的 stage-level 资源分配思想**与 vllm-omni 的 disaggregated inference 高度一致（ISSUE #4590）
2. **Orchestrator 硬编码线性链**是 DAG parallel 的核心障碍（ISSUE #4590）
3. **`SupportsComponentDiscovery` 可以桥接到 stage 分离**，实现通用化（PR #3076 → PR #3208）
4. **`OmniDiffusionConfig` 滥用**需要在配置重构中解决（ISSUE #4021）
5. **PR #3208 的 `_STAGE_COMPONENTS` 可以被通用方案替代**，不再需要每个模型手写（PR #3076 扩展）

### 下一步

1. 扩展 `SupportsComponentDiscovery` 协议（PR #3076），加 `get_stage_components()` 方法
2. 改 Orchestrator 支持 DAG 拓扑（ISSUE #4590）
3. 从 main 分支开始实现，提交 PR
4. 持续跟踪 ISSUE #4021 的配置重构进展，协调 SubModuleConfig 设计

---

*最后更新: 2026-06-27*
