# 文献调研框架 — Stage分离与异构部署推理优化

> 目标会议：待定（请与导师确认，这决定了文献调研的深度和广度）

## 核心发现

**vLLM-Omni 已有公开论文**（arXiv:2602.02204，2026年2月），其核心思路与你的专利交底书高度重叠：将多模态推理拆分为独立 stage（Encode/Prefill/Decode/Generator），各 stage 独立部署、独立批处理。你的工作的差异化可能在于：

1. **昇腾 NPU 异构硬件的具体适配**（310P/910C/910B/CPU 四级部署）——vLLM-Omni 论文在 A100/H100 上跑
2. **三级分层缓存机制**（模型权重/中间特征/推理参数）——vLLM-Omni 主要做 KV-cache
3. **针对昇腾硬件的具体性能调优**

**先确认**：你们的工作是 vLLM-Omni 的上游修改吗？代码在哪个仓库？如果没有代码实现，论文就只能是纯方案设计（design-only paper），这会显著影响可投的会议级别。

---

## 必读论文（按阅读顺序）

### 第一梯队：直接相关，必须精读

| # | 论文 | 会议/年份 | 与你的关系 |
|---|------|-----------|-----------|
| 1 | **vLLM-Omni: Fully Disaggregated Serving for Any-to-Any Multimodal Models** | arXiv 2026.02 | **最直接相关**。你们的 baseline。拆分了 E/P/D/G 四阶段，独立部署+动态批处理。必须逐字精读 |
| 2 | **Qwen-Image Technical Report** | arXiv 2025.08 | 你们的目标模型。搞清楚原始 pipeline（VAE→Qwen2.5-VL→MMDiT→VAE-Decoder），你的 stage 切分依据来自这里 |
| 3 | **Splitwise: Efficient Generative LLM Inference Using Phase Splitting** | ISCA 2024 | disaggregated inference 的开创性工作。虽然做 LLM 不是扩散模型，但 prefill/decode 分离的思想是你的理论来源 |
| 4 | **vLLM: Efficient Memory Management for LLM Serving with PagedAttention** | SOSP 2023 | vLLM 生态的根基。vLLM-Omni 继承自此 |

### 第二梯队：重要参考，需要引用

| # | 论文 | 会议/年份 | 与你的关系 |
|---|------|-----------|-----------|
| 5 | **DistServe** | OSDI 2024 | prefill/decode 分离 + 各自独立优化并行策略，方法论可参考 |
| 6 | **Mooncake** | FAST 2025 (Best Paper) | KVCache 中心化的分离架构，Kimi 的生产系统。缓存设计可参考 |
| 7 | **DistriFusion** | CVPR 2024 (Highlight) | 扩散模型的多 GPU 并行推理，patch 级别并行 |
| 8 | **PipeFusion** | NeurIPS 2025 | 扩散 Transformer 的流水线并行，stage 间 stale feature map 复用 |
| 9 | **xDiT** | arXiv 2024.11 | DiT 推理并行引擎，支持序列并行+流水线并行+CFG并行混合 |
| 10 | **PRESERVE** | arXiv 2025.01 | 华为苏黎世出品。模型权重+KV Cache 预取，与你的缓存预加载直接相关 |

### 第三梯队：背景支撑，选择性引用

| # | 论文 | 会议/年份 | 与你的关系 |
|---|------|-----------|-----------|
| 11 | **SARATHI** | arXiv 2023.08 | chunked prefill + hybrid batching，你的批处理策略的理论来源 |
| 12 | **HeteGen** | MLSys 2024 | CPU+GPU 异构张量并行，异构部署方法论 |
| 13 | **SiPipe** | arXiv 2025.06 | CPU-GPU 异构流水线，利用 CPU 做辅助计算 |
| 14 | **LMCache** | arXiv 2025.10 | 层次化 KV 缓存，可参考其缓存层级设计 |
| 15 | **IMPRESS** | FAST 2025 | 基于重要性的缓存淘汰策略，你的缓存淘汰机制可引用 |
| 16 | **SwiftDiffusion** | arXiv 2024.07 | Stable Diffusion 生产级推理系统，ControlNet/LoRA 分离 |
| 17 | **TetriInfer** | arXiv 2024.01 | 分离式推理 + 两级调度 |
| 18 | **Hybrid SD** | arXiv 2024.08 | 边缘-云端协同扩散推理，stage 拆分的思想类似 |

---

## 文献阅读要点（针对每篇论文要提取什么）

### 读 vLLM-Omni 时要提取的

- [ ] OmniConnector 的 E/P/D/G 四阶段分别做什么
- [ ] 各阶段的参数规模、耗时占比（是否已有 profile 数据）
- [ ] 跨阶段数据传输机制（和你的流水线调度对比）
- [ ] 他们的实验在什么硬件上跑的（A100/H100 vs 你的昇腾）
- [ ] 他们是否提到了异构硬件适配（大概率没有，这是你的差异化空间）
- [ ] 局限性/未来工作中是否有提到 NPU 适配

### 读 Splitwise 时要提取的

- [ ] prefill 和 decode 的计算特征差异（compute-bound vs memory-bound）
- [ ] 如何形式化描述"硬件亲和性"（你可能可以类比到你的 4 个 stage）
- [ ] 跨阶段 KV cache 传输的延迟代价
- [ ] 实验评估用了哪些指标

### 读 Qwen-Image 时要提取的

- [ ] 精确的推理流程图
- [ ] VAE Encoder、Qwen2.5-VL、MMDiT、VAE Decoder 各模块的参数规模和 FLOPs
- [ ] 官方推理耗时数据
- [ ] 和 Stable Diffusion 3 的架构差异

### 读 PRESERVE 时要提取的

- [ ] 预取机制的具体实现
- [ ] 如何重叠通信和计算
- [ ] 华为背景——可能与昇腾平台有关联

---

## 文献综述写作框架（对应论文 Introduction + Related Work 章）

### Section: Related Work

按以下结构组织，每个子节 3-4 段，最终落在"现有工作的不足"上：

#### 2.1 Efficient LLM Serving

- 讲 vLLM (PagedAttention) → SARATHI (chunked prefill) → vLLM-Omni (multimodal disaggregation)
- **落点**：现有工作主要集中在 GPU 平台（A100/H100），未考虑昇腾 NPU 异构硬件的特性

#### 2.2 Disaggregated Inference

- 讲 Splitwise → DistServe → Mooncake → vLLM-Omni
- **落点**：现有分离式推理主要做 prefill/decode 的两阶段分离，你的工作是四个 stage 的细粒度分离 + 四级硬件精准匹配

#### 2.3 Pipeline Parallelism for Diffusion Models

- 讲 DistriFusion → PipeFusion → xDiT
- **落点**：现有并行方案是同构 GPU 内部的并行，没有跨 CPU/NPU 的异构流水线

#### 2.4 Heterogeneous Deployment

- 讲 HeteGen → SiPipe → Hybrid SD
- **落点**：现有异构部署未针对昇腾 NPU 多型号（310P/910C/910B）做亲和性匹配

#### 2.5 Caching for Model Inference

- 讲 LMCache → IMPRESS → PRESERVE
- **落点**：现有缓存方案主要做 KV-Cache，缺少模型权重+中间特征+推理参数的三级分层缓存

---

## 差异化定位分析

| 维度 | vLLM-Omni | 你们的工作 |
|------|-----------|-----------|
| Stage 分离 | E/P/D/G 四阶段 | VAE-Encoder / LLM特征编码 / MMDiT生成 / VAE-Decoder |
| 硬件平台 | NVIDIA GPU (A100/H100) | 昇腾 NPU (310P/910C/910B) + CPU |
| 异构部署 | 同构 GPU pool | 四级异构硬件精准匹配 |
| 缓存 | KV-cache (DBCache/TeaCache) | 三级缓存（权重/特征/参数） |
| 调度 | OmniRouter + 动态批处理 | 专属任务队列 + 动态批处理 + 异步流水线 |

**核心差异化论点**：现有 disaggregated inference 在同构 GPU 上已经成熟，但针对昇腾 NPU 多型号 + CPU 的异构平台尚无系统性方案。你们的工作填补了这个 gap。

---

## 后续行动清单

1. **本周**：精读 vLLM-Omni 论文 + Qwen-Image 论文（这是论文 baseline 的基础）
2. **下周**：读 Splitwise + DistServe + PRESERVE + PipeFusion（填充 Related Work）
3. **同步进行**：Clone vLLM-Omni 仓库，看懂代码中的 stage 抽象怎么实现
4. **关键决策**：找导师确认——代码到底有没有实现？如果没有实现，论文定位需要调整
