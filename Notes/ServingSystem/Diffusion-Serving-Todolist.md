---
type: Note
related_to: "[[ServingSystem]]"
status: Active
---

# Diffusion Model Serving 论文阅读清单

> Diffusion Model Serving 领域核心论文阅读计划。包含 Zotero 报告中的论文，以及这些论文引用的该领域最重要论文。
>
> **阅读策略**：先读 Serving 系统论文建立领域认知，再读基础模型和优化技术论文补充细节。

---

## 📌 阅读优先级

> [!important] 建议阅读顺序
> 1. **Serving 系统论文**（理解该领域在解决什么问题）
> 2. **基础架构论文**（DiT, SD, Flux 等模型架构）
> 3. **优化技术论文**（缓存、少步生成、量化等）
> 4. **LLM Serving 参考**（借鉴 LLM 领域的调度和系统设计思想）

---

## 一、核心 Serving 系统论文

> [!note] 本领域最重要的系统论文，解决"如何高效部署 Diffusion Model"的问题

### 1.1 DiT-Serve（⭐ 必读）
- [x] **DiT-Serve: An Efficient Serving Engine for Diffusion Transformers** (Luo et al., 2025)
  - [OpenReview](https://openreview.net/forum?id=NGNRc7rZBg)
  - 核心贡献：step-level batching + Brick Attention（binpack 不同 context length 的请求）
  - 解决问题：空间利用率低（padding）、时间利用率短请求等待长请求）
  - 2-3× 吞吐提升，3-4× 延迟降低

### 1.2 TetriServe
- [x] **TetriServe: Efficient DiT Serving for Heterogeneous Image Generation** (Lu et al., 2026)
  - [arXiv:2510.01565](https://arxiv.org/abs/2510.01565)
  - 核心贡献：弹性序列并行（dynamic sequence parallelism）
  - 解决问题：异构工作负载（混合分辨率 + 不同 deadline）下的 SLO 达成

### 1.3 TridentServe
- [x] **TridentServe: A Stage-level Serving System for Diffusion Pipelines** (Xia et al., 2025)
  - [arXiv:2510.02838](https://arxiv.org/abs/2510.02838)
  - 核心贡献：stage-level 资源分配（encode–diffuse–decode 三阶段不同资源需求）
  - 解决问题：现有系统对每个 stage 分配相同资源，导致利用率低
  - 2.5× 平均延迟降低，3.6×/4.1× P95 延迟降低

### 1.4 GENSERVE
- [x] **GENSERVE: Efficient Co-Serving of Heterogeneous Diffusion Model Workloads** (Ye et al., 2026)
  - [arXiv:2604.04335](https://arxiv.org/abs/2604.04335)
  - 核心贡献：T2I 和 T2V 异构工作负载的 co-serving
  - 关键机制：step-level 资源适配、视频请求抢占、弹性序列并行动态批处理、SLO-aware 调度
  - SLO 达成率提升 44%

### 1.5 StreamDiffusionV2（⭐ 必读）
- [ ] **StreamDiffusionV2: A Streaming System for Dynamic and Interactive Video Generation** (Feng et al., 2026)
  - [arXiv:2511.07399](https://arxiv.org/abs/2511.07399) — **MLSys 2026**
  - 核心贡献：实时流式视频生成的完整系统
  - 关键技术：SLO-aware batching scheduler、sink-token-guided rolling KV cache、motion-aware noise controller、pipeline orchestration（跨 denoising steps 和 network layers 并行）
  - 14B 模型 58.28 FPS，1.3B 模型 64.52 FPS（4×H100）

### 1.6 DistriFusion（⭐ 必读）
- [ ] **DistriFusion: Distributed Parallel Inference for High-Resolution Diffusion Models** (Li et al., 2024)
  - [CVPR 2024](https://arxiv.org/abs/2402.19481)
  - 核心贡献：displaced patch parallelism，利用相邻去噪步骤的相关性实现异步通信
  - 近线性多 GPU 加速，生成质量无损

---

## 二、流式与实时推理系统

> [!note] 面向实时交互场景的推理系统

### 2.1 StreamDiffusion（v1）
- [ ] **StreamDiffusion: A Pipeline-level Solution for Real-Time Interactive Generation** (Kodaira et al., 2023)
  - [GitHub](https://github.com/cumulo-autumn/StreamDiffusion)
  - 核心贡献：pipeline-level 流式推理，~100+ FPS
  - 关键技术：Stochastic Similarity Filter (SSF)、Residual CFG (RCFG)、IO Queues

### 2.2 FastServe
- [ ] **FastServe: Efficient Online Serving of Diffusion Models** (Wu et al., 2024)
  - 核心贡献：preemptive scheduling for diffusion models
  - 关键技术：skip-join MLFQ（Multi-Level Feedback Queue），避免 convoy effect
  - 利用扩散模型迭代特性进行细粒度任务调度

---

## 三、并行与分布式推理技术

> [!note] 多 GPU 场景下的并行策略

### 3.1 USP
- [ ] **USP: A Unified Sequence Parallelism Approach for Long Context Generative AI** (2024)
  - 核心贡献：统一序列并行框架，结合 Ring Attention 和 DeepSpeed-Ulysses
  - 适用于高分辨率图像和视频生成的长序列场景

### 3.2 PipeFusion
- [ ] **PipeFusion: Pipeline Parallelism for Diffusion Transformer Inference** (2024)
  - 核心贡献：利用相邻去噪步骤的相似性实现 pipeline parallelism
  - 不同 GPU 同时处理不同去噪步骤，减少通信开销

---

## 四、推理优化技术

> [!note] 不依赖分布式部署的单卡优化方法

### 4.1 缓存优化
- [ ] **DeepCache: Accelerating Diffusion Models for Free** (Ma et al., CVPR 2024)
  - [arXiv:2312.00858](https://arxiv.org/abs/2312.00858) — [GitHub](https://github.com/horseee/DeepCache)
  - 核心贡献：利用 U-Net 时间冗余，缓存高层特征复用，~2× 加速
  - 无需训练，plug-and-play

- [ ] **FasterDiT** — DiT 架构的特征缓存优化

### 4.2 少步生成与蒸馏
- [ ] **Consistency Models** (Song et al., ICML 2023)
  - [arXiv:2303.01469](https://arxiv.org/abs/2303.01469)
  - 核心贡献：单步/少步生成，从扩散轨迹直接映射到干净数据
  - 两种训练方式：Consistency Distillation (CD) 和 Consistency Training (CT)

- [ ] **Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference** (Luo et al., 2023)
  - [arXiv:2310.04378](https://arxiv.org/abs/2310.04378)
  - 核心贡献：在潜空间中实现 2-4 步高质量生成
  - LCM-LoRA：通用 adapter，可应用于各种微调 SD 模型

- [ ] **LCM-LoRA** — Latent Consistency Models 的 LoRA adapter 版本

### 4.3 量化
- [ ] **Q-Diffusion** — Diffusion Model 的量化方法
- [ ] **PTQ4DiT** — DiT 架构的训练后量化
- [ ] **DiTAS** — Diffusion Transformer Architecture Search

### 4.4 注意力优化
- [ ] **DiTFastAttn** — DiT 的训练后稀疏注意力加速
- [ ] **FlashAttention** (Dao et al., 2022) — [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) — 高效注意力计算（基础技术）

---

## 五、基础模型架构

> [!note] 理解 Serving 系统所服务的模型架构

### 5.1 Flux
- [ ] **Flow Matching for Generative Modeling** (Lipman et al., ICLR 2023)
  - [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)
  - Flow Matching 的理论基础，Flux 模型的核心技术

---

## 六、LLM Serving 参考论文

> [!tip] LLM Serving 领域的成熟系统设计，可借鉴其调度、批处理和资源管理思想

### 6.1 经典系统
- [ ] **vLLM: Efficient Memory Management for Large Language Model Serving with PagedAttention** (Kwon et al., SOSP 2023)
  - [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
  - PagedAttention、连续批处理、高效显存管理

- [ ] **Orca: A Distributed Serving System for Transformer-Based Generative Models** (Yu et al., OSDI 2022)
  - 连续批处理（iteration-level scheduling）的开创性工作

- [ ] **Splitwise: Efficient Generative LLM Inference Using Phase Splitting** (Patel et al., 2023)
  - Microsoft Research，prefill/decode 阶段分离到不同硬件
  - 对 Diffusion Serving 中的 stage-level 分离有参考价值

### 6.2 调度与批处理
- [ ] **Sarathi: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills** (2023)
  - chunked prefill + decode 混合批处理

- [ ] **LoRAX: Multi-LoRA Serving** (Predibase, 2023)
  - 多 LoRA adapter 的高效共服务
  - 对多风格/多任务 Diffusion 服务有参考价值

### 6.3 异构部署
- [ ] **Helix: Serving Large Language Models over Heterogeneous GPUs and Network via Max-Flow** (Mei et al., ASPLOS 2025)
  - [ACM DL](https://dl.acm.org/doi/10.1145/3669940.3707215)（Zotero 报告中的论文）
  - Max-Flow 算法优化异构 GPU 网络上的 LLM 服务

---

## 七、综合参考

### 7.1 技术报告
- [ ] **EECS-2025-46.pdf** — Berkeley 技术报告（Zotero 报告中的附件）
  - [PDF](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/Archive/EECS-2025-46.pdf)
  - 可能是 DiT-Serve 或相关系统的学位论文

### 7.2 综述论文
- [ ] **A Survey on Efficient Inference for Large Language Models** (2024)
  - 虽然聚焦 LLM，但量化、KV-cache、推测解码等技术可迁移到 DiT

- [ ] **Diffusion Models: A Comprehensive Survey of Methods and Applications** (ACM Computing Surveys, 2024)
  - 覆盖完整 pipeline，包括推理优化

---

## 📊 论文分类速查表

| 类别 | 论文 | 核心技术 | 年份 |
|------|------|----------|------|
| **Serving 系统** | DiT-Serve | step-level batching, Brick Attention | 2025 |
| **Serving 系统** | TetriServe | 弹性序列并行 | 2026 |
| **Serving 系统** | TridentServe | stage-level 资源分配 | 2025 |
| **Serving 系统** | GENSERVE | T2I/T2V co-serving | 2026 |
| **流式系统** | StreamDiffusionV2 | 流式 pipeline orchestration | 2026 |
| **流式系统** | StreamDiffusion | pipeline-level 流式推理 | 2023 |
| **分布式推理** | DistriFusion | displaced patch parallelism | 2024 |
| **分布式推理** | USP | 统一序列并行 | 2024 |
| **分布式推理** | PipeFusion | pipeline parallelism for DiT | 2024 |
| **缓存优化** | DeepCache | U-Net 特征缓存 | 2024 |
| **少步生成** | Consistency Models | 单步生成 | 2023 |
| **少步生成** | LCM | 潜空间少步生成 | 2023 |
| **模型架构** | DiT | Transformer-based diffusion | 2023 |
| **模型架构** | Stable Diffusion | 潜空间扩散 | 2022 |
| **模型架构** | SANA | 线性注意力 DiT | 2024 |
| **LLM 参考** | vLLM | PagedAttention | 2023 |
| **LLM 参考** | Splitwise | prefill/decode 分离 | 2023 |
| **LLM 参考** | Helix | 异构 GPU 部署 | 2025 |

---

## 🎯 关键术语

| 术语 | 定义 |
|------|------|
| **DiT** | Diffusion Transformer，用 Transformer 替代 U-Net 的扩散模型架构 |
| **SLO** | Service Level Objective，服务级别目标（延迟、吞吐等） |
| **Step-level Batching** | 每个去噪步骤进行调度和批处理，而非整个推理过程 |
| **Sequence Parallelism** | 将序列维度分片到多个 GPU |
| **Brick Attention** | DiT-Serve 提出的注意力算法，binpack 不同 context length |
| **Displaced Patch Parallelism** | DistriFusion 的 patch 级并行，利用相邻步骤相关性 |
| **Pipeline Parallelism** | 将模型/推理拆分到多个设备流水线执行 |
| **Continuous Batching** | 请求级别而非批次级别的动态批处理 |
| **Prefill/Decode** | LLM 的两个推理阶段，计算特性不同 |

---

*最后更新: 2026-06-18*
*标签: #DiffusionServing #DiT #分布式推理 #系统优化*
