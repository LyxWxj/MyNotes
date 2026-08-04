---
type: Note
related_to:
  - "[[Diffusion]]"
  - "[[DMD]]"
status: Active
tags:
  - diffusion
  - training
  - survey
  - pretraining
  - alignment
  - distillation
---

# Diffusion 模型训练框架调研

> 调研范围：扩散模型（Diffusion / Flow Matching / 扩散语言模型 dLLM）的**训练生态**，按训练阶段划分为预训练（Pretraining）、后训练（Post-training：微调 + 偏好对齐 + 强化学习）、蒸馏（Distillation）三类，覆盖官方开源训练代码与社区训练框架。

---

## 一、总览：训练阶段划分

扩散模型的完整生命周期可划分为四个阶段，不同阶段的目标、数据与计算特性差异显著：

1. **预训练（Pretraining）**：从零训练基础模型，目标是学习数据分布。图像/视频生成模型使用大规模图文/视频数据，常见目标为 score matching、Flow Matching 或 masked diffusion；这一阶段对算力、数据与分布式训练框架要求最高，绝大多数框架由模型团队自研并随模型开源。
2. **后训练（Post-training）**：在预训练基座上继续优化，包括
   - **微调（Fine-tuning）**：全参微调或 LoRA 等参数高效微调，适配特定风格/任务；
   - **偏好对齐（Preference Alignment）**：用人类偏好数据（win/lose 对或 reward model）提升生成质量与审美（DPO 系方法）；
   - **强化学习（RL）**：把去噪过程建模为 MDP，用策略梯度（PPO/GRPO 等）或 reward 梯度直接优化。
3. **蒸馏（Distillation）**：把多步教师模型压缩为少步/单步学生模型，用于推理加速，典型方法有 ADD、LCM、一致性模型、DMD、SCoT 等。
4. **推理与服务**：不属于训练，但训练出的少步模型直接决定服务成本（仓库已有专门调研：`Notes/Diffusion/ServingSystem/业界现状调研报告.md`）。

值得注意的趋势：扩散模型与 LLM 的工程栈正在融合——扩散语言模型（dLLM，如 LLaDA、Dream）的后训练直接复用 LLM 的 RL/对齐框架（GRPO、PPO），而 Alibaba ROLL、ms-swift 等通用框架同时覆盖 LLM 与扩散模型。

---

## 二、预训练框架

### 2.1 官方开源训练代码（图像 / 视频基础模型）

| 框架 / 模型 | 团队 | 训练能力 | 备注 |
| --- | --- | --- | --- |
| [Meta DiT](https://github.com/facebookresearch/DiT) | Meta | 训练 + 推理 | 最早开源的 DiT 训练代码（含 ImageNet 分类条件），是后续所有 DiT 框架的基线 |
| [OpenDiT](https://github.com/NUS-HPC-AI-Lab/OpenDiT) | NUS HPC-AI-Lab | 高性能训练 + 推理 | 基于 Colossal-AI 的 DiT 训练系统，支持文本到视频/图像，含序列并行与高效注意力；用 8×A100 训练 80k steps 验证精度 |
| [Open-Sora 2.0](https://github.com/hpcai-tech/Open-Sora) | HPC-AI Tech | 分布式训练全流程 | Flow Matching + 3D VAE，11B 参数仅用 224 卡、约 20 万美元完成训练；权重、推理、训练代码全开源 |
| [Lumina-T2X](https://github.com/Alpha-VLLM/Lumina-T2X) | Alpha-VLLM | 统一预训练框架 | 基于 Flag-DiT 的 Text-to-Any-Modality 框架，覆盖图像/视频/3D，支持多人格 + 多分辨率打包训练；Lumina-Next 在 2025 年技术报告中探索 μP（maximal update parameterization）规模化，8192 batch 下取得最优配置，超参数可无需 ImageNet 超调直接迁移 |
| [CogVideoX](https://github.com/THUDM/CogVideoX) | 智谱 | 官方微调脚本 | 官方仓库含 full-finetune 与 LoRA 训练脚本；社区另有 [cogvideox-factory](https://github.com/a-r-r-o-w/cogvideox-factory)（TorchAO + DeepSpeed 显存优化） |
| [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) | 腾讯 | 官方训练代码 | 2025-12-05 官方开源 `train.py`，支持分布式训练、FSDP、context parallel、gradient checkpointing；另有官方 I2V LoRA 训练 |
| [Wan 2.1/2.2](https://github.com/Wan-Video/Wan2.1) | 阿里 | 权重 + 推理（训练走社区） | 官方开源权重与推理；训练主要依赖社区工具（ai-toolkit、DiffSynth-Studio 等支持 LoRA/全参训练）；Wan 3.0 于 2026-04 开源 1.3B/14B 权重 |
| [Goku](https://github.com/Saiyan-World/goku) | 字节跳动 | 预训练（论文伴生） | CVPR 2025 Highlight，Rectified Flow Transformer 视频基础模型 |
| [Cosmos / Predict1](https://github.com/NVIDIA/ai-esr-ecosystem) | NVIDIA | 预训练 + 后训练指南 | 官方提供 diffusion 后训练指南、tokenizer 后训练、预训练脚本（NVIDIA docs） |

**未开源官方训练代码、只能靠社区微调的模型**：Stable Diffusion 3.5、FLUX 1/2 等。这类模型权重开源，但训练/全参微调流程不公开，社区通过 Diffusers 官方 examples、[kohya-ss](https://github.com/kohya-ss/sd-scripts)、[ai-toolkit](https://github.com/ostris/ai-toolkit) 等工具进行 LoRA/全参微调。

### 2.2 社区通用训练生态（图像 + 视频）

- **Diffusers（Hugging Face）**：事实标准微调库，官方 `examples` 覆盖 SD/SDXL/Flux/DiT 的 LoRA 与全参微调、DreamBooth、文本反演等，是大多数后训练工作的基础。
- **ai-toolkit（Ostris）**：工业级 LoRA/全参微调工具，广泛支持 FLUX、SDXL、Wan 等，训练配置高度可复现。
- **DiffSynth-Studio（ModelScope）**：异构计算框架，支持 SD/SDXL/Flux/HunyuanVideo/Wan 等模型的 LoRA 与全参训练。
- **kohya-ss/sd-scripts**：社区最流行的 SD/SDXL/FLUX LoRA 训练脚本，生态成熟。
- **FastVideo**：面向视频扩散模型（Sora 类架构）的训练框架，支持分布式训练、数据流与评估，开源于 model-hub（[fastvideo](https://github.com/hpcaitech/FastVideo)）。

### 2.3 中国生态

- **PaddleMIX 2.0（百度）**：多模态训练框架，支持 DiT 3B 级预训练，含张量并行、分组切片并行等分布式策略。
- **MindONE（MindSpore 社区）**：昇腾生态的扩散模型训练/推理库，支持 SD/SDXL/CogVideoX/Open-Sora 等训练。
- **ms-swift（魔搭社区）**：统一微调/对齐工具链，除 LLM 外也覆盖扩散模型的微调与对齐。
- **FastVideo**：由 HPC-AI Tech（Open-Sora 团队）开源，视频模型训练事实标准之一。

---

## 三、后训练与对齐框架

### 3.1 强化学习（RL）

将去噪过程看作 **MDP**（状态 = 当前带噪图像，动作 = 预测去噪结果），即可用策略梯度微调扩散模型，这是扩散模型对齐的核心范式。

| 方法 | 核心思想 | 官方代码 |
| --- | --- | --- |
| **DDPO** | 把去噪过程当作 MDP，用 REINFORCE 类策略梯度微调，是扩散 RL 的奠基工作 | [kvablack/ddpo-pytorch](https://github.com/kvablack/ddpo-pytorch) |
| **DPOK** | DDPO 基础上加入 KL 约束的在线优化，训练更稳定 | [yinbinhan/fine-tuning-of-diffusion-models](https://github.com/yinbinhan/fine-tuning-of-diffusion-models) |
| **ReFL（Reward Feedback Learning）** | 直接通过 reward model 的梯度反向传播微调 SD，NeurIPS 2023 ImageReward 附带；该思想已被 Alibaba ROLL 吸收为 Reward FL 算法 | [THUDM/ImageReward](https://github.com/THUDM/ImageReward) |
| **VADER** | 视频扩散对齐，通过 reward 梯度端到端反向传播，支持 HPS/PickScore/YOLO 等多样 reward，可微调 VideoCrafter、OpenSora、SVD 等 | [mihirp1998/VADER](https://github.com/mihirp1998/VADER) |
| **GRPO 系** | LLM 的 GRPO 被引入扩散：DanceGRPO（视觉生成，2025-05）、GRPO + 扩散噪声放大（2025-06）、d1/diffu-GRPO（dLLM 数学推理，2025-04）、MaskGRPO（多模态离散扩散）等 | 分散在各论文伴生仓库 |

### 3.2 偏好优化（DPO 系）

不训练 reward model，直接用成对偏好数据做直接偏好优化，把 DPO 推广到去噪过程的每一步。

| 方法 | 核心思想 | 官方代码 |
| --- | --- | --- |
| **D3PO** | 把 DPO 推广到 MDP 框架，无需 reward model | [yk7333/D3PO](https://github.com/yk7333/D3PO) |
| **Diffusion-DPO** | 逐去噪步应用 DPO，并按噪声级别加权损失；Salesforce 出品，影响最大 | [SalesforceAIResearch/DiffusionDPO](https://github.com/SalesforceAIResearch/DiffusionDPO) |
| **Diffusion-KTO** | 无需成对数据的 KTO 变体，仅需整体偏好标签 | — |
| **SPO** | Step-by-step Preference Optimization：丢弃 DPO 的“逐点传播”策略，按步骤用 step-aware 偏好模型选 win/lose 对监督，细节可控性更强（CVPR 2025） | [RockeyCoss/SPO](https://github.com/RockeyCoss/SPO) |
| **RDPO** | 重新审视 DPO 风格框架的若干缺陷，提出修订版逐步骤 DPO（ICCV 2025） | — |
| **SPIN-Diffusion** | 自博弈（self-play）式逐轮微调，把教师/学生对抗当作博弈，每轮蒸馏式提升（NeurIPS 2024，UCLA） | 论文伴生仓库 |
| **Diffusion-NPO** | 负样本偏好优化，只利用负偏好数据实现对齐，避免正样本主导（ICLR 2025 / AAAI 2026） | [G-U-N/Diffusion-NPO](https://github.com/G-U-N/Diffusion-NPO) |

偏好对齐的综述参考：
- [*Alignment and Safety of Diffusion Models via RL and Reward Modeling*](https://arxiv.org/abs/2505.17352)（arXiv 2505.17352）
- *Preference alignment on diffusion models*（Computer Science Review, 2026），其分类框架为：RL 系（PPO/GRPO/ReFL/DDPO/DPOK/VADER）+ DPO 系（D3PO/Diffusion-DPO/SPO/KTO）+ 其他。

### 3.3 扩散语言模型（dLLM）后训练新进展

dLLM（如 LLaDA、Dream、SDAR）的后训练正在复用并扩展 LLM RL 栈，是 2025 下半年以来最活跃的方向：

- **d1 / diffu-GRPO（2025-04）**：把 GRPO 应用到 masked diffusion LLM，训练出首个开源 dLLM 推理模型 d1。
- **DiRL（2025-12，[OpenMOSS/DiRL](https://github.com/OpenMOSS/DiRL)，arXiv [2512.22234](https://arxiv.org/abs/2512.22234)）**：dLLM 后训练框架，两阶段 SFT + RL；RL 用自研 **DiPO**（无偏 GRPO 实现）；工程上整合 FlexAttention blockwise 训练与 LMDeploy 推理，8B 模型在数学推理上达到 SOTA。
- **DARE（2026-04，arXiv [2604.04215](https://arxiv.org/abs/2604.04215)）**：dLLM 对齐与强化执行器，统一 worker/dataflow/workflow，在 PPO 风格流水线上支持 SFT 与多种扩散专用 RL 算法，覆盖 LLaDA、Dream、SDAR、LLaDA2.x 等模型家族。
- **Alibaba ROLL（[alibaba/ROLL](https://github.com/alibaba/ROLL)）**：大规模 RL 库，支持 GRPO/PPO/REINFORCE++/TOPR/RAFT 等算法；扩散侧提供 **WanTrainingModule**，支持 Wan2.2 + LoRA 注入 + reward scoring，并内置 Reward FL（ReFL）算法；训练后端可切换 Megatron-LM/DeepSpeed/FSDP2，推理后端接 vLLM/SGLang。

---

## 四、蒸馏框架

蒸馏的目标：用少步（1~4 步）学生模型逼近多步教师模型的输出分布，是推理加速的关键手段，与采样器、服务系统正交。

| 方法 | 核心思想 | 备注 |
| --- | --- | --- |
| **ADD（SD-Turbo / SDXL-Turbo）** | 对抗蒸馏（Adversarial Distillation），用判别器 + score distillation 把教师压到 1~4 步 | Stability 出品，NeurIPS 2024 报告 |
| **LCM / LCM-LoRA** | 潜在一致性蒸馏，直接在 latent 空间训练一致性模型，支持 LoRA 形式 | 曾广泛用于 WebUI，生态成熟 |
| **Consistency Models / CD / CFM** | 让模型沿 ODE 轨迹自洽（任意两个时刻输出一致），实现少步甚至单步采样 | Consistency Distillation 与 Consistency Flow Matching 两条路线 |
| **Rectified Flow 重流** | 通过 reflow 把弯曲轨迹变直，减少所需步数 | 与仓库笔记 [[RectifiedFlow]]、[[FlowMatching]] 直接相关 |
| **DMD / DMD2** | 分布匹配蒸馏：不要求逐点复刻教师轨迹，只匹配学生与教师的输出分布，实现单步生成 | 已有专门笔记 [[DMD]] |
| **SCoT** | Straight Consistent Trajectory：统一一致性模型与矫正流，通过速度损失实现学生轨迹拉直，1~2 步高质量生成（arXiv [2502.16972](https://arxiv.org/abs/2502.16972)，2025） | 无需像 reflow 那样数值求解 ODE |
| **Shortcut Models** | 单次训练直接得到支持少步/单步的模型，无需“预训练 + 蒸馏”两阶段（Frans et al., 2024） | 与一致性模型、CTM、mean flow 统一在 flow map 视角下 |
| **Flow Maps** | *How to build a consistency model: Learning flow maps via self-distillation*（NeurIPS 2025），给出 consistency model / CTM / shortcut / mean flow 的统一数学框架 | [nmboffi/flow-maps](https://github.com/nmboffi/flow-maps) |

蒸馏综述参考：[*A survey on pre-trained diffusion model distillations*](https://arxiv.org/abs/2502.08364)（arXiv 2502.08364）。

另外，蒸馏也被用于 dLLM 后端压缩（例如 mochi-1 的 distilled transformer 42 层版本），说明蒸馏思想已跨模态复用。

---

## 五、框架汇总表

| 框架 | 类型 | 官方仓库 | 一句话说明 |
| --- | --- | --- | --- |
| OpenDiT | 预训练 | [NUS-HPC-AI-Lab/OpenDiT](https://github.com/NUS-HPC-AI-Lab/OpenDiT) | Colossal-AI 上的高性能 DiT 训练/推理系统 |
| Open-Sora 2.0 | 预训练 | [hpcai-tech/Open-Sora](https://github.com/hpcai-tech/Open-Sora) | 11B 视频模型，224 卡 20 万美元级训练，全流程开源 |
| Lumina-T2X / Lumina-Next | 预训练 | [Alpha-VLLM/Lumina-T2X](https://github.com/Alpha-VLLM/Lumina-T2X) | Text-to-Any-Modality 统一框架（Flag-DiT） |
| CogVideoX | 预训练/微调 | [THUDM/CogVideoX](https://github.com/THUDM/CogVideoX) | 官方 full-finetune/LoRA 脚本 |
| HunyuanVideo | 预训练/微调 | [Tencent/HunyuanVideo](https://github.com/Tencent/HunyuanVideo) | 2025-12 开源训练代码（FSDP/context parallel） |
| Wan 2.x / 3.0 | 预训练（训练靠社区） | [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) | 权重开源；ai-toolkit/DiffSynth 支持训练 |
| DDPO | RL 后训练 | [kvablack/ddpo-pytorch](https://github.com/kvablack/ddpo-pytorch) | 扩散 RL 奠基工作 |
| Diffusion-DPO | 偏好对齐 | [SalesforceAIResearch/DiffusionDPO](https://github.com/SalesforceAIResearch/DiffusionDPO) | DPO 推广到去噪过程 |
| SPO | 偏好对齐 | [RockeyCoss/SPO](https://github.com/RockeyCoss/SPO) | 逐步偏好优化（CVPR 2025） |
| Diffusion-NPO | 偏好对齐 | [G-U-N/Diffusion-NPO](https://github.com/G-U-N/Diffusion-NPO) | 只用负偏好对齐（ICLR 2025） |
| ROLL | RL 通用库（含扩散） | [alibaba/ROLL](https://github.com/alibaba/ROLL) | GRPO/PPO/RAFT + WanTrainingModule |
| DiRL | dLLM 后训练 | [OpenMOSS/DiRL](https://github.com/OpenMOSS/DiRL) | dLLM SFT + RL（DiPO 无偏 GRPO） |
| DARE | dLLM 后训练 | arXiv [2604.04215](https://arxiv.org/abs/2604.04215) | dLLM 对齐与强化执行器 |
| LCM | 蒸馏 | [luosiallen/latent-consistency-model](https://github.com/luosiallen/latent-consistency-model) | 潜在一致性蒸馏 |
| DMD2 | 蒸馏 | [tianweiy/DMD2](https://github.com/tianweiy/DMD2) | 分布匹配单步蒸馏（见 [[DMD]]） |
| Flow Maps | 蒸馏 | [nmboffi/flow-maps](https://github.com/nmboffi/flow-maps) | consistency model 统一框架（NeurIPS 2025） |
| ai-toolkit / Diffusers / kohya | 社区微调 | [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit) 等 | 通用 LoRA/全参微调生态 |
| PaddleMIX 2.0 | 预训练（国产） | [PaddlePaddle/PaddleMIX](https://github.com/PaddlePaddle/PaddleMIX) | 百度多模态训练框架，DiT 3B 预训练 |
| MindONE | 预训练/推理（国产） | [mindspore-lab/mindone](https://github.com/mindspore-lab/mindone) | 昇腾生态扩散训练库 |
| ms-swift | 微调/对齐（国产） | [modelscope/ms-swift](https://github.com/modelscope/ms-swift) | 统一微调对齐工具链 |

---

## 六、发展趋势与结论

1. **预训练框架“随模型开源”成为常态**：Open-Sora、HunyuanVideo、CogVideoX 等均把分布式训练代码随权重一起发布，训练能力本身成为模型影响力的一部分；而 SD3.5/FLUX 等闭源训练的代表性模型则把微调生态留给了社区工具。
2. **后训练正在从“微调”转向“对齐 + RL”**：DDPO/DPOK 奠定了扩散 RL 范式，DPO 系方法解决了 reward model 依赖问题，而 2025 年 SPO、RDPO、Diffusion-NPO 等开始系统性修正早期方法的缺陷——这个演进与 LLM 的 RLHF → DPO → 无偏 GRPO 路径高度同构。
3. **LLM 与扩散的工程栈融合**：dLLM 后训练（DiRL、DARE）直接复用 PPO/GRPO 流水线，Alibaba ROLL 一套库同时服务 LLM 与视频扩散；未来扩散训练框架大概率并入统一的多模态 RL 框架。
4. **蒸馏是推理成本的胜负手**：ADD、LCM、DMD、SCoT、Flow Maps 等不断压低采样步数（4 步 → 1 步），且对服务系统的吞吐与延迟有直接放大效应；蒸馏与步级调度（见仓库 `Notes/Diffusion/ServingSystem` 系列）是推理优化的两条正交主线。
5. **国产生态补位明显**：PaddleMIX、MindONE、ms-swift、FastVideo 覆盖了从预训练到微调/对齐的完整链路，与 Hugging Face Diffusers 生态形成互补。

---

## 相关笔记

- [[DMD]] — 分布匹配蒸馏详解
- [[RectifiedFlow]]、[[FlowMatching]] — 蒸馏与少步采样的理论基础
- [[wan]]、[[cogvideox]] — 具体模型笔记
- `Notes/Diffusion/ServingSystem/业界现状调研报告.md` — 训练成果的推理服务侧落地
