# World Model 文献清单

---

## 📖 推荐阅读顺序

> 按依赖关系和知识递进组织，从理论基础到具体系统

### 第一阶段：理论基础（必读）

| 顺序 | 论文 | 笔记 | 为什么先读 |
|------|------|------|-----------|
| 1 | **World Models Survey** | [笔记](00-综述-World-Models-Survey.md) | 建立全局认知框架，理解四维分类 |
| 2 | **LeCun's Path** | [笔记](01-JEPA-LeCun-Path.md) | JEPA 理论基础，理解表示空间预测 vs 像素空间预测 |
| 3 | **I-JEPA** | [笔记](02-I-JEPA.md) | JEPA 的第一个实例化，理解核心机制 |

### 第二阶段：JEPA 进阶

| 顺序 | 论文 | 笔记 | 为什么这个顺序 |
|------|------|------|--------------|
| 4 | **V-JEPA 2.1** | [笔记](03-V-JEPA-2.1.md) | 从图像到视频 JEPA，密集特征学习 |
| 5 | **LeWorldModel** | [笔记](05-LeWorldModel.md) | JEPA 作为世界模型的端到端训练 |

### 第三阶段：MBRL 世界模型

| 顺序 | 论文 | 笔记 | 为什么这个顺序 |
|------|------|------|--------------|
| 6 | **DreamerV3** | [笔记](04-DreamerV3.md) | 经典 RSSM 世界模型，通用 RL 算法 |

### 第四阶段：扩散世界模型平台

| 顺序 | 论文 | 笔记 | 为什么这个顺序 |
|------|------|------|--------------|
| 7 | **Cosmos (v1)** | [笔记](07-Cosmos.md) | DiT 世界基础模型平台，AdaLN-LoRA 等优化 |
| 8 | **GWM** | [笔记](08-GWM.md) | 3D 高斯 + DiT 的机器人世界模型 |

### 第五阶段：推理引擎与蒸馏

| 顺序 | 论文 | 笔记 | 为什么这个顺序 |
|------|------|------|--------------|
| 9 | **Inferix** | [笔记](06-Inferix.md) | Block-Diffusion 推理引擎，KV Cache 管理 |
| 10 | **minWM** | [笔记](10-minWM.md) | 全栈蒸馏管道，双向→少步 AR |
| 11 | **DreamX-World 1.0** | [笔记](09-DreamX-World.md) | E-PRoPE、记忆、事件控制 |

### 第六阶段：交互式世界模型前沿

| 顺序 | 论文 | 笔记 | 为什么这个顺序 |
|------|------|------|--------------|
| 12 | **Interactive Video WM Survey** | [笔记](11-Interactive-Video-WM-Survey.md) | 交互式世界模型全景 |
| 13 | **MoWorld** | 待补充 | NPU 部署，50 FPS |
| 14 | **BiWM** | 待补充 | 双向自回归范式 |
| 15 | **LingBot-World** | 待补充 | 分钟级长程一致性 |
| 16 | **Multiplayer WM** | 待补充 | 多玩家世界模型 |

### 第七阶段：机器人与具身

| 顺序 | 论文 | 笔记 | 为什么这个顺序 |
|------|------|------|--------------|
| 17 | **RoboDream** | 待补充 | 组合式世界模型 |
| 18 | **Qwen-RobotWorld** | 待补充 | 语言条件控制 |

### 第八阶段：世界-语言-动作

| 顺序 | 论文 | 笔记 | 为什么这个顺序 |
|------|------|------|--------------|
| 19 | **WLA Model** | 待补充 | 统一世界+语言+动作 |
| 20 | **World Model Self-Distillation** | 待补充 | 自蒸馏 + RL |
| 21 | **Cosmos 3** | 待补充 | 全模态统一架构 |

### 第九阶段：专项应用

| 顺序 | 论文 | 笔记 | 为什么这个顺序 |
|------|------|------|--------------|
| 22 | **NVIDIA OmniDreams** | 待补充 | 自动驾驶仿真 |
| 23 | **World Action Models Survey** | 待补充 | 具身 WAMs 综述 |
| 24 | **Learning to Model the World** | 待补充 | 形式化数学定义 |

---

## 📝 笔记索引

| 文件 | 论文 | 核心价值 |
|------|------|---------|
| [00-综述](00-综述-World-Models-Survey.md) | World Models Survey | 四维分类框架，领域全景 |
| [01-JEPA](01-JEPA-LeCun-Path.md) | LeCun's Path | 六模块认知架构，JEPA 理论 |
| [02-I-JEPA](02-I-JEPA.md) | I-JEPA | 无增强自监督，表示空间预测 |
| [03-V-JEPA-2.1](03-V-JEPA-2.1.md) | V-JEPA 2.1 | 密集特征，深度自监督 |
| [04-DreamerV3](04-DreamerV3.md) | DreamerV3 | Symlog，通用 RL，Minecraft |
| [05-LeWorldModel](05-LeWorldModel.md) | LeWorldModel | 端到端 JEPA，SIGReg |
| [06-Inferix](06-Inferix.md) | Inferix | 块扩散推理引擎 |
| [07-Cosmos](07-Cosmos.md) | Cosmos v1 | 世界基础模型平台 |
| [08-GWM](08-GWM.md) | GWM | 3D 高斯世界模型 |
| [09-DreamX-World](09-DreamX-World.md) | DreamX-World 1.0 | E-PRoPE，记忆，事件控制 |
| [10-minWM](10-minWM.md) | minWM | 全栈蒸馏，237× 加速 |
| [11-Interactive-Video-WM-Survey](11-Interactive-Video-WM-Survey.md) | Interactive Video WM Survey | 交互式世界模型全景 |

---

## 论文清单

### 推理引擎 (Inference Engine)

| 论文 | 年份 | 关键特性 | 笔记 | 链接 | PDF |
|------|------|----------|------|------|-----|
| **Inferix** | 2026 | Block-Diffusion 半自回归解码，KV Cache 管理，交互式视频流 | [笔记](06-Inferix.md) | [arXiv](http://arxiv.org/abs/2511.20714) | [PDF](http://arxiv.org/pdf/2511.20714) |

### 交互式世界模型 (Interactive World Models)

| 论文 | 年份 | 关键特性 | 笔记 | 链接 | PDF |
|------|------|----------|------|------|-----|
| **MoWorld** | 2026 | 50 FPS 实时交互，NPU 部署，无需高端 GPU | 待补充 | [arXiv](https://arxiv.org/abs/2607.06216) | [PDF](https://arxiv.org/pdf/2607.06216) |
| **BiWM** | 2026 | 双向自回归范式，DMD 蒸馏，支持 Wan2.1-1.3B 到 LTX-2.3-22B | 待补充 | [arXiv](https://arxiv.org/abs/2606.10135) | [PDF](https://arxiv.org/pdf/2606.10135) |
| **minWM** | 2026 | 全栈开源框架，相机控制，因果一致性蒸馏 | [笔记](10-minWM.md) | [arXiv](https://arxiv.org/abs/2605.30263) | [PDF](https://arxiv.org/pdf/2605.30263) |
| **DreamX-World 1.0** | 2026 | 通用交互模型，E-PRoPE 相机控制，8×RTX 5090 达 16 FPS | [笔记](09-DreamX-World.md) | [arXiv](https://arxiv.org/abs/2606.16993) | [PDF](https://arxiv.org/pdf/2606.16993) |
| **LingBot-World** | 2026 | 开源世界模拟器，分钟级长程一致性，<1s 延迟 | 待补充 | [arXiv](http://arxiv.org/abs/2601.20540) | [PDF](http://arxiv.org/pdf/2601.20540) |
| **Multiplayer Interactive WM** | 2026 | 首个多玩家世界模型，Rocket League，5B 参数，20 FPS | 待补充 | [arXiv](https://arxiv.org/abs/2607.05352) | [PDF](https://arxiv.org/pdf/2607.05352) |

### 机器人操控 (Robotic Manipulation)

| 论文 | 年份 | 关键特性 | 笔记 | 链接 | PDF |
|------|------|----------|------|------|-----|
| **GWM** | 2025 | 高斯世界模型，3D-VAE + DiT，ICCV 2025 | [笔记](08-GWM.md) | [arXiv](https://arxiv.org/abs/2508.17600) | [PDF](https://arxiv.org/pdf/2508.17600) |
| **RoboDream** | 2026 | 组合式世界模型，可扩展机器人数据合成 | 待补充 | [arXiv](https://arxiv.org/abs/2606.02577) | [PDF](https://arxiv.org/pdf/2606.02577) |
| **Qwen-RobotWorld** | 2026 | 语言条件视频世界模型，8.6M 视频语料，20+ 具身类型 | 待补充 | [arXiv](https://arxiv.org/abs/2606.17030) | [PDF](https://arxiv.org/pdf/2606.17030) |

### 自动驾驶 (Autonomous Driving)

| 论文 | 年份 | 关键特性 | 笔记 | 链接 | PDF |
|------|------|----------|------|------|-----|
| **NVIDIA OmniDreams** | 2026 | 闭环自动驾驶仿真，实时生成世界模型 | 待补充 | [arXiv](https://arxiv.org/abs/2606.03159) | [PDF](https://arxiv.org/pdf/2606.03159) |

### 世界 - 语言 - 动作模型 (World-Language-Action)

| 论文 | 年份 | 关键特性 | 笔记 | 链接 | PDF |
|------|------|----------|------|------|-----|
| **WLA Model** | 2026 | 统一世界建模 + 语言推理 + 动作合成，RTX 5090 40ms 推理 | 待补充 | [arXiv](https://arxiv.org/abs/2606.05979) | [PDF](https://arxiv.org/pdf/2606.05979) |
| **World Model Self-Distillation** | 2026 | 自蒸馏 +RL，VLM 生成任务，视频扩散模型蒸馏 | 待补充 | [arXiv](https://arxiv.org/abs/2606.12072) | [PDF](https://arxiv.org/pdf/2606.12072) |
| **DreamerV3** | 2023 | 通用 RL 算法，环境模型+想象未来，Minecraft 从零收集钻石 | [笔记](04-DreamerV3.md) | [arXiv](https://arxiv.org/abs/2301.04104) | [PDF](https://arxiv.org/pdf/2301.04104) |

### JEPA 架构系列 (Joint-Embedding Predictive Architecture)

| 论文 | 年份 | 关键特性 | 笔记 | 链接 | PDF |
|------|------|----------|------|------|-----|
| **LeCun's Path** | 2022 | JEPA 奠基论文，自主机器智能路线图 | [笔记](01-JEPA-LeCun-Path.md) | [OpenReview](https://openreview.net/forum?id=BZ5a1r-kVsf) | [PDF](https://openreview.net/pdf?id=BZ5a1r-kVsf) |
| **I-JEPA** | 2023 | 图像自监督学习，无数据增强，ViT-Huge/14 | [笔记](02-I-JEPA.md) | [arXiv](http://arxiv.org/abs/2301.08243) | [PDF](http://arxiv.org/pdf/2301.08243) |
| **V-JEPA 2.1** | 2026 | 视频密集特征学习，深度自监督，多模态分词器 | [笔记](03-V-JEPA-2.1.md) | [arXiv](http://arxiv.org/abs/2603.14482) | [PDF](http://arxiv.org/pdf/2603.14482) |
| **LeWorldModel** | 2026 | 端到端 JEPA，~15M 参数，单 GPU 训练，规划速度 48x 提升 | [笔记](05-LeWorldModel.md) | [arXiv](http://arxiv.org/abs/2603.19312) | [PDF](http://arxiv.org/pdf/2603.19312) |

### NVIDIA Cosmos 系列

| 论文 | 年份 | 关键特性 | 笔记 | 链接 | PDF |
|------|------|----------|------|------|-----|
| **Cosmos (v1)** | 2025 | 世界基础模型平台，开源开放权重 | [笔记](07-Cosmos.md) | [arXiv](https://arxiv.org/abs/2501.03575v3) | [PDF](https://arxiv.org/pdf/2501.03575v3) |
| **Cosmos 3** | 2026 | 全模态世界模型，统一 MoT 架构，语言 + 图像 + 视频 + 音频 + 动作 | 待补充 | [arXiv](https://arxiv.org/abs/2606.02800) | [PDF](https://arxiv.org/pdf/2606.02800) |

### 综述论文 (Surveys)

| 论文 | 年份 | 涵盖内容 | 笔记 | 链接 | PDF |
|------|------|----------|------|------|-----|
| **World Models: A Comprehensive Survey** | 2026 | 四维分类：架构、方法论、推理策略、应用领域 | [笔记](00-综述-World-Models-Survey.md) | [arXiv](https://arxiv.org/abs/2606.00133) | [PDF](https://arxiv.org/pdf/2606.00133) |
| **World Action Models Survey** | 2026 | 具身预测 - 动作模型 (WAMs) | 待补充 | [arXiv](https://arxiv.org/abs/2606.20781) | [PDF](https://arxiv.org/pdf/2606.20781) |
| **Towards Interactive Video World Modeling** | 2026 | 前沿、挑战、基准、未来趋势 | [笔记](11-Interactive-Video-WM-Survey.md) | [arXiv](http://arxiv.org/abs/2606.01164) | [PDF](http://arxiv.org/pdf/2606.01164) |
| **Learning to Model the World** | 2025 | 四分支分类：观测级生成、潜空间、RL-based、对象中心；形式化数学定义 | 待补充 | [TechRxiv](https://doi.org/10.36227/techrxiv.177274570.09578608/v1) | [PDF](https://www.techrxiv.org/doi/pdf/10.36227/techrxiv.177274570.09578608/v1?download=true&redirectToLatest=false) |
