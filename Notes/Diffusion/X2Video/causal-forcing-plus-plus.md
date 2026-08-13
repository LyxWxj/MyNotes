---
type: Note
related_to: "[[X2Video]]"
status: Active
url: https://arxiv.org/abs/2605.15141
---

# Causal Forcing++：面向实时交互视频的可扩展少步自回归扩散蒸馏

> [!info] 论文信息
> - **标题**：Causal Forcing++: Scalable Few-Step Autoregressive Diffusion Distillation for Real-Time Interactive Video Generation
> - **作者**：Min Zhao、Hongzhou Zhu、Kaiwen Zheng、Zihan Zhou、Bokai Yan、Xinyuan Li、Xiao Yang、Chongxuan Li、Jun Zhu
> - **机构**：清华大学、ShengShu、中国人民大学
> - **日期**：2026-06-01（arXiv v3）
> - **arXiv**：[2605.15141](https://arxiv.org/abs/2605.15141)
> - **论文 HTML**：https://arxiv.org/html/2605.15141v3
> - **代码**：https://github.com/thu-ml/Causal-Forcing
> - **世界模型代码**：https://github.com/shengshu-ai/minWM

## 一句话总结

**Causal Forcing++（CF++）将 Causal Forcing 的第 2 阶段从「离线全轨迹的因果 ODE 蒸馏」替换为「在线、相邻时间步的因果一致性蒸馏（causal consistency distillation, causal CD）」**。二者学习相同的 AR 条件流映射，但 causal CD 无需预计算和保存完整 PF-ODE 轨迹；因此能在逐帧、1–2 步这一更激进的实时设置中获得更强初始化，并把第 2 阶段成本降约 4 倍。

## 问题：逐帧、少步 AR 让初始化成为瓶颈

双向视频扩散模型一次联合生成整段视频，质量高但首帧等待长，且用户无法根据已生成画面实时改变后续条件。自回归扩散（AR diffusion）则在帧或块之间因果展开、在当前单元内做扩散采样；配合 temporal KV cache，可形成流式、可交互的视频生成。

此前的 CausVid、Self Forcing 与 Causal Forcing 通常工作在「每个 AR chunk 含约 3 个 latent frame、每个 chunk 采样 4 步」的区间。本文将目标推进到：

- **帧级 AR**：每次只生成 1 个 latent frame，反馈粒度更细；
- **1–2 步采样**：进一步降低每帧等待；
- **自回归 self-rollout**：生成前缀来自学生自身，因此误差会持续累积。

在这一设置中，少步学生的初始能力决定后续 asymmetric DMD 能否稳定优化。论文归纳了已有三种初始化的互补缺陷：

| 初始化方案 | 代表工作 | 问题 |
|---|---|---|
| 双向教师 ODE 蒸馏到 AR 学生 | CausVid、Self Forcing | 教师依赖未来帧，AR 学生不可见，目标与因果生成架构不一致 |
| 直接使用多步 AR diffusion | LiveAvatar、WorldPlay | 没有少步能力；缩小 AR 单元、减少采样步数后，单步误差在 rollout 中放大 |
| AR 教师的因果 ODE 蒸馏 | Causal Forcing | 学习目标正确，却要为每条样本预生成、存储完整 PF-ODE 轨迹，难以扩展 |

### 为什么双向教师的 ODE 初始化不成立

对第 $i$ 帧而言，双向教师的 PF-ODE 轨迹会依赖未来噪声帧 $x_t^{>i}$；但 AR 学生仅可输入当前噪声帧与历史 $\left(x_t^i, x_t^{<i}\right)$。因此，同一份学生输入可能因不同未来上下文对应多个干净目标，违反 Causal Forcing 提出的**帧级注入性**。以 MSE 回归时，最优学生会趋向条件期望：

$$
G_\theta^*(x_t^i, x_t^{<i}, t)
= \mathbb{E}[x_0^i \mid x_t^i, x_t^{<i}, t]。
$$

条件均值会平均掉多个可能未来中的高频细节，形成模糊且与真实 AR 流映射不对齐的初始化。逐帧、少步 rollout 会把这种误差进一步放大。

> [!important] 关键判断
> Asymmetric DMD 是**初始化后的分布细化器**，而不是能从任意多步 AR 模型中可靠学出少步生成器的完整训练器。缺少强少步初始化时，后续 DMD 无法承担巨大的纠偏负担。

## 方法：Causal Forcing++ 三阶段管线

CF++ 保留 Causal Forcing 的第 1、3 阶段，只替换第 2 阶段。

```text
Stage 1: Teacher forcing AR diffusion
    双向基础视频模型 -> 多步 AR 教师

Stage 2: Causal consistency distillation（本文）
    AR 教师的一次在线 ODE 局部步进 -> 少步 AR 学生初始化

Stage 3: Asymmetric DMD + student self-rollout
    双向模型/critic 负责分布匹配；AR 学生按自身历史 rollout
```

### Stage 1：训练多步 AR 教师

从 Wan2.1-1.3B 出发，以 teacher forcing 和因果注意力将双向视频扩散模型微调为多步 AR diffusion teacher。训练时第 $i$ 帧以真实干净前缀 $x_{\mathrm{gt}}^{<i}$ 为条件，因而学到 AR 条件分布；这也是第 2 阶段能够保持因果对齐的前提。

### Stage 2：Causal CD 初始化

令 $G_\theta$ 表示学生的去噪/一致性函数，$\phi$ 是冻结的 AR 教师。采用 flow matching 参数化时：

$$
G_\theta(x_t^i, x_{\mathrm{gt}}^{<i}, t)
= x_t^i - t\,v_\theta(x_t^i, x_{\mathrm{gt}}^{<i}, t)。
$$

训练时，先从真实第 $i$ 帧构造 $x_t^i$，再在相同真实前缀条件下，由 AR 教师执行**一次** ODE 步得到 $\hat{x}_{t-\Delta t}^i$。学生当前输出与其 EMA 目标网络在相邻时间步的输出对齐：

$$
\min_\theta\;\mathbb{E}\left[
w(t)\,d\left(
G_\theta(x_t^i,x_{\mathrm{gt}}^{<i},t),
G_{\theta^-}(\hat{x}_{t-\Delta t}^i,x_{\mathrm{gt}}^{<i},t-\Delta t)
\right)
\right]。
$$

其中 $\theta^-$ 是 EMA 且停止梯度，$w(t)$ 为时间步权重，$d$ 是预设距离函数（本文使用平方范数）。

#### 与因果 ODE 蒸馏的关系

两种方法学习同一个对象：AR 教师的条件流映射

$$
f_\phi:(x_t^i, x_{\mathrm{gt}}^{<i},t) \mapsto x_0^i。
$$

- **因果 ODE 蒸馏**：从轨迹上的 $t$ 直接回归到 $0$，需离线生成完整轨迹和配对数据。
- **因果 CD**：让相邻时刻的预测一致；理论上，最优解与 $f_\phi$ 的误差由 ODE 求解器数值误差 $\mathcal{O}((\Delta t)^p)$ 控制。

因此，CF++ 的关键并非换了一个启发式 loss，而是以局部一致性监督近似同一个因果流映射。相邻时间步的目标间隔仅为 $\Delta t$，相较于直接从高噪声状态跳到终点的回归更容易优化。

> [!tip] 工程收益
> 在论文的 80K 视频规模下，因果 ODE 初始化约需 11,600 A800 GPU·hours 与 1,900 GiB 额外轨迹存储；causal CD 约需 2,900 A800 GPU·hours，且不需要额外轨迹存储。教师、数据或 AR 单元改变时，也无需重新生成离线轨迹集。

### Stage 3：Asymmetric DMD 与 self-rollout

用第 2 阶段的少步 AR 学生初始化后，沿用 Self Forcing/Causal Forcing 的 asymmetric DMD：学生保持因果 AR 架构并按自身生成历史 rollout，而用于分布匹配的真实 score 模型/critic 保持双向。这样第 2 阶段解决「正确的 AR 流映射」，第 3 阶段利用较强双向模型的分布知识细化感知质量。

## 为什么不以 Causal DMD 替代 Causal CD

论文也考察了全因果的 score distillation（causal DMD）作为第 2 阶段。它前几帧往往更锐利，却在更长 AR rollout 中出现剧烈漂移和相机跳变，最终质量低于 causal CD。

作者的解释是：DMD 更偏 **mode-seeking**，对历史前缀的微小偏移更敏感；一旦学生历史偏离训练时的真实前缀，误差会迅速放大，造成更严重的 exposure bias。Causal CD 的 mode-covering 特性使其在存在历史偏移时保留更多高质量概率质量，因此更适合作为 AR 少步初始化。

## 实验设置

- 基座模型：Wan2.1-1.3B；Stage 3 的真实 score 模型为 Wan2.1-14B。
- 生成规格：480 × 832、81 帧、逐 latent frame AR。
- 数据：Stage 1/2 使用含 OpenVid 样本的 80K 视频集；Stage 3 使用 VidProM。
- 训练：三个阶段分别训练 20K、5K、1K step，batch size 为 64。
- Causal CD：48 个离散时间步、Euler 求解器、平方范数。
- 评测：VBench、VisionReward、Dynamic Degree、Instruction Following；吞吐和首帧延迟均在单张 A800 测量，且不含 VAE 时间。

低于 4 步时，Stage 3 沿用 ASD 技巧：首个 latent frame 始终用 4 步生成，后续 20 个 latent frame 使用目标的 1 或 2 步。因此表中的 1、2、4 步首帧延迟相同。

## 主要结果

### 与既有方法比较

| 模型 | FPS ↑ | 首帧延迟（s）↓ | VBench Total ↑ | Quality ↑ | Dynamic ↑ | VisionReward ↑ | Instruction Following ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| CausVid | 10.4 | 0.60 | 81.33 | 83.98 | 62 | 5.741 | 12 |
| Self Forcing | 10.4 | 0.60 | 83.74 | 84.48 | 57 | 5.820 | 48 |
| Causal Forcing | 10.4 | 0.60 | 84.04 | 84.59 | 68 | 6.326 | 56 |
| **CF++（1-step）** | **20.7** | **0.27** | 83.35 | 84.50 | 66 | 5.412 | 38 |
| **CF++（2-step）** | **14.1** | **0.27** | **84.14** | 84.89 | 64 | **6.661** | 51 |
| **CF++（4-step）** | 8.69 | **0.27** | 84.10 | **84.94** | **71** | **6.798** | 47 |

论文重点主张的逐帧 2 步模型，相对 4-step chunk-wise Causal Forcing：VBench Total +0.10、VBench Quality +0.30、VisionReward +0.335，同时首帧延迟从 0.60 s 降到 0.27 s（约 55%，论文概述为 50%）。

### 初始化消融：为什么选择 Causal CD

| 2-step 初始化 | VBench Total ↑ | VisionReward ↑ | Stage 2 时间（A800 GPU·h）↓ | 额外存储（GiB）↓ |
|---|---:|---:|---:|---:|
| Self Forcing ODE | 79.44 | 2.826 | 5,000 | 1,500 |
| 多步 AR diffusion | 82.43 | 3.645 | - | 0 |
| 因果 ODE | 83.77 | 6.224 | 11,600 | 1,900 |
| Causal DMD | 83.73 | 6.108 | 2,900 | 0 |
| **Causal CD（CF++）** | **84.14** | **6.661** | **2,900** | **0** |

因果 CD 在 1、2、4 步下均匹配或超过因果 ODE。尤其 2 步时，其 VBench Total 与 VisionReward 均为消融中最高，说明局部一致性监督同时改善了计算成本与初始化质量。

## 向动作条件世界模型扩展

论文以 Genie3 风格的相机位姿作为 action，展示 CF++ 可蒸馏出交互式世界模型：

1. 用 WorldPlay 构造带相机位姿标注的训练集；
2. 通过 PRoPE 向 Wan2.1-1.3B 注入位姿信息，获得双向的相机条件视频扩散模型；
3. 使用 CF++ 将其蒸馏为 chunk-wise 4-step、可交互的 AR 世界模型。

这部分是定性演示。论文明确将「将动作条件版本进一步压缩到逐帧 2 步」留作未来工作，不能将其视为已经验证的完全实时世界模型结果。

## 实现与研究启示

- **少步初始化是独立问题**：先确认学生是否获得正确的因果流映射，再做 DMD 分布细化；不能期待后处理式 DMD 修复根本性的 AR 架构错配。
- **局部教师步进替代离线数据加工**：如果蒸馏目标是 ODE 流映射，可优先评估相邻时间步的一致性约束，减少离线轨迹数据和版本耦合。
- **AR 训练须把 rollout 稳定性作为第一等指标**：首帧锐利不代表长视频好。应同时观察后期帧、相机漂移与历史偏移下的退化。
- **延迟口径需审慎比较**：本文的延迟在 A800 上测量且不含 VAE；与不同硬件或端到端管线的报告值不可直接横比。
- **当前低延迟技巧有边界**：1/2-step 配置仍让首帧使用 4 步，意味着「每个后续帧的少步」与「端到端首帧完全一两步」并非同一主张。

## 与前作的关系

| 工作                   | 修复的问题                               | 局限                                | CF++ 的推进                            |
| -------------------- | ----------------------------------- | --------------------------------- | ----------------------------------- |
| CausVid              | 建立 ODE init + asymmetric DMD 框架     | 双向教师 ODE 目标与 AR 学生不对齐；训练/推理上下文不一致 | 保留蒸馏框架，但用因果初始化和 self-rollout 解决关键错配 |
| Self Forcing         | Stage 3 采用学生 self-rollout，缓解训练-推理差距 | Stage 2 仍从双向教师 ODE 初始化            | 不再使用违反帧级注入性的双向 ODE 回归               |
| Causal Forcing       | 用 AR 教师进行因果 ODE，修复初始化目标             | 预生成完整轨迹，代价和存储高                    | 用 online causal CD 学同一流映射，成本约降 4 倍  |
| **Causal Forcing++** | 正确、少步、可扩展的 AR 初始化                   | 动作条件版本仍只展示 chunk-wise 4-step      | 在逐帧 1–2 步下给出系统实证                    |

## 结论

CF++ 的主要贡献是把「因果 ODE 蒸馏正确但昂贵」转化为「因果一致性蒸馏正确且可扩展」：它用 AR 教师的一次在线局部步进监督，学习与全轨迹 ODE 蒸馏相同的 AR 条件流映射。在 Wan2.1-1.3B 的逐帧 2-step 设定中，方法同时实现更低首帧延迟、更高综合质量，以及显著更低的第 2 阶段训练与存储成本。

> [!quote] 引用
> Zhao, M., Zhu, H., Zheng, K., Zhou, Z., Yan, B., Li, X., Yang, X., Li, C., & Zhu, J. (2026). *Causal Forcing++: Scalable Few-Step Autoregressive Diffusion Distillation for Real-Time Interactive Video Generation*. arXiv:2605.15141.
