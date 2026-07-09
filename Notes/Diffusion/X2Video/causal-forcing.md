---
type: Note
related_to: "[[X2Video]]"
status: Active
url: http://arxiv.org/abs/2602.02214
---

# Causal Forcing: Autoregressive Diffusion Distillation Done Right for High-Quality Real-Time Interactive Video Generation

> [!info] 论文信息
> - **作者**: Hongzhou Zhu, Min Zhao, Guande He, Hang Su, Chongxuan Li, Jun Zhu
> - **机构**: 清华大学、中国人民大学、德克萨斯大学奥斯汀分校
> - **日期**: 2026-06-01
> - **会议**: ICML 2026
> - **arXiv**: [2602.02214](http://arxiv.org/abs/2602.02214)
> - **项目主页**: https://thu-ml.github.io/CausalForcing.github.io/
> - **代码**: https://github.com/thu-ml/Causal-Forcing

## 摘要

为了实现实时交互式视频生成，当前方法将预训练的双向视频扩散模型蒸馏为少步自回归（AR）模型，当全注意力被因果注意力替代时面临架构差距。然而，现有方法未能从理论上弥合这一差距。它们通过ODE蒸馏初始化AR学生，这需要帧级注入性，即在AR教师的PF-ODE下，每个噪声帧必须映射到唯一的干净帧。从双向教师蒸馏AR学生违反了这一条件，阻止了教师流映射的恢复，反而导致条件期望解，降低了性能。

> [!tip] 核心创新
> - **帧级注入性**：识别并形式化ODE蒸馏的关键要求，即每个噪声帧必须映射到唯一的干净帧。
> - **架构差距分析**：揭示现有方法（如Self Forcing）在理论上存在的缺陷。
> - **因果ODE蒸馏**：使用自回归教师进行ODE初始化，弥合架构差距。
> - **三阶段方法**：教师强制AR扩散训练、因果ODE蒸馏、非对称DMD。

## 主要贡献

1. **理论分析**：识别帧级注入性作为ODE蒸馏的必要原则，揭示现有方法的理论缺陷。
2. **提出Causal Forcing**：一种三阶段方法，通过自回归教师进行因果ODE蒸馏，弥合架构差距。
3. **优越性能**：在所有指标上超越所有基线，显著超越最先进方法Self Forcing。
4. **实现实时交互**：实现高质量、低延迟的实时交互式视频生成。

> [!important] 性能提升
> - **Dynamic Degree**：超越Self Forcing 19.3%。
> - **VisionReward**：超越Self Forcing 8.7%。
> - **Instruction Following**：超越Self Forcing 16.7%。
> - **推理延迟**：保持与Self Forcing相同的推理延迟。

## 问题分析：现有方法的局限

### 架构差距

> [!warning] 架构差距的挑战
> - **采样步骤差距**：减少多步采样到少步采样，这是共享的挑战。
> - **架构差距**：将具有全注意力的双向模型转换为仅基于过去上下文的因果注意力架构，这是更根本的挑战。
> - **现有方法**：Self Forcing等方法未能理论上解决架构差距。

### 帧级注入性

> [!important] 帧级注入性原则
> - **定义**：对于映射ϕ_AR: (x_i^t, t) → x_i^0，帧级注入性成立的条件是：对于任意t ∈ (0,1]，对于任意两个噪声视频{x_j^t}_{j=1}^N, {y_j^t}_{j=1}^N，如果x_i^t = y_i^t，则ϕ_AR(x_i^t, t) = ϕ_AR(y_i^t, t)。
> - **直觉**：双向扩散模型使用所有帧去噪第i帧，因此即使x_i^t固定，不同的x_{>i}^t也可能产生不同的x_i^0。Self Forcing中，AR学生在没有x_{>i}^t的情况下被监督，导致信息丢失，违反帧级注入性。
> - **后果**：违反帧级注入性导致回归学生无法恢复教师的流映射，而是坍缩到条件期望：G_θ^*(x_i^t, x_{<i}^t, t) = E[x_i^0 | x_i^t, x_{<i}^t, t]，这导致模糊的视觉结果。

> [!abstract] 引理3.2（PF-ODE的帧级非注入性，非正式）
> 设x_{1:N}^t满足双向扩散模型的PF-ODE。如果ϕ_Bi(x_{1:N}^t, t)_i相对于x_{other}^t不是几乎处处常数，则对于所有t ∈ (0,1]，对于所有x_{1:N}^t ∈ R^d，存在y_{1:N}^t ∈ R^d，使得y_i^t = x_i^t，且ϕ_Bi(x_{1:N}^t, t)_i ≠ ϕ_Bi(y_{1:N}^t, t)_i。

### Self Forcing的缺陷

> [!warning] Self Forcing的问题
> - **ODE蒸馏阶段**：使用双向教师蒸馏AR学生，违反帧级注入性。
> - **DMD阶段**：无法弥合架构差距，如图2所示。
> - **结果**：性能显著低于标准DMD（蒸馏双向学生）。

## 方法：Causal Forcing

> [!note] 三阶段方法
> 1. **阶段1：自回归扩散训练**：使用教师强制（TF）训练AR扩散模型作为教师。
> 2. **阶段2：因果ODE蒸馏**：使用AR教师进行ODE初始化，满足帧级注入性。
> 3. **阶段3：非对称DMD**：应用与Self Forcing相同的DMD程序，获得少步AR学生。

### 阶段1：自回归扩散训练

> [!important] 教师强制 vs 扩散强制
> - **教师强制（TF）**：基于干净前缀x_{<i}^0进行条件化，学习p_data(x_i^0 | x_{<i}^0)。
> - **扩散强制（DF）**：基于噪声前缀x_{<i}^t进行条件化，学习p_DF(x_i^0 | x_{<i}^t)。
> - **发现**：与普遍看法相反，TF比DF更适合训练AR扩散模型，理论上和经验上都是如此。DF由于训练-推理差距导致视频崩溃。

> [!note] 训练策略
> - **拼接策略**：将干净视频与噪声副本拼接，应用因果注意力掩码，使x_i^t能够关注x_{<i}^0。
> - **教师模型**：使用TF训练的AR扩散模型作为后续ODE蒸馏的教师。

### 阶段2：因果ODE蒸馏

> [!tip] 关键创新
> - **自回归教师**：使用AR教师而非双向教师进行ODE初始化。
> - **帧级注入性**：由于教师是自回归的，其PF-ODE自然满足帧级注入性。
> - **流映射学习**：学生能够准确学习流映射，因为配对数据满足注入性条件。

> [!abstract] 命题3.3（当前Self Forcing ODE蒸馏中的分布不匹配）
> 使用双向扩散模型PF-ODE的配对数据训练因果帧级模型G_θ: (x_i^t, t) → x_i^0，最优解不遵循数据分布：G_θ^*(x_i^t, t) = E[x_i^0 | x_i^t, t] ≁ p_data(x_i^0)。

### 阶段3：非对称DMD

- 应用与Self Forcing相同的DMD程序。
- 使用因果ODE蒸馏初始化AR学生。
- 使用双向基础模型作为教师。
- 进一步提升性能，获得少步AR学生。

## 实验结果

> [!success] 性能表现
> - **全面超越**：在所有指标上超越所有基线模型。
> - **显著提升**：Dynamic Degree提升19.3%，VisionReward提升8.7%，Instruction Following提升16.7%。
> - **实时性能**：保持与Self Forcing相同的推理延迟，实现实时交互式视频生成。
> - **高质量生成**：生成高质量、高动态、符合指令的视频。

> [!note] 与现有方法的比较
> - **Self Forcing**：当前SOTA，Causal Forcing显著超越。
> - **标准DMD**：蒸馏双向学生，Causal Forcing与之竞争或超越。
> - **其他基线**：包括Wan、LTX-Video、CogVideoX等，Causal Forcing全面超越。

## 理论贡献

> [!important] 理论分析
> - **帧级注入性**：形式化ODE蒸馏的关键要求。
> - **架构差距分析**：揭示双向教师到AR学生蒸馏的理论缺陷。
> - **分布不匹配**：证明当前Self Forcing ODE蒸馏导致分布不匹配。
> - **教师强制优势**：理论上证明TF比DF更适合AR扩散训练。

## 应用场景

> [!tip] 实时交互应用
> - **世界建模**：实时生成世界动态，支持交互式探索。
> - **游戏模拟**：实时生成游戏画面，支持交互式游戏体验。
> - **具身智能**：实时生成机器人视角视频，支持机器人学习。
> - **交互式内容创作**：支持用户实时控制和互动。

## 局限性与未来工作

> [!warning] 当前局限
> - 尽管实现了高质量实时生成，但在某些极端复杂场景下，生成质量可能仍有提升空间。
> - 三阶段训练流程相对复杂，可能需要进一步简化。
> - 在某些高分辨率场景下，计算成本可能仍需优化。

> [!quote] 未来方向
> - 探索更简单的训练流程，减少阶段数量。
> - 扩展到更高分辨率和更长视频生成。
> - 增强实时交互控制能力。
> - 与多模态模型结合，实现更丰富的交互体验。

## 总结

Causal Forcing 是一种创新的自回归扩散蒸馏方法，通过识别帧级注入性这一关键原则，揭示了现有方法的理论缺陷，并提出了有效的解决方案。通过使用自回归教师进行因果ODE蒸馏，Causal Forcing 弥合了架构差距，实现了高质量、低延迟的实时交互式视频生成。该方法在所有指标上显著超越最先进方法 Self Forcing，为实时交互式视频应用提供了强大的技术基础。

> [!quote] 引用
> Zhu, H., Zhao, M., He, G., Su, H., Li, C., & Zhu, J. (2026). Causal Forcing: Autoregressive Diffusion Distillation Done Right for High-Quality Real-Time Interactive Video Generation. *International Conference on Machine Learning (ICML)*.
