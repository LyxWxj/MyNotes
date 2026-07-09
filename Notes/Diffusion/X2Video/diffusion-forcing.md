---
type: Note
related_to: "[[X2Video]]"
status: Active
url: http://arxiv.org/abs/2407.01392
---

# Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion

> [!info] 论文信息
> - **作者**: Boyuan Chen, Diego Marti Monso, Yilun Du, Max Simchowitz, Russ Tedrake, Vincent Sitzmann
> - **日期**: 2024-12-10
> - **arXiv**: [2407.01392](http://arxiv.org/abs/2407.01392)
> - **项目主页**: https://boyuan.space/diffusion-forcing

## 摘要

Diffusion Forcing 是一种新的训练范式，其中扩散模型被训练来对一组具有独立每 token 噪声级别的 token 进行去噪。该方法将 Diffusion Forcing 应用于序列生成建模，通过训练一个因果 next-token 预测模型来生成一个或多个未来 token，而无需完全扩散过去的 token。该方法结合了 next-token 预测模型（如可变长度生成）和全序列扩散模型（如引导采样到理想轨迹的能力）的优势。

> [!tip] 核心思想
> 将扩散过程中的噪声级别视为一种"部分掩码"，每个 token 可以有不同的噪声级别，从而统一了 next-token 预测和全序列扩散的优势。

## 主要贡献

1. **提出 Diffusion Forcing**：一种新的概率序列模型，具有 next-token 预测模型的灵活性，同时能够像全序列扩散模型一样进行长程引导。
2. **决策制定框架**：引入一种新的决策制定框架，允许将 Diffusion Forcing 同时用作策略（policy）和规划器（planner）。
3. **理论证明**：证明在适当条件下，优化所提出的训练目标可以最大化所有子序列的似然下界。
4. **广泛实验评估**：在视频生成、基于模型的规划、视觉模仿学习和时间序列预测等多个领域评估 Causal Diffusion Forcing（CDF），展示了其独特能力。

> [!important] 关键创新
> - **可变长度生成**：可以生成从下一个 token 到数千个 token 的序列，即使对于连续 token 也能保持稳定。
> - **引导采样**：像全序列扩散一样，接受引导以获得高回报的生成。
> - **蒙特卡洛引导（MCG）**：利用因果性、灵活范围和可变噪声调度，显著提高高回报生成的采样效果。

## 方法概述

### 统一视角：噪声作为部分掩码

- **时间轴掩码**：Teacher forcing 将每个 token xt 在时间 t 掩码，并从过去 x1:t−1 进行预测。
- **噪声轴掩码**：全序列前向扩散逐渐向数据添加噪声，可以视为沿噪声轴的部分掩码。
- **Diffusion Forcing**：结合两个轴，每个 token 可以有不同的噪声级别 kt，形成 (x_kt_t)1≤t≤T 的序列。

### 因果扩散强制（Causal Diffusion Forcing, CDF）

- 使用因果架构（如 RNN 或掩码 Transformer）实现。
- 训练模型一次性去噪整个序列，每个 token 有独立的噪声级别。
- 采样时，CDF 逐渐将高斯噪声帧序列去噪为干净样本，不同帧在每个去噪步骤可能有不同的噪声级别。

> [!note] 算法流程
> **训练**：
> 1. 采样观测轨迹 (x1, ..., xT)
> 2. 对每个 t，采样独立噪声级别 kt ∈ {0, 1, ..., K}
> 3. 前向扩散：x_kt_t = ForwardDiffuse(xt, kt)
> 4. 更新隐藏状态 zt ∼ pθ(zt|zt−1, x_kt_t, kt)
> 5. 预测噪声 ϵθ(zt−1, x_kt_t, kt)
> 6. 计算 MSE 损失并反向传播
>
> **采样**：
> 1. 初始化 x1, . . . , xT ∼ N(0, σ²_K I)
> 2. 对每个去噪步骤 m = M−1, ..., 0
> 3. 对每个 t，采样新隐藏状态，计算去噪步骤
> 4. 应用引导：x1:H ← AddGuidance(x_new_1:H, ∇x log c(x_new_1:H))

## 理论保证

> [!abstract] 定理 3.1（非正式）
> Diffusion Forcing 训练过程（算法 1）优化了对所有子序列 token 的期望对数似然 ln pθ((x_kt_t)1≤t≤T) 的重新加权证据下界（ELBO），其中期望是对噪声级别 k1:T ∼ [K]^T 和根据前向过程添加噪声的 x_kt_t 取平均。此外，在适当条件下，优化 (3.1) 同时最大化了所有噪声级别序列的似然下界。

## 实验结果

论文在多个领域进行了广泛实验：

1. **视频生成**：稳定长序列自回归视频生成，超越训练范围。
2. **基于模型的规划**：在决策制定任务中，通过蒙特卡洛引导显著提高性能。
3. **视觉模仿学习**：作为策略和规划器同时使用。
4. **时间序列预测**：展示在连续数据上的稳定性。

> [!success] 关键结果
> - 在视频生成中，CDF 能够生成比训练序列更长的视频，而基线方法会发散。
> - 在规划任务中，MCG 引导比标准引导方法获得更高的回报。
> - 在决策制定中，CDF 同时作为策略和规划器，实现更好的性能。

## 应用与扩展

- **树搜索**：支持高效的树搜索，用于决策制定。
- **组合性**：可以组合训练数据中观察到的子序列，具有用户确定的记忆范围。
- **因果不确定性**：不同 token 可以有不同的不确定性级别，反映其在序列中的位置。

## 局限性与未来工作

- 当前实现主要基于 RNN 架构，Transformer 实现可能进一步提升性能。
- 在非常长的序列上，计算成本可能成为挑战。
- 未来工作可以探索更复杂的引导策略和更大规模的应用。

## 相关工作

- **Teacher Forcing**：传统 next-token 预测训练方法，但无法引导采样且在连续数据上不稳定。
- **全序列扩散**：能够引导采样，但仅限于固定长度序列且非因果。
- **AR-Diffusion**：使用因果架构进行全序列文本扩散，但噪声级别沿时间轴线性相关。
- **Diffusion Forcing** 统一了这些方法的优势，提供了更灵活的框架。

## 总结

Diffusion Forcing 是一种创新的训练范式，成功结合了 next-token 预测和全序列扩散的优点。通过允许每个 token 有独立的噪声级别，它实现了可变长度生成、引导采样和长程稳定性。该方法在视频生成、规划和决策制定等多个领域展示了强大的能力，为序列生成建模提供了新的方向。

> [!quote] 引用
> Chen, B., Monso, D. M., Du, Y., Simchowitz, M., Tedrake, R., & Sitzmann, V. (2024). Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion. *Advances in Neural Information Processing Systems*, 37.
