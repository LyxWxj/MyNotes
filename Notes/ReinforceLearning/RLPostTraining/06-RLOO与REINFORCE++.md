# RLOO 与 REINFORCE++（无价值模型的策略梯度算法）

> RLOO：Ahmadian et al., *Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs*, arXiv:2402.14799（Google DeepMind, 2024）
> REINFORCE++：Hu et al. (GLM 团队), *REINFORCE++: A Simple and Efficient Approach for Aligning Large Language Models*, arXiv:2501.03262
> 相关笔记：[[02-GRPO（组相对策略优化）]]、[[01-PPO（近端策略优化）]]

## 一、定位与动机

PPO 的性能很大程度来自价值模型（critic）提供的低方差基线；GRPO 则用"组内归一化"替代 critic。**RLOO 与 REINFORCE++ 属于第三条路线：回到最朴素的 REINFORCE，用更轻的基线/归一化手段减方差，彻底不要 critic。**

共同动机：

1. **critic 昂贵且难训**：价值模型与策略同规模，训练成本翻倍；在稀疏的 outcome 奖励下价值估计不准，反而引入偏差；
2. **实现极简**：不需要 GAE、不需要 λ、不需要 value loss，一条 loss 公式即可；容易在任意框架（verl、TRL、ms-swift）中落地；
3. 在**可验证奖励（数学/代码）**场景，奖励本身信噪比高，复杂的优势估计收益有限，简单方法反而更稳。

## 二、RLOO：REINFORCE with Leave-One-Out baseline

### 2.1 核心思想

对每个 prompt $x$ 采样 $K$ 个回复 $\{y_1,\dots,y_K\}$。经典 REINFORCE 用"组内均值"做基线；RLOO 用**留一法（leave-one-out）**：估计第 $i$ 个回复的优势时，排除它自己，用其余 $K-1$ 个的均值做基线：

$$\widehat{A}(x,y_i)=R(x,y_i)-\frac{1}{K-1}\sum_{j\neq i}R(x,y_j)$$

等价于：

$$\widehat{A}(x,y_i)=\frac{K}{K-1}\Big(R(x,y_i)-\frac{1}{K}\sum_{j=1}^{K}R(x,y_j)\Big)$$

- 与组均值基线相比只差常数因子 $K/(K-1)$，但**估计上无偏性更好**：自己的奖励不出现在基线里，避免自我消减（self-penalty）；
- 用整条回复的奖励 $R(x,y_i)$（**轨迹级**），不做 token 级 credit assignment；
- 不需要 critic、不需要参考模型（可选加 KL 正则），损失：

$$\mathcal{L}_{\mathrm{RLOO}}=-\mathbb{E}_{x}\left[\sum_{i=1}^{K}\widehat{A}(x,y_i)\log\pi_\theta(y_i|x)\right]$$

### 2.2 算法过程

```
1. 对每个 prompt 采样 K 个回复，保存旧 logprobs
2. 计算奖励 R(x, y_i)
3. 留一法基线：Â_i = R(x,y_i) - mean(R_{j≠i})
4. 更新策略：L = -Σ_i Â_i · log π_θ(y_i|x)（轨迹级 logprob）
5. 可选：加 KL 正则（对 ref）；无 clip、无归一化
```

### 2.3 特点

- **对奖励尺度敏感**：不做 std 归一化，奖励绝对值影响更新幅度，reward scale 不同需调学习率；
- 无 clip：需要小学习率 + KL 正则保证稳定；
- 论文结论：在偏好对齐场景（helpfulness/harmlessness）中，**RLOO 用 4 个采样即可匹配甚至超过 PPO**，且对 KL 系数更鲁棒——"回到基础"也能赢。

## 三、REINFORCE++：GLM 的工程化 REINFORCE

REINFORCE++ 是智谱 GLM 团队在 RLOO 基础上做的工程化升级，目标是在**大规模推理 RL**中同时拿到"无 critic 的简单"和"PPO 级的稳定"。

### 3.1 四项关键设计

1. **token 级 KL 惩罚（累计回报形式）**：对第 $t$ 个 token，回报 = 最终奖励减去**从 t 到序列末尾累计的 KL**：

$$R(q,o_t)=r(o_{1:T},q)-\beta\sum_{i=t}^{T}\mathbb{D}_{\mathrm{KL}}(i)$$

其中 $\mathbb{D}_{\mathrm{KL}}(i)=\mathrm{KL}\big(\pi_\theta(\cdot|q,o_{<i})\,\|\,\pi_{\mathrm{ref}}(\cdot|q,o_{<i})\big)$。这样 KL 惩罚随生成进度**逐 token 摊销**，避免"KL 只在序列末尾惩罚"导致的熵崩塌；

2. **PPO 式 clip**：保留 clip 目标控制 off-policy 风险：

$$\mathcal{L}=-\mathbb{E}\left[\sum_t\min\Big(r_t(\theta)\widehat{A}_t,\ \mathrm{clip}\big(r_t(\theta),1-\varepsilon,1+\varepsilon\big)\widehat{A}_t\Big)\right]$$

3. **全局 advantage 归一化**：不用组内统计，而是在**整个 batch 上**把 advantage 归一化为均值 0、标准差 1（whitening），对奖励尺度鲁棒；
4. **无 critic / 无 GAE**：省掉价值模型约一半训练显存与算力。

### 3.2 算法过程

```
1. 对每个 prompt 采样若干回复，保存旧 logprobs
2. 计算最终奖励 r(q, o_{1:T})
3. per-token 回报：R(q, o_t) = r - β·Σ_{i=t..T} KL_i（token 级 KL 累计）
4. 全局 batch 归一化 advantage（均值 0 / 标准差 1）
5. token 级 PPO-clip 目标更新策略
```

## 四、三兄弟对比：RLOO vs REINFORCE++ vs GRPO

| 维度 | RLOO | REINFORCE++ | GRPO |
|---|---|---|---|
| 基线/归一化 | 留一法（组内） | 全局 batch whitening | 组内 mean + std |
| clip | 无 | 有（PPO 式） | 有（可设 ε=1 关闭） |
| KL 正则 | 可选（对 ref） | 必需（per-token 累计） | 必需（对 ref） |
| 优化单元 | 轨迹级 | token 级 | token 级 |
| 对奖励尺度敏感 | 敏感 | 不敏感（全局归一化） | 不敏感（组内归一化） |
| 稳定性 | 一般（需小 LR） | 好 | 好（长序列略差，见 [[04-GSPO（组序列策略优化）]]） |
| 实现成本 | 最低 | 低 | 低 |

## 五、优缺点与适用场景

**优点**
- 无 critic：显存/算力省一半，分布式训练简单（GLM-4.5 大规模验证 REINFORCE++）；
- 实现极简，任意框架 50 行内可落地；
- 对可验证奖励任务效果好（奖励信噪比高，不需要价值模型）。

**缺点**
- 样本效率低于 PPO（无逐步价值信息），需要更大的 rollout batch；
- 对 reward scale / 数据分布敏感（RLOO 尤甚）；
- 轨迹级信号在长序列上粒度粗，需靠 KL + 归一化控方差。

**适用场景**：数学/代码等可验证奖励的大规模推理 RL；算力受限（省 critic 显存）；快速原型验证；作为 GRPO/GSPO 的轻量替代。

## 六、参考

- RLOO：Ahmadian et al., arXiv:2402.14799
- REINFORCE++：Hu et al., arXiv:2501.03262
- 原始 REINFORCE：Williams, *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning*, Machine Learning 1992
- 实现参考：verl `compute_rloo_outcome_advantage`、`compute_reinforce_plus_plus_outcome_advantage`（`core_algos.py`）
