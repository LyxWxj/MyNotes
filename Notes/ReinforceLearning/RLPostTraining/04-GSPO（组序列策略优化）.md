# GSPO（组序列策略优化，Group Sequence Policy Optimization）

> 论文：Zheng et al. (Qwen Team), *Group Sequence Policy Optimization*, arXiv:2507.18071（2025）
> 相关笔记：[[02-GRPO（组相对策略优化）]]、[[07-DAPO]]、[[01-PPO（近端策略优化）]]

## 一、定位与动机

GSPO 是 Qwen 团队在 2025 年提出的 RL 算法，**已用于 Qwen3 系列的大规模 RL 训练**。它直接挑战 GRPO 的一个"默认设计"：per-token 的重要性比率（importance ratio）。

论文指出的核心问题：

1. **token 级重要性采样在数学上不成立**：重要性采样要求"对行为分布采样的多个样本求平均"来校正分布偏移，而 GRPO 的每个 token 位置只采样了一个 token，单个样本的比率 $\pi_\theta(y_t|x,y_{<t})/\pi_{\theta_{\mathrm{old}}}(y_t|x,y_{<t})$ 本质上是噪声，不是分布校正；
2. **方差随序列长度累积**：token 级噪声在长回复（长 CoT）上逐 token 累积，clip 机制会进一步放大异常 token 的影响，最终导致**不可逆的训练崩溃（model collapse）**——在 MoE 大模型、长 CoT 任务上尤其严重；
3. **优化单元与奖励单元不一致**：奖励是序列级的（整个回复得一个分），优化却在 token 级做 off-policy 校正。

GSPO 的回答：**把重要性比率、clip、奖励、优化全部放到序列级**。

## 二、核心思想

用**序列似然**定义重要性比率，并做长度归一化：

$$s_i(\theta)=\left(\frac{\pi_\theta(y_i|x)}{\pi_{\theta_{\mathrm{old}}}(y_i|x)}\right)^{\frac{1}{|y_i|}}=\exp\left(\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\log\frac{\pi_\theta(y_{i,t}|x,y_{i,<t})}{\pi_{\theta_{\mathrm{old}}}(y_{i,t}|x,y_{i,<t})}\right)$$

- 这是逐 token log-ratio 的**算术平均**再取指数（等价于几何平均），有明确的统计含义：回复在"平均每 token"意义上偏离旧策略多少；
- 长度归一化保证不同长度回复的比率在同一数值范围，clip 范围才有意义；
- 整个回复作为一个"动作"做 clip、加权、优化——与序列级奖励天然对齐。

## 三、数学原理

### 3.1 目标函数

$$\mathcal{J}_{\mathrm{GSPO}}(\theta)=\mathbb{E}_{x\sim\mathcal{D},\{y_i\}_{i=1}^G\sim\pi_{\theta_{\mathrm{old}}}}\left[\frac{1}{G}\sum_{i=1}^{G}\min\Big(s_i(\theta)\widehat{A}_i,\ \mathrm{clip}\big(s_i(\theta),1-\varepsilon,1+\varepsilon\big)\widehat{A}_i\Big)\right]$$

优势仍用组内归一化（与 GRPO 相同）：

$$\widehat{A}_i=\frac{r(x,y_i)-\mathrm{mean}\big(\{r(x,y_j)\}_{j=1}^G\big)}{\mathrm{std}\big(\{r(x,y_j)\}_{j=1}^G\big)}$$

### 3.2 梯度分析：GSPO vs GRPO

忽略 clip，两者的梯度（权重部分）对比：

$$\nabla_\theta\mathcal{J}_{\mathrm{GRPO}}\propto \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\widehat{A}_i\cdot\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\underbrace{\frac{\pi_\theta(y_{i,t}|x,y_{i,<t})}{\pi_{\theta_{\mathrm{old}}}(y_{i,t}|x,y_{i,<t})}}_{\text{每个 token 各自加权}}\nabla_\theta\log\pi_\theta(y_{i,t}|x,y_{i,<t})\right]$$

$$\nabla_\theta\mathcal{J}_{\mathrm{GSPO}}\propto \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\underbrace{s_i(\theta)\widehat{A}_i}_{\text{整条回复一个权重}}\cdot\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\nabla_\theta\log\pi_\theta(y_{i,t}|x,y_{i,<t})\right]$$

- GRPO：回复内每个 token 用**自己的**比率加权（同一条回复内权重可以相差几个数量级），这些不平衡权重随训练累积，是不稳定的根源；
- GSPO：回复内所有 token **等权重**（都是 $s_i\widehat{A}_i$），只保留了"整条回复是否 off-policy"这一个序列级信号，方差大幅降低。

### 3.3 方差分析

论文用理论分析证明：在相同条件下，**GSPO 梯度估计器的方差不超过 GRPO**（clip 前），且 GSPO 的 clip 触发比例远低于 GRPO（实验中低约两个数量级），说明 GSPO 的 off-policy 校正本身更"温和"，训练信号更干净。

## 四、算法过程

```
1. 固定 π_ref，初始化 π_θ
2. 循环（每轮）：
   a. 对每个 prompt 采样 G 个回复，保存旧 logprobs（per token）
   b. 计算奖励 r(x, y_i)
   c. 组内归一化得到 Â_i（序列级）
   d. 序列级 IS：s_i = exp(mean_t(log π_θ(y_{i,t}) - log π_θold(y_{i,t})))
   e. 更新策略（切 minibatch、多 epoch）：
      L = 1/G Σ_i min(s_i·Â_i, clip(s_i, 1±ε)·Â_i)（可选 KL/熵正则）
   f. π_θold ← π_θ
```

注意：实现上只需把"per-token log-ratio"在 mask 内先求和平均成序列标量，再广播回每个 token，**计算量与 GRPO 完全相同**（verl/ms-swift 均提供 `importance_sampling_level=sequence` 开关）。

## 五、GSPO-token 变体

多轮对话（multi-turn RL）等场景需要更细粒度的 token 级优势时，论文提出 GSPO-token：

$$w_{i,t}^{\mathrm{GSPO\text{-}token}}=\mathrm{sg}\big[s_i\big]\cdot\frac{\pi_\theta(y_{i,t}|x,y_{i,<t})}{\mathrm{sg}\big[\pi_\theta(y_{i,t}|x,y_{i,<t})\big]}$$

其中 $\mathrm{sg}[\cdot]$ 表示 stop-gradient（detach）。

- 序列级部分 $s_i$ 用**停梯度**的常数权重（提供序列级尺度），token 级部分只保留"当前 token 自身概率变化"的梯度；
- 论文证明：**当所有 token 的 advantage 相同时（GRPO 的标准设置），GSPO-token 与 GSPO 等价**；
- 价值在于为未来"细粒度 token 优势"（如过程奖励、value model 提供 token 级 Â）预留扩展空间，同时不重蹈 GRPO 的方差覆辙。

## 六、关键超参与实践

| 超参 | GSPO 论文取值 | 说明 |
|---|---|---|
| ε | 3e-4 量级 | 序列级比率是几何平均，数值远小于 token 级比率，clip 范围随之小 2~3 个数量级 |
| 组大小 G | 8~64 | 与 GRPO 相同 |
| KL 系数 β | 0（Qwen 实践） | 论文框架保留 KL 项，Qwen3 大规模训练中设 β=0，靠 clip + 数据策略稳定 |
| minibatch / epoch | 4 minibatches / 若干 epoch | 与 GRPO 相同 |

## 七、与 GRPO 的对比

| 维度 | GRPO | GSPO |
|---|---|---|
| 重要性比率 | token 级（每 token 各自） | 序列级（长度归一化几何平均） |
| clip 单元 | 单 token | 整条回复 |
| 优化单元 | token | 序列（与奖励单元一致） |
| 梯度方差 | 随长度累积、易崩 | 低、稳定 |
| MoE / 超长 CoT | 需要额外稳定策略 | 天然稳定（论文实验验证） |
| 实现成本 | — | 几乎为零（同计算量） |

## 八、优缺点与适用场景

**优点**：训练稳定（尤其 MoE、长 CoT）；与 GRPO 完全同构，**一行开关即可切换**；Qwen3 大规模验证；为细粒度优势（GSPO-token）留了接口。

**缺点**：仍然依赖组内归一化（outcome 级信号），**样本效率与 GRPO 相当**，不解决稀疏奖励/熵崩塌问题（可与 DAPO 类技巧组合）；序列级 clip 对"单 token 突变的恶意攻击"（reward hacking 的一种）不如 token 级敏感。

**适用场景**：大规模推理 RL（数学、代码）；MoE 架构；超长 CoT；对训练稳定性要求极高的生产环境——当前（2025 下半年起）很多团队把它当作 GRPO 的默认替代品。

## 九、参考

- GSPO：Zheng et al. (Qwen Team), arXiv:2507.18071
- Qwen3 技术报告：*Qwen3 Technical Report*, arXiv:2505.09388
- 实现参考：verl（`importance_sampling_level` 参数）、ms-swift（`--importance_sampling_level sequence/sequence_token`）
