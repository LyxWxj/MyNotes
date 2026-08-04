# GRPO（组相对策略优化，Group Relative Policy Optimization）

> 论文：Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*, arXiv:2402.03300（2024）
> 大规模应用：DeepSeek-R1, arXiv:2501.12948
> 相关笔记：[[01-PPO（近端策略优化）]]、[[04-GSPO（组序列策略优化）]]、[[07-DAPO]]、[[08-VAPO]]

## 一、定位与动机

GRPO 是 DeepSeek 为**大规模 LLM RL 后训练**设计的算法，是当前数学/代码推理 RL（DeepSeek-R1、Qwen、GLM 等）的主流基线，也是 GSPO、DAPO、VAPO 等新一代算法的出发点。

针对 PPO 的两个痛点：

1. **价值模型（critic）成本高**：value model 与 policy 同规模，训练显存/算力几乎翻倍；
2. **价值估计在稀疏奖励下不可靠**：LLM 推理任务只在序列末尾给出"对/错"奖励，中间 token 没有即时奖励，critic 很难学准，GAE 的偏差会污染整个优势估计。

GRPO 的思路：**彻底去掉 critic，用同一 prompt 的一组采样结果的相对好坏来估计 advantage**。

## 二、核心思想

对每个 prompt $x$，从当前策略采样 $G$ 个回复 $\{y_1,\dots,y_G\}$，用**组内归一化奖励**作为每个回复（进而每个 token）的 advantage：

$$\widehat{A}_i = \frac{r(x,y_i)-\mathrm{mean}\big(\{r(x,y_j)\}_{j=1}^{G}\big)}{\mathrm{std}\big(\{r(x,y_j)\}_{j=1}^{G}\big)}$$

- 同一组内所有 token 共享同一个 advantage；
- 组内归一化同时起到了"去基线（减均值）"和"尺度归一化（除标准差）"两个作用，等价于用组均值做 baseline 的 REINFORCE 的加强版；
- 显式 KL 惩罚替代 critic 提供的约束，防止策略漂移过远（PPO 中也有 KL，但 GRPO 将其作为唯一正则来源，且用对参考策略的无偏估计）。

## 三、数学原理

### 3.1 优化目标

$$\mathcal{J}_{\mathrm{GRPO}}(\theta)=\mathbb{E}_{x\sim\mathcal{D},\{y_i\}_{i=1}^G\sim\pi_{\theta_{\mathrm{old}}}}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\min\Big(w_{i,t}(\theta)\widehat{A}_{i,t},\ \mathrm{clip}\big(w_{i,t}(\theta),1-\varepsilon,1+\varepsilon\big)\widehat{A}_{i,t}\Big)\right] - \beta\,\mathbb{D}_{\mathrm{KL}}[\pi_\theta\|\pi_{\mathrm{ref}}]$$

其中 per-token 重要性比率：

$$w_{i,t}(\theta)=\frac{\pi_\theta(y_{i,t}|x,y_{i,<t})}{\pi_{\theta_{\mathrm{old}}}(y_{i,t}|x,y_{i,<t})},\qquad \widehat{A}_{i,t}=\widehat{A}_i$$

### 3.2 KL 惩罚项

GRPO 用**无偏的 KL 估计**（Kool et al. 2019 的估计器，避免 `log` 展开的偏置）：

$$\mathbb{D}_{\mathrm{KL}}[\pi_\theta\|\pi_{\mathrm{ref}}]=\mathbb{E}\left[\frac{\pi_{\mathrm{ref}}(y|x)}{\pi_\theta(y|x)}-\log\frac{\pi_{\mathrm{ref}}(y|x)}{\pi_\theta(y|x)}-1\right]=\mathbb{E}\left[\exp\big(\log\pi_{\mathrm{ref}}-\log\pi_\theta\big)-\big(\log\pi_{\mathrm{ref}}-\log\pi_\theta\big)-1\right]$$

per-token 计算、per-token 求和，乘 $\beta$ 后从 reward 或 loss 中扣除（实现上通常从 loss 中扣）。

### 3.3 与 REINFORCE 的关系

不 clip 时（ε 很大），GRPO 梯度近似为：

$$\nabla_\theta\mathcal{J}\approx \mathbb{E}\left[\frac{1}{G}\sum_i \widehat{A}_i\cdot\frac{1}{|y_i|}\sum_t\nabla_\theta\log\pi_\theta(y_{i,t}|x,y_{i,<t})\right]$$

即"组内相对 REINFORCE"：好于组平均的回复整体提升概率，差于组平均的整体压低概率。**clip 的作用**：防止个别 token 的比率偏离过大（尤其负 advantage 时压低概率导致梯度爆炸）。

## 四、算法过程

```
1. 固定参考策略 π_ref（通常是 SFT 模型），初始化 π_θ
2. 循环（每轮）：
   a. 对每个 prompt x，采样 G 个回复（如 G=8~64），保存旧 logprobs
   b. 计算每个回复的奖励 r(x, y_i)（规则验证器 / RM）
   c. 组内归一化得到 advantage Â_i（每个回复一个值，组内所有 token 共享）
   d. 计算 token 级 KL（对 π_ref）作为正则
   e. 切 minibatch、多 epoch 更新策略：
      L = 1/G Σ_i 1/|y_i| Σ_t min(w_t·Â_i, clip(w_t, 1±ε)·Â_i) - β·KL
   f. π_θold ← π_θ
```

## 五、变体与工程技巧

| 变体 / 技巧 | 内容 |
|---|---|
| ε=1（不 clip） | 大规模推理 RL 中常用 ε=1，等价于只保留 min 中的无 clip 项；因为 KL 已足够约束 |
| 关闭 std 归一化 | 组内只减均值、不除标准差（Dr.GRPO 讨论；verl `compute_grpo_outcome_advantage` 可配） |
| Dr.GRPO（动态采样） | arXiv:2504.08919：过滤"全对/全错"的组（同 DAPO Dynamic Sampling），并支持移除 std 归一化 |
| Global Batch Normalization（GBN） | DeepSeek-V3.1 起：advantage 用**全局 batch** 统计而非组内统计，缓解组内方差在 G 小时有偏的问题 |
| Token 级 loss 归一化 | DAPO 指出按 `Σ_i \|o_i\|` 总 token 数归一化比"先每条平均再组平均"更公平（见 [[07-DAPO]]） |
| KL 系数调度 | β 一般取 0.001~0.01；Qwen3/GSPO 实践中甚至设 β=0，靠 clip + 数据策略稳定 |

## 六、与 PPO 的对比

| 维度 | PPO | GRPO |
|---|---|---|
| 价值模型 | 需要 critic（同规模） | 不需要 |
| 优势来源 | GAE（critic 逐步估计） | 组内归一化奖励（outcome 级） |
| 方差控制 | critic 基线 + λ 调度 | 组内统计（G 越大越准） |
| 偏差来源 | critic 估计误差 | 组均值非无偏基线（组数有限） |
| KL 正则 | 可选（对 ref 或 old） | 必需（对 ref，无 critic 时的唯一锚点） |
| 显存/算力 | 高（双模型） | 低（单模型 + ref 前向） |
| 奖励粒度 | 可支持 token 级/过程奖励 | 天然面向 outcome 奖励（组级） |
| 适合场景 | 通用、稠密奖励、agentic | 大规模推理 RL（可验证奖励） |

## 七、优缺点与适用场景

**优点**：去掉 critic 省一半资源；组内归一化简单有效；在数学/代码等"可验证奖励"任务上稳定收敛（DeepSeek-R1 大规模验证）。

**缺点**：
- **样本效率低**：outcome 级 advantage 比 token 级 GAE 粗糙，需要大 G / 大数据；
- **对 G 敏感**：G 太小组内统计方差大；G 太大采样成本高；
- **长序列不稳定**：per-token 重要性比率随序列长度累积方差，MoE / 超长 CoT 训练易崩（GSPO 论文的核心发现）；
- **熵崩塌 / reward hacking**：无 critic 约束时策略易坍缩到捷径（DAPO 解决）。

**适用场景**：大规模推理 RL 的事实标准；有规则验证器的数学、代码任务；多模态验证类任务（verl-omni 的 FlowGRPO 也以它为蓝本）。

## 八、参考

- DeepSeekMath：Shao et al., arXiv:2402.03300
- DeepSeek-R1：arXiv:2501.12948
- Dr.GRPO：*Dynamically Scaled Group Relative Policy Optimization*, arXiv:2504.08919
- 本地参考实现：`verl/verl/trainer/ppo/core_algos.py` 的 `compute_grpo_outcome_advantage`（可关 std 归一化）、`compute_policy_loss`
