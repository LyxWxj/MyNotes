# DPO（直接偏好优化，Direct Preference Optimization）

> 论文：Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*, arXiv:2305.18290（NeurIPS 2023）
> 相关笔记：[[05-KTO]]、[[01-PPO（近端策略优化）]]

## 一、定位与动机

经典 RLHF 是**三阶段**流水线：SFT → 训练奖励模型（RM）→ 用 PPO 做 RL。这个流水线有三个问题：

1. **复杂**：RM + 策略 + 参考模型 + 价值模型四套模型；
2. **不稳定**：RL 阶段对超参、初始化敏感，容易崩溃；
3. **样本效率低**：on-policy 采样昂贵。

DPO 的核心洞察：**最优策略与奖励函数之间存在一一对应关系**。既然 RM 的作用只是给偏好数据打分，那么可以直接把"偏好数据"变成策略的监督信号，**跳过显式 RM 和 RL 采样**，把对齐问题转化为一个简单的分类（logistic）损失。DPO 是当前 offline 偏好对齐（SFT 后直接对齐）的事实标准。

## 二、核心思想

从 RLHF 目标出发，**奖励模型的最优解可以写成策略的形式**（隐式奖励）：

$$r(x,y)=\beta\log\frac{\pi_\theta(y|x)}{\pi_{\mathrm{ref}}(y|x)}+\beta\log Z(x)$$

把隐式奖励代入 Bradley-Terry 偏好模型，配对比较的"人类偏好概率"也变成只依赖策略比率的表达式，于是可以直接最大化偏好数据的似然——**无需 RM、无需采样**。

## 三、数学推导（简化版）

### 3.1 RLHF 目标及其闭式解

$$\max_{\pi_\theta}\ \mathbb{E}_{x\sim\mathcal{D},y\sim\pi_\theta}\big[r(x,y)\big]-\beta\,\mathbb{D}_{\mathrm{KL}}\big[\pi_\theta(y|x)\,\|\,\pi_{\mathrm{ref}}(y|x)\big]$$

带 KL 约束的奖励最大化有闭式解（与熵正则 RL / 能量模型同源）：

$$\pi_r(y|x)=\frac{1}{Z(x)}\pi_{\mathrm{ref}}(y|x)\exp\left(\frac{r(x,y)}{\beta}\right)$$

反解出奖励：

$$r(x,y)=\beta\log\frac{\pi_r(y|x)}{\pi_{\mathrm{ref}}(y|x)}+\beta\log Z(x)$$

### 3.2 Bradley-Terry 模型

人类偏好概率（配对比较）：

$$p(y_w\succ y_l|x)=\sigma\big(r(x,y_w)-r(x,y_l)\big)=\frac{1}{1+\exp\big(r(x,y_l)-r(x,y_w)\big)}$$

### 3.3 DPO 损失

把隐式奖励代入 BT 模型，$Z(x)$ 在相减中消去：

$$p(y_w\succ y_l|x)=\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)}-\beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}\right)$$

最大化对数似然即最小化：

$$\mathcal{L}_{\mathrm{DPO}}(\theta)=-\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)}-\beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}\right)\right]$$

### 3.4 梯度含义

$$\nabla_\theta\mathcal{L}_{\mathrm{DPO}}=-\beta\,\mathbb{E}\left[\underbrace{\sigma\left(\beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}-\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)}\right)}_{\text{犯错程度权重}}\Big(\nabla_\theta\log\pi_\theta(y_w|x)-\nabla_\theta\log\pi_\theta(y_l|x)\Big)\right]$$

- 升高 $y_w$ 的概率（+ 项）、降低 $y_l$ 的概率（− 项）；
- **权重 = 当前策略犯错的严重程度**：如果策略已经明显偏好 $y_w$，σ→0，梯度自动变小（自正则，无需 clip）；
- β 控制对偏离 π_ref 的惩罚：β 越大，策略越不敢偏离参考模型（同 RLHF 中的温度/正则强度）。

## 四、算法过程

```
1. 准备偏好数据集 {(x, y_w, y_l)}
2. 对每个样本：
   a. 用冻结的 π_ref 计算 y_w、y_l 的 logprob（可离线缓存）
   b. 用 π_θ 计算 y_w、y_l 的 logprob
   c. 计算 log-ratio 差，套 sigmoid + log 得到 loss
   d. 反向传播（只更新 π_θ）
3. 无需采样、无需 RM、无需价值模型
```

全程 offline，训练稳定，因此也成为后续许多算法的"骨架"（KTO、IPO、SimPO、ORPO 等都是替换其中的组件）。

## 五、变体与已知问题

| 变体 | 改动 | arXiv |
|---|---|---|
| IPO | 用均方误差替代 log-sigmoid，避免 DPO 的过拟合/正则化偏差 | 2310.12036 |
| KTO | 去掉配对，只用 desirable/undesirable 二值信号 | 2402.01306 |
| SimPO | 去掉 π_ref，直接在平均对数概率上加 margin | 2405.14734 |
| ORPO | 与 SFT 合并：单阶段"偏好 + 负对数似然" | 2403.07691 |
| cDPO | 用约束优化解决 DPO 与偏好似然不一致的问题 | 2406.06536 |
| DPO-Positive | 增加正样本 NLL 项（模仿学习） | 2406.03123 |

**已知问题**
- **长度偏差**：token 级 logprob 求和使 DPO 系统性偏好长回复（token 越多、概率越低、但求和掩盖了这一点），实践中需长度归一化或数据过滤；
- **分布外问题**：offline 数据来自 SFT 分布，策略更新后数据不更新（无采样）；
- **参考模型退化**：π_ref 与 π_θ 差异过大时 loss 饱和，梯度消失。

## 六、与 PPO 的对比

| 维度 | PPO | DPO |
|---|---|---|
| 数据 | on-policy 采样 | offline 偏好对（可复用） |
| 奖励模型 | 需要 | 不需要（隐式奖励） |
| 价值模型 | 需要 | 不需要 |
| 训练目标 | 最大化奖励（RL） | 最大化偏好似然（分类） |
| 稳定性 | 差（需大量调参） | 好（监督式训练） |
| 上限 | 高（可在线探索、可加新奖励） | 受限于离线数据质量 |

## 七、适用场景

- **SFT 后的标准偏好对齐**（Chat 模型的最后一步），尤其是算力有限、希望稳定收敛的团队；
- 人类偏好数据丰富但**没有规则验证器**的场景（对话、写作、风格对齐）；
- 作为 online 算法的**初始化或混合组件**（如 online DPO、与 PPO 的 KL 正则共用）。

不适合：需要持续探索、奖励函数可实时计算（数学/代码验证）的高强度推理任务——那些场景 GRPO 系更合适。

## 八、参考

- DPO：Rafailov et al., arXiv:2305.18290
- 理论延伸：*A General Theoretical Paradigm to Understand Learning from Human Preferences*（IPO，arXiv:2310.12036）
- 本地实现参考：verl 的 `verl/trainer/dpo_trainer.py`（DPO/KTO/IPO 统一实现）
