# PostTraining: SFT-RLHF

---

## Midtaining / Two-phase training

> [!note] 两阶段训练
> - Data Mixture of Stable Stage
> - Data Mixture of Decay Stage: 融合 Instruction Tuning 的数据到其中。

---

## RLHF

> [!note] RLHF 的两个阶段
> - **Imitation (SFT)**
>
>   Fit $\hat p(y|x) = p*(y|x)$ for some reference distribution p*(y|x)
>
> - **Optimization (RLHF)**
>
>   Find $\hat p(y|x)$ such that $\mathcal \max_p E_p[R(y,x)]$ for a reward R(y,x)
>   最大化某种我们可以测量的奖励函数

### Data

首先一些人（众包）针对 output pairs 按照某些标准打分，得到大量有分数的 output pairs。通过对比 output pairs 和分数训练奖励模型，让奖励模型对 LM 的输出进行打分，然后训练。

### How

#### PPO

$$objective(\phi)=E_{(x,y)\sim D_{\pi RL}}[r_\theta(x,y)-\beta \log(\pi^{RL}_\phi(y|x))/\pi^{SFT}(y|x)] + \gamma E_{x-D_pretrain}[\log(\pi^{RL}_\phi (x))]$$

> [!info] 符号含义
> - $\pi^{RL}_\phi$ 是学习到的 RL 策略
> - $\pi^{SFT}$ 是监督训练后的模型
> - $D_{pretrain}$ 是预训练分布
> - $\beta$ 是 KL 散度奖励系数
> - 预训练损失系数 $\gamma$，他们分别控制了 KL 惩罚与预训练梯度

在 PPO 中，$\gamma$ 为 0，也就是：

$$
objective(\phi)=E_{(x,y)\sim D_{\pi RL}}[r_\theta(x,y)-\beta \log(\pi^{RL}_\phi(y|x)/\pi^{SFT}(y|x))
$$

只保留策略散度，这一项限制了 RL 模型的输出不会离 SFT 的输出分布太远。

> [!tip] 两项的作用
> - 奖励项：$r_\theta(x,y)$ 是训练好的奖励模型打分。
> - 约束项：$\beta \log \frac{\pi^{RL}}{\pi^{SFT}}$ 限制了新模型 $\pi^{RL}$ 不要为了刷分而输出连语法都不通顺的怪诞文本（这在 RL 微调早期极易发生）。

##### Reward model

我们训练一个奖励模型去预测给定一个输入 $x$ 人类会如何认为输出 ${y \in [y_0,y_1]}$ 中的哪一个更好，如果实际上 $y_i$ 是人类认为最好的，那么对于这个奖励模型的损失是：

$$
loss(r_\theta) = -E_{(x,y_0,y_1)-D}[\log(\sigma(r_\theta(x,y_i)-r_\theta(x,y_{1-i})))]
$$

##### 人类反馈策略

我们希望使用上方已经训练好的奖励模型去训练可以生成高质量内容的策略。我们将奖励模型的输出作为我们使用 PPO 算法最大化对于所有 LM 输出的奖励（逐个 Token）。

$$
R(x,y) = r_\theta(x,y)-\beta \log(\pi^{RL}_\phi(y|x)/\pi^{SFT}(y|x))
$$

##### PPO-at a conceptual level

> [!note] PPO 演进过程
>
> **Attempt 1: Policy gradients**
> $$
> \begin{aligned}
> J(\theta) = E_{z\sim\pi_\theta}[R(z)] = \sum_z p_\theta(z)R(z)
> \\
> \nabla_\theta E_{p_\theta}(R(z)) = E_{p_\theta}[R(z)\nabla_\theta\log p_\theta(z)]
> \end{aligned}
> $$
>
> **Attempt 2: TRPO (linearize the problem around the current policy)**
> $$
> \max_\theta \hat {\mathbb E_t}[\frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}\hat{A_t}]
> $$
> $$
> s.t. \hat {\mathbb E}[KL[\pi_{\theta_{old}}(.|s_t),\pi_{\theta}(.|s_t)]] \le \delta
> $$
>
> **Attempt 3: PPO (Clip the ratios at some eps)**
> $$
> L(s, a, \theta_k, \theta) = \min(\frac{\pi_\theta(a|s)}{\pi_{\theta_{k}} (a|s)}A^{\pi_{\theta_{k}}}(s,a),clip(\frac{\pi_\theta(a|s)}{\pi_{\theta_{k}} (a|s)},1-\epsilon,\epsilon)A^{\pi_{\theta_{k}}}(s,a))
> $$

> [!question] Can we avoid doing any RL?
> Some reasonable stuff people thought about:
> - Train the model with a control token-(SFT on the pairs, prepend [GOOD] to be chosen, [BAD] not to be chosen).
> - Train the model on only preferred output.
> - Train a reward model, get LM outputs, train on the preferred output.
> - Train a reward model, get 1024 LM outputs, take the best one.

###### 1. $r(\theta)$：概率比率

在 PPO 的上下文中，这个没有下标、带括号的 $r(\theta)$ 特指**当前策略**与**旧策略**在同一动作上的**概率比值**：

$$
r(\theta) = \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}
$$

> [!info] $r(\theta)$ 含义
> - **含义**：衡量"新版模型 $\theta$"想选这个动作的意愿，比"旧版模型 $\theta_{old}$"**强了多少倍**。
> - **作用**：这是一个**缩放因子**。如果这个动作好，我们希望比值变大（概率提升）；如果动作不好，我们希望比值变小。
> - **取值范围**：$[0, +\infty)$。

###### 2. $r_\theta(x,y)$：奖励分数

这是你笔记里 **Reward Model（奖励模型）** 输出的标量值，那个下标 $\theta$ 是**奖励模型自身的参数**（通常在 RLHF 阶段被冻结）。

$$
R = r_\theta(x,y)
$$

> [!info] $r_\theta(x,y)$ 含义
> - **含义**：输入问题 $x$ 和回答 $y$，奖励模型给这个回答打的**分数**（比如 0.8 分或 -0.2 分）。
> - **作用**：告诉策略梯度算法，这个回答 $y$ **到底好不好**。
> - **取值范围**：取决于训练奖励模型时的归一化方式，通常在 $[-10, 10]$ 之间。

###### 3. $\hat{A}$：优势函数

$$
\hat{A} = Q(s,a) - V(s)
$$

> [!info] $\hat{A}$ 含义
> - **通俗解释**：在状态 $s$ 下，选择了动作 $a$ 之后，最终拿到的总回报比"平均水平/预期水平"**高出多少**。
> - **为什么需要它**：如果不减去基线 $V(s)$，策略梯度会有巨大的方差（比如一个回答拿了100分，但在那个语境下随便答答都能拿99分，那么其实这个动作只值1分的优势，模型更新步子就很小）。
> - **在 RLHF 中的落地**：在大模型微调里，我们通常不单独估算 $V(s)$，而是利用**序列级奖励**直接计算优势。

###### 4. 串联起来看 PPO 里的运算

在你提供的公式里，两者是这样协作的：

$$
\text{PPO目标} = \min \left( r(\theta) \cdot \hat{A}, \quad \text{clip}(r(\theta), 1-\epsilon, 1+\epsilon) \cdot \hat{A} \right)
$$

> [!example] 符号角色对照
> | 符号 | 角色类比 | 具体含义 |
> | :--- | :--- | :--- |
> | $r(\theta)$ | **油门/刹车踏板** | 控制"我要不要多生成这个词" |
> | $\hat{A}$ | **导航指令** | 告诉踏板"这个方向对了(+1)还是错了(-1)" |
> | $r_\theta(x,y)$ | **最终目的地评分** | 用来计算出这个导航指令 $\hat{A}$ 的原始数据 |

###### 总结防晕指南

> [!abstract] 符号识别指南
> - **看到 $r(\theta)$（无下标、带括号）**：想**比例**。这是 PPO 特有的**截断对象**。
> - **看到 $r_\theta$（有下标 $\theta$）**：想**分数**。这是奖励模型打的分。
> - **看到 $\hat{A}$**：想**净收益**。正数代表比预期好，负数代表比预期差。

---

#### DPO-derivation from the RLHF formula

Our goal is to optimize:

$$
\max_{\pi_\theta}\mathbb E_{x \sim \mathcal D, y\sim \pi_\theta (y|x)}[r_\phi(x,y)] - \beta \mathbb D_{KL}[\pi_\theta(y|x)||\pi_{ref}(y|x)]
$$

Assume that the policy $\pi$ is the set of all policies (nonparametric assumption). The maximizer is:

$$
\pi_r(y|x) = \frac{1}{Z(x)}\pi_{ref}(y|x)exp(\frac{1}{\beta}r(x,y))
$$

Solve for the implied reward:

$$
r(x,y) = \beta\log\frac{\pi_r(y|x)}{\pi_{ref}(y|x)} + \beta\log Z(x)
$$

##### DPO derivation 2

We can now optimize the implied reward as a reward model via the Stiennon objective:

$$
loss(r_\theta) = -E_{(x,y_0,y_1)\sim D}[\log(\sigma(r_\theta(x,y_i)-r_\theta(x,y_{1-i})))]
$$

$$
r(x,y) = \beta\log\frac{\pi_r(y|x)}{\pi_{ref}(y|x)} + \beta\log Z(x)
$$

This gives the DPO objective:

$$
\mathcal L_{DPO}(\pi_\theta;\pi_{ref})=-E_{(x,y_0,y_1)-D}[\log(\sigma(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}))]
$$

> [!important] DPO 推导的关键步骤
> 1. Make a nonparametric assumption (links $\pi_\theta$ and r in closed form)
> 2. Parametrize reward r via the policy
> 3. Optimize the reward using supervised losses (which in turn, optimizes the policy)

---

## 一、DPO 解决了什么问题？

> [!warning] RLHF-PPO 的痛点
> 回顾标准 RLHF 的 PPO 流程，你需要同时维护或运转 **四个模型**：
>
> 1. **Policy Model**（待训练的目标模型）
> 2. **Reference Model**（冻结的 SFT 模型，用于算 KL 散度）
> 3. **Reward Model**（单独训练的评分模型）
> 4. **Value/Critic Model**（PPO 需要的价值网络）
>
> 这导致**显存爆炸**、**训练不稳定**（容易奖励崩溃），而且流程极长。

> [!important] DPO 的核心洞察
> 在偏好数据（人类选 A 不选 B）已经给定的前提下，**最优策略的概率分布**与**奖励函数**之间存在一个**闭式映射关系**。
>
> 利用这个关系，我们**不需要显式地训练一个奖励模型，也不需要跑 PPO 采样**，直接用一个**分类损失**就能让模型学会偏好。

---

## 二、从 RLHF 控制问题到 DPO 闭式解

### 1. RLHF 的通用优化目标

我们希望找到一个策略 $\pi_\theta$，在最大化奖励的同时，不要偏离参考策略（通常是 SFT 模型）太远：

$$
\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)} \left[ r_\phi(x,y) \right] - \beta \cdot \mathbb{D}_{KL} \left[ \pi_\theta(y|x) \;\|\; \pi_{\text{ref}}(y|x) \right]
$$

### 2. 数学魔法：非参假设下的闭式最优解

> [!important] 闭式最优解
> 如果暂时忘记参数化（假设 $\pi_\theta$ 可以是**任意**概率分布，即非参假设），这个带 KL 约束的最大化问题存在一个**解析解**：
>
> $$\pi_r^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left( \frac{1}{\beta} r(x,y) \right)$$
>
> 其中 $Z(x) = \sum_y \pi_{\text{ref}}(y|x) \exp(r(x,y)/\beta)$ 是归一化配分函数。
>
> **这个公式的意思**：最优策略 $\pi^*$ 的输出概率，正比于参考策略的概率 $\times$ 奖励值的指数放大。

### 3. 倒果为因：把奖励函数"解"出来

既然训练好的最优策略 $\pi^*$ 与奖励 $r$ 有上述一一对应关系，我们反过来可以把**隐含的奖励函数**用策略的概率比表示出来：

$$
\log \pi_r^*(y|x) = \log \pi_{\text{ref}}(y|x) + \frac{1}{\beta} r(x,y) - \log Z(x)
$$

移项得 **隐含奖励公式**（这是 DPO 最关键的一步）：

$$
r(x,y) = \beta \log \frac{\pi_r^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)
$$

---

## 三、用分类损失代替打分训练

### 1. Bradley-Terry 偏好模型

在 RLHF 数据构造中，人类不是直接给分，而是比较 $y_w$（胜者）和 $y_l$（败者）。人类认为 $y_w$ 优于 $y_l$ 的概率服从 Bradley-Terry 模型：

$$
P(y_w \succ y_l | x) = \sigma \left( r(x, y_w) - r(x, y_l) \right)
$$

$\sigma$ 是 Sigmoid 函数。

### 2. 代入隐含奖励：见证奇迹

> [!important] 配分函数抵消
> 把刚才解出来的 $r(x,y)$ 表达式代入 $r(x, y_w) - r(x, y_l)$：
>
> $$r(x, y_w) - r(x, y_l) = \beta \log \frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)} + \underbrace{\beta \log Z(x) - \beta \log Z(x)}_{\text{消掉了！}}$$
>
> **归一化常数 $Z(x)$ 完美抵消！** 这是 DPO 不需要训练奖励模型的根本原因。

### 3. DPO 最终损失函数

现在，我们只需将待训练的策略 $\pi_\theta$ 当作那个"最优策略 $\pi^*$"，最大化偏好概率（即最小化负对数似然）：

$$
\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = - \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]
$$

> [!tip] 直观理解
> - 当模型认为**胜者概率比**远大于**败者概率比**时，Sigmoid 输入为正，损失减小。
> - $\beta$ 控制偏好强度：$\beta$ 越小，模型越自由（允许更大偏离参考模型）；$\beta$ 越大，越保守（必须紧贴参考模型）。

---

## 四、DPO 的梯度分析：它在对模型做什么？

对上述损失求关于参数 $\theta$ 的梯度，可以得到更清晰的物理图像：

$$
\nabla_\theta \mathcal{L}_{\text{DPO}} = - \beta \mathbb{E} \left[ \underbrace{\sigma \left( \hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w) \right)}_{\text{错误程度权重}} \cdot \left( \nabla_\theta \log \pi_\theta(y_w|x) - \nabla_\theta \log \pi_\theta(y_l|x) \right) \right]
$$

其中 $\hat{r}_\theta(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$。

> [!important] 梯度揭示了 DPO 的工作机制
> 1. **增加胜者概率**：梯度方向是 $+ \nabla \log \pi_\theta(y_w)$。
> 2. **降低败者概率**：梯度方向是 $- \nabla \log \pi_\theta(y_l)$。
> 3. **自适应权重**：权重项 $\sigma(\cdots)$ 是 **"模型当前有多错"** 的度量。
>    - 如果模型已经强烈偏好胜者（$\hat{r}_w \gg \hat{r}_l$），Sigmoid 值趋近 0，权重几乎为 0——**学好了就少动**。
>    - 如果模型搞反了（$\hat{r}_l > \hat{r}_w$），Sigmoid 值趋近 1，权重最大——**犯错了就狠改**。
>
> 这正是 **DPO 不需要 Critic 网络却能稳定更新的核心原因**：它自带**自适应步长控制**。

---

## 五、DPO vs RLHF-PPO：一图看懂差异

> [!example] DPO vs RLHF-PPO 对比
> | 对比维度 | **RLHF-PPO** | **DPO** |
> | :--- | :--- | :--- |
> | **模型数量** | 4 个（Policy, Ref, Reward, Critic） | **2 个**（Policy, Ref） |
> | **显存占用** | 极大 | 极小（接近 SFT） |
> | **训练流程** | 采样 $\to$ 打分 $\to$ 计算优势 $\to$ 更新 | **直接读取数据** $\to$ **计算 Loss** $\to$ **更新** |
> | **稳定性** | 需要调 $\beta$、$\gamma$、Clip 等 | 只需调 $\beta$ |
> | **数学本质** | 在线策略强化学习 | **监督分类学习**（带隐含奖励） |

---

## 六、DPO 的局限与边界

> [!warning] DPO 的局限
> 尽管 DPO 简单高效，但它并非万能：
>
> 1. **离线偏好依赖**：DPO 只能利用**已有的静态偏好数据**。如果模型生成的内容偏离数据分布太远，DPO 无法像 PPO 那样通过在线采样纠偏。
> 2. **奖励过度优化**：虽然 DPO 有 KL 约束，但在 $\beta$ 设得太小或训练太久时，依然会导致模型利用概率空间的漏洞去刷高"隐含奖励"。
> 3. **配分函数假设**：推导假设了 $Z(x)$ 在 $y_w$ 和 $y_l$ 相减时消掉。这要求**奖励模型确实是 Bradley-Terry 模型下的最优解**，如果数据生成过程违背该假设，DPO 效果会下降。

---

## 七、一句话总结

> [!abstract] DPO 的本质
> **DPO 的本质是把"强化学习微调"这个复杂的最大期望奖励问题，利用闭式解等价变形为一个简单的"二分类交叉熵"问题，从而砍掉了显式的奖励模型和 PPO 采样过程。**
