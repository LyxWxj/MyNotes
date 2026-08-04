---
type: Note
related_to:
  - "[[FlowMatching]]"
  - "[[DDIM]]"
  - "[[Diffusion]]"
status: Active
url: https://arxiv.org/abs/2209.03003
code: https://github.com/gnobitab/RectifiedFlow
---

# Rectified Flow（整流流）

> [!cite] 来源
> - Liu, Gong & Liu. *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*. arXiv:2209.03003（ICLR 2023）
> - Liu. *Rectified Flow: A Marginal Preserving Approach to Optimal Transport*. arXiv:2209.14577
> - Hammour Yue. *Diffusion学习笔记（二十）——深入理解Rectified Flow，完善统一扩散框架*. 知乎（作者专栏：https://www.zhihu.com/people/bai-e-ji-wan-qi/posts）
> - 相关论文：Rectified Diffusion（arXiv:2410.07303）、Stochastic Interpolants（arXiv:2303.08797）、2-Rectifications Are Enough（arXiv:2410.14949）
> - 官方项目页：https://www.cs.utexas.edu/~lqiang/rectflow/html/intro.html
> - 关联笔记：[[FlowMatching]]、[[DDIM]]、[[DDPM]]

---

## 1. 核心思想

### 1.1 问题定义：传输映射（Transport Map）

给定两个分布 $\pi_0, \pi_1$，找传输映射 $T: \mathbb{R}^d \to \mathbb{R}^d$，使得当 $Z_0 \sim \pi_0$ 时 $Z_1 := T(Z_0) \sim \pi_1$，即 $(Z_0, Z_1)$ 构成 $\pi_0, \pi_1$ 的一个耦合（transport plan）。

- **生成建模**：$\pi_1$ = 数据分布，$\pi_0$ = 标准高斯。
- **域迁移（transfer）**：$\pi_0, \pi_1$ 都是观测到的分布，做图像翻译、风格迁移、域适应。

> [!note] 为什么不用最优传输（OT）？
> Monge OT 问题 $\min_T \mathbb{E}[c(T(Z_0)-Z_0)]$ 在高维大数据场景下难以高效求解；而且传输成本与下游学习性能并不完全一致。Rectified Flow 的答案：**不追求"成本最优"，而是显式偏好"路径是直线"的 ODE**——直线既是最短路径（与 OT 的 Monge 解有联系），又恰好计算上最友好。

### 1.2 为什么直线路径这么重要？

> [!important] 直线 = 无离散误差
> 若 ODE $\mathrm{d}Z_t = v(Z_t, t)\mathrm{d}t$ 的路径完全笔直，则 $Z_t = Z_0 + t\,v(Z_0, 0)$，**单步 Euler 即可精确求解**，无需任何时间离散化。这让推理速度接近 GAN/VAE 等一步模型，同时保留 ODE/SDE 多步模型在训练上的稳定性质。

### 1.3 线性插值及其"非因果"问题

给定端点 $X_0 \sim \pi_0, X_1 \sim \pi_1$，定义线性插值：

$$X_t = tX_1 + (1-t)X_0, \quad t \in [0,1]$$

它满足一个平凡的 ODE：$\mathrm{d}X_t = (X_1 - X_0)\mathrm{d}t$，确实把 $\pi_0$ 送到 $\pi_1$，但：

> [!warning] 非因果（non-causal / anticipating）
> 更新 $X_t$ 需要终点 $X_1$ 的信息，而 $t<1$ 时并不知道终点。表现在轨迹上：多条插值线在中间相交，交点处的更新方向不唯一，无法用因果 ODE $\mathrm{d}Z_t = v(Z_t,t)\mathrm{d}t$ 描述。

### 1.4 整流（Rectification）：把插值"因果化"

把 $X_t$ 用 L2 投影到"可因果模拟的 ODE"空间，即求解最小二乘回归：

$$\min_v \int_0^1 \mathbb{E}\left[\left\| (X_1 - X_0) - v(X_t, t) \right\|^2\right]\mathrm{d}t$$

其理论最优解为**条件期望**：

$$v(z,t) = \mathbb{E}[X_1 - X_0 \mid X_t = z]$$

即"所有在时刻 $t$ 穿过点 $z$ 的直线的平均方向"。由该 $v$ 驱动的 ODE 就是由 $(X_0, X_1)$ 诱导的 **rectified flow**。

> [!tip] 类比：修路与交通
> 线性插值 $X_t$ 相当于在 $\pi_0$ 与 $\pi_1$ 之间修好"道路"；rectified flow 则是粒子在这些道路上**短视、无记忆、不交叉**地流动——在交叉点处重新接线（rewiring），无视原始端点配对，最终形成一条新的、更确定性的配对。

---

## 2. 训练算法

Rectified Flow 的训练就是一个普通的回归问题：

> [!example] 训练步骤（与 [[FlowMatching]] 的 CFM 完全一致）
> 1. 采样端点 $(X_0, X_1) \sim \pi_0 \times \pi_1$（通常取**独立耦合**，即随机配对）+ 时间 $t \sim U[0,1]$
> 2. 构造中间点 $X_t = tX_1 + (1-t)X_0$
> 3. 最小化 $\left\|v_\theta(X_t, t) - (X_1 - X_0)\right\|^2$

- 无需 GAN 式对抗训练、无需变分推断、无需精心设计噪声调度 $\beta_t$。
- **时间对称**：交换 $X_0, X_1$ 并翻转 $v$ 的符号得到等价问题，因此前向（噪声→数据）与反向（数据→噪声）采样同样容易。

---

## 3. 关键性质（理论保证）

### 3.1 边际保持（Marginal Preserving）

> [!important] 核心性质
> $$\mathrm{Law}(Z_t) = \mathrm{Law}(X_t), \quad \forall t \in [0,1]$$
>
> rectified flow 的轨迹与插值轨迹在**每个时刻的边际分布完全相同**，因此 $(Z_0, Z_1)$ 是 $\pi_0, \pi_1$ 的合法耦合。

### 3.2 传输成本单调不增

$$ \mathbb{E}[c(Z_1 - Z_0)] \leq \mathbb{E}[c(X_1 - X_0)], \quad \forall \text{凸函数 } c $$

> [!note] 直觉
> rectification 把任意耦合（典型地是独立耦合 $\pi_0 \times \pi_1$）变成**确定性耦合**，且对所有凸成本函数同时保证成本不增。这正是"整流"一词的由来：把混乱的随机配对整理成更经济的确定性配对。

> [!tip] 证明思路（两次 Jensen）
> 因为 $Z_1 - Z_0 = \int_0^1 v(Z_t, t)\,\mathrm{d}t = \int_0^1 \mathbb{E}[X_1 - X_0 \mid Z_t]\,\mathrm{d}t$，对凸函数 $c$ 先用 Jensen 把 $c\left(\int \mathbb{E}[\cdot]\,\mathrm{d}t\right)$ 拆进积分，再用条件期望的 Jensen 拆进期望，最后回到 $\mathbb{E}[c(X_1-X_0)]$。直观上：ODE 在每个点取"平均方向"，沿平均方向的运输成本不会超过原本任意一条连线。

### 3.3 轨迹不交叉（Non-crossing）

在 $v$ 满足 Lipschitz 条件时 ODE 解存在且唯一（Picard-Lindelöf 定理），因此任意两条轨迹不会在 $t<1$ 时沿不同方向穿过同一点——交叉被重接线消除。

---

## 4. Reflow：迭代整流，逼近直线插值

记整流算子为 $\boldsymbol{Z} = \mathtt{Rectflow}((X_0, X_1))$，递归应用：

$$\boldsymbol{Z}^{k+1} = \mathtt{Rectflow}((Z_0^k, Z_1^k)), \quad \boldsymbol{Z}^0 = (X_0, X_1)$$

即用第 $k$ 代 flow 采样出的端点对 $(Z_0^k, Z_1^k)$ 作为新的训练数据，再训练下一代 flow。实现上：先用当前 ODE 采样合成一个配对数据集，再在上面跑第 2 节的回归训练。

### 4.1 直线度度量

$$S(\boldsymbol{Z}) = \int_0^1 \mathbb{E}\left[\left\| Z_1 - Z_0 - v(Z_t, t) \right\|^2\right]\mathrm{d}t$$

$S=0$ 当且仅当轨迹是完美直线。理论保证：

> [!important] 拉直定理
> $$\min_{k \leq K} S(\boldsymbol{Z}^k) = O(1/K)$$
>
> 随着 reflow 迭代次数 $K$ 增加，路径的直线度以 $O(1/K)$ 收敛到 0。

### 4.2 Reflow 的双重作用

1. **降低传输成本**：每次迭代都让耦合更接近最优（凸成本单调不增）。
2. **拉直轨迹**：离散误差随迭代下降，最终单步 Euler 即可。

> [!tip] 实践经验
> 图像生成中 **reflow 一次就够**（2-rectified flow 的轨迹已接近直线）。代价是额外训练开销：需要用 teacher ODE 采样合成配对数据集。[[FlowMatching]] 的标准形式没有 reflow，因此通常需要 5-20 步；rectified flow + reflow 之后可以 1-2 步生成。

### 4.3 本质是"贴近插值方程"，不是"拉直"

> [!warning] "拉直"一词容易误导
> 1-rectified flow 的真实 ODE 本来就是弯曲的（见 §5），并不是"原本是直线、因训练误差变弯、再训练把它拉直"。reflow 的正确理解是：**用 1-rectified 采样出的配对点 $(Z_0, Z_1)$ 重新构造直线插值并训练新 ODE，让新 ODE 贴近这条插值方程**。因为线性插值的速度 $Z_1 - Z_0$ 是常数，新 ODE 拟合它之后自然趋向直线。

### 4.4 Reflow 只能加速，不提升生成质量

> [!note] 2-rectified 的效果 ≤ 1-rectified
> 第二次训练以第一个模型产生的配对点为基底，因此 reflow **理论上只能加速**（用更少步数达到同等质量），生成质量上限至多与 1-rectified 持平。2-rectified 的价值在于把"多步质量"压缩进"一步"，而不是提升天花板；继续迭代（3-rectified 及以上）收益递减，甚至可能变差（这一点与论文定理 3.7 的 $O(1/K)$ 收敛描述并不完全吻合，详见 §9 局限）。

---

## 5. 深入辨析：真实 ODE 不是直线

> [!warning] 常见误解
> 训练损失的最优解不是 $X_1 - X_0$ 本身，而是它的**条件期望**：
> $$\frac{\mathrm{d}X_t}{\mathrm{d}t} = v^*(X_t, t) = \mathbb{E}_{X_0, X_1}[X_1 - X_0 \mid X_t]$$
> 推导：把回归损失的平方展开成三项，交叉项用全期望公式（tower property）把 $\mathbb{E}_{X_0,X_1}[\cdot]$ 整理成 $\mathbb{E}_{X_t}[\cdot]$，损失最终化为 $\mathbb{E}_{X_t}\left[\left\|v_\theta - \mathbb{E}[X_1-X_0 \mid X_t]\right\|^2\right] + C$，最优解即条件期望。

### 5.1 为什么通常不是直线

对于某个中间状态 $X_t$，一般存在**多对** $(X_0, X_1)$ 的连线穿过它（连续分布下"中间区域"几乎总有相交的线）。条件期望是对这些方向的**平均**，方向随 $X_t$ 变化 → ODE 是弯曲的，**与普通扩散模型一样不能一步采样**。

> [!note] 什么时候才是直线？
> 只有当穿过每个点 $X_t$ 的连线**至多一条**时（如分布边缘区域），$v^*(X_t,t) = X_1 - X_0$ 才是常数。复杂分布不可能用互不相交的直线连起两侧的所有点对——这正是 ODE"轨迹不交叉"约束的直接推论。

### 5.2 直线性藏在插值选择里

从随机插值框架看 $v(X_t, t) = \mathbb{E}[\dot{X}_t \mid X_t]$：线性插值的速度 $\dot{X}_t = X_1 - X_0$ 是常数，条件期望的"内核"是常数。**直线来自插值选择，而非训练本身**——这就是为什么 reflow 能把 ODE 逼近直线，而 DDIM 类（非线性插值）即使 reflow 也只是贴近弯曲的插值曲线。

### 5.3 两个容易混淆的 ODE

- **插值 ODE**：$\frac{\mathrm{d}X_t}{\mathrm{d}t} = X_1 - X_0$——真实但**非因果**（依赖终点 $X_1$）；
- **真实 ODE**：$\frac{\mathrm{d}X_t}{\mathrm{d}t} = \mathbb{E}[X_1 - X_0 \mid X_t]$——因果、可模拟，是 rectified flow 真正对应的 ODE；
- 训练出的 $v_\theta$ 是后者（真实 ODE）的近似，而不是前者的近似。

---

## 6. 与其他方法的关系

### 6.1 统一框架：因果化任意插值

rectified flow 可以推广到**任意光滑插值过程** $X_t$（连接 $X_0, X_1$）：

$$v(z,t) = \mathbb{E}[\dot{X}_t \mid X_t = z], \quad \min_v \int_0^1 \mathbb{E}\left[\|\dot{X}_t - v(X_t, t)\|^2\right]\mathrm{d}t$$

此时边际保持仍然成立，但**不再保证凸成本下降，也不保证 reflow 后变直**——直线性依赖线性插值。

### 6.2 DDIM / 概率流 ODE 是特例

取 $X_t = \alpha_t X_1 + \beta_t \xi$（$\xi \sim \mathcal{N}(0,I)$），则：

- **VP-ODE（等价于 [[DDIM]]）**、sub-VP-ODE、VE-ODE 都是该框架的特例；
- 它们的 $\alpha_t, \beta_t$ 是非线性的（如 $\sqrt{\bar\alpha_t}, \sqrt{1-\bar\alpha_t}$），导致**轨迹弯曲、速度随时间变化**，Euler 积分产生离散误差，需要更多步；
- rectified flow 取 $\alpha_t = t, \beta_t = 1-t$ 的线性选择，恰好是其中"最直"的一条。

### 6.3 与 Flow Matching 的关系

> [!important] 同一训练目标，不同叙事
> [[FlowMatching]]（Lipman et al. 2022/2023）与 Rectified Flow（Liu et al. 2022）**独立提出**，在线性插值下训练损失完全相同（都是 CFM 损失，最优解都是 $v(z,t)=\mathbb{E}[X_1-X_0 \mid X_t=z]$）。
>
> - **Flow Matching 视角**：从条件概率路径构造边际向量场（条件 → 边际），强调路径选择的自由度；
> - **Rectified Flow 视角**：强调"因果化"、边际保持、传输成本不增、以及 **reflow 迭代拉直**（FM 没有）。

### 6.4 与 DDPM（SDE）的关系

- 分数/扩散模型的训练可以看作"因果化非光滑的随机插值"；在 SDE 情形下 $v(z,t) = \lim_{s \to t^+}\mathbb{E}[(X_s - X_t)/(s-t) \mid X_t = z]$，极限与期望不可交换，因此路径"粗糙"。
- rectified flow 坚持纯 ODE（奥卡姆剃刀）：ODE 与 SDE 可以互相转换而不改变边际分布（Song et al. 2021），所以随机噪声只应在确有需要时加入。

### 6.5 ε-prediction 视角：所有扩散模型都能"一步生成"

> [!important] 配对数据二次训练 = 让 ODE 贴近插值方程
> 以 DDPM 的插值 $X_t = \sqrt{\alpha_t}X_0 + \sqrt{1-\alpha_t}\,\varepsilon$ 为例（$X_0$ 为数据、$\varepsilon$ 为噪声）。若用配对数据二次训练（reflow），最优网络将满足：
> $$Z_t = \sqrt{\alpha_t}Z_0 + \sqrt{1-\alpha_t}\,\varepsilon^{**}_\theta(Z_t, t)$$
> 即网络学会了插值方程本身。此时采样只需**反解插值方程**：
> $$\hat{Z}_0 = \frac{Z_t - \sqrt{1-\alpha_t}\,\varepsilon^{**}_\theta(Z_t,t)}{\sqrt{\alpha_t}}$$
> 无需任何校正步骤 → **一步生成**。这对任意 ε-prediction 扩散模型（VP/VE/sub-VP ODE、扩散桥等）都成立。
>
> Rectified Flow 只是该机制在线性插值下的特例：反解插值方程恰好退化为欧拉一步。**太关注"直线"反而容易忽略加速的本质**——在 ε-prediction 视角下，RF 与 Diffusion 的定义、训练、加速完全统一（实证见 InstaFlow、Rectified Diffusion）。

### 6.6 与随机插值（Stochastic Interpolants）的关系

> [!note] 两条等价的统一路线
> - **Flow Matching**：从 $X_1 = \varphi_t(X_0, X_t)$ 出发，关注条件路径 $p(X_t \mid X_0)$（前向过程），利于 ε-prediction；
> - **随机插值 / Rectified Flow**：从 $X_t = I_t(X_0, X_1)$ 出发，直接构造速度 $v(X_t,t) = \mathbb{E}[\dot{X}_t \mid X_t]$，利于 v-prediction。
>
> 两者完全等价（FM 的线性示例即 Rectified Flow）；"速度 = 插值速度的条件期望"正是随机插值框架的核心定理。

---

## 7. 与最优传输（OT）的关系

- rectified flow **不求解** Monge OT，但每次 rectification 使所有凸传输成本单调不增，可以看作"逐步走向 OT"的简单算法。
- **c-Rectified Flow**（Liu 2022, arXiv:2209.14577）：迭代构造逼近 Monge map 的变体。
- 后续理论工作（如 Bansal et al. 2024, arXiv:2410.14949）给出了 straightness 与 Wasserstein 收敛之间的联系。

---

## 8. 应用与影响

- **InstaFlow**（Liu et al. 2024, ICLR）：把 Stable Diffusion 变成真正的**单步 T2I 模型**。关键发现：reflow 的核心作用是改善噪声-图像配对（assignment），使后续 1-step 蒸馏可行；COCO FID 23.3、每图 0.09s，训练仅 199 A100 GPU days。
- **PeRFlow**（arXiv:2405.07510）：分段（piecewise）rectified flow，把预训练扩散模型"分段拉直"，作为即插即用的加速方案。
- **现代 T2I 模型**（SD3、Flux 等）普遍采用 flow matching / rectified flow 形式的 v-prediction 训练，通常不跑 reflow，靠大规模数据与低步数调度器实现 1-4 步采样。

---

## 9. 局限与开放问题

- 论文定理 3.7（直线度 $O(1/K)$）描述的是"前 K 次中最好的一次"，实际中 2-rectified 已足够直，继续 rectification 效果几乎不变甚至变差——理论收敛速度与实际收益不完全吻合。
- 更紧的理论结果：*2-Rectifications Are Enough for Straight Flows*（arXiv:2410.14949）证明 2 次 rectification 足以达到足够直，并给出比 $S(\boldsymbol{Z})$ 更紧的上界。
- ε-prediction 的统一视角目前仅对**生成任务**（先验为高斯）成立；Rectified Flow 的优势在于还能处理**两个一般分布之间的传输**（域迁移、图生图），这一场景下 reflow 加速的收敛理论仍是空白。

---

## 10. 一句话总结

> [!abstract] Rectified Flow = 直线插值 + 最小二乘回归（因果化）+ reflow 贴近插值方程
>
> 用一个与标准监督学习无异的回归损失，统一生成建模与域迁移。注意：1-rectified 的真实 ODE 是条件期望（弯曲的），并不能一步采样；一步生成来自 reflow——把 ODE 贴近由配对点构造的直线插值方程。其理论保证（边际保持、凸成本不增、直线度 $O(1/K)$）提供了"从多步 ODE 走向单步生成"的可证明路径，且在 ε-prediction 视角下与整个扩散家族统一。
