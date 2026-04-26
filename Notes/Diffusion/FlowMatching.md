# Flow Matching（流匹配）

> 来源：知乎专栏《VAE、Diffusion、Flow Matching 系统讲解（三）Flow Matching 流匹配》- 冰锐

## 1. 核心思想

### 从 Diffusion 到 Flow Matching 的动机

Diffusion 模型效果很好，但存在两个根本性的遗留问题：

1. **噪声调度需要精心设计**：$\beta_t$ 的变化曲线（linear、cosine 等）对生成质量影响很大，但没有明确的理论指导——全靠经验调参
2. **弯曲的概率路径导致采样慢**：$\sqrt{\bar\alpha_t}$ 和 $\sqrt{1-\bar\alpha_t}$ 都是 $t$ 的非线性函数，从噪声到数据的路径是弯曲的。ODE solver（如 Euler 方法）积分时，弯曲路径的一阶近似误差大，需要更多步

**Flow Matching 的核心直觉**：既然弯曲是问题的根源，那就直接走直线。

### 形式化

学一个向量场（速度场）$v_\theta(x, t)$，定义连续的 ODE：

$$\frac{dx}{dt} = v_\theta(x, t), \quad t \in [0, 1]$$

- $t=0$：噪声 $x_0 \sim \mathcal{N}(0, I)$
- $t=1$：数据 $x_1 \sim p_{\text{data}}$

训练好后，从噪声出发沿 ODE 积分到 $t=1$，得到的 $x_1$ 就是生成的数据。

### 记号对照（重要）

| | Diffusion 记号 | Flow Matching 记号 |
|---|---|---|
| 数据 | $x_0$ | $x_1$ |
| 噪声 | $x_T$（或 $\varepsilon$） | $x_0$ |
| 时间方向 | $t: 0 \to T$（数据→噪声） | $t: 0 \to 1$（噪声→数据） |

**后文中 $x_0$ = 噪声，$x_1$ = 数据。**

### 与 Diffusion 的本质区别

Diffusion 先设计好前向/反向过程（SDE），再从中推导训练目标。Flow Matching 直接从"学一个 ODE 的速度场"出发，跳过了 SDE 的所有复杂机制（马尔可夫链、反向后验、变分下界），整个框架从设计到训练都更直接。

## 2. 连续正规化流（CNF）数学基础

### 2.1 流映射

给定向量场 $v_t(x)$，流映射 $\phi_t: \mathbb{R}^d \to \mathbb{R}^d$ 定义为：

$$\frac{d\phi_t(x)}{dt} = v_t(\phi_t(x)), \quad \phi_0(x) = x$$

**直觉**：把每个点 $x$ 想象成一个粒子。$\phi_t(x)$ 是粒子在时间 $t$ 的位置。在每个时刻，粒子按照当前位置处的速度 $v_t$ 移动。

**关键性质**：在温和条件下（$v_t(x)$ 关于 $x$ 是 Lipschitz 连续的），流映射是一一映射（双射）——不同的起点一定到达不同的终点，粒子永远不会"撞到一起"。这由 Picard-Lindelöf 定理保证。

**为什么"不撞到一起"很重要？** 如果两个不同的噪声点在流动过程中撞到同一位置，信息丢失，映射不可逆——意味着无法从图像反推回噪声（没有 latent space），概率密度计算也会出问题。

### 2.2 连续性方程

大量粒子同时按照 $v_t$ 流动，概率密度 $p_t(x)$ 的演化满足连续性方程：

$$\frac{\partial p_t(x)}{\partial t} + \nabla \cdot (p_t(x)\, v_t(x)) = 0$$

- $\nabla \cdot \vec{F} = \sum_i \frac{\partial F_i}{\partial x_i}$：散度，衡量向量场在某点是"发散"还是"汇聚"
- 散度为正（流出 > 流入）→ 密度下降；散度为负（流入 > 流出）→ 密度上升
- 这是质量守恒的体现：粒子不会凭空出现或消失

**核心推论**：给定初始分布 $p_0$（噪声）和向量场 $v_t$，$p_t$ 在每个时刻都被唯一确定。如果 $v_t$ 选得好，$p_1$ 就等于 $p_{\text{data}}$——这就是 Flow Matching 的目标。

### 2.3 梯度 vs 散度速查

| 运算 | 符号 | 输入 → 输出 | 含义 | 本文中的出现 |
|---|---|---|---|---|
| 梯度 | $\nabla f$ | 标量 → 向量 | 函数变化最快的方向 | score function $\nabla \log p$ |
| 散度 | $\nabla \cdot \vec{F}$ | 向量 → 标量 | 向量场在某点"发散"还是"汇聚" | 连续性方程 $\nabla \cdot (p_t v_t)$ |

## 3. 为什么不直接匹配边际向量场？

### 3.1 原始 Flow Matching 目标

最直接的想法：用 L2 损失让 $v_\theta$ 去拟合"真实的"边际速度场 $u_t(x)$：

$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{t \sim U[0,1],\, x \sim p_t}\left[\|v_\theta(x, t) - u_t(x)\|^2\right]$$

| | DDPM | Flow Matching |
|---|---|---|
| 网络预测 | 噪声 $\epsilon_\theta(x_t, t)$ | 速度 $v_\theta(x, t)$ |
| 拟合目标 | 真实噪声 $\epsilon$ | 真实速度 $u_t(x)$ |
| 损失函数 | $\|\epsilon - \epsilon_\theta\|^2$ | $\|u_t - v_\theta\|^2$ |

### 3.2 为什么不可用

$u_t(x)$ 依赖于整个数据分布 $p_{\text{data}}$，无法从有限样本中直接计算：

$$p_t(x) = \int p_t(x|x_1)\,p_{\text{data}}(x_1)\,dx_1$$

需要对整个数据分布积分——和 Diffusion 中 $q(x_t) = \int q(x_t|x_0)q_{\text{data}}(x_0)dx_0$ 的困难完全一样。

**与 Diffusion 的完美平行**：

- Diffusion：$q(x_{t-1}|x_t)$ 不可算 → "给定 $x_0$"后变成可算的 $q(x_{t-1}|x_t, x_0)$
- Flow Matching：$u_t(x)$ 不可算 → "给定 $x_1$"后变成可算的 $u_t(x|x_1)$

**一句话**：边际不可算 → 加条件后可算 → 用网络学那个可算的量（三个模型共用的母题）。

## 4. Conditional Flow Matching（核心突破）

### 4.1 关键洞察

Lipman et al. (2023) 的核心发现：不需要知道边际向量场 $u_t(x)$！可以用条件向量场来等价训练。

### 4.2 条件概率路径

给定一对样本（$x_0 \sim \mathcal{N}(0,I)$，$x_1 \sim p_{\text{data}}$），最简单的选择——线性插值：

$$x_t = (1-t)\,x_0 + t\,x_1$$

这是从 $x_0$ 到 $x_1$ 最短、最简单的路径，完全笔直。

对应的条件分布：

$$p_t(x | x_1) = \mathcal{N}(x;\, t\,x_1,\, (1-t)^2 I)$$

验证边界：$t=0$ 时 $p_0 = \mathcal{N}(0, I)$（噪声），$t=1$ 时 $p_1 = \delta(x - x_1)$（数据点）。

### 4.3 条件向量场

对 $x_t = (1-t)x_0 + tx_1$ 两端对 $t$ 求导：

$$\frac{dx_t}{dt} = x_1 - x_0$$

这就是条件向量场 $u_t(x|x_1) = x_1 - x_0$——**一个不依赖于 $t$ 的常数向量！**

每个粒子从噪声点出发，以恒定速度沿直线匀速运动到数据点。没有加速度、没有弯曲。

**和 Diffusion 的对比**：Diffusion 的条件向量场是 $\dot{\sqrt{\bar\alpha_t}}x_1 + \dot{\sqrt{1-\bar\alpha_t}}x_0$（随时间变化），FM 的条件向量场是常数 $x_1 - x_0$（极度简化）。

### 4.4 CFM 损失

$$\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t \sim U[0,1],\, x_0 \sim \mathcal{N}(0,I),\, x_1 \sim p_{\text{data}}}\left[\|v_\theta(x_t, t) - (x_1 - x_0)\|^2\right]$$

其中 $x_t = (1-t)x_0 + tx_1$。

**训练步骤对比**：

| 步骤 | Flow Matching (CFM) | Diffusion (DDPM) |
|---|---|---|
| 采样 | $x_0 \sim \mathcal{N}(0,I)$，$x_1$ 为数据 | $x_0$ 为数据，$\epsilon \sim \mathcal{N}(0,I)$ |
| 选时间 | $t \sim U[0,1]$ | $t \sim U\{1,...,T\}$ |
| 造输入 | $x_t = (1-t)x_0 + tx_1$ | $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$ |
| 损失 | $\|v_\theta(x_t, t) - (x_1 - x_0)\|^2$ | $\|\epsilon_\theta(x_t, t) - \epsilon\|^2$ |

两者的结构惊人地相似——都是"造一个中间点 $x_t$，让网络预测某个目标"。区别在于插值方式不同（线性 vs $\sqrt{\bar\alpha_t}$ 加权）和预测目标不同（速度 vs 噪声）。**Flow Matching 的插值和目标都更简单。**

### 4.5 为什么 CFM 等价于 FM？（核心定理）

**定理（Lipman et al. 2023）**：$\nabla_\theta \mathcal{L}_{\text{CFM}} = \nabla_\theta \mathcal{L}_{\text{FM}}$

虽然两个损失的值不同，但对网络参数 $\theta$ 的梯度完全相同。关键在于展开 $\|v_\theta - \text{target}\|^2 = \|v_\theta\|^2 - 2\langle v_\theta, \text{target}\rangle + \|\text{target}\|^2$：

| 项 | FM | CFM | 相同？ |
|---|---|---|---|
| $\mathbb{E}[\|v_\theta\|^2]$ | $x$ 来自 $p_t$ | $x_t$ 来自条件采样后插值 | ✓（两种采样等价） |
| $\mathbb{E}[\langle v_\theta, \text{目标}\rangle]$ | 目标是 $u_t(x)$（平均速度） | 目标是 $u_t(x\|x_1)$（个体速度） | ✓（个体平均 = 平均） |
| $\mathbb{E}[\|\text{目标}\|^2]$ | $\|u_t(x)\|^2$ | $\|x_1-x_0\|^2$ | ✗（但此项不含 $\theta$） |

第三项虽然不同，但求梯度时消失（与 $\theta$ 无关）。所以梯度完全一致。

**核心直觉**：L2 损失的最优解是条件期望。每次训练用一个个体的速度 $x_1-x_0$，但 L2 损失会自动让网络学到所有个体速度的平均——这就是边际速度场。和 Diffusion 中"训练预测单个 $\epsilon$，自动学到 $\mathbb{E}[\epsilon|x_t]$"完全一样的道理。

## 5. 与 Diffusion 的数学联系

### 5.1 统一框架：概率路径

两者都可以看作定义了一条概率路径 $p_t(x)$，核心区别在于路径的"形状"：

| | Diffusion | Flow Matching |
|---|---|---|
| 概率路径 | $p_t(x) = \int \mathcal{N}(x; \sqrt{\bar\alpha_t}x_1, (1-\bar\alpha_t)I) q(x_1) dx_1$ | $p_t(x) = \int \mathcal{N}(x; tx_1, (1-t)^2I) q(x_1) dx_1$ |
| 条件插值 | $x_t = \sqrt{\bar\alpha_t}x_1 + \sqrt{1-\bar\alpha_t}x_0$ | $x_t = (1-t)x_0 + tx_1$ |
| 路径形状 | **弯曲**（$\sqrt{\bar\alpha_t}$ 是 $t$ 的非线性函数） | **直线**（系数 $(1-t)$ 和 $t$ 是线性的） |
| 学习目标 | 噪声 $\epsilon$（等价于 score） | 速度 $v = x_1 - x_0$ |
| 时间范围 | $t \in \{0, 1, ..., T\}$（离散） | $t \in [0, 1]$（连续） |

**"弯曲"的来源**：$\bar\alpha_t = \prod_{s=1}^t \alpha_s$ 本身是连乘积（非线性），再开根号后 $\sqrt{\bar\alpha_t}$ 更加非线性。

### 5.2 路径曲率的影响

当用数值 ODE solver（如 Euler 方法）采样时：

$$x_{t+\Delta t} \approx x_t + v(x_t, t) \cdot \Delta t$$

- **直线路径**：速度确实不变（$v = x_1 - x_0$ 是常数），Euler 方法可以精确积分，理论 1 步就够
- **弯曲路径**：速度随 $t$ 变化，Euler 近似产生截断误差，步数太少误差累积

**这就是 Flow Matching 能用更少步数的根本原因**：直线路径对 ODE solver 更友好，5-20 步就能达到 Diffusion 需要 20-50 步的质量。

### 5.3 Diffusion 是 Flow Matching 的特例

如果在 FM 框架下选择 Diffusion 的插值方式：

$$x_t = \sqrt{\bar\alpha_t}x_1 + \sqrt{1-\bar\alpha_t}x_0$$

条件向量场变为：

$$u_t(x|x_1) = \frac{d\sqrt{\bar\alpha_t}}{dt}x_1 + \frac{d\sqrt{1-\bar\alpha_t}}{dt}x_0$$

这是时间相关的（不像线性 FM 的常数速度）。用这个向量场做 FM 训练，得到的模型与 Diffusion 的 Probability Flow ODE 完全等价。

**结论**：Diffusion 是 Flow Matching 在弯曲路径下的特例。Flow Matching 通过选择更优的路径（直线），在同一框架内获得更好的采样效率。

## 6. 路径选择

Flow Matching 框架的强大之处：路径的选择是自由的。

| 路径类型 | 插值公式 | 条件向量场 | 特点 |
|---|---|---|---|
| 线性（默认） | $x_t = (1-t)x_0 + tx_1$ | $x_1 - x_0$（常数） | 最简单，路径笔直，Euler 友好 |
| 高斯（等价 Diffusion） | $x_t = \sqrt{\bar\alpha_t}x_1 + \sqrt{1-\bar\alpha_t}x_0$ | $\dot{\sqrt{\bar\alpha_t}}x_1 + \dot{\sqrt{1-\bar\alpha_t}}x_0$（时变） | 路径弯曲，与 DDPM PF-ODE 等价 |
| 最优传输（OT） | 通过 OT 配对 $(x_0, x_1)$ 后做线性插值 | $x_1 - x_0$（配对后的常数） | 路径不交叉，向量场最平滑，采样最高效 |

## 7. 采样过程

训练好 $v_\theta$ 后，采样就是求解 ODE $\frac{dx}{dt} = v_\theta(x, t)$ 从 $t=0$ 到 $t=1$。

### 7.1 Euler 方法（最简单）

把 $[0, 1]$ 均匀分成 $N$ 步，步长 $\Delta t = 1/N$：

$$x_{t+\Delta t} = x_t + v_\theta(x_t, t) \cdot \Delta t$$

从 $x_0 \sim \mathcal{N}(0, I)$ 开始，迭代 $N$ 步得到 $x_1$。$N$ 越大越精确但越慢。线性 FM 通常 $N = 5 \sim 20$ 就够了。

### 7.2 Midpoint 方法（二阶）

```
k = v_θ(x_t, t)
x_mid = x_t + k · Δt/2          （先走半步探路）
v_mid = v_θ(x_mid, t + Δt/2)    （在中点重新评估）
x_{t+Δt} = x_t + v_mid · Δt     （用中点速度走完整步）
```

每步需要 2 次网络推理，但精度从 $O(1/N)$ 提升到 $O(1/N^2)$，实际中通常已足够。

**与 DDIM 的对比**：DDIM 本质上也是在求解 ODE（Probability Flow ODE）。Flow Matching 的优势在于 ODE 的"路径更直"，用同样阶数的 solver 需要更少步数。

## 8. Optimal Transport Flow Matching

### 8.1 路径交叉问题

在 CFM 中随机配对 $(x_0, x_1)$，不同样本的路径可能在中间时刻交叉。在交叉点处，向量场同时指向不同方向——网络被迫输出平均（接近零向量），生成质量下降。

### 8.2 OT-CFM 的解决方案

在 mini-batch 内用最优传输找到总搬运代价最小的配对：

$$\pi^* = \arg\min_\pi \sum_{i,j} \pi_{ij}\,\|x_0^{(i)} - x_1^{(j)}\|^2$$

其中 $\pi_{ij} \in \{0, 1\}$ 是配对矩阵，每个 $x_0^{(i)}$ 恰好配一个 $x_1^{(j)}$。

**直觉**：OT 配对就像"搬家公司调度"——每辆车分配到最近的目的地，总路程最短。最优配对一定没有交叉（如有交叉，交换目的地总路程变短，矛盾）。

**效果**：向量场更平滑、采样步数更少、FID 等指标优于普通 CFM。代价是每个 mini-batch 需要解离散 OT 问题（匈牙利算法，$O(B^3)$）。

## 9. 特点

| 优势 | 劣势 |
|---|---|
| 采样快（路径直，5-20 步） | OT 配对在高维空间只是近似 |
| 训练简单（只需一个 L2 回归损失） | 生态尚在发展，社区资源不如 Diffusion |
| 理论优雅（基于 CNF + 最优传输） | classifier-free guidance 等最佳实践还在探索 |
| 无噪声调度（线性插值天然有效） | |
| 统一性强（Diffusion 是它的特例） | |

## 10. 应用

| 领域 | 代表工作 | 说明 |
|---|---|---|
| 图像生成 | Stable Diffusion 3, Flux | 用 FM 替代 Diffusion ODE 采样，更快更好 |
| 语音合成 | Meta Voicebox | 非自回归语音生成，速度远快于自回归方法 |
| 文本生成 | Flow Matching for text | 用连续流替代离散去噪 |
| 视频生成 | 多个新工作 | 利用 FM 快速采样优势处理高维视频数据 |

## 11. 核心速记卡

**一句话总结**：直线插值 $x_t = (1-t)x_0 + tx_1$、常数速度 $v = x_1 - x_0$、不需要噪声调度——这就是 Flow Matching 的全部。

**核心逻辑链**：

1. 目标：学一个速度场 $v_\theta(x, t)$ 驱动 ODE，把噪声分布变成数据分布
2. 直接匹配边际速度场不可行（依赖整个 $p_{\text{data}}$，和 Diffusion 困难一样）
3. 核心突破：给定一对 $(x_0, x_1)$，定义线性路径 → 条件速度 $= x_1 - x_0$（常数！）
4. CFM 损失：$\|v_\theta(x_t, t) - (x_1 - x_0)\|^2$ → L2 损失自动学到平均（边际速度）
5. 采样：ODE 积分 $x_{t+\Delta t} = x_t + v_\theta \cdot \Delta t$，5-20 步即可

**关键公式**：

- $x_t = (1-t)x_0 + tx_1$ —— 线性插值（路径笔直）
- $u_t(x|x_1) = x_1 - x_0$ —— 条件速度场（常数，不依赖 $t$）
- $\mathcal{L}_{\text{CFM}} = \|v_\theta(x_t, t) - (x_1 - x_0)\|^2$ —— CFM 损失

**和 Diffusion 的核心区别**：

- Diffusion：弯曲路径 → 需要噪声调度、更多采样步数
- Flow Matching：直线路径 → 不需要调度、ODE solver 更友好、步数更少

**为什么 CFM = FM？** L2 损失的最优解是条件期望。每次用个体速度训练，自动学到所有个体速度的平均 = 边际速度场。和 Diffusion 中"训练预测单个 $\epsilon$，自动学到 $\mathbb{E}[\epsilon|x_t]$"完全一样。
