# Diffusion Models（扩散模型）

> [!cite] 来源
> 知乎专栏《VAE、Diffusion、Flow Matching 系统讲解（二）Diffusion 扩散模型完全指南》- 冰锐

---

## 1. 核心思想

> [!note] 核心思想
> 如果知道如何把一张图片逐渐"腐蚀"成噪声，那么学会"逆转腐蚀"的过程就等于学会了生成。

**两步走**：

1. **前向过程（固定，不需要学习）**：对数据逐步加高斯噪声，$T$ 步后变成纯噪声
2. **反向过程（需要学习）**：训练神经网络逐步去噪，把纯噪声变回数据

```
前向: x_0 → x_1 → x_2 → ... → x_T    (逐步加噪，信号衰减)
反向: x_T → x_{T-1} → ... → x_0      (逐步去噪，网络预测噪声)
```

> [!info] 信噪比
> $\mathrm{SNR}(t) = \frac{\bar\alpha_t}{1-\bar\alpha_t}$，随 $t$ 增大单调递减。

---

## 2. 前向过程（加噪）

### 2.1 单步加噪

定义噪声调度 $\beta_1, \beta_2, \ldots, \beta_T$（通常 $\beta_t \in [0.0001, 0.02]$，$T=1000$）：

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}\, x_{t-1}, \beta_t I)$$

重参数化形式：

$$x_t = \sqrt{1-\beta_t}\, x_{t-1} + \sqrt{\beta_t}\, \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I)$$

定义 $\alpha_t = 1 - \beta_t$，等价形式：

$$x_t = \sqrt{\alpha_t}\, x_{t-1} + \sqrt{1-\alpha_t}\, \epsilon_t$$

> [!tip] 方差守恒
> 系数设计保证了 $(\sqrt{\alpha_t})^2 + (\sqrt{1-\alpha_t})^2 = 1$，每步加噪后方差不发散也不坍缩。

### 2.2 闭式解（核心推导）

定义 $\bar{\alpha}_t = \prod_{s=1}^{t}\alpha_s$。利用独立高斯的可加性（$a\epsilon_1 + b\epsilon_2 \sim \mathcal{N}(0, (a^2+b^2)I)$），可以一步直接算出任意时刻的 $x_t$：

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

或写为分布形式：

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}\, x_0, (1-\bar{\alpha}_t)\, I)$$

当 $t = T$ 时，$\bar{\alpha}_T \approx 0$，所以 $x_T \approx \mathcal{N}(0, I)$。

> [!important] 这是 Diffusion 中最重要的数学推导之一
> 闭式解让训练时可以随机采样任意 $t$，一步构造 $x_t$，无需逐步加噪。

---

## 3. 反向过程（核心难点）

### 3.1 问题：$q(x_{t-1}|x_t)$ 为什么不可算？

用贝叶斯定理展开：

$$q(x_{t-1}|x_t) = \frac{q(x_t|x_{t-1}) \cdot q(x_{t-1})}{q(x_t)}$$

- $q(x_t|x_{t-1})$ 是已知的高斯 ✓
- $q(x_{t-1})$ 和 $q(x_t)$ 是**边际分布**，需要对整个数据分布积分：

$$q(x_{t-1}) = \int q(x_{t-1}|x_0) \cdot q_{\text{data}}(x_0) \, dx_0$$

这是 $N$ 个高斯的**混合**（不是线性组合），是一个极其复杂的多峰分布，没有闭式表达。

> [!warning] 关键区分——高斯的"线性组合" vs "混合"
> - **线性组合** $Z = aX + bY$：对随机变量的采样值做算术运算，结果仍是高斯（如前向过程中的噪声合并）
> - **混合** $p(x) = \sum_i w_i \cdot \mathcal{N}(x; \mu_i, \sigma_i^2)$：对概率密度函数做加权平均，结果一般不是高斯（多峰）

### 3.2 突破：给定 $x_0$ 后，一切变成已知高斯

$$q(x_{t-1}|x_t, x_0) = \frac{q(x_t|x_{t-1}) \cdot q(x_{t-1}|x_0)}{q(x_t|x_0)}$$

右边三项全都是已知的高斯分布！三个已知高斯做贝叶斯运算，结果仍是高斯。

> [!tip] 本质
> 给定 $x_0$ 后，"$N$ 个高斯的混合"坍缩成了"一个确定的高斯"。

### 3.3 配方推导

通过配方（completing the square）求出后验的均值和方差：

$$\tilde\beta_t = \frac{\beta_t(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}$$

$$\tilde\mu_t = \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,x_t + \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\,x_0$$

$$q(x_{t-1}|x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t I)$$

> [!info] $\tilde\mu_t$ 的直觉
> $\tilde\mu_t$ 是 $x_t$（当前噪声图）和 $x_0$（原始数据）之间的"折中"——噪声大时（$t$ 大）更依赖 $x_0$，噪声小时更依赖 $x_t$。

### 3.4 用噪声 $\epsilon$ 替换 $x_0$（连接神经网络）

推理时没有 $x_0$。利用前向闭式解反解 $x_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon}{\sqrt{\bar\alpha_t}}$，代入 $\tilde\mu_t$：

$$\tilde\mu_t = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon\right)$$

> [!important] 核心结论
> $\tilde\mu_t$ 只依赖于 $x_t$（已知）和 $\epsilon$（唯一未知量）。训练神经网络 $\epsilon_\theta(x_t, t)$ 预测 $\epsilon$ 即可完成反向去噪。

### 3.5 完整逻辑链

> [!abstract] 反向过程逻辑链
> 1. 想要 $q(x_{t-1}|x_t)$ → **不可算**（需要整个数据分布）
> 2. 退而求其次 $q(x_{t-1}|x_t, x_0)$ → **可算**（三个已知高斯配方）
> 3. 后验均值 $\tilde\mu_t(x_t, x_0)$ → 依赖 $x_0$，推理时没有
> 4. 用 $\epsilon$ 替换 $x_0$ → $\tilde\mu_t(x_t, \epsilon)$
> 5. 训练网络预测 $\epsilon$ → $\epsilon_\theta(x_t, t) \approx \epsilon$

---

## 4. 损失函数推导

### 4.1 变分下界（VLB）

目标：最大化 $\log p_\theta(x_0)$。与 VAE 完全一致的思路——引入前向过程 $q(x_{1:T}|x_0)$ 作为"桥梁"，Jensen 不等式推出下界：

$$\log p_\theta(x_0) \geq \mathbb{E}_{q}\left[\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}\right]$$

> [!example] 对比：VAE vs Diffusion
> | 对比 | VAE | Diffusion |
> |---|---|---|
> | 隐变量 | $z$（单个向量） | $x_1, x_2, \ldots, x_T$（整条马尔可夫链） |
> | 近似后验 | $q_\phi(z\|x_0)$（需学习的 Encoder） | $q(x_{1:T}\|x_0)$（固定前向加噪，无需学习） |
> | 生成模型 | $p_\theta(x_0, z) = p_\theta(x_0\|z) \cdot p(z)$ | $p_\theta(x_{0:T}) = p(x_T) \cdot \prod p_\theta(x_{t-1}\|x_t)$ |

### 4.2 分解为逐步 KL

通过贝叶斯翻转和伸缩消去（Telescoping），VLB 分解为：

$$-\text{ELBO} = \underbrace{D_{\text{KL}}(q(x_T|x_0) \| p(x_T))}_{L_T} + \sum_{t=2}^{T} \underbrace{D_{\text{KL}}(q(x_{t-1}|x_t, x_0) \| p_\theta(x_{t-1}|x_t))}_{L_{t-1}} - \underbrace{\mathbb{E}_{q}[\log p_\theta(x_0|x_1)]}_{L_0}$$

> [!note] 各项含义
> - $L_T$：前向终点与先验的匹配，常数（$\approx 0$）
> - $L_{t-1}$：**核心项**，模型的去噪一步与真实反向后验的匹配
> - $L_0$：最终一步的重建损失

### 4.3 核心项 $L_{t-1}$ 的计算

$$L_{t-1} = D_{\text{KL}}(q(x_{t-1}|x_t, x_0) \| p_\theta(x_{t-1}|x_t))$$

- 真实后验：高斯，均值 $\tilde\mu_t(x_t, x_0)$，方差 $\tilde\beta_t$
- 模型分布：也设为高斯，方差固定为 $\tilde\beta_t$，只学均值 $\mu_\theta(x_t, t)$

> [!tip] 固定方差的简化
> 固定方差大大简化了问题（Improved DDPM 后来学了方差，但提升有限）。

两个方差相同的高斯之间的 KL 散度 = 均值差的平方：

$$L_{t-1} = \frac{1}{2\tilde\beta_t}\|\tilde\mu_t(x_t, x_0) - \mu_\theta(x_t, t)\|^2$$

### 4.4 参数化为噪声预测

将真实后验均值的 $\epsilon$ 替换为网络预测 $\epsilon_\theta$：

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t, t)\right)$$

代入后 $x_t$ 项抵消：

$$L_{t-1} = \frac{\beta_t^2}{2\tilde\beta_t \alpha_t (1-\bar\alpha_t)}\|\epsilon - \epsilon_\theta(x_t, t)\|^2$$

### 4.5 简化损失

> [!important] Ho et al. 2020 的简化损失
> 去掉时间步权重系数直接用简化损失训练效果更好：
>
> $$\mathcal{L}_{\text{simple}} = \mathbb{E}_{x_0, \epsilon \sim \mathcal{N}(0,I), t \sim U\{1,T\}}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$
>
> 其中 $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$。

> [!question] 为什么去掉权重反而更好？
> 简化损失对所有时间步均匀权重，给高噪声级别（负责全局结构）更多关注，实践中产生更好的生成质量。

---

## 5. 三种等价的预测目标

### 5.1 核心关系

一切都源于前向闭式解：$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$

给定 $x_t$（已知），$\epsilon$、$x_0$、$v$ 三者可以互相转化：

$$\epsilon = \frac{x_t - \sqrt{\bar\alpha_t}\,x_0}{\sqrt{1-\bar\alpha_t}}, \quad x_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon}{\sqrt{\bar\alpha_t}}$$

速度 $v$ 是 $\epsilon$ 和 $x_0$ 的线性组合：

$$v = \sqrt{\bar\alpha_t}\,\epsilon - \sqrt{1-\bar\alpha_t}\,x_0$$

> [!info] $v$ 的来源
> 令 $\cos\theta = \sqrt{\bar\alpha_t},\ \sin\theta = \sqrt{1-\bar\alpha_t}$，前向过程变为 $x_t = \cos\theta \cdot x_0 + \sin\theta \cdot \epsilon$，$v = \frac{dx_t}{d\theta} = -\sin\theta \cdot x_0 + \cos\theta \cdot \epsilon$，即插值路径上的切线方向。

### 5.2 等价性

$$\|x_0 - x_{0,\theta}\|^2 = \frac{1-\bar\alpha_t}{\bar\alpha_t}\,\|\epsilon - \epsilon_\theta\|^2$$

$$\|v - v_\theta\|^2 = \frac{1}{\bar\alpha_t}\,\|\epsilon - \epsilon_\theta\|^2$$

三个损失只差常数因子，有相同的最优解。

### 5.3 三种目标对比

> [!example] 三种预测目标对比
> | | $\epsilon$ 预测 | $x_0$ 预测 | $v$ 预测 |
> |---|---|---|---|
> | 网络输出 | 预测噪声 | 预测去噪后的原图 | 信号与噪声的"旋转" |
> | $t$ 小（噪声少） | 噪声信号微弱 | 接近 $x_t$，容易 | 稳定 |
> | $t$ 大（噪声多） | 接近 $x_t$，容易 | 与 $x_t$ 差异大，困难 | 稳定 |
> | 数值稳定性 | $t \to 0$ 时不稳定 | $t \to T$ 时不稳定 | **两端都稳定** |
> | 代表工作 | DDPM (Ho 2020) | DALL·E (Ramesh et al.) | Stable Diffusion v2+ |

> [!tip] $v$ 预测两端都稳定的原因
> 系数满足 $(\sqrt{\bar\alpha_t})^2 + (\sqrt{1-\bar\alpha_t})^2 = 1$，反推公式 $x_0 = \sqrt{\bar\alpha_t}\,x_t - \sqrt{1-\bar\alpha_t}\,v$ 中没有除法，采样过程不会数值爆炸。

---

## 6. 采样

### 6.1 DDPM 采样（随机，T 步）

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\,\epsilon_\theta(x_t, t)\right) + \sigma_t\, z, \quad z \sim \mathcal{N}(0, I)$$

这是一个 SDE 的离散化，每步加入新的随机噪声。必须走完所有 $T$ 步。

### 6.2 DDIM 采样（确定性，可加速）

> [!note] DDIM 核心洞察
> 反向过程不一定要是随机的马尔可夫链。唯一约束只有前向边际分布不变。

**两步走策略**：

1. 用网络估计 $\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\,\epsilon_\theta(x_t, t)}{\sqrt{\bar\alpha_t}}$
2. "重新加噪"到目标时刻：$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\,\hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}}\,\epsilon_\theta(x_t, t)$

> [!question] 为什么可以跳步？
> 公式中只出现累积系数 $\bar\alpha$，不依赖单步衰减系数 $\alpha_t$。定义子序列 $\tau = [T, T-k, T-2k, \ldots, 0]$，每步直接跳 $k$ 个时间步。

> [!tip] 增量 vs 坐标直觉
> - DDPM 公式含 $\alpha_t$、$\beta_t$（单步增量）：描述"台阶高度"，必须一级一级走
> - DDIM 公式只含 $\bar\alpha_t$、$\bar\alpha_{t-1}$（累积坐标）：描述"海拔高度"，直接传送

> [!example] DDPM vs DDIM 对比
> | | DDPM | DDIM |
> |---|---|---|
> | 随机性 | 有（每步加新噪声） | 无（确定性，可复现） |
> | 步数 | 必须 T 步 | 可跳步（如 50 步） |
> | 隐空间插值 | 不支持 | 支持（$x_T \leftrightarrow x_0$ 确定性双射） |

> [!abstract] 本质
> DDIM 去掉了随机噪声，把 SDE 转化成了 ODE。

---

## 7. Score Matching 视角

### 7.1 Score Function 定义

$$s(x) = \nabla_x \log p(x)$$

> [!info] Score Function 直觉
> 想象数据分布 $p(x)$ 是一个地形图，"海拔"代表概率密度。score function 是每一点的上坡方向——指向概率密度增大最快的方向。
>
> - 在峰值处，score $\approx 0$（已经在山顶）
> - 在低谷处，score 指向最近的峰值

> [!question] 为什么学 score 而不直接学 $p(x)$？
> 直接学 $p(x)$ 需要保证积分为 1（归一化常数 $Z$ 在高维空间极难计算），而 $\nabla_x \log p(x) = \nabla_x \log \frac{p^*(x)}{Z} = \nabla_x \log p^*(x)$，对 $x$ 求梯度时 $Z$ 直接消失。

### 7.2 预测噪声 = 学 Score

> [!important] 核心结论
> $$\epsilon_\theta(x_t, t) \approx -\sqrt{1-\bar\alpha_t}\,\nabla_{x_t}\log q(x_t)$$

推导过程：边际分布的 score = 条件 score 关于后验的期望：

$$\nabla_{x_t}\log q(x_t) = \mathbb{E}_{q(x_0|x_t)}\left[\nabla_{x_t}\log q(x_t|x_0)\right]$$

条件分布 $q(x_t|x_0)$ 是高斯，其 score 为：

$$\nabla_{x_t} \log q(x_t|x_0) = -\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}$$

而 L2 损失的最优解恰好是条件期望 $\epsilon_\theta^*(x_t, t) = \mathbb{E}[\epsilon|x_t]$，所以网络学到的就是 score。

> [!tip] 直觉
> 噪声把你推离数据，score 把你拉回数据，两者差一个负号和一个缩放因子。

### 7.3 Probability Flow ODE

Song et al. 2021 证明：对于任意 SDE，都存在一个确定性的 ODE，使得两者在每个时刻的概率分布完全相同：

$$\frac{dx}{dt} = f(x,t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)$$

> [!note] DDIM 与 PF-ODE 的联系
> DDIM 的确定性采样本质上就是在求解 Probability Flow ODE。用 PF-ODE 的优势：确定性（每次结果可复现）、可跳步加速、为连接 Flow Matching 提供桥梁。

---

## 8. 核心速记卡

> [!abstract] 一句话总结
> 前向逐步加噪（固定）→ 反向逐步去噪（学习），网络预测噪声 $\epsilon$，等价于学习 score function。

> [!abstract] 核心逻辑链
> 1. 前向闭式解：$x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$（一步到位）
> 2. 反向不可算 → 给定 $x_0$ 后可算 → 配方得闭式解
> 3. 用 $\epsilon$ 替换 $x_0$ → 训练网络预测 $\epsilon$
> 4. 损失：ELBO → VLB → MSE = $\|\epsilon - \epsilon_\theta\|^2$
> 5. 采样：DDPM（SDE，随机）或 DDIM（ODE，确定性/可加速）

> [!important] 关键公式
> - $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$ —— 前向闭式解
> - $\tilde\mu_t = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon\right)$ —— 后验均值（只用 $\epsilon$）
> - $\mathcal{L} = \|\epsilon - \epsilon_\theta(x_t, t)\|^2$ —— 简化损失
> - $\epsilon_\theta \approx -\sqrt{1-\bar\alpha_t}\,\nabla\log q(x_t)$ —— 预测噪声 = 学 score

> [!tip] 与下一章的衔接
> Diffusion 的弯曲路径导致采样慢。Flow Matching 把路径拉直，从 SDE/score 框架切换到 ODE/velocity 框架。
