---
type: Note
related_to: "[[Diffusion]]"
status: Active
url: https://arxiv.org/abs/2006.11239
code: https://github.com/hojonathanho/diffusion
---

# Denoising Diffusion Implicit Models (DDIM)

---

## Non‑Markovian Forward Process

> [!note] DDIM 的核心思想
> DDPM 的前向过程是马尔可夫链，每一步只依赖上一步。DDIM 则采用一个**非马尔可夫**的前向过程，它直接定义在给定 $x_0$ 和 $x_t$ 的条件下 $x_{t-1}$ 的分布。这样做的目的是保持边际分布 $q(x_t|x_0)$ 与 DDPM 完全一致，从而可以复用同一个训练好的噪声预测网络。

### 边际分布（与 DDPM 相同）

$$
q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)\mathbf{I})
$$

- $\alpha_t = 1-\beta_t$，$\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$。
- $\beta_t$ 为噪声调度，通常与 DDPM 中的定义一致。

### 条件后验（设计自由度）

> [!important] 条件后验分布族
> DDIM 定义一族分布 $q_\sigma(x_{t-1}|x_t, x_0)$，使得联合分布 $q_\sigma(x_{1:T}|x_0) = q_\sigma(x_T|x_0)\prod_{t=2}^T q_\sigma(x_{t-1}|x_t, x_0)$ 的边缘 $q_\sigma(x_t|x_0)$ 依然是 $\mathcal{N}(\sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)\mathbf{I})$。

对于任意实数 $\sigma_t \ge 0$，该后验定义为：

$$
q_\sigma(x_{t-1}|x_t, x_0) = \mathcal{N}\left(x_{t-1}; \sqrt{\bar{\alpha}_{t-1}}x_0 + \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2}\,\frac{x_t - \sqrt{\bar{\alpha}_t}x_0}{\sqrt{1-\bar{\alpha}_t}},\; \sigma_t^2\mathbf{I}\right)
$$

- **均值**：由 $x_0$ 的预测和 $x_t$ 到 $x_0$ 的方向共同决定。
- **方差**：$\sigma_t^2\mathbf{I}$ 是一个自由参数，控制前向过程的随机性。

直接计算可验证，若 $x_t \sim q(x_t|x_0)$，则上述 $x_{t-1}$ 的边际分布确为 $q(x_{t-1}|x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_{t-1}}x_0, (1-\bar{\alpha}_{t-1})\mathbf{I})$。

> [!tip] 特例
> 当取
> $$\sigma_t = \sqrt{\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}}\sqrt{1-\frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}}$$
> 时，该过程退化为 DDPM 的马尔可夫前向过程。

---

## Reverse Process

> [!note] 生成过程
> DDIM 的生成过程（反向过程）直接从 $p(x_T)=\mathcal{N}(x_T;0,\mathbf{I})$ 开始，按以下步骤迭代去噪：

$$
p_\theta^{(t)}(x_{t-1}|x_t) =
\begin{cases}
\mathcal{N}\left(\mu_\theta(x_t, t), \sigma_t^2\mathbf{I}\right) & \text{if } t > 1 \\
\mathrm{deterministic\;given\;} x_1 & \text{if } t = 1
\end{cases}
$$

其中均值 $\mu_\theta(x_t, t)$ 利用网络预测的噪声 $\epsilon_\theta(x_t, t)$ 来定义：

首先由 $x_t$ 和 $\epsilon_\theta$ 得到对 $x_0$ 的估计：
$$
\hat{x}_0(x_t, t) = \frac{1}{\sqrt{\bar{\alpha}_t}}\big(x_t - \sqrt{1-\bar{\alpha}_t}\,\epsilon_\theta(x_t, t)\big)
$$

然后代入后验均值表达式，得到：
$$
\mu_\theta(x_t, t) = \sqrt{\bar{\alpha}_{t-1}}\,\hat{x}_0(x_t, t) + \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2}\,\epsilon_\theta(x_t, t)
$$

> [!important] 一步采样公式
> $$
> x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\underbrace{\left(\frac{x_t - \sqrt{1-\bar{\alpha}_t}\,\epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}\right)}_{\text{predicted }x_0} + \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2}\,\epsilon_\theta(x_t, t) + \sigma_t z
> $$
> 其中 $z \sim \mathcal{N}(0,\mathbf{I})$ 仅在 $t>1$ 且 $\sigma_t > 0$ 时添加。

---

## Deterministic DDIM

> [!important] 确定性 DDIM
> 当所有 $\sigma_t = 0$ 时，反向过程变为**完全确定性的**：
>
> $$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\,\hat{x}_0(x_t, t) + \sqrt{1-\bar{\alpha}_{t-1}}\,\epsilon_\theta(x_t, t)$$

- 初始噪声 $x_T$ 和模型参数 $\theta$ 唯一决定了生成结果 $x_0$。
- 这种确定性特性使 DDIM 拥有类似于 GAN 的隐空间插值性质：在 $x_T$ 之间做插值，生成图像会在语义上平滑过渡。

---

## Accelerated Sampling

> [!note] 加速采样原理
> 因为 DDIM 的前向边缘分布对任意子序列 $\tau = \{\tau_1, \tau_2, \dots, \tau_S\}$（$\tau_1 < \tau_2 < \dots < \tau_S = T$）也保持一致，所以可用这些时间步直接进行生成，大幅减少迭代次数。

### 子序列生成公式

给定 $x_{\tau_s}$，一步生成更早的子步骤：
$$
x_{\tau_{s-1}} = \sqrt{\bar{\alpha}_{\tau_{s-1}}}\,\hat{x}_0(x_{\tau_s}, \tau_s) + \sqrt{1-\bar{\alpha}_{\tau_{s-1}}-\sigma_{\tau_s}^2}\,\epsilon_\theta(x_{\tau_s}, \tau_s) + \sigma_{\tau_s} z
$$

- 只需 $S \ll T$ 步，显著加快采样速度（如原论文中 $T=1000$，$S=10$ 步即可得到高质量样本）。
- 方差项 $\sigma_{\tau_s}$ 可沿用通用定义或设为 0（确定性子序列采样）。

---

## Training (Loss)

> [!tip] 训练策略
> DDIM **不重新训练**网络，而是直接复用 DDPM 预训练好的噪声预测模型 $\epsilon_\theta$。因此训练目标与 DDPM 完全相同。

### Simplified DDPM Loss

$$
L_{\text{simple}} = \mathbb{E}_{x_0, \epsilon, t}\Big[\|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon,\, t)\|^2\Big]
$$

- $x_0$ 为真实数据，$\epsilon \sim \mathcal{N}(0,\mathbf{I})$，$t \sim \mathrm{Uniform}(1,\dots,T)$。
- DDIM 的推导保证了只要边际 $q(x_t|x_0)$ 与 DDPM 一致，训练目标就无需改变。

---

## 总结

### 一、前向过程（非马尔可夫）

#### 1. 边际分布

$$
q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)\mathbf{I})
$$

> [!note] 与 DDPM 完全一致
> 保证可复用 DDPM 训练的网络。

#### 2. 后验分布族

$$
q_\sigma(x_{t-1}|x_t, x_0) = \mathcal{N}\left(x_{t-1}; \sqrt{\bar{\alpha}_{t-1}}x_0 + \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2}\,\frac{x_t - \sqrt{\bar{\alpha}_t}x_0}{\sqrt{1-\bar{\alpha}_t}},\; \sigma_t^2\mathbf{I}\right)
$$

- $\sigma_t$ 决定随机性；$\sigma_t=0$ 时为确定性的隐式过程，$\sigma_t=\tilde{\beta}_t$ 时恢复 DDPM。

### 二、反向过程（生成）

#### 3. 单步去噪（含重参数化）

$$
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\,\hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2}\,\epsilon_\theta(x_t, t) + \sigma_t z
$$
其中
$$
\hat{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}\big(x_t - \sqrt{1-\bar{\alpha}_t}\,\epsilon_\theta(x_t, t)\big)
$$

- 当 $\sigma_t=0$ 时过程确定，给定 $x_T$ 唯一决定生成结果。

#### 4. 加速采样

- 选择子序列 $\tau \subset \{1,\dots,T\}$，用相同公式以 $S$ 步生成。
- 生成速度提升 $T/S$ 倍，质量几乎不下降。

### 三、训练

- 完全沿用 DDPM 的简化损失 $L_{\text{simple}}$，无需额外训练。
- DDIM 是一种**采样方法**，对预训练扩散模型即插即用。

> [!abstract] DDIM 核心优势
> 1. **确定性生成**：$\sigma_t=0$ 时隐空间连续，可支持语义插值。
> 2. **加速采样**：通过子序列步数可压缩到原来的 10%–5% 仍能保持高质量。
> 3. **兼容性**：无需修改训练过程，直接应用在 DDPM 模型上即可获得上述改进。
