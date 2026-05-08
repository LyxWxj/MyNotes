---
type: Note
related_to: "[[Diffusion]]"
status: Active
url: https://arxiv.org/abs/2010.02502
code: https://github.com/hojonathanho/diffusion
---

# Denoising Diffution Probabilistic Models (DDPM)

## Forward Process

前向传播的过程对输入数据$x_0 \sim q(x_0)$添加噪声，执行$T$个时间步
$$
\begin{aligned}
q(x_t|x_{t-1}) &= \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t\mathbf{I})\\
q(x_{1:T}|x_0) &= \prod_{t=1}^T q(x_t|x_{t-1})
\end{aligned}
$$
$\mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1})$表示高斯分布$\mathcal{N}(\sqrt{1-\beta_t}x_{t-1})$在$x_t$处的概率密度函数值
$\beta_1,...,\beta_T$是预先定义的噪声调度，
我们可以通过下列公式直接从$x_0$采样任意时间步的$x_t$：
$$
q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)\mathbf{I})
$$
其中$\alpha_t = 1-\beta_t$，$\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$。

## Reverse Process

反向过程是一个从$p(x_T)=\mathcal{N}(x_T; 0, \mathbf{I})$开始逐步去噪$T$步的过程
$$
\begin{aligned}
p_\theta(x_{t-1}|x_t) &= \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))\\
p_\theta(x_{0:T}) &= p(x_T)\prod_{t=1}^T p_\theta(x_{t-1}|x_t)\\
p_\theta(x_0) &= \int p_\theta(x_{0:T})dx_{1:T}
\end{aligned}
$$
$\theta$是需要训练的参数。

## Loss

我们优化根据似然的负对数ELBO(根据Jenson‘s inequality).

### Jensen 不等式（凹函数）

对于凹函数 $\psi$（如 $\log$）：
$$
\psi(\mathbb{E}[X]) \ge \mathbb{E}[\psi(X)]
$$

### ELBO（证据下界）

$$
\log p(x) \ge \text{ELBO}(q) := \mathbb{E}_{q(z)}\left[\log \frac{p(x,z)}{q(z)}\right]
$$

- $p(x)$：边际似然（证据），难直接计算
- $q(z)$：变分分布，近似后验 $p(z|x)$
- 最大化 ELBO ⇔ 最小化 $D_{KL}(q(z)\|p(z|x))$

### 两者关系

由 Jensen 不等式（$\log$ 为凹函数）：
$$
\log p(x) = \log \mathbb{E}_q\left[\frac{p(x,z)}{q(z)}\right] \ge \mathbb{E}_q\left[\log \frac{p(x,z)}{q(z)}\right] = \text{ELBO}
$$

损失：
$$
\begin{aligned}
L &= \mathbb{E}_q[-\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}]\\
&= \mathbb{E}_q[-\log p_\theta(x_T) - \sum_{t=1}^T \log \frac{p_\theta(x_{t-1}|x_t)}{q(x_{t-1}|x_t,x_0)}]\\
&= \mathbb{E}_q[-\log \frac{p(x_T)}{q(x_T|x_0)} - \sum_{t=2}^T \log \frac{p_\theta(x_{t-1}|x_t)}{q(x_{t-1}|x_t,x_0)}-\log p_\theta(x_0|x1)]\\
&=\mathbb{E}_q[D_{KL}(q(X_T|x_0)\|p(x_T)) + \sum_{t=2}^T D_{KL}(q(x_{t-1}|x_t,x_0)\|p_\theta(x_{t-1}|x_t))]
\end{aligned}
$$
因为$\beta$是常量，所以$D_{KL}(q(x_T|x_t)\|p(x_T))$是常量，所以实际上这个损失在衡量从每个时间步$t-1$的反向过程的概率分布的KL散度$L_{t-1}$

### Computing Loss

$L_{t-1} = D_{KL}(q(x_{t-1}|x_t,x_0)\|p_\theta(x_{t-1}|x_t))$，q是理想的后验分布，p是模型预测的分布，

首先$q(x_{t-1}|x_t,x_0)$的计算：
$$
\begin{aligned}
q(x_{t-1}|x_t,x_0)&=\mathcal{N}(x_{t-1};\tilde{\mu}_t(x_t,x_0),\tilde{\beta}_t\mathbf{I})\\
\tilde {\mu}_t(x_t,x_0)&=\frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t x_0+\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})x_t}{1- \bar{\alpha}_t}\\
\tilde{\beta_t}&=\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t
\end{aligned}
$$

在论文中$\Sigma_\theta(x_t, t)=\sigma_t^2\mathbf{I}$,   $\sigma^2_t$被设置为常量$\beta_t$或$\tilde{\beta}_t$。

另一方面：
$$
p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1};\mu_\theta(x_t,t), \sigma_t^2\mathbf{I})
$$
对于给定的噪声$\epsilon \sim \mathcal{N}(0, \mathbf{I})$,using $q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)\mathbf{I})$
$$
\begin{aligned}
x_t(x_0,\epsilon) &= \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon\\
x_0&=\frac{1}{\sqrt{\bar{\alpha}_t}}(x_t(x_0,\epsilon)-\sqrt{1-\bar{\alpha}_t}\epsilon)
\end{aligned}
$$
两个分布具有相同的方差，KL散度可以化简为均值的距离。
$$
\begin{aligned}
L_{t-1} &= D_{KL}(q(x_{t-1}|x_t,x_0)\|p_\theta(x_{t-1}|x_t))\\
&= \mathbb{E}_q[\frac{1}{2\sigma_t^2}\|\tilde{\mu}_t(x_t,x_0)-\mu_\theta(x_t,t)\|^2]\\  
&=E_{x_0,\epsilon}[\frac{1}{2\sigma_t^2}\|\frac{1}{\sqrt{\bar{\alpha}_t}}(x_t(x_0,\epsilon)-\ \frac{\beta_t}{\sqrt{1-\bar {\alpha}_t}}\epsilon)-\mu_\theta(x_t(x_0,\epsilon),t)\|^2]
\end{aligned}
$$
然后用一个预测噪音的模型来重参数化$\mu_\theta$
$$
\begin{aligned}
\mu_\theta(x_t,t) &= \tilde{\mu}(x_t, \frac{1}{\sqrt{\hat{\alpha}_t}}(x_t - \sqrt{1-\hat{\alpha}_t}\epsilon_\theta(x_t,t)))\\
&= \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t,t))
\end{aligned}
$$
$\epsilon_\theta$是从$(x_t,t)$预测的噪声

然后得到

$$
L_{t-1} = \mathbb{E}_{x_0, \epsilon}[\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar{\alpha}_t)}\|\epsilon -\epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon,t)\|]
$$
因此，其实我们在预测噪声

### Simplified Loss

简化系数
$$
L_{t-1} = \mathbb{E}_{x_0, \epsilon}[\|\epsilon -\epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon,t)\|^2]
$$

## 总结

---

### 一、前向过程（Forward Process）

#### 1. 单步转移核

$$
q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t\mathbf{I})
$$

- **含义**：在已知 $x_{t-1}$ 的条件下，$x_t$ 服从高斯分布。
- **均值**：$\sqrt{1-\beta_t}\,x_{t-1}$（保留上一时刻信息，略为衰减）
- **方差**：$\beta_t\mathbf{I}$（添加各向同性高斯噪声）
- $\beta_t$：预先定义的噪声强度（通常随 $t$ 增大而增大）

#### 2. 完整前向路径的联合分布

$$
q(x_{1:T}|x_0) = \prod_{t=1}^T q(x_t|x_{t-1})
$$

- **含义**：给定初始数据 $x_0$，整个噪声序列 $x_1,\dots,x_T$ 的联合概率。
- **马尔可夫性质**：每一步只依赖上一步，所以联合分布等于各条件概率的乘积。

#### 3. 任意时间步的边际分布（重参数化）

$$
q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)\mathbf{I})
$$

- **含义**：直接从 $x_0$ 采样 $x_t$，无需迭代 $t$ 步。
- 定义 $\alpha_t = 1-\beta_t$，$\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$。
- **推导**：利用高斯分布的叠加性质，累积的噪声方差为 $1-\bar{\alpha}_t$。

---

### 二、反向过程（Reverse Process）

#### 4. 单步去噪分布

$$
p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
$$

- **含义**：从噪声 $x_t$ 恢复更干净的 $x_{t-1}$，均值和方差由神经网络 $\theta$ 预测。
- 通常将 $\Sigma_\theta$ 固定为常数（如 $\beta_t$ 或 $\tilde{\beta}_t$），只学习均值。

#### 5. 反向过程的联合分布

$$
p_\theta(x_{0:T}) = p(x_T)\prod_{t=1}^T p_\theta(x_{t-1}|x_t)
$$

- **含义**：从先验 $p(x_T)=\mathcal{N}(0,\mathbf{I})$ 开始，逐步去噪生成整个序列。

#### 6. 生成的边际分布

$$
p_\theta(x_0) = \int p_\theta(x_{0:T})\, dx_{1:T}
$$

- **含义**：对中间变量积分，得到生成数据 $x_0$ 的概率（实际计算时通过采样近似）。

---

### 三、损失函数（ELBO 推导）

#### 7. ELBO 的一般形式（变分推断）

$$
\log p(x) \ge \text{ELBO}(q) := \mathbb{E}_{q(z)}\left[\log \frac{p(x,z)}{q(z)}\right]
$$

- 由 Jensen 不等式（$\log$ 为凹函数）得到。
- 在 DDPM 中：$x = x_0$，$z = x_{1:T}$，$q(z) = q(x_{1:T}|x_0)$，$p(x,z) = p_\theta(x_{0:T})$。

#### 8. DDPM 的负 ELBO（损失）

$$
L = \mathbb{E}_q\left[-\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}\right]
$$

- 即 $\mathbb{E}_q[-\log p_\theta(x_{0:T}) + \log q(x_{1:T}|x_0)]$，最小化 $L$ 等价于最大化 ELBO。

#### 9. 展开为 KL 散度之和

$$
\begin{aligned}
L &= \mathbb{E}_q\Big[-\log p_\theta(x_T) - \sum_{t=1}^T \log \frac{p_\theta(x_{t-1}|x_t)}{q(x_{t-1}|x_t,x_0)}\Big] \\
&= \mathbb{E}_q\Big[-\log \frac{p(x_T)}{q(x_T|x_0)} - \sum_{t=2}^T \log \frac{p_\theta(x_{t-1}|x_t)}{q(x_{t-1}|x_t,x_0)} - \log p_\theta(x_0|x_1)\Big] \\
&= \mathbb{E}_q\Big[ D_{KL}(q(x_T|x_0)\|p(x_T)) + \sum_{t=2}^T D_{KL}(q(x_{t-1}|x_t,x_0)\|p_\theta(x_{t-1}|x_t)) \Big]
\end{aligned}
$$

- 利用条件概率分解和贝叶斯公式，将损失分解为多个 KL 散度。
- 第一项 $D_{KL}(q(x_T|x_0)\|p(x_T))$ 是常数（因为 $\beta_t$ 固定，与 $\theta$ 无关），可忽略。
- 剩余项逐时间步比较真实后验 $q(x_{t-1}|x_t,x_0)$ 与模型预测 $p_\theta(x_{t-1}|x_t)$。

---

### 四、真实后验的解析形式

#### 10. 条件后验 $q(x_{t-1}|x_t,x_0)$

$$
q(x_{t-1}|x_t,x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t,x_0), \tilde{\beta}_t\mathbf{I})
$$
$$
\tilde{\mu}_t(x_t,x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t x_0 + \sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})x_t}{1-\bar{\alpha}_t}
$$
$$
\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t
$$

- 由前向过程的马尔可夫性和高斯条件分布公式推导得出。
- 这是**给定 $x_0$ 和 $x_t$ 时 $x_{t-1}$ 的真实分布**，用于指导模型学习。

---

### 五、模型简化与损失化简

#### 11. 固定方差与预测均值

$$
p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t,t), \sigma_t^2\mathbf{I}), \quad \sigma_t^2 = \beta_t \text{ 或 } \tilde{\beta}_t
$$

- 方差不学习，设为常数。
- 模型只需拟合均值 $\mu_\theta(x_t,t)$ 与真实后验均值 $\tilde{\mu}_t(x_t,x_0)$。

#### 12. 重参数化 $x_t$ 与 $x_0$ 的关系

$$
x_t(x_0,\epsilon) = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, \quad \epsilon \sim \mathcal{N}(0,\mathbf{I})
$$
$$
x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}\big(x_t(x_0,\epsilon) - \sqrt{1-\bar{\alpha}_t}\epsilon\big)
$$

- 这是前向过程的重参数化技巧，便于将随机性显式表达。

#### 13. KL 散度的化简（预测噪声）

$$
L_{t-1} = \mathbb{E}_{x_0,\epsilon}\left[\frac{1}{2\sigma_t^2}\left\|\tilde{\mu}_t(x_t,x_0) - \mu_\theta(x_t,t)\right\|^2\right]
$$
代入 $\tilde{\mu}_t$ 和 $x_t$ 表达式，可重写为：
$$
L_{t-1} = \mathbb{E}_{x_0,\epsilon}\left[\frac{1}{2\sigma_t^2}\left\|\frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon\right) - \mu_\theta(x_t,t)\right\|^2\right]
$$

- 若令 $\mu_\theta(x_t,t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t,t)\right)$，则 KL 散度简化为预测噪声 $\epsilon$ 与 $\epsilon_\theta$ 的均方误差。

#### 14. 最终简化损失（常用版本）

$$
L_{t-1} = \mathbb{E}_{x_0,\epsilon}\left[\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar{\alpha}_t)}\left\|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon,\, t)\right\|^2\right]
$$

- 实际实现中常忽略加权系数，直接最小化 $\|\epsilon - \epsilon_\theta(x_t,t)\|^2$，因为不同 $t$ 的系数对优化影响不大。

---

### 总结：DDPM 核心流程

1. **前向**：固定噪声调度，将 $x_0$ 逐步加噪至 $x_T \sim \mathcal{N}(0,\mathbf{I})$。
2. **反向**：学习神经网络 $\epsilon_\theta(x_t,t)$，预测当前噪声，从而逐步去噪。
3. **训练**：随机采样时间步 $t$，从真实数据 $x_0$ 采样噪声 $\epsilon$，构造 $x_t$，然后最小化 $\|\epsilon - \epsilon_\theta(x_t,t)\|^2$。
4. **生成**：从 $x_T \sim \mathcal{N}(0,\mathbf{I})$ 开始，迭代 $t=T..1$ 用 $p_\theta(x_{t-1}|x_t)$ 采样，得到 $x_0$。

希望这个逐公式解释能帮助你巩固 DDPM 的数学细节。如果你对某一步的推导还有疑问，可以继续提问。
