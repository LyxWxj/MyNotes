# VAE（变分自编码器）

> 来源：知乎专栏《VAE、Diffusion、Flow Matching 系统讲解（一）：VAE 变分自编码器》- 冰锐

## 1. 核心思想

生成模型的目标是学会数据分布 $p(x)$，从而能采样生成新数据。

**VAE 的策略**：假设数据由低维隐变量 $z$ 生成（先采样 $z$，再经 Decoder 映射成 $x$）：

$$p(x) = \int p(x|z)\,p(z)\,dz$$

这基于**流形假设**：高维数据的实际自由度远低于维度。例如 28×28 的手写数字（784 维），变化因素只有笔画粗细、倾斜角度等约 10-20 个因素。VAE 用一个低维向量 $z$（如 20 维）表示这些潜在因素。

**两个核心困难**：

1. $p(x) = \int p(x|z)p(z)dz$ 需要对所有可能的 $z$ 求积分，高维空间不可行
2. 训练还需要后验 $p(z|x) = \frac{p(x|z)p(z)}{p(x)}$，但分母 $p(x)$ 就是不可解的积分

**VAE 的解法——变分推断**：既然真实后验 $p(z|x)$ 算不出来，就训练一个 Encoder 网络 $q_\phi(z|x)$（输出高斯分布的参数 $\mu, \sigma$）去近似它。然后优化一个可计算的下界 ELBO 来替代不可解的积分。

"变分"来自变分法（Calculus of Variations）——在一族函数中找最优的那个。VAE 中的"变分推断"指的是在参数化的分布族 $\{q_\phi(z|x)\}$ 中搜索最接近真实后验的分布。

## 2. 架构

```txt
x  →  [Encoder q_φ(z|x)]  →  μ, σ²
                                ↓
                         z = μ + σ·ε    (ε ~ N(0,I), 重参数化)
                                ↓
z  →  [Decoder p_θ(x|z)]  →  x̂ (重建)
```

- **Encoder** $q_\phi(z|x)$：数据 → 隐变量分布的参数（均值 $\mu$、方差 $\sigma^2$）
- **Decoder** $p_\theta(x|z)$：隐变量 → 重建数据

## 3. ELBO 推导（核心数学）

### 3.1 出发点：最大化对数似然

希望最大化 $\log p_\theta(x)$，但 $p_\theta(x) = \int p_\theta(x|z)p(z)dz$ 不可计算。

用 $\log$ 而非 $p(x)$ 的原因：(1) 数值稳定——$p(x)$ 在高维空间极小（如 $10^{-300}$），取 log 后正常；(2) 乘法变加法——联合似然 $\prod p(x_i)$ 取 log 后变为 $\sum \log p(x_i)$；(3) $\log$ 单调递增，最优解等价。

### 3.2 引入变分分布

引入近似后验 $q_\phi(z|x)$，做重要性采样：

$$\log p_\theta(x) = \log \int \frac{p_\theta(x, z)}{q_\phi(z|x)} \cdot q_\phi(z|x) \, dz = \log \mathbb{E}_{q_\phi(z|x)}\left[\frac{p_\theta(x, z)}{q_\phi(z|x)}\right]$$

### 3.3 Jensen 不等式下推

$\log$ 是凹函数，$\log \mathbb{E}[X] \geq \mathbb{E}[\log X]$：

$$\log p_\theta(x) \geq \mathbb{E}_{q_\phi(z|x)}\left[\log \frac{p_\theta(x, z)}{q_\phi(z|x)}\right]$$

这个下界就是 **ELBO**（Evidence Lower Bound，证据下界）。

### 3.4 ELBO 的分解

把联合概率拆开 $p_\theta(x,z) = p_\theta(x|z) \cdot p(z)$：

$$\text{ELBO} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{\text{KL}}(q_\phi(z|x) \| p(z))$$

- **第一项（重建项）**：Decoder 从 $z$ 重建 $x$ 的质量
- **第二项（KL 正则项）**：Encoder 输出的分布偏离先验 $\mathcal{N}(0,I)$ 的程度

### 3.5 ELBO 与真实似然的关系

$$\log p_\theta(x) = \text{ELBO} + D_{\text{KL}}(q_\phi(z|x) \| p_\theta(z|x))$$

因为 $D_{\text{KL}} \geq 0$，所以 $\log p_\theta(x) \geq \text{ELBO}$。最大化 ELBO 同时做到两件事：

1. 最大化数据似然（让生成模型变好）
2. 最小化近似间隙（让 Encoder 近似更准确）

### 3.6 损失函数最终形式

$$\mathcal{L}_{\text{VAE}} = -\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] + D_{\text{KL}}(q_\phi(z|x) \| p(z))$$

| 项 | 含义 | 直觉 |
|---|---|---|
| $-\mathbb{E}[\log p_\theta(x\|z)]$ | Decoder 从 $z$ 重建 $x$ 的负对数似然 | 若 Decoder 输出高斯分布，就是 MSE |
| $D_{\text{KL}}(q \| p)$ | Encoder 输出的分布偏离先验 $\mathcal{N}(0,I)$ 的程度 | 让隐空间平滑可采样 |

**为什么重建损失是 MSE？** 假设 $p_\theta(x|z) = \mathcal{N}(x; \mu_\theta(z), \sigma^2 I)$，则 $-\log p_\theta(x|z) = \frac{\|x - \mu_\theta(z)\|^2}{2\sigma^2} + \text{const}$，忽略常数后就是 MSE。

## 4. 重参数化技巧

**问题**：从 $q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \sigma^2_\phi(x))$ 中采样 $z$ 的操作不可微，梯度无法从 Decoder 流回 Encoder。

**解决方案**：把随机性"外包"给外部噪声：

$$z = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

- $\epsilon$ 从固定分布采样（与参数 $\phi$ 无关）
- $z$ 对 $\mu$ 和 $\sigma$ 是可微的确定性函数：$\frac{\partial z}{\partial \mu} = 1, \frac{\partial z}{\partial \sigma} = \epsilon$

重参数化并非 VAE 独有，是随机计算图中的通用方法。**Diffusion 中的前向过程 $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon$ 本质上也是重参数化。**

## 5. KL 散度闭式解

当 $q(z|x) = \mathcal{N}(\mu, \text{diag}(\sigma^2))$，$p(z) = \mathcal{N}(0, I)$ 时，KL 散度有闭式解：

$$D_{\text{KL}} = -\frac{1}{2}\sum_{j=1}^{d}\left(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

这避免了蒙特卡洛采样估计 KL 散度带来的方差。

## 6. VAE 的本质困境

### 6.1 重建 vs 正则化的张力

两项损失之间存在根本性矛盾：

- **重建项**想让 Encoder 把尽可能多的信息塞进 $z$ → $\sigma$ 越小越好
- **KL 项**想让 $q(z|x)$ 接近 $\mathcal{N}(0,I)$ → $\mu \to 0$，$\sigma \to 1$

这种张力导致 VAE 生成的图像往往偏模糊——它是所有可能重建的"平均"。

### 6.2 后验坍缩（Posterior Collapse）

当 Decoder 太强大时（如自回归 Decoder），它可以忽略 $z$ 直接从上下文生成 $x$。此时 $q(z|x) \approx p(z)$，$z$ 不再携带任何信息。

**缓解方法**：KL 退火、$\beta$-VAE（调节 KL 权重）、Free bits（给每维 KL 设最小值）。

## 7. 特点

| 优势 | 劣势 |
|---|---|
| 训练快（一次前传，无需迭代采样） | 生成质量一般（后验近似 + 高斯假设导致模糊） |
| 隐空间有明确结构，可做插值、属性编辑 | 表达能力受限于高斯假设 |
| 可做数据压缩（作为其他模型的前端） | |

## 8. 应用

- Stable Diffusion 的图像编解码器（KL-VAE，把 512×512 图压缩到 64×64 隐空间）
- 异常检测、数据增强、半监督学习

## 9. 核心速记卡

**一句话总结**：真实后验 $p(z|x)$ 算不出来 → 用 Encoder $q_\phi(z|x)$ 近似 → 优化 ELBO（重建 + KL）作为替代目标。

**核心逻辑链**：

1. 目标：最大化 $\log p(x)$，但积分 $\int p(x|z)p(z)dz$ 不可算
2. 引入近似后验 $q_\phi(z|x)$，用 Jensen 不等式得到下界 ELBO
3. ELBO = 重建项 - KL 项
4. 采样 $z$ 不可微 → 重参数化：$z = \mu + \sigma \cdot \varepsilon$
5. 训练 Encoder + Decoder，最小化 $-\text{ELBO}$

**关键公式**：

- $\log p(x) = \text{ELBO} + D_{\text{KL}}(q_\phi \| p(z|x))$ —— ELBO 是对数似然的下界
- $\mathcal{L} = -\mathbb{E}[\log p_\theta(x|z)] + D_{\text{KL}}(q_\phi \| p(z))$ —— 损失 = 重建 + KL
- $z = \mu + \sigma \cdot \varepsilon,\ \varepsilon \sim \mathcal{N}(0,I)$ —— 重参数化技巧

**与下一章的衔接**：VAE 一步跳太远（从 $z$ 直接到 $x$），导致模糊。Diffusion 的策略是"把一步大跳拆成多步小走"——逐步去噪，每步只需做微小调整。
