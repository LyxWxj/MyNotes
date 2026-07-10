---
type: Note
related_to: "[[Diffusion]]"
status: Active
tags:
  - distillation
  - diffusion
  - DMD
  - score-matching
url:
  - https://tianweiy.github.io/dmd/
  - https://tianweiy.github.io/dmd2/
---

# DMD 与 DMD2：分布匹配蒸馏详解

## 一、背景问题：为什么需要蒸馏？

扩散模型（Diffusion Models）能生成高质量图像，但采样需要几十到几百步迭代去噪，每一步都是一个完整的神经网络前向传播。例如 Stable Diffusion 生成一张 512×512 图像需要约 2590ms。

**目标**：将多步扩散模型"蒸馏"成一个**一步生成器** $G_\theta$，将随机噪声 $z$ 直接映射为图像，同时尽量保持教师模型的质量。

---

## 二、DMD（Distribution Matching Distillation）

> 论文：*One-step Diffusion with Distribution Matching Distillation* (Yin et al., 2024)

### 2.1 核心思想：逐点匹配 vs 分布匹配

#### 什么是逐点匹配（Paired Matching）？

逐点匹配的核心假设是：**给定同一个噪声输入 $z$，学生模型的输出应该和教师模型的输出完全一致**。

形式化地说，教师模型有一个从噪声到图像的映射 $f_{\text{teacher}}: z \mapsto y$（通过多步采样实现），逐点匹配要求学生模型 $G_\theta$ 满足：

$$
\forall z \sim \mathcal{N}(0, I): \quad G_\theta(z) \approx f_{\text{teacher}}(z)
$$

对应的损失函数通常是逐像素或逐特征的回归：

$$
\mathcal{L}_{\text{paired}} = \mathbb{E}_{z \sim \mathcal{N}(0,I)} \left[ d\left( G_\theta(z),\ f_{\text{teacher}}(z) \right) \right]
$$

其中 $d$ 可以是 L2、L1、LPIPS 等距离度量。

**具体例子**：

1. **直接回归（Luhman et al.）**：离线用教师模型对大量噪声 $z$ 跑完整采样流程，得到配对数据集 $\{(z_i, y_i)\}$，然后训练学生做监督回归：
   $$
   \mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \| G_\theta(z_i) - y_i \|^2
   $$

2. **Progressive Distillation**：教师模型先从 1024 步蒸馏到 512 步，再从 512 步蒸馏到 256 步……每一轮都要求学生对**同一个噪声输入**，在**一半的步数内**产生和教师相同的输出。

3. **Consistency Distillation**：要求学生模型在 ODE 轨迹上的**任意两个时间点** $t_i$ 和 $t_j$，对同一个初始噪声，输出一致的结果：
   $$
   \mathcal{L} = \| G_\theta(x_{t_i}, t_i) - G_\theta(x_{t_j}, t_j) \|^2
   $$
   这也是一种逐点匹配——在同一条轨迹内部做配对约束。

#### 逐点匹配的根本问题

逐点匹配有一个隐含的**过强假设**：它认为教师模型的采样路径是最优的，学生必须严格复刻。

但事实是：
- 扩散模型的采样路径**不是唯一的**——同一个分布可以用不同的 ODE/SDE 轨迹来采样
- 教师模型的单条采样路径可能**不是最高效的**——它可能绕了弯路
- 强制学生复刻教师的路径，**上限就是教师本身**，无法超越

类比：就像要求一个学生抄写名家书法的每一笔的精确位置，而不是学习整体的风格和结构。抄写再精确，也只能无限接近原作，不可能自成一派。

#### 什么是分布匹配（Distribution Matching）？

分布匹配放松了这个约束：**不要求对同一个 $z$ 产生相同的输出，只要求学生生成的所有图像在整体上看起来"来自同一个分布"**。

$$
p_{\text{student}}(x) \approx p_{\text{teacher}}(x)
$$

形式化地说，最小化两个分布之间的散度：

$$
\mathcal{L}_{\text{dist}} = D(p_{\text{fake}} \| p_{\text{real}})
$$

这意味着：
- 学生对噪声 $z_1$ 生成的图像，不需要和教师对 $z_1$ 生成的一样
- 只要学生生成的图像集合在统计上和教师的不可区分即可
- 学生有**自由度**去找到更高效的生成路径

类比：GAN 就是典型的分布匹配——生成器不需要对某个噪声生成特定图片，只要整体分布骗过判别器就行。

#### 直观对比

```
逐点匹配 (Paired Matching):
  噪声 z₁ → 教师 → 图像 y₁    噪声 z₁ → 学生 → 图像 ŷ₁  ← 要求 ŷ₁ ≈ y₁
  噪声 z₂ → 教师 → 图像 y₂    噪声 z₂ → 学生 → 图像 ŷ₂  ← 要求 ŷ₂ ≈ y₂
  噪声 z₃ → 教师 → 图像 y₃    噪声 z₃ → 学生 → 图像 ŷ₃  ← 要求 ŷ₃ ≈ y₃
  每个点都要对齐！

分布匹配 (Distribution Matching):
  噪声 z₁ → 教师 → 图像 y₁
  噪声 z₂ → 教师 → 图像 y₂      整体分布: p_real
  噪声 z₃ → 教师 → 图像 y₃              ↕ 要求分布一致
  ...                              整体分布: p_fake
  噪声 z₁' → 学生 → 图像 ŷ₁'
  噪声 z₂' → 学生 → 图像 ŷ₂'     ← 不需要 z 和教师相同
  噪声 z₃' → 学生 → 图像 ŷ₃'     ← 不需要单点对应
  只要求整体分布一致！
```

#### 为什么分布匹配更优？

| 维度 | 逐点匹配 | 分布匹配 |
|------|----------|----------|
| 约束强度 | 强（逐点对应） | 弱（整体一致） |
| 学生自由度 | 低（必须复刻路径） | 高（可探索新路径） |
| 能否超越教师 | ❌ 上限 = 教师 | ✅ 理论上可以 |
| 训练数据需求 | 需要配对 $(z, y)$ | 只需要无配对样本 |
| 模式坍缩风险 | 低（有配对监督） | 需要额外设计避免 |

DMD 的核心洞察：**用分布匹配替代逐点匹配，释放学生的表达潜力，同时用扩散模型的 score function 作为分布级别的监督信号**。

### 2.2 数学推导

目标是最小化学生生成分布 $p_{\text{fake}}$ 和真实分布 $p_{\text{real}}$ 之间的 KL 散度：

$$
\mathcal{L}_{\text{DMD}} = D_{\text{KL}}(p_{\text{fake}} \| p_{\text{real}}) = \mathbb{E}_{x \sim p_{\text{fake}}} \left[ \log p_{\text{fake}}(x) - \log p_{\text{real}}(x) \right]
$$

#### 从 KL 散度到 Score Function 的推导

**为什么不能直接对期望求导？**

直觉上可能会这样推导：

$$
\nabla_\theta D_{\text{KL}} \stackrel{?}{=} \mathbb{E}_{x \sim p_{\text{fake}}} \left[ \frac{\nabla_\theta p_{\text{fake}}(x)}{p_{\text{fake}}(x)} - \frac{\nabla_\theta p_{\text{real}}(x)}{p_{\text{real}}(x)} \right]
$$

这里有两个问题：
1. $p_{\text{real}}$ 不依赖 $\theta$（教师模型是冻结的），所以第二项确实为零——这部分没错
2. 但**第一项也不对**：$\nabla_\theta$ 不能直接穿过期望符号，因为**期望的采样分布 $p_{\text{fake}}$ 本身依赖 $\theta$**

类比：$\nabla_w \mathbb{E}_{x \sim p_w}[f(x)] \neq \mathbb{E}_{x \sim p_w}[\nabla_w f(x)]$，除非 $p_w$ 不依赖 $w$。

**正确推导：Leibniz 积分法则**

将 KL 散度写成积分形式：

$$
D_{\text{KL}} = \int p_{\text{fake}}(x) \left[ \log p_{\text{fake}}(x) - \log p_{\text{real}}(x) \right] dx
$$

对 $\theta$ 求导，用 Leibniz 法则（积分和求导交换顺序）+ 乘积法则：

$$
\nabla_\theta D_{\text{KL}} = \int \nabla_\theta \left\{ p_{\text{fake}}(x) \cdot \left[ \log p_{\text{fake}}(x) - \log p_{\text{real}}(x) \right] \right\} dx
$$

展开乘积法则，拆成两项：

$$
= \underbrace{\int \left[ \log p_{\text{fake}} - \log p_{\text{real}} \right] \cdot \nabla_\theta p_{\text{fake}} \, dx}_{\text{项 A：被积函数的导数}} + \underbrace{\int p_{\text{fake}} \cdot \nabla_\theta \left[ \log p_{\text{fake}} - \log p_{\text{real}} \right] dx}_{\text{项 B：对数内部的导数}}
$$

**项 B 的化简**：

$$
\text{项 B} = \int p_{\text{fake}} \cdot \frac{\nabla_\theta p_{\text{fake}}}{p_{\text{fake}}} dx - \int p_{\text{fake}} \cdot \underbrace{\frac{\nabla_\theta p_{\text{real}}}{p_{\text{real}}}}_{=0} dx = \int \nabla_\theta p_{\text{fake}} \, dx = \nabla_\theta \underbrace{\int p_{\text{fake}} \, dx}_{=1} = 0
$$

项 B 为零的原因：概率分布的归一化条件 $\int p_{\text{fake}} dx = 1$，对其求导必然为零。直觉上，$p_{\text{fake}}$ 形状怎么变，它在全空间的积分始终是 1。

**项 A 的化简**：

利用对数导数技巧（log-derivative trick）：

$$
\nabla_\theta p_{\text{fake}}(x) = p_{\text{fake}}(x) \cdot \nabla_\theta \log p_{\text{fake}}(x)
$$

代入项 A：

$$
\text{项 A} = \int \left[ \log p_{\text{fake}} - \log p_{\text{real}} \right] \cdot p_{\text{fake}} \cdot \nabla_\theta \log p_{\text{fake}} \, dx
$$

$$
= \mathbb{E}_{x \sim p_{\text{fake}}} \left[ \left( \log p_{\text{fake}}(x) - \log p_{\text{real}}(x) \right) \cdot \nabla_\theta \log p_{\text{fake}}(x) \right]
$$

现在对 $\nabla_\theta \log p_{\text{fake}}(x)$ 用链式法则。注意 $x = G_\theta(z)$，所以 $\log p_{\text{fake}}(x)$ 通过 $x$ 间接依赖 $\theta$：

$$
\nabla_\theta \log p_{\text{fake}}(x) = \nabla_x \log p_{\text{fake}}(x) \cdot \frac{\partial G_\theta(z)}{\partial \theta} = s_{\text{fake}}(x) \cdot \frac{dG_\theta(z)}{d\theta}
$$

类似地，$\nabla_x \log p_{\text{real}}(x) = s_{\text{real}}(x)$。

进一步，可以证明（通过对 fake score 做分部积分）：

$$
\mathbb{E}_{x \sim p_{\text{fake}}} \left[ \log p_{\text{fake}}(x) \cdot s_{\text{fake}}(x) \right] = -\mathbb{E}_{x \sim p_{\text{fake}}} \left[ s_{\text{fake}}(x) \right]
$$

最终化简得到：

$$
\boxed{\nabla_\theta D_{\text{KL}} = \mathbb{E}_{z \sim \mathcal{N}(0,I)} \left[ \left( s_{\text{fake}}(x) - s_{\text{real}}(x) \right) \cdot \frac{dG_\theta(z)}{d\theta} \right]}
$$

其中 $s(x) = \nabla_x \log p(x)$ 是**得分函数（Score Function）**。

**与 REINFORCE 的联系**：上述推导的核心技巧——对数导数技巧——和 REINFORCE 算法完全相同：

$$
\nabla_\theta \mathbb{E}_{x \sim p_\theta}[f(x)] = \mathbb{E}_{x \sim p_\theta}[f(x) \cdot \nabla_\theta \log p_\theta(x)]
$$

这个技巧把"对分布求导"转化为"对对数概率求导"，使得梯度可以通过采样来估计。

**直觉理解**：
- $s_{\text{real}}(x)$：把图像往"更真实"的方向推
- $-s_{\text{fake}}(x)$：把图像往"更不像假图"的方向推
- 两者之差 = 让假图更真实、同时更不像假图

#### 公式中各项的含义

**$G_\theta(z)$ 是什么？**

$G_\theta$ 是 DMD 要训练的**一步生成器**：

$$
G_\theta: \mathbb{R}^{H \times W \times C} \to \mathbb{R}^{H \times W \times C}
$$

- **输入**：随机噪声 $z \sim \mathcal{N}(0, I)$（和扩散模型采样时的初始噪声同维度）
- **输出**：一张图像 $x = G_\theta(z)$
- **架构**：和教师扩散模型的 denoiser（如 UNet）**完全相同**，但**去掉了时间条件输入** $t$

为什么去掉时间条件？因为教师模型 $\mu_{\text{base}}(x_t, t)$ 需要在不同时间步 $t$ 上做去噪，所以需要 $t$ 作为条件输入。而一步生成器只做**一次前向传播**，不存在"第几步"的概念，所以去掉 $t$。

**初始化方式**：

$$
G_\theta(z) = \mu_{\text{base}}(z, T-1), \quad \forall z
$$

即用教师模型在**最大噪声级别**（$t = T-1 = 999$）的权重来初始化。下面详细解释这个初始化在不同架构下的具体含义。

**DiT 架构下的初始化**：

DiT 中，时间步 $t$ 不是直接修改权重矩阵，而是通过 **adaLN（自适应层归一化）** 注入：

$$
\text{adaLN}(x, t) = \gamma(t) \cdot \text{LayerNorm}(x) + \beta(t)
$$

其中 $\gamma(t), \beta(t)$ 是由时间步嵌入 $e(t)$ 通过一个小 MLP 生成的调制参数。因此 DiT 的参数分为两类：

| 参数类型 | 是否依赖 $t$ | 例子 |
|----------|-------------|------|
| **共享权重** | ❌ | Attention 的 $W_Q, W_K, W_V, W_O$，MLP 权重，LayerNorm 基础参数 |
| **时间调制参数** | ✅ | adaLN 的缩放 $\gamma(t)$ 和平移 $\beta(t)$ |

初始化 $G_\theta(z) = \mu_{\text{base}}(z, T-1)$ 的具体操作是：
1. **复制教师模型的全部共享权重**（Attention、MLP、LayerNorm 等）
2. **去掉时间步条件输入**——即去掉 adaLN 中根据 $t$ 生成 $\gamma(t), \beta(t)$ 的那部分网络，退化为普通 LayerNorm

或者等价地，固定 $t = T-1$ 时的调制参数：

$$
G_\theta(z) = \mu_{\text{base}}(z, t=T-1) \implies \text{adaLN}(x) = \gamma(T-1) \cdot \text{LN}(x) + \beta(T-1)
$$

**UNet 架构下的初始化**：

如果教师是 UNet（如 Stable Diffusion），时间步通过 sinusoidal embedding + FiLM 调制注入，逻辑相同：复制权重，去掉（或固定）时间条件。

**为什么选择 $t = T-1$？**

因为一步生成器的输入是**纯高斯噪声** $z \sim \mathcal{N}(0, I)$，对应扩散过程中**噪声最大的时刻**。教师模型在 $t = T-1$ 时做的事情正是"从纯噪声一步预测干净图像的均值"，和一步生成器的目标完全一致。

```
教师模型在不同时间步的行为：
  t = 999: 纯噪声 → 预测大致结构（粗粒度去噪）
  t = 500: 中等噪声 → 预测更多细节
  t = 100: 轻微噪声 → 精细去噪
  t = 0:   干净图像 → 输出

一步生成器 = 教师在 t=999 的"粗粒度去噪"能力的直接复制
```

**梯度公式中各项的角色**：

$$
\nabla_\theta D_{\text{KL}} = \underbrace{\left( s_{\text{fake}}(x) - s_{\text{real}}(x) \right)}_{\text{方向信号：往哪推}} \cdot \underbrace{\frac{dG_\theta(z)}{d\theta}}_{\text{Jacobian：怎么改参数}}
$$

- $G_\theta(z)$：前向传播，噪声 → 图像
- $\frac{dG_\theta(z)}{d\theta}$：生成器输出对参数的 Jacobian，通过反向传播计算
- $s_{\text{fake}}(x) - s_{\text{real}}(x)$：score 之差，告诉图像"该往哪个方向变"
- 两者相乘，就是更新生成器参数 $\theta$ 的梯度

### 2.3 为什么要加噪声？（Score-SDE 的关键）

直接在原始分布上计算 score 有问题：真实分布和假分布可能不重叠（特别是训练初期），导致梯度无意义。

**解决方案**：对图像加不同程度的高斯噪 声，创造一族"模糊"分布 $p_{\text{real},t}$ 和 $p_{\text{fake},t}$，它们在整个空间上都有支撑，因此总是重叠的：

$$
\nabla_\theta \mathcal{L}_{\text{DMD}} = \mathbb{E}_{z,t} \left[ w_t \cdot \alpha_t \cdot \left( s_{\text{fake}}(x_t, t) - s_{\text{real}}(x_t, t) \right) \cdot \frac{dG_\theta}{d\theta} \right]
$$

其中 $x_t$ 是对生成图像 $x = G_\theta(z)$ 加噪后的结果。

### 2.4 两个扩散模型分别做什么？

| 模型 | 角色 | 初始化 | 是否更新 |
|------|------|--------|----------|
| $\mu_{\text{real}}$ | 建模真实数据的 score | 预训练教师模型 | **冻结** |
| $\mu_{\text{fake}}$ | 建模学生生成分布的 score | 预训练教师模型 | **动态更新** |

**三者的关系**：$\mu_{\text{base}}$、$\mu_{\text{real}}$、$\mu_{\text{fake}}$ 都从同一个预训练教师模型出发：

```
μ_base（预训练教师模型）
  ├── 复制一份 → μ_real（冻结，估计真实数据的 score s_real）
  │                  用于：把假图往"更真实"的方向推
  │                  不更新，因为真实数据分布是固定的
  │
  └── 复制一份 → μ_fake（动态更新，估计假图分布的 score s_fake）
                     用于：把假图往"更不像假图"的方向推
                     需要持续更新，因为假图分布随 G_θ 的变化而变化
```

- $\mu_{\text{base}}$ 和 $\mu_{\text{real}}$ 是**同一个模型的不同称呼**：$\mu_{\text{base}}$ 强调其"预训练教师"的原始身份，$\mu_{\text{real}}$ 强调其在 DMD 框架中"真实 score 估计器"的角色
- $\mu_{\text{fake}}$ 也由 $\mu_{\text{base}}$ 初始化而来，但训练过程中会不断更新参数

$\mu_{\text{fake}}$ 通过标准的去噪损失持续训练，以跟踪学生生成分布的变化：

$$
\mathcal{L}_\phi^{\text{denoise}} = \| \mu_\phi^{\text{fake}}(x_t, t) - x_0 \|_2^2
$$

### 2.5 回归损失（Regression Loss）

DMD 发现仅靠分布匹配梯度训练不稳定，因此加了一个**回归损失**作为正则化：

1. **离线预计算**：用教师模型 + 确定性采样器对大量噪声 $z$ 生成对应的图像 $y$
2. **在线训练**：$\mathcal{L}_{\text{reg}} = \text{LPIPS}(G_\theta(z), y)$

这个回归损失确保学生模型的输出在**大尺度结构**上和教师一致。

### 2.6 DMD 的完整训练流程

```
┌─────────────────────────────────────────────────────┐
│  输入: 预训练扩散模型 μ_base, 训练数据               │
│                                                     │
│  1. 初始化生成器 G_θ = μ_base(z, T-1)               │
│  2. 初始化 fake score μ_φ = μ_base                  │
│  3. 离线预计算 (z, y) 配对数据集                     │
│                                                     │
│  训练循环:                                          │
│    a. 采样 z ~ N(0,I), 生成假图 x = G_θ(z)          │
│    b. 对 x 加不同程度噪声 → x_t                     │
│    c. 用 μ_real 和 μ_fake 分别计算 score             │
│    d. 计算分布匹配梯度 ∇_θ L_DMD                    │
│    e. 计算回归损失 L_reg = LPIPS(x, y)              │
│    f. 更新 θ: ∇_θ(L_DMD + L_reg)                   │
│    g. 更新 φ: 用假图训练 μ_fake 的去噪损失          │
└─────────────────────────────────────────────────────┘
```

### 2.7 DMD 的局限

1. **回归损失昂贵**：为 SDXL 预计算配对数据约需 700 A100 天
2. **回归损失限制上限**：学生的质量被教师的采样路径绑定，无法超越教师
3. **仅支持一步生成**

### 2.8 $\mu_{\text{fake}}$ 的意义：它是谁、做什么、为什么需要它

#### $\mu_{\text{fake}}$ 不是生成器

$\mu_{\text{fake}}$ **不是生成器**，它是一个**完整的扩散模型**（和教师架构相同），用来**估计假图分布的 score function**。它虽然理论上也能通过多步采样生成图像，但这不是它的目的。

三个模型的角色对比：

| | $G_\theta$ | $\mu_{\text{fake}}$ | $\mu_{\text{real}}$ |
|---|---|---|---|
| **是什么** | 一步生成器 | 完整扩散模型 | 完整扩散模型 |
| **能否生成图像** | ✅ 一步生成 | ✅ 但需要多步采样 | ✅ 但需要多步采样 |
| **实际用途** | 生成假图 | 估计 $s_{\text{fake}}$（score） | 估计 $s_{\text{real}}$（score） |
| **输入** | 噪声 $z$ | 噪声图像 $x_t$ + 时间步 $t$ | 噪声图像 $x_t$ + 时间步 $t$ |
| **输出** | 图像 | 去噪预测（用于算 score） | 去噪预测（用于算 score） |

#### 为什么需要 $\mu_{\text{fake}}$：它是计算梯度的桥梁

核心问题：**我们需要计算 KL 散度对 $\theta$ 的梯度，这需要 $s_{\text{fake}}(x) = \nabla_x \log p_{\text{fake}}(x)$，但这个量无法直接获取。**

为什么无法直接获取？
- $p_{\text{fake}}(x)$ 是由 $G_\theta$ **隐式定义**的：$x = G_\theta(z), z \sim \mathcal{N}(0,I)$
- 我们可以**采样**（跑一次 $G_\theta$ 就得到一张假图）
- 但我们**无法计算密度** $p_{\text{fake}}(x)$，更无法求其 score

所以需要 $\mu_{\text{fake}}$ 作为**桥梁**：

$$
G_\theta \xrightarrow{\text{生成假图}} \text{假图数据集} \xrightarrow{\text{训练}} \mu_{\text{fake}} \xrightarrow{\text{计算}} s_{\text{fake}} \xrightarrow{\text{提供梯度}} G_\theta
$$

$\mu_{\text{fake}}$ 通过在假图上做去噪训练，学会了假图分布的 score function。这个 score 是计算 $G_\theta$ 梯度所**必需的中间量**。

#### $G_\theta$ 和 $\mu_{\text{fake}}$ 的反馈循环

两个模型形成一个**动态耦合的反馈循环**：

```
G_θ 生成假图 x = G_θ(z)
        │
        ▼
μ_fake 在假图上训练（去噪损失），学到假图的 score
        │
        ▼
s_fake 被用来计算 G_θ 的梯度: ∇_θ = s_fake - s_real
        │
        ▼
G_θ 更新参数 → 生成的假图分布变了
        │
        ▼
μ_fake 需要重新跟踪新的假图分布 → 回到顶部
```

- $\mu_{\text{fake}}$ 是 $G_\theta$ 的**批评者**：$G_\theta$ 生成假图，$\mu_{\text{fake}}$ 学习这些假图的分布，然后用 score 告诉 $G_\theta$ 怎么改进
- $\mu_{\text{fake}}$ 必须**持续追赶** $G_\theta$：因为 $G_\theta$ 不断更新，假图分布不断变化，$\mu_{\text{fake}}$ 必须及时更新以提供准确的梯度信号

这正是 DMD2 用**双时间尺度更新规则**（$\mu_{\text{fake}}$ 更新 5 次 vs $G_\theta$ 更新 1 次）的原因——让批评者比生成器学得更快，才能准确跟踪。

#### 最终产物

**DMD 的产物是 $G_\theta$，$\mu_{\text{fake}}$ 训练完就丢弃。**

```
训练阶段（需要三个模型）:
  μ_real(冻结)  ──→  提供 s_real  ──┐
                                     ├──→ s_fake - s_real ──→ 更新 G_θ
  μ_fake(训练)  ──→  提供 s_fake  ──┘
  G_θ(训练)     ──→  生成假图给 μ_fake 学习

推理阶段（只需要一个模型）:
  噪声 z ──→ G_θ ──→ 图像 x
  μ_real 和 μ_fake 全部丢弃！
```

类比：$G_\theta$ 是学生画家，$\mu_{\text{fake}}$ 是评论家。评论家自己不需要画画（生成图像），他只需要能准确评价画作（估计 score）就行。训练完成后，只需要画家就够了。

### 2.9 为什么需要 $\mu_{\text{fake}}$？——从 GAN 的视角进一步理解

要理解 $\mu_{\text{fake}}$ 的意义，需要先理解 GAN，然后对比 DMD 和 GAN 的异同。

#### GAN（生成对抗网络）简介

GAN 由 Goodfellow et al. (2014) 提出，核心思想是**两个网络对抗博弈**：

- **生成器 $G$**：噪声 $z \to$ 假图，目标是骗过判别器
- **判别器 $D$**：图像 $\to$ 真/假概率 $D(x) \in [0,1]$，目标是分辨真假

训练目标是一个极小极大博弈：

$$
\min_G \max_D \; \mathcal{L}(G, D) = \mathbb{E}_{x \sim p_{\text{real}}} [\log D(x)] + \mathbb{E}_{z \sim \mathcal{N}(0,I)} [\log(1 - D(G(z)))]
$$

```
GAN 训练流程（每个训练步）:
  ① 采样真实图像 x ~ p_real
  ② 采样噪声 z，生成假图 x_fake = G(z)
  ③ 更新 D：让 D(x_real) → 1, D(x_fake) → 0
  ④ 更新 G：让 D(G(z)) → 1（骗过 D）
```

GAN 的关键特性：**分布匹配**——不要求噪声 $z$ 和图像一一对应，只要整体分布骗过判别器即可。但 GAN 的训练不稳定，容易模式坍缩。

#### DMD 与 GAN 的对比

**相同点**：
1. 都是分布匹配，不要求逐点对应
2. 都不需要配对数据
3. 都用两个网络对抗：一个生成，一个评估

**不同点**：

| | GAN | DMD |
|---|---|---|
| **批评者** | 学习的判别器 $D$（二分类网络） | 预训练扩散模型 $\mu_{\text{real}}$ + 动态 $\mu_{\text{fake}}$ |
| **批评者的信号** | 一个标量：$D(x) \in [0,1]$（真/假概率） | 一个向量场：$s_{\text{fake}} - s_{\text{real}}$（像素级梯度方向） |
| **信号粒度** | 全局（整张图一个分数） | 局部（每个像素位置有独立的调整方向） |
| **稳定性** | 不稳定，容易模式坍缩 | 相对稳定（score function 提供更丰富的梯度信号） |
| **起点** | 从零开始训练 | 从预训练扩散模型出发 |
| **教师模型** | 不需要 | 需要预训练扩散模型作为 $\mu_{\text{real}}$ |

**信号粒度的差异（关键区别）**：

```
GAN 的判别器反馈：
  假图 ──→ D ──→ 0.2（整张图只有一个分数：20%像真的）
  生成器只知道"这张图整体不太像"，但不知道具体哪里不像。

DMD 的 score 反馈：
  假图 ──→ s_fake - s_real ──→ 每个像素位置一个方向向量
  生成器精确知道"这个像素应该往左偏一点、那个像素应该变亮一些"
```

这就是为什么 DMD 的训练比 GAN 更稳定——**反馈信号的信息量大得多**。

#### $\mu_{\text{fake}}$ 的意义

回到最初的问题：$\mu_{\text{fake}}$ 到底有什么用？

在 GAN 中，判别器 $D$ 同时承担两个职责：
1. 判断"这张图有多像真图"（对应 DMD 中的 $s_{\text{real}}$）
2. 隐式地了解"当前假图长什么样"（对应 DMD 中的 $s_{\text{fake}}$）

DMD 将这两个职责**拆开**，分别交给两个扩散模型：
- $\mu_{\text{real}}$（冻结）：负责"什么是真图"——对应 GAN 中判别器对真图的判断
- $\mu_{\text{fake}}$（动态更新）：负责"当前假图长什么样"——对应 GAN 中判别器对假图的判断

$s_{\text{fake}} - s_{\text{real}}$ 这个梯度本身就具有对抗性质：
- $s_{\text{real}}$：把假图往真实分布的方向推（类似 GAN 中"让 $D$ 给高分"）
- $-s_{\text{fake}}$：把假图推离当前假图分布（防止模式坍缩，类似 GAN 中"让假图不要聚集在少数模式上"）

| GAN 的角色 | DMD 中的对应 |
|---|---|
| 判别器 $D$ | $\mu_{\text{real}}$ + $\mu_{\text{fake}}$ 共同构成"判别器" |
| $D(x) \to 1$（对真图） | $s_{\text{real}}$ 把假图往真实方向推 |
| $D(x) \to 0$（对假图） | $-s_{\text{fake}}$ 把假图推离当前假图分布 |

所以 $\mu_{\text{fake}}$ 的意义是：**它是 DMD 版本的"判别器"的一半**，负责建模当前假图分布的 score，使得梯度信号 $s_{\text{fake}} - s_{\text{real}}$ 能同时包含"往真实方向推"和"远离假图模式"两个分量。训练完成后它功成身退，只留下 $G_\theta$ 作为最终产物。

---

## 三、DMD2（Improved Distribution Matching Distillation）

> 论文：*Improved Distribution Matching Distillation for Fast Image Synthesis* (Yin et al., 2024)

DMD2 针对 DMD 的三个核心问题提出了三项改进：

### 3.1 改进一：去掉回归损失 + 双时间尺度更新规则

**问题**：去掉回归损失后训练不稳定，生成图像的亮度等统计量剧烈波动。

**根因分析**：fake score 模型 $\mu_{\text{fake}}$ 没有足够准确地跟踪学生生成分布的变化。因为学生分布本身在不断变化，$\mu_{\text{fake}}$ 需要持续"追赶"，但更新速度不够快导致梯度有偏。

**解决方案 — 双时间尺度更新规则（Two Time-scale Update Rule）**：

$$
\mu_{\text{fake}} \text{ 更新 } 5 \text{ 次} \quad \longleftrightarrow \quad G_\theta \text{ 更新 } 1 \text{ 次}
$$

让 fake score 模型比生成器更新得更频繁，确保它能准确追踪生成分布。灵感来自 Heusel et al. (TTUR, 2017)。

**效果**：无需回归损失即可稳定训练，消除了昂贵的离线数据预计算。

### 3.2 改进二：引入 GAN 损失 + 真实数据

**问题**：DMD 的学生模型从未接触过真实数据，只通过教师模型的 score 间接学习。教师模型对真实分布的 score 估计不完美，误差会传播给学生。

**解决方案**：在 fake diffusion model 的 UNet 瓶颈处加一个分类头作为判别器：

$$
\mathcal{L}_{\text{GAN}} = \mathbb{E}_{x \sim p_{\text{real}}} [\log D(F(x,t))] + \mathbb{E}_{z} [-\log D(F(G_\theta(z),t))]
$$

其中 $F$ 是加噪操作（前向扩散过程）。

**关键设计**：
- 判别器在**加噪后的样本**上做真假分类（借鉴 DiffusionGAN）
- 这个 GAN 目标与分布匹配哲学一致：不需要配对数据，不依赖教师采样路径
- 学生直接接触真实数据，可以**超越教师**

### 3.3 改进三：支持多步生成 + 向后模拟

**问题**：SDXL 等大模型难以一步蒸馏（容量限制 + 优化景观复杂）。

**解决方案**：支持 N 步生成器。推理时交替执行去噪和加噪：

```
z_0 ~ N(0,I)  →  去噪 x̂_t1 = G(x_t1, t1)  →  加噪 x_t2 = α·x̂_t1 + σ·ε
            →  去噪 x̂_t2 = G(x_t2, t2)  →  加噪 x_t3  →  ...  →  x̂_tN
```

4步模型的时间表：`999, 749, 499, 249`

**训练-推理不匹配问题**：之前的方法用加噪的**真实图像**训练多步生成器，但推理时输入是**生成器自己上一步的输出**，存在 domain mismatch。

**解决方案 — 向后模拟（Backward Simulation）**：训练时用学生生成器自己的输出（模拟推理过程）作为输入，而不是真实图像。

### 3.4 DMD2 完整训练流程

```
┌──────────────────────────────────────────────────────────┐
│  每个训练步:                                              │
│                                                          │
│  步骤 1: 训练判别器 D 和 fake score μ_fake                │
│    - 采样真实图像 x ~ p_real                              │
│    - 采样噪声 z, 生成假图 x_fake = G_θ(z)                 │
│    - 更新 μ_fake (去噪损失, 更新 5 次)                    │
│    - 更新 D (GAN 分类损失)                                │
│                                                          │
│  步骤 2: 训练生成器 G_θ                                   │
│    - 计算分布匹配梯度: s_fake - s_real                    │
│    - 计算 GAN 损失: 让假图骗过判别器                       │
│    - 更新 θ: ∇_θ(L_DMD + L_GAN)                          │
│                                                          │
│  (多步模式下, 训练时用生成器自己的输出模拟推理过程)         │
└──────────────────────────────────────────────────────────┘
```

---

## 四、DMD vs DMD2 对比总结

| 特性 | DMD | DMD2 |
|------|-----|------|
| 回归损失 | ✅ 需要（离线预计算配对数据） | ❌ 去掉 |
| 训练稳定性来源 | 回归损失正则化 | 双时间尺度更新规则 |
| 是否接触真实数据 | ❌ 只通过教师间接学习 | ✅ 通过 GAN 判别器 |
| 能否超越教师 | ❌ 被回归损失绑定 | ✅ 可以超越 |
| 支持步数 | 仅一步 | 一步 / 多步 |
| 训练-推理不匹配 | 无（仅一步） | 用 backward simulation 解决 |
| ImageNet-64 FID | 2.62 | **1.28** |
| COCO 2014 FID | 11.49 | **8.35** |
| 推理加速 | ~100× | ~500× |

---

## 五、一句话总结

> **DMD**：用两个扩散模型分别建模真实/假分布的 score，通过 score 之差做梯度，实现分布级蒸馏；但需要回归损失稳定训练。
>
> **DMD2**：去掉了回归损失（用双时间尺度更新替代），引入 GAN 让学生直接看真实数据，支持多步生成——最终学生可以超越教师。
