# 贝叶斯家族

贝叶斯定理 · 先验与后验 · 朴素贝叶斯 · 贝叶斯网络 · 贝叶斯推断

---

## 贝叶斯定理 (Bayes' Theorem)

---

### 推导贝叶斯定理

从联合概率与条件概率的关系出发：

$$P(A, B) = P(A \mid B) \cdot P(B)$$

交换 $A$ 和 $B$ 的角色：

$$P(B, A) = P(B \mid A) \cdot P(A)$$

**核心洞察**：$A$ 和 $B$ 同时发生的事件与 $B$ 和 $A$ 同时发生的事件是同一事件——顺序无关：

$$P(A, B) = P(B, A)$$

因此：

$P(A \mid B) \cdot P(B) = P(B \mid A) \cdot P(A)$

两边同时除以 $P(B)$：

$$\boxed{\;P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}\;}$$

---

这就是**贝叶斯定理**——它让我们可以"翻转"条件方向。

**用假设-证据的语言表述**（$H$ = 假设，$D$ = 数据）：

$$\underbrace{P(H \mid D)}_{\text{后验}} = \frac{\overbrace{P(D \mid H)}^{\text{似然}} \cdot \underbrace{P(H)}_{\text{先验}}}{\underbrace{P(D)}_{\text{证据}}}$$

**$\theta$ 是什么？** 在参数估计中，$\theta$ 是 $H$ 的一个具体实例——每个可能的参数值都是一个假设。例如，"$\theta = 0.7$"是假设"硬币正面朝上的概率为 0.7"。所以 $P(\theta)$ 和 $P(H)$ 含义相同——参数的先验分布。

---

### 贝叶斯定理：直觉理解

### 先验 (Prior) $P(H)$

在看到数据**之前**的信念

"今天有 20% 的概率下雨"

### 似然 (Likelihood) $P(D \mid H)$

给定假设下数据的概率

"如果下雨，地面有 95% 的概率是湿的"

### 后验 (Posterior) $P(H \mid D)$

看到数据**之后**更新的信念

"地面是湿的——下雨的概率是多少？"

$$P(\text{rain} \mid \text{wet}) = \frac{P(\text{wet} \mid \text{rain}) \cdot P(\text{rain})}{P(\text{wet})} = \frac{0.95 \times 0.20}{0.27} = 0.704$$

**证据** (Evidence) $P(\text{wet})$ 通过全概率公式计算：

$$P(\text{wet}) = P(\text{wet} \mid \text{rain})P(\text{rain}) + P(\text{wet} \mid \text{sunny})P(\text{sunny}) = 0.95 \times 0.20 + 0.10 \times 0.80 = 0.27$$

---

### 证据的作用

证据 $P(D) = \sum_H P(D \mid H) P(H)$ 是一个归一化常数，确保后验概率之和为 1。

**在实践中**，我们通常只关心相对后验：

$$P(H \mid D) \propto P(D \mid H) \cdot P(H)$$

"后验 ∝ 似然 × 先验"

**比较两个假设**：

$$\frac{P(H_1 \mid D)}{P(H_2 \mid D)} = \frac{P(D \mid H_1)}{P(D \mid H_2)} \cdot \frac{P(H_1)}{P(H_2)}$$

**后验几率 = 似然比 × 先验几率**

---

## 先验与后验 (Prior & Posterior)

---

### 选择先验

先验 $P(H)$ 编码了看到数据**之前**的知识。

| 先验类型 | 含义 | 示例 |
| ---------- | ------- | ------- |
| **无信息先验** (Uninformative) | 最小化偏差 | $\theta \sim \text{Uniform}(0, 1)$ |
| **弱信息先验** (Weakly informative) | 温和的正则化 | $\theta \sim \text{Beta}(2, 2)$ |
| **强信息先验** (Strongly informative) | 基于先验知识 | $\theta \sim \text{Beta}(50, 50)$ |

**Beta 分布** (Beta distribution) —— 概率参数 $\theta \in [0, 1]$ 的标准先验：

$$\text{Beta}(\theta; \alpha, \beta) = \frac{1}{B(\alpha, \beta)} \theta^{\alpha-1} (1-\theta)^{\beta-1}$$

- $\alpha$："成功"的先验权重
- $\beta$："失败"的先验权重

---

### 共轭先验 (Conjugate Priors)

**问题**：贝叶斯定理需要计算证据：

$$P(D) = \int P(D \mid \theta) P(\theta) \, d\theta$$

这个积分通常是难以处理的——没有解析解。

**解决方案**：选择一个先验 $P(\theta)$，使其与似然 $P(D \mid \theta)$ 相乘后，后验属于**同一分布族**。这就是**共轭先验**。

**为什么有用**：使用共轭先验时，积分有解析解。后验只是参数更新后的同一分布——不需要积分，只需加法。

当先验和后验属于**同一分布族**时，更新只需加法。

| 似然 | 共轭先验 | 后验更新 |
| ---------- | --------------- | ---------------- |
| 二项分布 (Binomial) | Beta 分布 | $\text{Beta}(\alpha + h,\; \beta + t)$ |
| 高斯分布（已知方差） (Gaussian) | 高斯分布 | $\mathcal{N}(\mu_{\text{post}}, \sigma_{\text{post}}^2)$ |
| 泊松分布 (Poisson) | Gamma 分布 | $\text{Gamma}(\alpha + \sum x_i,\; \beta + n)$ |
| 多项分布 (Multinomial) | 狄利克雷分布 (Dirichlet) | $\text{Dir}(\alpha_1 + n_1, \ldots, \alpha_k + n_k)$ |

**Beta-Binomial 示例**：

先验：$\theta \sim \text{Beta}(5, 5)$ —— "可能是公平的"

数据：20 次抛掷，15 次正面

后验：$\theta \mid D \sim \text{Beta}(5+15, 5+5) = \text{Beta}(20, 10)$

后验均值 $= \frac{20}{30} = 0.67$ —— 从先验 (0.5) 向数据 (0.75) 偏移。

---

### 高斯-高斯共轭 (Gaussian-Gaussian Conjugacy)

已知方差 $\sigma^2$，估计均值 $\mu$：

先验：$\mu \sim \mathcal{N}(\mu_0, \sigma_0^2)$

数据：$x_1, \ldots, x_n \sim \mathcal{N}(\mu, \sigma^2)$

后验：$\mu \mid \mathbf{x} \sim \mathcal{N}(\mu_{\text{post}}, \sigma_{\text{post}}^2)$

$$\frac{1}{\sigma_{\text{post}}^2} = \frac{1}{\sigma_0^2} + \frac{n}{\sigma^2}$$

$$\mu_{\text{post}} = \sigma_{\text{post}}^2 \left(\frac{\mu_0}{\sigma_0^2} + \frac{n\bar{x}}{\sigma^2}\right)$$

> [!tip] 直觉理解
> 后验精度 = 先验精度 + 数据精度。数据越多，先验的影响越小。

$$\mu_{\text{post}} = w_{\text{prior}} \cdot \mu_0 + w_{\text{data}} \cdot \bar{x}$$

权重与精度（方差的倒数）成正比。

---

### MAP vs MLE

| | 极大似然估计 (MLE) | 最大后验估计 (MAP) | 完全贝叶斯 |
|---|---|---|---|
| 目标 | $\arg\max P(D \mid \theta)$ | $\arg\max P(\theta \mid D)$ | $P(\theta \mid D)$ 完整分布 |
| 使用先验？ | 否 | 是 | 是 |
| 结果 | 点估计 | 点估计 | 完整分布 |
| 正则化 | 无 | 对应 L2/L1 | 自动 |

**MAP 推导**：

$$\hat{\theta}_{\text{MAP}} = \arg\max P(\theta \mid D) = \arg\max \left[\log P(D \mid \theta) + \log P(\theta)\right]$$

**使用高斯先验** $\theta \sim \mathcal{N}(0, \tau^2)$：

$$\log P(\theta) = -\frac{\theta^2}{2\tau^2} + \text{const}$$

MAP = MLE + L2 正则化，其中 $\lambda = \frac{\sigma^2}{\tau^2}$

---

### MAP 示例：估计学生智商

假设我们测量一名学生的智商。智商分数在总体中服从 $\mathcal{N}(\mu, 15^2)$。

**数据**：5 次测试分数：130, 125, 135, 128, 132

**MLE**（无先验）：

$\hat{\mu}_{\text{MLE}} = \frac{130 + 125 + 135 + 128 + 132}{5} = 130$

**MAP** 使用先验 $\mu \sim \mathcal{N}(100, 15^2)$（"总体均值为 100"）：

使用高斯-高斯共轭公式：

$$\hat{\mu}_{\text{MAP}} = \frac{\sigma^2 \cdot \mu_0 + n \cdot \sigma_0^2 \cdot \bar{x}}{\sigma^2 + n \cdot \sigma_0^2} = \frac{225 \times 100 + 5 \times 225 \times 130}{225 + 5 \times 225} = \frac{22500 + 146250}{1350} = 124.4$$

**比较**：

| 方法 | 估计值 | 解释 |
| ------ | -------- | -------------- |
| MLE | 130.0 | "数据显示 130" |
| MAP | 124.4 | "数据显示 130，但先验将其拉向 100" |
| 完全贝叶斯 | $\mathcal{N}(124.4, \sigma_{\text{post}}^2)$ | "这是 $\mu$ 的完整分布" |

**更多数据**（50 个分数，均值 = 130）：MAP ≈ 129.3 —— 先验影响消失。

**更少数据**（1 个分数 = 130）：MAP ≈ 115 —— 先验强烈拉向 100。

---

## 朴素贝叶斯 (Naive Bayes)

---

### 朴素贝叶斯分类器

**目标**：给定特征 $\mathbf{x} = [x_1, \ldots, x_n]$，预测类别 $y$。

**贝叶斯定理**：

$$P(y \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid y) \cdot P(y)}{P(\mathbf{x})} \propto P(\mathbf{x} \mid y) \cdot P(y)$$

**"朴素"假设**：给定类别 $y$，所有特征**条件独立**：

$$P(x_1, \ldots, x_n \mid y) = \prod_{i=1}^n P(x_i \mid y)$$

**为什么叫"朴素"？** 这个假设在实践中几乎从不成立（特征通常是相关的），但它却出奇地好用。

**预测规则**：

$$\hat{y} = \arg\max_y P(y) \prod_{i=1}^n P(x_i \mid y)$$

在对数空间中（避免下溢）：

$$\hat{y} = \arg\max_y \left[\log P(y) + \sum_{i=1}^n \log P(x_i \mid y)\right]$$

---

### 朴素贝叶斯的三种变体

| 变体 | $P(x_i \mid y)$ 假设 | 适用场景 |
| ------- | --------------------------- | -------- |
| **高斯朴素贝叶斯** (Gaussian NB) | $x_i \mid y \sim \mathcal{N}(\mu_{iy}, \sigma_{iy}^2)$ | 连续特征 |
| **多项式朴素贝叶斯** (Multinomial NB) | $x_i$ 是计数（词频） | 文本分类 |
| **伯努利朴素贝叶斯** (Bernoulli NB) | $x_i \in \{0, 1\}$ | 二值特征 |

**高斯朴素贝叶斯**：

$$P(x_i \mid y) = \frac{1}{\sqrt{2\pi\sigma_{iy}^2}} \exp\!\left(-\frac{(x_i - \mu_{iy})^2}{2\sigma_{iy}^2}\right)$$

每个特征在每个类别中都有自己的均值和方差。

---

### 朴素贝叶斯：垃圾邮件过滤示例

**特征**：邮件中的词频（词袋模型）

**训练**：

- $P(\text{spam}) = \frac{\text{垃圾邮件数量}}{\text{邮件总数}}$
- $P(\text{"free"} \mid \text{spam}) = \frac{\text{垃圾邮件中"free"的出现次数}}{\text{垃圾邮件中的总词数}}$
- 对每个词和每个类别进行计算

**预测**：

$P(\text{spam} \mid \text{free}, \text{money}) \propto P(\text{spam}) \prod_{i} P(w_i \mid \text{spam})$

选择概率更高的类别。

**拉普拉斯平滑** (Laplace smoothing)：防止 $P(w_i \mid y) = 0$（一个未见过的词会使所有概率归零）：

$$P(w_i \mid y) = \frac{\text{count}(w_i, y) + \alpha}{\text{count}(y) + \alpha \cdot |V|}$$

$\alpha = 1$ 是拉普拉斯平滑，$|V|$ 是词汇表大小。

---

### 朴素贝叶斯：逐步示例

将一封包含 "free" 和 "money" 的邮件分类为垃圾邮件或正常邮件。

**训练数据**：10 封邮件（6 封垃圾邮件，4 封正常邮件）

| 词 | 垃圾邮件中（共 6 封） | 正常邮件中（共 4 封） |
| ---- | ------------------ | ----------------- |
| "free" | 5 | 1 |
| "money" | 4 | 0 |
| "meeting" | 1 | 3 |

**第 1 步 — 先验**：

$$P(\text{spam}) = \frac{6}{10} = 0.6, \quad P(\text{ham}) = \frac{4}{10} = 0.4$$

---

**第 2 步 — 似然**（使用拉普拉斯平滑，$|V| = 3$）：

$$P(\text{"free"} \mid \text{spam}) = \frac{5 + 1}{6 + 3} = \frac{6}{9}, \quad P(\text{"free"} \mid \text{ham}) = \frac{1 + 1}{4 + 3} = \frac{2}{7}$$

$$P(\text{"money"} \mid \text{spam}) = \frac{4 + 1}{6 + 3} = \frac{5}{9}, \quad P(\text{"money"} \mid \text{ham}) = \frac{0 + 1}{4 + 3} = \frac{1}{7}$$

**第 3 步 — 后验**：

$$P(\text{spam} \mid \text{free}, \text{money}) \propto 0.6 \times \frac{6}{9} \times \frac{5}{9} = 0.6 \times 0.222 = 0.133$$

$$P(\text{ham} \mid \text{free}, \text{money}) \propto 0.4 \times \frac{2}{7} \times \frac{1}{7} = 0.4 \times 0.041 = 0.016$$

**第 4 步 — 归一化**：$P(\text{spam}|\text{(邮件包含 "free", "money")}) = \frac{0.133}{0.133 + 0.016} = 0.893$

**结果**：89.3% 的概率是垃圾邮件。分类为**垃圾邮件**。

---

### 多项式朴素贝叶斯：词频示例

前面的示例只检查了词**是否出现**（伯努利）。多项式朴素贝叶斯 (Multinomial Naive Bayes) 计算每个词**出现的次数**。

**训练数据**：所有邮件中的词频

| 词 | 垃圾邮件中的计数（共 20 个词） | 正常邮件中的计数（共 15 个词） |
| ---- | ------------------------------ | ----------------------------- |
| "free" | 8 | 1 |
| "money" | 6 | 0 |
| "meeting" | 2 | 5 |
| "project" | 4 | 9 |

---

**新邮件**：包含 "free" 两次，"money" 一次 → 特征向量 $\mathbf{x} = [2, 1, 0, 0]$

**似然**（使用拉普拉斯平滑，$|V| = 4$）：

$$P(\text{"free"} \mid \text{spam}) = \frac{8 + 1}{20 + 4} = \frac{9}{24}, \quad P(\text{"free"} \mid \text{ham}) = \frac{1 + 1}{15 + 4} = \frac{2}{19}$$

$$P(\text{"money"} \mid \text{spam}) = \frac{6 + 1}{20 + 4} = \frac{7}{24}, \quad P(\text{"money"} \mid \text{ham}) = \frac{0 + 1}{15 + 4} = \frac{1}{19}$$

**后验**（以词频作为指数）：

$$P(\text{spam}) \cdot P(\text{"free"} \mid \text{spam})^2 \cdot P(\text{"money"} \mid \text{spam})^1 = 0.6 \times \left(\frac{9}{24}\right)^2 \times \frac{7}{24} = 0.0295$$

$$P(\text{ham}) \cdot P(\text{"free"} \mid \text{ham})^2 \cdot P(\text{"money"} \mid \text{ham})^1 = 0.4 \times \left(\frac{2}{19}\right)^2 \times \frac{1}{19} = 0.00047$$

**归一化**：$P(\text{spam}) = \frac{0.0295}{0.0295 + 0.00047} = 0.984$

**结果**：98.4% 的概率是垃圾邮件。重复出现的 "free" 使其比伯努利版本更有把握。

---

### 朴素贝叶斯：优缺点

**优点**：

- 训练和预测速度极快
- 无需迭代优化
- 在高维数据（文本）中表现良好
- 不易过拟合（强正则化）
- 可解释性强

**缺点**：

- 强独立性假设
- 特征相关时表现较差
- 概率估计校准不佳
- 对未见过的特征敏感

**为什么朴素假设仍然有效？** 分类只需要**排序**，而不需要精确的概率。即使概率值被扭曲，只要正确的类别排名最高，预测就是正确的。

---

## 贝叶斯网络 (Bayesian Networks)

---

### 什么是贝叶斯网络？

**贝叶斯网络**是一个**有向无环图 (Directed Acyclic Graph, DAG)**，表示变量之间的因果/依赖关系。

**组成部分**：

- **节点**：随机变量
- **有向边**：因果方向（$A \to B$ 表示 $A$ 影响 $B$）
- **条件概率表 (Conditional Probability Table, CPT)**：$P(X_i \mid \text{parents}(X_i))$

**关键性质**：在给定父节点的条件下，每个节点与所有非后代节点条件独立。

$$P(X_i \mid \text{parents}(X_i), \text{non-descendants}) = P(X_i \mid \text{parents}(X_i))$$

**联合分布分解**：

$$P(X_1, X_2, \ldots, X_n) = \prod_{i=1}^n P(X_i \mid \text{parents}(X_i))$$

这是**贝叶斯网络的链式法则**——将完整的联合分布分解为局部条件概率。

---

### 示例：洒水器问题

**场景**：

- 多云 (C) 影响下雨 (R) 和洒水器 (S)
- 下雨 (R) 和洒水器 (S) 影响草地湿润 (W)

**图结构**：

$$C \to R, \quad C \to S, \quad R \to W, \quad S \to W$$

**联合分布**：

$$P(C, R, S, W) = P(C) \cdot P(R \mid C) \cdot P(S \mid C) \cdot P(W \mid R, S)$$

**条件概率表**：

$P(C=1) = 0.5$

$P(R=1 \mid C=1) = 0.8$, $P(R=1 \mid C=0) = 0.1$

$P(S=1 \mid C=1) = 0.1$, $P(S=1 \mid C=0) = 0.5$

$P(W=1 \mid R, S)$:

| R | S | P(W=1) |
|---|---|--------|
| 0 | 0 | 0.001  |
| 0 | 1 | 0.9    |
| 1 | 0 | 0.99   |
| 1 | 1 | 0.999  |

---

### 贝叶斯网络中的推断

**问题**：已知草地是湿的 (W=1)，洒水器打开的概率是多少？

$$P(S=1 \mid W=1) = \frac{P(S=1, W=1)}{P(W=1)} = \frac{\sum_{C,R} P(C, R, S=1, W=1)}{\sum_{C,R,S} P(C, R, S, W=1)}$$

**代码**：

```python
P_C = {0: 0.5, 1: 0.5}
P_R_given_C = {(0,0): 0.9, (0,1): 0.1, (1,0): 0.2, (1,1): 0.8}
P_S_given_C = {(0,0): 0.5, (0,1): 0.5, (1,0): 0.9, (1,1): 0.1}
P_W_given_RS = {(0,0,0): 0.999, (0,0,1): 0.001, (0,1,0): 0.1, (0,1,1): 0.9,
                (1,0,0): 0.01, (1,0,1): 0.99, (1,1,0): 0.001, (1,1,1): 0.999}
num = sum(P_C[c] * P_R_given_C[(c,r)] * P_S_given_C[(c,1)] * P_W_given_RS[(r,1,1)]
          for c in [0,1] for r in [0,1])
den = sum(P_C[c] * P_R_given_C[(c,r)] * P_S_given_C[(c,s)] * P_W_given_RS[(r,s,1)]
          for c in [0,1] for r in [0,1] for s in [0,1])
print(f"P(S=1|W=1) = {num/den:.4f}")  # ≈ 0.2572
```

---

### 条件独立结构

三种基本结构决定独立性：

### 链式 (Chain)

$A \to B \to C$

给定 $B$：$A \perp C$

未给定 $B$：$A$ 和 $C$ 相关

### 叉式 (Fork)

$A \leftarrow B \to C$

给定 $B$：$A \perp C$

未给定 $B$：$A$ 和 $C$ 相关

### 对撞 (Collider)

$A \to B \leftarrow C$

给定 $B$：$A$ 和 $C$ 相关！

未给定 $B$：$A \perp C$

**对撞结构违反直觉**：$A$ 和 $C$ 在不观察 $B$ 时是独立的，但一旦观察 $B$ 就变得相关。这被称为**解释消除** (explaining away) 效应。

---

### d-分离 (d-Separation)

一种用于判断贝叶斯网络中条件独立性的算法：

**规则**：给定观测集 $Z$，$X$ 和 $Y$ 被 $Z$ **d-分离**，当且仅当 $X$ 和 $Y$ 之间的**所有路径**都被阻断。

如果路径包含满足以下条件之一的节点，则该路径被阻断：

1. **链式或叉式**：中间节点在 $Z$ 中（已观测）
2. **对撞**：中间节点及其后代**不在** $Z$ 中（未观测）

**d-分离 $\Rightarrow$ 条件独立**：

$$X \perp Y \mid Z \quad \text{如果 } X \text{ 和 } Y \text{ 在给定 } Z \text{ 时被 d-分离}$$

**在洒水器问题中**：

- $C$ 和 $W$ 不独立（路径 $C \to R \to W$ 和 $C \to S \to W$）
- 给定 $R$ 和 $S$：$C \perp W$（所有路径被阻断）

---

## 贝叶斯推断 (Bayesian Inference)

---

### 完整的贝叶斯推断流程

**第 1 步**：选择模型（似然）

$$P(D \mid \theta) = \prod_{i=1}^N P(x_i \mid \theta)$$

**第 2 步**：选择先验

$$P(\theta)$$

**第 3 步**：计算后验

$$P(\theta \mid D) = \frac{P(D \mid \theta) P(\theta)}{P(D)} \propto P(D \mid \theta) P(\theta)$$

**第 4 步**：使用后验

- **点估计**：后验均值、中位数或众数 (MAP)
- **区间估计**：可信区间 (credible interval)
- **预测**：$P(x_{\text{new}} \mid D) = \int P(x_{\text{new}} \mid \theta) P(\theta \mid D) \, d\theta$

---

### 推导后验预测分布

我们想要 $P(x_{\text{new}} \mid D)$ —— 给定已观测数据，新数据的概率。

**第 1 步**：对 $\theta$ 边际化（全概率公式）：

$$P(x_{\text{new}} \mid D) = \int P(x_{\text{new}}, \theta \mid D) \, d\theta$$

**第 2 步**：对联合分布应用链式法则：

$$P(x_{\text{new}}, \theta \mid D) = P(x_{\text{new}} \mid \theta, D) \cdot P(\theta \mid D)$$

**第 3 步**：关键假设——给定 $\theta$，新数据 $x_{\text{new}}$ 与旧数据 $D$ 独立：

$$P(x_{\text{new}} \mid \theta, D) = P(x_{\text{new}} \mid \theta)$$

这很合理：一旦你知道了参数 $\theta$，旧数据不会提供额外信息。

**第 4 步**：代回：

$$\boxed{\;P(x_{\text{new}} \mid D) = \int P(x_{\text{new}} \mid \theta) \cdot P(\theta \mid D) \, d\theta\;}$$

---

### 后验预测分布 (Posterior Predictive Distribution)

要预测新数据点 $x_{\text{new}}$，对所有可能的 $\theta$ 积分：

$$P(x_{\text{new}} \mid D) = \int P(x_{\text{new}} \mid \theta) P(\theta \mid D) \, d\theta$$

> [!tip] 直觉理解
> 不要使用单一的"最佳" $\theta$ —— 要对**所有** $\theta$ 按其后验概率加权求平均。

**与 MLE 比较**：

| | MLE/MAP | 贝叶斯 |
|---|---|---|
| 预测 | $P(x_{\text{new}} \mid \hat{\theta})$ | $\int P(x_{\text{new}} \mid \theta) P(\theta \mid D) d\theta$ |
| 不确定性 | 忽略 $\hat{\theta}$ 的不确定性 | 自动包含 |
| 小样本 | 过拟合风险高 | 先验起到正则化作用 |

---

**数值近似**：从后验中采样：

$$P(x_{\text{new}} \mid D) \approx \frac{1}{S} \sum_{s=1}^S P(x_{\text{new}} \mid \theta^{(s)}), \quad \theta^{(s)} \sim P(\theta \mid D)$$

---

### 后验预测：具体示例

估计一名学生的智商。观测到 5 个分数，均值为 130。

**后验**：$\mu \mid D \sim \mathcal{N}(124.4, 6^2)$（来自前面的 MAP 示例）

现在预测：这名学生下一次测试会得多少分？

**MLE 方法**（点估计）：

模型说 $x_{\text{new}} \sim \mathcal{N}(\mu, 15^2)$，但 $\mu$ 未知。MLE 代入 $\hat{\mu} = 130$：

$$x_{\text{new}} \sim \mathcal{N}(\hat{\mu}, 15^2) = \mathcal{N}(130, 15^2)$$

将 $\hat{\mu}$ 当作真实的 $\mu$。预测：均值 130，标准差 15。

**贝叶斯方法**（对后验积分）：

$$P(x_{\text{new}} \mid D) = \int \mathcal{N}(x_{\text{new}} \mid \mu, 15^2) \cdot \mathcal{N}(\mu \mid 124.4, 6^2) \, d\mu$$

结果：$x_{\text{new}} \mid D \sim \mathcal{N}(124.4, 15^2 + 6^2) = \mathcal{N}(124.4, 261)$

预测：均值 124.4，标准差 $\sqrt{261} \approx 16.2$

---

**比较**：

| | MLE | 贝叶斯 |
|---|---|---|
| 预测均值 | 130.0 | 124.4 |
| 预测标准差 | 15.0 | 16.2 |
| 为什么不同？ | 忽略 $\mu$ 的不确定性 | 包含 $\mu$ 的不确定性 |

**额外的不确定性**（$16.2 > 15$）来源于不知道真实的 $\mu$。贝叶斯预测对未知事物更加诚实。

**采样解释**：

1. 从后验 $\mathcal{N}(124.4, 6^2)$ 中抽取 $\mu^{(s)}$
2. 从 $\mathcal{N}(\mu^{(s)}, 15^2)$ 中抽取 $x^{(s)}$
3. 重复 1000 次 → $x^{(s)}$ 的直方图近似 $P(x_{\text{new}} \mid D)$

每个样本是一个两步过程：先采样参数，再根据该参数采样数据。

---

### 贝叶斯线性回归 (Bayesian Linear Regression)

标准回归：$\hat{y} = X\hat{\theta}$（一个点估计）

贝叶斯回归：对 $\theta$ 设置先验，得到后验分布

先验：$\theta \sim \mathcal{N}(0, \alpha^{-1}I)$（$\alpha$ 控制正则化强度）

似然：$y \mid X, \theta \sim \mathcal{N}(X\theta, \beta^{-1}I)$（$\beta = 1/\sigma^2$）

后验（高斯-高斯共轭）：

$$\theta \mid X, y \sim \mathcal{N}(\mu_N, \Sigma_N)$$

$$\Sigma_N = (\alpha I + \beta X^T X)^{-1}$$

$$\mu_N = \beta \Sigma_N X^T y$$

**预测分布**也是高斯分布：

$$y_{\text{new}} \mid x_{\text{new}}, X, y \sim \mathcal{N}(\mu_N^T x_{\text{new}},\; \sigma_{\text{pred}}^2)$$

预测方差有**两部分**：噪声 $\beta^{-1}$ + 参数不确定性。

---

### 当后验没有解析解时

大多数现实世界的后验无法解析计算。

**方法 1：MCMC（马尔可夫链蒙特卡洛，Markov Chain Monte Carlo）**

从后验中采样，用样本近似。

- **Metropolis-Hastings**：通用采样器
- **Gibbs 采样**：一次采样一个维度
- **HMC（哈密顿蒙特卡洛，Hamiltonian Monte Carlo）**：利用梯度信息进行高效采样

**方法 2：变分推断 (Variational Inference)**

用简单分布 $q(\theta)$ 近似后验，最小化 KL 散度：

$$\min_{q} D_{\text{KL}}(q(\theta) \| P(\theta \mid D))$$

比 MCMC 更快，但可能牺牲精度。

**方法 3：拉普拉斯近似 (Laplace Approximation)**

将后验近似为高斯分布（在 MAP 处进行二阶泰勒展开）。

---

### 贝叶斯 vs 频率派 (Bayesian vs Frequentist)

| | 频率派 (Frequentist) | 贝叶斯 (Bayesian) |
|---|---|---|
| 概率含义 | 长期频率 | 信念程度 |
| 参数 | 固定但未知 | 随机变量 |
| 推断 | 点估计 + 置信区间 | 后验 + 可信区间 |
| 先验 | 不使用 | 核心组成部分 |
| 计算 | 通常有解析解 | 通常是近似的 |
| 小样本 | 不稳定 | 先验有助于正则化 |

**置信区间 vs 可信区间**：

- 95% 置信区间 (Confidence interval)："如果我们重复实验 100 次，95 个区间包含真实值"
- 95% 可信区间 (Credible interval)："参数以 95% 的概率落在这个区间内"

后者更直观，但需要先验。

---

### 总结

### 基础

- 贝叶斯定理
- 先验 → 后验
- 共轭先验
- MAP vs MLE

### 模型

- 朴素贝叶斯分类器
- 贝叶斯网络
- d-分离
- 条件独立

### 推断

- 后验预测分布
- 贝叶斯线性回归
- MCMC / 变分推断
- 贝叶斯 vs 频率派

$$\boxed{\;P(H \mid D) = \frac{P(D \mid H) \cdot P(H)}{P(D)}\;}$$

贝叶斯思维的精髓：**用数据更新信念，用后验做出决策。**
