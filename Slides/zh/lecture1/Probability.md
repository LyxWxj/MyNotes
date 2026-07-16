## 第三部分：概率与统计 (Probability & Statistics)
### 你将需要的内容
本节涵盖完成 W0D5 教程所需的所有内容：
**教程 1 — 分布与采样 (Distributions & Sampling)**
- 均匀分布 (Uniform)、二项分布 (Binomial)、泊松分布 (Poisson)、高斯分布 (Gaussian)
- 使用 NumPy 进行采样
- 直方图 (Histograms)
**教程 2 — 推断 (Inference)**
- 条件概率 / 联合概率 / 边际概率 (Conditional / Joint / Marginal probability)
- 似然函数与最大似然估计 (Likelihood & MLE)
- 贝叶斯推断 (Bayesian inference)：先验 → 后验 (Prior → Posterior)
- 马尔可夫链 (Markov chains)

---

### 随机变量与分布 (Random Variables & Distributions)

---

### 随机变量 (Random Variable)
**随机变量 (random variable)** $X: \Omega \to \mathbb{R}$ 将随机结果映射为数值。
**离散型 (Discrete)**：取可数个值
概率质量函数 (PMF)：$P(X = x_k) = p_k$，$\sum_k p_k = 1$
示例：脉冲计数 $X \in \{0, 1, 2, \ldots\}$
**连续型 (Continuous)**：取任意实数值
概率密度函数 (PDF)：$p(x) \geq 0$，$\int_{-\infty}^{\infty} p(x)\, dx = 1$
$P(a \leq X \leq b) = \int_a^b p(x)\, dx$
注意：对于连续型随机变量，$P(X = a) = 0$。

---

### 均匀分布 (Uniform Distribution)
$X \sim \mathcal{U}(a, b)$ — 在 $[a, b]$ 区间内所有值具有相等的概率：
$$p(x) = \frac{1}{b - a} \quad \text{for } x \in [a, b]$$
**NumPy 采样**：
```python
np.random.seed(0)                          # reproducible results
samples = np.random.uniform(0, 1, size=10) # 10 samples from U(0,1)
```
**应用**：随机初始化、探索状态空间、随机游走 (random walk)。
**随机游走 (random walk)** 组合均匀步长：每一步在 $x$ 和 $y$ 方向上随机移动：
```python
x[step+1] = x[step] + (np.random.uniform() - 0.5) * step_size
#                       ^^^^^^^^^^^^^^^^^^^^^^^^
#                       centered around 0: range [-0.5, 0.5]
```

---

### 二项分布 (Binomial Distribution)
$n$ 次独立的二元试验 (binary trials)，每次成功概率为 $p$：
$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$
其中 $\binom{n}{k} = \frac{n!}{k!(n-k)!}$ 计算从 $n$ 次试验中选择 $k$ 次成功的方式数。
**示例**：大鼠在 T 型迷宫中，10 次试验，$p = 0.5$（随机选择）。左转 7 次的概率是多少？
$$P(k=7 \mid n=10, p=0.5) = \binom{10}{7}(0.5)^7(0.5)^3 = 120 \times 0.000977 = 0.117$$
**采样**：
```python
samples = np.random.binomial(n=10, p=0.5, size=1000)
# each element = number of left turns in 10 trials
# histogram peaks at k=5 (the expected value np)
```

---

### 泊松分布 (Poisson Distribution)
对固定时间间隔内事件数量建模，平均速率为 $\lambda$：
$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$$
**示例**：一个神经元以平均速率 $\lambda = 4$ 次脉冲/秒发放。在一秒内恰好发放 7 次脉冲的概率是多少？
$$P(k=7 \mid \lambda=4) = \frac{4^7 e^{-4}}{7!} = \frac{16384 \times 0.0183}{5040} \approx 0.060$$
**采样**：
```python
spike_counts = np.random.poisson(lam=4, size=100)
# each element = number of spikes in one interval
# histogram is asymmetric for small λ (can't have negative spikes)
```
**何时使用**：计数离散事件（脉冲、光子到达、突变）。当 $n \to \infty$，$p \to 0$，$np = \lambda$ 时，泊松分布是二项分布的极限。

---

### 高斯（正态）分布 (Gaussian / Normal Distribution)
最重要的连续分布：
$$X \sim \mathcal{N}(\mu, \sigma^2) \quad \Rightarrow \quad p(x) = \frac{1}{sigma\sqrt{2\pi}} \exp\!\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$
| 参数 | 符号 | 含义 |
| --------- | ------ | ------- |
| 均值 (Mean)      | $\mu$  | 峰值的中心 |
| 标准差 (Std Dev)   | $\sigma$ | 宽度 / 分散程度 |
| 方差 (Variance)  | $\sigma^2$ | 分散程度的平方 |
**68-95-99.7 法则**：$P(\mu \pm 1\sigma) \approx 68\%$，$P(\mu \pm 2\sigma) \approx 95\%$，$P(\mu \pm 3\sigma) \approx 99.7\%$
**采样**：
```python
samples = np.random.normal(mu=5, sigma=1, size=1000)
```

---

### 从零实现高斯分布 (Implementing a Gaussian from Scratch)
教程要求你手动实现 PDF：
```python
def my_gaussian(x_points, mu, sigma):
    px = 1 / (2 * np.pi * sigma**2)**0.5 * np.exp(-(x_points - mu)**2 / (2 * sigma**2))
    # Normalize numerically (step size = 0.1)
    px = px / (0.1 * sum(px))
    return px
```
**为什么要归一化？** 解析 PDF 在 $(-\infty, \infty)$ 上积分为 1，但我们在有限网格上计算（$-8$ 到 $9$，步长 $0.1$）。数值求和 $\neq 1$，因此我们除以 $0.1 \times \text{sum}$ 来强制归一化。
**使用 scipy**（替代方案）：
```python
from scipy.stats import norm
px = norm.pdf(x, loc=mu, scale=sigma)  # analytical, already normalized
```

---

### 直方图作为密度估计器 (Histogram as Density Estimator)
```python
plt.hist(samples, bins=30, density=True)
```
`density=True` 进行归一化使总面积 = 1，使其可与 PDF 进行比较。
**样本统计量 (Sample statistics)**（来自教程 1）：
```python
np.mean(samples)    # sample mean → converges to μ
np.std(samples)     # sample std → converges to σ
```
**关键洞察**：当样本量较少时（$n < 50$），直方图噪声较大。当 $n > 500$ 时，钟形曲线形状变得清晰。这就是**大数定律 (Law of Large Numbers)** 的实际体现。

---

### 概率规则 (Probability Rules)

---

### 条件概率、联合概率和边际概率 (Conditional, Joint, and Marginal)
对于事件 $A$ 和 $B$，其中 $P(B) > 0$：
**条件概率 (Conditional probability)** — 在 $B$ 发生条件下 $A$ 发生的概率：
$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}$$
**联合概率 (Joint probability)** — $A$ 和 $B$ 同时发生：
$$P(A \cap B) = P(A \mid B) \cdot P(B) = P(B \mid A) \cdot P(A)$$
**边际概率 (Marginal probability)** — 无论 $B$ 是否发生，$A$ 的概率：
$$P(A) = P(A \mid B_1)P(B_1) + P(A \mid B_2)P(B_2) + \cdots = \sum_i P(A \mid B_i)P(B_i)$$
对于连续情况：$P(A) = \int P(A \mid y)\, p(y)\, dy$

---

### 示例：视觉神经元 (Example: Visual Neurons)
40% 对水平方向有响应（$P(h_+) = 0.4$），30% 对垂直方向有响应（$P(v_+) = 0.3$）。
**独立 (Independence)** → 联合概率 = 乘积：
$$P(h_+ \cap v_+) = P(h_+) \cdot P(v_+) = 0.4 \times 0.3 = 0.12$$
**不独立** → 使用条件概率：
已知 $P(h_+ \mid v_+) = 0.1$，则：
$$P(h_+ \cap v_+) = P(h_+ \mid v_+) \cdot P(v_+) = 0.1 \times 0.3 = 0.03$$
**边际恢复 (Marginal recovery)**（验证）：
$$P(v_+) = P(v_+ \mid h_+)P(h_+) + P(v_+ \mid h_0)P(h_0)$$
你需要 $P(v_+ \mid h_+)$ 和 $P(v_+ \mid h_0)$ — 从联合概率和边际概率计算得出。

---

### 贝叶斯定理 (Bayes' Theorem)
$$\boxed{\;P(Y \mid X) = \frac{P(X \mid Y) \cdot P(Y)}{P(X)}\;}$$
### 先验 $P(Y)$ (Prior)
数据**之前**的信念
### 似然 $P(X \mid Y)$ (Likelihood)
给定假设下的数据
### 后验 $P(Y \mid X)$ (Posterior)
更新后的信念
**医学检测示例**：疾病率 1%，检测灵敏度 95%，假阳性率 10%。
$$P(\text{disease} \mid +) = \frac{0.95 \times 0.01}{0.95 \times 0.01 + 0.10 \times 0.99} = 8.8\%$$
先验（罕见疾病）在证据较弱（一次检测）时占主导地位。

---

### 似然与最大似然估计 (Likelihood & MLE)

---

### 对数规则 (Logarithm Rules)
为什么要对乘积取对数？因为对数将乘法转换为加法：
| 规则 | 公式 | 重要性 |
| ---- | ------- | -------------- |
| 乘积 → 求和 | $\log(a \cdot b) = \log a + \log b$ | 似然是乘积 → 对数似然是求和 |
| 幂次 → 乘法 | $\log(a^k) = k \log a$ | 简化指数项 |
| 单调性 | $a > b \Leftrightarrow \log a > \log b$ | $\arg\max L = \arg\max \log L$ — 结果相同 |
**数值原因**：对于 1000 个数据点，$p(x_i) \approx 0.01$ → $L = 0.01^{1000} = 10^{-2000}$ → **下溢 (underflow)** 为零。
$\log L = 1000 \times \log(0.01) = -4605$ — 一个可处理的数值。
**关键性质**：最大化 $L$ 等价于最大化 $\log L$（对数是单调递增的）。因此我们可以自由地在两者之间切换。

---

### 似然函数 (Likelihood Function)
给定数据 $\mathbf{x} = (x_1, \ldots, x_n)$，参数 $(\mu, \sigma)$ 的**似然**为：
$$L(\mu, \sigma) = \prod_{i=1}^n p(x_i \mid \mu, \sigma)$$
**对数似然 (Log-likelihood)**（对乘积取对数）：
$$\log L = \log \prod_{i=1}^n p(x_i \mid \mu, \sigma) = \sum_{i=1}^n \log\, p(x_i \mid \mu, \sigma)$$
**代码**：
```python
from scipy.stats import norm
def compute_log_likelihood(x, mu, sigma):
    return np.sum(norm.logpdf(x, mu, sigma))
x = np.random.normal(5, 1, size=1000)
print(compute_log_likelihood(x, 4, 0.1))   # bad guess → very negative
print(compute_log_likelihood(x, 5, 1))     # good guess → less negative
```
对数似然始终 $\leq 0$。越接近 0 = 拟合越好。

---

### 最大似然估计 (Maximum Likelihood Estimation)
找到使对数似然最大化的参数：
$$\hat{\theta}_{\text{MLE}} = \arg\max_{\theta} \log L(\theta)$$
**高斯分布的解析解**（令导数为 0）：
$$\hat{\mu} = \frac{1}{n}\sum_{i=1}^n x_i, \qquad \hat{\sigma}^2 = \frac{1}{n}\sum_{i=1}^n (x_i - \hat{\mu})^2$$
**数值解**（当无闭式解时）：
```python
from scipy.optimize import minimize
def neg_log_likelihood(theta, x):
    mu, sigma = theta
    return -np.sum(norm.logpdf(x, mu, sigma))
x = np.random.normal(5, 1, size=1000)
result = minimize(neg_log_likelihood, x0=[2, 2], args=(x,),
                  bounds=((None, None), (0, None)))
print(f"mu={result.x[0]:.3f}, sigma={result.x[1]:.3f}")
```
**为什么要最小化负值？** 最大化 $L$ = 最小化 $-L$。`scipy.optimize.minimize` 只能进行最小化。

---

### 似然的网格搜索 (Grid Search for Likelihood)
在优化之前，通过网格搜索建立直觉：
```python
mean_vals = np.linspace(1, 10, 10)
sigma_vals = np.array([0.7, 0.8, 0.9, 1, 1.2, 1.5, 2, 3, 4, 5])
likelihood = np.zeros((len(sigma_vals), len(mean_vals)))
for i, mu in enumerate(mean_vals):
    for j, sigma in enumerate(sigma_vals):
        likelihood[j, i] = np.sum(norm.logpdf(x, mu, sigma))
```
绘制为热力图。热力图的峰值 = 最大似然估计值。应该接近生成数据的真实 $(\mu, \sigma)$。

---

### 贝叶斯推断 (Bayesian Inference)

---

### 先验、似然、后验 (Prior, Likelihood, Posterior)
$$\underbrace{P(\theta \mid D)}_{\text{后验}} = \frac{\overbrace{P(D \mid \theta)}^{\text{似然}} \cdot \underbrace{P(\theta)}_{\text{先验}}}{P(D)}$$
**共轭先验 (Conjugate priors)**：当先验 × 似然 = 与先验相同的分布族时，更新只是算术运算。
**Beta-二项共轭 (Beta-Binomial conjugacy)**：
| | 分布 | 参数 |
|---|---|---|
| 先验 (Prior) | $\text{Beta}(\alpha, \beta)$ | 编码关于概率 $\theta$ 的信念 |
| 数据 (Data) | $n$ 次抛掷中 $h$ 次正面，$t$ 次反面 | |
| 后验 (Posterior) | $\text{Beta}(\alpha + h, \beta + t)$ | 更新后的信念 |
**Beta 概率密度函数 (Beta PDF)**：$f(\theta; \alpha, \beta) = \frac{1}{B(\alpha, \beta)}\theta^{\alpha-1}(1-\theta)^{\beta-1}$

---

### 贝叶斯推断：抛硬币示例 (Bayesian Inference: Coin Flip Example)
先验：$\theta \sim \text{Beta}(5, 5)$ — "可能是公平的，以 0.5 为中心"
数据：20 次抛掷，15 次正面
后验：$\theta \mid D \sim \text{Beta}(5+15, 5+5) = \text{Beta}(20, 10)$
后验均值 = $\frac{20}{20+10} = 0.67$ — 从先验 (0.5) 向数据 (0.75) 偏移。
**Beta PDF 代码**：
```python
from scipy.stats import beta
theta = np.linspace(0, 1, 100)
prior_pdf = beta.pdf(theta, 5, 5)
posterior_pdf = beta.pdf(theta, 20, 10)
```
**最大似然估计 (MLE)** = $15/20 = 0.75$（忽略先验）。**最大后验估计 (MAP)** = Beta(20,10) 的众数 $\approx 0.67$（包含先验）。
随着数据增多，后验集中在最大似然估计上 — 先验被"冲刷掉"。

---

### 经典推断与贝叶斯推断 (Classical vs Bayesian Inference)
```python
# Classical (MLE)
mean_classic = np.mean(x)
var_classic = np.var(x)
# Bayesian (with prior as pseudo-data)
prior = np.array([4, 6])              # two pseudo-observations
x_with_prior = np.hstack((x, prior))
mean_bayes = np.mean(x_with_prior)
var_bayes = np.var(x_with_prior)
```
**比较**：当数据点较少时，贝叶斯估计更稳定（由先验正则化）。当数据点较多时，两者收敛到相同的结果。
**要点**：贝叶斯推断给出完整的分布，而不仅仅是点估计。在数据有限时很有帮助。

---

### 马尔可夫链 (Markov Chains)

---

### 马尔可夫性质 (The Markov Property)
如果一个随机过程的未来仅取决于现在，而不取决于过去，则该过程具有**马尔可夫性质 (Markov property)**：
$$P(X_{t+1} \mid X_t, X_{t-1}, \ldots, X_0) = P(X_{t+1} \mid X_t)$$
**类比**：醉汉的下一步只取决于他现在的位置，而不取决于他如何到达那里。整个历史都是无关紧要的。
**与非马尔可夫过程对比**：
- 马尔可夫："我现在在十字路口" → 下一步是确定的
- 非马尔可夫："我在十字路口，但我从北方来" → 下一步可能不同
实际上，许多系统并不是真正的马尔可夫过程，但我们可以通过在状态中包含足够的信息来*使其*成为马尔可夫过程。例如：对于运动物体，仅位置不是马尔可夫的，但位置 + 速度是。
**重要性**：马尔可夫性质使我们能够在不跟踪完整历史的情况下计算 $P(X_{t+k} \mid X_t)$。这是隐马尔可夫模型 (Hidden Markov Models)、MCMC 采样和强化学习 (reinforcement learning) 的基础。

---

### 状态转移矩阵 (State Transition Matrix)
对于具有 $n$ 个状态的系统，**转移矩阵 (transition matrix)** $T$ 是一个 $n \times n$ 矩阵，其中：
$$T_{ij} = P(\text{next state} = j \mid \text{current state} = i)$$
**性质**：
- 每行是一个概率分布：$\sum_{j=1}^n T_{ij} = 1$
- 所有元素非负：$T_{ij} \geq 0$
- $T$ 是一个**随机矩阵 (stochastic matrix)**（行随机矩阵）
**示例**：大鼠在 3 区域迷宫中（暗区 = 1，筑巢区 = 2，亮区 = 3）
$$T = \begin{bmatrix} 0.2 & 0.6 & 0.2 \\ 0.6 & 0.3 & 0.1 \\ 0.8 & 0.2 & 0.0 \end{bmatrix}$$
阅读第 1 行："如果大鼠在区域 1（暗区），有 20% 的概率停留，60% 的概率移动到筑巢区，20% 的概率移动到亮区。"

---

**解读矩阵**：
| 条目 | 值 | 含义 |
| ----- | ----- | ------- |
| $T_{11} = 0.2$ | 留在暗区 | 大鼠倾向于离开暗区 |
| $T_{21} = 0.6$ | 筑巢区 → 暗区 | 大鼠经常从筑巢区退回暗区 |
| $T_{31} = 0.8$ | 亮区 → 暗区 | 大鼠强烈避免留在亮区 |
| $T_{33} = 0.0$ | 亮区 → 亮区 | 大鼠从不留在亮区 |

---

### 状态演化：从矩阵到概率 (State Evolution: From Matrix to Probabilities)
如何计算经过 $k$ 步后处于每个状态的概率？
**一步**：如果当前状态已知（例如在区域 2），表示为行向量 $\mathbf{p}_0 = [0, 1, 0]$：
$$\mathbf{p}_1 = \mathbf{p}_0 \cdot T = [0, 1, 0] \cdot T = [0.6,\; 0.3,\; 0.1]$$
1 步后：60% 概率在暗区，30% 在筑巢区，10% 在亮区。
**两步**：再次应用 $T$：
$$\mathbf{p}_2 = \mathbf{p}_1 \cdot T = \mathbf{p}_0 \cdot T^2$$
**$k$ 步**：$\mathbf{p}_k = \mathbf{p}_0 \cdot T^k$
**代码**：
```python
T = np.array([[0.2, 0.6, 0.2],
              [0.6, 0.3, 0.1],
              [0.8, 0.2, 0.0]])
p0 = np.array([0, 1, 0])               # start in area 2
p4 = p0 @ np.linalg.matrix_power(T, 4)  # after 4 transitions
print(f"P(area 2 after 4 steps) = {p4[1]:.4f}")  # 0.4311
```
**矩阵乘法有效的原因**：$\mathbf{p}_0 \cdot T$ 对每个 $j$ 计算 $\sum_i p_i \cdot T_{ij}$ — 这正是全概率公式 (law of total probability) $P(\text{next}=j) = \sum_i P(\text{next}=j \mid \text{current}=i) P(\text{current}=i)$。

---

### 稳态与时间平均 (Steady State & Time Averaging)
当 $k \to \infty$ 时，$\mathbf{p}_k$ 收敛到**稳态 (steady state)** $\boldsymbol{\pi}$，与起始位置无关：
$$\boldsymbol{\pi} = \boldsymbol{\pi} \cdot T$$
> [!tip] 直觉
> 经过多次转移后，系统"遗忘"了它的起始位置。稳态是在每个区域中花费时间的长期比例。
**代码**（通过运行 100 步近似）：
```python
p_random = np.ones((1, 3)) / 3          # uniform initial distribution
p_avg = p_random @ np.linalg.matrix_power(T, 100)
# Result: ≈ [[0.447, 0.421, 0.132]]
```
**时间平均**：从任意状态开始，经过许多步后在每个区域的时间比例：
```python
p_random = np.ones((1, 3)) / 3          # uniform initial distribution
p_avg = p_random @ np.linalg.matrix_power(T, 100)
print(p_avg)  # ≈ [[0.447, 0.421, 0.132]]
```
**关键洞察**：稳态不依赖于起始状态（对于遍历链 (ergodic chains)）。这是大数定律在马尔可夫链中的类比。
**与朴素贝叶斯的联系**：分类分布 (categorical distribution)（多结果伯努利分布）出现在马尔可夫链中 — $T$ 的每一行是关于下一个状态的分类分布。

---

### 总结 (Summary)
### 分布 (Distributions)
- **均匀分布 (Uniform)**：`np.random.uniform`
- **二项分布 (Binomial)**：`np.random.binomial`
- **泊松分布 (Poisson)**：`np.random.poisson`
- **高斯分布 (Gaussian)**：`np.random.normal`
### 概率规则 (Probability Rules)
- **条件概率 (Conditional)**：$P(A|B) = P(A,B)/P(B)$
- **联合概率 (Joint)**：$P(A,B) = P(A|B)P(B)$
- **边际概率 (Marginal)**：求和/积分消去
- **贝叶斯定理 (Bayes)**：后验 ∝ 似然 × 先验
### 推断 (Inference)
- **似然 (Likelihood)**：`norm.logpdf`，求和
- **最大似然估计 (MLE)**：`scipy.optimize.minimize`
- **贝叶斯 (Bayesian)**：Beta 先验 + 二项数据
- **马尔可夫 (Markov)**：$\mathbf{p} \cdot T^k$
$$\boxed{\;P(Y \mid X) = \frac{P(X \mid Y) \cdot P(Y)}{P(X)}\;}$$
