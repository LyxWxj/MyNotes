# 马尔可夫过程与卡尔曼滤波

Hidden Markov Models · Kalman Filter · Forward Inference · Smoothing

---

## 总览

本课件涵盖三个核心主题，从离散隐状态到连续隐状态的推断：

| 主题 | 模型 | 隐状态类型 | 推断方法 |
|------|------|-----------|----------|
| **隐马尔可夫模型 (HMM)** | 离散状态 + 高斯观测 | 离散 ($s_t \in \{0,1\}$) | 前向推断 (Forward) |
| **卡尔曼滤波 (1D)** | 连续状态 + 高斯观测 | 连续 ($s_t \in \mathbb{R}$) | 贝叶斯更新 |
| **卡尔曼滤波 (2D)** | 多维状态 + 平滑 | 连续 ($s_t \in \mathbb{R}^n$) | 滤波 + 平滑 + EM |

**核心思想**: 我们无法直接观测隐藏状态，只能通过含噪声的测量来推断。贝叶斯推断是统一框架。

---

## 第一部分：隐马尔可夫模型 (HMM)

---

### 什么是马尔可夫性质？

**马尔可夫性质**: 未来只依赖于现在，与过去无关。

$$p(s_t | s_{t-1}, s_{t-2}, \ldots, s_1) = p(s_t | s_{t-1})$$

**马尔可夫链**: 满足马尔可夫性质的状态序列 $\{s_1, s_2, \ldots, s_T\}$

```
s_1 → s_2 → s_3 → ... → s_T
```

**转移矩阵** $D$: 描述状态之间的转移概率

$$D = \begin{bmatrix} p(s_t=+1|s_{t-1}=+1) & p(s_t=-1|s_{t-1}=+1) \\ p(s_t=+1|s_{t-1}=-1) & p(s_t=-1|s_{t-1}=-1) \end{bmatrix}$$

**状态更新**: $P_t = P_{t-1} D$，其中 $P_t = [p(s_t=+1), p(s_t=-1)]$

$P_{t-1} =[p(s_{t-1}=1),p(s_{t-1}=-1)]$

---

### 隐马尔可夫模型 (HMM) 的结构

HMM = 马尔可夫链 + 观测模型

```
隐藏状态:   s_1 → s_2 → s_3 → ... → s_T
               ↓      ↓      ↓           ↓
观测值:      m_1    m_2    m_3    ...   m_T
```

**两个关键组件**:

| 组件 | 数学描述 | 含义 |
|------|---------|------|
| **状态转移** | $p(s_t \| s_{t-1})$ | 状态如何随时间变化 |
| **观测模型** | $p(m_t \| s_t)$ | 状态如何产生观测值 |

**二元 HMM 示例**:

- 隐状态: $s_t \in \{+1, -1\}$（如：鱼群在左/右）
- 转移概率: $p_{\text{switch}}$（切换概率）
- 观测: $m_t | s_t \sim \mathcal{N}(\mu_{s_t}, \sigma^2)$

---

### HMM 数据生成

从 HMM 中采样数据：

**Step 1: 生成隐状态序列**

```python
def sample(model, T):
    S = np.zeros((T,), dtype=int)
    S[0] = np.random.choice([0, 1], p=model.startprob)
    for t in range(1, T):
        # 根据转移矩阵采样下一个状态
        transition_vector = model.transmat[S[t-1], :]
        S[t] = np.random.choice([0, 1], p=transition_vector)
```

**Step 2: 生成观测值**

```python
    # 根据状态生成高斯观测
    means = model.means[S]
    scales = np.sqrt(model.vars[S])
    M = np.random.normal(loc=means, scale=scales, size=(T,))
```

```python
transmat = np.array([[1 - switch_prob, switch_prob],
                     [switch_prob, 1 - switch_prob]])
```

---

### 预测未来：不确定性增长

即使完全知道当前状态，未来也会变得不确定。

**无观测时的预测**:

$$P_t = P_{t-1} D = P_0 D^t$$

```python
def simulate_prediction_only(model, nstep):
    predictive_probs = []
    prob = model.startprob
    for i in range(nstep):
        predictive_probs.append(prob)
        prob = prob @ model.transmat  # 一步预测
    return predictive_probs
```

**关键观察**:

- 切换概率越大 → 忘记得越快
- 切换概率 = 0.5 → 立即达到均匀分布
- 切换概率 > 0.5 → 状态倾向于切换（振荡）
**神经科学应用**: 这就是为什么我们在变化的世界中需要不断更新感知！

---

### 前向推断 (Forward Inference)

给定观测序列 $m_{1:t}$，推断当前隐状态 $s_t$。

**递归贝叶斯推断**，每一步包含两个操作：

**1. 预测 (Predict)**: 用转移矩阵将昨天的后验变成今天的先验

$$\text{today's prior} = p(s_t | m_{1:t-1}) = p(s_{t-1} | m_{1:t-1}) \cdot D$$

**2. 更新 (Update)**: 结合新的观测 $m_t$ 计算后验

$$\text{posterior} \propto \text{prior} \times \text{likelihood} = p(m_t | s_t) \cdot p(s_t | m_{1:t-1})$$

```python
def one_step_update(model, posterior_tm1, M_t):
    # 预测
    prediction = posterior_tm1 @ model.transmat
    # 计算似然
    likelihood = compute_likelihood(model, M_t)
    # 更新（贝叶斯规则）
    posterior_t = prediction * likelihood
    posterior_t /= np.sum(posterior_t)  # 归一化
    return prediction, likelihood, posterior_t
```

---

### 前向推断的完整流程

```
初始状态 P_0
    │
    ▼
┌─────────────────────────────────────────────┐
│  时间 t=1,2,...,T                            │
│                                              │
│  1. 预测: prior = posterior_{t-1} × D        │
│                                              │
│  2. 计算似然: L = p(m_t | s_t)               │
│                                              │
│  3. 更新: posterior ∝ prior × L              │
│                                              │
│  4. 归一化: posterior = posterior / sum       │
└─────────────────────────────────────────────┘
    │
    ▼
后验概率序列 {P(s_t|m_{1:t})}
```

**影响推断质量的因素**:

- **噪声越大** → 似然越平坦 → 后验更依赖先验
- **切换越快** → 先验更不确定 → 后验更依赖观测

---

### HMM 总结

### 关键概念

- **马尔可夫性质**: 未来只依赖现在
- **隐状态**: 不能直接观测
- **转移矩阵**: 描述状态演化
- **观测模型**: 状态→观测的映射

### 推断算法

- **预测**: 先验 = 后验 × D
- **更新**: 后验 ∝ 先验 × 似然
- **归一化**: 确保概率和为 1
**应用场景**:
- 离子通道开/关状态
- 睡眠/觉醒状态切换
- 鱼群位置追踪
- 语音识别中的音素序列

---

## 第二部分：卡尔曼滤波 (1D)

---

### 从离散到连续：为什么需要卡尔曼滤波？

HMM 处理**离散**隐状态，但很多真实系统是**连续**的。

| | HMM | 卡尔曼滤波 |
|---|---|---|
| **隐状态** | 离散 (有限个) | 连续 (实数) |
| **状态转移** | 转移矩阵 | 线性动力学 + 噪声 |
| **观测模型** | 任意分布 | 线性 + 高斯噪声 |
| **推断** | 前向算法 | 贝叶斯更新 (解析解) |

**卡尔曼滤波是 HMM 的连续版本！**

核心假设：所有分布都是**高斯**的 → 数学上可解析求解

---

### 卡尔曼滤波的数学模型

**状态方程** (动力学模型):

$$s_t = D \cdot s_{t-1} + w_t, \quad w_t \sim \mathcal{N}(0, \sigma_p^2)$$

- $D$: 动力学参数（状态如何演化）
- $w_t$: 过程噪声（动力学的不确定性）
**观测方程** (测量模型):

$$m_t = s_t + \eta_t, \quad \eta_t \sim \mathcal{N}(0, \sigma_m^2)$$

- $\eta_t$: 测量噪声（传感器的不确定性）
**完整模型**:

```
隐状态: s_0 → s_1 → s_2 → ... → s_T
           ↓      ↓      ↓           ↓
观测值:    m_1    m_2    m_3    ...   m_T
```

所有变量都是**高斯分布**！

---

### 卡尔曼滤波：贝叶斯更新

每一步都是高斯 × 高斯 = 高斯

**Step 1: 预测** — 将昨天的后验变成今天的先验

$$\text{prior} = \mathcal{N}(D \cdot \mu_{t-1}, D^2 \cdot \sigma_{t-1}^2 + \sigma_p^2)$$

**Step 2: 更新** — 结合似然计算后验

后验 = 先验 × 似然 = 高斯 × 高斯 = 高斯

**信息加权** (关键洞见):

$$\frac{1}{\sigma_{\text{post}}^2} = \frac{1}{\sigma_{\text{prior}}^2} + \frac{1}{\sigma_{\text{likelihood}}^2}$$

$$\mu_{\text{post}} = g_{\text{prior}} \cdot \mu_{\text{prior}} + g_{\text{likelihood}} \cdot m_t$$

其中 $g = \frac{\text{信息}}{\text{总信息}}$ 是信息权重

---

### 卡尔曼滤波的直觉理解

**信息加权平均**:

```
后验均值 = (先验信息 × 先验均值 + 似然信息 × 观测值) / 总信息
```

| 场景    | 先验权重 | 似然权重 | 结果    |
| ----- | ---- | ---- | ----- |
| 测量噪声小 | 低    | 高    | 接近观测值 |
| 过程噪声小 | 高    | 低    | 接近预测值 |
| 两者相当  | 中    | 中    | 加权平均  |

**经典卡尔曼增益**:

$$K = \frac{\sigma_{\text{prior}}^2}{\sigma_{\text{prior}}^2 + \sigma_m^2}$$

$$\mu_{\text{post}} = \mu_{\text{prior}} + K(m_t - \mu_{\text{prior}})$$

$K$ 接近 1 → 信任测量；$K$ 接近 0 → 信任预测

---

### 卡尔曼滤波代码实现

```python
gaussian = namedtuple('Gaussian', ['mean', 'cov'])
def kalman_filter_step(posterior, D, process_noise, measurement_noise, m):
    # Step 1: 预测
    prior_mean = D * posterior.mean
    prior_cov = D**2 * posterior.cov + process_noise
    # Step 2: 更新
    likelihood = gaussian(m, measurement_noise)
    # 信息加权
    info_prior = 1 / prior_cov
    info_likelihood = 1 / likelihood.cov
    info_posterior = info_prior + info_likelihood
    # 后验均值 = 加权平均
    prior_weight = info_prior / info_posterior
    likelihood_weight = info_likelihood / info_posterior
    posterior_mean = prior_weight * prior_mean + likelihood_weight * m
    # 后验方差 = 1/总信息
    posterior_cov = 1 / info_posterior
    return gaussian(posterior_mean, posterior_cov)
```

---

### 卡尔曼滤波的性质

**1. 后验方差随时间减小**

$$\sigma_t^2 < \sigma_{t-1}^2$$

每次获得新观测，不确定性都减小。

**2. 稳态方差**

当 $t \to \infty$，后验方差收敛到：

$$\sigma_{\infty}^2 = \frac{\sigma_p^2}{1-D^2} \cdot \frac{1}{\text{SNR}+1}$$

其中 $\text{SNR} = \sigma_p^2 / \sigma_m^2$ 是信噪比

**3. 估计误差与后验方差一致**

估计误差 $\hat{s}_t - s_t$ 的分布与后验分布 $\mathcal{N}(0, \sigma_t^2)$ 匹配！

**4. 在线算法**

只使用过去的数据，可以实时运行。

---

### 影响卡尔曼滤波性能的因素

**动力学参数 $D$**:

- $|D| < 1$: 稳定系统，状态衰减到 0
- $|D| = 1$: 随机游走
- $|D| > 1$: 不稳定系统，状态发散
**过程噪声 $\sigma_p$**:
- 大 → 状态变化快 → 预测不确定 → 更依赖测量
- 小 → 状态变化慢 → 预测准确 → 更依赖先验
**测量噪声 $\sigma_m$**:
- 大 → 测量不可靠 → 更依赖预测
- 小 → 测量可靠 → 更依赖观测
**信噪比 SNR = $\sigma_p^2 / \sigma_m^2$**:
- 高 SNR → 滤波效果好
- 低 SNR → 需要更多时间才能准确定位

---

### 卡尔曼滤波的神经科学应用

| 应用 | 隐状态 | 观测值 |
|------|--------|--------|
| **脑机接口** | 运动意图 | 神经活动 |
| **EEG 分析** | 脑活动 | 头皮电压 |
| **感知追踪** | 物体位置 | 视网膜图像 |
| **运动控制** | 肢体位置 | 本体感觉 |

**大脑就是卡尔曼滤波器？**

- 大脑需要从含噪声的感觉输入推断世界状态
- 贝叶斯推断是描述大脑感知的有力框架
- 卡尔曼滤波提供了在线、递归的推断算法

---

## 第三部分：二维卡尔曼滤波与平滑

---

### 从 1D 到 2D：向量形式

**状态方程**:

$$\mathbf{s}_t = D \mathbf{s}_{t-1} + \mathbf{w}_t, \quad \mathbf{w}_t \sim \mathcal{N}(0, Q)$$

**观测方程**:

$$\mathbf{m}_t = H \mathbf{s}_t + \boldsymbol{\eta}_t, \quad \boldsymbol{\eta}_t \sim \mathcal{N}(0, R)$$

**参数矩阵**:

| 符号 | 名称 | 维度 | 含义 |
|------|------|------|------|
| $D$ | 状态转移矩阵 | $n \times n$ | 状态如何演化 |
| $Q$ | 过程噪声协方差 | $n \times n$ | 动力学不确定性 |
| $H$ | 观测矩阵 | $m \times n$ | 状态如何映射到观测 |
| $R$ | 观测噪声协方差 | $m \times m$ | 测量不确定性 |

---

### 2D 卡尔曼滤波：预测步骤

**预测均值**:

$$\hat{\mu}_t^{\text{pred}} = D \hat{\mu}_{t-1}$$

**预测协方差**:

$$\hat{\Sigma}_t^{\text{pred}} = D \hat{\Sigma}_{t-1} D^\top + Q$$

**直觉**:

- 均值通过动力学矩阵传播
- 协方差被 $D$ 缩放，并加上过程噪声

---

### 2D 卡尔曼滤波：更新步骤

**卡尔曼增益矩阵**:

$$K_t = \hat{\Sigma}_t^{\text{pred}} H^\top (H \hat{\Sigma}_t^{\text{pred}} H^\top + R)^{-1}$$

**滤波均值**:

$$\hat{\mu}_t^{\text{filter}} = \hat{\mu}_t^{\text{pred}} + K_t (\mathbf{m}_t - H \hat{\mu}_t^{\text{pred}})$$

**滤波协方差**:

$$\hat{\Sigma}_t^{\text{filter}} = (I - K_t H) \hat{\Sigma}_t^{\text{pred}}$$

**创新项** $\mathbf{m}_t - H \hat{\mu}_t^{\text{pred}}$：实际观测与预测观测的差异

$K_t$ 决定了如何在预测和观测之间权衡

---

### 2D 卡尔曼滤波代码

```python
def kalman_filter(data, params):
    D, Q, H, R = params['D'], params['Q'], params['H'], params['R']
    I = np.eye(D.shape[0])
    mu = np.zeros((len(data), D.shape[0]))
    sigma = np.zeros((len(data), D.shape[0], D.shape[0]))
    for t, y in enumerate(data):
        if t == 0:
            mu_pred = params['mu_0']
            sigma_pred = params['sigma_0']
        else:
            mu_pred = D @ mu[t-1]
            sigma_pred = D @ sigma[t-1] @ D.T + Q
        # 卡尔曼增益
        K = sigma_pred @ H.T @ np.linalg.inv(H @ sigma_pred @ H.T + R)
        # 更新
        mu[t] = mu_pred + K @ (y - H @ mu_pred)
        sigma[t] = (I - K @ H) @ sigma_pred
    return mu, sigma
```

---

### 卡尔曼平滑 (Kalman Smoothing)

**滤波**只使用过去的数据，**平滑**使用全部数据（过去 + 未来）。

**平滑均值** (反向传播):

$$\hat{\mu}_t^{\text{smooth}} = \hat{\mu}_t^{\text{filter}} + J_t (\hat{\mu}_{t+1}^{\text{smooth}} - D \hat{\mu}_t^{\text{filter}})$$

**平滑协方差**:

$$\hat{\Sigma}_t^{\text{smooth}} = \hat{\Sigma}_t^{\text{filter}} + J_t (\hat{\Sigma}_{t+1}^{\text{smooth}} - P_t) J_t^\top$$

**平滑增益**:

$$J_t = \hat{\Sigma}_t^{\text{filter}} D^\top P_t^{-1}$$

其中 $P_t = D \hat{\Sigma}_t^{\text{filter}} D^\top + Q$ 是 $t+1$ 时刻的预测协方差

---

### 滤波 vs 平滑

| | 滤波 (Filtering) | 平滑 (Smoothing) |
|---|---|---|
| **使用数据** | $m_{1:t}$ (过去) | $m_{1:T}$ (全部) |
| **方向** | 前向 ($t=0 \to T$) | 后向 ($t=T \to 0$) |
| **实时性** | 可以在线运行 | 需要批量处理 |
| **精度** | 较低 | 更高 |
| **MSE** | 较大 | 较小 |

```python
print(f"滤波 MSE: {np.mean((state - filtered_state_means)**2):.3f}")
print(f"平滑 MSE: {np.mean((state - smoothed_state_means)**2):.3f}")
# 平滑 MSE 显著更小！
```

**为什么平滑更好？** 因为它利用了未来的信息来修正过去的估计。

---

### 参数学习：EM 算法

当我们不知道系统参数 $D, Q, H, R$ 时，需要从数据中学习。

**期望最大化 (EM) 算法**:

| 步骤 | 操作 | 工具 |
|------|------|------|
| **E 步** | 固定参数，推断隐状态 | 卡尔曼滤波 + 平滑 |
| **M 步** | 固定隐状态，更新参数 | 最大似然估计 |

**M 步更新公式** (示例):

$$D^{\text{new}} = \left(\sum_{t=2}^T \mathbb{E}[s_t s_{t-1}^\top]\right) \left(\sum_{t=2}^T \mathbb{E}[s_{t-1} s_{t-1}^\top]\right)^{-1}$$

```python
import pykalman
kf = pykalman.KalmanFilter(n_dim_state=2, n_dim_obs=2,
                           em_vars=['transition_matrices', 'transition_covariance',
                                    'observation_matrices', 'observation_covariance'])
kf.em(data)  # 用EM算法学习参数
```

---

### 应用：眼动追踪数据平滑

**问题**: 眼动仪数据含噪声，需要平滑

**方法**: 用卡尔曼滤波平滑眼动轨迹

```python
# 设置模型
kf = pykalman.KalmanFilter(n_dim_state=2, n_dim_obs=2)
kf.initial_state_mean = data[0]
kf.initial_state_covariance = 0.1 * np.eye(2)
# 用EM学习参数
kf.em(data)
# 平滑数据
smoothed_mean, smoothed_cov = kf.smooth(data)
```

**结果**: 绿色曲线是平滑后的眼动轨迹，比原始数据（品红色）更平滑

**处理眨眼**: 使用 numpy 的 masked array，将负坐标标记为缺失值

---

## 总结

---

### 关键公式

**HMM 前向推断**:

$$P(s_t | m_{1:t}) \propto p(m_t | s_t) \cdot [P(s_{t-1}|m_{1:t-1}) \cdot D]$$

**1D 卡尔曼滤波**:

$$\mu_t^{\text{post}} = g_{\text{prior}} \cdot D\mu_{t-1} + g_{\text{likelihood}} \cdot m_t$$

$$\frac{1}{(\sigma_t^{\text{post}})^2} = \frac{1}{(D\sigma_{t-1})^2 + \sigma_p^2} + \frac{1}{\sigma_m^2}$$

**2D 卡尔曼滤波**:

$$K_t = \Sigma_t^{\text{pred}} H^\top (H \Sigma_t^{\text{pred}} H^\top + R)^{-1}$$

$$\mu_t^{\text{filter}} = \mu_t^{\text{pred}} + K_t (m_t - H\mu_t^{\text{pred}})$$

---

### 应用与扩展

**神经科学应用**:

- 脑机接口中的运动意图估计
- EEG/MEG 信号的源定位
- 感觉运动控制的状态估计
- 神经元集群的解码
**扩展模型**:

| 扩展 | 问题 | 方法 |
|------|------|------|
| 非线性动力学 | $s_t = f(s_{t-1}) + w_t$ | 扩展卡尔曼滤波 (EKF) |
| 非高斯噪声 | 重尾分布 | 粒子滤波 |
| 连续时间 | $ds = f(s)dt + \sigma dW$ | 卡尔曼 - 布西滤波 |
| 参数未知 | $D, Q, R$ 未知 | EM 算法 |

---
