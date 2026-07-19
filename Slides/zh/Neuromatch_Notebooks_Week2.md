# Neuromatch 笔记本 — 第 2 周

线性系统 (Linear Systems) · 生物神经元模型 (Biological Neuron Models) · 动力系统 (Dynamical Systems)

---

## 概述 (Overview)

第 2 周聚焦于**动力系统与神经模型 (dynamical systems and neural models)** ——从线性系统到生物神经元模型，再到网络动力学：

| 天数      | 主题                      | 核心技能                               |
| -------- | -------------------------- | ---------------------------------------- |
| **W2D3** | 线性系统 (Linear Systems)             | 欧拉积分 (Euler integration)、振荡 (Oscillations)、随机游走 (Random walks)、OU 过程、自回归模型 (AR models) |
| **W2D4** | 生物神经元模型 (Biological Neuron Models)   | 泄漏积分发放神经元 (LIF neuron)、电导突触 (Conductance synapses)、短期可塑性 (STP)、脉冲时间依赖可塑性 (STDP)               |
| **W2D5** | 动力系统 (Dynamical Systems)          | 放电频率模型 (Firing rate models)、Wilson-Cowan 模型、相平面分析 (Phase plane)、雅可比矩阵 (Jacobian)、极限环 (Limit cycles) |

**统一主题**：神经元和网络如何随时间演化，以及我们如何用数学方法对其动力学进行建模？

---

## W2D3：线性系统 (Linear Systems)

---

### Tutorial 1：一维微分方程 (One-Dimensional Differential Equations)

最简单的动力系统：$\dot{x} = ax$

**解析解 (Analytical solution)**：$x(t) = x_0 e^{at}$

| $a$                  | 行为 (Behavior)                        |
| -------------------- | ------------------------------- |
| $a < 0$              | 指数衰减 → 0 (Exponential decay → 0)           |
| $a > 0$              | 指数增长 → ∞ (Exponential growth → ∞)          |
| $a = \text{complex}$ | 振荡（伴随增长/衰减）(Oscillation with growth/decay) |

**前向欧拉积分 (Forward Euler integration)**（数值解 (numerical solution)）：

$$x(t_i) = x(t_{i-1}) + \dot{x}(t_{i-1}) \cdot dt$$

对于 $\dot{x} = ax$ 具体形式：$x[k] = x[k-1] + a \cdot x[k-1] \cdot dt$

**实现细节**：使用 `dtype=complex` 处理复数 $a$（振荡动力学需要）

---

### Tutorial 1：复数 $a$ 与振荡动力学

当 $a$ 为复数时（$a = \text{real} + i \cdot \text{imag}$），系统产生振荡：

$$x(t) = x_0 e^{(\text{real} + i \cdot \text{imag})t} = x_0 e^{\text{real} \cdot t} \cdot [\cos(\text{imag} \cdot t) + i \sin(\text{imag} \cdot t)]$$

**关键洞察 (Key insight)**：
- **实部 (Real part)** → 增长/衰减率（振幅包络）
- **虚部 (Imaginary part)** → 振荡频率

**稳定振荡条件**：设实部 = 0，虚部 = $2\pi f$

例如：产生 0.5 Hz 的稳定振荡 → 虚部 = $2\pi \times 0.5 = \pi \approx 3.14$

**增长振荡条件**：实部 > 0 且 虚部 ≠ 0

---

### Tutorial 1：二维线性系统 (Two-Dimensional Linear Systems)

扩展到二维：$\dot{\mathbf{x}} = \mathbf{A}\mathbf{x}$

$$\begin{bmatrix} \dot{x}_1 \\ \dot{x}_2 \end{bmatrix} = \begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}$$

**数值求解**：使用 `scipy.integrate.solve_ivp`（而非手动欧拉法）

**流线图 (Stream plot)**：在网格上计算 $\mathbf{A}\mathbf{x}$，箭头显示状态变化方向

**特征向量 (Eigenvectors)**：$\mathbf{A}\mathbf{x}$ 与 $\mathbf{x}$ 平行的方向（不变方向）

**特征值 (Eigenvalues)**：$\mathbf{A}\mathbf{x}$ 沿特征向量方向的拉伸/压缩因子

**稳定性分类 (Stability classification)**：

| 特征值类型 | 行为 |
|-----------|------|
| 均为负实数 | 稳定节点 (Stable node)（收敛到原点）|
| 均为正实数 | 不稳定节点 (Unstable node)（发散）|
| 符号相反 | 鞍点 (Saddle point) |
| 复数 | 振荡/螺旋 (Oscillation / Spiral) |

---

### Tutorial 2：马尔可夫过程 (Markov Processes)

**马尔可夫性质**：当前状态完全决定下一状态的转移（无记忆性）

**电报过程 (Telegraph process)**：两态离子通道模型

- 状态：关闭 (0) 和 打开 (1)
- 转移概率：$P(0 \to 1 | x=0) = \mu_{c2o}$，$P(1 \to 0 | x=1) = \mu_{o2c}$

**泊松过程性质**：
1. 事件概率独立于其他事件
2. 平均事件率在给定时间段内恒定
3. 两个事件不能同时发生

**状态转移矩阵 (State transition matrix)**：

$$\begin{bmatrix} C \\ O \end{bmatrix}_{k+1} = \begin{bmatrix} 1-\mu_{c2o} & \mu_{o2c} \\ \mu_{c2o} & 1-\mu_{o2c} \end{bmatrix} \begin{bmatrix} C \\ O \end{bmatrix}_k$$

- 每列之和 = 1（概率守恒）
- 矩阵元素含义：
  - $1 - \mu_{c2o}$：保持关闭的概率
  - $\mu_{c2o}$：从关闭转为打开的概率
  - $\mu_{o2c}$：从打开转为关闭的概率
  - $1 - \mu_{o2c}$：保持打开的概率

**概率传播算法**：$\mathbf{x}_{k+1} = \mathbf{A} \cdot \mathbf{x}_k$（矩阵-向量乘法）

**平衡态分析 (Equilibrium analysis)**：
- 特征值 = 1 对应**稳定平衡**特征向量
- 其他特征值对应瞬态衰减
- 平衡特征向量需归一化（元素之和 = 1）
- 打开的平衡概率：$\frac{\mu_{c2o}}{\mu_{c2o} + \mu_{o2c}}$

---

### Tutorial 3：随机游走与扩散 (Random Walks and Diffusion)

**随机游走 (Random walk)**：每一步以等概率移动 $\Delta x = \pm 1$

**位置更新**：$x_{k+1} = x_k + \Delta x$

**高斯步长的随机游走**：步骤从 $\mathcal{N}(\mu, \sigma)$ 采样

**高效向量化实现**：

```python
def random_walk_simulator(N, T, mu=0, sigma=1):
    steps = np.random.normal(mu, sigma, size=(N, T))
    sim = np.cumsum(steps, axis=1)
    return sim
```

**扩散过程性质 (Properties of diffusive process)**：
- 均值保持在 0 附近（与时间无关）
- **方差随时间线性增长**：$\text{Var} \propto t$（具体为 $\text{Var} = \sigma^2 t$）
- 分布随时间变宽但中心不变

---

### Tutorial 3：确定性衰减与 OU 过程

**基本衰减**：$x_{k+1} = \lambda x_k$，解：$x_k = x_0 \lambda^k$（$|\lambda| < 1$ 时衰减）

**带目标的衰减**：$x_{k+1} = x_\infty + \lambda(x_k - x_\infty)$

**解析解**：$x_k = x_\infty(1 - \lambda^k) + x_0 \lambda^k$

当 $k \to \infty$：$x_k \to x_\infty$

**Ornstein-Uhlenbeck (OU) 过程 / 漂移扩散模型 (Drift-Diffusion Model)**：

$$x_{k+1} = x_\infty + \lambda(x_k - x_\infty) + \sigma \eta$$

其中 $\eta \sim \mathcal{N}(0,1)$（标准正态分布）

**两个组成部分**：
- **漂移项 (Drift)**：$x_\infty + \lambda(x_k - x_\infty)$，将 $x$ 拉向 $x_\infty$
- **扩散项 (Diffusion)**：$\sigma \eta$，添加随机噪声

**平衡方差 (Equilibrium variance)**（关键结果）：

$$\text{Var}_{eq} = \frac{\sigma^2}{1 - \lambda^2}$$

**性质**：
- 仅依赖于 $\lambda$ 和 $\sigma$，**不依赖** $x_0$ 或 $x_\infty$
- 当 $\lambda \to 1$：方差发散（接近纯随机游走）
- 当 $\lambda \to 0$：方差趋近 $\sigma^2$（每步独立）

**经验方差计算**：运行长时间 $T$ 的模拟，取**后半段**的方差（假设系统已稳定）

```python
x[-round(T/2):].var()
```

**关键观察**：
- OU 过程的均值精确遵循确定性解
- 方差达到平衡（不像随机游走那样无限增长）
- 恢复漂移力防止了方差的无限增长

---

### Tutorial 4：自回归模型 (Autoregressive Models)

**视角转换**：给定数据，学习其动力学（逆向问题）

**一阶自回归 AR(1)**：$x_{k+1} = \lambda x_k + \eta$

**回归公式化**：$\mathbf{x}_2 = \lambda \mathbf{x}_1$
- $\mathbf{x}_1 = x[0:T-1]$（过去值）
- $\mathbf{x}_2 = x[1:T]$（未来值，偏移 1）

**最小二乘求解**：

```python
p, res, rnk, s = np.linalg.lstsq(x1, x2, rcond=None)
```

**添加截距项**：在 x1 前添加一列 1

```python
x1 = x1[:, np.newaxis]**[0, 1]  # 添加列：常数项和线性项
```

回归系数 $p[1]$ 即为估计的 $\hat{\lambda}$

**残差分析 (Residual analysis)**：
- 残差 = 数据 - 预测值：$\text{res} = x_2 - (p[0] + \hat{\lambda} \cdot x_1[:, 1])$
- 残差标准差应近似等于 $\sigma$（噪声参数）
- 残差直方图应近似正态分布

---

### Tutorial 4：高阶自回归模型

**r 阶 AR 模型**：$x_{k+1} = \alpha_0 + \alpha_1 x_k + \alpha_2 x_{k-1} + \dots + \alpha_r x_{k-r}$

共 $r+1$ 个系数需要拟合（包括截距 $\alpha_0$）

**时间延迟矩阵构建 (build_time_delay_matrices)**：

- $\mathbf{x}_1$：大小为 $[(r+1) \times (n-r)]$ 的矩阵
  - 第 0 行：全 1（截距项）
  - 第 1 行：$x[0:T-r]$（滞后 1）
  - 第 2 行：$x[1:T-r+1]$（滞后 2，通过 `np.roll` 实现）
  - ... 直到滞后 $r$
- $\mathbf{x}_2$：向量 $x[r:]$（要预测的值）

**np.roll 技巧**：`xprime = np.roll(xprime, -1)` 将数组左移 1 位

**预测与分类**：
- 对于二值 (+1/-1) 数据：预测 = $\text{sign}(\mathbf{x}_1^T \cdot \mathbf{p})$
- 错误率 = $\text{count}(x_2 \neq \text{prediction}) / \text{len}(x_2)$
- 随机猜测基线：错误率 = 0.5

**过拟合观察**：
- 扫描 AR 阶数从 r=1 到 r=20
- 存在**最佳点**（约 r=6 对于人类生成数据）
- 低 r：欠拟合（错过模式）
- 高 r：过拟合（拟合训练噪声，测试表现差）
- 体现了偏差-方差权衡 (bias-variance tradeoff)

**人类随机性 vs 机器随机性**：
- 人类在生成随机序列方面表现很差（可检测的模式）
- AR 模型可以利用这些模式进行优于随机的预测
- 机器生成的随机整数真正不可预测（错误率 ≈ 0.5）
- 二值编码：'0' → -1，'1' → +1（通过 `x*2 - 1`）

---

## W2D4：生物神经元模型 (Biological Neuron Models)

---

### Tutorial 1：泄漏积分发放模型 (LIF Model)

**核心膜电位方程（阈下动力学）**：

$$\tau_m \frac{dV}{dt} = -(V - E_L) + \frac{I}{g_L}$$

其中 $\tau_m = C_m / g_L$ 是膜时间常数，$g_L$ 是漏电导，$E_L$ 是静息电位

**脉冲与复位规则 (Spike-and-reset rule)**：

$$\text{if } V(t_{sp}) \geq V_{th}: \quad V(t) = V_{reset} \text{ for } t \in (t_{sp}, t_{sp} + \tau_{ref}]$$

**默认参数**：

| 参数 | 值 | 含义 |
|------|-----|------|
| $V_{th}$ | -55 mV | 脉冲阈值 |
| $V_{reset}$ | -75 mV | 复位电位 |
| $E_L$ | -75 mV | 静息电位 |
| $\tau_m$ | 10 ms | 膜时间常数 |
| $g_L$ | 10 nS | 漏电导 |
| $t_{ref}$ | 2 ms | 不应期 |
| $dt$ | 0.1 ms | 时间步长 |

**欧拉积分实现 (run_LIF)**：

```python
for it in range(Lt - 1):
    if tr > 0:                          # 不应期
        v[it] = V_reset
        tr = tr - 1
    elif v[it] >= V_th:                 # 脉冲！
        rec_spikes.append(it)
        v[it] = V_reset
        tr = tref / dt
    # 计算膜电位增量
    dv = (dt / tau_m) * (-(v[it] - E_L) + Iinj[it] / g_L)
    # 更新膜电位
    v[it + 1] = v[it] + dv
```

---

### Tutorial 1：不同类型的输入电流

**直流电流 (DC)**：恒定电流，产生规则脉冲（CV_ISI ≈ 0）

**高斯白噪声 (GWN)**：

$$I_{gwn} = \mu + \sigma \cdot \frac{\xi(t)}{\sqrt{dt/1000}}$$

其中 $\xi(t) \sim \mathcal{N}(0,1)$，除以 $\sqrt{dt/1000}$ 将离散时间噪声转换为正确的连续时间缩放（单位转换为秒）

**Ornstein-Uhlenbeck (OU) 过程（有色噪声）**：

$$\tau_\eta \frac{d\eta}{dt} = -\eta(t) + \sigma_\eta \sqrt{2\tau_\eta} \xi(t)$$

**性质**：
- $\mathbb{E}[\eta(t)] = \mu$
- 自协方差：$\text{Cov}[\eta(t), \eta(t+\tau)] = \sigma_\eta^2 e^{-|t-\tau|/\tau_\eta}$

**欧拉实现**：

```python
I_ou[it+1] = I_ou[it] + (dt/tau_ou)*(mu - I_ou[it]) + sqrt(2*dt/tau_ou)*sig*noise[it+1]
```

---

### Tutorial 1：放电频率与脉冲不规则性

**频率-电流曲线 (F-I curve)**：输出放电频率作为输入电流的函数

**脉冲间隔变异系数 (CV of ISI)**：

$$\text{CV}_{\text{ISI}} = \frac{\text{std}(\text{ISI})}{\text{mean}(\text{ISI})}$$

| CV 值 | 含义 |
|--------|------|
| 0 | 完全规则（时钟般）|
| 1 | 泊松过程（最大不规则性）|

**关键发现**：
- DC 输入产生规则脉冲（CV ≈ 0）
- GWN 输入产生不规则脉冲；更高的 $\sigma$ 增加 CV_ISI
- 增加 $\sigma$ 使 F-I 曲线变平滑
- 增加均值 $\mu$ 同时保持 $\sigma$ 固定会降低 CV_ISI（高频时更规则）

---

### Tutorial 2：相关输入与相关性转移

**相关输入模型**：

$$\frac{I_i}{g_L} = \mu_i + \sigma_i (\sqrt{1-c}\xi_i + \sqrt{c}\xi_c)$$

其中 $c \in [0,1]$ 控制公共输入的比例，$\xi_i$ 是独立噪声，$\xi_c$ 是共享的共同噪声

**样本相关系数 (Pearson)**：

$$r_{ij} = \frac{\text{cov}(I_i, I_j)}{\sqrt{\text{var}(I_i)} \sqrt{\text{var}(I_j)}}$$

其中 $\text{cov}(I_i, I_j) = \sum_{k=1}^{L}(I_i^k - \bar{I_i})(I_j^k - \bar{I_j})$

注意：严格样本协方差应除以 $L-1$，但在相关系数中分子分母的 $L-1$ 会抵消

**泊松脉冲生成器 (Poisson_generator)**：

```python
poisson_train = 1.0 * (u_rand < rate * (dt / 1000))
```

每个时间箱的脉冲概率 = $\text{rate} \times dt / 1000$

**相关泊松生成 (generate_corr_Poisson)**：
1. 生成频率为 $\lambda/c$ 的"母"泊松脉冲序列
2. 每个子神经元独立采样母序列中比例为 $c$ 的脉冲（通过随机打乱索引实现）

**Campbell 定理（泊松输入的突触电流均值和方差）**：

$$\mu_{\rm syn} = \lambda J \int P(t) dt$$

$$\sigma_{\rm syn} = \lambda J \int P(t)^2 dt$$

其中 $\lambda$ 是泊松率，$J$ 是 PSP 幅度，$P(t)$ 是突触后电流核

**关键发现**：
- 输出相关性 **总是小于** 输入相关性（LIF 充当"相关性滤波器"）
- 相关性转移函数近似线性
- 更高的均值 $\mu$ 和更高的 $\sigma$ 都会增加转移函数的斜率（更好的相关性传递）
- 更高的放电率导致更好的相关性传递

---

### Tutorial 3：基于电导的突触 (Conductance-Based Synapses)

**突触电导动力学**：

$$\frac{dg_{\rm syn}(t)}{dt} = \bar{g}_{\rm syn} \sum_k \delta(t-t_k) - \frac{g_{\rm syn}(t)}{\tau_{\rm syn}}$$

- $\bar{g}_{\rm syn}$：每个脉冲引起的最大电导变化（突触权重）
- $\tau_{\rm syn}$：突触时间常数（控制衰减速度）

**欧姆定律（电导转电流）**：

$$I_{\rm syn}(t) = g_{\rm syn}(t)(V(t) - E_{\rm syn})$$

- $E_E = 0$ mV（兴奋性反转电位，去极化）
- $E_I = -80$ mV（抑制性反转电位，超极化）

**总突触电流**：

$$I_{\rm syn} = -g_E(t)(V - E_E) - g_I(t)(V - E_I)$$

**电导 LIF 膜电位方程**：

$$\tau_m \frac{dV}{dt} = -(V - E_L) - \frac{g_E(t)}{g_L}(V - E_E) - \frac{g_I(t)}{g_L}(V - E_I) + \frac{I_{\rm inj}}{g_L}$$

**欧拉更新电导 (run_LIF_cond)**：

```python
gE[it+1] = gE[it] - (dt/tau_syn_E)*gE[it] + gE_bar * spike_train_ex[it+1]
gI[it+1] = gI[it] - (dt/tau_syn_I)*gI[it] + gI_bar * spike_train_in[it+1]
```

**默认突触参数**：
- 兴奋性：$g_E = 2.4$ nS，$E_E = 0$ mV，$\tau_E = 2$ ms
- 抑制性：$g_I = 2.4$ nS，$E_I = -80$ mV，$\tau_I = 5$ ms
- 80 个兴奋性、20 个抑制性突触前神经元，频率 10 Hz

**自由膜电位 (Free Membrane Potential, FMP)**：去除脉冲阈值的膜电位（人为设定 $V_{th} = \infty$）

- 平均 FMP > 阈值：**均值驱动体制** (Mean-driven regime)（规则放电，低 CV_ISI）
- 平均 FMP < 阈值：**波动驱动体制** (Fluctuation-driven regime)（不规则放电，高 CV_ISI）
- 兴奋/抑制平衡决定放电模式
- 突触输入是**有色噪声**（指数核滤波），不是白噪声

---

### Tutorial 3：短期突触可塑性 (Short-Term Plasticity, STP)

**三个动态变量模型**：

$$\frac{du_E}{dt} = -\frac{u_E}{\tau_f} + U_0(1-u_E^-)\delta(t-t_{sp})$$

$$\frac{dR_E}{dt} = \frac{1-R_E}{\tau_d} - u_E^+ R_E^- \delta(t-t_{sp})$$

$$\frac{dg_E}{dt} = -\frac{g_E}{\tau_E} + \bar{g}_E u_E^+ R_E^- \delta(t-t_{sp})$$

**变量含义**：

| 变量 | 含义 | 范围 | 衰减常数 |
|------|------|------|---------|
| $u$ | 释放概率（使用率） | $[0, 1]$ | $\tau_f$（易化时间常数）|
| $R$ | 可释放资源 | $[0, 1]$ | $\tau_d$（抑制时间常数）|
| $g$ | 突触后电导 | $[0, \bar{g}]$ | $\tau_E$（突触时间常数）|

**物理过程**：
```
脉冲到达 → u 增加（钙内流）
         → 消耗资源：R 减少
         → 产生电导：g 增加

脉冲之间 → u 衰减回 0（τ_f）
         → R 恢复到 1（τ_d）
         → g 衰减（τ_E）
```

**欧拉实现 (dynamic_syn)**：

```python
for it in range(Lt - 1):
    # 更新 u（释放概率）
    du = -(dt/tau_f) * u[it] + U0 * (1.0 - u[it]) * pre_spike_train[it+1]
    u[it+1] = u[it] + du
    
    # 更新 R（资源）- 注意使用更新后的 u[it+1]
    dR = (dt/tau_d) * (1.0 - R[it]) - u[it+1] * R[it] * pre_spike_train[it+1]
    R[it+1] = R[it] + dR
    
    # 更新 g（电导）- 注意使用更新后的 u[it+1] 和 R[it]
    dg = -(dt/tau_syn) * g[it] + g_bar * R[it] * u[it+1] * pre_spike_train[it+1]
    g[it+1] = g[it] + dg
```

**关键点**：脉冲到达时，先更新 $u$，再用新的 $u$ 更新 $R$ 和 $g$（顺序很重要！）

**短期抑制 (STD) vs 短期易化 (STF) 参数**：

| 参数 | STD | STF |
|------|-----|-----|
| $U_0$ | 0.5（高初始释放率）| 0.2（低初始释放率）|
| $\tau_d$ | 100 ms | 100 ms |
| $\tau_f$ | 50 ms（快速恢复）| 750 ms（慢速衰减）|

**STD 机制**：
- 高频输入时资源来不及恢复，电导持续减小
- $g_{10}/g_1$ 随输入率单调递减

**STF 机制**：
- $\tau_f$ 大时，$u$ 在脉冲间衰减慢，累积效应明显
- $g_{10}/g_1$ 随输入率非单调变化（先增后减）

---

### Tutorial 4：脉冲时间依赖可塑性 (STDP)

**STDP 权重变化规则（双相指数衰减）**：

$$\Delta W = \begin{cases} A_+ e^{(t_{pre}-t_{post})/\tau_+} & \text{if } t_{post} > t_{pre} \text{ (LTP)} \\ -A_- e^{-(t_{pre}-t_{post})/\tau_-} & \text{if } t_{post} < t_{pre} \text{ (LTD)} \end{cases}$$

其中 $\Delta t = t_{pre} - t_{post}$。为简化，设 $\tau_+ = \tau_- = \tau_{\rm stdp}$

**默认 STDP 参数**：
- $A_+ = 0.008$（LTP 幅度）
- $A_- = A_+ \times 1.10 = 0.0088$（LTD 幅度，略大——不对称）
- $\tau_{\rm stdp} = 20$ ms

**追踪变量 P(t) 和 M(t)（高效 STDP 实现）**：

对于每个突触前神经元 $i$：
$$\tau_+ \frac{dP}{dt} = -P$$
突触前脉冲到达时：$P(t) = P(t) + A_+$

对于每个突触后神经元：
$$\tau_- \frac{dM}{dt} = -M$$
突触后脉冲到达时：$M(t) = M(t) - A_-$

- $P(t)$ 始终为正（追踪最近的突触前活动，用于 LTP）
- $M(t)$ 始终为负（追踪最近的突触后活动，用于 LTD）

**P 的欧拉更新 (generate_P)**：

```python
dP = -(dt/tau_stdp)*P[:,it] + A_plus * spike_train[:,it+1]
P[:,it+1] = P[:,it] + dP
```

**使用追踪变量的权重更新规则**：

当突触前神经元 $i$ 发放时（LTD）：
$$\bar{g}_i = \bar{g}_i + M(t) \cdot \bar{g}_{max}$$
- $M$ 为负，所以权重减小
- 钳制：若 $\bar{g}_i < 0$，设 $\bar{g}_i = 0$

当突触后神经元发放时（LTP）：
$$\bar{g}_i = \bar{g}_i + P_i(t) \cdot \bar{g}_{max} \quad \forall i$$
- $P$ 为正，所以权重增大
- 钳制：若 $\bar{g}_i > \bar{g}_{max}$，设 $\bar{g}_i = \bar{g}_{max}$

**带 STDP 突触的 LIF 膜电位方程**：

$$\tau_m \frac{dV}{dt} = -(V - E_L) - g_E(t)(V - E_E)$$

其中 $g_E(t) = \sum_i g_i(t)$，每个 $g_i(t)$ 使用动态更新的 $\bar{g}_i$

**默认突触参数（STDP 模拟）**：
- $\bar{g}_E = 0.024$ nS（每个突触的最大电导）
- $g_{E,init} = 0.014 - 0.024$ nS（初始电导）
- $E_E = 0$ mV，$\tau_E = 5$ ms
- $N = 300$ 个突触前神经元，频率 10-15 Hz，$dt = 1$ ms

**关键发现**：
- 不相关泊松输入时，许多突触随时间减弱（LTD 主导，因为 $A_- > A_+$）
- 权重分布随时间演化；出现双峰分布（许多权重接近 0，一些接近 $g_{max}$）
- 相关输入时：相关突触前神经元维持其权重（更高的 pre-before-post 配对机会），不相关突触抑制
- STDP 实现**无监督学习**：携带相关/相关信息的突触被选择性增强

---

## W2D5：动力系统 (Dynamical Systems)

---

### Tutorial 1：单群放电频率模型

**前馈放电频率动力学 (Eq. 1)**：

$$\tau \frac{dr}{dt} = -r + F(I_{\rm ext})$$

**Sigmoid 传递函数 / F-I 曲线 (Eq. 2)**：

$$F(x; a, \theta) = \frac{1}{1 + e^{-a(x-\theta)}} - \frac{1}{1 + e^{a\theta}}$$

- $a$ = 增益 (gain)，$\theta$ = 阈值 (threshold)
- 第二项确保 $F(0; a, \theta) = 0$

**实现**：

```python
def F(x, a, theta):
    f = (1 + np.exp(-a * (x - theta)))**-1 - (1 + np.exp(a * theta))**-1
    return f
```

**递归网络动力学 (Eq. 3)**：

$$\tau \frac{dr}{dt} = -r + F(w \cdot r + I_{\rm ext})$$

其中 $w$ 是递归突触权重（E 到 E）

**$w = 0$ 时的解析解**：

$$r(t) = r(0) + [F(I_{\rm ext}; a, \theta) - r(0)](1 - e^{-t/\tau})$$

---

### Tutorial 1：不动点与稳定性

**不动点条件 (Eq. 4)**：$\frac{dr}{dt} = 0$ 时的 $r$ 值

$$-r^* + F(w \cdot r^* + I_{\rm ext}; a, \theta) = 0$$

**Sigmoid 传递函数的导数 (Eq. 5)**：

$$\frac{dF}{dx} = a \cdot e^{-a(x-\theta)} \cdot (1 + e^{-a(x-\theta)})^{-2}$$

**特征值（稳定性分析）(Eq. 4 in Bonus)**：

$$\lambda = \frac{-1 + w \cdot F'(w \cdot r^* + I_{\rm ext}; a, \theta)}{\tau}$$

| $\lambda$ | 稳定性 |
|-----------|--------|
| $\lambda < 0$ | 稳定（吸引）|
| $\lambda > 0$ | 不稳定（排斥）|

**实现**：

```python
def eig_single(fp, tau, a, theta, w, I_ext, **other_pars):
    eig = (-1 + w * dF(w * fp + I_ext, a, theta)) / tau
    return eig
```

**默认参数**：$\tau = 1.0$ ms，$a = 1.2$，$\theta = 2.8$，$w = 0.0$，$I_{\rm ext} = 0.0$，$T = 20$ ms，$dt = 0.1$ ms

---

### Tutorial 1：OU 噪声输入

**OU 过程**：

$$\tau_\eta \frac{d\eta}{dt} = -\eta(t) + \sigma_\eta \sqrt{2\tau_\eta} \xi(t)$$

**欧拉更新**：

```python
I_ou[it+1] = I_ou[it] + dt/tau_ou * (0 - I_ou[it]) + sqrt(2*dt/tau_ou) * sig * noise[it+1]
```

**关键发现**：在多个不动点存在时，噪声输入可以驱动系统在不动点之间转换

---

### Tutorial 2：Wilson-Cowan 模型

**两个耦合群（兴奋 + 抑制）(Eq. 1)**：

$$\tau_E \frac{dr_E}{dt} = -r_E + F_E(w_{EE}r_E - w_{EI}r_I + I_E^{\rm ext}; a_E, \theta_E)$$

$$\tau_I \frac{dr_I}{dt} = -r_I + F_I(w_{IE}r_E - w_{II}r_I + I_I^{\rm ext}; a_I, \theta_I)$$

**欧拉更新**：

```python
r_E[k+1] = r_E[k] + (dt/τ_E)*(-r_E[k] + F(w_EE*r_E[k] - w_EI*r_I[k] + I_ext_E, a_E, θ_E))
r_I[k+1] = r_I[k] + (dt/τ_I)*(-r_I[k] + F(w_IE*r_E[k] - w_II*r_I[k] + I_ext_I, a_I, θ_I))
```

**默认参数**：

| 参数 | 值 | 含义 |
|------|-----|------|
| $\tau_E$ | 1.0 ms | E 群时间常数 |
| $\tau_I$ | 2.0 ms | I 群时间常数 |
| $a_E$ | 1.2 | E 群增益 |
| $a_I$ | 1.0 | I 群增益 |
| $\theta_E$ | 2.8 | E 群阈值 |
| $\theta_I$ | 4.0 | I 群阈值 |
| $w_{EE}$ | 9.0 | E→E 连接强度 |
| $w_{EI}$ | 4.0 | I→E 连接强度 |
| $w_{IE}$ | 13.0 | E→I 连接强度 |
| $w_{II}$ | 11.0 | I→I 连接强度 |

---

### Tutorial 2：零线 (Nullclines)

**零线定义**：$\frac{dr_E}{dt} = 0$ 或 $\frac{dr_I}{dt} = 0$ 的曲线

**E 零线 ($\frac{dr_E}{dt} = 0$, Eq. 2)**：
$$-r_E + F_E(w_{EE}r_E - w_{EI}r_I + I_E^{\rm ext}; a_E, \theta_E) = 0$$

**I 零线 ($\frac{dr_I}{dt} = 0$, Eq. 3)**：
$$-r_I + F_I(w_{IE}r_E - w_{II}r_I + I_I^{\rm ext}; a_I, \theta_I) = 0$$

**零线显式表达 (Eqs. 4-5)**：

$$\text{E 零线：} \quad r_I = \frac{1}{w_{EI}}[w_{EE}r_E - F_E^{-1}(r_E; a_E, \theta_E) + I_E^{\rm ext}]$$

$$\text{I 零线：} \quad r_E = \frac{1}{w_{IE}}[w_{II}r_I + F_I^{-1}(r_I; a_I, \theta_I) - I_I^{\rm ext}]$$

**逆传递函数 (Eq. 6)**：

$$F^{-1}(x; a, \theta) = -\frac{1}{a} \ln\left[\frac{1}{x + \frac{1}{1+e^{a\theta}}} - 1\right] + \theta$$

**零线的性质**：
- E 零线将相平面分为 $\frac{dr_E}{dt} > 0$ 和 $\frac{dr_E}{dt} < 0$ 两个区域
- I 零线将相平面分为 $\frac{dr_I}{dt} > 0$ 和 $\frac{dr_I}{dt} < 0$ 两个区域
- 两条零线的交点是系统的**不动点**

---

### Tutorial 2：向量场 (Vector Field)

**向量场定义**：在相平面每个点上显示 $(\frac{dr_E}{dt}, \frac{dr_I}{dt})$ 的箭头

```python
def EIderivs(rE, rI, tau_E, a_E, theta_E, wEE, wEI, I_ext_E,
             tau_I, a_I, theta_I, wIE, wII, I_ext_I, **other_pars):
    drEdt = (-rE + F(wEE*rE - wEI*rI + I_ext_E, a_E, theta_E)) / tau_E
    drIdt = (-rI + F(wIE*rE - wII*rI + I_ext_I, a_I, theta_I)) / tau_I
    return drEdt, drIdt
```

**关键观察**：
- 轨迹遵循向量场方向
- 不同轨迹最终到达两个不动点之一（取决于初始条件）
- 轨迹收敛的点是零线曲线的交点

---

### Tutorial 3：雅可比矩阵与稳定性

**系统重写**：

$$\frac{dr_E}{dt} = G_E(r_E, r_I) = \frac{1}{\tau_E}[-r_E + F_E(w_{EE}r_E - w_{EI}r_I + I_E^{\rm ext}; a, \theta)]$$

$$\frac{dr_I}{dt} = G_I(r_E, r_I) = \frac{1}{\tau_I}[-r_I + F_I(w_{IE}r_E - w_{II}r_I + I_I^{\rm ext}; a, \theta)]$$

**雅可比矩阵 (Jacobian, Eq. 7)**：

$$J = \begin{bmatrix} \frac{\partial G_E}{\partial r_E} & \frac{\partial G_E}{\partial r_I} \\ \frac{\partial G_I}{\partial r_E} & \frac{\partial G_I}{\partial r_I} \end{bmatrix}$$

**雅可比矩阵元素 (Eqs. 8-11)**：

$$J[0,0] = \frac{\partial G_E}{\partial r_E} = \frac{1}{\tau_E}[-1 + w_{EE} F_E'(w_{EE}r_E^* - w_{EI}r_I^* + I_E^{\rm ext})]$$

$$J[0,1] = \frac{\partial G_E}{\partial r_I} = \frac{1}{\tau_E}[-w_{EI} F_E'(w_{EE}r_E^* - w_{EI}r_I^* + I_E^{\rm ext})]$$

$$J[1,0] = \frac{\partial G_I}{\partial r_E} = \frac{1}{\tau_I}[w_{IE} F_I'(w_{IE}r_E^* - w_{II}r_I^* + I_I^{\rm ext})]$$

$$J[1,1] = \frac{\partial G_I}{\partial r_I} = \frac{1}{\tau_I}[-1 - w_{II} F_I'(w_{IE}r_E^* - w_{II}r_I^* + I_I^{\rm ext})]$$

**矩阵符号表示**：

$$J = T^{-1}(FW - I)$$

其中：
- $T = \begin{bmatrix} \tau_E & 0 \\ 0 & \tau_I \end{bmatrix}$（时间常数矩阵）
- $F = \begin{bmatrix} F_E' & 0 \\ 0 & F_I' \end{bmatrix}$（增益导数矩阵）
- $W = \begin{bmatrix} w_{EE} & -w_{EI} \\ w_{IE} & -w_{II} \end{bmatrix}$（连接矩阵）
- $I$ 是单位矩阵

**稳定性准则**：
- $\det(J) > 0$ 时不动点稳定（两个特征值实部为负）
- $\det(FW - I) = (F_E' w_{EI})(F_I' w_{IE}) - (F_I' w_{II} + 1)(F_E' w_{EE} - 1) > 0$

**实现**：

```python
def get_eig_Jacobian(fp, tau_E, a_E, theta_E, wEE, wEI, I_ext_E,
                     tau_I, a_I, theta_I, wIE, wII, I_ext_I, **other_pars):
    rE, rI = fp
    J = np.zeros((2, 2))
    J[0, 0] = (-1 + wEE * dF(wEE*rE - wEI*rI + I_ext_E, a_E, theta_E)) / tau_E
    J[0, 1] = (-wEI * dF(wEE*rE - wEI*rI + I_ext_E, a_E, theta_E)) / tau_E
    J[1, 0] = (wIE * dF(wIE*rE - wII*rI + I_ext_I, a_I, theta_I)) / tau_I
    J[1, 1] = (-1 - wII * dF(wIE*rE - wII*rI + I_ext_I, a_I, theta_I)) / tau_I
    evals = np.linalg.eig(J)[0]
    return evals
```

---

### Tutorial 3：零线斜率分析 (Nullcline Slope Analysis)

**E 零线斜率 (Eq. 12)**：

$$\left(\frac{dr_I}{dr_E}\right)_{\text{E零线}} = \frac{F_E' w_{EE} - 1}{F_E' w_{EI}}$$

**I 零线斜率 (Eq. 13)**：

$$\left(\frac{dr_I}{dr_E}\right)_{\text{I零线}} = \frac{F_I' w_{IE}}{F_I' w_{II} + 1}$$

**性质**：
- I 零线斜率始终为正
- E 零线斜率的符号取决于 $(F_E' w_{EE} - 1)$

**结论 1**：在稳定不动点处，I 零线比 E 零线更陡峭

**结论 2**：向抑制群添加输入时
- E 零线保持不变
- I 零线向左平移 $\delta I_I^{\rm ext} / w_{IE}$

---

### Tutorial 3：极限环与振荡 (Limit Cycles and Oscillations)

**振荡产生的条件**：特征值变为**复数**

**振荡参数**：$w_{EE}=6.4$，$w_{EI}=4.8$，$w_{IE}=6.0$，$w_{II}=1.2$，$I_E^{\rm ext}=0.8$

- 轨迹在相平面中形成**极限环 (limit cycle)**
- 兴奋 (E) 和抑制 (I) 群交替活跃
- 频率由特征值的虚部决定
- 振荡稳定性由特征值实部决定（正实部 → 振荡增长，负实部 → 振荡衰减）

**分岔 (Bifurcation)**：随着参数变化，系统行为发生剧烈变化
- 改变 $\tau_I$ 可以在稳态与振荡之间切换
- 零线保持不变，但向量场发生变化
- 直觉：$\tau_I$ 较小时，抑制活动变化快于兴奋活动，导致振荡

---

### Tutorial 3：抑制稳定网络 (Inhibition-Stabilized Network, ISN)

**基于 $\frac{\partial G_E}{\partial r_E}$ 的两种模式**：

$$\frac{\partial G_E}{\partial r_E} = \frac{1}{\tau_E}[-1 + w_{EE} F_E'] = \frac{1}{\tau_E}(F_E' w_{EE} - 1)$$

| 模式 | 条件 | E 零线斜率 | 行为 |
|------|------|----------|------|
| **非 ISN (non-ISN)** | $F_E' w_{EE} - 1 < 0$ | 负 | 增加对 I 的抑制 → E 减少 |
| **ISN** | $F_E' w_{EE} - 1 > 0$ | 正 | 增加对 I 的抑制 → E 矛盾地增加 |

**ISN 在皮层中很常见**：强的反复性兴奋 ($w_{EE}$ 较大) 创造了一种需要抑制来维持稳定的模式

**ISN 的矛盾行为**：
- 正常情况：抑制 I → E 增加（减少抑制）
- ISN 情况：抑制 I → E 也减少（因为 E 的自兴奋太强，需要 I 来稳定）

---

### Tutorial 3：工作记忆：持续活动 (Working Memory: Persistent Activity)

**机制**：多个不动点 + 噪声

1. 系统从低活动不动点开始
2. 短暂脉冲将状态推过不稳定不动点
3. 系统在高活动不动点稳定下来
4. 这代表了对刺激的"记忆"

**实现**：OU 噪声 + 短暂电流脉冲

```python
def my_inject(pars, t_start, t_lag=10.):
    I = np.zeros(Lt)
    N_start = int(t_start / dt)
    N_lag = int(t_lag / dt)
    I[N_start:N_start + N_lag] = 1.
    return I
```

**关键参数**：
- 脉冲幅度 $S_E$ 决定是否触发转换
- 临界脉冲幅度：刚好足够将状态推过不稳定不动点
- 足够大的脉冲：系统切换到持续活动
- 脉冲结束后：系统保持在高活动状态（工作记忆）

---

## 总结 (Summary)

---

### 第 2 周：核心概念 (Key Concepts)

### W2D3：线性系统 (Linear Systems)

- 欧拉积分 (Euler integration)
- 特征值分析 (Eigenvalue analysis)
- 马尔可夫过程与状态转移矩阵 (Markov processes & state transition matrices)
- 随机游走与扩散过程 (Random walks & diffusion processes)
- OU 过程与平衡方差 (OU process & equilibrium variance)
- 自回归模型与时间延迟矩阵 (AR models & time-delay matrices)

### W2D4：神经元模型 (Neuron Models)

- LIF 神经元动力学与欧拉积分 (LIF neuron dynamics & Euler integration)
- DC/GWN/OU 输入类型 (DC/GWN/OU input types)
- 相关输入与相关性转移 (Correlated inputs & correlation transfer)
- 基于电导的突触 (Conductance-based synapses)
- 自由膜电位与放电体制 (FMP & firing regimes)
- 短期可塑性：抑制与易化 (STP: depression & facilitation)
- STDP 学习规则与权重更新 (STDP learning rule & weight updates)
- P/M 追踪变量 (P/M trace variables)

### W2D5：网络动力学 (Network Dynamics)

- 放电频率模型与 sigmoid 传递函数 (Firing rate model & sigmoid transfer function)
- 不动点与特征值稳定性 (Fixed points & eigenvalue stability)
- Wilson-Cowan 模型与 E/I 耦合 (Wilson-Cowan model & E/I coupling)
- 零线与向量场 (Nullclines & vector fields)
- 雅可比矩阵与线性化 (Jacobian matrix & linearization)
- 零线斜率分析 (Nullcline slope analysis)
- 极限环与分岔 (Limit cycles & bifurcations)
- 抑制稳定网络 (Inhibition-stabilized network)
- 工作记忆与持续活动 (Working memory & persistent activity)

---

### 关键公式汇总 (Key Formulas)

$$\tau_m \frac{dV}{dt} = -(V-E_L) + \frac{I}{g_L} \quad \text{(LIF neuron)}$$

$$\tau_m \frac{dV}{dt} = -(V-E_L) - \frac{g_E}{g_L}(V-E_E) - \frac{g_I}{g_L}(V-E_I) + \frac{I_{\rm inj}}{g_L} \quad \text{(Conductance-based LIF)}$$

$$x_{k+1} = x_\infty + \lambda(x_k - x_\infty) + \sigma\eta \quad \text{(OU process)}$$

$$\text{Var}_{eq} = \frac{\sigma^2}{1-\lambda^2} \quad \text{(OU equilibrium variance)}$$

$$\tau \frac{dr}{dt} = -r + F(w \cdot r + I_{\rm ext}) \quad \text{(Firing rate model)}$$

$$F(x; a, \theta) = \frac{1}{1+e^{-a(x-\theta)}} - \frac{1}{1+e^{a\theta}} \quad \text{(Sigmoid transfer function)}$$

$$\tau_E \frac{dr_E}{dt} = -r_E + F_E(w_{EE}r_E - w_{EI}r_I + I_E^{\rm ext}) \quad \text{(Wilson-Cowan E)}$$

$$\tau_I \frac{dr_I}{dt} = -r_I + F_I(w_{IE}r_E - w_{II}r_I + I_I^{\rm ext}) \quad \text{(Wilson-Cowan I)}$$

$$\lambda = \frac{-1 + w \cdot F'(w \cdot r^* + I_{\rm ext})}{\tau} \quad \text{(Eigenvalue/stability)}$$

$$J = T^{-1}(FW - I) \quad \text{(Jacobian matrix)}$$

$$\frac{dr_I}{dr_E}\bigg|_{\text{E零线}} = \frac{F_E' w_{EE} - 1}{F_E' w_{EI}} \quad \text{(E nullcline slope)}$$

$$\frac{dr_I}{dr_E}\bigg|_{\text{I零线}} = \frac{F_I' w_{IE}}{F_I' w_{II} + 1} \quad \text{(I nullcline slope)}$$

$$\Delta W = \begin{cases} A_+ e^{\Delta t/\tau_+} & \Delta t < 0 \text{ (LTP)} \\ -A_- e^{-\Delta t/\tau_-} & \Delta t > 0 \text{ (LTD)} \end{cases} \quad \text{(STDP rule)}$$

---

### 教程之间的逻辑联系

| 教程 | 模型 | 维度 | 关键分析 |
|------|------|------|---------|
| W2D3 T1 | $\dot{x} = ax$ | 1D | 欧拉积分，特征值 |
| W2D3 T2 | 马尔可夫过程 | 2D | 状态转移矩阵，平衡态 |
| W2D3 T3 | OU 过程 | 1D | 随机游走，漂移扩散，平衡方差 |
| W2D3 T4 | 自回归模型 | 1D | 时间延迟矩阵，回归拟合 |
| W2D4 T1 | LIF 神经元 | 1D | 膜电位动力学，F-I 曲线，CV_ISI |
| W2D4 T2 | 相关 LIF | 2×1D | 相关输入，相关性转移 |
| W2D4 T3 | 电导 LIF + STP | 1D | 突触电导，u-R-g 动力学 |
| W2D4 T4 | LIF + STDP | N×1D | 权重更新，无监督学习 |
| W2D5 T1 | 单群放电频率 | 1D | F-I 曲线，不动点，特征值稳定性 |
| W2D5 T2 | Wilson-Cowan | 2D | 零线，向量场，相平面 |
| W2D5 T3 | WC + 分析 | 2D | 雅可比矩阵，极限环，ISN，工作记忆 |

**渐进关系**：从单群单特征值，到双群 2×2 雅可比矩阵（两个特征值可为实数或复数），实现更丰富的动力学（振荡和双稳态）。
