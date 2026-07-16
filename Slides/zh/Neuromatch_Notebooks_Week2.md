# Neuromatch 笔记本 — 第 2 周

线性系统 (Linear Systems) · 生物神经元模型 (Biological Neuron Models) · 动力系统 (Dynamical Systems)

---

## 概述 (Overview)

第 2 周聚焦于**动力系统与神经模型 (dynamical systems and neural models)** ——从线性系统到生物神经元模型，再到网络动力学：

| 天数      | 主题                      | 核心技能                               |
| -------- | -------------------------- | ---------------------------------------- |
| **W2D3** | 线性系统 (Linear Systems)             | 欧拉积分 (Euler integration)、振荡 (Oscillations)、自回归模型 (AR models) |
| **W2D4** | 生物神经元模型 (Biological Neuron Models)   | 泄漏积分发放神经元 (LIF neuron)、突触 (Synapses)、脉冲时间依赖可塑性 (STDP)               |
| **W2D5** | 动力系统 (Dynamical Systems)          | 放电频率模型 (Firing rate models)、Wilson-Cowan 模型、相平面分析 (Phase plane) |

**统一主题**：神经元和网络如何随时间演化，以及我们如何用数学方法对其动力学进行建模？

---

## W2D3：线性系统 (Linear Systems)

---

### 第 1 节：一维微分方程 (One-Dimensional Differential Equations)

最简单的动力系统：$\dot{x} = ax$

**解析解 (Analytical solution)**：$x(t) = x_0 e^{at}$

| $a$                  | 行为 (Behavior)                        |
| -------------------- | ------------------------------- |
| $a < 0$              | 指数衰减 → 0 (Exponential decay → 0)           |
| $a > 0$              | 指数增长 → ∞ (Exponential growth → ∞)          |
| $a = \text{complex}$ | 振荡（伴随增长/衰减）(Oscillation with growth/decay) |

**前向欧拉积分 (Forward Euler integration)**（数值解 (numerical solution)）：

$$x(t_i) = x(t_{i-1}) + \dot{x}(t_{i-1}) \cdot dt$$

---

### 第 2 节：振荡动力学 (Oscillatory Dynamics)

当 $a$ 为复数时（$a = \text{real} + i \cdot \text{imag}$），系统产生振荡：

**关键洞察 (Key insight)**：

- 实部 (Real part) → 增长/衰减率
- 虚部 (Imaginary part) → 振荡频率

$$x(t) = x_0 e^{(\text{real} + i \cdot \text{imag})t} = x_0 e^{\text{real} \cdot t} \cdot [\cos(\text{imag} \cdot t) + i \sin(\text{imag} \cdot t)]$$

**对于频率为 $f$ 的稳定振荡 (Stable oscillation)**：设实部 = 0，虚部 = $2\pi f$

---

### 第 3 节：二维线性系统 (Two-Dimensional Linear Systems)

扩展到二维：$\dot{\mathbf{x}} = \mathbf{A}\mathbf{x}$

$$\begin{bmatrix} \dot{x}_1 \\ \dot{x}_2 \end{bmatrix} = \begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}$$

**系统函数 (System function)**：

```python
def system(t, x, a00, a01, a10, a11):
    x1dot = a00 * x[0] + a01 * x[1]
    x2dot = a10 * x[0] + a11 * x[1]
    return np.array([x1dot, x2dot])
```

**特征值决定行为 (Eigenvalues determine behavior)**：

- 均为负 → 稳定节点 (Stable node)（收敛到原点）
- 均为正 → 不稳定节点 (Unstable node)（发散）
- 符号相反 → 鞍点 (Saddle point)
- 复数 → 振荡（螺旋）(Oscillation / Spiral)

---

### 随机游走与扩散 (Random Walks and Diffusion)

随机游走 (Random walk)：每一步以等概率移动 $\Delta x = \pm 1$。

```python
def random_walk_simulator(N, T, mu=0, sigma=1):
    steps = np.random.normal(mu, sigma, size=(N, T))
    sim = np.cumsum(steps, axis=1)
    return sim
```

**随机游走的性质 (Properties of random walks)**：

- 均值保持在 0 附近（与时间无关）
- 方差随时间线性增长：$\text{Var} \propto t$
- 这是一种**扩散过程 (diffusive process)**

---

### Ornstein-Uhlenbeck (OU) 过程 (Ornstein-Uhlenbeck Process)

将确定性漂移与随机扩散相结合：

$$x_{k+1} = x_\infty + \lambda(x_k - x_\infty) + \sigma \eta$$

```python
def simulate_ddm(lam, sig, x0, xinfty, T):
    t = np.arange(0, T, 1.)
    x = np.zeros_like(t)
    x[0] = x0
    for k in range(len(t)-1):
        x[k+1] = xinfty + lam * (x[k] - xinfty) + sig * np.random.standard_normal()
    return t, x
```

**平衡方差 (Equilibrium variance)**（当 $\lambda < 1$ 时）：

$$\text{Var} = \frac{\sigma^2}{1 - \lambda^2}$$

与随机游走不同，由于存在向 $x_\infty$ 的恢复漂移，方差会**趋于饱和 (saturates)**。

---

### 自回归模型 (Autoregressive Models)

转换视角：给定数据，学习其动力学。

**一阶自回归 (First-order AR)**：$x_{k+1} = \lambda x_k + \eta$

**高阶自回归 (Higher-order AR)**：$x_{k+1} = \alpha_0 + \alpha_1 x_k + \alpha_2 x_{k-1} + \dots + \alpha_r x_{k-r}$

**残差 (Residual)** = 数据 - 预测值：

```python
res = x2 - (p[0] + lam_hat * x1[:, 1])
```

**关键发现 (Key finding)**：人类在生成随机序列方面表现很差！自回归模型 (AR model) 可以比随机猜测更好地预测人类的"随机"输入（误差 < 0.5）。

六阶自回归模型 (6th-order AR model) 在欠拟合与过拟合之间找到了最佳平衡点。

---

## W2D4：生物神经元模型 (Biological Neuron Models)

---

### 泄漏积分发放模型 (Leaky Integrate-and-Fire Model, LIF)

最简单的神经元数学模型：

$$\tau_m \frac{dV}{dt} = -(V - E_L) + \frac{I}{g_L}$$

**参数 (Parameters)**：

| 符号 | 含义 | 典型值 |
|--------|---------|---------------|
| $\tau_m$ | 膜时间常数 (Membrane time constant) | 10 ms |
| $g_L$ | 漏电导 (Leak conductance) | 10 nS |
| $E_L$ | 静息电位 (Resting potential) | −75 mV |
| $V_{th}$ | 脉冲阈值 (Spike threshold) | −55 mV |
| $V_{reset}$ | 复位电位 (Reset potential) | −75 mV |
| $t_{ref}$ | 不应期 (Refractory time) | 2 ms |

---

### LIF 神经元：欧拉积分 (LIF Neuron: Euler Integration)

```python
def run_LIF(pars, Iinj, stop=False):
    # ... parameter setup ...
    for it in range(Lt - 1):
        if tr > 0:                          # refractory period
            v[it] = V_reset
            tr = tr - 1
        elif v[it] >= V_th:                 # spike!
            rec_spikes.append(it)
            v[it] = V_reset
            tr = tref / dt
        # Calculate the increment of the membrane potential
        dv = (dt / tau_m) * (-(v[it] - E_L) + Iinj[it] / g_L)
        # Update the membrane potential
        v[it + 1] = v[it] + dv
```

**关键输出 (Key outputs)**：膜电位轨迹 `v` 和脉冲时间 `sp`

---

### 放电频率与脉冲不规则性 (Firing Rate and Spike Irregularity)

**频率-电流曲线 (F-I curve)**：输出放电频率作为输入电流的函数。

**脉冲间隔变异系数 (CV of ISI, Coefficient of Variation of Inter-Spike Intervals)**：

$$\text{CV}_{\text{ISI}} = \frac{\text{std}(\text{ISI})}{\text{mean}(\text{ISI})}$$

| CV 值 | 含义 (Meaning)                                |
| -------- | -------------------------------------- |
| 0        | 完全规则（时钟般）(Perfectly regular / Clock-like)         |
| 1        | 泊松过程（最大不规则性）(Poisson process / Maximum irregularity) |

```python
def isi_cv_LIF(spike_times):
    if len(spike_times) >= 2:
        isi = np.diff(spike_times)
        cv = np.std(isi) / np.mean(isi)
    return isi, cv
```

---

### 输入相关性与输出相关性 (Input Correlations and Output Correlations)

相关输入如何影响输出相关性？

**两个神经元的相关输入 (Correlated input)**：

$$\frac{I_i}{g_L} = \mu + \sigma(\sqrt{1-c}\xi_i + \sqrt{c}\xi_c)$$

其中 $c \in [0,1]$ 控制公共输入的比例。

**样本相关系数 (Sample correlation coefficient)**：

```python
def my_CC(i, j):
    cov = np.sum((i - np.mean(i)) * (j - np.mean(j)))
    var_i = np.sum((i - np.mean(i))**2)
    var_j = np.sum((j - np.mean(j))**2)
    return cov / np.sqrt(var_i * var_j)
```

**关键发现 (Key finding)**：输出相关性 < 输入相关性。神经元充当了"相关性滤波器 (correlation filter)"的角色。

---

### 基于电导的突触 (Conductance-Based Synapses)

真实神经元接收的突触输入被建模为电导变化：

$$\tau_m \frac{dV}{dt} = -(V-E_L) - \frac{g_E}{g_L}(V-E_E) - \frac{g_I}{g_L}(V-E_I) + \frac{I_{\text{inj}}}{g_L}$$

**自由膜电位 (Free Membrane Potential, FMP)**：去除脉冲阈值的膜电位（人为设定 $V_{th} = \infty$）。

- 平均 FMP 高于阈值 → 规则放电 (Regular firing)
- 平均 FMP 低于阈值 → 不规则的噪声驱动放电 (Irregular, noise-driven firing)
- **兴奋/抑制平衡 (Balance of excitation/inhibition)** 决定放电模式

---

### 短期可塑性 (Short-Term Plasticity, STP)

突触可以根据近期脉冲历史改变强度：

**短期抑制 (Short-Term Depression, STD)**：突触随重复使用而减弱

- 参数：$U_0 = 0.5$，$\tau_d = 100$ ms，$\tau_f = 50$ ms
**短期易化 (Short-Term Facilitation, STF)**：突触随重复使用而增强
- 参数：$U_0 = 0.2$，$\tau_d = 100$ ms，$\tau_f = 750$ ms
**短期可塑性动力学 (STP dynamics)**：

$$\frac{du}{dt} = -\frac{u}{\tau_f} + U_0(1-u^-)\delta(t-t_{sp})$$

$$\frac{dR}{dt} = \frac{1-R}{\tau_d} - u^+ R^- \delta(t-t_{sp})$$

---

### 脉冲时间依赖可塑性 (Spike-Timing Dependent Plasticity, STDP)

突触权重根据突触前和突触后脉冲的**时间 (timing)** 进行变化：

$$\Delta W = \begin{cases} A_+ e^{(t_{pre}-t_{post})/\tau_+} & \text{if } t_{post} > t_{pre} \text{ (LTP)} \\ -A_- e^{-(t_{pre}-t_{post})/\tau_-} & \text{if } t_{post} < t_{pre} \text{ (LTD)} \end{cases}$$

**追踪变量 (Tracking variables)**：

```python
def generate_P(pars, pre_spike_train_ex):
    A_plus, tau_stdp = pars['A_plus'], pars['tau_stdp']
    dt = pars['dt']
    P = np.zeros(pre_spike_train_ex.shape)
    for it in range(Lt - 1):
        dP = -(dt / tau_stdp) * P[:, it] + A_plus * pre_spike_train_ex[:, it + 1]
        P[:, it + 1] = P[:, it] + dP
    return P
```

**关键洞察 (Key insight)**：STDP 使来自**相关 (correlated)** 突触前神经元的突触得到增强，而不相关的突触则被减弱——这是一种无监督学习 (unsupervised learning) 的形式。

---

## W2D5：动力系统 (Dynamical Systems)

---

### 单群放电频率模型 (Single Population Firing Rate Model)

不再对单个神经元进行建模，而是对一个群的**平均放电频率 (average firing rate)** 进行建模：

$$\tau \frac{dr}{dt} = -r + F(w \cdot r + I_{\text{ext}})$$

**Sigmoid 传递函数 (Sigmoid transfer function)**：

$$F(x; a, \theta) = \frac{1}{1 + e^{-a(x-\theta)}} - \frac{1}{1 + e^{a\theta}}$$

```python
def F(x, a, theta):
    f = (1 + np.exp(-a * (x - theta)))**-1 - (1 + np.exp(a * theta))**-1
    return f
```

---

### 不动点与稳定性 (Fixed Points and Stability)

**不动点 (Fixed point)**：$\frac{dr}{dt} = 0$ 时的 $r$ 值

$$-r^* + F(w \cdot r^* + I_{\text{ext}}) = 0$$

**特征值 (Eigenvalue)**（稳定性 (stability)）：

$$\lambda = \frac{-1 + w \cdot F'(w \cdot r^* + I_{\text{ext}})}{\tau}$$

| $\lambda$ | 稳定性 (Stability) |
|-----------|-----------|
| $\lambda < 0$ | 稳定（吸引）(Stable / Attracting) |
| $\lambda > 0$ | 不稳定（排斥）(Unstable / Repelling) |

```python
def eig_single(fp, tau, a, theta, w, I_ext, **other_pars):
    eig = (-1 + w * dF(w * fp + I_ext, a, theta)) / tau
    return eig
```

---

### Wilson-Cowan 模型：兴奋/抑制群 (Wilson-Cowan Model: E/I Populations)

两个耦合群（兴奋 (Excitatory) + 抑制 (Inhibitory)）：

$$\tau_E \frac{dr_E}{dt} = -r_E + F_E(w_{EE}r_E - w_{EI}r_I + I_E^{\text{ext}})$$

$$\tau_I \frac{dr_I}{dt} = -r_I + F_I(w_{IE}r_E - w_{II}r_I + I_I^{\text{ext}})$$

```python
def simulate_wc(tau_E, a_E, theta_E, tau_I, a_I, theta_I,
                wEE, wEI, wIE, wII, I_ext_E, I_ext_I,
                rE_init, rI_init, dt, range_t, **other_pars):
    for k in range(Lt - 1):
        drE = (dt/tau_E) * (-rE[k] + F(wEE*rE[k] - wEI*rI[k] + I_ext_E[k], a_E, theta_E))
        drI = (dt/tau_I) * (-rI[k] + F(wIE*rE[k] - wII*rI[k] + I_ext_I[k], a_I, theta_I))
        rE[k+1] = rE[k] + drE
        rI[k+1] = rI[k] + drI
    return rE, rI
```

---

### 相平面分析 (Phase Plane Analysis)

绘制 $r_E$ 与 $r_I$ 的关系图以可视化系统动力学：

**零线 (Nullclines)**：$\frac{dr_E}{dt} = 0$ 或 $\frac{dr_I}{dt} = 0$ 的曲线

```python
def get_E_nullcline(rE, a_E, theta_E, wEE, wEI, I_ext_E, **other_pars):
    rI = 1/wEI * (wEE * rE - F_inv(rE, a_E, theta_E) + I_ext_E)
    return rI
def get_I_nullcline(rI, a_I, theta_I, wIE, wII, I_ext_I, **other_pars):
    rE = 1/wIE * (wII * rI + F_inv(rI, a_I, theta_I) - I_ext_I)
    return rE
```

**向量场 (Vector field)**：在每个点上显示 $(\frac{dr_E}{dt}, \frac{dr_I}{dt})$ 的箭头

```python
def EIderivs(rE, rI, tau_E, a_E, theta_E, wEE, wEI, I_ext_E,
             tau_I, a_I, theta_I, wIE, wII, I_ext_I, **other_pars):
    drEdt = (-rE + F(wEE*rE - wEI*rI + I_ext_E, a_E, theta_E)) / tau_E
    drIdt = (-rI + F(wIE*rE - wII*rI + I_ext_I, a_I, theta_I)) / tau_I
    return drEdt, drIdt
```

---

### 雅可比矩阵与稳定性 (Jacobian Matrix and Stability)

对于二维 Wilson-Cowan 系统，稳定性由**雅可比矩阵 (Jacobian)** 决定：

$$J = \begin{bmatrix} \frac{\partial G_E}{\partial r_E} & \frac{\partial G_E}{\partial r_I} \\ \frac{\partial G_I}{\partial r_E} & \frac{\partial G_I}{\partial r_I} \end{bmatrix}$$

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

### 极限环与振荡 (Limit Cycles and Oscillations)

当特征值变为**复数 (complex)** 时，系统产生振荡：

**振荡参数 (Oscillatory parameters)**：$w_{EE}=6.4$，$w_{EI}=4.8$，$w_{IE}=6.0$，$w_{II}=1.2$，$I_E^{\text{ext}}=0.8$

- 轨迹在相平面中形成**极限环 (limit cycle)**
- 兴奋 (E) 和抑制 (I) 群交替活跃
- 频率由特征值的虚部决定
**分岔 (Bifurcation)**：随着参数变化，系统行为发生剧烈变化
- 改变 $\tau_I$ 可以在稳态与振荡之间切换
- 零线保持不变，但向量场发生变化

---

### 抑制稳定网络 (Inhibition-Stabilized Network, ISN)

基于 $\frac{\partial G_E}{\partial r_E}$ 的两种模式：

| 模式 (Regime) | 条件 (Condition) | 行为 (Behavior) |
|--------|-----------|----------|
| **非 ISN (non-ISN)** | $\frac{\partial G_E}{\partial r_E} < 0$ | 增加对 I 的抑制 → E 减少 |
| **ISN** | $\frac{\partial G_E}{\partial r_E} > 0$ | 增加对 I 的抑制 → E 也减少（矛盾地）(paradoxically) |

**ISN 在皮层中很常见 (ISN is common in cortex)**：强的反复性兴奋 (recurrent excitation)（$w_{EE}$ 较大）创造了一种需要抑制来维持稳定的模式。

---

### 工作记忆：持续活动 (Working Memory: Persistent Activity)

短暂输入可以触发**持续活动 (sustained activity)**，其持续时间超过刺激本身：

**机制 (Mechanism)**：多个不动点 + 噪声

1. 系统从低活动不动点开始
2. 短暂脉冲将状态推过不稳定不动点
3. 系统在高活动不动点稳定下来
4. 这代表了对刺激的"记忆"
**Wilson-Cowan 模型展示了这一现象**：
- 无脉冲时：系统保持在静息状态
- 足够大的脉冲：系统切换到持续活动
- 临界脉冲幅度决定了转换

---

## 总结 (Summary)

---

### 第 2 周：核心概念 (Key Concepts)

### W2D3：线性系统 (Linear Systems)

- 欧拉积分 (Euler integration)
- 特征值分析 (Eigenvalue analysis)
- 随机游走与 OU 过程 (Random walks & OU process)
- 自回归模型 (Autoregressive models)

### W2D4：神经元模型 (Neuron Models)

- LIF 神经元动力学 (LIF neuron dynamics)
- 基于电导的突触 (Conductance-based synapses)
- 短期可塑性 (Short-term plasticity)
- STDP 学习规则 (STDP learning rule)

### W2D5：网络动力学 (Network Dynamics)

- 放电频率模型 (Firing rate models)
- Wilson-Cowan 模型
- 相平面分析 (Phase plane analysis)
- 不动点与稳定性 (Fixed points & stability)

---

### 关键公式 (Key Formulas)

$$\tau_m \frac{dV}{dt} = -(V-E_L) + \frac{I}{g_L} \quad \text{(LIF neuron)}$$

$$\tau \frac{dr}{dt} = -r + F(w \cdot r + I_{\text{ext}}) \quad \text{(Firing rate model)}$$

$$\tau_E \frac{dr_E}{dt} = -r_E + F_E(w_{EE}r_E - w_{EI}r_I + I_E^{\text{ext}}) \quad \text{(Wilson-Cowan)}$$

$$\lambda = \frac{-1 + w \cdot F'(w \cdot r^* + I_{\text{ext}})}{\tau} \quad \text{(Eigenvalue/stability)}$$

$$\text{Var} = \frac{\sigma^2}{1-\lambda^2} \quad \text{(OU equilibrium variance)}$$

---
