# Neuromatch 笔记本 — 第 0 周

Python 与 LIF 神经元 · 脉冲与不应期 · 线性代数 · 微积分 · 概率论

---

## 概述 (Overview)

第 0 周为计算神经科学 (Computational Neuroscience) 建立数学和编程基础：

| 天数       | 主题                     | 核心技能                               |
| -------- | -------------------------- | ---------------------------------------- |
| **W0D1** | Python 与 LIF 模型         | 欧拉积分 (Euler Integration)、NumPy 基础          |
| **W0D2** | 脉冲与代码组织             | 布尔索引 (Boolean Indexing)、类 (Classes)                |
| **W0D3** | 线性代数 (Linear Algebra)             | 向量 (Vectors)、矩阵 (Matrices)、特征值 (Eigenvalues)           |
| **W0D4** | 微积分 (Calculus)                   | 微分 (Differentiation)、积分 (Integration)、链式法则 (Chain Rule) |
| **W0D5** | 概率与统计 (Probability & Statistics)   | 分布 (Distributions)、泊松脉冲 (Poisson Spiking)           |

**统一主题**：从第一性原理模拟一个神经元，然后建立数学来分析它。

---

## W0D1：Python 基础与 LIF 模型

---

### LIF 神经元模型 (The LIF Neuron Model)

**漏积分放电** (Leaky-Integrate-and-Fire) 神经元是最简单的脉冲神经元模型：

$$\tau_m \frac{d}{dt}V(t) = E_L - V(t) + R\,I(t) \quad \text{if } V(t) \leq V_{threshold}$$

**重置条件** (Reset Condition)：

$$V(t) = V_{reset} \quad \text{if } V(t) > V_{threshold}$$

**物理直觉**：膜就像一个漏电的电容器 —— 它积分输入电流，但也向 $E_L$ 方向泄漏电荷。

| 参数               | 符号        | 值            | 含义              |
| ---------------------- | ----------- | ------------- | ------------------- |
| 膜时间常数 (Membrane Time Constant) | $\tau_m$    | 20 ms         | V 响应的速度       |
| 漏电位 (Leak Potential)         | $E_L$       | -60 mV        | 静息电压           |
| 重置电压 (Reset Voltage)          | $V_{reset}$ | -70 mV        | 脉冲后的电压       |
| 阈值 (Threshold)              | $V_{th}$    | -50 mV        | 脉冲触发点         |
| 膜电阻 (Membrane Resistance)    | $R$         | 100 M$\Omega$ | 输入灵敏度         |
| 平均输入电流 (Mean Input Current)     | $I_{mean}$  | 250 pA        | 驱动强度           |

---

### 关键公式：欧拉离散化 (Euler Discretization)

该常微分方程 (ODE) 通常无法解析求解。我们对其进行离散化：

$$V(t + \Delta t) = V(t) + \frac{\Delta t}{\tau_m}\left(E_L - V(t) + R\,I(t)\right)$$

这是**前向欧拉法** (Forward Euler Method) —— 所有笔记本中重复使用最多的技术。

**代码模式**（出现在每个模拟笔记本中）：

```python
v = el                              # initial condition V(0) = E_L
for step in range(step_end):
    t = step * dt
    i = compute_input(t)            # synaptic input at this step
    v = v + dt/tau * (el - v + r*i) # Euler update
```

**欧拉法为何有效**：对于足够小的 $\Delta t$，$dV/dt$ 在一步内的线性近似是准确的。误差为 $o(\Delta t)$ —— 步长减半，误差减半。

---

### 推导：从 ODE 到差分方程 (From ODE to Difference Equation)

**第 1 步**：从连续膜方程 (ODE) 开始：

$$\tau_m \frac{dV}{dt} = E_L - V(t) + R\,I(t)$$

**第 2 步**：用有限差分 (Finite Difference) 近似导数 —— 将 $\frac{dV}{dt}$ 替换为 $\frac{\Delta V}{\Delta t}$：

$$\tau_m\frac{\Delta V}{\Delta t}=\tau_m \frac{V(t + \Delta t) - V(t)}{\Delta t} = E_L - V(t) + R\,I(t)$$

这不再是微分方程 —— 而是将 $V(t+\Delta t)$ 与 $V(t)$ 联系起来的代数方程。

**第 3 步**：两边乘以 $\frac{\Delta t}{\tau_m}$：

$$V(t + \Delta t) - V(t) = \frac{\Delta t}{\tau_m}\left(E_L - V(t) + R\,I(t)\right)$$

---

### 推导：迭代更新规则 (Iterative Update Rule)

**第 4 步**：将 $V(t)$ 移到右边：

$$\boxed{\;V(t + \Delta t) = V(t) + \frac{\Delta t}{\tau_m}\left(E_L - V(t) + R\,I(t)\right)\;}$$

这是**前向欧拉更新规则** (Forward Euler Update Rule) —— 给定当前状态 $V(t)$ 和输入 $I(t)$，我们可以计算下一个状态 $V(t+\Delta t)$。

**第 5 步**：转换为代码 —— 让 `v` 表示 $V(t)$：

```python
v = v + dt / tau * (el - v + r * i)
#  ↑   ↑
#  |   └── right side: evaluates old V(t)
#  └────── left side:  becomes new V(t+Δt) after assignment
```

**同一个变量** `v` 出现在等号两边，因为 Python 先计算右边，然后将结果赋值给左边。

**几何直觉**：欧拉法将曲线 $V(t)$ 近似为一系列短直线段，每段的斜率为在当前点计算的 $\frac{dV}{dt}$。

---

### 输入电流模型 (Input Current Models)

**正弦输入** (Sinusoidal Input)（确定性的）

$$I(t) = I_{mean}\left(1 + \sin\left(\frac{2\pi}{0.01}\,t\right)\right)$$

周期 = 10 ms，在 $0$ 和 $2 I_{mean}$ 之间振荡。

**随机输入** (Random Input)（随机性的）

$$I(t) = I_{mean}\left(1 + 0.1\sqrt{\frac{t_{max}}{\Delta t}}\,\xi(t)\right)$$

其中 $\xi(t) \sim \mathcal{U}(-1, 1)$。

缩放因子 $0.1\sqrt{t_{max}/\Delta t}$ 控制噪声幅度。

**编程难点**：因子 $0.1\sqrt{t_{max}/\Delta t}$ **不是**任意的 —— 它确保无论步长如何，噪声具有相同的统计功率。这是随机模拟中的常见模式。

---

### 从标量到向量到矩阵 (From Scalar to Vector to Matrix)

教程有意通过三个层次的表示进行递进：

| 层次          | 存储方式                      | 循环方式                                | 性能              |
| ------------- | ----------------------------- | ----------------------------------- | ----------------- |
| **标量** (Scalar)    | `v = el`                      | `for step: v = v + …`             | 慢，一个神经元    |
| **一维数组** (1-D Array) | `v_n = [el]*n`                | `for j: v_n[j] = v_n[j] + …`      | 中等，n 个神经元  |
| **二维数组** (2-D Array) | `v_n = el*np.ones([n,steps])` | `v_n[:,step] = v_n[:,step-1] + …` | 快，向量化        |

每个层次模拟的是**同一个 LIF 神经元** —— 区别在于我们如何在内存中组织数据以及如何编写循环。

---

### NumPy 向量化 (NumPy Vectorization)

二维版本 `v_n[:,step] = …` **同时**操作所有神经元 —— 这就是 NumPy 向量化，比 Python 循环快几个数量级。

**为什么？** Python 的 `for` 循环在运行时解释每次迭代。NumPy 将整个列操作委托给优化的 C/Fortran 代码 —— 一次函数调用代替 $n$ 次循环迭代。

**形状约定**：`v_n[j, step]` = 神经元 $j$ 在时间步 $step$ 的膜电位。

```
v_n.shape = (n_neurons, n_steps)
         step 0   step 1   step 2   ...
neuron 0 [ -60      -58      -55    ... ]
neuron 1 [ -60      -59      -57    ... ]
neuron 2 [ -60      -57      -53    ... ]
```

**关键索引**：`v_n[j, step]` → 标量，`v_n[j, :]` → 一个神经元的轨迹，`v_n[:, step]` → 一个时间点的所有神经元。最后一种形式是向量化更新的目标。

---

### N 次实现的样本统计 (Sample Statistics over N Realizations)

$N$ 个独立神经元接收不同的随机输入：

**样本均值** (Sample Mean)：

$$\langle V(t)\rangle = \frac{1}{N}\sum_{n=1}^N V_n(t)$$

**样本方差** (Sample Variance)（带贝塞尔校正 Bessel's Correction）：

$$\text{Var}(t) = \frac{1}{N-1}\sum_{n=1}^N \left(V_n(t) - \langle V(t)\rangle\right)^2$$

**标准差** (Standard Deviation)：$\sigma(t) = \sqrt{\text{Var}(t)}$

我们绘制 $\sigma$（而不是方差），因为它与 $V$ 具有相同的单位（毫伏），便于视觉比较。

---

### 在 NumPy 中计算统计量 (Computing Statistics in NumPy)

$N$ 个神经元存储为形状为 `(n, steps)` 的 `v_n`：

```python
v_mean = np.mean(v_n, axis=0)   # shape: (steps,)
v_std  = np.std(v_n, axis=0)    # shape: (steps,)
```

`axis=0` 表示"折叠神经元维度" —— 每个元素是一个时间步。

**为什么是 `axis=0`？** 数组形状为 `(n_neurons, n_steps)`。轴 0 是神经元轴。`np.mean(…, axis=0)` 跨神经元求平均，每个时间步留下一个值。

**可视化**：将均值绘制为粗线，$\pm\sigma$ 绘制为阴影带：

```python
plt.plot(t_range, v_mean, 'C0', label='mean')
plt.plot(t_range, v_mean + v_std, 'C7', label='mean ± std')
plt.plot(t_range, v_mean - v_std, 'C7')
```

---

## W0D2：脉冲、不应期与代码组织 (Spikes, Refractory Period & Code Organization)

---

### 向 LIF 添加脉冲 (Adding Spikes to LIF)

重置条件 —— 当 $V \geq V_{th}$ 时，记录脉冲并重置。

---

### 什么是布尔数组？ (What is a Boolean Array?)

对数组进行比较会返回一个 `True`/`False` 数组：

```python
v = np.array([-55, -48, -60, -50, -45])   # 5 neurons
vth = -50                                   # threshold
spiked = (v >= vth)
# Result: [False,  True, False,  True,  5]
#              ↑          ↑          ↑
#         -55<-50   -48>=-50   -45>=-50
```

然后**使用布尔数组作为索引**来仅选择 True 元素：

```python
v[spiked]            # array([-48, -50, -45])  — values that crossed threshold
v[spiked] = -70      # reset ONLY those neurons to V_reset
```

**两行脉冲逻辑**（同时应用于所有 $n$ 个神经元）：

```python
spiked = (v_n[:, step] >= vth)   # which neurons spiked?
v_n[spiked, step] = vr           # reset only those to V_reset
```

不需要 `for` 循环 —— NumPy 在一次操作中处理所有 $n$ 个神经元。

---

### 存储脉冲数据 (Storing Spike Data)

**两种记录脉冲的方式**：

**字典列表** (Dictionary of Lists)

```python
spikes = {j: [] for j in range(n)}
# When neuron j spikes:
spikes[j] += [t]
```

存储实际的脉冲时间。灵活但较慢。

**二元光栅数组** (Binary Raster Array)

```python
raster = np.zeros([n, steps])
# When neurons spike:
raster[spiked, step] = 1.
```

高效网格：1 = 脉冲，0 = 无脉冲。

使用 `plt.scatter(times, neuron_ids)` 或 `plt.imshow(raster)` 绘图。

---

### 布尔索引 —— 关键优化 (Boolean Indexing — The Critical Optimization)

**不使用布尔索引**（慢，练习 2）：

```python
for j in range(n):
    if v_n[j, step] >= vth:
        v_n[j, step] = vr
        spikes[j] += [t]
        spikes_n[step] += 1
```

**使用布尔索引**（快，练习 3-4）：

```python
spiked = (v_n[:, step] >= vth)   # one vectorized comparison
v_n[spiked, step] = vr           # one vectorized assignment
for j in np.where(spiked)[0]:    # only loop over actual spikes
    spikes[j] += [t]
```

**为什么这很重要**：对于 $n = 500$ 个神经元和 $150$ 个时间步，内循环运行 $500 \times 150 = 75{,}000$ 次。布尔索引将其减少为仅对实际发放脉冲的神经元进行迭代 —— 通常只是很小的一部分。

---

### 不应期 (Refractory Period)

脉冲后，将 $V = V_{reset}$ 钳制 (Clamp) 持续时间 $t_{ref}$：

```python
# Track last spike time per neuron
last_spike = -t_ref * np.ones([n])   # initialized so all start unclamped
# In the simulation loop:
spiked = (v_n[:, step] >= vth)
v_n[spiked, step] = vr
last_spike[spiked] = t               # record spike time
clamped = (last_spike + t_ref > t)   # still within refractory period
v_n[clamped, step] = vr              # force to reset voltage
```

**操作顺序至关重要**：

1. 欧拉积分 → 计算 $V$
2. 检测脉冲 ($V \geq V_{th}$) → 重置、记录
3. 应用不应期钳制 → 覆盖不应期神经元的 $V$

如果在步骤 2 之前执行步骤 3，会错误地钳制刚刚发放脉冲的神经元。

**随机不应期**：$t_{ref} = \mu + \sigma\,\mathcal{N}(0,1)$，裁剪到 $\geq 0$。

---

### 代码重构：函数和类 (Code Refactoring: Functions and Classes)

相同的模拟逻辑以三种方式重新组织：

**原始循环** (Raw Loop)

所有逻辑内联：

```python
for step in range(step_end):
    v_n[:,step] = v_n[:,step-1] \
      + dt/tau*(el - v_n[:,step-1] \
      + r*i[:,step])
    spiked = (v_n[:,step] >= vth)
    v_n[spiked,step] = vr
```

难以重用或测试。

**函数** (Functions)

```python
def ode_step(v, i, dt):
    return v + dt/tau*(el-v+r*i)
def spike_clamp(v, delta):
    spiked = (v >= vth)
    v[spiked] = vr
    clamped = (delta < t_ref)
    v[clamped] = vr
    return v, spiked
```

可重用、可测试。

**类** (Class)

```python
class LIFNeurons:
    def __init__(self, n, ...):
        self.v = el*np.ones(n)
        self.last_spike = ...
    def ode_step(self, dt, i):
        self.v = self.v + ...
        self.spiked = (self.v>=vth)
```

有状态的、封装的。

---

### LIFNeurons 类 —— 关键属性 (The LIFNeurons Class — Key Attributes)

```python
class LIFNeurons:
    def __init__(self, n, t_ref_mu=0.01, t_ref_sigma=0.002,
                 tau=20e-3, el=-60e-3, vr=-70e-3, vth=-50e-3, r=100e6):
        self.n = n
        self.tau, self.el, self.vr, self.vth, self.r = tau, el, vr, vth, r
        self.t_ref = t_ref_mu + t_ref_sigma * np.random.normal(size=n)
        self.t_ref[self.t_ref < 0] = 0
        self.v = el * np.ones(n)
        self.spiked = self.v >= vth
        self.last_spike = -self.t_ref * np.ones(n)
        self.t, self.steps = 0., 0
```

| 属性              | 类型              | 含义                        |
| ----------------- | ----------------- | ----------------------------- |
| `self.v`          | array `(n,)`      | 当前膜电位                    |
| `self.spiked`     | bool array `(n,)` | 哪些神经元刚刚发放脉冲        |
| `self.last_spike` | array `(n,)`      | 每个神经元上次脉冲的时间      |
| `self.t_ref`      | array `(n,)`      | 每个神经元的不应期            |
| `self.t`          | float             | 模拟时钟                      |
| `self.steps`      | int               | 积分步数                      |

---

## W0D3：线性代数（向量与矩阵）(Linear Algebra — Vectors & Matrices)

---

### 向量 —— 点积与几何 (Vectors — Dot Product and Geometry)

**点积** (Dot Product) 将代数与几何联系起来：

$$\mathbf{x} \cdot \mathbf{y} = \sum_i x_i y_i = \|\mathbf{x}\|\,\|\mathbf{y}\|\cos\theta$$

**向量长度** (Vector Length)：$\|\mathbf{x}\| = \sqrt{\sum_i x_i^2}$；

**单位向量** (Unit Vector)：$\tilde{\mathbf{x}} = \mathbf{x} / \|\mathbf{x}\|$

**神经科学应用**：单个神经元的放电率是一个点积：$y = \mathbf{w} \cdot \mathbf{r} = \sum_i w_i r_i$

其中 $\mathbf{w}$ 是权重向量，$\mathbf{r}$ 是输入放电率向量。

**线性组合** (Linear Combination)：$\mathbf{y} = \sum_i \alpha_i \mathbf{b}_i$

一组向量的**张成空间** (Span) 是所有可能的线性组合。如果向量是**线性无关** (Linearly Independent) 的，它们构成其张成空间的**基** (Basis)。

**代码**：`np.dot(x, y)`、`np.linalg.norm(x)`

---

### 矩阵 —— 线性变换 (Matrices — Linear Transformations)

矩阵 $W$ 表示线性变换 (Linear Transformation) $\mathbf{y} = W\mathbf{x}$：

$$\begin{bmatrix} y_1 \\ y_2 \end{bmatrix} = \begin{bmatrix} W_{11} & W_{12} \\ W_{21} & W_{22} \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}$$

**关键操作**：

| 操作             | 公式                                          | 代码                       |
| --------------------- | ------------------------------------------------ | -------------------------- |
| 矩阵-向量乘法         | $\mathbf{y} = W\mathbf{x}$                       | `y = W @ x`                |
| 矩阵求逆        | $W^{-1}W = I$                                    | `W_inv = np.linalg.inv(W)` |
| 矩阵乘法 | $C_{ij} = \text{row}_i(A) \cdot \text{col}_j(B)$ | `C = A @ B`                |

**神经科学应用**：权重矩阵将输入神经元群体映射到输出神经元：

$$\mathbf{r}_{out} = W \mathbf{r}_{in}$$

$W$ 的**秩** (Rank) 揭示了映射的内在维度。

---

### 特征值与特征向量 (Eigenvalues and Eigenvectors)

特征值方程 (Eigenvalue Equation) $W\mathbf{v} = \lambda\mathbf{v}$ 找到不旋转的方向：

- $\mathbf{v}$：特征向量 (Eigenvector) —— 被 $W$ 保持的方向
- $\lambda$：特征值 (Eigenvalue) —— 沿该方向的缩放因子
> [!tip] 直觉
> 如果 $\lambda > 1$，变换沿 $\mathbf{v}$ 方向拉伸；如果 $0 < \lambda < 1$，则压缩；如果 $\lambda < 0$，则翻转。

**代码**：

```python
eigenvalues, eigenvectors = np.linalg.eig(W)
```

**在神经科学中**：连接矩阵 $W$ 的特征值决定了网络动力学的稳定性。如果所有 $|\lambda| < 1$，扰动衰减（稳定）。如果任何 $|\lambda| > 1$，扰动增长（不稳定）。这是 W2D3（线性动力系统 Linear Dynamical Systems）的核心内容。

---

### `np.dot` vs `*` vs `@`

| 运算符          | 语义        | 一维             | 二维                   | 三维+             |
| ----------------- | --------------- | --------------- | --------------------- | ---------------- |
| `*`               | 逐元素 (Element-wise)    | $x_i \cdot y_i$ | $A_{ij} \cdot B_{ij}$ | 逐元素     |
| `np.dot`          | 点积/乘积     | 标量（内积）  | 矩阵乘法       | 轴收缩 |
| `@` / `np.matmul` | 矩阵乘法 | 不支持   | 矩阵乘法       | 批量矩阵乘法     |

**实用建议**：使用 `@` 进行矩阵乘法，`*` 进行逐元素运算，`np.dot` 进行一维内积。

**常见错误**：当你想用 `@` 时使用了 `*`。

```python
# Wrong: element-wise, not matrix multiply
C = A * B
# Right: matrix multiply
C = A @ B
```

---

## W0D4：微积分（微分与积分）(Calculus — Differentiation & Integration)

---

### 数值微分 (Numerical Differentiation)

**有限差分** (Finite Difference) —— 近似导数：

$$f'(a) \approx \frac{f(a+h) - f(a)}{h}$$

**中心差分** (Central Difference)（更精确）：

$$f'(a) \approx \frac{f(a+h) - f(a-h)}{2h}$$

**使用 SymPy 的代码**（符号计算，精确）：

```python
import sympy as sp
t = sp.Symbol('t')
f = t * sp.exp(-t / tau)
df = sp.diff(f, t)              # exact derivative
integral = sp.integrate(f, t)   # exact integral
```

**符号计算为何重要**：对于 alpha 函数 $f(t) = t\,e^{-t/\tau}$，导数告诉我们峰值何时出现 ($t = \tau$)，增益告诉我们神经元的灵敏度。

---

### 微分法则 (Differentiation Rules)

**乘积法则** (Product Rule)

$$\frac{d}{dt}[u \cdot v] = v\frac{du}{dt} + u\frac{dv}{dt}$$

示例：$\frac{d}{dt}[t \cdot e^{-t/\tau}] = e^{-t/\tau} + t \cdot (-\frac{1}{\tau})e^{-t/\tau}$

**链式法则** (Chain Rule)

$$\frac{dr}{da} = \frac{dr}{dt} \cdot \frac{dt}{da}$$

示例：如果 $r = \sigma(V)$ 且 $V = RI$，则 $\frac{dr}{dI} = \sigma'(V) \cdot R$

**偏导数** (Partial Derivatives)

$$\frac{\partial f}{\partial x_1}\bigg|_{x_2 \text{ fixed}}$$

对于 $f(x_1, x_2) = x_1^2 x_2$：$\frac{\partial f}{\partial x_1} = 2x_1 x_2$

**链式法则是反向传播** (Backpropagation, W1D5) **的数学基础**。当损失 $L$ 通过中间变量依赖于权重 $\mathbf{w}$ 时，链式法则通过整个计算图连接 $\frac{\partial L}{\partial \mathbf{w}}$。

---

### 神经传递函数 (Neural Transfer Functions)

**Sigmoid** (Logistic) 传递函数将输入电流映射到放电率：

$$\sigma(x; a, \theta) = \frac{1}{1 + e^{-a(x - \theta)}}$$

**增益** (Gain) = 传递函数的导数：

$$g = \frac{d\sigma}{dx} = a\,\sigma(1 - \sigma)$$

| 参数 | 含义   | 效果                                        |
| --------- | --------- | --------------------------------------------- |
| $a$       | 陡度 (Steepness) | $a$ 越大 → 过渡越尖锐               |
| $\theta$  | 阈值 (Threshold) | 沿 x 轴平移曲线                 |
| $g$       | 增益 (Gain)      | 在 $x = \theta$ 处最大，此时 $\sigma = 0.5$ |

**生物学解释**：增益决定了神经元对阈值附近小输入变化的灵敏度。

---

### 数值积分 (Numerical Integration)

**黎曼和** (Riemann Sum) —— 近似积分：

$$\int_a^b f(x)\,dx \approx \sum_{i} f(x_i)\,\Delta x$$

**使用 `np.cumsum` 的代码**：

```python
dx = 0.01
x = np.arange(0, 10, dx)
y = np.sin(x)
cumulative_integral = np.cumsum(y) * dx
```

`np.cumsum` 返回累积和：$[y_0,\; y_0{+}y_1,\; y_0{+}y_1{+}y_2,\; \ldots]$

乘以 $\Delta x$，得到每个点处曲线下方的累积面积。

**微分是高通滤波，积分是低通滤波**：导数放大快速变化（高频）；积分将其平滑（累积慢趋势）。这是神经科学中信号处理的基础。

---

## W0D5：概率与统计 (Probability & Statistics)

---

### 离散分布 (Discrete Distributions)

**二项分布** (Binomial Distribution)

$$P(k \mid n, p) = \binom{n}{k} p^k (1-p)^{n-k}$$

在 $n$ 次独立试验中 $k$ 次成功，每次成功概率为 $p$。

```python
samples = np.random.binomial(n=10, p=0.5, size=1000)
```

**泊松分布** (Poisson Distribution)

$$P(k \mid \lambda) = \frac{\lambda^k e^{-\lambda}}{k!}$$

固定区间内的事件数，速率为 $\lambda$。

```python
samples = np.random.poisson(lam=5, size=1000)
```

**二项分布的极限**：当 $n \to \infty$，$p \to 0$，$np = \lambda$ 时。

---

### 连续分布 (Continuous Distributions)

**高斯（正态）分布** (Gaussian/Normal Distribution)

$$f(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

```python
samples = np.random.normal(mu=0, sigma=1, size=1000)
from scipy.stats import norm
pdf = norm.pdf(x, loc=0, scale=1)
```

**均匀分布** (Uniform Distribution)

$$f(x \mid a, b) = \frac{1}{b-a} \quad \text{for } x \in [a,b]$$

```python
samples = np.random.uniform(-1, 1, size=1000)
```

在 W0D1-W0D2 中广泛用于随机输入电流：$\xi(t) \sim \mathcal{U}(-1,1)$。

---

### 泊松脉冲模型 (Poisson Spiking Model)

神经脉冲序列通常被建模为**泊松过程** (Poisson Process)：

| 层次           | 变量                     | 分布                           |
| --------------- | -------------------------- | -------------------------------- |
| 单次试验    | 窗口内的脉冲计数      | Poisson($\lambda \Delta t$)      |
| 跨试验   | 脉冲计数变异性    | 泊松 —— Fano 因子 = 1        |
| 连续时间 | 脉冲间间隔      | 指数分布($\lambda$)           |
| 群体      | 跨神经元的放电率 | 高斯（中心极限定理 Central Limit Theorem） |

**Fano 因子** (Fano Factor)：$\text{FF} = \frac{\text{Var}(\text{spike count})}{\text{Mean}(\text{spike count})}$

- FF = 1 对应泊松分布（方差 = 均值）
- FF < 1 对应更规则的脉冲
- FF > 1 对应突发性脉冲

---

### 泊松脉冲：代码 (Poisson Spiking: Code)

```python
# Generate Poisson spike counts
spike_counts = np.random.poisson(lam=rate * duration, size=n_trials)
# Compute inter-spike intervals
isis = np.diff(spike_times)
# Coefficient of variation (irregularity measure)
cv_isi = np.std(isis) / np.mean(isis)
```

- `np.diff` 计算连续差值：`[t1-t0, t2-t1, t3-t2, …]`
- `cv_isi = 1` 对应泊松分布（指数 ISI），`< 1` 对应规则脉冲，`> 1` 对应突发性脉冲

---

### 直方图作为密度估计器 (Histogram as Density Estimator)

```python
plt.hist(data, bins=50, density=True, histtype='stepfilled')
```

设置 `density=True` 将直方图归一化，使其积分为 1 —— 可与概率密度函数 (PDF) 比较。

**与分布的联系**：泊松脉冲计数的直方图应看起来像泊松 PMF；膜电位的直方图（多个神经元）应看起来像高斯分布（中心极限定理）。

**W0D1 应用**：在 $t = t_{max}/10$ 和 $t = t_{max}$ 处绘制 $V(t)$ 的直方图，展示分布如何随时间演化 —— 随着神经元接收不同的随机输入，分布逐渐扩散。

---

## 横切主题 (Cross-Cutting Themes)

---

### 欧拉积分模式 (The Euler Integration Pattern)

这个单一模式出现在**每个模拟笔记本**中（W0D1、W0D2、W2D3、W2D4、W2D5）：

```python
for step in range(step_end):
    t = step * dt
    # 1. Compute input at this time step
    # 2. Euler update: x_new = x_old + dt * f(x_old, t)
    # 3. Apply constraints (threshold, reset, clamp)
```

| 笔记本 | $f(x,t)$                         | 约束条件                    |
| -------- | -------------------------------- | ------------------------------ |
| W0D1     | $\frac{1}{\tau}(E_L - V + RI)$   | 无（无脉冲）               |
| W0D2     | 相同                             | 脉冲重置 + 不应期钳制 |
| W2D3     | $Ax$（线性系统）             | 无                           |
| W2D4     | 与 LIF 相同 + 噪声              | 脉冲 + 不应期             |
| W2D5     | $\frac{1}{\tau}(-r + F(wr + I))$ | 速率裁剪                  |

---

### 布尔索引模式 (Boolean Indexing Pattern)

用向量化掩码替换显式循环 —— 出现在 W0D2 及后续每个笔记本中：

```python
# Instead of:
for j in range(n):
    if condition[j]:
        v[j] = value
# Use:
mask = condition          # boolean array
v[mask] = value           # vectorized assignment
```

**神经科学中常见的布尔运算**：

```python
spiked = (v >= vth)                    # threshold detection
clamped = (t - last_spike < t_ref)     # refractory check
in_range = (t_start <= t) & (t <= t_end)  # time window
```

`&` (与)、`|` (或)、`~` (非) 运算符对布尔数组进行逐元素操作。使用括号：`(a > 1) & (b < 2)`。

---

### 正规方程 (The Normal Equations)

最小二乘解 (Least-Squares Solution) 出现在 W0D3、W1D2、W1D3 中：

$$\hat{\boldsymbol{\theta}} = (X^T X)^{-1} X^T \mathbf{y}$$

**出现位置**：

| 场景                    | $X$              | $\mathbf{y}$ | $\hat{\boldsymbol{\theta}}$ |
| -------------------------- | ---------------- | ------------ | --------------------------- |
| 线性回归 (W1D2)   | 特征矩阵   | 标签       | 模型权重               |
| 线性-高斯 GLM (W1D3) | 设计矩阵    | 脉冲计数 | 刺激滤波器             |
| 群体解码        | 神经响应 | 刺激     | 解码器权重             |

**代码**：

```python
theta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
# Or more numerically stable:
theta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
```

---

### 推导正规方程 (Deriving the Normal Equations)

**目标**：找到使均方误差 (Mean Squared Error) 最小的 $\hat{\boldsymbol{\theta}}$。

**第 1 步 —— 定义目标函数**：

$$L(\boldsymbol{\theta}) = \frac{1}{N} \|\mathbf{y} - X\boldsymbol{\theta}\|^2 = \frac{1}{N}(\mathbf{y} - X\boldsymbol{\theta})^T(\mathbf{y} - X\boldsymbol{\theta})$$

- **损失** $L$ 衡量预测 $X\theta$ 与真实值 $\mathbf{y}$ 之间的差距 —— 最小化它以找到最佳拟合。
- **范数平方**：$\|\mathbf{v}\|^2 = \mathbf{v}^T\mathbf{v} = v_1^2 + \cdots + v_n^2$

**第 2 步 —— 展开**：

$$L = \frac{1}{N}\left(\mathbf{y}^T\mathbf{y} - 2\boldsymbol{\theta}^T X^T\mathbf{y} + \boldsymbol{\theta}^T X^T X \boldsymbol{\theta}\right)$$

使用转置性质 —— 见下一页。

---

**第 3 步 —— 求梯度并设为零**：

$$\frac{\partial L}{\partial \boldsymbol{\theta}} = \frac{1}{N}\left(-2X^T\mathbf{y} + 2X^T X\boldsymbol{\theta}\right) = 0$$

**第 4 步 —— 求解**：

$$X^T X\hat{\boldsymbol{\theta}} = X^T\mathbf{y} \quad \Longrightarrow \quad \hat{\boldsymbol{\theta}} = (X^T X)^{-1} X^T \mathbf{y}$$

$X^T\mathbf{y}$ 是 $\mathbf{y}$ 在 $X$ 列空间上的投影；$(X^TX)^{-1}$ 将其映射回参数空间。

---

### 推导细节 (Derivation Details)

**第 2 步使用的转置性质** (Transpose Properties)：

| 性质 | 规则                               |
| -------- | ---------------------------------- |
| 乘积  | $(AB)^T = B^TA^T$ —— 顺序反转 |
| 求和      | $(A+B)^T = A^T + B^T$              |

**第 2 步的完整展开**：

$$\|\mathbf{y} - X\theta\|^2 = (\mathbf{y} - X\theta)^T(\mathbf{y} - X\theta)$$

$$= \mathbf{y}^T\mathbf{y} - \mathbf{y}^TX\theta - (X\theta)^T\mathbf{y} + (X\theta)^TX\theta$$

$$= \mathbf{y}^T\mathbf{y} - 2\theta^TX^T\mathbf{y} + \theta^TX^TX\theta$$

交叉项合并是因为 $\mathbf{y}^TX\theta$ 是标量，所以 $\mathbf{y}^TX\theta = (\mathbf{y}^TX\theta)^T = \theta^TX^T\mathbf{y}$。

---

**第 3 步使用的矩阵微积分规则** (Matrix Calculus Rules)：

| 导数                                                                         | 结果        |
| ---------------------------------------------------------------------------------- | ------------- |
| $\frac{\partial}{\partial \theta} \theta^T X^Ty$                                   | $X^T y$       |
| $\frac{\partial}{\partial \theta} \theta^T X^TX\theta = (X\theta)^2=X^TX \theta^2$ | $2X^TX\theta$ |

**$(X^TX)^{-1}$ 何时存在？**

- 当 $X$ 的列是**线性无关**的（满列秩）时
- 实践中：使用 `np.linalg.lstsq` —— 通过 SVD 处理秩亏情况
- 等价于直接求解线性系统 $X^TX\hat{\boldsymbol{\theta}} = X^T\mathbf{y}$

---

### 总结 (Summary)

### 编程 (Programming)

- **欧拉积分**循环模式
- **NumPy 向量化**：标量 → 一维 → 二维
- **布尔索引**用于脉冲检测
- **`np.cumsum`** 用于数值积分
- **SymPy** 用于符号微积分
- **重构**：循环 → 函数 → 类

### 建模 (Modeling)

- **LIF 神经元**：膜方程 + 重置
- **不应期**：脉冲后钳制
- **传递函数**：Sigmoid、增益
- **泊松脉冲**：脉冲计数、ISI、CV
- **线性变换**：$W\mathbf{x}$、特征值

### 计算 (Calculation)

- **ODE 的欧拉离散化**
- **点积**、矩阵乘法
- **有限差分**导数
- **黎曼和**积分
- **链式法则**用于梯度
- **二项分布、泊松分布、高斯分布**

---

## 练习 (Exercises)

---

### 练习 1：实现 `cumsum`

实现一个函数 `my_cumsum(a)`，返回列表的累积和，不使用 `np.cumsum`。

**定义**：$\text{cumsum}[i] = \sum_{j=0}^{i} a[j]$

**示例**：`my_cumsum([1, 2, 3, 4])` → `[1, 3, 6, 10]`

```python
def my_cumsum(a):
    result = []
    total = 0
    for x in a:
        ...  # update total and append to result
    return result
```

**挑战**：实现二维版本 `my_cumsum2d(a, axis)`，沿 `axis=0`（行）或 `axis=1`（列）求和。

---

### 练习 2：实现 `diff`

实现一个函数 `my_diff(a)`，返回连续差值，不使用 `np.diff`。

**定义**：$\text{diff}[i] = a[i+1] - a[i]$，$\text{diff}[0]=a[0]$

**示例**：`my_diff([1, 3, 6, 10])` → `[1, 2, 3, 4]`

注意：输出长度为 `len(a)`。

```python
def my_diff(a):
    result = [a[0]]                        # diff[0] = a[0]
    for i in range(...):
        ...  # compute a[i+1] - a[i] and append
    return result
```

**应用**：`my_diff(spike_times)` 给出脉冲间间隔 (ISI)，用于在 W0D5 中计算 `cv_isi = std(ISI) / mean(ISI)`。

---

### 练习 3：滑动平均 (Running Mean)

实现 `running_mean(a)`，其中位置 $i$ 处的元素是前 $i+1$ 个元素的均值。

**定义**：$\text{result}[i] = \frac{1}{i+1}\sum_{j=0}^{i} a[j]$

**示例**：`running_mean([2, 4, 6, 8])` → `[2.0, 3.0, 4.0, 5.0]`

```python
def running_mean(a):
    result = []
    total = 0
    for i, x in enumerate(a):
        total += x
        result.append(total / (i + 1))
    return result
```

**挑战**：**不使用** `total` 重写函数 —— 只允许使用 `a` 和 `result`。

**提示**：思考 `result[-1]` 与前一个累积和的关系。你能从中恢复 `total` 吗？

**联系**：`running_mean(a) = cumsum(a) / [1, 2, 3, …, n]` —— 归一化的累积和，用于跟踪平均放电率随时间的变化。

---

### 练习 4：Softmax —— 定义与基本实现 (Softmax — Definition & Basic Implementation)

**Softmax** 将 logits 转换为概率分布：

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

**性质**：所有输出为正，和为 1，保持顺序。

**用途**：分类输出层、注意力机制 (Attention Mechanisms)、策略网络 (Policy Networks)。

**基本两遍实现**（无数值稳定性）：

```python
def softmax_basic(z):
    # Pass 1: compute exponentials
    exp_z = []
    total = 0
    for zi in z:
        exp_z.append(np.exp(zi))
        total += exp_z[-1]
    # Pass 2: normalization
    result = []
    for e in exp_z:
        result.append(e / total)
    return result
```

**问题**：对于大的 $z_i$，$e^{z_i}$ 会溢出（例如 $e^{1000} = \infty$）。标准修复：先减去最大值。

---

### 练习 4（续）：在线 Softmax (Online Softmax)

**在线 Softmax** 在单次遍历中计算 Softmax —— 适用于完整向量不可用的流式数据。

**思路**：维护一个运行最大值 $m$ 和运行和 $s$：

```python
def softmax_online(z):
    m = -float('inf')   # running max
    s = 0.0             # running sum of exp(z_i - m)
    result = []
    for zi in z:
        if zi > m:
            # new max found — how to rescale previous s?
            ...
        else:
            ...
        result.append(np.exp(zi - m))
    return [r / s for r in result]
```

**提示**：当新的最大值 $m_{\text{new}}$ 到来时，所有之前的指数都是相对于 $m_{\text{old}}$ 的。什么因子能将它们转换到新的参考框架？

---

### 练习 5：蒙特卡洛估计 (Monte Carlo Estimation)

**目标**：使用随机采样估计 $\int_{-\infty}^{\infty} \mathcal{N}(x \mid 0, 1)\,dx \approx 1$。

**蒙特卡洛原理** (Monte Carlo Principle)：采样 $x_i \sim \mathcal{U}(a,b)$，然后：

$$\int_a^b f(x)\,dx \approx \frac{b-a}{N}\sum_{i=1}^{N} f(x_i)$$

```python
def gaussian(x, mu=0, sigma=1):
    # ...
def monte_carlo_gaussian(N, a=-10, b=10):
    # ...
```

---

### 练习 5（续）：收敛性 (Convergence)

```python
for N in [100, 1000, 10000, 100000]:
    est = monte_carlo_gaussian(N)
    print(f"N={N:>6d}  estimate={est:.6f}  error={abs(est-1.0):.6f}")
```

| N       | 估计值 | 误差 |
| ------- | -------- | ----- |
| 100     | 0.987    | 0.013 |
| 1,000   | 1.002    | 0.002 |
| 10,000  | 0.999    | 0.001 |
| 100,000 | 1.000    | 0.000 |

**收敛速率**：误差 $\propto 1/\sqrt{N}$ —— 精度提高 10 倍需要 100 倍的样本。很慢！

**问题**：大多数样本落在 $\mathcal{N}(x) \approx 0$ 的区域 —— 在低密度区域浪费了计算。
