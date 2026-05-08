---
type: Note
related_to:
  - "[[Diffusion]]"
  - "[[DDIM]]"
  - "[[DPM-Solver]]"
status: Active
---

# ODE 数值方法：Euler 法与 Runge-Kutta 法

## 1. 问题形式

给定常微分方程初值问题：

$$\frac{dx}{dt} = v(x, t), \quad x(t_0) = x_0$$

目标：从 $t_0$ 出发，逐步推进到 $t_1, t_2, \dots, t_N$，得到各时刻的近似解 $x_n \approx x(t_n)$。

在扩散模型中，$v(x, t)$ 就是概率流 ODE 的漂移场（由神经网络给出）。

## 2. Euler 法（一阶方法）

### 2.1 基本思想

在当前点 $(t_n, x_n)$ 处，用切线方向 $v(x_n, t_n)$ 做线性外推：

$$x_{n+1} = x_n + h \cdot v(x_n, t_n)$$

其中 $h = t_{n+1} - t_n$ 为步长。

### 2.2 几何解释

从 $(t_n, x_n)$ 出发，沿切线方向走一步 $h$，到达下一个点。这是最简单的"沿斜率走"的策略。

### 2.3 截断误差

局部截断误差为 $O(h^2)$，全局误差为 $O(h)$，因此 Euler 法是**一阶方法**。

$$x(t_{n+1}) - x_{n+1} = O(h^2)$$

### 2.4 在扩散模型中的对应

Euler 法对应 **DDIM 的一步更新**：

$$x_{n+1} = \frac{\alpha_{n+1}}{\alpha_n} x_n + (\sigma_{n+1} - \sigma_n \frac{\alpha_{n+1}}{\alpha_n}) \epsilon_\theta(x_n, t_n)$$

步数少时误差大，生成质量差。

## 3. Runge-Kutta 法族

### 3.1 核心思想

Euler 法只用一个点的斜率，精度有限。Runge-Kutta 法通过在步长内**多个中间点**采样斜率，加权平均得到更精确的估计。

### 3.2 二阶 Runge-Kutta 法（RK2 / 中点法）

**步骤**：

1. 计算起点斜率：$k_1 = v(x_n, t_n)$
2. 用 $k_1$ 做半步 Euler，估计中点：$x_{n+1/2} = x_n + \frac{h}{2} k_1$
3. 在中点处计算斜率：$k_2 = v(x_{n+1/2}, t_n + \frac{h}{2})$
4. 用中点斜率做整步更新：$x_{n+1} = x_n + h \cdot k_2$

每步需要 **2 次函数评估**，局部截断误差为 $O(h^3)$，是**二阶方法**。

#### 变体：改进 Euler 法（Heun 法）

$$k_1 = v(x_n, t_n)$$
$$k_2 = v(x_n + h k_1, t_n + h)$$
$$x_{n+1} = x_n + \frac{h}{2}(k_1 + k_2)$$

用起点和终点斜率的平均值代替中点斜率。

### 3.3 四阶 Runge-Kutta 法（RK4）

**最经典的高阶方法**，每步需要 4 次函数评估：

$$k_1 = v(x_n, t_n)$$
$$k_2 = v(x_n + \frac{h}{2} k_1, t_n + \frac{h}{2})$$
$$k_3 = v(x_n + \frac{h}{2} k_2, t_n + \frac{h}{2})$$
$$k_4 = v(x_n + h k_3, t_n + h)$$

$$x_{n+1} = x_n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

局部截断误差为 $O(h^5)$，全局误差为 $O(h^4)$，是**四阶方法**。

#### 权重直觉

- $k_1$：起点斜率（权重 1/6）
- $k_2, k_3$：中点斜率（各权重 2/6 = 1/3）
- $k_4$：终点斜率（权重 1/6）

本质上是用 Simpson 积分公式的思想来近似积分。

### 3.4 自适应步长

利用高阶和低阶解的差异估计误差：

1. 同时计算 $x_{n+1}^{(4)}$（RK4）和 $x_{n+1}^{(5)}$（RK5，Dormand-Prince 对）
2. 估计误差：$\text{err} = \|x_{n+1}^{(5)} - x_{n+1}^{(4)}\|$
3. 若 err < 容差：接受步长，下一步可增大 $h$
4. 若 err > 容差：拒绝步长，减小 $h$ 重算

这就是 **RK45 / Dormand-Prince** 方法，MATLAB 的 `ode45` 和 SciPy 的 `solve_ivp` 默认使用。

## 4. 方法对比

| 方法 | 阶数 | 每步评估次数 | 局部误差 | 适用场景 |
|------|------|-------------|----------|----------|
| Euler | 1 | 1 | $O(h^2)$ | 简单快速，步数充足时 |
| RK2（中点法） | 2 | 2 | $O(h^3)$ | 中等精度需求 |
| RK4 | 4 | 4 | $O(h^5)$ | 高精度，步长较大时 |
| RK45（自适应） | 4/5 | 6 | $O(h^5)$ | 通用，自动调步长 |

## 5. 在扩散采样中的应用

### 5.1 为什么通用 RK 方法不是最优？

扩散模型的概率流 ODE 具有**半线性结构**：

$$\frac{dx}{dt} = f(t)x + g(t)\epsilon_\theta(x, t)$$

通用 RK 方法将整个右端项视为黑盒，对线性部分 $f(t)x$ 也做数值近似，浪费了精度。

DPM-Solver 的做法是：**精确求解线性部分 + 高阶近似非线性部分**，等价于对半线性 ODE 的定制化 RK 方法。

### 5.2 DPM-Solver 与 RK 的对应

| DPM-Solver | 等价的 RK 思想 | 额外评估次数 |
|------------|---------------|-------------|
| DPM-Solver-1 | Euler（对非线性部分） | 0 |
| DPM-Solver-2 | 中点法（对非线性部分） | 1 |
| DPM-Solver-3 | 三阶方法（对非线性部分） | 2 |

区别在于：DPM-Solver 的"中点"是在**对数信噪比空间**中选取的，且线性部分始终精确处理。

### 5.3 实践建议

- **10-20 步**：DPM-Solver-2 是最佳选择（精度/速度平衡）
- **10 步以内**：DPM-Solver-3 或 DPM-Solver++
- **50 步以上**：普通 Euler / DDIM 已足够
- **需要自适应步长**：DPM-Solver 支持自适应，或用通用 RK45

## 6. 参考

- Butcher, J. C. (2016). Numerical Methods for Ordinary Differential Equations. Wiley.
- Lu, C., et al. (2022). DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps. NeurIPS 2022.
