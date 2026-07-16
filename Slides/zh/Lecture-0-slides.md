# 第0讲：机器学习的数学基础

微积分 · 线性代数 · 概率论

---

## 概览

三个主要组成部分：

| 科目 | 用途 |
| ------------------ | ---------------------------------------------------------------- |
| **微积分** (Calculus) | 优化——通过梯度下降更新模型参数 |
| **线性代数** (Linear Algebra) | 表示——以向量/矩阵的形式存储和计算数据 |
| **概率论** (Probability) | 建模——用概率分布描述不确定性 |

快速浏览这三个**组成部分**，尽快进入机器学习。

---

**微积分**

---

## 第1部分：微积分

### 极限 (Limit)：定义

**极限**描述了当 $x$ 无限接近某一点 $a$ 时，$f(x)$ 所趋近的值：

$$\lim_{x \to a} f(x) = L$$

形式化定义：对于任意 $\varepsilon > 0$，存在 $\delta > 0$ 使得

$$0 < |x - a| < \delta \implies |f(x) - L| < \varepsilon$$

> [!tip] 直觉
> 通过选择足够接近 $a$ 的 $x$（但 $x \neq a$），我们可以使 $f(x)$ 任意接近 $L$。

---

### 极限 (Limit)：示例

**直接代入法**（当函数连续时）：

$$\lim_{x \to 2} (x^2 + 1) = 2^2 + 1 = 5$$

**不定式** $\frac{0}{0}$——因式分解并约分：

$$\lim_{x \to 1} \frac{x^2 - 1}{x - 1} = \lim_{x \to 1} \frac{(x-1)(x+1)}{x-1} = \lim_{x \to 1}(x+1) = 2$$

**重要极限**：

$$\lim_{x \to 0} \frac{\sin x}{x} = 1$$

$$\lim_{x \to 0} \frac{e^x - 1}{x} = 1$$

$$\lim_{x \to \infty}\left(1 + \frac{1}{x}\right)^x = e$$

---

### 极限 (Limit)：性质

**1. 线性性质**：

$$\lim_{x \to a} [f(x) + g(x)] = \lim_{x \to a} f(x) + \lim_{x \to a} g(x)$$

$$\lim_{x \to a} [c \cdot f(x)] = c \cdot \lim_{x \to a} f(x)$$

**2. 乘积与商**：

$$\lim_{x \to a} [f(x) \cdot g(x)] = \lim_{x \to a} f(x) \cdot \lim_{x \to a} g(x)$$

$$\lim_{x \to a} \frac{f(x)}{g(x)} = \frac{\displaystyle\lim_{x \to a} f(x)}{\displaystyle\lim_{x \to a} g(x)} \quad \text{(if } \lim_{x \to a} g(x) \neq 0\text{)}$$

**3. 夹逼定理 (Squeeze Theorem)**：如果在 $a$ 附近 $g(x) \leq f(x) \leq h(x)$，且 $\lim_{x \to a} g(x) = \lim_{x \to a} h(x) = L$，则 $\lim_{x \to a} f(x) = L$。

---

### 连续性 (Continuity)

函数 $f$ 在点 $a$ 处**连续**，如果：

$$\boxed{\lim_{x \to a} f(x) = f(a)}$$

这需要**三个条件**：

1. $f(a)$ 有定义（该点存在）
2. $\lim_{x \to a} f(x)$ 存在（左极限和右极限一致）
3. $\lim_{x \to a} f(x) = f(a)$（极限等于函数值）
> [!tip] 直觉
> 函数图像在 $a$ 处**没有断裂、跳跃或空洞**。你可以不用抬笔就画过 $a$。

---

### 间断点的类型 (Types of Discontinuity)

**可去间断点 (Removable)**

<svg width="160" height="180" viewBox="0 0 160 180">

  <line x1="20" y1="160" x2="140" y2="160" stroke="#666" stroke-width="1"/>

  <line x1="20" y1="160" x2="20" y2="20" stroke="#666" stroke-width="1"/>

  <circle cx="80" cy="90" r="5" fill="none" stroke="#4fc3f7" stroke-width="2.5"/>

  <circle cx="80" cy="120" r="4" fill="#4fc3f7"/>

</svg>

$\lim_{x \to a} f(x) = L$ 存在，但 $f(a) \neq L$ 或 $f(a)$ 未定义

**跳跃间断点 (Jump)**

<svg width="160" height="180" viewBox="0 0 160 180">

  <line x1="20" y1="160" x2="140" y2="160" stroke="#666" stroke-width="1"/>

  <line x1="20" y1="160" x2="20" y2="20" stroke="#666" stroke-width="1"/>

  <line x1="30" y1="130" x2="80" y2="130" stroke="#ff8a65" stroke-width="2.5"/>

  <circle cx="80" cy="130" r="4" fill="#ff8a65"/>

  <line x1="80" y1="70" x2="130" y2="70" stroke="#ff8a65" stroke-width="2.5"/>

  <circle cx="80" cy="70" r="4" fill="none" stroke="#ff8a65" stroke-width="2.5"/>

  <line x1="80" y1="70" x2="80" y2="130" stroke="#ff8a65" stroke-width="1" stroke-dasharray="4"/>

</svg>

$\lim_{x \to a^-} f(x) \neq \lim_{x \to a^+} f(x)$

**无穷间断点 / 本质间断点 (Infinite / Essential)**

<svg width="160" height="180" viewBox="0 0 160 180">

  <line x1="20" y1="160" x2="140" y2="160" stroke="#666" stroke-width="1"/>

  <line x1="20" y1="160" x2="20" y2="20" stroke="#666" stroke-width="1"/>

  <line x1="80" y1="20" x2="80" y2="160" stroke="#ce93d8" stroke-width="1" stroke-dasharray="4"/>

</svg>

$\lim_{x \to a} f(x) = \pm\infty$ 或不存在

---

### 连续性与可微性 (Continuity & Differentiability)

函数 $f$ 在 $a$ 处**可微 (differentiable)**，如果导数存在：

$$f'(a) = \lim_{\Delta x \to 0} \frac{f(a + \Delta x) - f(a)}{\Delta x} \quad \text{存在}$$

**关键关系**：

$$\text{在 } a \text{ 处可微} \implies \text{在 } a \text{ 处连续}$$

**但反之不成立！** 函数可以在 $a$ 处连续但不可微。

**为什么？** 如果 $f'(a)$ 存在，则：

$$\lim_{\Delta x \to 0}[f(a+\Delta x) - f(a)] = \lim_{\Delta x \to 0} \frac{f(a+\Delta x) - f(a)}{\Delta x} \cdot \Delta x = f'(a) \cdot 0 = 0$$

$$\implies \lim_{\Delta x \to 0} f(a+\Delta x) = f(a) \implies \text{连续}$$

---

### 反例：连续但不可微

<svg width="280" height="260" viewBox="0 0 280 260">
  <line x1="20" y1="220" x2="260" y2="220" stroke="#666" stroke-width="1"/>
  <line x1="140" y1="220" x2="140" y2="30" stroke="#666" stroke-width="1"/>
  <!-- |x| -->
  <circle cx="140" cy="220" r="5" fill="#4fc3f7"/>
</svg>
**$f(x) = |x|$** 在 $x = 0$ 处：
- **连续**：$\lim_{x \to 0}|x| = 0 = f(0)$ ✓
- **不可微**：左导数和右导数不一致：
$$f'_-(0) = \lim_{\Delta x \to 0^-}\frac{|\Delta x|}{\Delta x} = -1$$
$$f'_+(0) = \lim_{\Delta x \to 0^+}\frac{|\Delta x|}{\Delta x} = +1$$
尖锐的**角点**意味着不存在唯一的切线。

---

### 总结：蕴含链

**可微 (Differentiable)**

$f'(a)$ 存在

$\Longrightarrow$

**连续 (Continuous)**

$\lim_{x\to a}f(x)=f(a)$

$\Longrightarrow$

**极限存在 (Limit Exists)**

$\lim_{x\to a}f(x)=L$

**逆命题为假**（每个箭头都是单向的）：

| 逆命题 | 反例 |
| ------------------------------------------- | ---------------------------------------------------------------- |
| 连续 $\not\Rightarrow$ 可微 | $f(x)=\|x\|$ 在 $x=0$ 处（角点） |
| 极限存在 $\not\Rightarrow$ 连续 | $f(x)=\begin{cases}x^2 & x\neq 0 \\ 1 & x=0\end{cases}$ 在 $x=0$ 处 |

---

### 可微性：直观理解

**可微 (Differentiable)**

<svg width="160" height="180" viewBox="0 0 160 180">

  <line x1="20" y1="160" x2="140" y2="160" stroke="#666" stroke-width="1"/>

  <line x1="20" y1="160" x2="20" y2="20" stroke="#666" stroke-width="1"/>

  <line x1="55" y1="85" x2="105" y2="75" stroke="#ff5252" stroke-width="1.5" stroke-dasharray="4"/>

  <circle cx="80" cy="80" r="4" fill="#4caf50"/>

</svg>

光滑曲线——处处存在切线

**角点 (Corner)（不可微）**

<svg width="160" height="180" viewBox="0 0 160 180">

  <line x1="20" y1="160" x2="140" y2="160" stroke="#666" stroke-width="1"/>

  <line x1="20" y1="160" x2="20" y2="20" stroke="#666" stroke-width="1"/>

  <circle cx="80" cy="60" r="4" fill="#ff9800"/>

  <line x1="50" y1="100" x2="80" y2="60" stroke="#ff5252" stroke-width="1.5" stroke-dasharray="4"/>

  <line x1="80" y1="60" x2="110" y2="100" stroke="#ff5252" stroke-width="1.5" stroke-dasharray="4"/>

</svg>

角点——两个切线方向，没有唯一切线

**尖点 (Cusp)（不可微）**

<svg width="160" height="180" viewBox="0 0 160 180">

  <line x1="20" y1="160" x2="140" y2="160" stroke="#666" stroke-width="1"/>

  <line x1="20" y1="160" x2="20" y2="20" stroke="#666" stroke-width="1"/>

  <circle cx="80" cy="60" r="4" fill="#ce93d8"/>

  <line x1="80" y1="60" x2="80" y2="160" stroke="#ff5252" stroke-width="1.5" stroke-dasharray="4"/>

</svg>

尖点——切线是垂直的（斜率为 $\infty$）

---

### 极限 → 导数 (Limit → Derivative)

导数**定义**为一个极限：

$$f'(x) = \lim_{\Delta x \to 0} \frac{f(x + \Delta x) - f(x)}{\Delta x}$$

**联系**：没有极限，我们就无法严格定义导数。极限捕捉了"瞬时变化率"的概念——当割线上的两点合并为一点时，切线的斜率。

**示例**：从第一性原理推导 $f(x) = x^2$ 的导数：

$$f'(x) = \lim_{\Delta x \to 0} \frac{(x+\Delta x)^2 - x^2}{\Delta x} = \lim_{\Delta x \to 0} \frac{2x\Delta x + (\Delta x)^2}{\Delta x} = \lim_{\Delta x \to 0}(2x + \Delta x) = 2x$$

---

### 导数定义 (Derivative-Definition)

对于函数 $y=f(x)$，在点 $x$ 处的导数为：

$$f'(x) = \lim_{\Delta x \to 0} \frac{f(x + \Delta x) - f(x)}{\Delta x} = \frac{dy}{dx}$$

**示例**：$f(x) = x^3/3 - x$

$$f'(x) = \frac{d}{dx}\left(\frac{x^3}{3} - x\right) = x^2 - 1$$

$\frac{df}{dx}$ 的符号表示当 $x$ 增加时 $f$ 是否增加。

问题：我们有一些 $x_0$ 和 $x_1 = x_0 - \beta f'(x_0)$（$\beta$ 足够小），那么 $f(x_0) \mathbf{O} f(x_1)$？

---

### 直觉 (Intuition)

$$x_1 = x_0 - \beta f'(x_0)$$

- 如果 $f'(x_0) > 0$：$f$ 递增，$x_1 < x_0$（向左移动）$\Rightarrow$ $f$ 减小
- 如果 $f'(x_0) < 0$：$f$ 递减，$x_1 > x_0$（向右移动）$\Rightarrow$ $f$ 减小
- 无论如何 $x_1$ 都朝着减小 $f$ 的方向移动

$$\boxed{f(x_0) \geq f(x_1)}$$

### 推导（泰勒展开 (Taylor Expansion)）

$$x_1 = x_0 - \beta f'(x_0)$$

$$f(x_1) \approx f(x_0) + f'(x_0)(x_1 - x_0)$$

代入 $x_1 - x_0 = -\beta f'(x_0)$：

$$f(x_1) \approx f(x_0) - \beta\left(f'(x_0)\right)^2$$

由于 $\beta > 0$ 且 $\left(f'(x_0)\right)^2 \geq 0$：

$$\boxed{f(x_1) \leq f(x_0)}$$

**泰勒展开 (Taylor Expansion)**：对于光滑函数 $f$ 在点 $a$ 附近：

$$f(x) = f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \cdots+\frac{f^{(n)}(a)}{n!}(x-a)^n+o((x-a)^n)$$

对于 $a=0$，$f(x) = f(0) + f'(0)(x) + \frac{f''(0)}{2!}(x)^2 + \cdots+\frac{f^{(n)}(0)}{n!}(x)^n+o((x)^n)$

---

### 图形解释 (Graphical Interpretation)

<DerivativeChart :tangent-x="1" />
在图表中：
- **蓝色曲线**：$f(x) = \frac{x^3}{3} - x$
- **红色虚线**：在 $x=1$ 处的切线
- 切线斜率 = $f'(1) = 1^2 - 1 = 0$
**关键点**：极值点 $\rightarrow f'(x)=0$。
**优化的基础**
当 f'(x) =0 时，f(x) 会怎样？

---

### 泰勒展开可视化 (Taylor Expansion Visualization)

<TaylorExpansion type="poly6" />
<TaylorExpansion type="ln" />
$$f(x) = f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \cdots+\frac{f^{(n)}(a)}{n!}(x-a)^n+o((x-a)^n)$$

---

### 泰勒展开可视化-2

<TaylorExpansion type="exp" />
<TaylorExpansion type="sin" />
$$f(x) = f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \cdots+\frac{f^{(n)}(a)}{n!}(x-a)^n+o((x-a)^n)$$

---

### 问题 (Questions)

**Q1**：对于 $f(x) = x^6 - 3x^4 + 2x^3 + x$ 在 $a=0$ 处：为什么 $T_2(x)$ 等于 $T_1(x)$，$T_5(x)$ 等于 $T_4(x)$？

**Q2**：对于 $\sin x$ 在 $a=0$ 处：为什么每个偶数阶展开都等于前一个奇数阶展开？（即 $T_{2k} = T_{2k-1}$）

提示：看看哪些系数为零以及为什么。

**A1**：$f(x) = x^6 - 3x^4 + 2x^3 + x$ **没有** $x^2$ 或 $x^5$ 项。在 $a=0$ 处，泰勒系数 $c_2 = \frac{f''(0)}{2!} = 0$ 且 $c_5 = \frac{f^{(5)}(0)}{5!} = 0$，所以添加这些项不会改变任何东西。

**A2**：$\sin x$ 是一个**奇函数** ($\sin(-x) = -\sin x$)，所以它在 $a=0$ 处的泰勒级数只包含**奇数**次幂：$\sin x = x - \frac{x^3}{3!} + \frac{x^5}{5!} - \cdots$。所有偶数阶系数都为零，因此 $T_{2k} = T_{2k-1}$。

---

### 常用求导法则 (Common Differentiation Rules)

| 函数 $y$ | 导数 $\dfrac{dy}{dx}$ | 微分 $dy$ | 法则 |
| ------------- | ------------------------------------- | ------------------------------- | ----------------- |
| $y = cf(x)$ | $c\dfrac{df}{dx}$ | $dy=c\, df=cf'(x)dx$ | 常数倍法则 |
| $y = f + g$ | $\dfrac{df}{dx} + \dfrac{dg}{dx}$ | $dy=df + dg=(f'(x)+g'(x))dx$ | 加法法则 |
| $y = f(g(x))$ | $\dfrac{df}{dg} \cdot \dfrac{dg}{dx}$ | $dy=f'(g)\, dg=f'(g(x))g'(x)dx$ | **链式法则 (Chain rule)** |

**链式法则 (Chain rule)** 对于神经网络中的反向传播至关重要。它计算复合函数的导数：逐层反向传播梯度。

$$\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial z} \cdot \frac{\partial z}{\partial w_1}$$

---

### 梯度下降可视化 (Gradient Descent Visualization)

<GradientDescent />
**两个起始点**：
- **红色路径**：从 $x_0 = -0.5$ 开始
- **绿色路径**：从 $x_0 = 4.5$ 开始
- 两者都收敛到 $x = 2$
**目标**：找到 $f(x) = (x-2)^2 + 1$ 的最小值
**回顾**：我们证明了在每一步 $x_1 = x_0 - \eta f'(x_0)$ 中 $f(x_0) \geq f(x_1)$。
这保证了函数值在每次迭代中**永远不会增加**。
我们称之为**梯度下降 (Gradient Descent)**。
**更新规则**：$x_{n+1} = x_n - \eta \cdot f'(x_n)$，其中 $\eta = 0.3$
无论从哪一侧开始，梯度下降总是朝着最小值下降。

---

### 学习率：太小与太大 (Learning Rate: Too Small vs Too Large)

<GradientDescentLR />
**函数**：
$$f(x) = (x - 2)^2 + 1$$
**梯度**：
$$f'(x) = 2(x - 2)$$
**更新规则**：
$$x_{n+1} = x_n - \eta \cdot f'(x_n)$$
**尝试用滑块调整 $\eta$** 并观察：
**权衡**：小的 $\eta$ 安全但慢。大的 $\eta$ 快但有风险。在机器学习训练中选择合适的学习率至关重要。

---

### 多个局部最小值 (Multiple Local Minima)

<GradientDescentMulti />
**函数**：
$$f(x) = x^4 - 4x^3 + 2x^2 + 4x$$
**梯度**：
$$f'(x) = 4x^3 - 12x^2 + 4x + 4$$
**临界点 (Critical points)**（$f'(x) = 0$ 的位置）：

| 点 | 类型 |
| ----------------- | --------- |
| $x \approx -0.41$ | 局部最小值 (Local min) |
| $x = 1$ | 局部最大值 (Local max) |
| $x \approx 2.41$ | 局部最小值 (Local min) |

---

### 偏导数 (Partial Derivative)

对于多变量函数 $f(x_1, x_2, \ldots, x_n)$，偏导数是当**只有一个变量**变化时的变化率：

$$\frac{\partial f}{\partial x_i} = \lim_{\Delta x_i \to 0} \frac{f(x_1, \ldots, x_i + \Delta x_i, \ldots, x_n) - f(x_1, \ldots, x_n)}{\Delta x_i}$$

示例：

$$
y=f(x,y,z),\frac{\partial f}{\partial x}=\lim_{\Delta x_i}\frac{f(x+\Delta x_i,y,z)}{\Delta x_i}
$$

**梯度 (Gradient)** 是所有偏导数组成的向量：

$$\nabla f = \left[\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n}\right]$$

梯度指向**最速上升**的方向。在机器学习中，我们沿着**相反**方向更新参数以最小化损失：

$$\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla L(\mathbf{w})$$

其中 $\eta$ 是学习率。

---

### 二阶导数 (Second-Order Derivative)

**二阶导数**是导数的导数，衡量变化率本身如何变化：

$$f''(x) = \frac{d^2 f}{dx^2} = \frac{d}{dx}\left(\frac{df}{dx}\right) = \lim_{\Delta x \to 0} \frac{f'(x + \Delta x) - f'(x)}{\Delta x}$$

**几何意义**：$f''(x)$ 描述了函数的**凹凸性 (concavity)**。

- $f''(x) > 0$：函数**上凹 (concave up)**（向上弯曲，像杯子）→ 临界点处为局部最小值
- $f''(x) < 0$：函数**下凹 (concave down)**（向下弯曲，像帽子）→ 临界点处为局部最大值
- $f''(x) = 0$：可能的**拐点 (inflection point)**（凹凸性改变）
**示例**：$f(x) = \frac{x^3}{3} - x$
$f'(x) = x^2 - 1, \quad f''(x) = 2x$
在 $x = 1$ 处：$f'(1) = 0$（临界点），$f''(1) = 2 > 0$ → **局部最小值**
在 $x = -1$ 处：$f'(-1) = 0$（临界点），$f''(-1) = -2 < 0$ → **局部最大值**

---

### 二阶导数与优化 (Second-Order Derivative & Optimization)

在点 $a$ 附近的**二阶泰勒展开 (Second-Order Taylor Expansion)**：

$$f(x) \approx f(a) + f'(a)(x-a) + \frac{f''(a)}{2}(x-a)^2$$

在临界点 $f'(a) = 0$ 处：

$$f(x) \approx f(a) + \frac{f''(a)}{2}(x-a)^2$$

- 如果 $f''(a) > 0$：$f(x) \geq f(a)$ → **局部最小值**
- 如果 $f''(a) < 0$：$f(x) \leq f(a)$ → **局部最大值**
**在机器学习中**：二阶导数告诉我们损失曲面的**曲率 (curvature)**。
- 大的 $f''$ → 陡峭的曲率 → 小步长更安全
- 小的 $f''$ → 平坦区域 → 可以采取更大的步长
这启发了**二阶优化方法 (second-order optimization methods)**，如牛顿法 (Newton's method)：

$$x_{n+1} = x_n - \frac{f'(x_n)}{f''(x_n)}$$

---

### 二阶偏导数 (Second-Order Partial Derivative)

对于多变量函数 $f(x_1, x_2, \ldots, x_n)$，我们可以对同一个变量求两次偏导数：

$$\frac{\partial^2 f}{\partial x_i^2} = \frac{\partial}{\partial x_i}\left(\frac{\partial f}{\partial x_i}\right)$$

**示例**：$f(x, y) = x^2 y + 3xy^2$

$$\frac{\partial f}{\partial x} = 2xy + 3y^2, \quad \frac{\partial^2 f}{\partial x^2} = 2y$$

$$\frac{\partial f}{\partial y} = x^2 + 6xy, \quad \frac{\partial^2 f}{\partial y^2} = 6x$$

**解释**：$\frac{\partial^2 f}{\partial x_i^2}$ 衡量 $f$ 沿 $x_i$ 方向的凹凸性，保持所有其他变量不变。

---

### 二阶混合偏导数 (Second-Order Mixed Partial Derivative)

**混合偏导数 (mixed partial derivative)** 涉及对**不同变量**求偏导数：

$$\frac{\partial^2 f}{\partial x_j \partial x_i} = \frac{\partial}{\partial x_j}\left(\frac{\partial f}{\partial x_i}\right)$$

**示例**：$f(x, y) = x^2 y + 3xy^2$

$$\frac{\partial f}{\partial x} = 2xy + 3y^2 \quad \Rightarrow \quad \frac{\partial^2 f}{\partial y \partial x} = \frac{\partial}{\partial y}(2xy + 3y^2) = 2x + 6y$$

$$\frac{\partial f}{\partial y} = x^2 + 6xy \quad \Rightarrow \quad \frac{\partial^2 f}{\partial x \partial y} = \frac{\partial}{\partial x}(x^2 + 6xy) = 2x + 6y$$

**克莱罗定理（混合偏导数的对称性）(Clairaut's Theorem)**：

如果 $f$ 具有连续的二阶偏导数，则：

$$\boxed{\frac{\partial^2 f}{\partial x_j \partial x_i} = \frac{\partial^2 f}{\partial x_i \partial x_j}}$$

求导的顺序不重要！

---

### 海森矩阵 (Hessian Matrix)

**海森矩阵 (Hessian matrix)** 收集了多变量函数的所有二阶偏导数：

$$H(f) = \begin{bmatrix} \dfrac{\partial^2 f}{\partial x_1^2} & \dfrac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \dfrac{\partial^2 f}{\partial x_1 \partial x_n} \\[6pt] \dfrac{\partial^2 f}{\partial x_2 \partial x_1} & \dfrac{\partial^2 f}{\partial x_2^2} & \cdots & \dfrac{\partial^2 f}{\partial x_2 \partial x_n} \\[6pt] \vdots & \vdots & \ddots & \vdots \\[6pt] \dfrac{\partial^2 f}{\partial x_n \partial x_1} & \dfrac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \dfrac{\partial^2 f}{\partial x_n^2} \end{bmatrix}$$

**示例**：$f(x, y) = x^2 y + 3xy^2$

$$H = \begin{bmatrix} \dfrac{\partial^2 f}{\partial x^2} & \dfrac{\partial^2 f}{\partial x \partial y} \\[6pt] \dfrac{\partial^2 f}{\partial y \partial x} & \dfrac{\partial^2 f}{\partial y^2} \end{bmatrix} = \begin{bmatrix} 2y & 2x + 6y \\ 2x + 6y & 6x \end{bmatrix}$$

---

### 海森矩阵与优化 (Hessian & Optimization)

在点 $\mathbf{a}$ 附近的**多变量二阶泰勒展开 (Multivariable second-order Taylor expansion)**：

$$f(\mathbf{x}) \approx f(\mathbf{a}) + \nabla f(\mathbf{a})^T (\mathbf{x} - \mathbf{a}) + \frac{1}{2}(\mathbf{x} - \mathbf{a})^T H(\mathbf{a}) (\mathbf{x} - \mathbf{a})$$

在临界点 $\nabla f(\mathbf{a}) = \mathbf{0}$ 处：

$$f(\mathbf{x}) \approx f(\mathbf{a}) + \frac{1}{2}(\mathbf{x} - \mathbf{a})^T H(\mathbf{a}) (\mathbf{x} - \mathbf{a})$$

利用海森矩阵**分类临界点**：

- $H$ **正定 (positive definite)** → 所有特征值 $> 0$ → **局部最小值**
- $H$ **负定 (negative definite)** → 所有特征值 $< 0$ → **局部最大值**
- $H$ **不定 (indefinite)** → 符号混合 → **鞍点 (saddle point)**
**在机器学习中**：海森矩阵为优化算法提供信息。牛顿法使用 $H^{-1}$ 来适应每个方向的步长，在光滑损失曲面上比梯度下降收敛更快。

$$\mathbf{w} \leftarrow \mathbf{w} - H^{-1} \nabla L(\mathbf{w})$$

---

### 不定积分 (Indefinite Integral)

**不定积分 (indefinite integral)** 是反导数——所有导数为 $f(x)$ 的函数族：

$$\int f(x)\, dx = F(x) + C$$

其中 $F'(x) = f(x)$，$C$ 是任意常数（因为 $(F(x)+C)' = f(x)$）。

**常用积分公式**：

| $f(x)$ | $\int f(x)\, dx$ |
| ------------------- | -------------------------- |
| $x^n$ ($n \neq -1$) | $\dfrac{x^{n+1}}{n+1} + C$ |
| $\dfrac{1}{x}$ | $\ln\|x\| + C$ |
| $e^x$ | $e^x + C$ |

---

### 不定积分的性质 (Properties of Indefinite Integrals)

**1. 线性性质（加法法则）**：

$$\int \left(f(x) + g(x)\right) dx = \int f(x)\, dx + \int g(x)\, dx$$

**2. 常数倍法则**：

$$\int C f(x)\, dx = C \int f(x)\, dx$$

等价形式：$\int C f(x)\, dx = \int f(x)\, d(Cx) = C \int f(x)\, dx$

**示例**：$\int (3x^2 + 2x)\, dx = 3\int x^2\, dx + 2\int x\, dx = 3 \cdot \frac{x^3}{3} + 2 \cdot \frac{x^2}{2} + C = x^3 + x^2 + C$
**练习**：$\int x + e^x \, dx,\int x + \frac{1}{x}\,dx$

---

### 凑微分法（Integration by Substitution）

$$\int f'(x)dx=\int df(x)=f(x)+C$$

若 $\int f(x)\, dx = F(x) + C$，则：

$$\boxed{\int f(g(x))\, g'(x)\, dx =\int f(g(x))dg(x)=\int f(u)du=F(u)+C= F(g(x)) + C}$$

**例**：$\int 2x \cos(x^2)\, dx$

令 $u = x^2$，则 $du = 2x\, dx$。

$$\int 2x \cos(x^2)\, dx = \int \cos u\, du = \sin u + C = \sin(x^2) + C$$

**例**：$\int e^{3x}\, dx$

令 $u = 3x$，则 $du = 3\, dx$，故 $dx = \frac{1}{3} du \int e^{3x}\, dx = \frac{1}{3}\int e^u\, du = \frac{1}{3}e^{3x} + C$

**练习**：$\int e^{-x} \,dx$

---

### 分部积分法（Integration by Parts）

由乘积法则 $(uv)' = u'v + uv'$，可得：

$$\boxed{\int u\, dv = uv - \int v\, du} or \boxed{\int u\, dv + \int v\, du=uv}$$

**例**：$\int x e^x\, dx$

令 $u = x$，$dv = e^x dx$。则 $du = dx$，$v = e^x$。

$$\int x e^x\, dx = x e^x - \int e^x\, dx = x e^x - e^x + C = (x-1)e^x + C$$

**练习**：$\int \ln\,x \, dx,\int xe^{-x}\,dx$

---

### 定积分（Integral）

**定积分**（Integral）是导数的逆运算，表示在 $[a, b]$ 区间上的累积量：

$$\int_a^b f(x)\, dx = F(a)-F(b)=\left[F(x)\right]^{b}_{a}=\lim_{n \to \infty} \sum_{i=1}^{n} f(x_i) \Delta x$$

**精确计算**——使用微积分基本定理（Fundamental Theorem of Calculus）：

$$\int_{0.5}^{2} \left(\frac{x^2}{2} + 0.3\right) dx = \left[\frac{x^3}{6} + 0.3x\right]_{0.5}^{2}$$

$$= \underbrace{\left(\frac{2^3}{6} + 0.3 \times 2\right)}_{F(2) = \frac{29}{15} \approx 1.933} - \underbrace{\left(\frac{0.5^3}{6} + 0.3 \times 0.5\right)}_{F(0.5) = \frac{41}{240} \approx 0.171} = \frac{141}{80} \approx 1.7625$$

<RiemannChart :a="0.5" :b="2" :n="n" />

---

### 什么是微分方程？

**微分方程**（Differential Equation）是将函数与其导数相关联的方程。

**简单例子**：

$$\frac{dy}{dt} = ky$$

这表示：_$y$ 的变化率与 $y$ 本身成正比。_

**解**是函数，而非数值：

$$y(t) = Ce^{kt}$$

其中 $C$ 是由初始条件确定的任意常数。

---

**日常例子**：

- **种群增长**：$\frac{dP}{dt} = rP$ —— 增长率与当前种群规模成正比
- **放射性衰变**：$\frac{dN}{dt} = -\lambda N$ —— 衰变速率与剩余原子数成正比
- **冷却**：$\frac{dT}{dt} = -k(T - T_{\text{env}})$ —— 牛顿冷却定律
**核心思想**：我们不直接指定函数，而是指定一个**关于它如何变化的规则**。求解常微分方程（ODE）意味着找到满足该规则的函数。

---

### 常微分方程（ODE）

**常微分方程**（Ordinary Differential Equation, ODE）将函数 $y(t)$ 与其导数相关联。**阶数**（Order）是出现的最高导数阶数。

**例子**：

| 常微分方程              | 阶数 | 类型                        |
| -------------------- | ----- | ----------------------- |
| $y' + 2y = 0$        | 1阶   | 线性、齐次（Linear, homogeneous）     |
| $y' + 2y = e^t$      | 1阶   | 线性、非齐次（Linear, non-homogeneous） |
| $y'' + 3y' + 2y = 0$ | 2阶   | 线性、齐次（Linear, homogeneous）     |

**为什么ODE在机器学习中很重要**：许多动力系统（如神经常微分方程 Neural ODE、扩散模型 Diffusion Models）被表述为ODE。训练它们需要求解ODE并对解进行微分。

---

### 一阶线性ODE

**一阶线性ODE**（First-Order Linear ODE）具有如下形式：$y' + p(t)\,y = g(t)$

**求解方法**——积分因子法（Integrating Factor）：

1. 计算 $\mu(t) = e^{\int p(t)\,dt}$
2. 两边同乘：$\mu(t)\,y' + \mu(t)\,p(t)\,y = \mu(t)\,g(t)$
3. 左边变为 $\frac{d}{dt}[\mu(t)\,y]$
4. 积分：$\mu(t)\,y = \int \mu(t)\,g(t)\,dt + C$
**例**：$y' + 2y = e^t$

$$\mu(t) = e^{\int 2\,dt} = e^{2t}$$

$$\frac{d}{dt}\!\left[e^{2t}y\right] = e^{2t} \cdot e^t = e^{3t}$$

$$e^{2t}y = \frac{1}{3}e^{3t} + C \quad\Rightarrow\quad \boxed{y = \frac{1}{3}e^{t} + Ce^{-2t}}$$

---

### 二阶线性ODE

**二阶线性ODE**（Second-Order Linear ODE）具有如下形式：

$y'' + a\,y' + b\,y = g(t)$

**齐次情形**（$g(t) = 0$）：求解**特征方程**（Characteristic Equation）

$r^2 + ar + b = 0$

| 根                            | 通解（General Solution）                                      |
| -------------------------------- | ----------------------------------------------------- |
| 两个不等实根 $r_1 \neq r_2$          | $y = C_1 e^{r_1 t} + C_2 e^{r_2 t}$                   |
| 重根 $r_1 = r_2 = r$         | $y = (C_1 + C_2 t)\,e^{rt}$                           |
| 复根 $r = \alpha \pm \beta i$ | $y = e^{\alpha t}(C_1 \cos\beta t + C_2 \sin\beta t)$ |

**例**：$y'' + 3y' + 2y = 0$

$$r^2 + 3r + 2 = 0 \;\Rightarrow\; (r+1)(r+2) = 0 \;\Rightarrow\; r_1 = -1,\; r_2 = -2$$

$$\boxed{y = C_1 e^{-t} + C_2 e^{-2t}}$$

---

**线性代数**

---

## 第二部分：线性代数

### 基本数据结构

**标量**（Scalar）：单个数值

$$x = 5, \quad x \in \mathbb{R}$$

**向量**（Vector）：有序的数值列表

$$\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ v_3 \end{bmatrix}, \quad \mathbf{v} \in \mathbb{R}^3$$

**矩阵**（Matrix）：二维数组

$$\mathbf{A} = \begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \\ a_{31} & a_{32} \end{bmatrix}, \quad \mathbf{A} \in \mathbb{R}^{3 \times 2}$$

**张量**（Tensor）：推广到 $n$ 维

- 0阶张量 = 标量
- 1阶张量 = 向量
- 2阶张量 = 矩阵
- 3阶及以上 = 高阶张量
**维度记法**：$\mathbb{R}^{m \times n}$ 表示具有 $m$ 行 $n$ 列的实数矩阵。
在机器学习框架（PyTorch、TensorFlow）中，所有数据都以张量的形式存储和计算。

---

### 张量维度：可视化示例

**0维——标量**：单个数值

$$x = 5$$

**1维——向量**：数值列表

$$\mathbf{v} = [3, 1, 4, 1, 5]$$

**2维——矩阵**：灰度图像（高度 × 宽度）

<GrayscaleTensor :rows="8" :cols="8" :cell-size="20" />

形状：$8 \times 8$（H × W）

**3维张量**：彩色图像（通道数 × 高度 × 宽度）

形状：$3 \times 8 \times 8$（C × H × W）

**4维张量**：彩色图像的批次

$$\text{Shape: } N \times C \times H \times W $$

- $N$：批次大小（Batch Size，图像数量）
- $H \times W$：空间维度
- $C$：通道数（Channels，RGB为3，RGBA为4）
在PyTorch中：`torch.Size([32, 4, 224, 224])` = 32张 224×224 的RGBA图像

---

### 向量的几何表示

<VectorChart />
向量可以表示为从原点出发的**有向线段**：
$$\mathbf{v} = \begin{bmatrix} 3 \\ 1 \end{bmatrix}, \quad \mathbf{u} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$$
**范数**（Norm，即长度）：
$$\|\mathbf{v}\| = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}$$
**单位向量**（Unit Vector）：范数为1的向量，$\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|}$
在机器学习中，一个数据样本（例如一张图像、一个用户画像）通常被表示为高维向量。

---

### 向量运算

**加法**：逐分量求和

$$\mathbf{u} + \mathbf{v} = \begin{bmatrix} u_1 + v_1 \\ u_2 + v_2 \\ \vdots \end{bmatrix}$$

**标量乘法**：每个分量乘以标量

$$c\mathbf{v} = \begin{bmatrix} cv_1 \\ cv_2 \\ \vdots \end{bmatrix}$$

**标量加法**：

$$c+\mathbf{v} = \begin{bmatrix} c \\ c \\ \vdots \end{bmatrix}+\begin{bmatrix} v_1 \\ v_2 \\ \vdots \end{bmatrix}=\begin{bmatrix} c+v_1 \\ c+v_2 \\ \vdots \end{bmatrix}$$

**点积**（Dot Product，即内积 Inner Product）：

$$\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^{n} u_i v_i = u_1 v_1 + u_2 v_2 + \cdots + u_n v_n$$

例如，$\mathbf{u} = [u_1, u_2, u_3,u_4]$ 和 $\mathbf{v} = [v_1, v_2, v_3,v_4]$：

$$\mathbf{u} \cdot \mathbf{v} = u_1 v_1 + u_2 v_2 + u_3 v_3 + u_4 v_4$$

几何解释：

$$\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\|\|\mathbf{v}\|\cos\theta$$

点积的几何解释：

- $\mathbf{u} \cdot \mathbf{v} > 0$：夹角 < 90°（大致同向）
- $\mathbf{u} \cdot \mathbf{v} = 0$：**正交**（Orthogonal，即垂直）
- $\mathbf{u} \cdot \mathbf{v} < 0$：夹角 > 90°（大致反向）
在机器学习中，神经网络的单层本质上就是输入向量和权重向量的**点积**。

---

### 向量转置

**列向量**（Column Vector）通过转置变为**行向量**（Row Vector）（反之亦然）：

$$\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ v_3 \end{bmatrix} \quad \Rightarrow \quad \mathbf{v}^T = \begin{bmatrix} v_1 & v_2 & v_3 \end{bmatrix}$$

**性质**：

- $(\mathbf{v}^T)^T = \mathbf{v}$
- $(c\mathbf{v})^T = c\mathbf{v}^T$
- $(\mathbf{u} + \mathbf{v})^T = \mathbf{u}^T + \mathbf{v}^T$
**重要性**：转置在行向量和列向量之间进行转换，这对于矩阵乘法以及定义内积/外积至关重要。

---

### 向量乘法：行 × 列

**行向量 × 列向量 → 标量（点积）**

给定 $\mathbf{a}, \mathbf{b} \in \mathbb{R}^k$（均为 $k$ 维向量）：

$$\mathbf{a}^T \mathbf{b} = \begin{bmatrix} a_1 & a_2 & \cdots & a_k \end{bmatrix} \begin{bmatrix} b_1 \\ b_2 \\ \vdots \\ b_k \end{bmatrix} = a_1 b_1 + a_2 b_2 + \cdots + a_k b_k$$

形状：$(1 \times k) \cdot (k \times 1) = 1 \times 1$

**例**：

$$\begin{bmatrix} 1 & 2 & 3 \end{bmatrix} \begin{bmatrix} 4 \\ 5 \\ 6 \end{bmatrix} = 1 \times 4 + 2 \times 5 + 3 \times 6 = 32$$

结果：$1 \times 1$ 标量

---

### 向量乘法：列 × 行

**列向量 × 行向量 → 矩阵（外积 Outer Product）**

给定 $\mathbf{a} \in \mathbb{R}^m$（$m$ 维）和 $\mathbf{b} \in \mathbb{R}^n$（$n$ 维）：

$$\mathbf{a} \mathbf{b}^T = \begin{bmatrix} a_1 \\ a_2 \\ \vdots \\ a_m \end{bmatrix} \begin{bmatrix} b_1 & b_2 & \cdots & b_n \end{bmatrix} = \begin{bmatrix} a_1 b_1 & a_1 b_2 & \cdots & a_1 b_n \\ a_2 b_1 & a_2 b_2 & \cdots & a_2 b_n \\ \vdots & \vdots & \ddots & \vdots \\ a_m b_1 & a_m b_2 & \cdots & a_m b_n \end{bmatrix}$$

形状：$(m \times 1) \cdot (1 \times n) = m \times n$

**例**：

$$\begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix} \begin{bmatrix} 4 & 5 & 6 \end{bmatrix} = \begin{bmatrix} 4 & 5 & 6 \\ 8 & 10 & 12 \\ 12 & 15 & 18 \end{bmatrix}$$

结果：$m \times n$ 矩阵

---

### 矩阵-向量乘法

对于 $\mathbf{A} \in \mathbb{R}^{m \times n}$ 和 $\mathbf{x} \in \mathbb{R}^n$：

$$\mathbf{A}\mathbf{x} = \begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} = \begin{bmatrix} a_{11}x_1 + a_{12}x_2 + \cdots + a_{1n}x_n \\ a_{21}x_1 + a_{22}x_2 + \cdots + a_{2n}x_n \\ \vdots \\ a_{m1}x_1 + a_{m2}x_2 + \cdots + a_{mn}x_n \end{bmatrix}=\mathbf{y}\in \mathbb{R}^m$$

形状：$(m \times n) \cdot (n \times 1) = m \times 1$

**例**：

$$\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \begin{bmatrix} 5 \\ 6 \end{bmatrix} = \begin{bmatrix} 1 \times 5 + 2 \times 6 \\ 3 \times 5 + 4 \times 6 \end{bmatrix} = \begin{bmatrix} 17 \\ 39 \end{bmatrix}$$

---

### 矩阵-向量乘法：行视角

**A按行划分，y的每个元素是一个点积**

$$\mathbf{A} = \begin{bmatrix} \mathbf{a}_1^T \\ \mathbf{a}_2^T \\ \vdots \\ \mathbf{a}_m^T \end{bmatrix}, \quad \mathbf{y} = \mathbf{A}\mathbf{x} = \begin{bmatrix} \mathbf{a}_1^T \mathbf{x} \\ \mathbf{a}_2^T \mathbf{x} \\ \vdots \\ \mathbf{a}_m^T \mathbf{x} \end{bmatrix}$$

每个 $y_i = \mathbf{a}_i^T \mathbf{x}$ 是 $\mathbf{A}$ 的第 $i$ 行与 $\mathbf{x}$ 的点积。

**例**：

$$\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \begin{bmatrix} 5 \\ 6 \end{bmatrix} = \begin{bmatrix} \begin{bmatrix} 1 & 2 \end{bmatrix} \cdot \begin{bmatrix} 5 \\ 6 \end{bmatrix} \\ \begin{bmatrix} 3 & 4 \end{bmatrix} \cdot \begin{bmatrix} 5 \\ 6 \end{bmatrix} \end{bmatrix} = \begin{bmatrix} 17 \\ 39 \end{bmatrix}$$

---

### 矩阵-向量乘法：列视角

**A按列划分，y是列的加权和**

$$\mathbf{A} = \begin{bmatrix} \mathbf{a}_1 & \mathbf{a}_2 & \cdots & \mathbf{a}_n \end{bmatrix}, x= \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}\quad \mathbf{y} = \mathbf{A}\mathbf{x} = x_1 \mathbf{a}_1 + x_2 \mathbf{a}_2 + \cdots + x_n \mathbf{a}_n$$

$\mathbf{y}$ 是 $\mathbf{A}$ 的列的线性组合（Linear Combination），权重由 $\mathbf{x}$ 给出。

**例**：

$$\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \begin{bmatrix} 5 \\ 6 \end{bmatrix} = 5 \begin{bmatrix} 1 \\ 3 \end{bmatrix} + 6 \begin{bmatrix} 2 \\ 4 \end{bmatrix} = \begin{bmatrix} 5 \\ 15 \end{bmatrix} + \begin{bmatrix} 12 \\ 24 \end{bmatrix} = \begin{bmatrix} 17 \\ 39 \end{bmatrix}$$

---

### 矩阵-向量乘法：几何视角

矩阵-向量乘法 $\mathbf{y} = \mathbf{A}\mathbf{x}$ 表示向量 $\mathbf{x}$ 的**几何变换**（Geometric Transformation）：

- **缩放**（Scaling）：沿坐标轴拉伸或压缩
- **旋转**（Rotation）：绕原点旋转向量
- **剪切**（Shearing）：扭曲空间
- **反射**（Reflection）：沿某轴翻转
尝试不同的预设，观察矩阵如何将蓝色向量变换为琥珀色向量！
<LinearTransform />

---

### 矩阵乘法

对于 $\mathbf{A} \in \mathbb{R}^{m \times k}$ 和 $\mathbf{B} \in \mathbb{R}^{k \times n}$，乘积 $\mathbf{C} = \mathbf{A}_{m\times k}\mathbf{B}_{k\times n} \in \mathbb{R}^{m \times n}$ 为：

$$C_{ij} = \sum_{p=1}^{k} A_{ip} B_{pj}$$

**视角1：行 × 列（点积）**

$$
\mathbf{A} = \begin{bmatrix} \mathbf{a}_1 \\ \mathbf{a}_2 \\ \vdots \\ \mathbf{a}_m \end{bmatrix}, \quad \mathbf{B} = \begin{bmatrix} \mathbf{b}_1 & \mathbf{b}_2 & \cdots & \mathbf{b}_n \end{bmatrix}
$$

$$
C = \begin{bmatrix}
\mathbf{a}_1\cdot \mathbf{b}_1 & \mathbf{a}_1\cdot \mathbf{b}_2 & \cdots & \mathbf{a}_1\cdot \mathbf{b}_n \\
\mathbf{a}_2\cdot \mathbf{b}_1 & \mathbf{a}_2\cdot \mathbf{b}_2 & \cdots & \mathbf{a}_2\cdot \mathbf{b}_n \\
\vdots & \vdots & \ddots & \vdots \\
\mathbf{a}_m\cdot \mathbf{b}_1 & \mathbf{a}_m\cdot \mathbf{b}_2 & \cdots & \mathbf{a}_m \cdot \mathbf{b}_n
\end{bmatrix}=\begin{bmatrix}Ab_1&Ab_2&\cdots&Ab_n\end{bmatrix}
$$

---

### 矩阵乘法：视角2

**视角2：列 × 行（外积）**

$$\mathbf{A} = \begin{bmatrix} \mathbf{a}_1 & \mathbf{a}_2 & \cdots & \mathbf{a}_k \end{bmatrix}, \quad \mathbf{B} = \begin{bmatrix} \mathbf{b}_1 \\ \mathbf{b}_2 \\ \vdots \\ \mathbf{b}_k \end{bmatrix}$$

$$\mathbf{C} = a_1 b_1+a_2 b_2 + \cdots + a_k b_k=\sum_{p=1}^{k} \mathbf{a}_p \mathbf{b}_p^T$$

**例**：

$$\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \begin{bmatrix} 5 & 7 & 9 \\ 6 & 8 & 10 \end{bmatrix} = \begin{bmatrix} 1{\times}5+2{\times}6 & 1{\times}7+2{\times}8 & 1{\times}9+2{\times}10 \\ 3{\times}5+4{\times}6 & 3{\times}7+4{\times}8 & 3{\times}9+4{\times}10 \end{bmatrix} = \begin{bmatrix} 17 & 23 & 29 \\ 39 & 53 & 67 \end{bmatrix}$$

---

### 矩阵乘法可视化

$$C_{ij} = \sum_{p=1}^{k} A_{ip} B_{pj}$$

<MatrixMultiply :cell-size="50" />

---

### 矩阵转置

矩阵的**转置**（Transpose）交换行和列：

$$\mathbf{A} = \begin{bmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \end{bmatrix} \quad \Rightarrow \quad \mathbf{A}^T = \begin{bmatrix} a_{11} & a_{21} \\ a_{12} & a_{22} \\ a_{13} & a_{23} \end{bmatrix}$$

**性质**：

- $(\mathbf{A}^T)^T = \mathbf{A}$
- $(\mathbf{A} + \mathbf{B})^T = \mathbf{A}^T + \mathbf{B}^T$
- $(c\mathbf{A})^T = c\mathbf{A}^T$
- $(\mathbf{A}\mathbf{B})^T = \mathbf{B}^T \mathbf{A}^T$ ← 注意顺序反转！
**例**：

$$\begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}^T = \begin{bmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{bmatrix}$$

---

### 其他重要的矩阵运算

设 $\mathbf{A} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$，$\mathbf{B} = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}$，$c = 2$

| 运算           | 定义                     | 示例                                          |
| ------------------- | ------------------------------ | ------------------------------------------------ |
| **转置**       | $(A^T)_{ij} = A_{ji}$          | $\begin{bmatrix} 1 & 3 \\ 2 & 4 \end{bmatrix}$   |
| **矩阵加法** | $(A+B)_{ij} = A_{ij} + B_{ij}$ | $\begin{bmatrix} 6 & 8 \\ 10 & 12 \end{bmatrix}$ |
| **标量加法** | $(A+c)_{ij} = A_{ij} + c$      | $\begin{bmatrix} 3 & 4 \\ 5 & 6 \end{bmatrix}$   |

---

### 矩阵运算（续）

| 运算                   | 定义                               | 示例                                            |
| --------------------------- | ---------------------------------------- | -------------------------------------------------- |
| **逐元素乘法（哈达玛积 Hadamard Product）** | $(A \odot B)_{ij} = A_{ij} \cdot B_{ij}$ | $\begin{bmatrix} 5 & 12 \\ 21 & 32 \end{bmatrix}$  |
| **矩阵乘法**   | $(AB)_{ij} = \sum_k A_{ik}B_{kj}$        | $\begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}$ |

**重要**：矩阵乘法一般**不满足交换律**（Not Commutative）：$\mathbf{A}\mathbf{B} \neq \mathbf{B}\mathbf{A}$

**例**：

$$\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix} \neq \begin{bmatrix} 23 & 34 \\ 31 & 46 \end{bmatrix} = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$$

---

### 特殊矩阵

**单位矩阵**（Identity Matrix）$\mathbf{I}$：$\mathbf{I}\mathbf{A} = \mathbf{A}\mathbf{I} = \mathbf{A}$

$$\mathbf{I}_3 = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

**对角矩阵**（Diagonal Matrix）：仅主对角线上有非零元素

$$\mathbf{D} = \begin{bmatrix} d_1 & 0 & 0 \\ 0 & d_2 & 0 \\ 0 & 0 & d_3 \end{bmatrix}, \quad \mathbf{D}\mathbf{x} = \begin{bmatrix} d_1 x_1 \\ d_2 x_2 \\ d_3 x_3 \end{bmatrix}$$

**对称矩阵**（Symmetric Matrix）：$\mathbf{A} = \mathbf{A}^T$（例如协方差矩阵 Covariance Matrix、海森矩阵 Hessian）

$$\mathbf{S} = \begin{bmatrix} 1 & 2 & 3 \\ 2 & 5 & 4 \\ 3 & 4 & 6 \end{bmatrix}$$

---

**上三角矩阵**（Upper Triangular Matrix）：对角线以下的所有元素均为零

$$\mathbf{U} = \begin{bmatrix} u_{11} & u_{12} & u_{13} \\ 0 & u_{22} & u_{23} \\ 0 & 0 & u_{33} \end{bmatrix}$$

**下三角矩阵**（Lower Triangular Matrix）：对角线以上的所有元素均为零

$$\mathbf{L} = \begin{bmatrix} l_{11} & 0 & 0 \\ l_{21} & l_{22} & 0 \\ l_{31} & l_{32} & l_{33} \end{bmatrix}$$

---

**正交矩阵**（Orthogonal Matrix）：$\mathbf{Q}^T\mathbf{Q} = \mathbf{Q}\mathbf{Q}^T = \mathbf{I}$，即 $\mathbf{Q}^{-1} = \mathbf{Q}^T$

- 列（和行）构成**标准正交基**（Orthonormal Basis）
- 保持长度和角度不变：$\|\mathbf{Q}\mathbf{x}\| = \|\mathbf{x}\|$
- $\det(\mathbf{Q}) = \pm 1$
**例**（2D旋转）：

$$\mathbf{R} = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$

在机器学习中，正交矩阵用于**主成分分析 PCA**、**QR分解**和神经网络的**正交初始化**。

---

**正定矩阵**（Positive Definite Matrix）：对所有 $\mathbf{x} \neq \mathbf{0}$，$\mathbf{x}^T\mathbf{A}\mathbf{x} > 0$

**等价条件**：

- 所有特征值（Eigenvalue）$\lambda_i > 0$
- 所有顺序主子式（Leading Principal Minor）$> 0$
- 存在Cholesky分解：$\mathbf{A} = \mathbf{L}\mathbf{L}^T$
**例**：

$$\mathbf{A} = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}, \quad \mathbf{x}^T\mathbf{A}\mathbf{x} = 2x_1^2 + 2x_1x_2 + 2x_2^2 > 0$$

**半正定**（Semi-definite）：$\mathbf{x}^T\mathbf{A}\mathbf{x} \geq 0$（特征值 $\geq 0$，例如协方差矩阵）

在机器学习中：正定海森矩阵保证凸性；正定核函数（Gram矩阵）确保有效的相似性度量。

---

### 线性方程组

$m$ 个方程 $n$ 个未知数的线性方程组可以写为 $\mathbf{A}\mathbf{x} = \mathbf{b}$：

$$\begin{cases} a_{11}x_1 + a_{12}x_2 + \cdots + a_{1n}x_n = b_1 \\ a_{21}x_1 + a_{22}x_2 + \cdots + a_{2n}x_n = b_2 \\ \vdots \\ a_{m1}x_1 + a_{m2}x_2 + \cdots + a_{mn}x_n = b_m \end{cases}$$

**例**：

$$\begin{cases} 2x + 3y = 8 \\ x - y = 1 \end{cases} \quad \Leftrightarrow \quad \begin{bmatrix} 2 & 3 \\ 1 & -1 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 8 \\ 1 \end{bmatrix}$$

解：$x = \frac{11}{5}, \; y = \frac{6}{5}$

**三种情形**：

- **唯一解**：$\mathbf{A}$ 可逆（满秩 Full Rank）
- **无解**：不相容方程组（超定 Overdetermined）
- **无穷多解**：欠定方程组（Underdetermined）

---

### 矩阵逆

对于方阵 $\mathbf{A} \in \mathbb{R}^{n \times n}$，**逆矩阵**（Inverse）$\mathbf{A}^{-1}$ 满足：

$$\mathbf{A}\mathbf{A}^{-1} = \mathbf{A}^{-1}\mathbf{A} = \mathbf{I}$$

其中 $\mathbf{I}$ 是 $n \times n$ 单位矩阵。

**求解线性方程组**：若 $\mathbf{A}$ 可逆，则 $\mathbf{A}\mathbf{x} = \mathbf{b}$ 有唯一解：

$$\mathbf{x} = \mathbf{A}^{-1}\mathbf{b}$$

**性质**：

- $(\mathbf{A}^{-1})^{-1} = \mathbf{A}$
- $(\mathbf{A}\mathbf{B})^{-1} = \mathbf{B}^{-1}\mathbf{A}^{-1}$
- $(\mathbf{A}^T)^{-1} = (\mathbf{A}^{-1})^T$
**例**（2×2矩阵）：

$$\mathbf{A} = \begin{bmatrix} a & b \\ c & d \end{bmatrix} \quad \Rightarrow \quad \mathbf{A}^{-1} = \frac{1}{ad - bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$$

---

### 行列式

方阵 $\mathbf{A} \in \mathbb{R}^{n \times n}$ 的**行列式**（Determinant）是一个标量，用于判断 $\mathbf{A}$ 是否可逆：

$$\det(\mathbf{A}) \neq 0 \quad \Leftrightarrow \quad \mathbf{A} \text{ 可逆}$$

**2×2矩阵**：

$$\det\begin{bmatrix} a & b \\ c & d \end{bmatrix} = ad - bc$$

**几何解释**：$|\det(\mathbf{A})|$ 是变换的缩放因子；符号表示方向。

- $\det(\mathbf{A}) > 0$：保持方向
- $\det(\mathbf{A}) < 0$：反转方向
- $\det(\mathbf{A}) = 0$：空间坍缩（奇异 Singular）

---

### 矩阵的秩

矩阵 $\mathbf{A}$ 的**秩**（Rank）是列空间（或行空间）的维度，即线性无关列（或行）的最大数量。

$$\text{rank}(\mathbf{A}) \leq \min(m, n)$$

**满秩**（Full Rank）：$\text{rank}(\mathbf{A}) = \min(m, n)$

- 满秩方阵 ⟹ 可逆
- 满秩矩形矩阵 ⟹ 列/行线性无关
**例**：

$$\mathbf{A} = \begin{bmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \end{bmatrix} \quad \Rightarrow \quad \text{rank}(\mathbf{A}) = 1$$

第二行是第一行的 $2$ 倍，因此只有1个线性无关行。

在机器学习中，秩揭示了数据的**有效维度**。低秩近似（Low-Rank Approximation）用于PCA和降维。

---

### 特征值与特征向量

对于方阵 $\mathbf{A} \in \mathbb{R}^{n \times n}$，标量 $\lambda$ 是**特征值**（Eigenvalue），非零向量 $\mathbf{v}$ 是**特征向量**（Eigenvector），若：

$$\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$$

> [!tip] 直觉理解
> 矩阵 $\mathbf{A}$ 仅将特征向量 $\mathbf{v}$ **缩放** $\lambda$ 倍，而不改变其方向。
**求特征值**：求解特征方程（Characteristic Equation）：

$$\det(\mathbf{A} - \lambda\mathbf{I}) = 0$$
**示例**（Example）：

$$\mathbf{A} = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}, \quad \det(\mathbf{A} - \lambda\mathbf{I}) = (2-\lambda)^2 - 1 = 0$$

特征值（Eigenvalues）：$\lambda_1 = 3, \; \lambda_2 = 1$

在机器学习（ML）中，特征值是 **PCA**、**谱聚类（spectral clustering）** 以及理解优化算法 **稳定性（stability）** 的基础。

---

### 雅可比矩阵（Jacobian Matrix）

函数 $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$ 的 **雅可比矩阵（Jacobian matrix）** 包含所有一阶偏导数：

$$\mathbf{J} = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} = \begin{bmatrix} \frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\ \frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\ \vdots & \vdots & \ddots & \vdots \\ \frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n} \end{bmatrix}$$

**示例**：对于 $\mathbf{f}(x, y) = \begin{bmatrix} x^2 + y \\ xy \end{bmatrix}$：

$$\mathbf{J} = \begin{bmatrix} 2x & 1 \\ y & x \end{bmatrix}$$

在机器学习中，雅可比矩阵用于 **反向传播（backpropagation）**、**归一化流（normalizing flows）** 和 **隐式微分（implicit differentiation）**。

---

### 海森矩阵（Hessian Matrix）

标量函数 $f: \mathbb{R}^n \to \mathbb{R}$ 的 **海森矩阵（Hessian matrix）** 包含所有二阶偏导数：

$$\mathbf{H} = \nabla^2 f = \begin{bmatrix} \frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\ \frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\ \vdots & \vdots & \ddots & \vdots \\ \frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2} \end{bmatrix}$$

**示例**：对于 $f(x, y) = x^2 + 3xy + y^2$：

$$\mathbf{H} = \begin{bmatrix} 2 & 3 \\ 3 & 2 \end{bmatrix}$$

**性质**：

- $\mathbf{H}$ 是 **对称的（symmetric）**（施瓦茨定理 Schwarz's theorem）
- **正定（Positive definite）** ⟹ 局部最小值
- **负定（Negative definite）** ⟹ 局部最大值
- **不定（Indefinite）** ⟹ 鞍点（saddle point）

在机器学习中，海森矩阵用于 **二阶优化（second-order optimization）**（牛顿法 Newton's method）、**自然梯度（natural gradient）** 以及分析 **损失曲面曲率（loss landscape curvature）**。

---

**概率论（Probability）**

---

## 第三部分：概率与统计（Part 3: Probability & Statistics）

### 你需要掌握的内容（What You'll Need）

本节涵盖完成 W0D5 教程（Tutorials）所需的全部内容：

**教程 1 — 分布与采样（Distributions & Sampling）**

- 均匀分布（Uniform）、二项分布（Binomial）、泊松分布（Poisson）、高斯分布（Gaussian）
- 使用 NumPy 采样
- 直方图（Histograms）

**教程 2 — 推断（Inference）**
- 条件概率 / 联合概率 / 边际概率（Conditional / Joint / Marginal probability）
- 似然与最大似然估计（Likelihood & MLE）
- 贝叶斯推断（Bayesian inference）：先验 → 后验（Prior → Posterior）
- 马尔可夫链（Markov chains）

---

### 随机变量与分布（Random Variables & Distributions）

---

### 随机变量（Random Variable）

**随机变量（random variable）** $X: \Omega \to \mathbb{R}$ 将随机结果映射为数值。

**离散型（Discrete）**：取可数个值

PMF：$P(X = x_k) = p_k$，$\sum_k p_k = 1$

示例：脉冲计数 $X \in \{0, 1, 2, \ldots\}$

**连续型（Continuous）**：取任意实数值

PDF：$p(x) \geq 0$，$\int_{-\infty}^{\infty} p(x)\, dx = 1$

$P(a \leq X \leq b) = \int_a^b p(x)\, dx$

注意：对于连续型，$P(X = a) = 0$。

---

### 均匀分布（Uniform Distribution）

$X \sim \mathcal{U}(a, b)$ — 在 $[a, b]$ 区间内所有值等概率：

$$p(x) = \frac{1}{b - a} \quad \text{for } x \in [a, b]$$

**NumPy 中的采样**：

```python
np.random.seed(0)                          # reproducible results
samples = np.random.uniform(0, 1, size=10) # 10 samples from U(0,1)
```

**应用**：随机初始化、探索状态空间、随机游走（random walk）。

**随机游走（random walk）** 由均匀步长组合而成：每一步在 $x$ 和 $y$ 方向上随机移动：

```python
x[step+1] = x[step] + (np.random.uniform() - 0.5) * step_size
#                       ^^^^^^^^^^^^^^^^^^^^^^^^
#                       centered around 0: range [-0.5, 0.5]
```

---

### 二项分布（Binomial Distribution）

$n$ 次独立二元试验，每次成功概率为 $p$：

$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$

其中 $\binom{n}{k} = \frac{n!}{k!(n-k)!}$ 表示从 $n$ 次试验中选择 $k$ 次成功的方式数。

**示例**：大鼠在 T 型迷宫中，10 次试验，$p = 0.5$（随机选择）。出现 7 次左转的概率是多少？

$$P(k=7 \mid n=10, p=0.5) = \binom{10}{7}(0.5)^7(0.5)^3 = 120 \times 0.000977 = 0.117$$

**采样**：

```python
samples = np.random.binomial(n=10, p=0.5, size=1000)
# each element = number of left turns in 10 trials
# histogram peaks at k=5 (the expected value np)
```

---

### 泊松分布（Poisson Distribution）

建模在固定时间间隔内的事件发生次数，平均速率为 $\lambda$：

$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$$

**示例**：一个神经元以平均速率 $\lambda = 4$ 次脉冲/秒发放。在一秒钟内恰好产生 7 次脉冲的概率是多少？

$$P(k=7 \mid \lambda=4) = \frac{4^7 e^{-4}}{7!} = \frac{16384 \times 0.0183}{5040} \approx 0.060$$

**采样**：

```python
spike_counts = np.random.poisson(lam=4, size=100)
# each element = number of spikes in one interval
# histogram is asymmetric for small λ (can't have negative spikes)
```

**何时使用**：计数离散事件（脉冲、光子到达、突变）。泊松分布是二项分布在 $n \to \infty$，$p \to 0$，$np = \lambda$ 条件下的极限。

---

### 高斯（正态）分布（Gaussian (Normal) Distribution）

最重要的连续分布：

$$X \sim \mathcal{N}(\mu, \sigma^2) \quad \Rightarrow \quad p(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

| 参数 | 符号 | 含义 |
| --------- | ---------- | ------------------ |
| 均值（Mean） | $\mu$ | 峰值中心 |
| 标准差（Std Dev） | $\sigma$ | 宽度 / 分散程度 |
| 方差（Variance） | $\sigma^2$ | 分散程度的平方 |

**68-95-99.7 法则**：$P(\mu \pm 1\sigma) \approx 68\%$，$P(\mu \pm 2\sigma) \approx 95\%$，$P(\mu \pm 3\sigma) \approx 99.7\%$

**采样**：

```python
samples = np.random.normal(mu=5, sigma=1, size=1000)
```

---

### 从零实现高斯分布（Implementing a Gaussian from Scratch）

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

### 直方图作为密度估计器（Histogram as Density Estimator）

```python
plt.hist(samples, bins=30, density=True)
```

`density=True` 进行归一化使总面积 = 1，使其可与 PDF 比较。

**样本统计量（Sample statistics）**（来自教程 1）：

```python
np.mean(samples)    # sample mean → converges to μ
np.std(samples)     # sample std → converges to σ
```

**关键洞察**：样本量少时（$n < 50$），直方图噪声很大。当 $n > 500$ 时，钟形曲线形状变得清晰。这就是 **大数定律（Law of Large Numbers）** 的体现。

---

### 概率法则（Probability Rules）

---

### 条件概率、联合概率与边际概率（Conditional, Joint, and Marginal）

对于事件 $A$ 和 $B$，且 $P(B) > 0$：

**条件概率（Conditional probability）** — 在 $B$ 发生的条件下 $A$ 发生的概率：

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}$$

**联合概率（Joint probability）** — $A$ 和 $B$ 同时发生：

$$P(A \cap B) = P(A \mid B) \cdot P(B) = P(B \mid A) \cdot P(A)$$

**边际概率（Marginal probability）** — 不考虑 $B$ 时 $A$ 的概率：

$$P(A) = P(A \mid B_1)P(B_1) + P(A \mid B_2)P(B_2) + \cdots = \sum_i P(A \mid B_i)P(B_i)$$

对于连续情况：$P(A) = \int P(A \mid y)\, p(y)\, dy$

---

### 示例：视觉神经元（Example: Visual Neurons）

40% 的神经元对水平方向有响应（$P(h_+) = 0.4$），30% 对垂直方向有响应（$P(v_+) = 0.3$）。

**独立（Independence）** → 联合概率 = 乘积：

$$P(h_+ \cap v_+) = P(h_+) \cdot P(v_+) = 0.4 \times 0.3 = 0.12$$

**非独立** → 使用条件概率：

已知 $P(h_+ \mid v_+) = 0.1$，则：

$$P(h_+ \cap v_+) = P(h_+ \mid v_+) \cdot P(v_+) = 0.1 \times 0.3 = 0.03$$

**边际恢复（Marginal recovery）**（检验）：

$$P(v_+) = P(v_+ \mid h_+)P(h_+) + P(v_+ \mid h_0)P(h_0)$$

你需要 $P(v_+ \mid h_+)$ 和 $P(v_+ \mid h_0)$ — 从联合概率和边际概率计算。

---

### 贝叶斯定理（Bayes' Theorem）

$$\boxed{\;P(Y \mid X) = \frac{P(X \mid Y) \cdot P(Y)}{P(X)}\;}$$

### 先验 $P(Y)$（Prior）

获取数据 **之前** 的信念

### 似然 $P(X \mid Y)$（Likelihood）

给定假设下的数据

### 后验 $P(Y \mid X)$（Posterior）

更新后的信念

**医学检测示例**：疾病率 1%，检测灵敏度 95%，假阳性率 10%。

$$P(\text{disease} \mid +) = \frac{0.95 \times 0.01}{0.95 \times 0.01 + 0.10 \times 0.99} = 8.8\%$$

在证据较弱（单次检测）时，先验（罕见疾病）占主导地位。

---

### 似然与最大似然估计（Likelihood & MLE）

---

### 对数法则（Logarithm Rules）

为什么对乘积取对数？因为对数将乘法转换为加法：

| 法则 | 公式 | 重要性 |
| ---- | ------- | -------------- |
| 乘积 → 求和 | $\log(a \cdot b) = \log a + \log b$ | 似然是乘积 → 对数似然是求和 |
| 幂 → 乘法 | $\log(a^k) = k \log a$ | 简化指数项 |
| 单调性 | $a > b \Leftrightarrow \log a > \log b$ | $\arg\max L = \arg\max \log L$ — 结果相同 |

**数值原因**：对于 1000 个数据点，$p(x_i) \approx 0.01$ → $L = 0.01^{1000} = 10^{-2000}$ → **下溢（underflow）** 为零。

$\log L = 1000 \times \log(0.01) = -4605$ — 一个可控的数字。

**关键性质**：最大化 $L$ 与最大化 $\log L$ 相同（对数是单调递增的）。因此我们可以自由切换。

---

### 似然函数（Likelihood Function）

给定数据 $\mathbf{x} = (x_1, \ldots, x_n)$，参数 $(\mu, \sigma)$ 的 **似然（likelihood）** 为：

$$L(\mu, \sigma) = \prod_{i=1}^n p(x_i \mid \mu, \sigma)$$

**对数似然（Log-likelihood）**（对乘积取对数）：

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

### 最大似然估计（Maximum Likelihood Estimation）

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

**为什么最小化负值？** 最大化 $L$ = 最小化 $-L$。`scipy.optimize.minimize` 只能最小化。

---

### 似然的网格搜索（Grid Search for Likelihood）

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

### 贝叶斯推断（Bayesian Inference）

---

### 先验、似然、后验（Prior, Likelihood, Posterior）

$$\underbrace{P(\theta \mid D)}_{\text{posterior}} = \frac{\overbrace{P(D \mid \theta)}^{\text{likelihood}} \cdot \underbrace{P(\theta)}_{\text{prior}}}{P(D)}$$

**共轭先验（Conjugate priors）**：当先验 × 似然 = 与先验同族分布时，更新只需简单算术。

**Beta-二项共轭（Beta-Binomial conjugacy）**：

| | 分布 | 参数 |
|---|---|---|
| 先验（Prior） | $\text{Beta}(\alpha, \beta)$ | 编码关于概率 $\theta$ 的信念 |
| 数据（Data） | $n$ 次抛掷中 $h$ 次正面，$t$ 次反面 | |
| 后验（Posterior） | $\text{Beta}(\alpha + h, \beta + t)$ | 更新后的信念 |

**Beta PDF**：$f(\theta; \alpha, \beta) = \frac{1}{B(\alpha, \beta)}\theta^{\alpha-1}(1-\theta)^{\beta-1}$

---

### 贝叶斯推断：抛硬币示例（Bayesian Inference: Coin Flip Example）

先验：$\theta \sim \text{Beta}(5, 5)$ — "可能是公平的，以 0.5 为中心"

数据：20 次抛掷，15 次正面

后验：$\theta \mid D \sim \text{Beta}(5+15, 5+5) = \text{Beta}(20, 10)$

后验均值 = $\frac{20}{20+10} = 0.67$ — 从先验（0.5）向数据（0.75）偏移。

**Beta PDF 的代码**：

```python
from scipy.stats import beta
theta = np.linspace(0, 1, 100)
prior_pdf = beta.pdf(theta, 5, 5)
posterior_pdf = beta.pdf(theta, 20, 10)
```

**MLE** = $15/20 = 0.75$（忽略先验）。**MAP** = Beta(20,10) 的众数 $\approx 0.67$（包含先验）。

随着数据增多，后验集中于 MLE — 先验被"冲淡"。

---

### 经典推断 vs 贝叶斯推断（Classical vs Bayesian Inference）

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

**比较**：数据点较少时，贝叶斯估计更稳定（由先验正则化）。数据点较多时，两者收敛到相同答案。

**要点**：贝叶斯推断给出完整的分布，而不仅仅是点估计。在数据有限时很有帮助。

---

### 马尔可夫链（Markov Chains）

---

### 马尔可夫性质（The Markov Property）

如果一个随机过程的未来仅取决于现在，而不取决于过去，则该过程具有 **马尔可夫性质（Markov property）**：

$$P(X_{t+1} \mid X_t, X_{t-1}, \ldots, X_0) = P(X_{t+1} \mid X_t)$$

**类比**：一个醉汉的下一步只取决于他现在的位置，而不是他如何到达这里的。整个历史都是无关的。

**与非马尔可夫过程的对比**：

- 马尔可夫："我现在在十字路口" → 下一步确定
- 非马尔可夫："我在十字路口，但我从北方来" → 下一步可能不同

在现实中，许多系统并非真正的马尔可夫过程，但我们可以通过在状态中包含足够的信息来 _使其_ 成为马尔可夫过程。例如：对于运动物体，仅位置不是马尔可夫的，但位置 + 速度是。

**为什么重要**：马尔可夫性质使我们能够在不追踪完整历史的情况下计算 $P(X_{t+k} \mid X_t)$。这是隐马尔可夫模型（Hidden Markov Models）、MCMC 采样和强化学习（reinforcement learning）的基础。

---

### 状态转移矩阵（State Transition Matrix）

对于具有 $n$ 个状态的系统，**转移矩阵（transition matrix）** $T$ 是一个 $n \times n$ 矩阵，其中：

$$T_{ij} = P(\text{next state} = j \mid \text{current state} = i)$$

**性质**：

- 每一行是一个概率分布：$\sum_{j=1}^n T_{ij} = 1$
- 所有元素非负：$T_{ij} \geq 0$
- $T$ 是一个 **随机矩阵（stochastic matrix）**（行随机）

**示例**：大鼠在 3 区域迷宫中（暗区 = 1，巢穴区 = 2，亮区 = 3）

$$T = \begin{bmatrix} 0.2 & 0.6 & 0.2 \\ 0.6 & 0.3 & 0.1 \\ 0.8 & 0.2 & 0.0 \end{bmatrix}$$

阅读第 1 行："如果大鼠在区域 1（暗区），有 20% 的概率停留，60% 的概率移动到巢穴区，20% 的概率移动到亮区。"

---

**解读矩阵**：

| 元素 | 值 | 含义 |
| ----- | ----- | ------- |
| $T_{11} = 0.2$ | 停留在暗区 | 大鼠倾向于离开暗区 |
| $T_{21} = 0.6$ | 巢穴区 → 暗区 | 大鼠经常从巢穴区退回到暗区 |
| $T_{31} = 0.8$ | 亮区 → 暗区 | 大鼠强烈避免停留在亮区 |
| $T_{33} = 0.0$ | 亮区 → 亮区 | 大鼠从不留在亮区 |

---

### 状态演化：从矩阵到概率（State Evolution: From Matrix to Probabilities）

如何计算经过 $k$ 步后处于每个状态的概率？

**一步**：如果当前状态已知（例如在区域 2），表示为行向量 $\mathbf{p}_0 = [0, 1, 0]$：

$$\mathbf{p}_1 = \mathbf{p}_0 \cdot T = [0, 1, 0] \cdot T = [0.6,\; 0.3,\; 0.1]$$

经过 1 步后：60% 的概率在暗区，30% 在巢穴区，10% 在亮区。

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

**为什么矩阵乘法有效**：$\mathbf{p}_0 \cdot T$ 对每个 $j$ 计算 $\sum_i p_i \cdot T_{ij}$ — 这正是全概率公式 $P(\text{next}=j) = \sum_i P(\text{next}=j \mid \text{current}=i) P(\text{current}=i)$。

---

### 稳态与时间平均（Steady State & Time Averaging）

当 $k \to \infty$ 时，$\mathbf{p}_k$ 收敛到 **稳态（steady state）** $\boldsymbol{\pi}$，与起始位置无关：

$$\boldsymbol{\pi} = \boldsymbol{\pi} \cdot T$$

> [!tip] 直觉
> 经过多次转移后，系统"遗忘"了它的起始位置。稳态是在每个区域中花费时间的长期比例。

**代码**（通过运行 100 步近似）：

```python
p_random = np.ones((1, 3)) / 3          # uniform initial distribution
p_avg = p_random @ np.linalg.matrix_power(T, 100)
# Result: ≈ [[0.447, 0.421, 0.132]]
```

**时间平均（Time averaging）**：从任意状态开始，经过多步后在每个区域的时间比例：

```python
p_random = np.ones((1, 3)) / 3          # uniform initial distribution
p_avg = p_random @ np.linalg.matrix_power(T, 100)
print(p_avg)  # ≈ [[0.447, 0.421, 0.132]]
```

**关键洞察**：稳态不依赖于起始状态（对于遍历链 ergodic chains）。这是大数定律在马尔可夫链中的类比。

**与朴素贝叶斯（Naive Bayes）的联系**：分类分布（多结果伯努利分布 categorical distribution）出现在马尔可夫链中 — $T$ 的每一行都是关于下一个状态的分类分布。

---

### 总结（Summary）

### 分布（Distributions）

- **均匀分布（Uniform）**：`np.random.uniform`
- **二项分布（Binomial）**：`np.random.binomial`
- **泊松分布（Poisson）**：`np.random.poisson`
- **高斯分布（Gaussian）**：`np.random.normal`

### 概率法则（Probability Rules）

- **条件概率（Conditional）**：$P(A|B) = P(A,B)/P(B)$
- **联合概率（Joint）**：$P(A,B) = P(A|B)P(B)$
- **边际概率（Marginal）**：求和/积分消去
- **贝叶斯（Bayes）**：后验 ∝ 似然 × 先验

### 推断（Inference）

- **似然（Likelihood）**：`norm.logpdf`，求和
- **MLE**：`scipy.optimize.minimize`
- **贝叶斯（Bayesian）**：Beta 先验 + 二项数据
- **马尔可夫（Markov）**：$\mathbf{p} \cdot T^k$

$$\boxed{\;P(Y \mid X) = \frac{P(X \mid Y) \cdot P(Y)}{P(X)}\;}$$

---
