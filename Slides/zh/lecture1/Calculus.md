## 第一部分：微积分 (Calculus)
### 极限：定义 (Limit: Definition)
**极限 (Limit)** 描述了当 $x$ 无限接近某一点 $a$ 时，$f(x)$ 所趋近的值：
$$\lim_{x \to a} f(x) = L$$
形式化定义：对于任意 $\varepsilon > 0$，存在 $\delta > 0$，使得
$$0 < |x - a| < \delta \implies |f(x) - L| < \varepsilon$$
> [!tip] 直觉理解
> 通过选择 $x$ 足够接近 $a$（但 $x \neq a$），我们可以使 $f(x)$ 任意接近 $L$。

---

### 极限：示例 (Limit: Examples)
**直接代入法** (Direct Substitution)（当函数连续时）：
$$\lim_{x \to 2} (x^2 + 1) = 2^2 + 1 = 5$$
**不定式** (Indeterminate Form) $\frac{0}{0}$ —— 因式分解并约分：
$$\lim_{x \to 1} \frac{x^2 - 1}{x - 1} = \lim_{x \to 1} \frac{(x-1)(x+1)}{x-1} = \lim_{x \to 1}(x+1) = 2$$
**重要极限** (Important Limits)：
$$\lim_{x \to 0} \frac{\sin x}{x} = 1$$
$$\lim_{x \to 0} \frac{e^x - 1}{x} = 1$$
$$\lim_{x \to \infty}\left(1 + \frac{1}{x}\right)^x = e$$

---

### 极限：性质 (Limit: Properties)
**1. 线性性** (Linearity)：
$$\lim_{x \to a} [f(x) + g(x)] = \lim_{x \to a} f(x) + \lim_{x \to a} g(x)$$
$$\lim_{x \to a} [c \cdot f(x)] = c \cdot \lim_{x \to a} f(x)$$
**2. 乘积与商** (Product and Quotient)：
$$\lim_{x \to a} [f(x) \cdot g(x)] = \lim_{x \to a} f(x) \cdot \lim_{x \to a} g(x)$$
$$\lim_{x \to a} \frac{f(x)}{g(x)} = \frac{\displaystyle\lim_{x \to a} f(x)}{\displaystyle\lim_{x \to a} g(x)} \quad \text{(if } \lim_{x \to a} g(x) \neq 0\text{)}$$
**3. 夹逼定理** (Squeeze Theorem)：若在 $a$ 附近有 $g(x) \leq f(x) \leq h(x)$，且 $\lim_{x \to a} g(x) = \lim_{x \to a} h(x) = L$，则 $\lim_{x \to a} f(x) = L$。

---

### 连续性 (Continuity)
函数 $f$ 在点 $a$ 处**连续 (Continuous)**，当且仅当：
$$\boxed{\lim_{x \to a} f(x) = f(a)}$$
这需要满足**三个条件**：
1. $f(a)$ 有定义（该点存在）
2. $\lim_{x \to a} f(x)$ 存在（左极限与右极限相等）
3. $\lim_{x \to a} f(x) = f(a)$（极限等于函数值）
> [!tip] 直觉理解
> 函数图像在 $a$ 处**没有断裂、跳跃或空洞**。你可以用笔不离开纸面画过 $a$ 点。

---

### 间断点的类型 (Types of Discontinuity)
**可去间断点 (Removable / Hole)**
<svg width="160" height="180" viewBox="0 0 160 180">
  <line x1="20" y1="160" x2="140" y2="160" stroke="#666" stroke-width="1"/>
  <line x1="20" y1="160" x2="20" y2="20" stroke="#666" stroke-width="1"/>
  <circle cx="80" cy="90" r="5" fill="none" stroke="#4fc3f7" stroke-width="2.5"/>
  <circle cx="80" cy="120" r="4" fill="#4fc3f7"/>
</svg>
$\lim_{x \to a} f(x) = L$ 存在，但 $f(a) \neq L$ 或 $f(a)$ 无定义
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

### 连续性与可导性 (Continuity & Differentiability)
函数 $f$ 在点 $a$ 处**可导 (Differentiable)**，当且仅当导数存在：
$$f'(a) = \lim_{\Delta x \to 0} \frac{f(a + \Delta x) - f(a)}{\Delta x} \quad \text{存在}$$
**关键关系**：
$$\text{在 } a \text{ 处可导} \implies \text{在 } a \text{ 处连续}$$
**但反之不成立！** 函数可以在 $a$ 处连续但不可导。
**为什么？** 若 $f'(a)$ 存在，则：
$$\lim_{\Delta x \to 0}[f(a+\Delta x) - f(a)] = \lim_{\Delta x \to 0} \frac{f(a+\Delta x) - f(a)}{\Delta x} \cdot \Delta x = f'(a) \cdot 0 = 0$$
$$\implies \lim_{\Delta x \to 0} f(a+\Delta x) = f(a) \implies \text{连续}$$

---

### 反例：连续但不可导 (Counterexample: Continuous but NOT Differentiable)
<svg width="280" height="260" viewBox="0 0 280 260">
  <line x1="20" y1="220" x2="260" y2="220" stroke="#666" stroke-width="1"/>
  <line x1="140" y1="220" x2="140" y2="30" stroke="#666" stroke-width="1"/>
  <!-- |x| -->
  <circle cx="140" cy="220" r="5" fill="#4fc3f7"/>
</svg>
**$f(x) = |x|$** 在 $x = 0$ 处：
- **连续**：$\lim_{x \to 0}|x| = 0 = f(0)$ ✓
- **不可导**：左导数与右导数不相等：
$$f'_-(0) = \lim_{\Delta x \to 0^-}\frac{|\Delta x|}{\Delta x} = -1$$
$$f'_+(0) = \lim_{\Delta x \to 0^+}\frac{|\Delta x|}{\Delta x} = +1$$
尖锐的**角点 (Corner)** 意味着不存在唯一的切线。

---

### 总结：蕴含链 (Summary: Implication Chain)
**可导 (Differentiable)**
$f'(a)$ 存在
$\Longrightarrow$
**连续 (Continuous)**
$\lim_{x\to a}f(x)=f(a)$
$\Longrightarrow$
**极限存在 (Limit Exists)**
$\lim_{x\to a}f(x)=L$
**逆命题不成立** (Converse is FALSE)（每个箭头都是单向的）：
| 逆命题                                        | 反例                                                          |
| ------------------------------------------- | ------------------------------------------------------------- |
| 连续 $\not\Rightarrow$ 可导                  | $f(x)=\|x\|$ 在 $x=0$ 处（角点）                               |
| 极限存在 $\not\Rightarrow$ 连续               | $f(x)=\begin{cases}x^2 & x\neq 0 \\ 1 & x=0\end{cases}$ 在 $x=0$ 处 |

---

### 可导性：直观理解 (Differentiability: Visual Intuition)
**可导 (Differentiable)**
<svg width="160" height="180" viewBox="0 0 160 180">
  <line x1="20" y1="160" x2="140" y2="160" stroke="#666" stroke-width="1"/>
  <line x1="20" y1="160" x2="20" y2="20" stroke="#666" stroke-width="1"/>
  <line x1="55" y1="85" x2="105" y2="75" stroke="#ff5252" stroke-width="1.5" stroke-dasharray="4"/>
  <circle cx="80" cy="80" r="4" fill="#4caf50"/>
</svg>
光滑曲线——切线处处存在
**角点 (Corner，不可导)**
<svg width="160" height="180" viewBox="0 0 160 180">
  <line x1="20" y1="160" x2="140" y2="160" stroke="#666" stroke-width="1"/>
  <line x1="20" y1="160" x2="20" y2="20" stroke="#666" stroke-width="1"/>
  <circle cx="80" cy="60" r="4" fill="#ff9800"/>
  <line x1="50" y1="100" x2="80" y2="60" stroke="#ff5252" stroke-width="1.5" stroke-dasharray="4"/>
  <line x1="80" y1="60" x2="110" y2="100" stroke="#ff5252" stroke-width="1.5" stroke-dasharray="4"/>
</svg>
角点——两个切线方向，无唯一切线
**尖点 (Cusp，不可导)**
<svg width="160" height="180" viewBox="0 0 160 180">
  <line x1="20" y1="160" x2="140" y2="160" stroke="#666" stroke-width="1"/>
  <line x1="20" y1="160" x2="20" y2="20" stroke="#666" stroke-width="1"/>
  <circle cx="80" cy="60" r="4" fill="#ce93d8"/>
  <line x1="80" y1="60" x2="80" y2="160" stroke="#ff5252" stroke-width="1.5" stroke-dasharray="4"/>
</svg>
尖点——切线垂直（斜率为 $\infty$）

---

### 从极限到导数 (Limit → Derivative)
导数**定义**为一个极限：
$$f'(x) = \lim_{\Delta x \to 0} \frac{f(x + \Delta x) - f(x)}{\Delta x}$$
**联系**：没有极限，我们就无法严格定义导数。极限捕捉了"瞬时变化率"的概念——即割线上两点合并为一点时，切线的斜率。
**示例**：从第一性原理推导 $f(x) = x^2$ 的导数：
$$f'(x) = \lim_{\Delta x \to 0} \frac{(x+\Delta x)^2 - x^2}{\Delta x} = \lim_{\Delta x \to 0} \frac{2x\Delta x + (\Delta x)^2}{\Delta x} = \lim_{\Delta x \to 0}(2x + \Delta x) = 2x$$

---

### 导数定义 (Derivative-Definition)
对于函数 $y=f(x)$，在点 $x$ 处的导数为：
$$f'(x) = \lim_{\Delta x \to 0} \frac{f(x + \Delta x) - f(x)}{\Delta x} = \frac{dy}{dx}$$
**示例**：$f(x) = x^3/3 - x$
$$f'(x) = \frac{d}{dx}\left(\frac{x^3}{3} - x\right) = x^2 - 1$$
$\frac{df}{dx}$ 的符号表示当 $x$ 增大时 $f$ 是否增大。
问题：给定某个 $x_0$ 和 $x_1 = x_0 - \beta f'(x_0)$（$\beta$ 足够小），则 $f(x_0) \mathbf{O} f(x_1)$？

---

### 直觉理解 (Intuition)
$$x_1 = x_0 - \beta f'(x_0)$$
- 若 $f'(x_0) > 0$：$f$ 递增，$x_1 < x_0$（向左移动）$\Rightarrow$ $f$ 减小
- 若 $f'(x_0) < 0$：$f$ 递减，$x_1 > x_0$（向右移动）$\Rightarrow$ $f$ 减小
- 无论如何，$x_1$ 都朝着使 $f$ 减小的方向移动
$$\boxed{f(x_0) \geq f(x_1)}$$
### 推导（泰勒展开）(Derivation - Taylor Expansion)
$$x_1 = x_0 - \beta f'(x_0)$$
$$f(x_1) \approx f(x_0) + f'(x_0)(x_1 - x_0)$$
代入 $x_1 - x_0 = -\beta f'(x_0)$：
$$f(x_1) \approx f(x_0) - \beta\left(f'(x_0)\right)^2$$
由于 $\beta > 0$ 且 $\left(f'(x_0)\right)^2 \geq 0$：
$$\boxed{f(x_1) \leq f(x_0)}$$
**泰勒展开** (Taylor Expansion)：对于在点 $a$ 附近的光滑函数 $f$：
$$f(x) = f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \cdots+\frac{f^{(n)}(a)}{n!}(x-a)^n+o((x-a)^n)$$
当 $a=0$ 时，$f(x) = f(0) + f'(0)(x) + \frac{f''(0)}{2!}(x)^2 + \cdots+\frac{f^{(n)}(0)}{n!}(x)^n+o((x)^n)$

---

### 图形解释 (Graphical Interpretation)
<DerivativeChart :tangent-x="1" />
在图表中：
- **蓝色曲线**：$f(x) = \frac{x^3}{3} - x$
- **红色虚线**：在 $x=1$ 处的切线
- 切线斜率 = $f'(1) = 1^2 - 1 = 0$
**关键点**：极值点 (Extrema) 满足 $\rightarrow f'(x)=0$。
**优化的基础** (The Basis of Optimization)
当 $f'(x) = 0$ 时，$f(x)$ 的值如何？

---

### 泰勒展开可视化 (Taylor Expansion Visualization)
<TaylorExpansion type="poly6" />
<TaylorExpansion type="ln" />
$$f(x) = f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \cdots+\frac{f^{(n)}(a)}{n!}(x-a)^n+o((x-a)^n)$$

---

### 泰勒展开可视化-2 (Taylor Expansion Visualization-2)
<TaylorExpansion type="exp" />
<TaylorExpansion type="sin" />
$$f(x) = f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \cdots+\frac{f^{(n)}(a)}{n!}(x-a)^n+o((x-a)^n)$$

---

### 问题 (Questions)
**Q1**：对于 $f(x) = x^6 - 3x^4 + 2x^3 + x$ 在 $a=0$ 处：为什么 $T_2(x)$ 等于 $T_1(x)$，而 $T_5(x)$ 等于 $T_4(x)$？
**Q2**：对于 $\sin x$ 在 $a=0$ 处：为什么每个偶数阶展开都等于前一个奇数阶展开？（即 $T_{2k} = T_{2k-1}$）
提示：观察哪些系数为零，以及为什么。
**A1**：$f(x) = x^6 - 3x^4 + 2x^3 + x$ **没有** $x^2$ 或 $x^5$ 项。在 $a=0$ 处，泰勒系数 $c_2 = \frac{f''(0)}{2!} = 0$ 且 $c_5 = \frac{f^{(5)}(0)}{5!} = 0$，因此添加这些项不会改变结果。
**A2**：$\sin x$ 是**奇函数 (Odd Function)**（$\sin(-x) = -\sin x$），因此其在 $a=0$ 处的泰勒级数只包含**奇数次幂**：$\sin x = x - \frac{x^3}{3!} + \frac{x^5}{5!} - \cdots$。所有偶数阶系数为零，因此 $T_{2k} = T_{2k-1}$。

---

### 常用求导法则 (Common Differentiation Rules)
| 函数 $y$       | 导数 $\dfrac{dy}{dx}$                 | 微分 $dy$                        | 法则                |
| ------------- | ------------------------------------- | -------------------------------- | ------------------- |
| $y = cf(x)$   | $c\dfrac{df}{dx}$                     | $dy=c\, df=cf'(x)dx$            | 常数倍法则 (Constant Multiple) |
| $y = f + g$   | $\dfrac{df}{dx} + \dfrac{dg}{dx}$     | $dy=df + dg=(f'(x)+g'(x))dx$    | 和法则 (Sum)         |
| $y = f(g(x))$ | $\dfrac{df}{dg} \cdot \dfrac{dg}{dx}$ | $dy=f'(g)\, dg=f'(g(x))g'(x)dx$ | **链式法则 (Chain Rule)** |
**链式法则 (Chain Rule)** 对于神经网络中的反向传播 (Backpropagation) 至关重要。它计算复合函数的导数：逐层反向传播梯度。
$$\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial z} \cdot \frac{\partial z}{\partial w_1}$$

---

### 梯度下降可视化 (Gradient Descent Visualization)
<GradientDescent />
**两个起始点**：
- **红色路径**：从 $x_0 = -0.5$ 开始
- **绿色路径**：从 $x_0 = 4.5$ 开始
- 两者都收敛到 $x = 2$
**目标**：求 $f(x) = (x-2)^2 + 1$ 的最小值
**回顾**：我们已证明对于每一步 $x_1 = x_0 - \eta f'(x_0)$，都有 $f(x_0) \geq f(x_1)$。
这保证了每次迭代中函数值**永不增大**。
我们称之为**梯度下降** (Gradient Descent)。
**更新规则**：$x_{n+1} = x_n - \eta \cdot f'(x_n)$，其中 $\eta = 0.3$
无论从哪一侧开始，梯度下降总是朝着最小值方向下降。

---

### 学习率：过小与过大 (Learning Rate: Too Small vs Too Large)
<GradientDescentLR />
**函数**：
$$f(x) = (x - 2)^2 + 1$$
**梯度** (Gradient)：
$$f'(x) = 2(x - 2)$$
**更新规则**：
$$x_{n+1} = x_n - \eta \cdot f'(x_n)$$
**尝试用滑块调整 $\eta$** 并观察：
**权衡** (Tradeoff)：小的 $\eta$ 安全但缓慢。大的 $\eta$ 快速但有风险。在机器学习训练中，选择合适的学习率 (Learning Rate) 至关重要。

---

### 多个局部最小值 (Multiple Local Minima)
<GradientDescentMulti />
**函数**：
$$f(x) = x^4 - 4x^3 + 2x^2 + 4x$$
**梯度**：
$$f'(x) = 4x^3 - 12x^2 + 4x + 4$$
**临界点** (Critical Points)（$f'(x) = 0$ 处）：
| 点                | 类型         |
| ----------------- | ------------ |
| $x \approx -0.41$ | 局部最小值 (Local Min) |
| $x = 1$           | 局部最大值 (Local Max) |
| $x \approx 2.41$  | 局部最小值 (Local Min) |

---

### 偏导数 (Partial Derivative)
对于多元函数 (Multivariable Function) $f(x_1, x_2, \ldots, x_n)$，偏导数是**仅有一个变量变化**时的变化率：
$$\frac{\partial f}{\partial x_i} = \lim_{\Delta x_i \to 0} \frac{f(x_1, \ldots, x_i + \Delta x_i, \ldots, x_n) - f(x_1, \ldots, x_n)}{\Delta x_i}$$
示例：
$$
y=f(x,y,z),\frac{\partial f}{\partial x}=\lim_{\Delta x_i}\frac{f(x+\Delta x_i,y,z)}{\Delta x_i}
$$
**梯度 (Gradient)** 是所有偏导数组成的向量：
$$\nabla f = \left[\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n}\right]$$
梯度指向**最陡上升** (Steepest Ascent) 的方向。在机器学习中，我们沿**相反**方向更新参数以最小化损失函数：
$$\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla L(\mathbf{w})$$
其中 $\eta$ 是学习率 (Learning Rate)。

---

### 二阶导数 (Second-Order Derivative)
**二阶导数**是导数的导数，衡量变化率本身如何变化：
$$f''(x) = \frac{d^2 f}{dx^2} = \frac{d}{dx}\left(\frac{df}{dx}\right) = \lim_{\Delta x \to 0} \frac{f'(x + \Delta x) - f'(x)}{\Delta x}$$
**几何意义**：$f''(x)$ 描述函数的**凹凸性** (Concavity)。
- $f''(x) > 0$：函数**上凹** (Concave Up)（向上弯曲，如杯形）→ 临界点处为局部最小值
- $f''(x) < 0$：函数**下凹** (Concave Down)（向下弯曲，如帽形）→ 临界点处为局部最大值
- $f''(x) = 0$：可能是**拐点** (Inflection Point)（凹凸性改变）
**示例**：$f(x) = \frac{x^3}{3} - x$
$f'(x) = x^2 - 1, \quad f''(x) = 2x$
在 $x = 1$ 处：$f'(1) = 0$（临界点），$f''(1) = 2 > 0$ → **局部最小值**
在 $x = -1$ 处：$f'(-1) = 0$（临界点），$f''(-1) = -2 < 0$ → **局部最大值**

---

### 二阶导数与优化 (Second-Order Derivative & Optimization)
在点 $a$ 附近的**二阶泰勒展开** (Second-Order Taylor Expansion)：
$$f(x) \approx f(a) + f'(a)(x-a) + \frac{f''(a)}{2}(x-a)^2$$
在临界点 $f'(a) = 0$ 处：
$$f(x) \approx f(a) + \frac{f''(a)}{2}(x-a)^2$$
- 若 $f''(a) > 0$：$f(x) \geq f(a)$ → **局部最小值**
- 若 $f''(a) < 0$：$f(x) \leq f(a)$ → **局部最大值**
**在机器学习中**：二阶导数告诉我们损失曲面 (Loss Surface) 的**曲率** (Curvature)。
- 大的 $f''$ → 陡峭曲率 → 小步长更安全
- 小的 $f''$ → 平坦区域 → 可以使用较大步长
这促使了**二阶优化方法** (Second-Order Optimization Methods) 的发展，如牛顿法 (Newton's Method)：
$$x_{n+1} = x_n - \frac{f'(x_n)}{f''(x_n)}$$

---

### 二阶偏导数 (Second-Order Partial Derivative)
对于多元函数 $f(x_1, x_2, \ldots, x_n)$，我们可以对同一变量求两次偏导数：
$$\frac{\partial^2 f}{\partial x_i^2} = \frac{\partial}{\partial x_i}\left(\frac{\partial f}{\partial x_i}\right)$$
**示例**：$f(x, y) = x^2 y + 3xy^2$
$$\frac{\partial f}{\partial x} = 2xy + 3y^2, \quad \frac{\partial^2 f}{\partial x^2} = 2y$$
$$\frac{\partial f}{\partial y} = x^2 + 6xy, \quad \frac{\partial^2 f}{\partial y^2} = 6x$$
**解释**：$\frac{\partial^2 f}{\partial x_i^2}$ 衡量 $f$ 沿 $x_i$ 方向的凹凸性，同时保持其他变量不变。

---

### 二阶混合偏导数 (Second-Order Mixed Partial Derivative)
**混合偏导数** (Mixed Partial Derivative) 涉及对**不同变量**求偏导数：
$$\frac{\partial^2 f}{\partial x_j \partial x_i} = \frac{\partial}{\partial x_j}\left(\frac{\partial f}{\partial x_i}\right)$$
**示例**：$f(x, y) = x^2 y + 3xy^2$
$$\frac{\partial f}{\partial x} = 2xy + 3y^2 \quad \Rightarrow \quad \frac{\partial^2 f}{\partial y \partial x} = \frac{\partial}{\partial y}(2xy + 3y^2) = 2x + 6y$$
$$\frac{\partial f}{\partial y} = x^2 + 6xy \quad \Rightarrow \quad \frac{\partial^2 f}{\partial x \partial y} = \frac{\partial}{\partial x}(x^2 + 6xy) = 2x + 6y$$
**克莱罗定理 (Clairaut's Theorem)（混合偏导数的对称性）**：
若 $f$ 具有连续的二阶偏导数，则：
$$\boxed{\frac{\partial^2 f}{\partial x_j \partial x_i} = \frac{\partial^2 f}{\partial x_i \partial x_j}}$$
求导顺序无关紧要！

---

### 海森矩阵 (Hessian Matrix)
**海森矩阵**收集了多元函数的所有二阶偏导数：
$$H(f) = \begin{bmatrix} \dfrac{\partial^2 f}{\partial x_1^2} & \dfrac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \dfrac{\partial^2 f}{\partial x_1 \partial x_n} \\[6pt] \dfrac{\partial^2 f}{\partial x_2 \partial x_1} & \dfrac{\partial^2 f}{\partial x_2^2} & \cdots & \dfrac{\partial^2 f}{\partial x_2 \partial x_n} \\[6pt] \vdots & \vdots & \ddots & \vdots \\[6pt] \dfrac{\partial^2 f}{\partial x_n \partial x_1} & \dfrac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \dfrac{\partial^2 f}{\partial x_n^2} \end{bmatrix}$$
**示例**：$f(x, y) = x^2 y + 3xy^2$
$$H = \begin{bmatrix} \dfrac{\partial^2 f}{\partial x^2} & \dfrac{\partial^2 f}{\partial x \partial y} \\[6pt] \dfrac{\partial^2 f}{\partial y \partial x} & \dfrac{\partial^2 f}{\partial y^2} \end{bmatrix} = \begin{bmatrix} 2y & 2x + 6y \\ 2x + 6y & 6x \end{bmatrix}$$

---

### 海森矩阵与优化 (Hessian & Optimization)
在点 $\mathbf{a}$ 附近的**多元二阶泰勒展开** (Multivariable Second-Order Taylor Expansion)：
$$f(\mathbf{x}) \approx f(\mathbf{a}) + \nabla f(\mathbf{a})^T (\mathbf{x} - \mathbf{a}) + \frac{1}{2}(\mathbf{x} - \mathbf{a})^T H(\mathbf{a}) (\mathbf{x} - \mathbf{a})$$
在临界点 $\nabla f(\mathbf{a}) = \mathbf{0}$ 处：
$$f(\mathbf{x}) \approx f(\mathbf{a}) + \frac{1}{2}(\mathbf{x} - \mathbf{a})^T H(\mathbf{a}) (\mathbf{x} - \mathbf{a})$$
利用海森矩阵**分类临界点**：
- $H$ **正定** (Positive Definite) → 所有特征值 $> 0$ → **局部最小值**
- $H$ **负定** (Negative Definite) → 所有特征值 $< 0$ → **局部最大值**
- $H$ **不定** (Indefinite) → 符号混合 → **鞍点** (Saddle Point)
**在机器学习中**：海森矩阵为优化算法提供信息。牛顿法使用 $H^{-1}$ 来自适应每个方向的步长，在光滑损失曲面上比梯度下降收敛更快。
$$\mathbf{w} \leftarrow \mathbf{w} - H^{-1} \nabla L(\mathbf{w})$$

---

### 不定积分 (Indefinite Integral)
**不定积分**是原函数 (Antiderivative)——所有导数为 $f(x)$ 的函数族：
$$\int f(x)\, dx = F(x) + C$$
其中 $F'(x) = f(x)$，$C$ 为任意常数（因为 $(F(x)+C)' = f(x)$）。
**常用积分公式** (Common Integration Formulas)：
| $f(x)$              | $\int f(x)\, dx$           |
| ------------------- | -------------------------- |
| $x^n$ ($n \neq -1$) | $\dfrac{x^{n+1}}{n+1} + C$ |
| $\dfrac{1}{x}$      | $\ln\|x\| + C$             |
| $e^x$               | $e^x + C$                  |

---

### 不定积分的性质 (Properties of Indefinite Integrals)
**1. 线性性（和法则）** (Linearity / Sum Rule)：
$$\int \left(f(x) + g(x)\right) dx = \int f(x)\, dx + \int g(x)\, dx$$
**2. 常数倍法则** (Constant Multiple Rule)：
$$\int C f(x)\, dx = C \int f(x)\, dx$$
等价形式：$\int C f(x)\, dx = \int f(x)\, d(Cx) = C \int f(x)\, dx$
**示例**：$\int (3x^2 + 2x)\, dx = 3\int x^2\, dx + 2\int x\, dx = 3 \cdot \frac{x^3}{3} + 2 \cdot \frac{x^2}{2} + C = x^3 + x^2 + C$
**练习**：$\int x + e^x \, dx,\int x + \frac{1}{x}\,dx$

---

### 换元积分法 (Integration by Substitution)
$$\int f'(x)dx=\int df(x)=f(x)+C$$
若 $\int f(x)\, dx = F(x) + C$，则：
$$\boxed{\int f(g(x))\, g'(x)\, dx =\int f(g(x))dg(x)=\int f(u)du=F(u)+C= F(g(x)) + C}$$
**示例**：$\int 2x \cos(x^2)\, dx$
令 $u = x^2$，则 $du = 2x\, dx$。
$$\int 2x \cos(x^2)\, dx = \int \cos u\, du = \sin u + C = \sin(x^2) + C$$
**示例**：$\int e^{3x}\, dx$
令 $u = 3x$，则 $du = 3\, dx$，所以 $dx = \frac{1}{3} du\r
\int e^{3x}\, dx = \frac{1}{3}\int e^u\, du = \frac{1}{3}e^{3x} + C$
**练习**：$\int e^{-x} \,dx$

---

### 分部积分法 (Integration by Parts)
由乘积法则 $(uv)' = u'v + uv'$，可得：
$$\boxed{\int u\, dv = uv - \int v\, du} or \boxed{\int u\, dv + \int v\, du=uv}$$
**示例**：$\int x e^x\, dx$
令 $u = x$，$dv = e^x dx$。则 $du = dx$，$v = e^x$。
$$\int x e^x\, dx = x e^x - \int e^x\, dx = x e^x - e^x + C = (x-1)e^x + C$$
**练习**：$\int \ln\,x \, dx,\int xe^{-x}\,dx$

---

### 积分 (Integral)
<script setup>
import { ref } from 'vue'
const n = ref(8)
</script>
**积分**是导数的逆运算，表示在 $[a, b]$ 上的累积量：
$$\int_a^b f(x)\, dx = F(a)-F(b)=\left[F(x)\right]^{b}_{a}=\lim_{n \to \infty} \sum_{i=1}^{n} f(x_i) \Delta x$$
<button class="px-3 py-1 rounded border border-gray-500 hover:bg-gray-400/20 font-mono" @click="n = Math.max(1, n - 1)">-</button>
  n = {{ n }}
  <button class="px-3 py-1 rounded border border-gray-500 hover:bg-gray-400/20 font-mono" @click="n += 1">+</button>
使用微积分基本定理 (Fundamental Theorem of Calculus) 进行**精确计算**：
$$\int_{0.5}^{2} \left(\frac{x^2}{2} + 0.3\right) dx = \left[\frac{x^3}{6} + 0.3x\right]_{0.5}^{2}$$
$$= \underbrace{\left(\frac{2^3}{6} + 0.3 \times 2\right)}_{F(2) = \frac{29}{15} \approx 1.933} - \underbrace{\left(\frac{0.5^3}{6} + 0.3 \times 0.5\right)}_{F(0.5) = \frac{41}{240} \approx 0.171} = \frac{141}{80} \approx 1.7625$$
<RiemannChart :a="0.5" :b="2" :n="n" />

---

### 什么是微分方程？(What is a Differential Equation?)
**微分方程** (Differential Equation) 是将函数与其导数相关联的方程。
**简单示例**：
$$\frac{dy}{dt} = ky$$
其含义为：_$y$ 的变化率与 $y$ 本身成正比。_
**解** (Solutions) 是函数，而非数值：
$$y(t) = Ce^{kt}$$
其中 $C$ 是由初始条件 (Initial Conditions) 确定的任意常数。

---

**日常示例**：
- **人口增长** (Population Growth)：$\frac{dP}{dt} = rP$ —— 增长率与当前人口成正比
- **放射性衰变** (Radioactive Decay)：$\frac{dN}{dt} = -\lambda N$ —— 衰变率与剩余原子数成正比
- **冷却** (Cooling)：$\frac{dT}{dt} = -k(T - T_{\text{env}})$ —— 牛顿冷却定律 (Newton's Law of Cooling)
**核心思想**：我们不是直接指定函数，而是指定一个**关于其如何变化的规则**。求解常微分方程 (ODE) 意味着找到满足该规则的函数。

---

### 常微分方程 (Ordinary Differential Equations, ODE)
**常微分方程**将函数 $y(t)$ 与其导数相关联。**阶数** (Order) 是最高导数的阶次。
**示例**：
| 常微分方程              | 阶数  | 类型                       |
| -------------------- | ---- | -------------------------- |
| $y' + 2y = 0$        | 一阶  | 线性齐次 (Linear, Homogeneous)    |
| $y' + 2y = e^t$      | 一阶  | 线性非齐次 (Linear, Non-Homogeneous) |
| $y'' + 3y' + 2y = 0$ | 二阶  | 线性齐次 (Linear, Homogeneous)    |
**为什么常微分方程在机器学习中重要**：许多动力系统 (Dynamical Systems)（如神经常微分方程 (Neural ODEs)、扩散模型 (Diffusion Models)）都被表述为常微分方程。训练它们需要求解常微分方程并对解进行微分。

---

### 一阶线性常微分方程 (First-Order Linear ODE)
**一阶线性常微分方程**的形式为：$y' + p(t)\,y = g(t)$
**求解方法** —— 积分因子法 (Integrating Factor)：
1. 计算 $\mu(t) = e^{\int p(t)\,dt}$
2. 两边同乘：$\mu(t)\,y' + \mu(t)\,p(t)\,y = \mu(t)\,g(t)$
3. 左边变为 $\frac{d}{dt}[\mu(t)\,y]$
4. 积分：$\mu(t)\,y = \int \mu(t)\,g(t)\,dt + C$
**示例**：$y' + 2y = e^t$
$$\mu(t) = e^{\int 2\,dt} = e^{2t}$$
$$\frac{d}{dt}\!\left[e^{2t}y\right] = e^{2t} \cdot e^t = e^{3t}$$
$$e^{2t}y = \frac{1}{3}e^{3t} + C \quad\Rightarrow\quad \boxed{y = \frac{1}{3}e^{t} + Ce^{-2t}}$$

---

### 二阶线性常微分方程 (Second-Order Linear ODE)
**二阶线性常微分方程**的形式为：
$y'' + a\,y' + b\,y = g(t)$
**齐次情况** (Homogeneous Case)（$g(t) = 0$）：求解**特征方程** (Characteristic Equation)
$r^2 + ar + b = 0$
| 根的类型                            | 通解 (General Solution)                                     |
| -------------------------------- | ---------------------------------------------------------- |
| 两个不等实根 $r_1 \neq r_2$         | $y = C_1 e^{r_1 t} + C_2 e^{r_2 t}$                      |
| 重根 $r_1 = r_2 = r$              | $y = (C_1 + C_2 t)\,e^{rt}$                              |
| 复根 $r = \alpha \pm \beta i$     | $y = e^{\alpha t}(C_1 \cos\beta t + C_2 \sin\beta t)$    |
**示例**：$y'' + 3y' + 2y = 0$
$$r^2 + 3r + 2 = 0 \;\Rightarrow\; (r+1)(r+2) = 0 \;\Rightarrow\; r_1 = -1,\; r_2 = -2$$
$$\boxed{y = C_1 e^{-t} + C_2 e^{-2t}}$$