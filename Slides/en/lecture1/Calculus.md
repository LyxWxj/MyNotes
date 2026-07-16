## Part 1: Calculus
### Limit: Definition
The **limit** describes the value that $f(x)$ approaches as $x$ gets arbitrarily close to a point $a$:
$$\lim_{x \to a} f(x) = L$$
Formally: for every $\varepsilon > 0$, there exists $\delta > 0$ such that
$$0 < |x - a| < \delta \implies |f(x) - L| < \varepsilon$$
> [!tip] Intuition
> We can make $f(x)$ as close to $L$ as we want by choosing $x$ sufficiently close to $a$ (but $x \neq a$).

---

### Limit: Examples
**Direct substitution** (when continuous):
$$\lim_{x \to 2} (x^2 + 1) = 2^2 + 1 = 5$$
**Indeterminate form** $\frac{0}{0}$ — factor and cancel:
$$\lim_{x \to 1} \frac{x^2 - 1}{x - 1} = \lim_{x \to 1} \frac{(x-1)(x+1)}{x-1} = \lim_{x \to 1}(x+1) = 2$$
**Important limits**:
$$\lim_{x \to 0} \frac{\sin x}{x} = 1$$
$$\lim_{x \to 0} \frac{e^x - 1}{x} = 1$$
$$\lim_{x \to \infty}\left(1 + \frac{1}{x}\right)^x = e$$

---

### Limit: Properties
**1. Linearity**:
$$\lim_{x \to a} [f(x) + g(x)] = \lim_{x \to a} f(x) + \lim_{x \to a} g(x)$$
$$\lim_{x \to a} [c \cdot f(x)] = c \cdot \lim_{x \to a} f(x)$$
**2. Product and Quotient**:
$$\lim_{x \to a} [f(x) \cdot g(x)] = \lim_{x \to a} f(x) \cdot \lim_{x \to a} g(x)$$
$$\lim_{x \to a} \frac{f(x)}{g(x)} = \frac{\displaystyle\lim_{x \to a} f(x)}{\displaystyle\lim_{x \to a} g(x)} \quad \text{(if } \lim_{x \to a} g(x) \neq 0\text{)}$$
**3. Squeeze Theorem**: If $g(x) \leq f(x) \leq h(x)$ near $a$, and $\lim_{x \to a} g(x) = \lim_{x \to a} h(x) = L$, then $\lim_{x \to a} f(x) = L$.

---

### Continuity
A function $f$ is **continuous** at point $a$ if:
$$\boxed{\lim_{x \to a} f(x) = f(a)}$$
This requires **three conditions**:
1. $f(a)$ is defined (the point exists)
2. $\lim_{x \to a} f(x)$ exists (left and right limits agree)
3. $\lim_{x \to a} f(x) = f(a)$ (limit equals function value)
> [!tip] Intuition
> The graph has **no breaks, jumps, or holes** at $a$. You can draw through $a$ without lifting your pen.

---

### Types of Discontinuity
**Removable (hole)**
<svg width="160" height="180" viewBox="0 0 160 180">
  <line x1="20" y1="160" x2="140" y2="160" stroke="#666" stroke-width="1"/>
  <line x1="20" y1="160" x2="20" y2="20" stroke="#666" stroke-width="1"/>
  <circle cx="80" cy="90" r="5" fill="none" stroke="#4fc3f7" stroke-width="2.5"/>
  <circle cx="80" cy="120" r="4" fill="#4fc3f7"/>
</svg>
$\lim_{x \to a} f(x) = L$ exists, but $f(a) \neq L$ or $f(a)$ undefined
**Jump**
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
**Infinite / Essential**
<svg width="160" height="180" viewBox="0 0 160 180">
  <line x1="20" y1="160" x2="140" y2="160" stroke="#666" stroke-width="1"/>
  <line x1="20" y1="160" x2="20" y2="20" stroke="#666" stroke-width="1"/>
  <line x1="80" y1="20" x2="80" y2="160" stroke="#ce93d8" stroke-width="1" stroke-dasharray="4"/>
</svg>
$\lim_{x \to a} f(x) = \pm\infty$ or does not exist

---

### Continuity & Differentiability
A function $f$ is **differentiable** at $a$ if the derivative exists:
$$f'(a) = \lim_{\Delta x \to 0} \frac{f(a + \Delta x) - f(a)}{\Delta x} \quad \text{exists}$$
**Key relationship**:
$$\text{Differentiable at } a \implies \text{Continuous at } a$$
**But NOT the converse!** A function can be continuous at $a$ but not differentiable there.
**Why?** If $f'(a)$ exists, then:
$$\lim_{\Delta x \to 0}[f(a+\Delta x) - f(a)] = \lim_{\Delta x \to 0} \frac{f(a+\Delta x) - f(a)}{\Delta x} \cdot \Delta x = f'(a) \cdot 0 = 0$$
$$\implies \lim_{\Delta x \to 0} f(a+\Delta x) = f(a) \implies \text{continuous}$$

---

### Counterexample: Continuous but NOT Differentiable
<svg width="280" height="260" viewBox="0 0 280 260">
  <line x1="20" y1="220" x2="260" y2="220" stroke="#666" stroke-width="1"/>
  <line x1="140" y1="220" x2="140" y2="30" stroke="#666" stroke-width="1"/>
  <!-- |x| -->
  <circle cx="140" cy="220" r="5" fill="#4fc3f7"/>
</svg>
**$f(x) = |x|$** at $x = 0$:
- **Continuous**: $\lim_{x \to 0}|x| = 0 = f(0)$ ✓
- **NOT differentiable**: left and right derivatives disagree:
$$f'_-(0) = \lim_{\Delta x \to 0^-}\frac{|\Delta x|}{\Delta x} = -1$$
$$f'_+(0) = \lim_{\Delta x \to 0^+}\frac{|\Delta x|}{\Delta x} = +1$$
The sharp **corner** means no unique tangent line exists.

---

### Summary: Implication Chain
**Differentiable**
$f'(a)$ exists
$\Longrightarrow$
**Continuous**
$\lim_{x\to a}f(x)=f(a)$
$\Longrightarrow$
**Limit Exists**
$\lim_{x\to a}f(x)=L$
**Converse is FALSE** (each arrow is one-directional):
| Converse                                    | Counterexample                                                   |
| ------------------------------------------- | ---------------------------------------------------------------- |
| Continuous $\not\Rightarrow$ Differentiable | $f(x)=\|x\|$ at $x=0$ (corner)                                   |
| Limit exists $\not\Rightarrow$ Continuous   | $f(x)=\begin{cases}x^2 & x\neq 0 \\ 1 & x=0\end{cases}$ at $x=0$ |

---

### Differentiability: Visual Intuition
**Differentiable**
<svg width="160" height="180" viewBox="0 0 160 180">
  <line x1="20" y1="160" x2="140" y2="160" stroke="#666" stroke-width="1"/>
  <line x1="20" y1="160" x2="20" y2="20" stroke="#666" stroke-width="1"/>
  <line x1="55" y1="85" x2="105" y2="75" stroke="#ff5252" stroke-width="1.5" stroke-dasharray="4"/>
  <circle cx="80" cy="80" r="4" fill="#4caf50"/>
</svg>
Smooth curve — tangent line exists everywhere
**Corner (not diff.)**
<svg width="160" height="180" viewBox="0 0 160 180">
  <line x1="20" y1="160" x2="140" y2="160" stroke="#666" stroke-width="1"/>
  <line x1="20" y1="160" x2="20" y2="20" stroke="#666" stroke-width="1"/>
  <circle cx="80" cy="60" r="4" fill="#ff9800"/>
  <line x1="50" y1="100" x2="80" y2="60" stroke="#ff5252" stroke-width="1.5" stroke-dasharray="4"/>
  <line x1="80" y1="60" x2="110" y2="100" stroke="#ff5252" stroke-width="1.5" stroke-dasharray="4"/>
</svg>
Corner — two tangent directions, no unique tangent
**Cusp (not diff.)**
<svg width="160" height="180" viewBox="0 0 160 180">
  <line x1="20" y1="160" x2="140" y2="160" stroke="#666" stroke-width="1"/>
  <line x1="20" y1="160" x2="20" y2="20" stroke="#666" stroke-width="1"/>
  <circle cx="80" cy="60" r="4" fill="#ce93d8"/>
  <line x1="80" y1="60" x2="80" y2="160" stroke="#ff5252" stroke-width="1.5" stroke-dasharray="4"/>
</svg>
Cusp — tangent is vertical ($\infty$ slope)

---

### Limit → Derivative
The derivative is **defined** as a limit:
$$f'(x) = \lim_{\Delta x \to 0} \frac{f(x + \Delta x) - f(x)}{\Delta x}$$
**Connection**: Without limits, we cannot rigorously define derivatives. The limit captures the idea of "instantaneous rate of change" — the slope of the tangent line as the two points on the secant line merge into one.
**Example**: Derivative of $f(x) = x^2$ from first principles:
$$f'(x) = \lim_{\Delta x \to 0} \frac{(x+\Delta x)^2 - x^2}{\Delta x} = \lim_{\Delta x \to 0} \frac{2x\Delta x + (\Delta x)^2}{\Delta x} = \lim_{\Delta x \to 0}(2x + \Delta x) = 2x$$

---

### Derivative-Definition
For a function $y=f(x)$, the derivative at point $x$ is:
$$f'(x) = \lim_{\Delta x \to 0} \frac{f(x + \Delta x) - f(x)}{\Delta x} = \frac{dy}{dx}$$
**Example**: $f(x) = x^3/3 - x$
$$f'(x) = \frac{d}{dx}\left(\frac{x^3}{3} - x\right) = x^2 - 1$$
The sign of $\frac{df}{dx}$ indicates if $f$ increases by an increasing $x$.
Question: we have some $x_0$ and $x_1 = x_0 - \beta f'(x_0)$ ($\beta$ is small enough), then $f(x_0) \mathbf{O} f(x_1)$?

---

### Intuition
$$x_1 = x_0 - \beta f'(x_0)$$
- If $f'(x_0) > 0$: $f$ is increasing, $x_1 < x_0$ (move left) $\Rightarrow$ $f$ decreases
- If $f'(x_0) < 0$: $f$ is decreasing, $x_1 > x_0$ (move right) $\Rightarrow$ $f$ decreases
- Anyway $x_1$ moves in the direction that reduces $f$
$$\boxed{f(x_0) \geq f(x_1)}$$
### Derivation (Taylor Expansion)
$$x_1 = x_0 - \beta f'(x_0)$$
$$f(x_1) \approx f(x_0) + f'(x_0)(x_1 - x_0)$$
Substitute $x_1 - x_0 = -\beta f'(x_0)$:
$$f(x_1) \approx f(x_0) - \beta\left(f'(x_0)\right)^2$$
Since $\beta > 0$ and $\left(f'(x_0)\right)^2 \geq 0$:
$$\boxed{f(x_1) \leq f(x_0)}$$
**Taylor Expansion**: For a smooth function $f$ near point $a$:
$$f(x) = f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \cdots+\frac{f^{(n)}(a)}{n!}(x-a)^n+o((x-a)^n)$$
For $a=0$,$f(x) = f(0) + f'(0)(x) + \frac{f''(0)}{2!}(x)^2 + \cdots+\frac{f^{(n)}(0)}{n!}(x)^n+o((x)^n)$

---

### Graphical Interpretation
<DerivativeChart :tangent-x="1" />
In the chart:
- **Blue curve**: $f(x) = \frac{x^3}{3} - x$
- **Red dashed line**: tangent at $x=1$
- Tangent slope = $f'(1) = 1^2 - 1 = 0$
**Key point**: points which are extrema $\rightarrow f'(x)=0$.
**The Basis of Optimization**
Points where f'(x) =0,then f(x)?

---

### Taylor Expansion Visualization
<TaylorExpansion type="poly6" />
<TaylorExpansion type="ln" />
$$f(x) = f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \cdots+\frac{f^{(n)}(a)}{n!}(x-a)^n+o((x-a)^n)$$

---

### Taylor Expansion Visualization-2
<TaylorExpansion type="exp" />
<TaylorExpansion type="sin" />
$$f(x) = f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \cdots+\frac{f^{(n)}(a)}{n!}(x-a)^n+o((x-a)^n)$$

---

### Questions
**Q1**: For $f(x) = x^6 - 3x^4 + 2x^3 + x$ at $a=0$: why does $T_2(x)$ equal $T_1(x)$, and $T_5(x)$ equal $T_4(x)$?
**Q2**: For $\sin x$ at $a=0$: why does every even-order expansion equal the previous odd-order one? (i.e. $T_{2k} = T_{2k-1}$)
Hint: Look at which coefficients are zero and why.
**A1**: $f(x) = x^6 - 3x^4 + 2x^3 + x$ has **no** $x^2$ or $x^5$ terms. At $a=0$, the Taylor coefficients $c_2 = \frac{f''(0)}{2!} = 0$ and $c_5 = \frac{f^{(5)}(0)}{5!} = 0$, so adding these terms changes nothing.
**A2**: $\sin x$ is an **odd function** ($\sin(-x) = -\sin x$), so its Taylor series at $a=0$ contains only **odd** powers: $\sin x = x - \frac{x^3}{3!} + \frac{x^5}{5!} - \cdots$. All even-order coefficients are zero, hence $T_{2k} = T_{2k-1}$.

---

### Common Differentiation Rules
| Function $y$  | Derivative $\dfrac{dy}{dx}$           | Differential $dy$               | Rule              |
| ------------- | ------------------------------------- | ------------------------------- | ----------------- |
| $y = cf(x)$   | $c\dfrac{df}{dx}$                     | $dy=c\, df=cf'(x)dx$            | Constant multiple |
| $y = f + g$   | $\dfrac{df}{dx} + \dfrac{dg}{dx}$     | $dy=df + dg=(f'(x)+g'(x))dx$    | Sum               |
| $y = f(g(x))$ | $\dfrac{df}{dg} \cdot \dfrac{dg}{dx}$ | $dy=f'(g)\, dg=f'(g(x))g'(x)dx$ | **Chain rule**    |
The **chain rule** is critical for backpropagation in neural networks. It computes derivatives of composite functions: propagating gradients backward layer by layer.
$$\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial z} \cdot \frac{\partial z}{\partial w_1}$$

---

### Gradient Descent Visualization
<GradientDescent />
**Two starting points**:
- **Red path**: start from $x_0 = -0.5$
- **Green path**: start from $x_0 = 4.5$
- Both converge to $x = 2$
**Goal**: find the minimum of $f(x) = (x-2)^2 + 1$
**Recall**: we proved $f(x_0) \geq f(x_1)$ for each step $x_1 = x_0 - \eta f'(x_0)$.
This guarantees the function value **never increases** at each iteration.
We call this **Gradient Descent**.
**Update rule**: $x_{n+1} = x_n - \eta \cdot f'(x_n)$, with $\eta = 0.3$
No matter which side we start, gradient descent always descends toward the minimum.

---

### Learning Rate: Too Small vs Too Large
<GradientDescentLR />
**Function**:
$$f(x) = (x - 2)^2 + 1$$
**Gradient**:
$$f'(x) = 2(x - 2)$$
**Update rule**:
$$x_{n+1} = x_n - \eta \cdot f'(x_n)$$
**Try adjusting $\eta$** with the slider and observe:
**Tradeoff**: Small $\eta$ is safe but slow. Large $\eta$ is fast but risky. Choosing the right learning rate is critical in ML training.

---

### Multiple Local Minima
<GradientDescentMulti />
**Function**:
$$f(x) = x^4 - 4x^3 + 2x^2 + 4x$$
**Gradient**:
$$f'(x) = 4x^3 - 12x^2 + 4x + 4$$
**Critical points** (where $f'(x) = 0$):
| Point             | Type      |
| ----------------- | --------- |
| $x \approx -0.41$ | Local min |
| $x = 1$           | Local max |
| $x \approx 2.41$  | Local min |

---

### Partial Derivative
For a multivariable function $f(x_1, x_2, \ldots, x_n)$, the partial derivative is the rate of change when **only one variable** varies:
$$\frac{\partial f}{\partial x_i} = \lim_{\Delta x_i \to 0} \frac{f(x_1, \ldots, x_i + \Delta x_i, \ldots, x_n) - f(x_1, \ldots, x_n)}{\Delta x_i}$$
Example:
$$
y=f(x,y,z),\frac{\partial f}{\partial x}=\lim_{\Delta x_i}\frac{f(x+\Delta x_i,y,z)}{\Delta x_i}
$$
The **gradient** is the vector of all partial derivatives:
$$\nabla f = \left[\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n}\right]$$
The gradient points in the direction of **steepest ascent**. In ML, we update parameters in the **opposite** direction to minimize the loss:
$$\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla L(\mathbf{w})$$
where $\eta$ is the learning rate.

---

### Second-Order Derivative
The **second-order derivative** is the derivative of the derivative, measuring how the rate of change itself changes:
$$f''(x) = \frac{d^2 f}{dx^2} = \frac{d}{dx}\left(\frac{df}{dx}\right) = \lim_{\Delta x \to 0} \frac{f'(x + \Delta x) - f'(x)}{\Delta x}$$
**Geometric meaning**: $f''(x)$ describes the **concavity** of the function.
- $f''(x) > 0$: the function is **concave up** (curves upward, like a cup) → local minimum at critical point
- $f''(x) < 0$: the function is **concave down** (curves downward, like a cap) → local maximum at critical point
- $f''(x) = 0$: possible **inflection point** (concavity changes)
**Example**: $f(x) = \frac{x^3}{3} - x$
$f'(x) = x^2 - 1, \quad f''(x) = 2x$
At $x = 1$: $f'(1) = 0$ (critical point), $f''(1) = 2 > 0$ → **local minimum**
At $x = -1$: $f'(-1) = 0$ (critical point), $f''(-1) = -2 < 0$ → **local maximum**

---

### Second-Order Derivative & Optimization
**Second-Order Taylor Expansion** near point $a$:
$$f(x) \approx f(a) + f'(a)(x-a) + \frac{f''(a)}{2}(x-a)^2$$
At a critical point where $f'(a) = 0$:
$$f(x) \approx f(a) + \frac{f''(a)}{2}(x-a)^2$$
- If $f''(a) > 0$: $f(x) \geq f(a)$ → **local minimum**
- If $f''(a) < 0$: $f(x) \leq f(a)$ → **local maximum**
**In machine learning**: The second derivative tells us about the **curvature** of the loss surface.
- Large $f''$ → steep curvature → small steps are safer
- Small $f''$ → flat region → can take larger steps
This motivates **second-order optimization methods** like Newton's method:
$$x_{n+1} = x_n - \frac{f'(x_n)}{f''(x_n)}$$

---

### Second-Order Partial Derivative
For a multivariable function $f(x_1, x_2, \ldots, x_n)$, we can take partial derivatives with respect to the same variable twice:
$$\frac{\partial^2 f}{\partial x_i^2} = \frac{\partial}{\partial x_i}\left(\frac{\partial f}{\partial x_i}\right)$$
**Example**: $f(x, y) = x^2 y + 3xy^2$
$$\frac{\partial f}{\partial x} = 2xy + 3y^2, \quad \frac{\partial^2 f}{\partial x^2} = 2y$$
$$\frac{\partial f}{\partial y} = x^2 + 6xy, \quad \frac{\partial^2 f}{\partial y^2} = 6x$$
**Interpretation**: $\frac{\partial^2 f}{\partial x_i^2}$ measures the concavity of $f$ along the $x_i$ direction, holding all other variables constant.

---

### Second-Order Mixed Partial Derivative
The **mixed partial derivative** involves taking partial derivatives with respect to **different variables**:
$$\frac{\partial^2 f}{\partial x_j \partial x_i} = \frac{\partial}{\partial x_j}\left(\frac{\partial f}{\partial x_i}\right)$$
**Example**: $f(x, y) = x^2 y + 3xy^2$
$$\frac{\partial f}{\partial x} = 2xy + 3y^2 \quad \Rightarrow \quad \frac{\partial^2 f}{\partial y \partial x} = \frac{\partial}{\partial y}(2xy + 3y^2) = 2x + 6y$$
$$\frac{\partial f}{\partial y} = x^2 + 6xy \quad \Rightarrow \quad \frac{\partial^2 f}{\partial x \partial y} = \frac{\partial}{\partial x}(x^2 + 6xy) = 2x + 6y$$
**Clairaut's Theorem (Symmetry of Mixed Partials)**:
If $f$ has continuous second-order partial derivatives, then:
$$\boxed{\frac{\partial^2 f}{\partial x_j \partial x_i} = \frac{\partial^2 f}{\partial x_i \partial x_j}}$$
The order of differentiation does not matter!

---

### Hessian Matrix
The **Hessian matrix** collects all second-order partial derivatives of a multivariable function:
$$H(f) = \begin{bmatrix} \dfrac{\partial^2 f}{\partial x_1^2} & \dfrac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \dfrac{\partial^2 f}{\partial x_1 \partial x_n} \\[6pt] \dfrac{\partial^2 f}{\partial x_2 \partial x_1} & \dfrac{\partial^2 f}{\partial x_2^2} & \cdots & \dfrac{\partial^2 f}{\partial x_2 \partial x_n} \\[6pt] \vdots & \vdots & \ddots & \vdots \\[6pt] \dfrac{\partial^2 f}{\partial x_n \partial x_1} & \dfrac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \dfrac{\partial^2 f}{\partial x_n^2} \end{bmatrix}$$
**Example**: $f(x, y) = x^2 y + 3xy^2$
$$H = \begin{bmatrix} \dfrac{\partial^2 f}{\partial x^2} & \dfrac{\partial^2 f}{\partial x \partial y} \\[6pt] \dfrac{\partial^2 f}{\partial y \partial x} & \dfrac{\partial^2 f}{\partial y^2} \end{bmatrix} = \begin{bmatrix} 2y & 2x + 6y \\ 2x + 6y & 6x \end{bmatrix}$$

---

### Hessian & Optimization
**Multivariable second-order Taylor expansion** near point $\mathbf{a}$:
$$f(\mathbf{x}) \approx f(\mathbf{a}) + \nabla f(\mathbf{a})^T (\mathbf{x} - \mathbf{a}) + \frac{1}{2}(\mathbf{x} - \mathbf{a})^T H(\mathbf{a}) (\mathbf{x} - \mathbf{a})$$
At a critical point where $\nabla f(\mathbf{a}) = \mathbf{0}$:
$$f(\mathbf{x}) \approx f(\mathbf{a}) + \frac{1}{2}(\mathbf{x} - \mathbf{a})^T H(\mathbf{a}) (\mathbf{x} - \mathbf{a})$$
**Classifying critical points** using the Hessian:
- $H$ **positive definite** → all eigenvalues $> 0$ → **local minimum**
- $H$ **negative definite** → all eigenvalues $< 0$ → **local maximum**
- $H$ **indefinite** → mixed signs → **saddle point**
**In ML**: The Hessian informs optimization algorithms. Newton's method uses $H^{-1}$ to adapt step size per direction, converging faster than gradient descent on smooth loss surfaces.
$$\mathbf{w} \leftarrow \mathbf{w} - H^{-1} \nabla L(\mathbf{w})$$

---

### Indefinite Integral
The **indefinite integral** is the antiderivative — the family of all functions whose derivative is $f(x)$:
$$\int f(x)\, dx = F(x) + C$$
where $F'(x) = f(x)$ and $C$ is an arbitrary constant (since $(F(x)+C)' = f(x)$).
**Common integration formulas**:
| $f(x)$              | $\int f(x)\, dx$           |
| ------------------- | -------------------------- |
| $x^n$ ($n \neq -1$) | $\dfrac{x^{n+1}}{n+1} + C$ |
| $\dfrac{1}{x}$      | $\ln\|x\| + C$             |
| $e^x$               | $e^x + C$                  |

---

### Properties of Indefinite Integrals
**1. Linearity (Sum Rule)**:
$$\int \left(f(x) + g(x)\right) dx = \int f(x)\, dx + \int g(x)\, dx$$
**2. Constant Multiple Rule**:
$$\int C f(x)\, dx = C \int f(x)\, dx$$
Equivalently: $\int C f(x)\, dx = \int f(x)\, d(Cx) = C \int f(x)\, dx$
**Example**: $\int (3x^2 + 2x)\, dx = 3\int x^2\, dx + 2\int x\, dx = 3 \cdot \frac{x^3}{3} + 2 \cdot \frac{x^2}{2} + C = x^3 + x^2 + C$
**Exerciese**: $\int x + e^x \, dx,\int x + \frac{1}{x}\,dx$

---

### Integration by Substitution (凑微分法)
$$\int f'(x)dx=\int df(x)=f(x)+C$$
If $\int f(x)\, dx = F(x) + C$, then:
$$\boxed{\int f(g(x))\, g'(x)\, dx =\int f(g(x))dg(x)=\int f(u)du=F(u)+C= F(g(x)) + C}$$
**Example**: $\int 2x \cos(x^2)\, dx$
Let $u = x^2$, then $du = 2x\, dx$.
$$\int 2x \cos(x^2)\, dx = \int \cos u\, du = \sin u + C = \sin(x^2) + C$$
**Example**: $\int e^{3x}\, dx$
Let $u = 3x$, then $du = 3\, dx$, so $dx = \frac{1}{3} du\r
\int e^{3x}\, dx = \frac{1}{3}\int e^u\, du = \frac{1}{3}e^{3x} + C$
**Exerciese**: $\int e^{-x} \,dx$

---

### Integration by Parts
From the product rule $(uv)' = u'v + uv'$, we get:
$$\boxed{\int u\, dv = uv - \int v\, du} or \boxed{\int u\, dv + \int v\, du=uv}$$
**Example**: $\int x e^x\, dx$
Let $u = x$, $dv = e^x dx$. Then $du = dx$, $v = e^x$.
$$\int x e^x\, dx = x e^x - \int e^x\, dx = x e^x - e^x + C = (x-1)e^x + C$$
**Exerciese**: $\int \ln\,x \, dx,\int xe^{-x}\,dx$

---

### Integral
<script setup>
import { ref } from 'vue'
const n = ref(8)
</script>
An **integral** is the inverse of the derivative, representing the accumulated quantity over $[a, b]$:
$$\int_a^b f(x)\, dx = F(a)-F(b)=\left[F(x)\right]^{b}_{a}=\lim_{n \to \infty} \sum_{i=1}^{n} f(x_i) \Delta x$$
<button class="px-3 py-1 rounded border border-gray-500 hover:bg-gray-400/20 font-mono" @click="n = Math.max(1, n - 1)">−</button>
  n = {{ n }}
  <button class="px-3 py-1 rounded border border-gray-500 hover:bg-gray-400/20 font-mono" @click="n += 1">+</button>
**Exact calculation** using the Fundamental Theorem of Calculus:
$$\int_{0.5}^{2} \left(\frac{x^2}{2} + 0.3\right) dx = \left[\frac{x^3}{6} + 0.3x\right]_{0.5}^{2}$$
$$= \underbrace{\left(\frac{2^3}{6} + 0.3 \times 2\right)}_{F(2) = \frac{29}{15} \approx 1.933} - \underbrace{\left(\frac{0.5^3}{6} + 0.3 \times 0.5\right)}_{F(0.5) = \frac{41}{240} \approx 0.171} = \frac{141}{80} \approx 1.7625$$
<RiemannChart :a="0.5" :b="2" :n="n" />

---

### What is a Differential Equation?
A **differential equation** is an equation that relates a function to its derivatives.
**Simple example**:
$$\frac{dy}{dt} = ky$$
This says: _the rate of change of $y$ is proportional to $y$ itself._
**Solutions** are functions, not numbers:
$$y(t) = Ce^{kt}$$
where $C$ is an arbitrary constant determined by initial conditions.

---

**Everyday examples**:
- **Population growth**: $\frac{dP}{dt} = rP$ — growth rate proportional to current population
- **Radioactive decay**: $\frac{dN}{dt} = -\lambda N$ — decay rate proportional to remaining atoms
- **Cooling**: $\frac{dT}{dt} = -k(T - T_{\text{env}})$ — Newton's law of cooling
**Key idea**: Instead of specifying a function directly, we specify a **rule about how it changes**. Solving the ODE means finding the function that satisfies this rule.

---

### Ordinary Differential Equations (ODE)
An **ODE** relates a function $y(t)$ to its derivatives. The **order** is the highest derivative present.
**Examples**:
| ODE                  | Order | Type                    |
| -------------------- | ----- | ----------------------- |
| $y' + 2y = 0$        | 1st   | Linear, homogeneous     |
| $y' + 2y = e^t$      | 1st   | Linear, non-homogeneous |
| $y'' + 3y' + 2y = 0$ | 2nd   | Linear, homogeneous     |
**Why ODEs matter in ML**: Many dynamical systems (e.g. neural ODEs, diffusion models) are formulated as ODEs. Training them requires solving ODEs and differentiating through the solutions.

---

### First-Order Linear ODE
A **first-order linear ODE** has the form: $y' + p(t)\,y = g(t)$
**Solution method** — Integrating Factor:
1. Compute $\mu(t) = e^{\int p(t)\,dt}$
2. Multiply both sides: $\mu(t)\,y' + \mu(t)\,p(t)\,y = \mu(t)\,g(t)$
3. Left side becomes $\frac{d}{dt}[\mu(t)\,y]$
4. Integrate: $\mu(t)\,y = \int \mu(t)\,g(t)\,dt + C$
**Example**: $y' + 2y = e^t$
$$\mu(t) = e^{\int 2\,dt} = e^{2t}$$
$$\frac{d}{dt}\!\left[e^{2t}y\right] = e^{2t} \cdot e^t = e^{3t}$$
$$e^{2t}y = \frac{1}{3}e^{3t} + C \quad\Rightarrow\quad \boxed{y = \frac{1}{3}e^{t} + Ce^{-2t}}$$

---

### Second-Order Linear ODE
A **second-order linear ODE** has the form:
$y'' + a\,y' + b\,y = g(t)$
**Homogeneous case** ($g(t) = 0$): solve the **characteristic equation**
$r^2 + ar + b = 0$
| Roots                            | General Solution                                      |
| -------------------------------- | ----------------------------------------------------- |
| Two real $r_1 \neq r_2$          | $y = C_1 e^{r_1 t} + C_2 e^{r_2 t}$                   |
| Repeated $r_1 = r_2 = r$         | $y = (C_1 + C_2 t)\,e^{rt}$                           |
| Complex $r = \alpha \pm \beta i$ | $y = e^{\alpha t}(C_1 \cos\beta t + C_2 \sin\beta t)$ |
**Example**: $y'' + 3y' + 2y = 0$
$$r^2 + 3r + 2 = 0 \;\Rightarrow\; (r+1)(r+2) = 0 \;\Rightarrow\; r_1 = -1,\; r_2 = -2$$
$$\boxed{y = C_1 e^{-t} + C_2 e^{-2t}}$$