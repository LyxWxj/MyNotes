# Lecture 0: Math Foundations for Machine Learning

Calculus · Linear Algebra · Probability

---

## Overview

Three main components:

| Subject            | Usage                                                            |
| ------------------ | ---------------------------------------------------------------- |
| **Calculus**       | Optimization — updating model parameters via gradient descent    |
| **Linear Algebra** | Representation — storing and computing data as vectors/matrices  |
| **Probability**    | Modeling — describing uncertainty with probability distributions |

Go through all these three **components**, dive into machine learning as quickly as possible.

---

**Calculus**

---

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

Let $u = 3x$, then $du = 3\, dx$, so $dx = \frac{1}{3} du \int e^{3x}\, dx = \frac{1}{3}\int e^u\, du = \frac{1}{3}e^{3x} + C$

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

An **integral** is the inverse of the derivative, representing the accumulated quantity over $[a, b]$:

$$\int_a^b f(x)\, dx = F(a)-F(b)=\left[F(x)\right]^{b}_{a}=\lim_{n \to \infty} \sum_{i=1}^{n} f(x_i) \Delta x$$

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

---

**Linear-Algebra**

---

## Part 2: Linear Algebra

### Basic Data Structures

**Scalar**: a single number

$$x = 5, \quad x \in \mathbb{R}$$

**Vector**: an ordered list of numbers

$$\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ v_3 \end{bmatrix}, \quad \mathbf{v} \in \mathbb{R}^3$$

**Matrix**: a 2D array

$$\mathbf{A} = \begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \\ a_{31} & a_{32} \end{bmatrix}, \quad \mathbf{A} \in \mathbb{R}^{3 \times 2}$$

**Tensor**: a generalization to $n$ dimensions

- 0th-order tensor = scalar
- 1st-order tensor = vector
- 2nd-order tensor = matrix
- 3rd-order and above = higher-order tensors
**Dimension notation**: $\mathbb{R}^{m \times n}$ denotes a real matrix with $m$ rows and $n$ columns.
In ML frameworks (PyTorch, TensorFlow), all data is stored and computed as tensors.

---

### Tensor Dimensions: Visual Examples

**0D — Scalar**: a single value

$$x = 5$$

**1D — Vector**: a list of values

$$\mathbf{v} = [3, 1, 4, 1, 5]$$

**2D — Matrix**: grayscale image (height × width)

<GrayscaleTensor :rows="8" :cols="8" :cell-size="20" />

Shape: $8 \times 8$ (H × W)

**3D Tensor**: color image (channels x height × width)

Shape: $3 \times 8 \times 8$ (C × H x W)

**4D Tensor**: batch of color images

$$\text{Shape: } N \times C \times H \times W $$

- $N$: batch size (number of images)
- $H \times W$: spatial dimensions
- $C$: channels (3 for RGB, 4 for RGBA)
In PyTorch: `torch.Size([32, 4, 224, 224])` = 32 RGBA images of 224×224

---

### Geometric Representation of Vectors

<VectorChart />
A vector can be represented as a **directed line segment** from the origin:
$$\mathbf{v} = \begin{bmatrix} 3 \\ 1 \end{bmatrix}, \quad \mathbf{u} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$$
**Norm (length)**:
$$\|\mathbf{v}\| = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}$$
**Unit vector**: a vector with norm 1, $\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|}$
In ML, a data sample (e.g. an image, a user profile) is typically represented as a high-dimensional vector.

---

### Vector Operations

**Addition**: component-wise sum

$$\mathbf{u} + \mathbf{v} = \begin{bmatrix} u_1 + v_1 \\ u_2 + v_2 \\ \vdots \end{bmatrix}$$

**Scalar multiplication**: multiply each component by the scalar

$$c\mathbf{v} = \begin{bmatrix} cv_1 \\ cv_2 \\ \vdots \end{bmatrix}$$

**Scalar addition**:

$$c+\mathbf{v} = \begin{bmatrix} c \\ c \\ \vdots \end{bmatrix}+\begin{bmatrix} v_1 \\ v_2 \\ \vdots \end{bmatrix}=\begin{bmatrix} c+v_1 \\ c+v_2 \\ \vdots \end{bmatrix}$$

**Dot product (inner product)**:

$$\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^{n} u_i v_i = u_1 v_1 + u_2 v_2 + \cdots + u_n v_n$$

For example, with $\mathbf{u} = [u_1, u_2, u_3,u_4]$ and $\mathbf{v} = [v_1, v_2, v_3,v_4]$:

$$\mathbf{u} \cdot \mathbf{v} = u_1 v_1 + u_2 v_2 + u_3 v_3 + u_4 v_4$$

Geometric interpretation:

$$\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\|\|\mathbf{v}\|\cos\theta$$

Geometric interpretation of the dot product:

- $\mathbf{u} \cdot \mathbf{v} > 0$: angle < 90° (roughly same direction)
- $\mathbf{u} \cdot \mathbf{v} = 0$: **orthogonal** (perpendicular)
- $\mathbf{u} \cdot \mathbf{v} < 0$: angle > 90° (roughly opposite)
In ML, a single layer of a neural network is essentially a **dot product** of the input and weight vectors.

---

### Vector Transpose

A **column vector** becomes a **row vector** (and vice versa) by transposing:

$$\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ v_3 \end{bmatrix} \quad \Rightarrow \quad \mathbf{v}^T = \begin{bmatrix} v_1 & v_2 & v_3 \end{bmatrix}$$

**Properties**:

- $(\mathbf{v}^T)^T = \mathbf{v}$
- $(c\mathbf{v})^T = c\mathbf{v}^T$
- $(\mathbf{u} + \mathbf{v})^T = \mathbf{u}^T + \mathbf{v}^T$
**Why it matters**: Transpose converts between row and column vectors, which is essential for matrix multiplication and defining inner/outer products.

---

### Vector Multiplication: Row × Column

**Row vector × Column vector → Scalar (Dot Product)**

Given $\mathbf{a}, \mathbf{b} \in \mathbb{R}^k$ (both $k$-dimensional vectors):

$$\mathbf{a}^T \mathbf{b} = \begin{bmatrix} a_1 & a_2 & \cdots & a_k \end{bmatrix} \begin{bmatrix} b_1 \\ b_2 \\ \vdots \\ b_k \end{bmatrix} = a_1 b_1 + a_2 b_2 + \cdots + a_k b_k$$

Shape: $(1 \times k) \cdot (k \times 1) = 1 \times 1$

**Example**:

$$\begin{bmatrix} 1 & 2 & 3 \end{bmatrix} \begin{bmatrix} 4 \\ 5 \\ 6 \end{bmatrix} = 1 \times 4 + 2 \times 5 + 3 \times 6 = 32$$

Result: $1 \times 1$ scalar

---

### Vector Multiplication: Column × Row

**Column vector × Row vector → Matrix (Outer Product)**

Given $\mathbf{a} \in \mathbb{R}^m$ ($m$-dimensional) and $\mathbf{b} \in \mathbb{R}^n$ ($n$-dimensional):

$$\mathbf{a} \mathbf{b}^T = \begin{bmatrix} a_1 \\ a_2 \\ \vdots \\ a_m \end{bmatrix} \begin{bmatrix} b_1 & b_2 & \cdots & b_n \end{bmatrix} = \begin{bmatrix} a_1 b_1 & a_1 b_2 & \cdots & a_1 b_n \\ a_2 b_1 & a_2 b_2 & \cdots & a_2 b_n \\ \vdots & \vdots & \ddots & \vdots \\ a_m b_1 & a_m b_2 & \cdots & a_m b_n \end{bmatrix}$$

Shape: $(m \times 1) \cdot (1 \times n) = m \times n$

**Example**:

$$\begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix} \begin{bmatrix} 4 & 5 & 6 \end{bmatrix} = \begin{bmatrix} 4 & 5 & 6 \\ 8 & 10 & 12 \\ 12 & 15 & 18 \end{bmatrix}$$

Result: $m \times n$ matrix

---

### Matrix-Vector Multiplication

For $\mathbf{A} \in \mathbb{R}^{m \times n}$ and $\mathbf{x} \in \mathbb{R}^n$:

$$\mathbf{A}\mathbf{x} = \begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} = \begin{bmatrix} a_{11}x_1 + a_{12}x_2 + \cdots + a_{1n}x_n \\ a_{21}x_1 + a_{22}x_2 + \cdots + a_{2n}x_n \\ \vdots \\ a_{m1}x_1 + a_{m2}x_2 + \cdots + a_{mn}x_n \end{bmatrix}=\mathbf{y}\in \mathbb{R}^m$$

Shape: $(m \times n) \cdot (n \times 1) = m \times 1$

**Example**:

$$\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \begin{bmatrix} 5 \\ 6 \end{bmatrix} = \begin{bmatrix} 1 \times 5 + 2 \times 6 \\ 3 \times 5 + 4 \times 6 \end{bmatrix} = \begin{bmatrix} 17 \\ 39 \end{bmatrix}$$

---

### Matrix-Vector Multiplication: Row Perspective

**A is partitioned by rows, each element of y is a dot product**

$$\mathbf{A} = \begin{bmatrix} \mathbf{a}_1^T \\ \mathbf{a}_2^T \\ \vdots \\ \mathbf{a}_m^T \end{bmatrix}, \quad \mathbf{y} = \mathbf{A}\mathbf{x} = \begin{bmatrix} \mathbf{a}_1^T \mathbf{x} \\ \mathbf{a}_2^T \mathbf{x} \\ \vdots \\ \mathbf{a}_m^T \mathbf{x} \end{bmatrix}$$

Each $y_i = \mathbf{a}_i^T \mathbf{x}$ is the dot product of the $i$-th row of $\mathbf{A}$ with $\mathbf{x}$.

**Example**:

$$\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \begin{bmatrix} 5 \\ 6 \end{bmatrix} = \begin{bmatrix} \begin{bmatrix} 1 & 2 \end{bmatrix} \cdot \begin{bmatrix} 5 \\ 6 \end{bmatrix} \\ \begin{bmatrix} 3 & 4 \end{bmatrix} \cdot \begin{bmatrix} 5 \\ 6 \end{bmatrix} \end{bmatrix} = \begin{bmatrix} 17 \\ 39 \end{bmatrix}$$

---

### Matrix-Vector Multiplication: Column Perspective

**A is partitioned by columns, y is a weighted sum of columns**

$$\mathbf{A} = \begin{bmatrix} \mathbf{a}_1 & \mathbf{a}_2 & \cdots & \mathbf{a}_n \end{bmatrix}, x= \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}\quad \mathbf{y} = \mathbf{A}\mathbf{x} = x_1 \mathbf{a}_1 + x_2 \mathbf{a}_2 + \cdots + x_n \mathbf{a}_n$$

$\mathbf{y}$ is a linear combination of the columns of $\mathbf{A}$, with weights given by $\mathbf{x}$.

**Example**:

$$\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \begin{bmatrix} 5 \\ 6 \end{bmatrix} = 5 \begin{bmatrix} 1 \\ 3 \end{bmatrix} + 6 \begin{bmatrix} 2 \\ 4 \end{bmatrix} = \begin{bmatrix} 5 \\ 15 \end{bmatrix} + \begin{bmatrix} 12 \\ 24 \end{bmatrix} = \begin{bmatrix} 17 \\ 39 \end{bmatrix}$$

---

### Matrix-Vector Multiplication: Geometry Perspective

Matrix-vector multiplication $\mathbf{y} = \mathbf{A}\mathbf{x}$ represents a **geometric transformation** of the vector $\mathbf{x}$:

- **Scaling**: stretch or shrink along axes
- **Rotation**: rotate vectors around the origin
- **Shearing**: skew the space
- **Reflection**: flip across an axis
Try different presets to see how the matrix transforms the blue vector into the amber one!
<LinearTransform />

---

### Matrix Multiplication

For $\mathbf{A} \in \mathbb{R}^{m \times k}$ and $\mathbf{B} \in \mathbb{R}^{k \times n}$, the product $\mathbf{C} = \mathbf{A}_{m\times k}\mathbf{B}_{k\times n} \in \mathbb{R}^{m \times n}$ is:

$$C_{ij} = \sum_{p=1}^{k} A_{ip} B_{pj}$$

**View 1: Row × Column (Dot Product)**

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

### Matrix Multiplication: View 2

**View 2: Column × Row (Outer Product)**

$$\mathbf{A} = \begin{bmatrix} \mathbf{a}_1 & \mathbf{a}_2 & \cdots & \mathbf{a}_k \end{bmatrix}, \quad \mathbf{B} = \begin{bmatrix} \mathbf{b}_1 \\ \mathbf{b}_2 \\ \vdots \\ \mathbf{b}_k \end{bmatrix}$$

$$\mathbf{C} = a_1 b_1+a_2 b_2 + \cdots + a_k b_k=\sum_{p=1}^{k} \mathbf{a}_p \mathbf{b}_p^T$$

**Example**:

$$\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \begin{bmatrix} 5 & 7 & 9 \\ 6 & 8 & 10 \end{bmatrix} = \begin{bmatrix} 1{\times}5+2{\times}6 & 1{\times}7+2{\times}8 & 1{\times}9+2{\times}10 \\ 3{\times}5+4{\times}6 & 3{\times}7+4{\times}8 & 3{\times}9+4{\times}10 \end{bmatrix} = \begin{bmatrix} 17 & 23 & 29 \\ 39 & 53 & 67 \end{bmatrix}$$

---

### Matrix Multiplication Visualization

$$C_{ij} = \sum_{p=1}^{k} A_{ip} B_{pj}$$

<MatrixMultiply :cell-size="50" />

---

### Matrix Transpose

The **transpose** of a matrix swaps rows and columns:

$$\mathbf{A} = \begin{bmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \end{bmatrix} \quad \Rightarrow \quad \mathbf{A}^T = \begin{bmatrix} a_{11} & a_{21} \\ a_{12} & a_{22} \\ a_{13} & a_{23} \end{bmatrix}$$

**Properties**:

- $(\mathbf{A}^T)^T = \mathbf{A}$
- $(\mathbf{A} + \mathbf{B})^T = \mathbf{A}^T + \mathbf{B}^T$
- $(c\mathbf{A})^T = c\mathbf{A}^T$
- $(\mathbf{A}\mathbf{B})^T = \mathbf{B}^T \mathbf{A}^T$ ← note the order reversal!
**Example**:

$$\begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}^T = \begin{bmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{bmatrix}$$

---

### Other Important Matrix Operations

Let $\mathbf{A} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$, $\mathbf{B} = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}$, $c = 2$

| Operation           | Definition                     | Example                                          |
| ------------------- | ------------------------------ | ------------------------------------------------ |
| **Transpose**       | $(A^T)_{ij} = A_{ji}$          | $\begin{bmatrix} 1 & 3 \\ 2 & 4 \end{bmatrix}$   |
| **Matrix Addition** | $(A+B)_{ij} = A_{ij} + B_{ij}$ | $\begin{bmatrix} 6 & 8 \\ 10 & 12 \end{bmatrix}$ |
| **Scalar Addition** | $(A+c)_{ij} = A_{ij} + c$      | $\begin{bmatrix} 3 & 4 \\ 5 & 6 \end{bmatrix}$   |

---

### Matrix Operations (cont.)

| Operation                   | Definition                               | Example                                            |
| --------------------------- | ---------------------------------------- | -------------------------------------------------- |
| **Element-wise (Hadamard)** | $(A \odot B)_{ij} = A_{ij} \cdot B_{ij}$ | $\begin{bmatrix} 5 & 12 \\ 21 & 32 \end{bmatrix}$  |
| **Matrix Multiplication**   | $(AB)_{ij} = \sum_k A_{ik}B_{kj}$        | $\begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}$ |

**Important**: Matrix multiplication is **not commutative** in general: $\mathbf{A}\mathbf{B} \neq \mathbf{B}\mathbf{A}$

**Example**:

$$\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix} \neq \begin{bmatrix} 23 & 34 \\ 31 & 46 \end{bmatrix} = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$$

---

### Special Matrices

**Identity Matrix** $\mathbf{I}$: $\mathbf{I}\mathbf{A} = \mathbf{A}\mathbf{I} = \mathbf{A}$

$$\mathbf{I}_3 = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

**Diagonal Matrix**: nonzero entries only on the main diagonal

$$\mathbf{D} = \begin{bmatrix} d_1 & 0 & 0 \\ 0 & d_2 & 0 \\ 0 & 0 & d_3 \end{bmatrix}, \quad \mathbf{D}\mathbf{x} = \begin{bmatrix} d_1 x_1 \\ d_2 x_2 \\ d_3 x_3 \end{bmatrix}$$

**Symmetric Matrix**: $\mathbf{A} = \mathbf{A}^T$ (e.g. covariance matrix, Hessian)

$$\mathbf{S} = \begin{bmatrix} 1 & 2 & 3 \\ 2 & 5 & 4 \\ 3 & 4 & 6 \end{bmatrix}$$

---

**Upper Triangular Matrix**: all entries below the diagonal are zero

$$\mathbf{U} = \begin{bmatrix} u_{11} & u_{12} & u_{13} \\ 0 & u_{22} & u_{23} \\ 0 & 0 & u_{33} \end{bmatrix}$$

**Lower Triangular Matrix**: all entries above the diagonal are zero

$$\mathbf{L} = \begin{bmatrix} l_{11} & 0 & 0 \\ l_{21} & l_{22} & 0 \\ l_{31} & l_{32} & l_{33} \end{bmatrix}$$

---

**Orthogonal Matrix**: $\mathbf{Q}^T\mathbf{Q} = \mathbf{Q}\mathbf{Q}^T = \mathbf{I}$, i.e. $\mathbf{Q}^{-1} = \mathbf{Q}^T$

- Columns (and rows) form an **orthonormal basis**
- Preserves lengths and angles: $\|\mathbf{Q}\mathbf{x}\| = \|\mathbf{x}\|$
- $\det(\mathbf{Q}) = \pm 1$
**Example** (2D rotation):

$$\mathbf{R} = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$

In ML, orthogonal matrices are used in **PCA**, **QR decomposition**, and **orthogonal initialization** of neural networks.

---

**Positive Definite Matrix**: $\mathbf{x}^T\mathbf{A}\mathbf{x} > 0$ for all $\mathbf{x} \neq \mathbf{0}$

**Equivalent conditions**:

- All eigenvalues $\lambda_i > 0$
- All leading principal minors $> 0$
- Cholesky decomposition exists: $\mathbf{A} = \mathbf{L}\mathbf{L}^T$
**Example**:

$$\mathbf{A} = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}, \quad \mathbf{x}^T\mathbf{A}\mathbf{x} = 2x_1^2 + 2x_1x_2 + 2x_2^2 > 0$$

**Semi-definite**: $\mathbf{x}^T\mathbf{A}\mathbf{x} \geq 0$ (eigenvalues $\geq 0$, e.g. covariance matrix)

In ML: positive definite Hessians guarantee convexity; positive definite kernels (Gram matrices) ensure valid similarity measures.

---

### Systems of Linear Equations

A system of $m$ linear equations in $n$ unknowns can be written as $\mathbf{A}\mathbf{x} = \mathbf{b}$:

$$\begin{cases} a_{11}x_1 + a_{12}x_2 + \cdots + a_{1n}x_n = b_1 \\ a_{21}x_1 + a_{22}x_2 + \cdots + a_{2n}x_n = b_2 \\ \vdots \\ a_{m1}x_1 + a_{m2}x_2 + \cdots + a_{mn}x_n = b_m \end{cases}$$

**Example**:

$$\begin{cases} 2x + 3y = 8 \\ x - y = 1 \end{cases} \quad \Leftrightarrow \quad \begin{bmatrix} 2 & 3 \\ 1 & -1 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 8 \\ 1 \end{bmatrix}$$

Solution: $x = \frac{11}{5}, \; y = \frac{6}{5}$

**Three cases**:

- **Unique solution**: $\mathbf{A}$ is invertible (full rank)
- **No solution**: inconsistent system (overdetermined)
- **Infinitely many solutions**: underdetermined system

---

### Matrix Inverse

For a square matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$, the **inverse** $\mathbf{A}^{-1}$ satisfies:

$$\mathbf{A}\mathbf{A}^{-1} = \mathbf{A}^{-1}\mathbf{A} = \mathbf{I}$$

where $\mathbf{I}$ is the $n \times n$ identity matrix.

**Solving linear systems**: If $\mathbf{A}$ is invertible, then $\mathbf{A}\mathbf{x} = \mathbf{b}$ has the unique solution:

$$\mathbf{x} = \mathbf{A}^{-1}\mathbf{b}$$

**Properties**:

- $(\mathbf{A}^{-1})^{-1} = \mathbf{A}$
- $(\mathbf{A}\mathbf{B})^{-1} = \mathbf{B}^{-1}\mathbf{A}^{-1}$
- $(\mathbf{A}^T)^{-1} = (\mathbf{A}^{-1})^T$
**Example** (2×2 matrix):

$$\mathbf{A} = \begin{bmatrix} a & b \\ c & d \end{bmatrix} \quad \Rightarrow \quad \mathbf{A}^{-1} = \frac{1}{ad - bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$$

---

### Determinant

The **determinant** of a square matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$ is a scalar that indicates whether $\mathbf{A}$ is invertible:

$$\det(\mathbf{A}) \neq 0 \quad \Leftrightarrow \quad \mathbf{A} \text{ is invertible}$$

**2×2 matrix**:

$$\det\begin{bmatrix} a & b \\ c & d \end{bmatrix} = ad - bc$$

**Geometric interpretation**: $|\det(\mathbf{A})|$ is the scaling factor of the transformation; the sign indicates orientation.

- $\det(\mathbf{A}) > 0$: preserves orientation
- $\det(\mathbf{A}) < 0$: reverses orientation
- $\det(\mathbf{A}) = 0$: collapses space (singular)

---

### Matrix Rank

The **rank** of a matrix $\mathbf{A}$ is the dimension of the column space (or row space), i.e., the maximum number of linearly independent columns (or rows).

$$\text{rank}(\mathbf{A}) \leq \min(m, n)$$

**Full rank**: $\text{rank}(\mathbf{A}) = \min(m, n)$

- Square matrix with full rank ⟹ invertible
- Rectangular matrix with full rank ⟹ columns/rows are linearly independent
**Example**:

$$\mathbf{A} = \begin{bmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \end{bmatrix} \quad \Rightarrow \quad \text{rank}(\mathbf{A}) = 1$$

The second row is $2 \times$ the first row, so only 1 linearly independent row.

In ML, rank reveals the **effective dimensionality** of data. Low-rank approximations are used in PCA and dimensionality reduction.

---

### Eigenvalues and Eigenvectors

For a square matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$, a scalar $\lambda$ is an **eigenvalue** and a nonzero vector $\mathbf{v}$ is an **eigenvector** if:

$$\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$$

> [!tip] Intuition
> The matrix $\mathbf{A}$ only **scales** the eigenvector $\mathbf{v}$ by $\lambda$, without changing its direction.
**Finding eigenvalues**: Solve the characteristic equation:

$$\det(\mathbf{A} - \lambda\mathbf{I}) = 0$$

**Example**:

$$\mathbf{A} = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}, \quad \det(\mathbf{A} - \lambda\mathbf{I}) = (2-\lambda)^2 - 1 = 0$$

Eigenvalues: $\lambda_1 = 3, \; \lambda_2 = 1$

In ML, eigenvalues are fundamental to **PCA**, **spectral clustering**, and understanding the **stability** of optimization algorithms.

---

### Jacobian Matrix

The **Jacobian matrix** of a function $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$ contains all first-order partial derivatives:

$$\mathbf{J} = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} = \begin{bmatrix} \frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\ \frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\ \vdots & \vdots & \ddots & \vdots \\ \frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n} \end{bmatrix}$$

**Example**: For $\mathbf{f}(x, y) = \begin{bmatrix} x^2 + y \\ xy \end{bmatrix}$:

$$\mathbf{J} = \begin{bmatrix} 2x & 1 \\ y & x \end{bmatrix}$$

In ML, the Jacobian is used in **backpropagation**, **normalizing flows**, and **implicit differentiation**.

---

### Hessian Matrix

The **Hessian matrix** of a scalar function $f: \mathbb{R}^n \to \mathbb{R}$ contains all second-order partial derivatives:

$$\mathbf{H} = \nabla^2 f = \begin{bmatrix} \frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\ \frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\ \vdots & \vdots & \ddots & \vdots \\ \frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2} \end{bmatrix}$$

**Example**: For $f(x, y) = x^2 + 3xy + y^2$:

$$\mathbf{H} = \begin{bmatrix} 2 & 3 \\ 3 & 2 \end{bmatrix}$$

**Properties**:

- $\mathbf{H}$ is **symmetric** (Schwarz's theorem)
- **Positive definite** ⟹ local minimum
- **Negative definite** ⟹ local maximum
- **Indefinite** ⟹ saddle point
In ML, the Hessian is used in **second-order optimization** (Newton's method), **natural gradient**, and analyzing **loss landscape curvature**.

---

**Probability**

---

## Part 3: Probability & Statistics

### What You'll Need

This section covers everything needed to complete the W0D5 Tutorials:

**Tutorial 1 — Distributions & Sampling**

- Uniform, Binomial, Poisson, Gaussian
- Sampling with NumPy
- Histograms
**Tutorial 2 — Inference**
- Conditional / Joint / Marginal probability
- Likelihood & MLE
- Bayesian inference (Prior → Posterior)
- Markov chains

---

### Random Variables & Distributions

---

### Random Variable

A **random variable** $X: \Omega \to \mathbb{R}$ maps random outcomes to numbers.

**Discrete**: takes countable values

PMF: $P(X = x_k) = p_k$, $\sum_k p_k = 1$

Example: spike count $X \in \{0, 1, 2, \ldots\}$

**Continuous**: takes any real value

PDF: $p(x) \geq 0$, $\int_{-\infty}^{\infty} p(x)\, dx = 1$

$P(a \leq X \leq b) = \int_a^b p(x)\, dx$

Note: $P(X = a) = 0$ for continuous.

---

### Uniform Distribution

$X \sim \mathcal{U}(a, b)$ — equal probability for all values in $[a, b]$:

$$p(x) = \frac{1}{b - a} \quad \text{for } x \in [a, b]$$

**Sampling in NumPy**:

```python
np.random.seed(0)                          # reproducible results
samples = np.random.uniform(0, 1, size=10) # 10 samples from U(0,1)
```

**Application**: random initialization, exploring state spaces, random walks.

A **random walk** combines uniform steps: at each step, move randomly in $x$ and $y$:

```python
x[step+1] = x[step] + (np.random.uniform() - 0.5) * step_size
#                       ^^^^^^^^^^^^^^^^^^^^^^^^
#                       centered around 0: range [-0.5, 0.5]
```

---

### Binomial Distribution

$n$ independent binary trials, each with success probability $p$:

$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$

where $\binom{n}{k} = \frac{n!}{k!(n-k)!}$ counts the number of ways to choose $k$ successes from $n$ trials.

**Example**: rat in a T-maze, 10 trials, $p = 0.5$ (random choice). What's the probability of 7 left turns?

$$P(k=7 \mid n=10, p=0.5) = \binom{10}{7}(0.5)^7(0.5)^3 = 120 \times 0.000977 = 0.117$$

**Sampling**:

```python
samples = np.random.binomial(n=10, p=0.5, size=1000)
# each element = number of left turns in 10 trials
# histogram peaks at k=5 (the expected value np)
```

---

### Poisson Distribution

Models the number of events in a fixed interval, with average rate $\lambda$:

$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$$

**Example**: a neuron fires at average rate $\lambda = 4$ spikes/second. What's the probability of exactly 7 spikes in one second?

$$P(k=7 \mid \lambda=4) = \frac{4^7 e^{-4}}{7!} = \frac{16384 \times 0.0183}{5040} \approx 0.060$$

**Sampling**:

```python
spike_counts = np.random.poisson(lam=4, size=100)
# each element = number of spikes in one interval
# histogram is asymmetric for small λ (can't have negative spikes)
```

**When to use**: counting discrete events (spikes, photon arrivals, mutations). Poisson is the limit of Binomial when $n \to \infty$, $p \to 0$, $np = \lambda$.

---

### Gaussian (Normal) Distribution

The most important continuous distribution:

$$X \sim \mathcal{N}(\mu, \sigma^2) \quad \Rightarrow \quad p(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

| Parameter | Symbol     | Meaning            |
| --------- | ---------- | ------------------ |
| Mean      | $\mu$      | Center of the peak |
| Std Dev   | $\sigma$   | Width / spread     |
| Variance  | $\sigma^2$ | Squared spread     |

**68-95-99.7 rule**: $P(\mu \pm 1\sigma) \approx 68\%$, $P(\mu \pm 2\sigma) \approx 95\%$, $P(\mu \pm 3\sigma) \approx 99.7\%$

**Sampling**:

```python
samples = np.random.normal(mu=5, sigma=1, size=1000)
```

---

### Implementing a Gaussian from Scratch

The Tutorial asks you to implement the PDF manually:

```python
def my_gaussian(x_points, mu, sigma):
    px = 1 / (2 * np.pi * sigma**2)**0.5 * np.exp(-(x_points - mu)**2 / (2 * sigma**2))
    # Normalize numerically (step size = 0.1)
    px = px / (0.1 * sum(px))
    return px
```

**Why normalize?** The analytical PDF integrates to 1 over $(-\infty, \infty)$, but we evaluate on a finite grid ($-8$ to $9$, step $0.1$). The numerical sum $\neq 1$, so we divide by $0.1 \times \text{sum}$ to force normalization.

**Using scipy** (alternative):

```python
from scipy.stats import norm
px = norm.pdf(x, loc=mu, scale=sigma)  # analytical, already normalized
```

---

### Histogram as Density Estimator

```python
plt.hist(samples, bins=30, density=True)
```

`density=True` normalizes so the total area = 1, making it comparable to a PDF.

**Sample statistics** (from Tutorial 1):

```python
np.mean(samples)    # sample mean → converges to μ
np.std(samples)     # sample std → converges to σ
```

**Key insight**: with few samples ($n < 50$), the histogram is noisy. With $n > 500$, the bell curve shape becomes clear. This is the **Law of Large Numbers** in action.

---

### Probability Rules

---

### Conditional, Joint, and Marginal

For events $A$ and $B$ with $P(B) > 0$:

**Conditional probability** — probability of $A$ given $B$ occurred:

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}$$

**Joint probability** — both $A$ and $B$ occur:

$$P(A \cap B) = P(A \mid B) \cdot P(B) = P(B \mid A) \cdot P(A)$$

**Marginal probability** — probability of $A$ regardless of $B$:

$$P(A) = P(A \mid B_1)P(B_1) + P(A \mid B_2)P(B_2) + \cdots = \sum_i P(A \mid B_i)P(B_i)$$

For continuous: $P(A) = \int P(A \mid y)\, p(y)\, dy$

---

### Example: Visual Neurons

40% respond to horizontal ($P(h_+) = 0.4$), 30% to vertical ($P(v_+) = 0.3$).

**Independence** → joint = product:

$$P(h_+ \cap v_+) = P(h_+) \cdot P(v_+) = 0.4 \times 0.3 = 0.12$$

**Not independent** → use conditional:

Given $P(h_+ \mid v_+) = 0.1$, then:

$$P(h_+ \cap v_+) = P(h_+ \mid v_+) \cdot P(v_+) = 0.1 \times 0.3 = 0.03$$

**Marginal recovery** (check):

$$P(v_+) = P(v_+ \mid h_+)P(h_+) + P(v_+ \mid h_0)P(h_0)$$

You need $P(v_+ \mid h_+)$ and $P(v_+ \mid h_0)$ — compute from joint and marginal.

---

### Bayes' Theorem

$$\boxed{\;P(Y \mid X) = \frac{P(X \mid Y) \cdot P(Y)}{P(X)}\;}$$

### Prior $P(Y)$

Belief **before** data

### Likelihood $P(X \mid Y)$

Data given hypothesis

### Posterior $P(Y \mid X)$

Updated belief

**Medical test**: disease rate 1%, test sensitivity 95%, false positive 10%.

$$P(\text{disease} \mid +) = \frac{0.95 \times 0.01}{0.95 \times 0.01 + 0.10 \times 0.99} = 8.8\%$$

Prior (rare disease) dominates with weak evidence (one test).

---

### Likelihood & MLE

---

### Logarithm Rules

Why take the log of a product? Because log turns multiplication into addition:

| Rule | Formula | Why it matters |
| ---- | ------- | -------------- |
| Product → Sum | $\log(a \cdot b) = \log a + \log b$ | Likelihood is a product → log-likelihood is a sum |
| Power → Multiply | $\log(a^k) = k \log a$ | Simplifies exponential terms |
| Monotone | $a > b \Leftrightarrow \log a > \log b$ | $\arg\max L = \arg\max \log L$ — same answer |

**Numerical reason**: $p(x_i) \approx 0.01$ for 1000 data points → $L = 0.01^{1000} = 10^{-2000}$ → **underflow** to zero.

$\log L = 1000 \times \log(0.01) = -4605$ — a manageable number.

**Key property**: maximizing $L$ is the same as maximizing $\log L$ (log is monotone increasing). So we can freely switch between them.

---

### Likelihood Function

Given data $\mathbf{x} = (x_1, \ldots, x_n)$, the **likelihood** of parameters $(\mu, \sigma)$ is:

$$L(\mu, \sigma) = \prod_{i=1}^n p(x_i \mid \mu, \sigma)$$

**Log-likelihood** (apply log to the product):

$$\log L = \log \prod_{i=1}^n p(x_i \mid \mu, \sigma) = \sum_{i=1}^n \log\, p(x_i \mid \mu, \sigma)$$

**Code**:

```python
from scipy.stats import norm
def compute_log_likelihood(x, mu, sigma):
    return np.sum(norm.logpdf(x, mu, sigma))
x = np.random.normal(5, 1, size=1000)
print(compute_log_likelihood(x, 4, 0.1))   # bad guess → very negative
print(compute_log_likelihood(x, 5, 1))     # good guess → less negative
```

Log-likelihood is always $\leq 0$. Closer to 0 = better fit.

---

### Maximum Likelihood Estimation

Find parameters that maximize the log-likelihood:

$$\hat{\theta}_{\text{MLE}} = \arg\max_{\theta} \log L(\theta)$$

**Analytical solution** for Gaussian (set derivative to 0):

$$\hat{\mu} = \frac{1}{n}\sum_{i=1}^n x_i, \qquad \hat{\sigma}^2 = \frac{1}{n}\sum_{i=1}^n (x_i - \hat{\mu})^2$$

**Numerical solution** (when no closed form):

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

**Why minimize negative?** Maximizing $L$ = minimizing $-L$. `scipy.optimize.minimize` only minimizes.

---

### Grid Search for Likelihood

Before optimization, build intuition with a grid search:

```python
mean_vals = np.linspace(1, 10, 10)
sigma_vals = np.array([0.7, 0.8, 0.9, 1, 1.2, 1.5, 2, 3, 4, 5])
likelihood = np.zeros((len(sigma_vals), len(mean_vals)))
for i, mu in enumerate(mean_vals):
    for j, sigma in enumerate(sigma_vals):
        likelihood[j, i] = np.sum(norm.logpdf(x, mu, sigma))
```

Plot as a heatmap. The peak of the heatmap = MLE estimate. Should be near the true $(\mu, \sigma)$ that generated the data.

---

### Bayesian Inference

---

### Prior, Likelihood, Posterior

$$\underbrace{P(\theta \mid D)}_{\text{posterior}} = \frac{\overbrace{P(D \mid \theta)}^{\text{likelihood}} \cdot \underbrace{P(\theta)}_{\text{prior}}}{P(D)}$$

**Conjugate priors**: when prior × likelihood = same family as prior, updating is just arithmetic.

**Beta-Binomial conjugacy**:

| | Distribution | Parameters |
|---|---|---|
| Prior | $\text{Beta}(\alpha, \beta)$ | encodes belief about probability $\theta$ |
| Data | $h$ heads, $t$ tails in $n$ flips | |
| Posterior | $\text{Beta}(\alpha + h, \beta + t)$ | updated belief |

**Beta PDF**: $f(\theta; \alpha, \beta) = \frac{1}{B(\alpha, \beta)}\theta^{\alpha-1}(1-\theta)^{\beta-1}$

---

### Bayesian Inference: Coin Flip Example

Prior: $\theta \sim \text{Beta}(5, 5)$ — "probably fair, centered at 0.5"

Data: 20 flips, 15 heads

Posterior: $\theta \mid D \sim \text{Beta}(5+15, 5+5) = \text{Beta}(20, 10)$

Posterior mean = $\frac{20}{20+10} = 0.67$ — shifted from prior (0.5) toward data (0.75).

**Code for Beta PDF**:

```python
from scipy.stats import beta
theta = np.linspace(0, 1, 100)
prior_pdf = beta.pdf(theta, 5, 5)
posterior_pdf = beta.pdf(theta, 20, 10)
```

**MLE** = $15/20 = 0.75$ (ignores prior). **MAP** = mode of Beta(20,10) $\approx 0.67$ (includes prior).

With more data, posterior concentrates on MLE — prior "washes out".

---

### Classical vs Bayesian Inference

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

**Comparison**: with few data points, Bayesian estimate is more stable (regularized by prior). With many data points, both converge to the same answer.

**Takeaway**: Bayesian inference gives you a full distribution, not just a point estimate. It helps when data is limited.

---

### Markov Chains

---

### The Markov Property

A stochastic process has the **Markov property** if the future depends only on the present, not the past:

$$P(X_{t+1} \mid X_t, X_{t-1}, \ldots, X_0) = P(X_{t+1} \mid X_t)$$

**Analogy**: a drunk person's next step depends only on where they are now, not how they got there. The entire history is irrelevant.

**Contrast with non-Markov**:

- Markov: "I'm at the crossroads now" → next step is determined
- Non-Markov: "I'm at the crossroads, but I came from the north" → next step might differ
In reality, many systems are not truly Markov, but we can _make_ them Markov by including enough information in the state. For example: position alone is not Markov for a moving object, but position + velocity is.
**Why it matters**: the Markov property lets us compute $P(X_{t+k} \mid X_t)$ without tracking the full history. This is the foundation of Hidden Markov Models, MCMC sampling, and reinforcement learning.

---

### State Transition Matrix

For a system with $n$ states, the **transition matrix** $T$ is an $n \times n$ matrix where:

$$T_{ij} = P(\text{next state} = j \mid \text{current state} = i)$$

**Properties**:

- Each row is a probability distribution: $\sum_{j=1}^n T_{ij} = 1$
- All entries are non-negative: $T_{ij} \geq 0$
- $T$ is a **stochastic matrix** (row-stochastic)
**Example**: rat in a 3-area maze (dark = 1, nesting = 2, bright = 3)

$$T = \begin{bmatrix} 0.2 & 0.6 & 0.2 \\ 0.6 & 0.3 & 0.1 \\ 0.8 & 0.2 & 0.0 \end{bmatrix}$$

Read row 1: "If the rat is in area 1 (dark), there's 20% chance it stays, 60% it moves to nesting, 20% it moves to bright."

---

**Reading the matrix**:

| Entry | Value | Meaning |
| ----- | ----- | ------- |
| $T_{11} = 0.2$ | stays in dark | rat tends to leave the dark area |
| $T_{21} = 0.6$ | nesting → dark | rat often retreats to dark from nesting |
| $T_{31} = 0.8$ | bright → dark | rat strongly avoids staying in bright |
| $T_{33} = 0.0$ | bright → bright | rat never stays in bright area |

---

### State Evolution: From Matrix to Probabilities

How do we compute the probability of being in each state after $k$ steps?

**One step**: if current state is known (e.g., in area 2), represent as row vector $\mathbf{p}_0 = [0, 1, 0]$:

$$\mathbf{p}_1 = \mathbf{p}_0 \cdot T = [0, 1, 0] \cdot T = [0.6,\; 0.3,\; 0.1]$$

After 1 step: 60% chance in dark, 30% in nesting, 10% in bright.

**Two steps**: apply $T$ again:

$$\mathbf{p}_2 = \mathbf{p}_1 \cdot T = \mathbf{p}_0 \cdot T^2$$

**$k$ steps**: $\mathbf{p}_k = \mathbf{p}_0 \cdot T^k$

**Code**:

```python
T = np.array([[0.2, 0.6, 0.2],
              [0.6, 0.3, 0.1],
              [0.8, 0.2, 0.0]])
p0 = np.array([0, 1, 0])               # start in area 2
p4 = p0 @ np.linalg.matrix_power(T, 4)  # after 4 transitions
print(f"P(area 2 after 4 steps) = {p4[1]:.4f}")  # 0.4311
```

**Why matrix multiplication works**: $\mathbf{p}_0 \cdot T$ computes $\sum_i p_i \cdot T_{ij}$ for each $j$ — this is exactly the law of total probability $P(\text{next}=j) = \sum_i P(\text{next}=j \mid \text{current}=i) P(\text{current}=i)$.

---

### Steady State & Time Averaging

As $k \to \infty$, $\mathbf{p}_k$ converges to **steady state** $\boldsymbol{\pi}$, regardless of starting position:

$$\boldsymbol{\pi} = \boldsymbol{\pi} \cdot T$$

> [!tip] Intuition
> after many transitions, the system "forgets" where it started. The steady state is the long-term proportion of time spent in each area.
**Code** (approximate by running 100 steps):

```python
p_random = np.ones((1, 3)) / 3          # uniform initial distribution
p_avg = p_random @ np.linalg.matrix_power(T, 100)
# Result: ≈ [[0.447, 0.421, 0.132]]
```

**Time averaging**: start from any state, after many steps the proportion of time in each area:

```python
p_random = np.ones((1, 3)) / 3          # uniform initial distribution
p_avg = p_random @ np.linalg.matrix_power(T, 100)
print(p_avg)  # ≈ [[0.447, 0.421, 0.132]]
```

**Key insight**: the steady state doesn't depend on the starting state (for ergodic chains). This is the Markov chain analogue of the Law of Large Numbers.

**Connection to Naive Bayes**: the categorical distribution (multi-outcome Bernoulli) appears in Markov chains — each row of $T$ is a categorical distribution over next states.

---

### Summary

### Distributions

- **Uniform**: `np.random.uniform`
- **Binomial**: `np.random.binomial`
- **Poisson**: `np.random.poisson`
- **Gaussian**: `np.random.normal`

### Probability Rules

- **Conditional**: $P(A|B) = P(A,B)/P(B)$
- **Joint**: $P(A,B) = P(A|B)P(B)$
- **Marginal**: sum/integrate out
- **Bayes**: posterior ∝ likelihood × prior

### Inference

- **Likelihood**: `norm.logpdf`, sum
- **MLE**: `scipy.optimize.minimize`
- **Bayesian**: Beta prior + Binomial data
- **Markov**: $\mathbf{p} \cdot T^k$

$$\boxed{\;P(Y \mid X) = \frac{P(X \mid Y) \cdot P(Y)}{P(X)}\;}$$

---
