# Neuromatch Notebooks — Week 0

Python & LIF Neurons · Spikes & Refractory Period · Linear Algebra · Calculus · Probability

---

## Overview

Week 0 builds the mathematical and programming foundations for computational neuroscience:

| Day      | Topic                      | Core Skill                               |
| -------- | -------------------------- | ---------------------------------------- |
| **W0D1** | Python & LIF Model         | Euler integration, NumPy basics          |
| **W0D2** | Spikes & Code Organization | Boolean indexing, classes                |
| **W0D3** | Linear Algebra             | Vectors, matrices, eigenvalues           |
| **W0D4** | Calculus                   | Differentiation, integration, chain rule |
| **W0D5** | Probability & Statistics   | Distributions, Poisson spiking           |

**The unifying theme**: simulate a neuron from first principles, then build the math to analyze it.

---

## W0D1: Python Basics & the LIF Model

---

### The LIF Neuron Model

The **Leaky-Integrate-and-Fire** neuron is the simplest spiking neuron model:

$$\tau_m \frac{d}{dt}V(t) = E_L - V(t) + R\,I(t) \quad \text{if } V(t) \leq V_{threshold}$$

**Reset condition**:

$$V(t) = V_{reset} \quad \text{if } V(t) > V_{threshold}$$

**Physical intuition**: the membrane acts like a leaky capacitor — it integrates input current but also leaks charge back toward $E_L$.

| Parameter              | Symbol      | Value         | Meaning             |
| ---------------------- | ----------- | ------------- | ------------------- |
| Membrane time constant | $\tau_m$    | 20 ms         | How fast V responds |
| Leak potential         | $E_L$       | -60 mV        | Resting voltage     |
| Reset voltage          | $V_{reset}$ | -70 mV        | Voltage after spike |
| Threshold              | $V_{th}$    | -50 mV        | Spike trigger       |
| Membrane resistance    | $R$         | 100 M$\Omega$ | Input sensitivity   |
| Mean input current     | $I_{mean}$  | 250 pA        | Drive strength      |

---

### Key Formula: Euler Discretization

The ODE cannot be solved analytically in general. We discretize it:

$$V(t + \Delta t) = V(t) + \frac{\Delta t}{\tau_m}\left(E_L - V(t) + R\,I(t)\right)$$

This is the **forward Euler method** — the single most reused technique across all notebooks.

**Code pattern** (appears in every simulation notebook):

```python
v = el                              # initial condition V(0) = E_L
for step in range(step_end):
    t = step * dt
    i = compute_input(t)            # synaptic input at this step
    v = v + dt/tau * (el - v + r*i) # Euler update
```

**Why Euler works**: for small enough $\Delta t$, the linear approximation of $dV/dt$ over one step is accurate. The error is $o(\Delta t)$ — halving the step size halves the error.

---

### Derivation: From ODE to Difference Equation

**Step 1**: Start with the continuous membrane equation (ODE):

$$\tau_m \frac{dV}{dt} = E_L - V(t) + R\,I(t)$$

**Step 2**: Approximate the derivative with a finite difference — replace $\frac{dV}{dt}$ with $\frac{\Delta V}{\Delta t}$:

$$\tau_m\frac{\Delta V}{\Delta t}=\tau_m \frac{V(t + \Delta t) - V(t)}{\Delta t} = E_L - V(t) + R\,I(t)$$

This is no longer a differential equation — it's an algebraic equation relating $V(t+\Delta t)$ to $V(t)$.

**Step 3**: Multiply both sides by $\frac{\Delta t}{\tau_m}$:

$$V(t + \Delta t) - V(t) = \frac{\Delta t}{\tau_m}\left(E_L - V(t) + R\,I(t)\right)$$

---

### Derivation: Iterative Update Rule

**Step 4**: Move $V(t)$ to the right side:

$$\boxed{\;V(t + \Delta t) = V(t) + \frac{\Delta t}{\tau_m}\left(E_L - V(t) + R\,I(t)\right)\;}$$

This is the **forward Euler update rule** — given the current state $V(t)$ and input $I(t)$, we can compute the next state $V(t+\Delta t)$.

**Step 5**: Translate to code — let `v` represent $V(t)$:

```python
v = v + dt / tau * (el - v + r * i)
#  ↑   ↑
#  |   └── right side: evaluates old V(t)
#  └────── left side:  becomes new V(t+Δt) after assignment
```

The **same variable** `v` appears on both sides because Python evaluates the right side first, then assigns the result to the left side.

**Geometric intuition**: the Euler method approximates the curve $V(t)$ as a sequence of short straight line segments, each with slope $\frac{dV}{dt}$ evaluated at the current point.

---

### Input Current Models

**Sinusoidal input** (deterministic)

$$I(t) = I_{mean}\left(1 + \sin\left(\frac{2\pi}{0.01}\,t\right)\right)$$

Period = 10 ms, oscillates between $0$ and $2 I_{mean}$.

**Random input** (stochastic)

$$I(t) = I_{mean}\left(1 + 0.1\sqrt{\frac{t_{max}}{\Delta t}}\,\xi(t)\right)$$

where $\xi(t) \sim \mathcal{U}(-1, 1)$.

The scaling factor $0.1\sqrt{t_{max}/\Delta t}$ controls noise amplitude.

**Programming difficulty**: the factor $0.1\sqrt{t_{max}/\Delta t}$ is **not** arbitrary — it ensures the noise has the same statistical power regardless of step size. This is a common pattern in stochastic simulations.

---

### From Scalar to Vector to Matrix

The tutorial deliberately progresses through three levels of representation:

| Level         | Storage                       | Loop                                | Performance       |
| ------------- | ----------------------------- | ----------------------------------- | ----------------- |
| **Scalar**    | `v = el`                      | `for step: v = v + …`             | Slow, one neuron  |
| **1-D array** | `v_n = [el]*n`                | `for j: v_n[j] = v_n[j] + …`      | Medium, n neurons |
| **2-D array** | `v_n = el*np.ones([n,steps])` | `v_n[:,step] = v_n[:,step-1] + …` | Fast, vectorized  |

Each level simulates the **same LIF neuron** — the difference is how we organize data in memory and how we write the loop.

---

### NumPy Vectorization

The 2-D version `v_n[:,step] = …` operates on **all neurons at once** — this is NumPy vectorization and is orders of magnitude faster than Python loops.

**Why?** Python `for` loops interpret each iteration at runtime. NumPy delegates the entire column operation to optimized C/Fortran code — one function call instead of $n$ loop iterations.

**Shape convention**: `v_n[j, step]` = membrane potential of neuron $j$ at time step $step$.

```
v_n.shape = (n_neurons, n_steps)
         step 0   step 1   step 2   ...
neuron 0 [ -60      -58      -55    ... ]
neuron 1 [ -60      -59      -57    ... ]
neuron 2 [ -60      -57      -53    ... ]
```

**Key indexing**: `v_n[j, step]` → scalar, `v_n[j, :]` → one neuron's trace, `v_n[:, step]` → all neurons at one time. The last form is the vectorized update target.

---

### Sample Statistics over N Realizations

With $N$ independent neurons receiving different random inputs:

**Sample mean**:

$$\langle V(t)\rangle = \frac{1}{N}\sum_{n=1}^N V_n(t)$$

**Sample variance** (with Bessel's correction):

$$\text{Var}(t) = \frac{1}{N-1}\sum_{n=1}^N \left(V_n(t) - \langle V(t)\rangle\right)^2$$

**Standard deviation**: $\sigma(t) = \sqrt{\text{Var}(t)}$

We plot $\sigma$ (not variance) because it has the same units as $V$ (millivolts), making it visually comparable.

---

### Computing Statistics in NumPy

With $N$ neurons stored as `v_n` of shape `(n, steps)`:

```python
v_mean = np.mean(v_n, axis=0)   # shape: (steps,)
v_std  = np.std(v_n, axis=0)    # shape: (steps,)
```

`axis=0` means "collapse the neuron dimension" — each element is a time step.

**Why `axis=0`?** The array is `(n_neurons, n_steps)`. Axis 0 is the neuron axis. `np.mean(…, axis=0)` averages across neurons, leaving one value per time step.

**Visualization**: plot the mean as a bold line, and $\pm\sigma$ as a shaded band:

```python
plt.plot(t_range, v_mean, 'C0', label='mean')
plt.plot(t_range, v_mean + v_std, 'C7', label='mean ± std')
plt.plot(t_range, v_mean - v_std, 'C7')
```

---

## W0D2: Spikes, Refractory Period & Code Organization

---

### Adding Spikes to LIF

The reset condition — when $V \geq V_{th}$, record a spike and reset.

---

### What is a Boolean Array?

A comparison on an array returns an array of `True`/`False`:

```python
v = np.array([-55, -48, -60, -50, -45])   # 5 neurons
vth = -50                                   # threshold
spiked = (v >= vth)
# Result: [False,  True, False,  True,  True]
#              ↑          ↑          ↑
#         -55<-50   -48>=-50   -45>=-50
```

Then **use the boolean array as an index** to select only the True elements:

```python
v[spiked]            # array([-48, -50, -45])  — values that crossed threshold
v[spiked] = -70      # reset ONLY those neurons to V_reset
```

**The two-line spike logic** (applied to all $n$ neurons at once):

```python
spiked = (v_n[:, step] >= vth)   # which neurons spiked?
v_n[spiked, step] = vr           # reset only those to V_reset
```

No `for` loop needed — NumPy handles all $n$ neurons in one operation.

---

### Storing Spike Data

**Two ways to record spikes**:

**Dictionary of lists**

```python
spikes = {j: [] for j in range(n)}
# When neuron j spikes:
spikes[j] += [t]
```

Stores actual spike times. Flexible but slower.

**Binary raster array**

```python
raster = np.zeros([n, steps])
# When neurons spike:
raster[spiked, step] = 1.
```

Efficient grid: 1 = spike, 0 = no spike.

Plot with `plt.scatter(times, neuron_ids)` or `plt.imshow(raster)`.

---

### Boolean Indexing — The Critical Optimization

**Without boolean indexing** (slow, Exercise 2):

```python
for j in range(n):
    if v_n[j, step] >= vth:
        v_n[j, step] = vr
        spikes[j] += [t]
        spikes_n[step] += 1
```

**With boolean indexing** (fast, Exercise 3-4):

```python
spiked = (v_n[:, step] >= vth)   # one vectorized comparison
v_n[spiked, step] = vr           # one vectorized assignment
for j in np.where(spiked)[0]:    # only loop over actual spikes
    spikes[j] += [t]
```

**Why this matters**: with $n = 500$ neurons and $150$ time steps, the inner loop runs $500 \times 150 = 75{,}000$ times. Boolean indexing reduces this to iterating only over neurons that actually spiked — typically a small fraction.

---

### Refractory Period

After a spike, clamp $V = V_{reset}$ for duration $t_{ref}$:

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

**Order of operations is critical**:

1. Euler integration → compute $V$
2. Detect spikes ($V \geq V_{th}$) → reset, record
3. Apply refractory clamp → override $V$ for refractory neurons
Doing step 3 before step 2 would incorrectly clamp neurons that just spiked.
**Random refractory periods**: $t_{ref} = \mu + \sigma\,\mathcal{N}(0,1)$, clipped to $\geq 0$.

---

### Code Refactoring: Functions and Classes

The same simulation logic is reorganized three ways:

**Raw loop**

All logic inline:

```python
for step in range(step_end):
    v_n[:,step] = v_n[:,step-1] \
      + dt/tau*(el - v_n[:,step-1] \
      + r*i[:,step])
    spiked = (v_n[:,step] >= vth)
    v_n[spiked,step] = vr
```

Hard to reuse or test.

**Functions**

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

Reusable, testable.

**Class**

```python
class LIFNeurons:
    def __init__(self, n, ...):
        self.v = el*np.ones(n)
        self.last_spike = ...
    def ode_step(self, dt, i):
        self.v = self.v + ...
        self.spiked = (self.v>=vth)
```

Stateful, encapsulated.

---

### The LIFNeurons Class — Key Attributes

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

| Attribute         | Type              | Meaning                       |
| ----------------- | ----------------- | ----------------------------- |
| `self.v`          | array `(n,)`      | Current membrane potential    |
| `self.spiked`     | bool array `(n,)` | Which neurons just spiked     |
| `self.last_spike` | array `(n,)`      | Time of last spike per neuron |
| `self.t_ref`      | array `(n,)`      | Refractory period per neuron  |
| `self.t`          | float             | Simulation clock              |
| `self.steps`      | int               | Number of integration steps   |

---

## W0D3: Linear Algebra (Vectors & Matrices)

---

### Vectors — Dot Product and Geometry

The **dot product** connects algebra to geometry:

$$\mathbf{x} \cdot \mathbf{y} = \sum_i x_i y_i = \|\mathbf{x}\|\,\|\mathbf{y}\|\cos\theta$$

**Vector length**: $\|\mathbf{x}\| = \sqrt{\sum_i x_i^2}$;

**Unit vector**: $\tilde{\mathbf{x}} = \mathbf{x} / \|\mathbf{x}\|$

**Neural application**: a single neuron's firing rate is a dot product: $y = \mathbf{w} \cdot \mathbf{r} = \sum_i w_i r_i$

where $\mathbf{w}$ is the weight vector and $\mathbf{r}$ is the input firing-rate vector.

**Linear combination**: $\mathbf{y} = \sum_i \alpha_i \mathbf{b}_i$

The **span** of a set of vectors is all possible linear combinations. If vectors are **linearly independent**, they form a **basis** for their span.

**Code**: `np.dot(x, y)`, `np.linalg.norm(x)`

---

### Matrices — Linear Transformations

A matrix $W$ represents a linear transformation $\mathbf{y} = W\mathbf{x}$:

$$\begin{bmatrix} y_1 \\ y_2 \end{bmatrix} = \begin{bmatrix} W_{11} & W_{12} \\ W_{21} & W_{22} \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}$$

**Key operations**:

| Operation             | Formula                                          | Code                       |
| --------------------- | ------------------------------------------------ | -------------------------- |
| Matrix-vector         | $\mathbf{y} = W\mathbf{x}$                       | `y = W @ x`                |
| Matrix inverse        | $W^{-1}W = I$                                    | `W_inv = np.linalg.inv(W)` |
| Matrix multiplication | $C_{ij} = \text{row}_i(A) \cdot \text{col}_j(B)$ | `C = A @ B`                |

**Neural application**: a weight matrix maps a population of input neurons to output neurons:

$$\mathbf{r}_{out} = W \mathbf{r}_{in}$$

The **rank** of $W$ reveals the intrinsic dimensionality of the mapping.

---

### Eigenvalues and Eigenvectors

The eigenvalue equation $W\mathbf{v} = \lambda\mathbf{v}$ finds directions that don't rotate:

- $\mathbf{v}$: eigenvector — direction preserved by $W$
- $\lambda$: eigenvalue — scaling factor along that direction
> [!tip] Intuition
> if $\lambda > 1$, the transformation stretches along $\mathbf{v}$; if $0 < \lambda < 1$, it compresses; if $\lambda < 0$, it flips.
**Code**:

```python
eigenvalues, eigenvectors = np.linalg.eig(W)
```

**In neuroscience**: eigenvalues of a connectivity matrix $W$ determine the stability of network dynamics. If all $|\lambda| < 1$, perturbations decay (stable). If any $|\lambda| > 1$, perturbations grow (unstable). This is central to W2D3 (Linear Dynamical Systems).

---

### `np.dot` vs `*` vs `@`

| Operator          | Semantic        | 1-D             | 2-D                   | 3-D+             |
| ----------------- | --------------- | --------------- | --------------------- | ---------------- |
| `*`               | Element-wise    | $x_i \cdot y_i$ | $A_{ij} \cdot B_{ij}$ | Element-wise     |
| `np.dot`          | Dot/product     | Scalar (inner)  | Matrix multiply       | Axis contraction |
| `@` / `np.matmul` | Matrix multiply | Not supported   | Matrix multiply       | Batch matmul     |

**Practical advice**: use `@` for matrix multiplication, `*` for element-wise, `np.dot` for 1-D inner products.

**Common mistake**: using `*` when you mean `@`.

```python
# Wrong: element-wise, not matrix multiply
C = A * B
# Right: matrix multiply
C = A @ B
```

---

## W0D4: Calculus (Differentiation & Integration)

---

### Numerical Differentiation

**Finite difference** — approximate the derivative:

$$f'(a) \approx \frac{f(a+h) - f(a)}{h}$$

**Central difference** (more accurate):

$$f'(a) \approx \frac{f(a+h) - f(a-h)}{2h}$$

**Code with SymPy** (symbolic, exact):

```python
import sympy as sp
t = sp.Symbol('t')
f = t * sp.exp(-t / tau)
df = sp.diff(f, t)              # exact derivative
integral = sp.integrate(f, t)   # exact integral
```

**Why symbolic matters**: for the alpha function $f(t) = t\,e^{-t/\tau}$, the derivative tells us when the peak occurs ($t = \tau$) and the gain tells us the neuron's sensitivity.

---

### Differentiation Rules

**Product Rule**

$$\frac{d}{dt}[u \cdot v] = v\frac{du}{dt} + u\frac{dv}{dt}$$

Example: $\frac{d}{dt}[t \cdot e^{-t/\tau}] = e^{-t/\tau} + t \cdot (-\frac{1}{\tau})e^{-t/\tau}$

**Chain Rule**

$$\frac{dr}{da} = \frac{dr}{dt} \cdot \frac{dt}{da}$$

Example: if $r = \sigma(V)$ and $V = RI$, then $\frac{dr}{dI} = \sigma'(V) \cdot R$

**Partial Derivatives**

$$\frac{\partial f}{\partial x_1}\bigg|_{x_2 \text{ fixed}}$$

For $f(x_1, x_2) = x_1^2 x_2$: $\frac{\partial f}{\partial x_1} = 2x_1 x_2$

**Chain rule is the mathematical foundation of backpropagation** (W1D5). When a loss $L$ depends on weights $\mathbf{w}$ through intermediate variables, the chain rule connects $\frac{\partial L}{\partial \mathbf{w}}$ through the entire computation graph.

---

### Neural Transfer Functions

The **sigmoid** (logistic) transfer function maps input current to firing rate:

$$\sigma(x; a, \theta) = \frac{1}{1 + e^{-a(x - \theta)}}$$

**Gain** = derivative of the transfer function:

$$g = \frac{d\sigma}{dx} = a\,\sigma(1 - \sigma)$$

| Parameter | Meaning   | Effect                                        |
| --------- | --------- | --------------------------------------------- |
| $a$       | Steepness | Larger $a$ → sharper transition               |
| $\theta$  | Threshold | Shifts the curve along x-axis                 |
| $g$       | Gain      | Maximum at $x = \theta$, where $\sigma = 0.5$ |

**Biological interpretation**: gain determines how sensitive a neuron is to small input changes near threshold.

---

### Numerical Integration

**Riemann sum** — approximate the integral:

$$\int_a^b f(x)\,dx \approx \sum_{i} f(x_i)\,\Delta x$$

**Code with `np.cumsum`**:

```python
dx = 0.01
x = np.arange(0, 10, dx)
y = np.sin(x)
cumulative_integral = np.cumsum(y) * dx
```

`np.cumsum` returns running sums: $[y_0,\; y_0{+}y_1,\; y_0{+}y_1{+}y_2,\; \ldots]$

Multiplied by $\Delta x$, this gives the accumulated area under the curve at each point.

**Differentiation as high-pass, integration as low-pass**: derivatives amplify fast changes (high frequencies); integrals smooth them out (accumulate slow trends). This is fundamental to signal processing in neuroscience.

---

## W0D5: Probability & Statistics

---

### Discrete Distributions

**Binomial Distribution**

$$P(k \mid n, p) = \binom{n}{k} p^k (1-p)^{n-k}$$

$k$ successes in $n$ independent trials, each with probability $p$.

```python
samples = np.random.binomial(n=10, p=0.5, size=1000)
```

**Poisson Distribution**

$$P(k \mid \lambda) = \frac{\lambda^k e^{-\lambda}}{k!}$$

Number of events in a fixed interval, rate $\lambda$.

```python
samples = np.random.poisson(lam=5, size=1000)
```

**Limit of Binomial**: when $n \to \infty$, $p \to 0$, $np = \lambda$.

---

### Continuous Distributions

**Gaussian (Normal) Distribution**

$$f(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

```python
samples = np.random.normal(mu=0, sigma=1, size=1000)
from scipy.stats import norm
pdf = norm.pdf(x, loc=0, scale=1)
```

**Uniform Distribution**

$$f(x \mid a, b) = \frac{1}{b-a} \quad \text{for } x \in [a,b]$$

```python
samples = np.random.uniform(-1, 1, size=1000)
```

Used extensively in W0D1-W0D2 for random input current: $\xi(t) \sim \mathcal{U}(-1,1)$.

---

### Poisson Spiking Model

Neural spike trains are commonly modeled as a **Poisson process**:

| Level           | Variable                   | Distribution                     |
| --------------- | -------------------------- | -------------------------------- |
| Single trial    | Spike count in window      | Poisson($\lambda \Delta t$)      |
| Across trials   | Spike count variability    | Poisson — Fano factor = 1        |
| Continuous time | Inter-spike intervals      | Exponential($\lambda$)           |
| Population      | Firing rate across neurons | Gaussian (Central Limit Theorem) |

**Fano factor**: $\text{FF} = \frac{\text{Var}(\text{spike count})}{\text{Mean}(\text{spike count})}$

- FF = 1 for Poisson (variance = mean)
- FF < 1 for more regular spiking
- FF > 1 for bursty spiking

---

### Poisson Spiking: Code

```python
# Generate Poisson spike counts
spike_counts = np.random.poisson(lam=rate * duration, size=n_trials)
# Compute inter-spike intervals
isis = np.diff(spike_times)
# Coefficient of variation (irregularity measure)
cv_isi = np.std(isis) / np.mean(isis)
```

- `np.diff` computes consecutive differences: `[t1-t0, t2-t1, t3-t2, …]`
- `cv_isi = 1` for Poisson (exponential ISI), `< 1` for regular, `> 1` for bursty

---

### Histogram as Density Estimator

```python
plt.hist(data, bins=50, density=True, histtype='stepfilled')
```

Setting `density=True` normalizes the histogram so it integrates to 1 — comparable to a PDF.

**Connection to distributions**: a histogram of Poisson spike counts should look like a Poisson PMF; a histogram of membrane potentials (many neurons) should look Gaussian (Central Limit Theorem).

**W0D1 application**: plotting histograms of $V(t)$ at $t = t_{max}/10$ and $t = t_{max}$ shows how the distribution evolves — it spreads out over time as neurons receive different random inputs.

---

## Cross-Cutting Themes

---

### The Euler Integration Pattern

This single pattern appears in **every simulation notebook** (W0D1, W0D2, W2D3, W2D4, W2D5):

```python
for step in range(step_end):
    t = step * dt
    # 1. Compute input at this time step
    # 2. Euler update: x_new = x_old + dt * f(x_old, t)
    # 3. Apply constraints (threshold, reset, clamp)
```

| Notebook | $f(x,t)$                         | Constraints                    |
| -------- | -------------------------------- | ------------------------------ |
| W0D1     | $\frac{1}{\tau}(E_L - V + RI)$   | None (no spikes)               |
| W0D2     | Same                             | Spike reset + refractory clamp |
| W2D3     | $Ax$ (linear system)             | None                           |
| W2D4     | Same as LIF + noise              | Spike + refractory             |
| W2D5     | $\frac{1}{\tau}(-r + F(wr + I))$ | Rate clipping                  |

---

### Boolean Indexing Pattern

Replace explicit loops with vectorized masks — appears in W0D2 and every subsequent notebook:

```python
# Instead of:
for j in range(n):
    if condition[j]:
        v[j] = value
# Use:
mask = condition          # boolean array
v[mask] = value           # vectorized assignment
```

**Common boolean operations in neuroscience**:

```python
spiked = (v >= vth)                    # threshold detection
clamped = (t - last_spike < t_ref)     # refractory check
in_range = (t_start <= t) & (t <= t_end)  # time window
```

The `&` (and), `|` (or), `~` (not) operators work element-wise on boolean arrays. Use parentheses: `(a > 1) & (b < 2)`.

---

### The Normal Equations

The least-squares solution appears across W0D3, W1D2, W1D3:

$$\hat{\boldsymbol{\theta}} = (X^T X)^{-1} X^T \mathbf{y}$$

**Where it appears**:

| Context                    | $X$              | $\mathbf{y}$ | $\hat{\boldsymbol{\theta}}$ |
| -------------------------- | ---------------- | ------------ | --------------------------- |
| Linear regression (W1D2)   | Feature matrix   | Labels       | Model weights               |
| Linear-Gaussian GLM (W1D3) | Design matrix    | Spike counts | Stimulus filter             |
| Population decoding        | Neural responses | Stimulus     | Decoder weights             |

**Code**:

```python
theta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
# Or more numerically stable:
theta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
```

---

### Deriving the Normal Equations

**Goal**: find $\hat{\boldsymbol{\theta}}$ that minimizes the mean squared error.

**Step 1 — Define the objective**:

$$L(\boldsymbol{\theta}) = \frac{1}{N} \|\mathbf{y} - X\boldsymbol{\theta}\|^2 = \frac{1}{N}(\mathbf{y} - X\boldsymbol{\theta})^T(\mathbf{y} - X\boldsymbol{\theta})$$

- **Loss** $L$ measures the gap between predictions $X\theta$ and ground truth $\mathbf{y}$ — minimize it to find the best fit.
- **Norm squared**: $\|\mathbf{v}\|^2 = \mathbf{v}^T\mathbf{v} = v_1^2 + \cdots + v_n^2$
**Step 2 — Expand**:

$$L = \frac{1}{N}\left(\mathbf{y}^T\mathbf{y} - 2\boldsymbol{\theta}^T X^T\mathbf{y} + \boldsymbol{\theta}^T X^T X \boldsymbol{\theta}\right)$$

Uses transpose properties — see next page.

---

**Step 3 — Take the gradient and set to zero**:

$$\frac{\partial L}{\partial \boldsymbol{\theta}} = \frac{1}{N}\left(-2X^T\mathbf{y} + 2X^T X\boldsymbol{\theta}\right) = 0$$

**Step 4 — Solve**:

$$X^T X\hat{\boldsymbol{\theta}} = X^T\mathbf{y} \quad \Longrightarrow \quad \hat{\boldsymbol{\theta}} = (X^T X)^{-1} X^T \mathbf{y}$$

$X^T\mathbf{y}$ is the projection of $\mathbf{y}$ onto the column space of $X$; $(X^TX)^{-1}$ maps it back to parameter space.

---

### Derivation Details

**Transpose properties** used in Step 2:

| Property | Rule                               |
| -------- | ---------------------------------- |
| Product  | $(AB)^T = B^TA^T$ — order reverses |
| Sum      | $(A+B)^T = A^T + B^T$              |

**Full expansion of Step 2**:

$$\|\mathbf{y} - X\theta\|^2 = (\mathbf{y} - X\theta)^T(\mathbf{y} - X\theta)$$

$$= \mathbf{y}^T\mathbf{y} - \mathbf{y}^TX\theta - (X\theta)^T\mathbf{y} + (X\theta)^TX\theta$$

$$= \mathbf{y}^T\mathbf{y} - 2\theta^TX^T\mathbf{y} + \theta^TX^TX\theta$$

The cross terms merge because $\mathbf{y}^TX\theta$ is a scalar, so $\mathbf{y}^TX\theta = (\mathbf{y}^TX\theta)^T = \theta^TX^T\mathbf{y}$.

---

**Matrix calculus rules** used in Step 3:

| Derivative                                                                         | Result        |
| ---------------------------------------------------------------------------------- | ------------- |
| $\frac{\partial}{\partial \theta} \theta^T X^Ty$                                   | $X^T y$       |
| $\frac{\partial}{\partial \theta} \theta^T X^TX\theta = (X\theta)^2=X^TX \theta^2$ | $2X^TX\theta$ |

**When does $(X^TX)^{-1}$ exist?**

- When columns of $X$ are **linearly independent** (full column rank)
- In practice: use `np.linalg.lstsq` — handles rank-deficient cases via SVD
- Equivalent to solving the linear system $X^TX\hat{\boldsymbol{\theta}} = X^T\mathbf{y}$ directly

---

### Summary

### Programming

- **Euler integration** loop pattern
- **NumPy vectorization**: scalar → 1-D → 2-D
- **Boolean indexing** for spike detection
- **`np.cumsum`** for numerical integration
- **SymPy** for symbolic calculus
- **Refactoring**: loop → function → class

### Modeling

- **LIF neuron**: membrane equation + reset
- **Refractory period**: post-spike clamp
- **Transfer function**: sigmoid, gain
- **Poisson spiking**: spike counts, ISI, CV
- **Linear transformation**: $W\mathbf{x}$, eigenvalues

### Calculation

- **Euler discretization** of ODEs
- **Dot product**, matrix multiply
- **Finite difference** derivatives
- **Riemann sum** integration
- **Chain rule** for gradients
- **Binomial, Poisson, Gaussian** distributions

---

## Exercises

---

### Exercise 1: Implement `cumsum`

Implement a function `my_cumsum(a)` that returns the cumulative sum of a list, without using `np.cumsum`.

**Definition**: $\text{cumsum}[i] = \sum_{j=0}^{i} a[j]$

**Example**: `my_cumsum([1, 2, 3, 4])` → `[1, 3, 6, 10]`

```python
def my_cumsum(a):
    result = []
    total = 0
    for x in a:
        ...  # update total and append to result
    return result
```

**Challenge**: implement the 2-D version `my_cumsum2d(a, axis)` that sums along `axis=0` (rows) or `axis=1` (columns).

---

### Exercise 2: Implement `diff`

Implement a function `my_diff(a)` that returns consecutive differences, without using `np.diff`.

**Definition**: $\text{diff}[i] = a[i+1] - a[i]$, $\text{diff}[0]=a[0]$

**Example**: `my_diff([1, 3, 6, 10])` → `[1, 2, 3, 4]`

Note: output length is `len(a)`.

```python
def my_diff(a):
    result = [a[0]]                        # diff[0] = a[0]
    for i in range(...):
        ...  # compute a[i+1] - a[i] and append
    return result
```

**Application**: `my_diff(spike_times)` gives inter-spike intervals (ISI), used to compute `cv_isi = std(ISI) / mean(ISI)` in W0D5.

---

### Exercise 3: Running Mean

Implement `running_mean(a)` where each element at position $i$ is the mean of the first $i+1$ elements.

**Definition**: $\text{result}[i] = \frac{1}{i+1}\sum_{j=0}^{i} a[j]$

**Example**: `running_mean([2, 4, 6, 8])` → `[2.0, 3.0, 4.0, 5.0]`

```python
def running_mean(a):
    result = []
    total = 0
    for i, x in enumerate(a):
        total += x
        result.append(total / (i + 1))
    return result
```

**Challenge**: rewrite the function **without** using `total` — only `a` and `result` are allowed.

**Hint**: think about how `result[-1]` relates to the previous cumulative sum. Can you recover `total` from it?

**Connection**: `running_mean(a) = cumsum(a) / [1, 2, 3, …, n]` — a normalized cumulative sum, useful for tracking average firing rate over time.

---

### Exercise 4: Softmax — Definition & Basic Implementation

**Softmax** converts logits into a probability distribution:

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

**Properties**: all outputs positive, sum to 1, preserves ordering.

**Used in**: classification output layers, attention mechanisms, policy networks.

**Basic 2-pass implementation** (no numerical stability):

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

**Problem**: $e^{z_i}$ overflows for large $z_i$ (e.g., $e^{1000} = \infty$). Standard fix: subtract the max first.

---

### Exercise 4 (cont.): Online Softmax

**Online softmax** computes softmax in a single pass — for streaming data where the full vector is unavailable.

**Idea**: maintain a running maximum $m$ and running sum $s$:

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

**Hint**: when a new max $m_{\text{new}}$ arrives, all previous exponentials were relative to $m_{\text{old}}$. What factor brings them into the new reference frame?

---

### Exercise 5: Monte Carlo Estimation

**Goal**: estimate $\int_{-\infty}^{\infty} \mathcal{N}(x \mid 0, 1)\,dx \approx 1$ using random sampling.

**Monte Carlo principle**: sample $x_i \sim \mathcal{U}(a,b)$, then:

$$\int_a^b f(x)\,dx \approx \frac{b-a}{N}\sum_{i=1}^{N} f(x_i)$$

```python
def gaussian(x, mu=0, sigma=1):
    # ...
def monte_carlo_gaussian(N, a=-10, b=10):
    # ...
```

---

### Exercise 5 (cont.): Convergence

```python
for N in [100, 1000, 10000, 100000]:
    est = monte_carlo_gaussian(N)
    print(f"N={N:>6d}  estimate={est:.6f}  error={abs(est-1.0):.6f}")
```

| N       | Estimate | Error |
| ------- | -------- | ----- |
| 100     | 0.987    | 0.013 |
| 1,000   | 1.002    | 0.002 |
| 10,000  | 0.999    | 0.001 |
| 100,000 | 1.000    | 0.000 |

**Convergence rate**: error $\propto 1/\sqrt{N}$ — 10x more accuracy needs 100x more samples. Slow!

**Problem**: most samples land where $\mathcal{N}(x) \approx 0$ — wasted effort in low-density regions.
