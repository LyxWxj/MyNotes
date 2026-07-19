# Neuromatch Notebooks — Week 2

Linear Systems · Biological Neuron Models · Dynamical Systems

---

## Overview

Week 2 focuses on **dynamical systems and neural models** — from linear systems to biological neuron models to network dynamics:

| Day      | Topic                      | Core Skills                               |
| -------- | -------------------------- | ---------------------------------------- |
| **W2D3** | Linear Systems             | Euler integration, Oscillations, Random walks, OU process, Autoregressive models |
| **W2D4** | Biological Neuron Models   | LIF neuron, Conductance synapses, Short-term plasticity (STP), Spike-timing dependent plasticity (STDP)               |
| **W2D5** | Dynamical Systems          | Firing rate models, Wilson-Cowan model, Phase plane analysis, Jacobian matrix, Limit cycles |

**Unifying theme**: How do neurons and networks evolve over time, and how can we model their dynamics mathematically?

---

## W2D3: Linear Systems

---

### Tutorial 1: One-Dimensional Differential Equations

The simplest dynamical system: $\dot{x} = ax$

**Analytical solution**: $x(t) = x_0 e^{at}$

| $a$                  | Behavior                        |
| -------------------- | ------------------------------- |
| $a < 0$              | Exponential decay → 0           |
| $a > 0$              | Exponential growth → ∞          |
| $a = \text{complex}$ | Oscillation with growth/decay |

**Forward Euler integration** (numerical solution):

$$x(t_i) = x(t_{i-1}) + \dot{x}(t_{i-1}) \cdot dt$$

For $\dot{x} = ax$ specifically: $x[k] = x[k-1] + a \cdot x[k-1] \cdot dt$

**Implementation detail**: Use `dtype=complex` to handle complex-valued $a$ (needed for oscillatory dynamics)

---

### Tutorial 1: Complex $a$ and Oscillatory Dynamics

When $a$ is complex ($a = \text{real} + i \cdot \text{imag}$), the system oscillates:

$$x(t) = x_0 e^{(\text{real} + i \cdot \text{imag})t} = x_0 e^{\text{real} \cdot t} \cdot [\cos(\text{imag} \cdot t) + i \sin(\text{imag} \cdot t)]$$

**Key insight**:
- **Real part** → growth/decay rate (amplitude envelope)
- **Imaginary part** → oscillation frequency

**Stable oscillation condition**: Set real part = 0, imaginary part = $2\pi f$

Example: Producing a 0.5 Hz stable oscillation → imaginary part = $2\pi \times 0.5 = \pi \approx 3.14$

**Growing oscillation condition**: real part > 0 AND imaginary part ≠ 0

---

### Tutorial 1: Two-Dimensional Linear Systems

Extension to 2D: $\dot{\mathbf{x}} = \mathbf{A}\mathbf{x}$

$$\begin{bmatrix} \dot{x}_1 \\ \dot{x}_2 \end{bmatrix} = \begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}$$

**Numerical solution**: Uses `scipy.integrate.solve_ivp` (not manual Euler in 2D)

**Stream plots**: Compute $\mathbf{A}\mathbf{x}$ at each grid point; arrows show direction of state change

**Eigenvectors**: Directions where $\mathbf{A}\mathbf{x}$ is parallel to $\mathbf{x}$ (invariant directions)

**Eigenvalues**: Factor by which $\mathbf{A}\mathbf{x}$ is stretched/shrunk along eigenvector directions

**Stability classification**:

| Eigenvalue Type | Behavior |
|----------------|----------|
| Both negative real | Stable node (converge to origin) |
| Both positive real | Unstable node (diverge) |
| Opposite signs | Saddle point |
| Complex | Oscillation / Spiral |

---

### Tutorial 2: Markov Processes

**Markov property**: Present state entirely determines the transition to the next state (memoryless)

**Telegraph process**: Two-state ion channel model

- States: Closed (0) and Open (1)
- Transition probabilities: $P(0 \to 1 | x=0) = \mu_{c2o}$, $P(1 \to 0 | x=1) = \mu_{o2c}$

**Poisson process properties**:
1. Event probability is independent of all other events
2. Average rate of events is constant within a given time period
3. Two events cannot occur simultaneously

**State transition matrix**:

$$\begin{bmatrix} C \\ O \end{bmatrix}_{k+1} = \begin{bmatrix} 1-\mu_{c2o} & \mu_{o2c} \\ \mu_{c2o} & 1-\mu_{o2c} \end{bmatrix} \begin{bmatrix} C \\ O \end{bmatrix}_k$$

- Each column sums to 1 (conservation of probability)
- Matrix entries:
  - $1 - \mu_{c2o}$: probability closed stays closed
  - $\mu_{c2o}$: probability closed transitions to open
  - $\mu_{o2c}$: probability open transitions to closed
  - $1 - \mu_{o2c}$: probability open stays open

**Probability propagation algorithm**: $\mathbf{x}_{k+1} = \mathbf{A} \cdot \mathbf{x}_k$ (matrix-vector multiply)

**Equilibrium analysis**:
- Eigenvalue = 1 corresponds to the **stable equilibrium** eigenvector
- Other eigenvalues correspond to transient decay
- Equilibrium eigenvector must be normalized (elements sum to 1)
- Equilibrium probability of being Open: $\frac{\mu_{c2o}}{\mu_{c2o} + \mu_{o2c}}$

---

### Tutorial 3: Random Walks and Diffusion

**Random walk**: At each step, move $\Delta x = \pm 1$ with equal probability

**Position update**: $x_{k+1} = x_k + \Delta x$

**Gaussian step random walk**: Steps drawn from $\mathcal{N}(\mu, \sigma)$

**Efficient vectorized implementation**:

```python
def random_walk_simulator(N, T, mu=0, sigma=1):
    steps = np.random.normal(mu, sigma, size=(N, T))
    sim = np.cumsum(steps, axis=1)
    return sim
```

**Diffusive process properties**:
- Mean stays near 0 (independent of time)
- **Variance grows linearly with time**: $\text{Var} \propto t$ (specifically $\text{Var} = \sigma^2 t$)
- Distribution widens over time but center remains unchanged

---

### Tutorial 3: Deterministic Decay and OU Process

**Basic decay**: $x_{k+1} = \lambda x_k$, solution: $x_k = x_0 \lambda^k$ (decays when $|\lambda| < 1$)

**Decay with target**: $x_{k+1} = x_\infty + \lambda(x_k - x_\infty)$

**Analytical solution**: $x_k = x_\infty(1 - \lambda^k) + x_0 \lambda^k$

As $k \to \infty$: $x_k \to x_\infty$

**Ornstein-Uhlenbeck (OU) process / Drift-Diffusion Model**:

$$x_{k+1} = x_\infty + \lambda(x_k - x_\infty) + \sigma \eta$$

where $\eta \sim \mathcal{N}(0,1)$ (standard normal)

**Two components**:
- **Drift**: $x_\infty + \lambda(x_k - x_\infty)$, pulls $x$ toward $x_\infty$
- **Diffusion**: $\sigma \eta$, adds random noise

**Equilibrium variance** (key result):

$$\text{Var}_{eq} = \frac{\sigma^2}{1 - \lambda^2}$$

**Properties**:
- Depends only on $\lambda$ and $\sigma$, **not** on $x_0$ or $x_\infty$
- As $\lambda \to 1$: variance diverges (approaches pure random walk)
- As $\lambda \to 0$: variance approaches $\sigma^2$ (each step independent)

**Empirical variance computation**: Run long simulation of duration $T$, take variance of the **second half** (assuming system has settled)

```python
x[-round(T/2):].var()
```

**Key observations**:
- Mean of OU process follows the deterministic solution exactly
- Variance reaches equilibrium (unlike random walk where it grows without bound)
- Restoring drift force prevents unbounded variance growth

---

### Tutorial 4: Autoregressive Models

**Perspective shift**: Given data, learn its dynamics (inverse problem)

**First-order autoregression AR(1)**: $x_{k+1} = \lambda x_k + \eta$

**Regression formulation**: $\mathbf{x}_2 = \lambda \mathbf{x}_1$
- $\mathbf{x}_1 = x[0:T-1]$ (past values)
- $\mathbf{x}_2 = x[1:T]$ (future values, shifted by 1)

**Least squares solution**:

```python
p, res, rnk, s = np.linalg.lstsq(x1, x2, rcond=None)
```

**Adding intercept term**: Prepend a column of 1s to x1

```python
x1 = x1[:, np.newaxis]**[0, 1]  # Add columns: constant and linear terms
```

Regression coefficient $p[1]$ is the estimated $\hat{\lambda}$

**Residual analysis**:
- Residuals = data - prediction: $\text{res} = x_2 - (p[0] + \hat{\lambda} \cdot x_1[:, 1])$
- Residual standard deviation should approximately equal $\sigma$ (noise parameter)
- Residual histogram should be approximately normal

---

### Tutorial 4: Higher-Order Autoregressive Models

**Order-$r$ AR model**: $x_{k+1} = \alpha_0 + \alpha_1 x_k + \alpha_2 x_{k-1} + \dots + \alpha_r x_{k-r}$

$r+1$ coefficients to fit (including intercept $\alpha_0$)

**Time-delay matrix construction (build_time_delay_matrices)**:

- $\mathbf{x}_1$: matrix of size $[(r+1) \times (n-r)]$
  - Row 0: all ones (intercept)
  - Row 1: $x[0:T-r]$ (lag 1)
  - Row 2: $x[1:T-r+1]$ (lag 2, achieved via `np.roll`)
  - ... up to lag $r$
- $\mathbf{x}_2$: vector $x[r:]$ (values to predict)

**np.roll trick**: `xprime = np.roll(xprime, -1)` shifts array left by 1 each iteration

**Prediction and classification**:
- For binary (+1/-1) data: prediction = $\text{sign}(\mathbf{x}_1^T \cdot \mathbf{p})$
- Error rate = $\text{count}(x_2 \neq \text{prediction}) / \text{len}(x_2)$
- Random chance baseline: error rate = 0.5

**Overfitting observation**:
- Sweeping AR orders from r=1 to r=20
- There is a **sweet spot** (around r=6 for human-generated data)
- Too low r: underfitting (misses patterns)
- Too high r: overfitting (fits training noise, poor on test)
- Demonstrates bias-variance tradeoff

**Human randomness vs machine randomness**:
- Humans are poor at generating random sequences (detectable patterns)
- AR models can exploit these patterns for better-than-chance predictions
- Machine-generated random integers are truly unpredictable (error ≈ 0.5)
- Binary encoding: '0' → -1, '1' → +1 (via `x*2 - 1`)

**Connections between tutorials**:
- Tutorial 1: Deterministic continuous-time dynamics ($\dot{x} = Ax$)
- Tutorial 2: Discrete-time probabilistic transitions (state transition matrix)
- Tutorial 3: Combining deterministic drift with stochastic diffusion (OU process)
- Tutorial 4: Fitting models to data (inverting the generative process via regression)
- The OU process $x_{k+1} = \lambda x_k + \sigma \eta$ is both the generative model (Tutorial 3) and the model being fit (Tutorial 4), closing the loop

---

## W2D4: Biological Neuron Models

---

### Tutorial 1: Leaky Integrate-and-Fire Model (LIF)

**Core membrane potential equation (subthreshold dynamics)**:

$$\tau_m \frac{dV}{dt} = -(V - E_L) + \frac{I}{g_L}$$

where $\tau_m = C_m / g_L$ is the membrane time constant, $g_L$ is leak conductance, $E_L$ is resting potential

**Spike-and-reset rule**:

$$\text{if } V(t_{sp}) \geq V_{th}: \quad V(t) = V_{reset} \text{ for } t \in (t_{sp}, t_{sp} + \tau_{ref}]$$

**Default parameters**:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| $V_{th}$ | -55 mV | Spike threshold |
| $V_{reset}$ | -75 mV | Reset potential |
| $E_L$ | -75 mV | Resting potential |
| $\tau_m$ | 10 ms | Membrane time constant |
| $g_L$ | 10 nS | Leak conductance |
| $t_{ref}$ | 2 ms | Refractory time |
| $dt$ | 0.1 ms | Time step |

**Euler integration implementation (run_LIF)**:

```python
for it in range(Lt - 1):
    if tr > 0:                          # Refractory period
        v[it] = V_reset
        tr = tr - 1
    elif v[it] >= V_th:                 # Spike!
        rec_spikes.append(it)
        v[it] = V_reset
        tr = tref / dt
    # Calculate the increment of the membrane potential
    dv = (dt / tau_m) * (-(v[it] - E_L) + Iinj[it] / g_L)
    # Update the membrane potential
    v[it + 1] = v[it] + dv
```

---

### Tutorial 1: Different Types of Input Currents

**Direct Current (DC)**: Constant current, produces regular spikes (CV_ISI ≈ 0)

**Gaussian White Noise (GWN)**:

$$I_{gwn} = \mu + \sigma \cdot \frac{\xi(t)}{\sqrt{dt/1000}}$$

where $\xi(t) \sim \mathcal{N}(0,1)$, dividing by $\sqrt{dt/1000}$ converts discrete-time noise to proper continuous-time scaling (units to seconds)

**Ornstein-Uhlenbeck (OU) process (colored noise)**:

$$\tau_\eta \frac{d\eta}{dt} = -\eta(t) + \sigma_\eta \sqrt{2\tau_\eta} \xi(t)$$

**Properties**:
- $\mathbb{E}[\eta(t)] = \mu$
- Autocovariance: $\text{Cov}[\eta(t), \eta(t+\tau)] = \sigma_\eta^2 e^{-|t-\tau|/\tau_\eta}$

**Euler implementation**:

```python
I_ou[it+1] = I_ou[it] + (dt/tau_ou)*(mu - I_ou[it]) + sqrt(2*dt/tau_ou)*sig*noise[it+1]
```

---

### Tutorial 1: Firing Rate and Spike Irregularity

**Frequency-Current curve (F-I curve)**: Output firing frequency as a function of input current

**Coefficient of Variation of ISI (CV_ISI)**:

$$\text{CV}_{\text{ISI}} = \frac{\text{std}(\text{ISI})}{\text{mean}(\text{ISI})}$$

| CV Value | Meaning |
|----------|---------|
| 0 | Perfectly regular (clock-like) |
| 1 | Poisson process (maximum irregularity) |

**Key findings**:
- DC input produces regular spiking (CV ≈ 0)
- GWN input produces irregular spiking; higher $\sigma$ increases CV_ISI
- Increasing $\sigma$ smooths the F-I curve
- Increasing mean $\mu$ while keeping $\sigma$ fixed decreases CV_ISI (more regular at higher rates)

---

### Tutorial 2: Correlated Inputs and Correlation Transfer

**Correlated input model**:

$$\frac{I_i}{g_L} = \mu_i + \sigma_i (\sqrt{1-c}\xi_i + \sqrt{c}\xi_c)$$

where $c \in [0,1]$ controls the fraction of common input, $\xi_i$ is independent noise, $\xi_c$ is shared common noise

**Sample correlation coefficient (Pearson)**:

$$r_{ij} = \frac{\text{cov}(I_i, I_j)}{\sqrt{\text{var}(I_i)} \sqrt{\text{var}(I_j)}}$$

where $\text{cov}(I_i, I_j) = \sum_{k=1}^{L}(I_i^k - \bar{I_i})(I_j^k - \bar{I_j})$

Note: Strict sample covariance should divide by $L-1$, but in the correlation coefficient the $L-1$ factors cancel

**Poisson spike train generator (Poisson_generator)**:

```python
poisson_train = 1.0 * (u_rand < rate * (dt / 1000))
```

Spike probability per bin = $\text{rate} \times dt / 1000$

**Correlated Poisson generation (generate_corr_Poisson)**:
1. Generate a "mother" Poisson train at rate $\lambda/c$
2. Each child independently samples a fraction $c$ of the mother's spikes (via shuffling indices)

**Campbell's theorem (mean and variance of synaptic current from Poisson input)**:

$$\mu_{\rm syn} = \lambda J \int P(t) dt$$

$$\sigma_{\rm syn} = \lambda J \int P(t)^2 dt$$

where $\lambda$ is Poisson rate, $J$ is PSP amplitude, $P(t)$ is postsynaptic current kernel

**Key findings**:
- Output correlation is **always smaller** than input correlation (LIF acts as a "correlation filter")
- Correlation transfer function is approximately linear
- Higher mean $\mu$ and higher $\sigma$ both increase the slope of the transfer function (better correlation transmission)
- Higher firing rates lead to better correlation transfer

---

### Tutorial 3: Conductance-Based Synapses

**Synaptic conductance dynamics**:

$$\frac{dg_{\rm syn}(t)}{dt} = \bar{g}_{\rm syn} \sum_k \delta(t-t_k) - \frac{g_{\rm syn}(t)}{\tau_{\rm syn}}$$

- $\bar{g}_{\rm syn}$: maximum conductance change per spike (synaptic weight)
- $\tau_{\rm syn}$: synaptic time constant (controls decay speed)

**Ohm's law (conductance to current)**:

$$I_{\rm syn}(t) = g_{\rm syn}(t)(V(t) - E_{\rm syn})$$

- $E_E = 0$ mV (excitatory reversal potential, depolarizing)
- $E_I = -80$ mV (inhibitory reversal potential, hyperpolarizing)

**Total synaptic current**:

$$I_{\rm syn} = -g_E(t)(V - E_E) - g_I(t)(V - E_I)$$

**Conductance-based LIF membrane equation**:

$$\tau_m \frac{dV}{dt} = -(V - E_L) - \frac{g_E(t)}{g_L}(V - E_E) - \frac{g_I(t)}{g_L}(V - E_I) + \frac{I_{\rm inj}}{g_L}$$

**Euler update for conductance (run_LIF_cond)**:

```python
gE[it+1] = gE[it] - (dt/tau_syn_E)*gE[it] + gE_bar * spike_train_ex[it+1]
gI[it+1] = gI[it] - (dt/tau_syn_I)*gI[it] + gI_bar * spike_train_in[it+1]
```

**Default synaptic parameters**:
- Excitatory: $g_E = 2.4$ nS, $E_E = 0$ mV, $\tau_E = 2$ ms
- Inhibitory: $g_I = 2.4$ nS, $E_I = -80$ mV, $\tau_I = 5$ ms
- 80 excitatory, 20 inhibitory presynaptic neurons at 10 Hz

**Free Membrane Potential (FMP)**: Membrane potential computed with spike threshold removed (set $V_{th} = \infty$)

- Mean FMP > threshold: **Mean-driven regime** (regular firing, low CV_ISI)
- Mean FMP < threshold: **Fluctuation-driven regime** (irregular firing, high CV_ISI)
- Balance of excitation/inhibition determines firing pattern
- Synaptic input is **colored noise** (exponential kernel filtering), not white noise

---

### Tutorial 3: Short-Term Synaptic Plasticity (STP)

**Three-variable dynamic model**:

$$\frac{du_E}{dt} = -\frac{u_E}{\tau_f} + U_0(1-u_E^-)\delta(t-t_{sp})$$

$$\frac{dR_E}{dt} = \frac{1-R_E}{\tau_d} - u_E^+ R_E^- \delta(t-t_{sp})$$

$$\frac{dg_E}{dt} = -\frac{g_E}{\tau_E} + \bar{g}_E u_E^+ R_E^- \delta(t-t_{sp})$$

**Variable meanings**:

| Variable | Meaning | Range | Decay Constant |
|----------|---------|-------|----------------|
| $u$ | Release probability (usage fraction) | $[0, 1]$ | $\tau_f$ (facilitation time constant) |
| $R$ | Available resource | $[0, 1]$ | $\tau_d$ (depression time constant) |
| $g$ | Postsynaptic conductance | $[0, \bar{g}]$ | $\tau_E$ (synaptic time constant) |

**Physical process**:
```
Spike arrives → u increases (calcium influx)
              → Consumes resources: R decreases
              → Produces conductance: g increases

Between spikes → u decays back to 0 (τ_f)
               → R recovers to 1 (τ_d)
               → g decays (τ_E)
```

**Euler implementation (dynamic_syn)**:

```python
for it in range(Lt - 1):
    # Update u (release probability)
    du = -(dt/tau_f) * u[it] + U0 * (1.0 - u[it]) * pre_spike_train[it+1]
    u[it+1] = u[it] + du
    
    # Update R (resource) - note use of updated u[it+1]
    dR = (dt/tau_d) * (1.0 - R[it]) - u[it+1] * R[it] * pre_spike_train[it+1]
    R[it+1] = R[it] + dR
    
    # Update g (conductance) - note use of updated u[it+1] and R[it]
    dg = -(dt/tau_syn) * g[it] + g_bar * R[it] * u[it+1] * pre_spike_train[it+1]
    g[it+1] = g[it] + dg
```

**Key point**: When a spike arrives, update $u$ first, then use the new $u$ to update $R$ and $g$ (order matters!)

**Short-Term Depression (STD) vs Short-Term Facilitation (STF) parameters**:

| Parameter | STD | STF |
|-----------|-----|-----|
| $U_0$ | 0.5 (high initial release rate) | 0.2 (low initial release rate) |
| $\tau_d$ | 100 ms | 100 ms |
| $\tau_f$ | 50 ms (fast recovery) | 750 ms (slow decay) |

**STD mechanism**:
- At high input rates, resources don't recover fast enough, conductance continuously decreases
- $g_{10}/g_1$ decreases monotonically with input rate

**STF mechanism**:
- When $\tau_f$ is large, $u$ decays slowly between spikes, accumulating effect
- $g_{10}/g_1$ changes non-monotonically with input rate (initially increases, then decreases)

---

### Tutorial 4: Spike-Timing Dependent Plasticity (STDP)

**STDP weight change rule (biphasic exponential decay)**:

$$\Delta W = \begin{cases} A_+ e^{(t_{pre}-t_{post})/\tau_+} & \text{if } t_{post} > t_{pre} \text{ (LTP)} \\ -A_- e^{-(t_{pre}-t_{post})/\tau_-} & \text{if } t_{post} < t_{pre} \text{ (LTD)} \end{cases}$$

where $\Delta t = t_{pre} - t_{post}$. For simplicity, $\tau_+ = \tau_- = \tau_{\rm stdp}$

**Default STDP parameters**:
- $A_+ = 0.008$ (LTP magnitude)
- $A_- = A_+ \times 1.10 = 0.0088$ (LTD magnitude, slightly larger — asymmetric)
- $\tau_{\rm stdp} = 20$ ms

**Trace variables P(t) and M(t) for efficient STDP implementation**:

For each presynaptic neuron $i$:
$$\tau_+ \frac{dP}{dt} = -P$$
On presynaptic spike: $P(t) = P(t) + A_+$

For each postsynaptic neuron:
$$\tau_- \frac{dM}{dt} = -M$$
On postsynaptic spike: $M(t) = M(t) - A_-$

- $P(t)$ is always positive (tracks recent presynaptic activity for LTP)
- $M(t)$ is always negative (tracks recent postsynaptic activity for LTD)

**Euler update for P (generate_P)**:

```python
dP = -(dt/tau_stdp)*P[:,it] + A_plus * spike_train[:,it+1]
P[:,it+1] = P[:,it] + dP
```

**Weight update rules using trace variables**:

When presynaptic neuron $i$ fires (LTD):
$$\bar{g}_i = \bar{g}_i + M(t) \cdot \bar{g}_{max}$$
- $M$ is negative, so weight decreases
- Clamp: if $\bar{g}_i < 0$, set $\bar{g}_i = 0$

When postsynaptic neuron fires (LTP):
$$\bar{g}_i = \bar{g}_i + P_i(t) \cdot \bar{g}_{max} \quad \forall i$$
- $P$ is positive, so weight increases
- Clamp: if $\bar{g}_i > \bar{g}_{max}$, set $\bar{g}_i = \bar{g}_{max}$

**LIF membrane equation with STDP synapses**:

$$\tau_m \frac{dV}{dt} = -(V - E_L) - g_E(t)(V - E_E)$$

where $g_E(t) = \sum_i g_i(t)$, each $g_i(t)$ uses the dynamically updated $\bar{g}_i$

**Default synapse parameters (STDP simulations)**:
- $\bar{g}_E = 0.024$ nS (max conductance per synapse)
- $g_{E,init} = 0.014 - 0.024$ nS (initial conductance)
- $E_E = 0$ mV, $\tau_E = 5$ ms
- $N = 300$ presynaptic neurons at 10-15 Hz, $dt = 1$ ms

**Key findings**:
- With uncorrelated Poisson inputs, many synapses weaken over time (LTD dominates due to $A_- > A_+$)
- Weight distribution evolves over time; bimodal distribution emerges (many weights near 0, some near $g_{max}$)
- With correlated inputs: correlated presynaptic neurons maintain their weights (higher chance of pre-before-post pairing), while uncorrelated synapses depress
- STDP enables **unsupervised learning**: synapses carrying correlated/relevant information are selectively strengthened

---

## W2D5: Dynamical Systems

---

### Tutorial 1: Single Population Firing Rate Model

**Feedforward firing rate dynamics (Eq. 1)**:

$$\tau \frac{dr}{dt} = -r + F(I_{\rm ext})$$

**Sigmoidal transfer function / F-I curve (Eq. 2)**:

$$F(x; a, \theta) = \frac{1}{1 + e^{-a(x-\theta)}} - \frac{1}{1 + e^{a\theta}}$$

- $a$ = gain, $\theta$ = threshold
- The second term ensures $F(0; a, \theta) = 0$

**Implementation**:

```python
def F(x, a, theta):
    f = (1 + np.exp(-a * (x - theta)))**-1 - (1 + np.exp(a * theta))**-1
    return f
```

**Recurrent network dynamics (Eq. 3)**:

$$\tau \frac{dr}{dt} = -r + F(w \cdot r + I_{\rm ext})$$

where $w$ is recurrent synaptic weight (E to E)

**Analytical solution for $w = 0$**:

$$r(t) = r(0) + [F(I_{\rm ext}; a, \theta) - r(0)](1 - e^{-t/\tau})$$

---

### Tutorial 1: Fixed Points and Stability

**Fixed point condition (Eq. 4)**: $r$ value when $\frac{dr}{dt} = 0$

$$-r^* + F(w \cdot r^* + I_{\rm ext}; a, \theta) = 0$$

**Derivative of sigmoid transfer function (Eq. 5)**:

$$\frac{dF}{dx} = a \cdot e^{-a(x-\theta)} \cdot (1 + e^{-a(x-\theta)})^{-2}$$

**Eigenvalue for stability analysis (Eq. 4 in Bonus)**:

$$\lambda = \frac{-1 + w \cdot F'(w \cdot r^* + I_{\rm ext}; a, \theta)}{\tau}$$

| $\lambda$ | Stability |
|-----------|-----------|
| $\lambda < 0$ | Stable (attracting) |
| $\lambda > 0$ | Unstable (repelling) |

**Implementation**:

```python
def eig_single(fp, tau, a, theta, w, I_ext, **other_pars):
    eig = (-1 + w * dF(w * fp + I_ext, a, theta)) / tau
    return eig
```

**Default parameters**: $\tau = 1.0$ ms, $a = 1.2$, $\theta = 2.8$, $w = 0.0$, $I_{\rm ext} = 0.0$, $T = 20$ ms, $dt = 0.1$ ms

---

### Tutorial 1: OU Noise Input

**OU process**:

$$\tau_\eta \frac{d\eta}{dt} = -\eta(t) + \sigma_\eta \sqrt{2\tau_\eta} \xi(t)$$

**Euler update**:

```python
I_ou[it+1] = I_ou[it] + dt/tau_ou * (0 - I_ou[it]) + sqrt(2*dt/tau_ou) * sig * noise[it+1]
```

**Key finding**: When multiple fixed points exist, noisy inputs can drive transitions between fixed points

---

### Tutorial 2: Wilson-Cowan Model

**Two coupled populations (excitatory + inhibitory) (Eq. 1)**:

$$\tau_E \frac{dr_E}{dt} = -r_E + F_E(w_{EE}r_E - w_{EI}r_I + I_E^{\rm ext}; a_E, \theta_E)$$

$$\tau_I \frac{dr_I}{dt} = -r_I + F_I(w_{IE}r_E - w_{II}r_I + I_I^{\rm ext}; a_I, \theta_I)$$

**Euler updates**:

```python
r_E[k+1] = r_E[k] + (dt/τ_E)*(-r_E[k] + F(w_EE*r_E[k] - w_EI*r_I[k] + I_ext_E, a_E, θ_E))
r_I[k+1] = r_I[k] + (dt/τ_I)*(-r_I[k] + F(w_IE*r_E[k] - w_II*r_I[k] + I_ext_I, a_I, θ_I))
```

**Default parameters**:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| $\tau_E$ | 1.0 ms | E population timescale |
| $\tau_I$ | 2.0 ms | I population timescale |
| $a_E$ | 1.2 | E population gain |
| $a_I$ | 1.0 | I population gain |
| $\theta_E$ | 2.8 | E population threshold |
| $\theta_I$ | 4.0 | I population threshold |
| $w_{EE}$ | 9.0 | E→E connection strength |
| $w_{EI}$ | 4.0 | I→E connection strength |
| $w_{IE}$ | 13.0 | E→I connection strength |
| $w_{II}$ | 11.0 | I→I connection strength |

---

### Tutorial 2: Nullclines

**Nullcline definition**: Curves where $\frac{dr_E}{dt} = 0$ or $\frac{dr_I}{dt} = 0$

**E nullcline ($\frac{dr_E}{dt} = 0$, Eq. 2)**:
$$-r_E + F_E(w_{EE}r_E - w_{EI}r_I + I_E^{\rm ext}; a_E, \theta_E) = 0$$

**I nullcline ($\frac{dr_I}{dt} = 0$, Eq. 3)**:
$$-r_I + F_I(w_{IE}r_E - w_{II}r_I + I_I^{\rm ext}; a_I, \theta_I) = 0$$

**Explicit nullcline expressions (Eqs. 4-5)**:

$$\text{E nullcline: } \quad r_I = \frac{1}{w_{EI}}[w_{EE}r_E - F_E^{-1}(r_E; a_E, \theta_E) + I_E^{\rm ext}]$$

$$\text{I nullcline: } \quad r_E = \frac{1}{w_{IE}}[w_{II}r_I + F_I^{-1}(r_I; a_I, \theta_I) - I_I^{\rm ext}]$$

**Inverse transfer function (Eq. 6)**:

$$F^{-1}(x; a, \theta) = -\frac{1}{a} \ln\left[\frac{1}{x + \frac{1}{1+e^{a\theta}}} - 1\right] + \theta$$

**Nullcline properties**:
- E nullcline divides the phase plane into regions where $\frac{dr_E}{dt} > 0$ and $\frac{dr_E}{dt} < 0$
- I nullcline divides the phase plane into regions where $\frac{dr_I}{dt} > 0$ and $\frac{dr_I}{dt} < 0$
- Intersections of the two nullclines are the system's **fixed points**

---

### Tutorial 2: Vector Field

**Vector field definition**: Map of arrows showing $(\frac{dr_E}{dt}, \frac{dr_I}{dt})$ at each point in the phase plane

```python
def EIderivs(rE, rI, tau_E, a_E, theta_E, wEE, wEI, I_ext_E,
             tau_I, a_I, theta_I, wIE, wII, I_ext_I, **other_pars):
    drEdt = (-rE + F(wEE*rE - wEI*rI + I_ext_E, a_E, theta_E)) / tau_E
    drIdt = (-rI + F(wIE*rE - wII*rI + I_ext_I, a_I, theta_I)) / tau_I
    return drEdt, drIdt
```

**Key observations**:
- Trajectories follow the vector field direction
- Different trajectories eventually reach one of two fixed points (depending on initial conditions)
- Points where trajectories converge are intersections of the nullcline curves

---

### Tutorial 3: Jacobian Matrix and Stability

**System rewrite**:

$$\frac{dr_E}{dt} = G_E(r_E, r_I) = \frac{1}{\tau_E}[-r_E + F_E(w_{EE}r_E - w_{EI}r_I + I_E^{\rm ext}; a, \theta)]$$

$$\frac{dr_I}{dt} = G_I(r_E, r_I) = \frac{1}{\tau_I}[-r_I + F_I(w_{IE}r_E - w_{II}r_I + I_I^{\rm ext}; a, \theta)]$$

**Jacobian matrix (Eq. 7)**:

$$J = \begin{bmatrix} \frac{\partial G_E}{\partial r_E} & \frac{\partial G_E}{\partial r_I} \\ \frac{\partial G_I}{\partial r_E} & \frac{\partial G_I}{\partial r_I} \end{bmatrix}$$

**Jacobian matrix elements (Eqs. 8-11)**:

$$J[0,0] = \frac{\partial G_E}{\partial r_E} = \frac{1}{\tau_E}[-1 + w_{EE} F_E'(w_{EE}r_E^* - w_{EI}r_I^* + I_E^{\rm ext})]$$

$$J[0,1] = \frac{\partial G_E}{\partial r_I} = \frac{1}{\tau_E}[-w_{EI} F_E'(w_{EE}r_E^* - w_{EI}r_I^* + I_E^{\rm ext})]$$

$$J[1,0] = \frac{\partial G_I}{\partial r_E} = \frac{1}{\tau_I}[w_{IE} F_I'(w_{IE}r_E^* - w_{II}r_I^* + I_I^{\rm ext})]$$

$$J[1,1] = \frac{\partial G_I}{\partial r_I} = \frac{1}{\tau_I}[-1 - w_{II} F_I'(w_{IE}r_E^* - w_{II}r_I^* + I_I^{\rm ext})]$$

**Matrix notation**:

$$J = T^{-1}(FW - I)$$

where:
- $T = \begin{bmatrix} \tau_E & 0 \\ 0 & \tau_I \end{bmatrix}$ (time constant matrix)
- $F = \begin{bmatrix} F_E' & 0 \\ 0 & F_I' \end{bmatrix}$ (gain derivative matrix)
- $W = \begin{bmatrix} w_{EE} & -w_{EI} \\ w_{IE} & -w_{II} \end{bmatrix}$ (connectivity matrix)
- $I$ is the identity matrix

**Stability criterion**:
- $\det(J) > 0$ at stable fixed point (both eigenvalues have negative real parts)
- $\det(FW - I) = (F_E' w_{EI})(F_I' w_{IE}) - (F_I' w_{II} + 1)(F_E' w_{EE} - 1) > 0$

**Implementation**:

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

### Tutorial 3: Nullcline Slope Analysis

**E nullcline slope (Eq. 12)**:

$$\left(\frac{dr_I}{dr_E}\right)_{\text{E-nullcline}} = \frac{F_E' w_{EE} - 1}{F_E' w_{EI}}$$

**I nullcline slope (Eq. 13)**:

$$\left(\frac{dr_I}{dr_E}\right)_{\text{I-nullcline}} = \frac{F_I' w_{IE}}{F_I' w_{II} + 1}$$

**Properties**:
- I nullcline slope is always positive
- E nullcline slope sign depends on $(F_E' w_{EE} - 1)$

**Conclusion 1**: At a stable fixed point, the I nullcline has a steeper slope than the E nullcline

**Conclusion 2**: When adding input to the inhibitory population
- E nullcline stays the same
- I nullcline shifts left by $\delta I_I^{\rm ext} / w_{IE}$

---

### Tutorial 3: Limit Cycles and Oscillations

**Condition for oscillations**: Eigenvalues become **complex**

**Oscillatory parameters**: $w_{EE}=6.4$, $w_{EI}=4.8$, $w_{IE}=6.0$, $w_{II}=1.2$, $I_E^{\rm ext}=0.8$

- Trajectories form a **limit cycle** in the phase plane
- Excitatory (E) and inhibitory (I) populations alternate in activity
- Frequency determined by the imaginary part of eigenvalues
- Oscillation stability determined by the real part (positive → growing, negative → decaying)

**Bifurcation**: Dramatic qualitative change in system behavior as parameters change
- Changing $\tau_I$ can switch between steady state and oscillations
- Nullclines stay the same, but vector field changes
- Intuition: When $\tau_I$ is small, inhibitory activity changes faster than excitatory, leading to oscillations

---

### Tutorial 3: Inhibition-Stabilized Network (ISN)

**Two regimes based on $\frac{\partial G_E}{\partial r_E}$**:

$$\frac{\partial G_E}{\partial r_E} = \frac{1}{\tau_E}[-1 + w_{EE} F_E'] = \frac{1}{\tau_E}(F_E' w_{EE} - 1)$$

| Regime | Condition | E Nullcline Slope | Behavior |
|--------|-----------|-------------------|----------|
| **non-ISN** | $F_E' w_{EE} - 1 < 0$ | Negative | Increase inhibition on I → E decreases |
| **ISN** | $F_E' w_{EE} - 1 > 0$ | Positive | Increase inhibition on I → E paradoxically increases |

**ISN is common in cortex**: Strong recurrent excitation ($w_{EE}$ large) creates a regime that requires inhibition to be stable

**ISN paradoxical behavior**:
- Normal case: Inhibit I → E increases (reduced inhibition)
- ISN case: Inhibit I → E also decreases (because E's self-excitation is too strong, needs I to stabilize)

---

### Tutorial 3: Working Memory — Persistent Activity

**Mechanism**: Multiple fixed points + noise

1. System starts at low-activity fixed point
2. Brief pulse pushes state past unstable fixed point
3. System settles at high-activity fixed point
4. This represents a "memory" of the stimulus

**Implementation**: OU noise + brief current pulse

```python
def my_inject(pars, t_start, t_lag=10.):
    I = np.zeros(Lt)
    N_start = int(t_start / dt)
    N_lag = int(t_lag / dt)
    I[N_start:N_start + N_lag] = 1.
    return I
```

**Key parameters**:
- Pulse amplitude $S_E$ determines whether transition is triggered
- Critical pulse amplitude: Just enough to push state past unstable fixed point
- Sufficiently large pulse: System switches to persistent activity
- After pulse ends: System maintains high-activity state (working memory)

---

## Summary

---

### Week 2: Key Concepts

### W2D3: Linear Systems

- Euler integration
- Eigenvalue analysis
- Markov processes and state transition matrices
- Random walks and diffusion processes
- OU process and equilibrium variance
- Autoregressive models and time-delay matrices

### W2D4: Neuron Models

- LIF neuron dynamics and Euler integration
- DC/GWN/OU input types
- Correlated inputs and correlation transfer
- Conductance-based synapses
- Free membrane potential and firing regimes
- Short-term plasticity: depression and facilitation
- STDP learning rule and weight updates
- P/M trace variables

### W2D5: Network Dynamics

- Firing rate model and sigmoid transfer function
- Fixed points and eigenvalue stability
- Wilson-Cowan model and E/I coupling
- Nullclines and vector fields
- Jacobian matrix and linearization
- Nullcline slope analysis
- Limit cycles and bifurcations
- Inhibition-stabilized network
- Working memory and persistent activity

---

### Key Formulas

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

$$\frac{dr_I}{dr_E}\bigg|_{\text{E-nullcline}} = \frac{F_E' w_{EE} - 1}{F_E' w_{EI}} \quad \text{(E nullcline slope)}$$

$$\frac{dr_I}{dr_E}\bigg|_{\text{I-nullcline}} = \frac{F_I' w_{IE}}{F_I' w_{II} + 1} \quad \text{(I nullcline slope)}$$

$$\Delta W = \begin{cases} A_+ e^{\Delta t/\tau_+} & \Delta t < 0 \text{ (LTP)} \\ -A_- e^{-\Delta t/\tau_-} & \Delta t > 0 \text{ (LTD)} \end{cases} \quad \text{(STDP rule)}$$

---

### Logical Connections Between Tutorials

| Tutorial | Model | Dimension | Key Analysis |
|----------|-------|-----------|--------------|
| W2D3 T1 | $\dot{x} = ax$ | 1D | Euler integration, eigenvalues |
| W2D3 T2 | Markov process | 2D | State transition matrix, equilibrium |
| W2D3 T3 | OU process | 1D | Random walk, drift-diffusion, equilibrium variance |
| W2D3 T4 | Autoregressive model | 1D | Time-delay matrices, regression fitting |
| W2D4 T1 | LIF neuron | 1D | Membrane dynamics, F-I curve, CV_ISI |
| W2D4 T2 | Correlated LIF | 2×1D | Correlated inputs, correlation transfer |
| W2D4 T3 | Conductance LIF + STP | 1D | Synaptic conductance, u-R-g dynamics |
| W2D4 T4 | LIF + STDP | N×1D | Weight updates, unsupervised learning |
| W2D5 T1 | Single population rate | 1D | F-I curve, fixed points, eigenvalue stability |
| W2D5 T2 | Wilson-Cowan | 2D | Nullclines, vector field, phase plane |
| W2D5 T3 | WC + analysis | 2D | Jacobian eigenvalues, limit cycles, ISN, working memory |

**Progressive relationship**: From single population with one eigenvalue, to two-population system requiring a 2×2 Jacobian matrix (two eigenvalues that can be real or complex), enabling richer dynamics including oscillations and bistability.
