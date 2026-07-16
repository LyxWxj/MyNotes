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
| Parameter | Symbol | Meaning |
| --------- | ------ | ------- |
| Mean      | $\mu$  | Center of the peak |
| Std Dev   | $\sigma$ | Width / spread |
| Variance  | $\sigma^2$ | Squared spread |
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
In reality, many systems are not truly Markov, but we can *make* them Markov by including enough information in the state. For example: position alone is not Markov for a moving object, but position + velocity is.
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