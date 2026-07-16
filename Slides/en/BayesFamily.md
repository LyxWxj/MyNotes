# Bayes Family

Bayes' Theorem · Prior & Posterior · Naive Bayes · Bayesian Networks · Bayesian Inference

---

## Bayes' Theorem

---

### Deriving Bayes' Theorem

Start from the relationship between joint and conditional probability:

$$P(A, B) = P(A \mid B) \cdot P(B)$$

Swap the roles of $A$ and $B$:

$$P(B, A) = P(B \mid A) \cdot P(A)$$

**Key insight**: $A$ and $B$ occurring together is the same event as $B$ and $A$ occurring together — order doesn't matter:

$$P(A, B) = P(B, A)$$

Therefore:

$P(A \mid B) \cdot P(B) = P(B \mid A) \cdot P(A)$

Divide both sides by $P(B)$:

$$\boxed{\;P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}\;}$$

---

This is **Bayes' theorem** — it lets us "flip" the conditioning direction.

**In hypothesis-evidence language** ($H$ = hypothesis, $D$ = data):

$$\underbrace{P(H \mid D)}_{\text{posterior}} = \frac{\overbrace{P(D \mid H)}^{\text{likelihood}} \cdot \underbrace{P(H)}_{\text{prior}}}{\underbrace{P(D)}_{\text{evidence}}}$$

**What is $\theta$?** In parameter estimation, $\theta$ is a specific instance of $H$ — each possible parameter value is a hypothesis. For example, "$\theta = 0.7$" is the hypothesis "the coin's heads probability is 0.7". So $P(\theta)$ and $P(H)$ mean the same thing — the prior over parameters.

---

### Bayes' Theorem: Intuition

### Prior $P(H)$

Belief **before** seeing data

"20% chance of rain today"

### Likelihood $P(D \mid H)$

Probability of data given hypothesis

"If raining, 95% chance ground is wet"

### Posterior $P(H \mid D)$

Updated belief **after** seeing data

"Ground is wet — chance of rain?"

$$P(\text{rain} \mid \text{wet}) = \frac{P(\text{wet} \mid \text{rain}) \cdot P(\text{rain})}{P(\text{wet})} = \frac{0.95 \times 0.20}{0.27} = 0.704$$

**Evidence** $P(\text{wet})$ via the law of total probability:

$$P(\text{wet}) = P(\text{wet} \mid \text{rain})P(\text{rain}) + P(\text{wet} \mid \text{sunny})P(\text{sunny}) = 0.95 \times 0.20 + 0.10 \times 0.80 = 0.27$$

---

### The Role of Evidence

The evidence $P(D) = \sum_H P(D \mid H) P(H)$ is a normalizing constant ensuring the posterior sums to 1.

**In practice**, we often only care about relative posterior:

$$P(H \mid D) \propto P(D \mid H) \cdot P(H)$$

"Posterior ∝ Likelihood × Prior"

**Comparing two hypotheses**:

$$\frac{P(H_1 \mid D)}{P(H_2 \mid D)} = \frac{P(D \mid H_1)}{P(D \mid H_2)} \cdot \frac{P(H_1)}{P(H_2)}$$

**Posterior odds = Likelihood ratio × Prior odds**

---

## Prior & Posterior

---

### Choosing a Prior

The prior $P(H)$ encodes knowledge **before** seeing data.

| Prior type | Meaning | Example |
| ---------- | ------- | ------- |
| **Uninformative** | Minimize bias | $\theta \sim \text{Uniform}(0, 1)$ |
| **Weakly informative** | Gentle regularization | $\theta \sim \text{Beta}(2, 2)$ |
| **Strongly informative** | Based on prior knowledge | $\theta \sim \text{Beta}(50, 50)$ |

**Beta distribution** — the standard prior for probability parameters $\theta \in [0, 1]$:

$$\text{Beta}(\theta; \alpha, \beta) = \frac{1}{B(\alpha, \beta)} \theta^{\alpha-1} (1-\theta)^{\beta-1}$$

- $\alpha$: prior weight for "success"
- $\beta$: prior weight for "failure"

---

### Conjugate Priors

**The problem**: Bayes' theorem requires computing the evidence:

$$P(D) = \int P(D \mid \theta) P(\theta) \, d\theta$$

This integral is usually intractable — no closed-form solution.

**The solution**: choose a prior $P(\theta)$ that, when multiplied by the likelihood $P(D \mid \theta)$, gives a posterior in the **same distribution family**. This is a **conjugate prior**.

**Why it helps**: with a conjugate prior, the integral has an analytic solution. The posterior is just the same distribution with updated parameters — no integration needed, just addition.

When prior and posterior belong to the **same distribution family**, updating is just addition.

| Likelihood | Conjugate Prior | Posterior Update |
| ---------- | --------------- | ---------------- |
| Binomial | Beta | $\text{Beta}(\alpha + h,\; \beta + t)$ |
| Gaussian (known variance) | Gaussian | $\mathcal{N}(\mu_{\text{post}}, \sigma_{\text{post}}^2)$ |
| Poisson | Gamma | $\text{Gamma}(\alpha + \sum x_i,\; \beta + n)$ |
| Multinomial | Dirichlet | $\text{Dir}(\alpha_1 + n_1, \ldots, \alpha_k + n_k)$ |

**Beta-Binomial example**:

Prior: $\theta \sim \text{Beta}(5, 5)$ — "probably fair"

Data: 20 flips, 15 heads

Posterior: $\theta \mid D \sim \text{Beta}(5+15, 5+5) = \text{Beta}(20, 10)$

Posterior mean $= \frac{20}{30} = 0.67$ — shifted from prior (0.5) toward data (0.75).

---

### Gaussian-Gaussian Conjugacy

Known variance $\sigma^2$, estimate mean $\mu$:

Prior: $\mu \sim \mathcal{N}(\mu_0, \sigma_0^2)$

Data: $x_1, \ldots, x_n \sim \mathcal{N}(\mu, \sigma^2)$

Posterior: $\mu \mid \mathbf{x} \sim \mathcal{N}(\mu_{\text{post}}, \sigma_{\text{post}}^2)$

$$\frac{1}{\sigma_{\text{post}}^2} = \frac{1}{\sigma_0^2} + \frac{n}{\sigma^2}$$

$$\mu_{\text{post}} = \sigma_{\text{post}}^2 \left(\frac{\mu_0}{\sigma_0^2} + \frac{n\bar{x}}{\sigma^2}\right)$$

> [!tip] Intuition
> posterior precision = prior precision + data precision. More data → less prior influence.

$$\mu_{\text{post}} = w_{\text{prior}} \cdot \mu_0 + w_{\text{data}} \cdot \bar{x}$$

Weights are proportional to precisions (inverse variances).

---

### MAP vs MLE

| | MLE | MAP | Full Bayesian |
|---|---|---|---|
| Objective | $\arg\max P(D \mid \theta)$ | $\arg\max P(\theta \mid D)$ | $P(\theta \mid D)$ full distribution |
| Uses prior? | No | Yes | Yes |
| Result | Point estimate | Point estimate | Full distribution |
| Regularization | None | Corresponds to L2/L1 | Automatic |

**MAP derivation**:

$$\hat{\theta}_{\text{MAP}} = \arg\max P(\theta \mid D) = \arg\max \left[\log P(D \mid \theta) + \log P(\theta)\right]$$

**With Gaussian prior** $\theta \sim \mathcal{N}(0, \tau^2)$:

$$\log P(\theta) = -\frac{\theta^2}{2\tau^2} + \text{const}$$

MAP = MLE + L2 regularization, where $\lambda = \frac{\sigma^2}{\tau^2}$

---

### MAP Example: Estimating a Student's IQ

Suppose we measure a student's IQ. IQ scores follow $\mathcal{N}(\mu, 15^2)$ in the population.

**Data**: 5 test scores: 130, 125, 135, 128, 132

**MLE** (no prior):

$\hat{\mu}_{\text{MLE}} = \frac{130 + 125 + 135 + 128 + 132}{5} = 130$

**MAP** with prior $\mu \sim \mathcal{N}(100, 15^2)$ ("population average is 100"):

Using the Gaussian-Gaussian conjugacy formula:

$$\hat{\mu}_{\text{MAP}} = \frac{\sigma^2 \cdot \mu_0 + n \cdot \sigma_0^2 \cdot \bar{x}}{\sigma^2 + n \cdot \sigma_0^2} = \frac{225 \times 100 + 5 \times 225 \times 130}{225 + 5 \times 225} = \frac{22500 + 146250}{1350} = 124.4$$

**Comparison**:

| Method | Estimate | Interpretation |
| ------ | -------- | -------------- |
| MLE | 130.0 | "The data says 130" |
| MAP | 124.4 | "Data says 130, but prior pulls toward 100" |
| Full Bayesian | $\mathcal{N}(124.4, \sigma_{\text{post}}^2)$ | "Here's the entire distribution over $\mu$" |

**With more data** (50 scores, mean = 130): MAP $\approx$ 129.3 — prior washes out.

**With less data** (1 score = 130): MAP $\approx$ 115 — prior pulls strongly toward 100.

---

## Naive Bayes

---

### Naive Bayes Classifier

**Goal**: given features $\mathbf{x} = [x_1, \ldots, x_n]$, predict class $y$.

**Bayes' theorem**:

$$P(y \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid y) \cdot P(y)}{P(\mathbf{x})} \propto P(\mathbf{x} \mid y) \cdot P(y)$$

**The "naive" assumption**: given class $y$, all features are **conditionally independent**:

$$P(x_1, \ldots, x_n \mid y) = \prod_{i=1}^n P(x_i \mid y)$$

**Why "naive"?** This assumption is almost never true in practice (features are usually correlated), yet it works surprisingly well.

**Prediction rule**:

$$\hat{y} = \arg\max_y P(y) \prod_{i=1}^n P(x_i \mid y)$$

In log space (to avoid underflow):

$$\hat{y} = \arg\max_y \left[\log P(y) + \sum_{i=1}^n \log P(x_i \mid y)\right]$$

---

### Three Variants of Naive Bayes

| Variant | $P(x_i \mid y)$ assumption | Use case |
| ------- | --------------------------- | -------- |
| **Gaussian NB** | $x_i \mid y \sim \mathcal{N}(\mu_{iy}, \sigma_{iy}^2)$ | Continuous features |
| **Multinomial NB** | $x_i$ is a count (word frequency) | Text classification |
| **Bernoulli NB** | $x_i \in \{0, 1\}$ | Binary features |

**Gaussian Naive Bayes**:

$$P(x_i \mid y) = \frac{1}{\sqrt{2\pi\sigma_{iy}^2}} \exp\!\left(-\frac{(x_i - \mu_{iy})^2}{2\sigma_{iy}^2}\right)$$

Each feature has its own mean and variance per class.

---

### Naive Bayes: Spam Filtering Example

**Features**: word counts in emails (bag-of-words model)

**Training**:

- $P(\text{spam}) = \frac{\text{spam count}}{\text{total emails}}$
- $P(\text{"free"} \mid \text{spam}) = \frac{\text{count of "free" in spam}}{\text{total words in spam}}$
- Compute for every word and every class
**Prediction**:
$P(\text{spam} \mid \text{free}, \text{money}) \propto P(\text{spam}) \prod_{i} P(w_i \mid \text{spam})$
Pick the class with higher probability.
**Laplace smoothing**: prevents $P(w_i \mid y) = 0$ (a single unseen word zeroes out everything):

$$P(w_i \mid y) = \frac{\text{count}(w_i, y) + \alpha}{\text{count}(y) + \alpha \cdot |V|}$$

$\alpha = 1$ is Laplace smoothing, $|V|$ is vocabulary size.

---

### Naive Bayes: Step-by-Step Example

Classify an email containing the words "free" and "money" as spam or ham.

**Training data**: 10 emails (6 spam, 4 ham)

| Word | In spam (total 6) | In ham (total 4) |
| ---- | ------------------ | ----------------- |
| "free" | 5 | 1 |
| "money" | 4 | 0 |
| "meeting" | 1 | 3 |

**Step 1 — Priors**:

$$P(\text{spam}) = \frac{6}{10} = 0.6, \quad P(\text{ham}) = \frac{4}{10} = 0.4$$

---

**Step 2 — Likelihoods** (with Laplace smoothing, $|V| = 3$):

$$P(\text{"free"} \mid \text{spam}) = \frac{5 + 1}{6 + 3} = \frac{6}{9}, \quad P(\text{"free"} \mid \text{ham}) = \frac{1 + 1}{4 + 3} = \frac{2}{7}$$

$$P(\text{"money"} \mid \text{spam}) = \frac{4 + 1}{6 + 3} = \frac{5}{9}, \quad P(\text{"money"} \mid \text{ham}) = \frac{0 + 1}{4 + 3} = \frac{1}{7}$$

**Step 3 — Posteriors**:

$$P(\text{spam} \mid \text{free}, \text{money}) \propto 0.6 \times \frac{6}{9} \times \frac{5}{9} = 0.6 \times 0.222 = 0.133$$

$$P(\text{ham} \mid \text{free}, \text{money}) \propto 0.4 \times \frac{2}{7} \times \frac{1}{7} = 0.4 \times 0.041 = 0.016$$

**Step 4 — Normalize**: $P(\text{spam}|\text{(emails contain "free","money")}) = \frac{0.133}{0.133 + 0.016} = 0.893$

**Result**: 89.3% spam. Classified as **spam**.

---

### Multinomial Naive Bayes: Word Frequency Example

The previous example only checked **whether** a word appears (Bernoulli). Multinomial Naive Bayes counts **how many times** each word appears.

**Training data**: word counts across all emails

| Word | Count in spam (total 20 words) | Count in ham (total 15 words) |
| ---- | ------------------------------ | ----------------------------- |
| "free" | 8 | 1 |
| "money" | 6 | 0 |
| "meeting" | 2 | 5 |
| "project" | 4 | 9 |

---

**New email**: contains "free" twice, "money" once → feature vector $\mathbf{x} = [2, 1, 0, 0]$

**Likelihoods** (with Laplace smoothing, $|V| = 4$):

$$P(\text{"free"} \mid \text{spam}) = \frac{8 + 1}{20 + 4} = \frac{9}{24}, \quad P(\text{"free"} \mid \text{ham}) = \frac{1 + 1}{15 + 4} = \frac{2}{19}$$

$$P(\text{"money"} \mid \text{spam}) = \frac{6 + 1}{20 + 4} = \frac{7}{24}, \quad P(\text{"money"} \mid \text{ham}) = \frac{0 + 1}{15 + 4} = \frac{1}{19}$$

**Posteriors** (use word frequency as exponent):

$$P(\text{spam}) \cdot P(\text{"free"} \mid \text{spam})^2 \cdot P(\text{"money"} \mid \text{spam})^1 = 0.6 \times \left(\frac{9}{24}\right)^2 \times \frac{7}{24} = 0.0295$$

$$P(\text{ham}) \cdot P(\text{"free"} \mid \text{ham})^2 \cdot P(\text{"money"} \mid \text{ham})^1 = 0.4 \times \left(\frac{2}{19}\right)^2 \times \frac{1}{19} = 0.00047$$

**Normalize**: $P(\text{spam}) = \frac{0.0295}{0.0295 + 0.00047} = 0.984$

**Result**: 98.4% spam. The repeated "free" makes it even more confident than the Bernoulli version.

---

### Naive Bayes: Pros and Cons

**Pros**:

- Extremely fast training and prediction
- No iterative optimization needed
- Works well in high dimensions (text)
- Hard to overfit (strong regularization)
- Interpretable
**Cons**:
- Strong independence assumption
- Poor when features are correlated
- Probability estimates are poorly calibrated
- Sensitive to unseen features
**Why does the naive assumption work anyway?** Classification only needs **ranking**, not accurate probabilities. Even if probability values are distorted, as long as the correct class ranks highest, the prediction is right.

---

## Bayesian Networks

---

### What is a Bayesian Network?

A **Bayesian Network** is a **Directed Acyclic Graph (DAG)** that represents causal/dependency relationships between variables.

**Components**:

- **Nodes**: random variables
- **Directed edges**: causal direction ($A \to B$ means $A$ influences $B$)
- **Conditional Probability Tables (CPT)**: $P(X_i \mid \text{parents}(X_i))$
**Key property**: each node is conditionally independent of all non-descendants given its parents.

$$P(X_i \mid \text{parents}(X_i), \text{non-descendants}) = P(X_i \mid \text{parents}(X_i))$$

**Joint distribution factorization**:

$$P(X_1, X_2, \ldots, X_n) = \prod_{i=1}^n P(X_i \mid \text{parents}(X_i))$$

This is the **chain rule for Bayesian networks** — factorizes the full joint into local conditional probabilities.

---

### Example: The Sprinkler Problem

**Scenario**:

- Cloudy (C) affects Rain (R) and Sprinkler (S)
- Rain (R) and Sprinkler (S) affect Wet Grass (W)
**Graph structure**:

$$C \to R, \quad C \to S, \quad R \to W, \quad S \to W$$

**Joint distribution**:

$$P(C, R, S, W) = P(C) \cdot P(R \mid C) \cdot P(S \mid C) \cdot P(W \mid R, S)$$

**CPTs**:

$P(C=1) = 0.5$

$P(R=1 \mid C=1) = 0.8$, $P(R=1 \mid C=0) = 0.1$

$P(S=1 \mid C=1) = 0.1$, $P(S=1 \mid C=0) = 0.5$

$P(W=1 \mid R, S)$:

| R | S | P(W=1) |
|---|---|--------|
| 0 | 0 | 0.001  |
| 0 | 1 | 0.9    |
| 1 | 0 | 0.99   |
| 1 | 1 | 0.999  |

---

### Inference in Bayesian Networks

**Question**: given that the grass is wet (W=1), what is the probability the sprinkler was on?

$$P(S=1 \mid W=1) = \frac{P(S=1, W=1)}{P(W=1)} = \frac{\sum_{C,R} P(C, R, S=1, W=1)}{\sum_{C,R,S} P(C, R, S, W=1)}$$

**Code**:

```python
P_C = {0: 0.5, 1: 0.5}
P_R_given_C = {(0,0): 0.9, (0,1): 0.1, (1,0): 0.2, (1,1): 0.8}
P_S_given_C = {(0,0): 0.5, (0,1): 0.5, (1,0): 0.9, (1,1): 0.1}
P_W_given_RS = {(0,0,0): 0.999, (0,0,1): 0.001, (0,1,0): 0.1, (0,1,1): 0.9,
                (1,0,0): 0.01, (1,0,1): 0.99, (1,1,0): 0.001, (1,1,1): 0.999}
num = sum(P_C[c] * P_R_given_C[(c,r)] * P_S_given_C[(c,1)] * P_W_given_RS[(r,1,1)]
          for c in [0,1] for r in [0,1])
den = sum(P_C[c] * P_R_given_C[(c,r)] * P_S_given_C[(c,s)] * P_W_given_RS[(r,s,1)]
          for c in [0,1] for r in [0,1] for s in [0,1])
print(f"P(S=1|W=1) = {num/den:.4f}")  # ≈ 0.2572
```

---

### Conditional Independence Structures

Three fundamental structures determine independence:

### Chain

$A \to B \to C$

Given $B$: $A \perp C$

Without $B$: $A$ and $C$ correlated

### Fork

$A \leftarrow B \to C$

Given $B$: $A \perp C$

Without $B$: $A$ and $C$ correlated

### Collider

$A \to B \leftarrow C$

Given $B$: $A$ and $C$ correlated!

Without $B$: $A \perp C$

**The collider is counter-intuitive**: $A$ and $C$ are independent without observing $B$, but become correlated once $B$ is observed. This is called the **explaining away** effect.

---

### d-Separation

An algorithm for determining conditional independence in Bayesian networks:

**Rule**: given observation set $Z$, $X$ and $Y$ are **d-separated** by $Z$ iff **all paths** between $X$ and $Y$ are blocked.

A path is blocked if it contains a node that satisfies either:

1. **Chain or Fork**: the middle node is in $Z$ (observed)
2. **Collider**: the middle node and its descendants are **not** in $Z$ (unobserved)
**d-separation $\Rightarrow$ conditional independence**:

$$X \perp Y \mid Z \quad \text{if } X \text{ and } Y \text{ are d-separated given } Z$$

**In the sprinkler problem**:

- $C$ and $W$ are not independent (paths $C \to R \to W$ and $C \to S \to W$)
- Given $R$ and $S$: $C \perp W$ (all paths blocked)

---

## Bayesian Inference

---

### The Full Bayesian Inference Pipeline

**Step 1**: Choose a model (likelihood)

$$P(D \mid \theta) = \prod_{i=1}^N P(x_i \mid \theta)$$

**Step 2**: Choose a prior

$$P(\theta)$$

**Step 3**: Compute the posterior

$$P(\theta \mid D) = \frac{P(D \mid \theta) P(\theta)}{P(D)} \propto P(D \mid \theta) P(\theta)$$

**Step 4**: Use the posterior

- **Point estimate**: posterior mean, median, or mode (MAP)
- **Interval estimate**: credible interval
- **Prediction**: $P(x_{\text{new}} \mid D) = \int P(x_{\text{new}} \mid \theta) P(\theta \mid D) \, d\theta$

---

### Deriving the Posterior Predictive

We want $P(x_{\text{new}} \mid D)$ — the probability of new data given what we've observed.

**Step 1**: marginalize over $\theta$ (law of total probability):

$$P(x_{\text{new}} \mid D) = \int P(x_{\text{new}}, \theta \mid D) \, d\theta$$

**Step 2**: apply the chain rule to the joint:

$$P(x_{\text{new}}, \theta \mid D) = P(x_{\text{new}} \mid \theta, D) \cdot P(\theta \mid D)$$

**Step 3**: key assumption — given $\theta$, the new data $x_{\text{new}}$ is independent of the old data $D$:

$$P(x_{\text{new}} \mid \theta, D) = P(x_{\text{new}} \mid \theta)$$

This makes sense: once you know the parameter $\theta$, the old data doesn't tell you anything extra.

**Step 4**: substitute back:

$$\boxed{\;P(x_{\text{new}} \mid D) = \int P(x_{\text{new}} \mid \theta) \cdot P(\theta \mid D) \, d\theta\;}$$

---

### Posterior Predictive Distribution

To predict a new data point $x_{\text{new}}$, integrate over all possible $\theta$:

$$P(x_{\text{new}} \mid D) = \int P(x_{\text{new}} \mid \theta) P(\theta \mid D) \, d\theta$$

> [!tip] Intuition
> don't use a single "best" $\theta$ — average over **all** $\theta$ weighted by their posterior probability.
**Compared to MLE**:

| | MLE/MAP | Bayesian |
|---|---|---|
| Prediction | $P(x_{\text{new}} \mid \hat{\theta})$ | $\int P(x_{\text{new}} \mid \theta) P(\theta \mid D) d\theta$ |
| Uncertainty | Ignores $\hat{\theta}$ uncertainty | Automatically included |
| Small samples | High overfitting risk | Prior regularizes |

---

**Numerical approximation**: sample from the posterior:

$$P(x_{\text{new}} \mid D) \approx \frac{1}{S} \sum_{s=1}^S P(x_{\text{new}} \mid \theta^{(s)}), \quad \theta^{(s)} \sim P(\theta \mid D)$$

---

### Posterior Predictive: Concrete Example

Estimate a student's IQ. Observed 5 scores with mean 130.

**Posterior**: $\mu \mid D \sim \mathcal{N}(124.4, 6^2)$ (from earlier MAP example)

Now predict: what score will this student get on the next test?

**MLE approach** (point estimate):

The model says $x_{\text{new}} \sim \mathcal{N}(\mu, 15^2)$, but $\mu$ is unknown. MLE plugs in $\hat{\mu} = 130$:

$$x_{\text{new}} \sim \mathcal{N}(\hat{\mu}, 15^2) = \mathcal{N}(130, 15^2)$$

Treat $\hat{\mu}$ as if it were the true $\mu$. Prediction: mean 130, std 15.

**Bayesian approach** (integrate over posterior):

$$P(x_{\text{new}} \mid D) = \int \mathcal{N}(x_{\text{new}} \mid \mu, 15^2) \cdot \mathcal{N}(\mu \mid 124.4, 6^2) \, d\mu$$

Result: $x_{\text{new}} \mid D \sim \mathcal{N}(124.4, 15^2 + 6^2) = \mathcal{N}(124.4, 261)$

Prediction: mean 124.4, std $\sqrt{261} \approx 16.2$

---

**Comparison**:

| | MLE | Bayesian |
|---|---|---|
| Predicted mean | 130.0 | 124.4 |
| Predicted std | 15.0 | 16.2 |
| Why different? | Ignores $\mu$ uncertainty | Includes $\mu$ uncertainty |

**The extra uncertainty** ($16.2 > 15$) comes from not knowing the true $\mu$. Bayesian prediction is more honest about what we don't know.

**Sampling interpretation**:

1. Draw $\mu^{(s)}$ from posterior $\mathcal{N}(124.4, 6^2)$
2. Draw $x^{(s)}$ from $\mathcal{N}(\mu^{(s)}, 15^2)$
3. Repeat 1000 times → histogram of $x^{(s)}$ approximates $P(x_{\text{new}} \mid D)$,
Each sample is a two-step process: first sample a parameter, then sample data given that parameter.

---

### Bayesian Linear Regression

Standard regression: $\hat{y} = X\hat{\theta}$ (one point estimate)

Bayesian regression: place a prior on $\theta$, get a posterior distribution

Prior: $\theta \sim \mathcal{N}(0, \alpha^{-1}I)$ ($\alpha$ controls regularization strength)

Likelihood: $y \mid X, \theta \sim \mathcal{N}(X\theta, \beta^{-1}I)$ ($\beta = 1/\sigma^2$)

Posterior (Gaussian-Gaussian conjugacy):

$$\theta \mid X, y \sim \mathcal{N}(\mu_N, \Sigma_N)$$

$$\Sigma_N = (\alpha I + \beta X^T X)^{-1}$$

$$\mu_N = \beta \Sigma_N X^T y$$

**Predictive distribution** is also Gaussian:

$$y_{\text{new}} \mid x_{\text{new}}, X, y \sim \mathcal{N}(\mu_N^T x_{\text{new}},\; \sigma_{\text{pred}}^2)$$

Prediction variance has **two parts**: noise $\beta^{-1}$ + parameter uncertainty.

---

### When the Posterior Has No Closed Form

Most real-world posteriors cannot be computed analytically.

**Method 1: MCMC (Markov Chain Monte Carlo)**

Sample from the posterior, approximate with samples.

- **Metropolis-Hastings**: general-purpose sampler
- **Gibbs Sampling**: sample one dimension at a time
- **HMC (Hamiltonian Monte Carlo)**: uses gradient information for efficient sampling
**Method 2: Variational Inference**
Approximate posterior with a simple distribution $q(\theta)$, minimize KL divergence:

$$\min_{q} D_{\text{KL}}(q(\theta) \| P(\theta \mid D))$$

Faster than MCMC, but may sacrifice accuracy.

**Method 3: Laplace Approximation**

Approximate posterior as Gaussian (second-order Taylor expansion around MAP).

---

### Bayesian vs Frequentist

| | Frequentist | Bayesian |
|---|---|---|
| Probability means | Long-term frequency | Degree of belief |
| Parameters | Fixed but unknown | Random variables |
| Inference | Point estimate + CI | Posterior + credible interval |
| Prior | Not used | Core component |
| Computation | Usually closed form | Often approximate |
| Small samples | Unstable | Prior helps regularize |

**Confidence interval vs Credible interval**:

- 95% CI: "if we repeat the experiment 100 times, 95 of the intervals contain the true value"
- 95% credible interval: "the parameter lies in this interval with 95% probability"
The latter is more intuitive, but requires a prior.

---

### Summary

### Foundations

- Bayes' theorem
- Prior → Posterior
- Conjugate priors
- MAP vs MLE

### Models

- Naive Bayes classifier
- Bayesian networks
- d-separation
- Conditional independence

### Inference

- Posterior predictive
- Bayesian linear regression
- MCMC / Variational inference
- Bayesian vs Frequentist

$$\boxed{\;P(H \mid D) = \frac{P(D \mid H) \cdot P(H)}{P(D)}\;}$$

The essence of Bayesian thinking: **update beliefs with data, make decisions with posteriors.**
