# Markov Processes and Kalman Filtering

Hidden Markov Models · Kalman Filter · Forward Inference · Smoothing

---

## Overview

This slides cover three core topics, from inference with discrete latent states to continuous latent states:

| Topic | Model | Latent State Type | Inference Method |
|------|------|-----------|----------|
| **Hidden Markov Model (HMM)** | Discrete states + Gaussian observations | Discrete ($s_t \in \{0,1\}$) | Forward Inference |
| **Kalman Filter (1D)** | Continuous states + Gaussian observations | Continuous ($s_t \in \mathbb{R}$) | Bayesian Update |
| **Kalman Filter (2D)** | Multi-dimensional states + smoothing | Continuous ($s_t \in \mathbb{R}^n$) | Filtering + Smoothing + EM |

**Core idea**: We cannot directly observe the hidden states; we can only infer them through noisy measurements. Bayesian inference is the unified framework.

---

## Part 1: Hidden Markov Model (HMM)

---

### What is the Markov Property?

**Markov Property**: The future depends only on the present, not on the past.

$$p(s_t | s_{t-1}, s_{t-2}, \ldots, s_1) = p(s_t | s_{t-1})$$

**Markov Chain**: A sequence of states $\{s_1, s_2, \ldots, s_T\}$ that satisfies the Markov property

```
s_1 → s_2 → s_3 → ... → s_T
```

**Transition Matrix** $D$: Describes the transition probabilities between states

$$D = \begin{bmatrix} p(s_t=+1|s_{t-1}=+1) & p(s_t=-1|s_{t-1}=+1) \\ p(s_t=+1|s_{t-1}=-1) & p(s_t=-1|s_{t-1}=-1) \end{bmatrix}$$

**State Update**: $P_t = P_{t-1} D$, where $P_t = [p(s_t=+1), p(s_t=-1)]$

$P_{t-1} =[p(s_{t-1}=1),p(s_{t-1}=-1)]$

---

### Structure of Hidden Markov Model (HMM)

HMM = Markov Chain + Observation Model

```
Hidden states:   s_1 → s_2 → s_3 → ... → s_T
                    ↓      ↓      ↓           ↓
Observations:    m_1    m_2    m_3    ...   m_T
```

**Two key components**:

| Component | Mathematical Description | Meaning |
|------|---------|------|
| **State Transition** | $p(s_t \| s_{t-1})$ | How states change over time |
| **Observation Model** | $p(m_t \| s_t)$ | How states generate observations |

**Binary HMM Example**:

- Hidden states: $s_t \in \{+1, -1\}$ (e.g., fish school on left/right)
- Transition probability: $p_{\text{switch}}$ (switching probability)
- Observations: $m_t | s_t \sim \mathcal{N}(\mu_{s_t}, \sigma^2)$

---

### HMM Data Generation

Sampling data from an HMM:

**Step 1: Generate hidden state sequence**

```python
def sample(model, T):
    S = np.zeros((T,), dtype=int)
    S[0] = np.random.choice([0, 1], p=model.startprob)
    for t in range(1, T):
        # Sample next state according to the transition matrix
        transition_vector = model.transmat[S[t-1], :]
        S[t] = np.random.choice([0, 1], p=transition_vector)
```

**Step 2: Generate observations**

```python
    # Generate Gaussian observations based on states
    means = model.means[S]
    scales = np.sqrt(model.vars[S])
    M = np.random.normal(loc=means, scale=scales, size=(T,))
```

```python
transmat = np.array([[1 - switch_prob, switch_prob],
                     [switch_prob, 1 - switch_prob]])
```

---

### Predicting the Future: Growing Uncertainty

Even if the current state is completely known, the future becomes uncertain.

**Prediction without observations**:

$$P_t = P_{t-1} D = P_0 D^t$$

```python
def simulate_prediction_only(model, nstep):
    predictive_probs = []
    prob = model.startprob
    for i in range(nstep):
        predictive_probs.append(prob)
        prob = prob @ model.transmat  # one-step prediction
    return predictive_probs
```

**Key observations**:

- Higher switching probability → faster forgetting
- Switching probability = 0.5 → immediately reaches uniform distribution
- Switching probability > 0.5 → states tend to switch (oscillate)

**Neuroscience application**: This is why we need to constantly update our perception in a changing world!

---

### Forward Inference

Given an observation sequence $m_{1:t}$, infer the current hidden state $s_t$.

**Recursive Bayesian inference**, with two operations at each step:

**1. Predict**: Use the transition matrix to turn yesterday's posterior into today's prior

$$\text{today's prior} = p(s_t | m_{1:t-1}) = p(s_{t-1} | m_{1:t-1}) \cdot D$$

**2. Update**: Combine the new observation $m_t$ to compute the posterior

$$\text{posterior} \propto \text{prior} \times \text{likelihood} = p(m_t | s_t) \cdot p(s_t | m_{1:t-1})$$

```python
def one_step_update(model, posterior_tm1, M_t):
    # Predict
    prediction = posterior_tm1 @ model.transmat
    # Compute likelihood
    likelihood = compute_likelihood(model, M_t)
    # Update (Bayes' rule)
    posterior_t = prediction * likelihood
    posterior_t /= np.sum(posterior_t)  # normalize
    return prediction, likelihood, posterior_t
```

---

### Complete Forward Inference Flow

```
Initial state P_0
    │
    ▼
┌─────────────────────────────────────────────┐
│  Time t=1,2,...,T                            │
│                                              │
│  1. Predict: prior = posterior_{t-1} × D     │
│                                              │
│  2. Compute likelihood: L = p(m_t | s_t)     │
│                                              │
│  3. Update: posterior ∝ prior × L            │
│                                              │
│  4. Normalize: posterior = posterior / sum    │
└─────────────────────────────────────────────┘
    │
    ▼
Posterior probability sequence {P(s_t|m_{1:t})}
```

**Factors affecting inference quality**:

- **Higher noise** → flatter likelihood → posterior relies more on prior
- **Faster switching** → more uncertain prior → posterior relies more on observations

---

### HMM Summary

### Key Concepts

- **Markov Property**: The future depends only on the present
- **Hidden States**: Cannot be directly observed
- **Transition Matrix**: Describes state evolution
- **Observation Model**: Mapping from states to observations

### Inference Algorithm

- **Predict**: Prior = Posterior × D
- **Update**: Posterior ∝ Prior × Likelihood
- **Normalize**: Ensure probabilities sum to 1

**Application Scenarios**:
- Ion channel open/close states
- Sleep/wake state switching
- Fish school location tracking
- Phoneme sequence in speech recognition

---

## Part 2: Kalman Filter (1D)

---

### From Discrete to Continuous: Why Do We Need the Kalman Filter?

HMM handles **discrete** hidden states, but many real-world systems are **continuous**.

| | HMM | Kalman Filter |
|---|---|---|
| **Hidden States** | Discrete (finite number) | Continuous (real numbers) |
| **State Transition** | Transition matrix | Linear dynamics + noise |
| **Observation Model** | Arbitrary distribution | Linear + Gaussian noise |
| **Inference** | Forward algorithm | Bayesian update (analytical solution) |

**The Kalman Filter is the continuous version of HMM!**

Core assumption: All distributions are **Gaussian** → mathematically solvable in closed form

---

### Mathematical Model of the Kalman Filter

**State Equation** (Dynamics Model):

$$s_t = D \cdot s_{t-1} + w_t, \quad w_t \sim \mathcal{N}(0, \sigma_p^2)$$

- $D$: Dynamics parameter (how states evolve)
- $w_t$: Process noise (uncertainty in dynamics)

**Observation Equation** (Measurement Model):

$$m_t = s_t + \eta_t, \quad \eta_t \sim \mathcal{N}(0, \sigma_m^2)$$

- $\eta_t$: Measurement noise (sensor uncertainty)

**Complete Model**:

```
Hidden states: s_0 → s_1 → s_2 → ... → s_T
                  ↓      ↓      ↓           ↓
Observations:  m_1    m_2    m_3    ...   m_T
```

All variables are **Gaussian distributions**!

---

### Kalman Filter: Bayesian Update

At each step: Gaussian × Gaussian = Gaussian

**Step 1: Predict** — Turn yesterday's posterior into today's prior

$$\text{prior} = \mathcal{N}(D \cdot \mu_{t-1}, D^2 \cdot \sigma_{t-1}^2 + \sigma_p^2)$$

**Step 2: Update** — Combine with likelihood to compute posterior

Posterior = Prior × Likelihood = Gaussian × Gaussian = Gaussian

**Information Weighting** (Key Insight):

$$\frac{1}{\sigma_{\text{post}}^2} = \frac{1}{\sigma_{\text{prior}}^2} + \frac{1}{\sigma_{\text{likelihood}}^2}$$

$$\mu_{\text{post}} = g_{\text{prior}} \cdot \mu_{\text{prior}} + g_{\text{likelihood}} \cdot m_t$$

where $g = \frac{\text{information}}{\text{total information}}$ is the information weight

---

### Intuition for the Kalman Filter

**Information-Weighted Average**:

```
Posterior mean = (Prior information × Prior mean + Likelihood information × Observation) / Total information
```

| Scenario | Prior Weight | Likelihood Weight | Result |
| ----- | ---- | ---- | ----- |
| Low measurement noise | Low | High | Close to observation |
| Low process noise | High | Low | Close to prediction |
| Comparable | Medium | Medium | Weighted average |

**Classical Kalman Gain**:

$$K = \frac{\sigma_{\text{prior}}^2}{\sigma_{\text{prior}}^2 + \sigma_m^2}$$

$$\mu_{\text{post}} = \mu_{\text{prior}} + K(m_t - \mu_{\text{prior}})$$

$K$ close to 1 → trust measurement; $K$ close to 0 → trust prediction

---

### Kalman Filter Code Implementation

```python
gaussian = namedtuple('Gaussian', ['mean', 'cov'])
def kalman_filter_step(posterior, D, process_noise, measurement_noise, m):
    # Step 1: Predict
    prior_mean = D * posterior.mean
    prior_cov = D**2 * posterior.cov + process_noise
    # Step 2: Update
    likelihood = gaussian(m, measurement_noise)
    # Information weighting
    info_prior = 1 / prior_cov
    info_likelihood = 1 / likelihood.cov
    info_posterior = info_prior + info_likelihood
    # Posterior mean = weighted average
    prior_weight = info_prior / info_posterior
    likelihood_weight = info_likelihood / info_posterior
    posterior_mean = prior_weight * prior_mean + likelihood_weight * m
    # Posterior variance = 1/total information
    posterior_cov = 1 / info_posterior
    return gaussian(posterior_mean, posterior_cov)
```

---

### Properties of the Kalman Filter

**1. Posterior variance decreases over time**

$$\sigma_t^2 < \sigma_{t-1}^2$$

Each new observation reduces uncertainty.

**2. Steady-state variance**

As $t \to \infty$, the posterior variance converges to:

$$\sigma_{\infty}^2 = \frac{\sigma_p^2}{1-D^2} \cdot \frac{1}{\text{SNR}+1}$$

where $\text{SNR} = \sigma_p^2 / \sigma_m^2$ is the signal-to-noise ratio

**3. Estimation error matches posterior variance**

The distribution of the estimation error $\hat{s}_t - s_t$ matches the posterior distribution $\mathcal{N}(0, \sigma_t^2)$!

**4. Online algorithm**

Uses only past data and can run in real-time.

---

### Factors Affecting Kalman Filter Performance

**Dynamics parameter $D$**:

- $|D| < 1$: Stable system, states decay to 0
- $|D| = 1$: Random walk
- $|D| > 1$: Unstable system, states diverge

**Process noise $\sigma_p$**:
- Large → fast state changes → uncertain predictions → rely more on measurements
- Small → slow state changes → accurate predictions → rely more on prior

**Measurement noise $\sigma_m$**:
- Large → unreliable measurements → rely more on predictions
- Small → reliable measurements → rely more on observations

**Signal-to-noise ratio SNR = $\sigma_p^2 / \sigma_m^2$**:
- High SNR → good filtering performance
- Low SNR → more time needed for accurate localization

---

### Neuroscience Applications of the Kalman Filter

| Application | Hidden State | Observation |
|------|--------|--------|
| **Brain-Computer Interface** | Motor intention | Neural activity |
| **EEG Analysis** | Brain activity | Scalp voltage |
| **Perceptual Tracking** | Object position | Retinal image |
| **Motor Control** | Limb position | Proprioception |

**Is the brain a Kalman filter?**

- The brain needs to infer world states from noisy sensory inputs
- Bayesian inference is a powerful framework for describing brain perception
- The Kalman filter provides an online, recursive inference algorithm

---

## Part 3: 2D Kalman Filter and Smoothing

---

### From 1D to 2D: Vector Form

**State Equation**:

$$\mathbf{s}_t = D \mathbf{s}_{t-1} + \mathbf{w}_t, \quad \mathbf{w}_t \sim \mathcal{N}(0, Q)$$

**Observation Equation**:

$$\mathbf{m}_t = H \mathbf{s}_t + \boldsymbol{\eta}_t, \quad \boldsymbol{\eta}_t \sim \mathcal{N}(0, R)$$

**Parameter Matrices**:

| Symbol | Name | Dimension | Meaning |
|------|------|------|------|
| $D$ | State transition matrix | $n \times n$ | How states evolve |
| $Q$ | Process noise covariance | $n \times n$ | Dynamics uncertainty |
| $H$ | Observation matrix | $m \times n$ | How states map to observations |
| $R$ | Observation noise covariance | $m \times m$ | Measurement uncertainty |

---

### 2D Kalman Filter: Prediction Step

**Predicted mean**:

$$\hat{\mu}_t^{\text{pred}} = D \hat{\mu}_{t-1}$$

**Predicted covariance**:

$$\hat{\Sigma}_t^{\text{pred}} = D \hat{\Sigma}_{t-1} D^\top + Q$$

**Intuition**:

- Mean propagates through the dynamics matrix
- Covariance is scaled by $D$ and augmented by process noise

---

### 2D Kalman Filter: Update Step

**Kalman Gain Matrix**:

$$K_t = \hat{\Sigma}_t^{\text{pred}} H^\top (H \hat{\Sigma}_t^{\text{pred}} H^\top + R)^{-1}$$

**Filtered mean**:

$$\hat{\mu}_t^{\text{filter}} = \hat{\mu}_t^{\text{pred}} + K_t (\mathbf{m}_t - H \hat{\mu}_t^{\text{pred}})$$

**Filtered covariance**:

$$\hat{\Sigma}_t^{\text{filter}} = (I - K_t H) \hat{\Sigma}_t^{\text{pred}}$$

**Innovation** $\mathbf{m}_t - H \hat{\mu}_t^{\text{pred}}$: The difference between actual observation and predicted observation

$K_t$ determines how to trade off between prediction and observation

---

### 2D Kalman Filter Code

```python
def kalman_filter(data, params):
    D, Q, H, R = params['D'], params['Q'], params['H'], params['R']
    I = np.eye(D.shape[0])
    mu = np.zeros((len(data), D.shape[0]))
    sigma = np.zeros((len(data), D.shape[0], D.shape[0]))
    for t, y in enumerate(data):
        if t == 0:
            mu_pred = params['mu_0']
            sigma_pred = params['sigma_0']
        else:
            mu_pred = D @ mu[t-1]
            sigma_pred = D @ sigma[t-1] @ D.T + Q
        # Kalman gain
        K = sigma_pred @ H.T @ np.linalg.inv(H @ sigma_pred @ H.T + R)
        # Update
        mu[t] = mu_pred + K @ (y - H @ mu_pred)
        sigma[t] = (I - K @ H) @ sigma_pred
    return mu, sigma
```

---

### Kalman Smoothing

**Filtering** uses only past data, while **smoothing** uses all data (past + future).

**Smoothed mean** (backward pass):

$$\hat{\mu}_t^{\text{smooth}} = \hat{\mu}_t^{\text{filter}} + J_t (\hat{\mu}_{t+1}^{\text{smooth}} - D \hat{\mu}_t^{\text{filter}})$$

**Smoothed covariance**:

$$\hat{\Sigma}_t^{\text{smooth}} = \hat{\Sigma}_t^{\text{filter}} + J_t (\hat{\Sigma}_{t+1}^{\text{smooth}} - P_t) J_t^\top$$

**Smoothing gain**:

$$J_t = \hat{\Sigma}_t^{\text{filter}} D^\top P_t^{-1}$$

where $P_t = D \hat{\Sigma}_t^{\text{filter}} D^\top + Q$ is the predicted covariance at time $t+1$

---

### Filtering vs Smoothing

| | Filtering | Smoothing |
|---|---|---|
| **Data Used** | $m_{1:t}$ (past) | $m_{1:T}$ (all) |
| **Direction** | Forward ($t=0 \to T$) | Backward ($t=T \to 0$) |
| **Real-time** | Can run online | Requires batch processing |
| **Accuracy** | Lower | Higher |
| **MSE** | Larger | Smaller |

```python
print(f"Filtering MSE: {np.mean((state - filtered_state_means)**2):.3f}")
print(f"Smoothing MSE: {np.mean((state - smoothed_state_means)**2):.3f}")
# Smoothing MSE is significantly smaller!
```

**Why is smoothing better?** Because it uses future information to correct past estimates.

---

### Parameter Learning: EM Algorithm

When we don't know the system parameters $D, Q, H, R$, we need to learn them from data.

**Expectation-Maximization (EM) Algorithm**:

| Step | Operation | Tool |
|------|------|------|
| **E Step** | Fix parameters, infer hidden states | Kalman filtering + smoothing |
| **M Step** | Fix hidden states, update parameters | Maximum likelihood estimation |

**M Step Update Formula** (example):

$$D^{\text{new}} = \left(\sum_{t=2}^T \mathbb{E}[s_t s_{t-1}^\top]\right) \left(\sum_{t=2}^T \mathbb{E}[s_{t-1} s_{t-1}^\top]\right)^{-1}$$

```python
import pykalman
kf = pykalman.KalmanFilter(n_dim_state=2, n_dim_obs=2,
                           em_vars=['transition_matrices', 'transition_covariance',
                                    'observation_matrices', 'observation_covariance'])
kf.em(data)  # Learn parameters using EM algorithm
```

---

### Application: Eye-Tracking Data Smoothing

**Problem**: Eye-tracker data contains noise and needs smoothing

**Method**: Use Kalman filter to smooth eye movement trajectories

```python
# Set up model
kf = pykalman.KalmanFilter(n_dim_state=2, n_dim_obs=2)
kf.initial_state_mean = data[0]
kf.initial_state_covariance = 0.1 * np.eye(2)
# Learn parameters using EM
kf.em(data)
# Smooth data
smoothed_mean, smoothed_cov = kf.smooth(data)
```

**Result**: The green curve is the smoothed eye movement trajectory, which is smoother than the original data (magenta)

**Handling blinks**: Use numpy's masked array to mark negative coordinates as missing values

---

## Summary

---

### Key Formulas

**HMM Forward Inference**:

$$P(s_t | m_{1:t}) \propto p(m_t | s_t) \cdot [P(s_{t-1}|m_{1:t-1}) \cdot D]$$

**1D Kalman Filter**:

$$\mu_t^{\text{post}} = g_{\text{prior}} \cdot D\mu_{t-1} + g_{\text{likelihood}} \cdot m_t$$

$$\frac{1}{(\sigma_t^{\text{post}})^2} = \frac{1}{(D\sigma_{t-1})^2 + \sigma_p^2} + \frac{1}{\sigma_m^2}$$

**2D Kalman Filter**:

$$K_t = \Sigma_t^{\text{pred}} H^\top (H \Sigma_t^{\text{pred}} H^\top + R)^{-1}$$

$$\mu_t^{\text{filter}} = \mu_t^{\text{pred}} + K_t (m_t - H\mu_t^{\text{pred}})$$

---

### Applications and Extensions

**Neuroscience Applications**:

- Motor intention estimation in brain-computer interfaces
- Source localization of EEG/MEG signals
- State estimation in sensorimotor control
- Neural population decoding

**Extended Models**:

| Extension | Problem | Method |
|------|------|------|
| Nonlinear dynamics | $s_t = f(s_{t-1}) + w_t$ | Extended Kalman Filter (EKF) |
| Non-Gaussian noise | Heavy-tailed distributions | Particle Filter |
| Continuous time | $ds = f(s)dt + \sigma dW$ | Kalman-Bucy Filter |
| Unknown parameters | $D, Q, R$ unknown | EM Algorithm |

---
