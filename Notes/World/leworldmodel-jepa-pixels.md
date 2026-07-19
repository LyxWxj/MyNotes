---
type: Note
related_to: "[[world-model]]"
status: Active
url: https://arxiv.org/abs/2603.19312
tags:
  - world-model
  - JEPA
  - representation-learning
  - end-to-end
  - planning
---

# LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels

## Meta

- **Authors:** Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, Randall Balestriero
- **Date:** 2026-03
- **arXiv ID:** 2603.19312
- **URL:** https://arxiv.org/abs/2603.19312
- **Zotero Key:** HACRF4TN
- **DOI:** 10.48550/arXiv.2603.19312

## Abstract

Joint Embedding Predictive Architectures (JEPAs) offer a compelling framework for learning world models in compact latent spaces, yet existing methods remain fragile, relying on complex multi-term losses, exponential moving averages, pre-trained encoders, or auxiliary supervision to prevent collapse. This paper introduces LeWorldModel (LeWM), the first JEPA that trains stably end-to-end from raw pixels using only two loss terms: a next-embedding prediction loss and a regularizer constraining latent embeddings to a Gaussian distribution. This reduces tunable loss hyperparameters from six to one compared to the only existing end-to-end alternative. With approximately 15 million parameters, it can be trained on a single GPU within hours, achieving planning speeds up to 48x faster than foundation-model-based world models while maintaining competitive performance across diverse 2D and 3D control tasks.

## Key Contributions

1. **First stable end-to-end JEPA from raw pixels** -- uses only two loss terms (prediction loss + Gaussian regularizer), no EMA, no stop-gradient, no frozen encoders
2. **Dramatic hyperparameter simplification** -- reduces tunable loss hyperparameters from 6 to 1 compared to PLDM (the only prior end-to-end alternative)
3. **Provable anti-collapse guarantee** -- SIGReg regularizer converges to isotropic Gaussian iff latent distribution matches, via Cramer-Wold theorem
4. **High efficiency** -- ~15M parameters, single-GPU training in hours, up to 48x faster planning than foundation-model-based world models
5. **Emergent physical structure** -- latent space encodes meaningful physical quantities (verified via probing); surprise evaluation detects physically implausible events

## Method

### JEPA Background

A Joint Embedding Predictive Architecture encodes observations into a compact latent space and models temporal dynamics by predicting latent representations of future observations. The key challenge is **representation collapse** -- the model mapping all inputs to nearly identical representations to trivially satisfy the prediction objective.

Prior approaches address collapse via:
- **I-JEPA / V-JEPA:** EMA of target encoder + stop-gradient (heuristic, lacks theoretical grounding)
- **DINO-WM:** Freeze a pre-trained DINOv2 encoder (avoids collapse but limits expressivity)
- **PLDM:** VICReg-based 7-term loss (suffers from training instabilities)

LeWM is the first JEPA that is simultaneously end-to-end, task-agnostic, pixel-based, reconstruction-free, reward-free, single-hyperparameter, and with provable anti-collapse guarantees.

### Architecture

**Encoder (ViT-Tiny):**
- Backbone: Vision Transformer (ViT-Tiny) from Hugging Face
- Patch size: 14, 12 layers, 3 attention heads, hidden dim 192 (~5M params)
- Uses `[CLS]` token embedding from the last layer
- Projection via 1-layer MLP with Batch Normalization (needed because ViT's final LayerNorm prevents effective optimization of the anti-collapse objective)

**Predictor (ViT-S):**
- 6 layers, 16 attention heads, dropout 10% (~10M params)
- Learned positional embeddings with causal masking over observation history
- Action conditioning via Adaptive Layer Normalization (AdaLN), initialized to zero
- History length: 3 for PushT and OGBench-Cube; 1 for TwoRoom
- Predicts next frame representation autoregressively with temporal causal masking
- Followed by a projector network (same as encoder's projection)

**Decoder (visualization only, not used in training):**
- Lightweight transformer decoder for diagnostic visualization
- Decodes `[CLS]` token embedding (192 dim) via cross-attention with learnable query tokens
- For 224x224 images with patch size 16: 196 learnable query tokens

**Input processing:**
- Frame-skip of 5 (consecutive actions grouped into single action block)
- Batch size: 128, sub-trajectories of size 4 frames
- Each frame: 224x224 pixels

### Loss Function

The complete LeWM training objective consists of only two terms:

$$\mathcal{L}_{\text{LeWM}} = \mathcal{L}_{\text{pred}} + \lambda \cdot \text{SIGReg}(\mathbf{Z})$$

**Prediction loss (teacher-forcing):**

$$\mathcal{L}_{\text{pred}} = \|\hat{\mathbf{z}}_{t+1} - \mathbf{z}_{t+1}\|_2^2, \quad \hat{\mathbf{z}}_{t+1} = \text{pred}_\phi(\mathbf{z}_t, \mathbf{a}_t)$$

**Comparison with PLDM's 7-term loss:**

$$\mathcal{L}_{\text{PLDM}} = \mathcal{L}_{\text{pred}} + \alpha\mathcal{L}_{\text{var}} + \beta\mathcal{L}_{\text{cov}} + \gamma\mathcal{L}_{\text{time-sim}} + \zeta\mathcal{L}_{\text{time-var}} + \nu\mathcal{L}_{\text{time-cov}} + \mu\mathcal{L}_{\text{IDM}}$$

### SIGReg Regularizer (Anti-Collapse)

SIGReg (Sketched-Isotropic-Gaussian Regularizer) encourages latent embeddings to match an isotropic Gaussian target distribution.

**Mechanism:**
1. Sample M unit-norm directions $\mathbf{u}^{(m)} \in \mathbb{S}^{D-1}$ uniformly on the hypersphere
2. Project embeddings: $\mathbf{h}^{(m)} = \mathbf{Z}\mathbf{u}^{(m)}$
3. Apply univariate Epps-Pulley test statistic on each projection
4. Aggregate:

$$\text{SIGReg}(\mathbf{Z}) = \frac{1}{M}\sum_{m=1}^{M} T(\mathbf{h}^{(m)})$$

**Epps-Pulley test statistic:**

$$T^{(m)} = \int_{-\infty}^{\infty} w(t)\left|\phi_N(t; \mathbf{h}^{(m)}) - \phi_0(t)\right|^2 dt$$

Where:
- $\phi_N(t; \mathbf{h}) = \frac{1}{N}\sum_{n=1}^{N} e^{it\mathbf{h}_n}$ is the empirical characteristic function (ECF)
- $w(t) = e^{-\frac{t^2}{2\lambda^2}}$ is a weighting function
- $\phi_0$ is the standard Gaussian $\mathcal{N}(0,1)$

**Theoretical guarantee (Cramer-Wold):**

$$\text{SIGReg}(\mathbf{Z}) \rightarrow 0 \iff \mathbb{P}_{\mathbf{Z}} \rightarrow \mathcal{N}(0, \mathbf{I})$$

By the Cramer-Wold theorem, matching all one-dimensional marginals is equivalent to matching the full joint distribution. This provides a provable anti-collapse guarantee.

**Practical details:**
- Integral approximation via trapezoid quadrature with T nodes in [0.2, 4]
- Default M = 1024 random projections
- Default lambda = 0.1
- Number of projections has negligible impact on downstream performance; lambda is the only effective hyperparameter
- Lambda can be optimized via simple bisection search: O(log n) vs PLDM's O(n^6) grid search
- No stop-gradient, no EMA, no additional stabilization heuristics -- gradients propagate through all components

### Planning via MPC

Given initial observation $o_1$ and goal $o_g$:
1. Encode: $\hat{z}_1 = \text{enc}_\theta(o_1)$ and $z_g = \text{enc}_\theta(o_g)$
2. Initialize candidate action sequence randomly
3. Autoregressive rollout: $\hat{z}_{t+1} = \text{pred}_\phi(\hat{z}_t, a_t)$ up to horizon H
4. Compute terminal cost: $C(\hat{z}_H) = \|\hat{z}_H - z_g\|_2^2$
5. Optimize: $a_{1:H}^* = \arg\min_{a_{1:H}} C(\hat{z}_H)$

**Solver:** Cross-Entropy Method (CEM)
- 300 sampled action sequences per iteration
- 30 optimization iterations
- Top 30 candidates selected as elites at each step
- Only the first K planned actions are executed before replanning from the updated observation (mitigates prediction error accumulation)

## Results

### Environments

| Environment | Type | Description |
|-------------|------|-------------|
| Push-T | 2D manipulation | Pushing a block toward a target configuration |
| OGBench-Cube | 3D manipulation | Robotic arm interacts with a cube |
| Two-Room | 2D navigation | Agent moves between rooms to reach targets |
| Reacher | 2D control | 2-joint arm reaching a target configuration |

All environments have continuous action spaces.

### Planning Performance

- LeWM outperforms PLDM on challenging tasks: **18% higher success rate on PushT**
- LeWM competitive with DINO-WM across tasks
- On PushT, LeWM (pixels-only) surpasses DINO-WM even when DINO-WM has access to additional proprioceptive information
- LeWM performs worse on Two-Room (simplest environment) -- low diversity and low intrinsic dimensionality make it harder for the encoder to match the isotropic Gaussian prior

### Planning Speed

- LeWM achieves **up to 48x faster planning** than DINO-WM, with full planning completing in under one second
- Approximately 200x fewer tokens than DINO-WM for encoding observations
- Planning speeds comparable to PLDM
- Planning time remains consistent across environments for a fixed planning setup

### Physical Probing Results (Push-T)

**Linear Probe:**

| Property | Model | MSE | r |
|----------|-------|-----|---|
| Agent Location | LeWM | 0.052 | 0.974 |
| Agent Location | PLDM | 0.090 | 0.955 |
| Agent Location | DINO-WM | 1.888 | 0.977 |
| Block Location | LeWM | 0.029 | 0.986 |
| Block Location | PLDM | 0.122 | 0.938 |
| Block Location | DINO-WM | 0.006 | 0.997 |
| Block Angle | LeWM | 0.187 | 0.902 |
| Block Angle | PLDM | 0.446 | 0.745 |
| Block Angle | DINO-WM | 0.050 | 0.979 |

**MLP Probe:**

| Property | Model | MSE | r |
|----------|-------|-----|---|
| Agent Location | LeWM | 0.004 | 0.998 |
| Agent Location | PLDM | 0.014 | 0.993 |
| Agent Location | DINO-WM | 0.003 | 0.999 |
| Block Location | LeWM | 0.001 | 0.999 |
| Block Location | PLDM | 0.011 | 0.994 |
| Block Location | DINO-WM | 0.002 | 0.999 |
| Block Angle | LeWM | 0.021 | 0.990 |
| Block Angle | PLDM | 0.056 | 0.972 |
| Block Angle | DINO-WM | 0.009 | 0.995 |

LeWM consistently outperforms PLDM while remaining competitive with DINO-WM. DINO-WM's strong probing performance may stem from its foundation-model pretraining on ~124M images.

### Violation-of-Expectation (VoE)

- LeWM consistently assigns higher surprise to frames containing physical violations
- Significant increase for teleportation perturbations across all environments (paired t-test, p < 0.01)
- Weaker and non-significant for color changes, indicating greater sensitivity to physical than visual perturbations

### Emergent Temporal Straightening

- LeWM's latent trajectories become increasingly straight on PushT over training as a purely emergent phenomenon
- LeWM achieves higher temporal straightness than PLDM, despite PLDM employing a dedicated temporal smoothness regularization term

### Ablation Studies

Key ablations (detailed in Appendix G):
1. **Training variance:** Consistency across runs examined
2. **Embedding dimensions:** Performance quickly saturates beyond a threshold
3. **Number of projections in SIGReg:** Largely unaffected -- no careful tuning needed
4. **Weight of SIGReg regularization (lambda):** The single effective hyperparameter
5. **Predictor size:** Varied predictor capacity
6. **Architecture:** Replaced ViT encoder with ResNet-18 -- competitive with both architectures
7. **Predictor dropout:** Varied dropout rates
8. **Planning solver:** Different planning configurations tested

## Related Work

### Generative World Models
- IRIS, DIAMOND, Delta-IRIS, OASIS, DreamerV4 -- model environments like Minecraft, Counter-Strike, Crafter
- Genie, HunyuanWorld -- generate interactive simulators
- Many generative WMs assume access to reward signals; LeWM focuses on reward-free setting

### JEPA for Self-Supervised Learning
- **I-JEPA** (Assran et al., 2023): images, uses EMA + stop-gradient
- **V-JEPA** (Bardes et al., 2023): video, EMA + stop-gradient
- **V-JEPA 2** (Assran et al., 2025): self-supervised video models for understanding, prediction, and planning
- **Echo-JEPA, Brain-JEPA:** medical data applications
- These approaches use EMA of target encoder with stop-gradient -- theoretical understanding remains limited

### JEPA for Action-Conditioned World Modeling
- Methods using pre-trained encoders: DINO-WM, OSVI-WM, Causal-JEPA, V-JEPA 2
- **PLDM:** learns end-to-end using VICReg with additional regularization; suffers from known training instabilities and scalability limitations

### Planning with Latent Dynamics
- World Models (Ha & Schmidhuber): learning policies from compact latent representations
- Dreamer series: RL in imagination via generative world models
- TD-MPC, TD-MPC2, Navigation World Models, DINO-WM, PLDM: planning at test time via MPC

## Baselines

| Method | Encoder | Collapse Prevention | End-to-End | Hyperparams |
|--------|---------|---------------------|------------|-------------|
| **LeWM** | ViT-Tiny (learned) | SIGReg (Gaussian prior) | Yes | 1 (lambda) |
| **DINO-WM** | DINOv2 (frozen) | Frozen encoder | No | -- |
| **PLDM** | ViT (learned) | VICReg (7-term loss) | Yes | 6 (alpha, beta, gamma, zeta, nu, mu) |
| **GCBC** | DINOv2 patch embeddings | -- | No | -- |
| **GCIVL** | DINOv2 patch embeddings | -- | No | -- |
| **GCIQL** | DINOv2 patch embeddings | -- | No | -- |

PLDM best hyperparameters from grid search (256 configs on Push-T): alpha=18.0, beta=12, gamma=0.2, zeta=0.7, nu=0.0, mu=0.0.

## Limitations

- Planning restricted to short horizons
- Relies on offline datasets with sufficient coverage
- Low data diversity weakens SIGReg in simple, low-dimensional environments where matching a high-dimensional Gaussian prior is harder
- Dependence on action labels (alleviated potentially via inverse dynamics modeling)
- CEM suffers from the curse of dimensionality; no global optimum guarantee in non-convex settings

## Notes

### Personal Thoughts

- **Elegant simplicity:** Reducing the anti-collapse problem to a single Gaussian regularizer with provable convergence is a significant conceptual contribution. The Cramer-Wold theorem provides a clean theoretical foundation that prior heuristic approaches (EMA, stop-gradient, VICReg) lack.
- **Practical impact:** Single hyperparameter (lambda) with O(log n) search is a major practical advantage over PLDM's O(n^6) grid search. This makes the method much more accessible for practitioners.
- **Scaling questions:** The paper uses ViT-Tiny (~5M params encoder). It would be interesting to see how the method scales to larger encoders and more complex environments. The reliance on offline datasets with sufficient coverage is a practical constraint.
- **Connection to LeCun's JEPA vision:** This paper is a direct realization of Yann LeCun's vision for JEPA as a path toward world models -- learning in latent space without pixel-level reconstruction. The fact that LeCun is a co-author reinforces this connection.
- **Surprise detection:** The VoE experiments showing sensitivity to physical but not visual perturbations suggest the latent space captures physical dynamics rather than superficial visual features. This is a strong indicator of meaningful representation learning.
- **Emergent straightening:** The fact that temporal latent trajectories become increasingly straight without any explicit regularization for this property is a fascinating emergent behavior, suggesting the model learns an inherently smooth dynamics model.
- **Comparison with V-JEPA 2:** V-JEPA 2 (same research group) also aims at world modeling but relies on pre-trained encoders. LeWM shows that end-to-end training from pixels is viable with the right anti-collapse mechanism.
