---
type: Note
related_to: "[[world-model]]"
status: Active
url: https://arxiv.org/abs/2605.30346
DOI: 10.48550/arXiv.2605.30346
ZoteroKey: RB9GMFKC
tags:
  - world-model
  - video-generation
  - benchmark
  - causality
  - evaluation
---

# YoCausal: How Far is Video Generation from World Model? A Causality Perspective

## Meta

- **Authors**: You-Zhe Xie, Yu-Hsuan Li, Jie-Ying Lee, Kaipeng Zhang, Yu-Lun Liu, Zhixiang Wang
- **Date**: 2026-05
- **arXiv**: [2605.30346](https://arxiv.org/abs/2605.30346)

## Abstract

Addresses whether video diffusion models truly understand causality or just overfit to temporal patterns. Existing benchmarks rely on synthetic data, creating sim-to-real gaps. YoCausal is a two-level benchmark inspired by the Violation of Expectation (VoE) paradigm from cognitive science. It uses temporally reversed real-world videos as natural counterfactual samples at zero cost, enabling an extensible evaluation protocol. Evaluating 13 state-of-the-art VDMs reveals a significant gap persists relative to human-level causal cognition.

## Key Contributions

1. **Two-level benchmark framework** for evaluating causal understanding in video diffusion models.
2. **Reverse Surprise Index (RSI)** -- Level 1 metric quantifying arrow-of-time perception via denoising loss.
3. **Causality Cognition Index (CCI)** -- Level 2 metric leveraging a VLM to stratify datasets into causal and non-causal subsets.
4. **Cost-free counterfactual generation** via temporal reversal of real-world videos.
5. **Arbitrarily extensible evaluation protocol** for ongoing assessment.

## Method

### VoE Paradigm Inspiration

- Draws from the Violation of Expectation paradigm in cognitive science.
- Uses temporally reversed real-world videos as "natural counterfactual samples" at zero cost.

### Level 1: Reverse Surprise Index (RSI)

- Quantifies arrow-of-time perception via denoising loss.
- Assesses whether models can perceive the direction of time.

### Level 2: Causality Cognition Index (CCI)

- Leverages a Vision-Language Model (VLM) to stratify datasets into causal and non-causal subsets.
- Disentangles genuine causal reasoning from temporal bias.

### Two-Level Design

- RSI measures temporal perception (necessary but not sufficient).
- CCI measures genuine causal understanding.
- The two metrics are complementary -- perceiving arrow of time does not imply understanding causality.

## Results

- Evaluated 13 state-of-the-art video diffusion models.
- **Finding 1**: Perceiving the arrow of time does not imply understanding causality -- RSI and CCI measure distinct capabilities.
- **Finding 2**: A significant gap persists relative to human-level causal cognition across all tested models.
- Current VDMs remain far from achieving world-model-level causal understanding, despite advances in temporal pattern learning.

## Related Work

- Video diffusion models (Sora, etc.): the subjects being evaluated.
- Cognitive science VoE paradigm: the theoretical foundation.
- World model benchmarks: prior work often uses synthetic data with sim-to-real gaps.

## Notes

- This is a critical benchmark paper -- it quantifies the gap between video generation and true world modeling.
- The VoE-inspired approach is elegant: using temporal reversal as a zero-cost counterfactual.
- Key insight: temporal pattern recognition (RSI) is not the same as causal reasoning (CCI).
- The finding that all 13 SOTA VDMs fail to match human-level causal cognition is sobering for the field.
- This suggests that scaling video generation alone may not lead to world models -- architectural innovations for causal reasoning are needed.
