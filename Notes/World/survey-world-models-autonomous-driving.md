---
type: Note
related_to: "[[world-model]]"
status: Active
url: https://arxiv.org/abs/2501.11260
DOI: 10.48550/arXiv.2501.11260
ZoteroKey: 7HNR4PNC
tags:
  - world-model
  - survey
  - autonomous-driving
---

# A Survey of World Models for Autonomous Driving

## Meta

- **Authors**: Tuo Feng, Wenguan Wang, Yi Yang
- **Date**: 2025-01 (v4: 2025-09-10)
- **arXiv**: [2501.11260](https://arxiv.org/abs/2501.11260)
- **Subject**: cs.RO, cs.CV
- **License**: CC BY 4.0
- **Resources**: [AwesomeWMAD](https://github.com/FengZicai/AwesomeWMAD) | [WMAD-Benchmarks](https://github.com/FengZicai/WMAD-Benchmarks)

## Abstract

A comprehensive survey of world models specifically designed for autonomous driving applications. The paper reviews advances in world models for AD, proposing a three-tiered taxonomy covering: (1) generation of future physical world representations using methods like diffusion models and 4D occupancy forecasting; (2) behavior planning combining rule-driven and learning-based approaches with cost map optimization and reinforcement learning; and (3) interaction between prediction and planning for multi-agent collaborative decision-making.

## Key Contributions

1. **Proposed a systematic three-tiered taxonomy** for categorizing world models in the autonomous driving domain.
2. **Comprehensive review of generation methods** spanning Image-, BEV-, OG-, and PC-based approaches for modeling scene evolution.
3. **Analysis of training paradigms** including self-supervised learning, multimodal pretraining, and generative data augmentation.
4. **Future research directions** highlighting challenges in self-supervised representation learning, multimodal fusion, and advanced simulation.

## Method

The survey organizes world models into three tiers:

- **Tier 1 -- Generation of Future Physical World**: Covers multiple representation formats (image, bird's-eye view, occupancy grids, point clouds) and leverages modern generative techniques (diffusion models, 4D occupancy forecasting).
- **Tier 2 -- Behavior Planning**: Merges rule-based and learned paradigms, using cost maps and RL for trajectory generation under complex traffic scenarios.
- **Tier 3 -- Prediction-Planning Interaction**: Achieves collaborative multi-agent decision-making through latent space diffusion and memory-augmented architectures.

## Results

- World models serve as a "linchpin technology" providing high-fidelity driving environment representations that integrate multimodal sensor data, semantics, and temporal dynamics.
- These models have "fundamentally transforming how vehicles interpret dynamic scenes and execute safe decision-making."
- The survey provides a "technical roadmap for harnessing the transformative potential of world models" toward safer autonomous driving.

## Related Work

This survey provides a meta-level overview connecting numerous prior works. Key related areas:
- Diffusion-based world models for driving (GAIA-1, DriveDreamer, etc.)
- BEV representation methods (BEVFormer, BEVDet)
- Occupancy forecasting (SurroundOcc, Occ3D)
- Action-conditioned video generation for driving

## Notes

- This is a good entry-point survey for understanding the landscape of world models in AD.
- The three-tiered taxonomy is a clean organizational framework.
- The accompanying GitHub repos (AwesomeWMAD, WMAD-Benchmarks) are useful for tracking related papers.
- Relevant to my research on DiT inference optimization -- many driving world models use diffusion architectures that face real-time inference challenges.
