---
type: Note
related_to: "[[world-model]]"
status: Active
url: https://arxiv.org/abs/2604.18564
DOI: 10.48550/arXiv.2604.18564
ZoteroKey: RIDPK2Y6
tags:
  - world-model
  - video-generation
  - multi-agent
  - multi-view
  - scalable
---

# MultiWorld: Scalable Multi-Agent Multi-View Video World Models

## Meta

- **Authors**: Haoyu Wu, Jiwen Yu, Yingtian Zou, Xihui Liu
- **Date**: 2026-04
- **arXiv**: [2604.18564](https://arxiv.org/abs/2604.18564)

## Abstract

A unified framework for multi-agent multi-view world modeling that supports flexible scaling of agent and view counts, and synthesizes different views in parallel. Most existing work handles only single-agent settings and cannot capture complex multi-agent interactions. MultiWorld addresses this with precise multi-agent control and multi-view consistency.

## Key Contributions

1. **Unified multi-agent multi-view framework** extending world models from single-agent to multi-agent multi-view settings.
2. **Multi-Agent Condition Module** enabling precise multi-agent controllability.
3. **Global State Encoder** ensuring coherent observations across different views.
4. **Flexible scaling** of both agent counts and view counts.
5. **Parallel synthesis** of different views for efficiency.

## Method

### Multi-Agent Condition Module

- Injects action conditioning for multiple agents simultaneously.
- Handles distinct per-agent controls within a single generation framework.
- Enables precise multi-agent controllability.

### Global State Encoder

- Ensures observations generated from different viewpoints remain coherent.
- Enforces multi-view consistency across synthesized frames.
- Maintains a shared global state that informs all view generators.

### Parallel Synthesis

- Rather than generating views sequentially, synthesizes different views in parallel.
- Improves throughput as view counts scale.

## Results

- Evaluated on two domains: **multi-player game environments** and **multi-robot manipulation tasks**.
- Outperforms baselines in video fidelity, action-following ability, and multi-view consistency across both settings.
- Paper is 15 pages with 10 figures.

## Related Work

- Single-agent world models: MultiWorld generalizes these to multi-agent settings.
- Multi-view video generation: prior work often handles views independently.
- Game simulators and robot manipulation: the two evaluation domains.

## Notes

- The multi-agent scaling is important for realistic simulation -- real-world driving involves many agents.
- Parallel view synthesis is a practical design choice for efficiency.
- The Global State Encoder for cross-view consistency is a key architectural insight.
- Relevant to autonomous driving world models (like CausalDrive) where multi-agent interaction is critical.
