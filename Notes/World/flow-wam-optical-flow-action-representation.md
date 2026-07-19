---
type: Note
related_to: "[[world-model]]"
status: Active
url: https://arxiv.org/abs/2607.13017
DOI: 10.48550/arXiv.2607.13017
ZoteroKey: 7HNJ39PX
tags:
  - robotics
  - world-model
  - diffusion
  - action-representation
  - optical-flow
---

# FlowWAM: Optical Flow as a Unified Action Representation for World Action Models

## Meta

- **Authors**: Yixiang Chen, Peiyan Li, Yuan Xu, Qisen Ma, Jiabing Yang, Kai Wang
- **Date**: 2026-07
- **arXiv**: [2607.13017](https://arxiv.org/abs/2607.13017)

## Abstract

Addresses World Action Models (WAMs) that use pretrained video generators for world modeling and action prediction. The core challenge is how to represent actions in a suitable form that aligns with pretrained video generators while carrying enough motion cues for accurate control. Existing numerical actions fail to align with video generators, and prior visual action representations overlook the temporal motion structure across frames. FlowWAM is a dual-stream diffusion framework that adopts optical flow as a unified, video-native action representation.

## Key Contributions

1. **Optical flow as a unified action representation** -- a video-native format that aligns with pretrained video generators while capturing temporal motion structure.
2. **Dual-stream diffusion framework** enabling both policy and world-model modes within a single architecture.
3. **Unlabeled video pretraining** -- flow extraction from raw videos removes the need for action labels, enabling large-scale data utilization.

## Method

### Optical Flow Action Representation

- Flow videos share the same format as RGB videos and encode rich per-pixel displacement.
- Flow can be easily extracted from raw videos without action labels.
- Serves as a format-compatible substitute for RGB that carries temporal motion information.

### Dual-Stream Diffusion Framework

- Jointly models optical flow and RGB video within a shared pretrained video generator.
- **Policy mode**: generates flow for action prediction.
- **World-model mode**: target flow sequences guide future video generation.

### Unlabeled Video Pretraining

- Since flow can be extracted from raw videos without action labels, FlowWAM can leverage large-scale unlabeled video data for pretraining.
- Removes the bottleneck of requiring paired action-video data.

## Results

- **RoboTwin manipulation**: 92.94% success rate (Clean setting) and 92.14% (Random), surpassing both VLA and WAM baselines.
- **WorldArena world modeling**: Best overall EWMScore of 63.71, with 18.4% relative improvement in trajectory accuracy.
- Flow-based representation delivers gains across both policy and world-model modes.

## Related Work

- Visual Action Representations (VAR): prior work overlooks temporal motion structure.
- Video Action Models (VAM): FlowWAM generalizes action representation to optical flow.
- Pretrained video generators (Sora-like): FlowWAM leverages these as backbone.

## Notes

- Optical flow as action representation is a clever bridge between numerical actions and video generation.
- The dual-mode design (policy + world-model) is elegant -- same framework serves both purposes.
- 92.94% on RoboTwin is impressive -- suggests flow-based actions capture motion well.
- The unlabeled pretraining capability is a significant advantage over methods requiring action annotations.
- The 18.4% improvement in trajectory accuracy on WorldArena shows the world-model mode is also effective.
