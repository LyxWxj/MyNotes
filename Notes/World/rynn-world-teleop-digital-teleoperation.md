---
type: Note
related_to: "[[world-model]]"
status: Active
url: https://arxiv.org/abs/2607.06558
DOI: 10.48550/arXiv.2607.06558
ZoteroKey: CFN2ZTQ8
tags:
  - robotics
  - world-model
  - real-time
  - action-conditioned
  - teleoperation
---

# RynnWorld-Teleop: An Action-Conditioned World Model for Digital Teleoperation

## Meta

- **Authors**: Haoyu Zhao, Xingyue Zhao, Hangyu Li, Biao Gong, Kehan Li, Siteng Huang, Xin Li, Deli Zhao, Zhongyu Li
- **Affiliation**: Alibaba DAMO Academy
- **Date**: 2026-07-07 (revised 2026-07-12)
- **arXiv**: [2607.06558](https://arxiv.org/abs/2607.06558)
- **Subject**: cs.RO

## Abstract

Introduces digital teleoperation, a paradigm that decouples data collection from physical constraints by replacing real robots with a generative world model. An operator's hand-pose stream drives a robot-centric world model that synthesizes high-fidelity egocentric videos from a single reference image. The recorded poses act as embodiment-agnostic action labels transferable to any target robot via standard retargeting. Enables 40+ FPS, real-time interactive generation on a single H100 GPU.

## Key Contributions

1. **Digital teleoperation paradigm** -- a novel framework that separates data collection from physical hardware and workspace constraints.
2. **RynnWorld-Teleop system** integrating depth-aware skeletal conditioning, progressive training, and autoregressive distillation.
3. **Real-time generation** at 40+ FPS on a single H100 GPU via single-pass inference distillation.
4. **Zero-shot Sim2Real transfer** -- policies trained exclusively on generated data transfer effectively to real robots.

## Method

### Digital Teleoperation Paradigm

- Replaces the real robot with a generative world model during data collection.
- Operator's hand-pose stream drives the world model to synthesize egocentric videos.
- Recorded poses are embodiment-agnostic action labels, transferable to any target robot via retargeting.

### Architecture Components

1. **Depth-aware skeletal conditioning**: provides geometric structure to the generation process.
2. **Progressive human-to-robot training on a video Diffusion Transformer**: gradually transforms human motion into robot-appropriate representations.
3. **Streaming autoregressive distillation**: compresses multi-step generative process into single-pass inference, enabling 40+ FPS.

### Pipeline

- Single reference image + hand-pose stream --> world model --> egocentric video.
- Generated video + pose stream --> complete state-action trajectories for imitation learning.

## Results

- Policies trained **exclusively** on RynnWorld-Teleop-generated data achieved effective zero-shot Sim2Real transfer across multiple bimanual manipulation tasks.
- Handles both dexterous and diverse bimanual tasks.
- Augmenting real-world datasets with digitally teleoperated data consistently improves success rates.
- 40+ FPS real-time generation on a single H100 GPU.

## Related Work

- Prior teleoperation approaches are constrained by physical hardware and workspace limitations.
- Video generation world models (Sora-like architectures) adapted for robotics.
- Action-conditioned video prediction for robot learning.

## Notes

- The "digital teleoperation" concept is clever: decouple data collection from physical constraints using a world model as the robot surrogate.
- 40+ FPS on H100 via autoregressive distillation -- the inference optimization is key. Single-pass distillation reduces the multi-step diffusion process.
- Embodiment-agnostic action labels are a powerful abstraction: collect once with hand poses, retarget to any robot.
- The zero-shot Sim2Real transfer result suggests the world model captures enough physical realism for policy learning.
- Related to my research on inference optimization: the distillation approach for real-time generation is directly applicable.
