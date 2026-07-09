---
type: Note
related_to: "[[3D-Generative-Models]]"
status: Active
url: https://github.com/wendashi/awesome-3D-Generative-Models
---

# Awesome 3D Generative Models

> Similar to the 2D Generative Model (such as Stable Diffusion, Flux) that builds a bridge between text and images, the 3D Generative Model is a bridge between text/images and 3D data.

**3D 创建的 pipeline 比 2D 复杂得多。** 模型分为 3D 物体生成和 3D 场景生成（World Models）。

## 3D Animation Pipeline

```
Modeling (Re-Topology / UV Unwrapping / Artist Mesh)
    → Texture (PBR)
        → Motion (Rigging & Skinning / Animation / Rendering / Lighting / Simulation)
```

---

## 目录

- [Review](#review)
- [Modeling - 3D Representation](#modeling---3d-representation)
- [Modeling - VAE For Mesh Reconstruction](#modeling---vae-for-mesh-reconstruction)
- [Modeling / Texture - Flow-matching DiT For Mesh & Texture Generation](#modeling--texture---flow-matching-dit-for-mesh--texture-generation)
- [Modeling - Re-topology / Artist Mesh Generation](#modeling---re-topology--artist-mesh-generation)
- [Modeling - UV Unwrapping](#modeling---uv-unwrapping)
- [Texture Generation](#texture-generation)
- [Motion - Rigging / Articulation / Dynamics / Animation / Simulation](#motion---rigging--articulation--dynamics--animation--simulation)
- [Acceleration For Mesh Generation](#acceleration-for-mesh-generation)
- [Post-train of 3D Base Models](#post-train-of-3d-base-models)
- [3D Scene Generation (World Models)](#3d-scene-generation-world-models)
- [Common Metrics](#common-metrics)
- [Benchmark](#benchmark)

---

## Review

1. [Feed-Forward-3D](https://fnzhan.com/projects/Feed-Forward-3D/)
2. [Production-Ready 3D Survey](https://github.com/hitcslj/Awesome-AIGC-3D)

---

## Modeling - 3D Representation

1. **VecSet** (TOG 2023) — [GitHub](https://github.com/1zb/3DShape2VecSet) — Used by Hunyuan3D2.1, TripoSG, Step1X-3D
2. **(SLat) TRELLIS** — [GitHub](https://github.com/microsoft/TRELLIS) — Dec 2024, CVPR'25
3. **Ghost on the Shell: An Expressive Representation of General 3D Shapes** — [GitHub](https://github.com/lzzcd001/GShell/) — ICLR 2024 Oral
4. **FastMesh: Efficient Artistic Mesh Generation via Component Decoupling** — [GitHub](https://github.com/jhkim0759/FastMesh) / [Project](https://jhkim0759.github.io/projects/FastMesh/)
5. **Geometry Distributions** — [GitHub](https://github.com/1zb/GeomDist) — ICCV 2025
6. **(VoxSet) LATTICE: Democratize High-Fidelity 3D Generation at Scale** — [Project](https://lattice3d.github.io/) — Nov 2025
7. **(O-Voxel) TRELLIS.2** — [GitHub](https://github.com/microsoft/TRELLIS.2/tree/main/o-voxel) — Dec 2025 → CVPR'26 Oral
8. **Faithful Contouring** — [GitHub](https://github.com/Luo-Yihao/FaithC) — Nov 2025 → CVPR'26 Oral
9. **FACE: A Face-based Autoregressive Representation for High-Fidelity and Efficient Mesh Generation** — [arXiv](https://arxiv.org/abs/2603.01515) — CVPR 2026

---

## Modeling - VAE For Mesh Reconstruction

1. **Sparc3D** — [GitHub](https://github.com/lizhihao6/Sparc3D) — Jun 2025
2. **TripoSF** — [GitHub](https://github.com/VAST-AI-Research/TripoSF) — Mar 2025
3. **Dora** — [GitHub](https://github.com/Seed3D/Dora) — Feb 2025, CVPR'25

---

## Modeling / Texture - Flow-matching DiT For Mesh & Texture Generation

1. **Ultra3D** — [Project](https://buaacyw.github.io/ultra3d/) — Jul 2025
2. **Hunyuan3D-2.1** — [GitHub](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) — Jun 2025
3. **Hunyuan3D-2** — [GitHub](https://github.com/Tencent-Hunyuan/Hunyuan3D-2) — Jan 2025
4. **TripoSG** — [GitHub](https://github.com/VAST-AI-Research/TripoSG) — Mar 2025
5. **Step1X-3D** — [GitHub](https://github.com/stepfun-ai/Step1X-3D) — May 2025 (with training code & dataset)
6. **Direct3D-S2** — [GitHub](https://github.com/DreamTechAI/Direct3D-S2) — May 2025
7. **Hi3DGen** — [GitHub](https://github.com/Stable-X/Stable3DGen) — Apr 2025, ICCV'25
8. **TRELLIS** — [GitHub](https://github.com/microsoft/TRELLIS) — Dec 2024, CVPR'25
9. **UniLat3D** — [GitHub](https://github.com/UniLat3D/UniLat3D) — Sep 2025
10. **UltraShape 1.0** — [GitHub](https://github.com/PKU-YuanGroup/UltraShape-1.0)
11. **Trelli2** — [GitHub](https://github.com/microsoft/TRELLIS.2) — CVPR'26
12. **Pixal3D** — [GitHub](https://github.com/TencentARC/Pixal3D) — SIGGRAPH'26

---

## Modeling - Re-topology / Artist Mesh Generation

1. **BPT** — [GitHub](https://github.com/tencent-hunyuan/bpt) — CVPR'25
2. **DeepMesh** — [GitHub](https://github.com/zhaorw02/DeepMesh) — ICCV'25
3. **MeshMosaic** — [GitHub](https://github.com/Xrvitd/MeshMosaic) — CVPR'26
4. **SATO** — [GitHub](https://github.com/Xrvitd/SATO) — SIGGRAPH'26

---

## Modeling - UV Unwrapping

1. **Auto-Regressive Surface Cutting** — [Project](https://victorcheung12.github.io/seamgpt/) — Jun 2025
2. **ArtUV: Artist-style UV Unwrapping** — [Project](https://chenyg59.github.io/ArtUV/) — Sep 2025
3. **SeamCrafter: Enhancing Mesh Seam Generation for Artist UV Unwrapping via Reinforcement Learning** — [Project](https://chenyg59.github.io/SeamCrafter/) — Sep 2025
4. **PartUV: Part-Based UV Unwrapping of 3D Meshes** — [GitHub](https://github.com/EricWang12/PartUV) — SIGGRAPH Asia'24
5. **MeshTailor: Cutting Seams via Generative Mesh Traversal** — [Project](https://meshtailor.github.io)

---

## Texture Generation

1. **SyncMVD** — [GitHub](https://github.com/LIU-Yuxin/SyncMVD) — SIGGRAPH Asia'24
2. **TEXGen** — [GitHub](https://github.com/CVMI-Lab/TEXGen) — SIGGRAPH Asia'24
3. **ARM** — [Project](https://arm-aigc.github.io/) — CVPR'25
4. **MaterialAnything** — [GitHub](https://github.com/3DTopia/MaterialAnything) — CVPR'25
5. **MaterialMVP** — [GitHub](https://github.com/ZebinHe/MaterialMVP) — ICCV'25
6. **RomanTex** — [GitHub](https://github.com/oakshy/RomanTex) — ICCV'25
7. **NaTex: Meet Native Texture Generation** — [GitHub](https://github.com/Zeqiang-Lai/NaTex) — Nov 2025
8. **SeqTex: Generate Mesh Textures in Video Sequence** — [GitHub](https://github.com/VAST-AI-Research/SeqTex) — SIGGRAPH Asia'25
9. **Chord: Chain of Rendering Decomposition for PBR Material Estimation from Generated Texture images** — [GitHub](https://github.com/ubisoft/ubisoft-laforge-chord) / [Project](https://ubisoft-laforge.github.io/world/chord/) — SIGGRAPH Asia'25
10. **LSRM: High-Fidelity Object-Centric Reconstruction via Scaled Context Windows** — [Project](https://lzqsd.github.io/LSRM.github.io/)

---

## Motion - Rigging / Articulation / Dynamics / Animation / Simulation

1. **Unirig** — [GitHub](https://github.com/VAST-AI-Research/UniRig) — SIGGRAPH'25
2. **AnimateAnyMesh** — [GitHub](https://github.com/JarrentWu1031/AnimateAnyMesh) — ICCV'25
3. **FreeArt3D: Training-Free Articulated Object Generation using 3D Diffusion** — [GitHub](https://github.com/CzzzzH/FreeArt3D) — SIGGRAPH Asia'25
4. **Pixie: Physics from Pixels** — [GitHub](https://github.com/vlongle/pixie) — Aug 2025
5. **SAFT: Shape and Appearance of Fabrics from Template via Differentiable Physical Simulations from Monocular Video** — [GitHub](https://github.com/vc-bonn/saft) — ICCV 2025
6. **BrickGPT: Generating Physically Stable and Buildable Brick Structures from Text** — [GitHub](https://github.com/AvaLovelace1/BrickGPT) — ICCV 2025 Best Paper
7. **PhysX-3D: Physical-Grounded 3D Asset Generation** — [GitHub](https://github.com/ziangcao0312/PhysX-3D) — NeurIPS 2025 Spotlight
8. **PhysX-Anything: Simulation-Ready Physical 3D Assets from Single Image** — [GitHub](https://github.com/ziangcao0312/PhysX-Anything)
9. **PGC: Physics-Based Gaussian Cloth from a Single Pose** — [Project](https://phys-gaussian-cloth.github.io/) — CVPR 2025 Highlight
10. **PhysAvatar: Learning the Physics of Dressed 3D Avatars from Visual Observations** — [GitHub](https://github.com/y-zheng18/PhysAvatar) — ECCV 2024
11. **DSO: Aligning 3D Generators with Simulation Feedback for Physical Soundness** — [GitHub](https://github.com/RuiningLi/dso) — ICCV 2025

---

## Acceleration For Mesh Generation

> lightning vecset decoder

1. **FlashVDM** — [GitHub](https://github.com/Tencent-Hunyuan/FlashVDM) — ICCV'25

---

## Post-train of 3D Base Models

1. **DeepMesh: Auto-Regressive Artist-Mesh Creation With Reinforcement Learning** — [GitHub](https://github.com/zhaorw02/DeepMesh) — ICCV 2025
2. **Nabla-R2D3: Effective and Efficient 3D Diffusion Alignment with 2D Rewards** — [GitHub](https://github.com/MobiusLqm/Nabla-R2D3) / [Project](https://nabla-r2d3.github.io/) — NeurIPS 2025
3. **DreamDPO: Aligning Text-to-3D Generation with Human Preferences via Direct Preference Optimization** — [GitHub](https://github.com/ZhenglinZhou/DreamDPO) — ICML 2025

---

## 3D Scene Generation (World Models)

### Open Source

1. [HY-World-2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0)
2. [WorldFM](https://inspatio.github.io/worldfm/)
3. [Lyra2 (NVIDIA)](https://research.nvidia.com/labs/sil/projects/lyra2/)
4. [World-Grow](https://world-grow.github.io/)
5. [Fantasy World](https://fantasy-amap.github.io/fantasy-world/)
6. [WorldGen](https://worldgen.github.io)

### Closed Source

1. [Marble (World Labs)](https://marble.worldlabs.ai/)
2. [Spaitial](https://spaitial.ai/)
3. [Genie (DeepMind)](https://deepmind.google/models/genie/)
4. [Odyssey](https://odyssey.ml/)
5. [Runway](https://runwayml.com/)
6. [Moonlake AI](https://moonlakeai.com/)

---

## Common Metrics

### Mesh Metrics

| 指标 | 方向 | 说明 |
|------|------|------|
| ULIP-T | ⬆ | [ULIP](https://github.com/salesforce/ULIP) 文本对齐 |
| ULIP-I | ⬆ | ULIP 图像对齐 |
| Uni3D-T | ⬆ | [Uni3D](https://github.com/baaivision/Uni3D) 文本对齐 |
| Uni3D-I | ⬆ | Uni3D 图像对齐 |
| CD (Chamfer Distance) | ⬇ | 几何相似度 |
| NC (Normal Consistency) | ⬇ | 法线一致性 |

> Code: [FlashVDM evaluation/app_metric.py](https://github.com/Tencent-Hunyuan/FlashVDM/blob/main/evaluation/app_metric.py)

### Texture Metrics

| 指标 | 方向 | 说明 |
|------|------|------|
| CLIP-FiD | ⬇ | CLIP 特征空间的 FID |
| CMMD | ⬇ | 多模态距离 |
| CLIP-I | ⬆ | CLIP 图像相似度 |
| LPIPS | ⬇ | 感知相似度 |

---

## Benchmark

1. **HY3D-Bench** — [GitHub](https://github.com/Tencent-Hunyuan/HY3D-Bench)

---

## Tags

`deep-neural-networks` `ai` `deep-learning` `3d` `dit` `diffusion-models` `aigc` `aigc3d` `hunyuan3d`
