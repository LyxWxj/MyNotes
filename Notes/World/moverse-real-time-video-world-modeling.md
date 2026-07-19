---
type: Note
related_to: "[[world-model]]"
status: Active
tags:
  - gaussian-splatting
  - world-model
  - real-time
  - panoramic
  - scene-roaming
  - video-generation
---

# MoVerse: Real-Time Video World Modeling with Panoramic Gaussian Scaffold

## 元信息

| 字段 | 内容 |
|------|------|
| **作者** | Yang Zhou, Ziheng Wang, Yuqin Lu, Haofeng Liu, Jun Liang, Shengfeng He, Jing Li |
| **日期** | 2026-06 |
| **arXiv** | [2606.13376](https://arxiv.org/abs/2606.13376) |
| **URL** | https://arxiv.org/abs/2606.13376 |
| **Zotero Key** | VRMJJJDS |
| **项目主页** | https://orange-3dv-team.github.io/MoVerse/ |

## 摘要 (Abstract)

MoVerse 是一个实时 video world model，能够从单张窄视角图像创建可交互导航的场景。系统解决的核心挑战是：输入仅观察到环境的一小部分，而漫游需要完整的周围世界，包括持久的几何结构和一致的输出。

## 核心贡献 (Key Contributions)

1. **三阶段流水线** — 全景扩展 -> 3D Gaussian Scaffold 构建 -> Gaussian 条件视频渲染
2. **拓扑感知扩散 (Topology-aware Diffusion)** — 将窄视角输入扩展为重力对齐的 360 度全景
3. **实时渲染** — 在单张 RTX 4090 GPU 上实现 8 FPS 的实时场景漫游
4. **世界构建与观察渲染分离** — 核心设计原则

## 方法 (Method)

### 三阶段流水线

#### Stage 1: 全景扩展 (Panoramic Expansion)
- 输入：单张窄视角图像
- 使用**拓扑感知扩散**将输入扩展为重力对齐的 360 度全景
- 在任何 3D 推理之前填充缺失的视野

#### Stage 2: 3D Gaussian Scaffold 构建
- 将全景图提升为**持久的 3D Gaussian scaffold**
- 使用全景几何感知的残差预测 (panoramic geometry-aware residual prediction)
- 创建密集的、可渲染的空间记忆

#### Stage 3: Gaussian 条件视频渲染
- 沿用户指定的相机轨迹渲染 scaffold
- 使用渲染结果作为条件生成照片级真实的视频
- **双向扩散教师 (Bidirectional Diffusion Teacher)**: 训练用于高质量输出
- **因果自回归学生 (Causal Autoregressive Student)**: 蒸馏用于有界延迟流式传输

### 设计理念

- 结合显式 3D 表示的**可控性和长距离一致性**与生成视频模型的**感知质量**
- 世界构建与观察渲染分离，使得 3D 结构持久化

## 实验结果 (Results)

- 在单张 NVIDIA RTX 4090 GPU 上实现 **8 FPS** 实时场景漫游
- 从单张窄视角图像创建可交互导航的完整场景
- 展示了从单张图像进行世界创建并实现交互式视频输出的实用路径

## 与其他工作的关系 (Related Work)

- **3D Gaussian Splatting**: 使用 Gaussian 表示作为 3D scaffold 的基础
- **Panoramic Image Generation**: 与全景图生成和外绘 (outpainting) 相关
- **Video World Models**: 与 Sora 等 video generation 模型的区别在于实时性和 3D 一致性
- **NeRF/3D Reconstruction**: 与 3D 重建相关，但侧重于生成而非重建
- **Scene Generation**: 与场景生成和 novel view synthesis 相关

## 个人笔记 (Notes)

- **三阶段设计**非常巧妙：先补全全景（2D），再构建 3D 结构，最后渲染视频
- **双向扩散教师 -> 因果自回归学生**的蒸馏策略是实现实时性的关键
- 8 FPS 在单卡上实现实时漫游是一个不错的性能指标
- Gaussian scaffold 作为持久化的 3D 表示，避免了每帧重新计算
- 这个工作将 world model 从"预测未来"扩展到了"创建可导航的世界"
- 与 NeRF-based 方法相比，Gaussian 表示在渲染速度上有天然优势
