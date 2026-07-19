---
type: Note
related_to: "[[world-model]]"
status: Active
url: http://arxiv.org/abs/2511.07399
tags:
  - world-model
  - video-generation
  - streaming
  - pipeline-parallelism
  - system
---

# StreamDiffusionV2: A Streaming System for Dynamic and Interactive Video Generation

## 元信息

| 字段 | 内容 |
|------|------|
| **作者** | Tianrui Feng, Zhi Li, Shuo Yang, Haocheng Xi, Muyang Li, Xiuyu Li, Lvmin Zhang, Keting Yang, Kelly Peng, Song Han, Maneesh Agrawala, Kurt Keutzer, Akio Kodaira, Chenfeng Xu |
| **会议** | MLSys 2026 |
| **日期** | 2026-02-22 |
| **arXiv ID** | 2511.07399 |
| **URL** | http://arxiv.org/abs/2511.07399 |
| **项目主页** | http://streamdiffusionv2.github.io |

## 摘要

StreamDiffusionV2 是一个无需训练（training-free）的交互式直播视频生成流水线。之前的图像级流式扩散模型受限于时间一致性；离线视频扩散系统主要优化吞吐量而非实时延迟。StreamDiffusionV2 针对直播场景的严格 SLO（首帧延迟最小、每帧满足 deadline、低抖动）进行设计，集成 SLO-aware 批调度器、block 调度器、sink-token 引导的滚动 KV cache、运动感知噪声控制器等系统级优化，并引入可扩展的流水线编排，在去噪步和网络层之间并行化扩散过程，实现近线性 FPS 扩展。

**性能数据**：无需 TensorRT 或量化，4x H100 上 14B 模型首帧 0.5s 内渲染，达到 58.28 FPS；1.3B 模型达到 64.52 FPS。

## 核心贡献

1. **SLO-aware 批调度器与 block 调度器**：为实时直播场景设计，确保首帧延迟和逐帧 deadline
2. **Sink-token 引导的滚动 KV Cache**：在流式视频生成中维护时间一致性
3. **运动感知噪声控制器**：根据运动动态调整噪声水平
4. **可扩展流水线编排**：跨去噪步和网络层并行化，实现近线性多 GPU 扩展
5. **异构 GPU 支持**：无缝扩展到异构 GPU 环境，支持灵活去噪步数（1-4步）

## 方法

- **训练无关**：不修改模型权重，纯系统级优化
- **流水线并行**：将扩散过程在去噪步骤和网络层两个维度上并行化
- **滚动 KV Cache**：借鉴 LLM 推理的 KV Cache 技术，用 sink token 引导保持长期一致性
- **SLO 感知调度**：区分在线直播（低延迟）和离线批处理（高吞吐）的调度策略
- **灵活去噪步数**：支持 1-4 步，平衡超低延迟和更高质量两种模式

## 实验结果

| 配置 | 首帧延迟 | FPS |
|------|---------|-----|
| 14B 模型, 4x H100 | < 0.5s | 58.28 |
| 1.3B 模型, 4x H100 | - | 64.52 |

- 不依赖 TensorRT 或量化，纯系统优化即达到此性能
- 近线性 FPS 扩展，不违反延迟保证

## 与其他工作的关系

- **StreamDiffusion v1**：基于图像级扩散，时间一致性有限，本工作升级到视频扩散
- **xDiT/FastVideo**：面向离线视频生成，优化吞吐量；StreamDiffusionV2 面向实时直播，优化延迟
- **Inferix**：面向世界模拟的 block-diffusion 推理引擎；StreamDiffusionV2 面向实时直播场景
- **vLLM/SGLang**：LLM 推理引擎；StreamDiffusionV2 将类似 SLO 调度思想引入视频生成

## 笔记

- 直播视频生成是一个非常实际的应用场景，SLO 约束（首帧延迟、逐帧 deadline）比离线生成严格得多
- "训练无关"的系统优化思路很有价值——不改模型权重，纯靠调度、并行、KV Cache 管理提升性能
- Sink-token 引导的滚动 KV Cache 是一个巧妙的技术，用少量关键 token 维护长期一致性
- 流水线并行（跨去噪步 + 跨网络层）实现近线性扩展，对多 GPU 部署很实用
- 该工作与 Inferix 形成互补：Inferix 关注世界模拟的长视频生成，StreamDiffusionV2 关注实时直播的低延迟生成
