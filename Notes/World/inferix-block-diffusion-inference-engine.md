---
type: Note
related_to: "[[world-model]]"
status: Active
url: http://arxiv.org/abs/2511.20714
tags:
  - world-model
  - video-generation
  - block-diffusion
  - inference-engine
---

# Inferix: A Block-Diffusion based Next-Generation Inference Engine for World Simulation

## 元信息

| 字段 | 内容 |
|------|------|
| **作者** | Inferix Team: Tianyu Feng, Yizeng Han, Jiahao He, Yuanyu He, Xi Lin, Teng Liu, Hanfeng Lu, Jiasheng Tang, Wei Wang, Zhiyuan Wang, Jichao Wu, Mingyang Yang, Yinghao Yu, Zeyu Zhang, Bohan Zhuang |
| **机构** | Zhejiang University, HKUST, Alibaba DAMO Academy, Alibaba TRE |
| **日期** | 2026-04-28 (v2) |
| **arXiv ID** | 2511.20714 |
| **URL** | http://arxiv.org/abs/2511.20714 |
| **代码** | https://github.com/alibaba-damo-academy/Inferix |

## 摘要

World models 是 agentic AI、embodied AI 和 gaming 领域的核心模拟器，能够生成长时、物理真实且可交互的高质量视频。Inferix 专门设计为下一代推理引擎，通过优化的半自回归（semi-autoregressive / block-diffusion）解码过程实现沉浸式世界合成。

关键创新在于 **block-diffusion 解码范式**：将扩散模型和自回归方法的优势结合——在每个 block 内应用扩散生成视频 token，同时以之前的 block 作为条件，从而生成更连贯稳定的视频序列。它通过重新引入 LLM 风格的 KV Cache 管理，克服了标准视频扩散的限制，实现了高效、可变长度、高质量的生成。

## 核心贡献

1. **提出 block-diffusion 专用推理引擎**：区别于高并发 LLM 推理系统（vLLM、SGLang）和经典视频扩散模型（xDiT），专门为世界模拟设计
2. **集成 InterVBench 评估基准**：面向分钟级长视频生成的细粒度评估，提出 Video Drift Error (VDE) 指标族
3. **支持交互式视频流**：RTMP 和 WebRTC 协议支持，实现动态叙事控制（不同 segment 使用不同 prompt）
4. **高级 KV Cache 管理**：支持 block-wise 内存管理、滑动窗口访问、MLA latent store、offload 到主存
5. **分布式世界合成**：支持 Ulysses 序列并行、Ring Attention 等多种并行策略
6. **内置性能分析**：零开销 profiler，支持自定义指标

## 方法

### 架构对比

| 特性 | 自回归 (AR) | 扩散 (Diffusion) | Block Diffusion (Semi-AR) |
|------|------------|-----------------|--------------------------|
| 生成长度 | 固定 | 固定 | **任意长度** |
| KV Cache | 有 | 无 | **有** |
| 可并行性 | 不可并行 | 可并行 | **block 内可并行** |

### 框架设计

- **并行策略**：Ulysses-style 序列并行 + Ring Attention，根据模型架构和网络拓扑自适应选择
- **KV 管理**：统一 KV 管理接口，支持 range-based chunked access 和 index-based selective fetch
- **支持的模型**：MAGI-1、CausVid、Self Forcing 等 block diffusion 模型
- **视频流**：支持不同视频 chunk 使用不同 prompt 控制，切换 prompt 时清除 cross-attention cache
- **量化**：集成 DAX 量化加速

### InterVBench

- 包含 1000 个长视频（>50秒），来源 DanceTrack、GOT-10k、HD-VILA-100M、ShareGPT4V
- 使用 GPT-4o 作为数据引擎，每 2-3 秒生成详细描述
- **VDE 指标族**：VDE-Clarity、VDE-Motion、VDE-Aesthetic、VDE-Background、VDE-Subject
- 同时集成 VBench 的 5 个维度：Subject Consistency、Background Consistency、Motion Smoothness、Aesthetic Quality、Image Quality

### 世界模拟推理的挑战

- **存储瓶颈**：KV Cache 占用大量 GPU 内存（需要保存前序 block 的上下文以避免 drift 和 forgetting）
- **计算瓶颈**：Wan2.1 14B 在单张 H20 上生成 5 秒视频需约 6800 秒，世界模拟更甚
- **解决方案**：量化（低比特计算）、稀疏注意力、减少去噪步数、利用推理冗余、分布式计算

## 实验结果

论文主要为系统设计和基准贡献，未报告详细的数值实验结果。InterVBench 提供了系统的评估框架。

## 与其他工作的关系

- **与 vLLM/SGLang 的区别**：那些面向高并发 LLM 推理，Inferix 专注于世界模拟的视频生成
- **与 xDiT/FastVideo 的区别**：那些面向经典视频扩散模型，Inferix 针对 block-diffusion 范式
- **与 MAGI-1 的关系**：MAGI-1 是从头训练的 block diffusion 模型，Inferix 支持其推理
- **与 CausVid/Self Forcing 的关系**：基于 Wan2.1 的 block diffusion 模型

## 笔记

- Block diffusion 是 AR 和 diffusion 之间的插值，在每个 block 内用 diffusion，block 间用自回归条件，兼具两者优势
- KV Cache 管理是从 LLM 推理借鉴到视频生成领域的关键技术，PageAttention、offload、压缩等均可复用
- InterVBench 的 VDE 指标思路很好——用相对变化率衡量长视频的时间一致性
- 该工作更偏系统/工程贡献，核心 insight 是"世界模型时代需要专用推理引擎，正如 LLM 时代需要 vLLM"
- 开源地址值得关注：https://github.com/alibaba-damo-academy/Inferix
