---
type: Note
related_to: "[[Audio]]"
status: Active
url: https://arxiv.org/abs/2506.07900
date: 2025-09-04
---

# MiniCPM4: Ultra-Efficient LLMs on End Devices

## 基本信息

- **作者**: MiniCPM Team (Chaojun Xiao, Yuxuan Li, Xu Han, ...)
- **机构**: 清华大学 / 面壁智能
- **时间**: 2025-06-09 (v1), 2025-09-04 (v2)
- **arXiv**: [2506.07900](https://arxiv.org/abs/2506.07900)
- **HuggingFace**: https://huggingface.co/openbmb/MiniCPM4-8B
- **GitHub**: https://github.com/openbmb/minicpm

## 摘要

MiniCPM4 是专为端侧设备设计的高效大语言模型，通过四个维度的系统创新实现效率提升：模型架构、训练数据、训练算法和推理系统。提供 0.5B 和 8B 两个版本，8B 变体在长序列理解与生成上展现出显著的速度提升。

## 核心创新

### 1. 模型架构：InfLLM v2 - 可训练稀疏注意力

**问题**: 自注意力机制的计算和内存需求对端侧设备处理长文档构成挑战

**方案**: 
- 基于动态稀疏注意力架构 InfLLM，引入高效的 kernel 设计和端到端专门训练
- **Token 级稀疏注意力**: 在 query 级别进行 token 级稀疏注意力计算
- **动态上下文块选择**: 将 KV cache 分割为块级单元，每个 query token 选择相关性最高的块
- **注意力稀疏度**: 达到 81% 的注意力稀疏度，同时保持与全注意力机制相当的长上下文处理能力

**与现有方法对比**:
- MoBA: 使用 query 块设计，无法在解码阶段加速
- NSA: 引入三种不同注意力组件，增加参数和存储成本
- InfLLM v2: 不引入额外参数，减少 60% 计算成本

### 2. 训练数据：UltraClean 与 UltraChat v2

#### UltraClean - 高质量预训练数据过滤

- **高效验证策略**: 利用接近训练完成的 LLM 作为基础，在最终训练步骤引入候选语料
- **高效质量分类器**: 基于"高质量种子数据有利于 LLM 训练"的假设
- **应用**: 对 FineWeb 和 Chinese FineWeb 数据集进行过滤，生成 UltraFineWeb

#### UltraChat v2 - 高质量 SFT 数据生成

- 聚焦于深度推理、上下文一致性和任务复杂性的多轮交互
- 结合专家模型和 prompt 工程生成挑战性对话
- 双阶段过滤：自动化验证 + 选择性人工审核

### 3. 训练算法

#### ModelTunnel v2 - 高效预训练策略搜索

- **改进的性能指标**: 使用 ScalingBench 替代语言模型损失作为可预测缩放的性能指标
- **搜索有效性验证**: 系统性验证 μP 结合超参数搜索的有效性

#### Chunk-wise Rollout - 负载均衡强化学习

- 限制每个 rollout 阶段的最大输出 token 预算
- 在后续迭代中恢复未完成轨迹的生成
- 稳定化技术：KL 损失、dual-clip、chunk 级重要性采样、garble filter

#### BitCPM4 - 三值 LLM 的量化感知训练

- 两阶段训练框架，用预训练高精度模型初始化量化阶段
- 使用 10× 更少的训练 token 达到与现有 QAT 方法相当的性能

### 4. 推理系统

#### CPM.cu - 轻量高效 CUDA 推理框架

- 静态内存管理、kernel 融合、高效投机采样实现
- 集成 InfLLM v2 的高效稀疏注意力 kernel
- FR-Spec 改进投机草稿速度
- P-GPTQ: 前缀感知的后训练量化
- SpecMQuant: 投机采样与量化的结合

#### ArkInfer - 跨平台部署系统

- 统一的基于执行器的架构和自适应后端接口
- 集成多种推理框架：NeuroPilot, Genie, RK-LLM, TensorRTLLM, llama.cpp
- 标准化 API 实现无缝跨平台部署

## 训练流程

1. **预训练**: 8.3T 高质量 token，WSD 学习率调度器
   - 7T token 用于 warmup 和稳定阶段
   - 1.3T token 用于退火阶段
2. **长上下文预训练**: 上下文窗口从 4K 扩展到 128K
3. **SFT 后训练**: 启用指令跟随能力
4. **混合推理模型 (MiniCPM4.1)**: 长 CoT 数据 SFT + 数学和编码任务的 RL

## 应用

### MiniCPM4-Survey: 可信综述生成
- 生成长序列、高连贯性和逻辑性的综述

### MiniCPM4-MCP: 工具使用
- 支持 Model Context Protocol，调用复杂函数获取外部资源

## 效率评估

在端侧 GPU 上的推理速度评估（128K 上下文）：
- **Jetson AGX Orin (64G)**: 相比 Llama-3-8B 等模型有显著速度提升
- **RTX 4090 (24G)**: 在 prefilling 和 decoding 上均表现优异

## 个人思考

MiniCPM4 在端侧高效 LLM 领域做了全面的系统性工作：

1. **InfLLM v2 的稀疏注意力设计**非常精巧，不引入额外参数同时实现 81% 稀疏度
2. **UltraClean 的高效验证策略**是亮点，利用接近训练完成的模型评估数据质量
3. **Chunk-wise Rollout** 解决了长 CoT RL 训练中的负载不均衡问题
4. **BitCPM4** 为极端资源受限设备提供了实用方案

对端侧部署有很强的指导意义，特别是稀疏注意力和量化技术的结合应用。
