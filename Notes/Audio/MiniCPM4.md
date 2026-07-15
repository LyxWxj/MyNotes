---
type: Note
related_to: "[[Audio]]"
status: Active
url: https://arxiv.org/abs/2506.07900
date: 2025-09-04
---

# MiniCPM4: Ultra-Efficient LLMs on End Devices

> [!info] 论文信息
> - **作者**: MiniCPM Team (Chaojun Xiao, Yuxuan Li, Xu Han, ...)
> - **机构**: 清华大学 / 面壁智能
> - **日期**: 2025-06-09 (v1), 2025-09-04 (v2)
> - **arXiv**: [2506.07900](https://arxiv.org/abs/2506.07900)
> - **HuggingFace**: https://huggingface.co/openbmb/MiniCPM4-8B
> - **GitHub**: https://github.com/openbmb/minicpm

## 摘要

MiniCPM4 是专为端侧设备设计的高效大语言模型，通过**四个维度的系统创新**实现效率提升：模型架构、训练数据、训练算法和推理系统。提供 0.5B 和 8B 两个版本，8B 变体在长序列理解与生成上展现出显著的速度提升。

> [!tip] 核心成就
> - 使用 Qwen3-8B **22% 的训练数据**达到可比性能
> - 128K 长文档处理速度提升 **7 倍**
> - 在端侧 GPU 上实现高效推理

## 核心创新概览

> [!abstract] 四大维度创新
> 1. **模型架构**: InfLLM v2 可训练稀疏注意力
> 2. **训练数据**: UltraClean 数据过滤 + UltraChat v2 SFT 数据
> 3. **训练算法**: ModelTunnel v2 + Chunk-wise Rollout + BitCPM4
> 4. **推理系统**: CPM.cu + ArkInfer 跨平台部署

---

## 1. 模型架构：InfLLM v2

> [!important] 核心问题
> 自注意力机制的计算和内存需求对端侧设备处理长文档构成重大挑战。

### 整体框架

> [!note] InfLLM v2 设计
> - 将 KV cache 分割为**块级单元**
> - 每个 query token 选择**相关性最高的块**进行注意力计算
> - 初始 token 和滑动窗口中的局部 token 始终被选择
> - 达到 **81% 的注意力稀疏度**，同时保持与全注意力相当的性能

### 动态上下文块选择

> [!tip] 两阶段选择策略
> 1. **语义 Kernel 计算**: 为每个块计算语义表示
> 2. **相关性评分 + Top-K 选择**: 基于相关性分数选择最相关的块

### 与现有方法对比

| 方法 | 解码加速 | Token 级稀疏 | 额外参数 | 计算成本降低 |
|------|---------|-------------|---------|------------|
| MoBA | ✗ | ✗ (块级) | - | - |
| NSA | ✓ | ✓ | 3 倍 KV 存储 | - |
| **InfLLM v2** | **✓** | **✓** | **无** | **60%** |

> [!success] InfLLM v2 的优势
> - 不引入额外参数，不影响短序列推理
> - Token 级稀疏注意力，相邻 token 可选择不同上下文
> - 同时加速 prefilling 和 decoding

---

## 2. 训练数据

### UltraClean - 高质量预训练数据过滤

> [!important] 核心创新：高效验证策略
> - **传统方法**: 从头训练 LLM 验证候选语料质量（成本高）
> - **UltraClean**: 利用**接近训练完成的 LLM**，在最终训练步骤引入候选语料
> - 使用性能提升作为数据质量指标

> [!note] 质量分类器设计
> - 基于假设："高质量种子数据有利于 LLM 训练"
> - 精心平衡正负样本集
> - 应用到 FineWeb 和 Chinese FineWeb → **UltraFineWeb**

### UltraChat v2 - 高质量 SFT 数据

> [!tip] 数据生成策略
> - 聚焦于**深度推理、上下文一致性和任务复杂性**的多轮交互
> - 推理类型：多跳推理、常识推理、领域特定问题解决
> - **双阶段过滤**: 自动化验证 + 选择性人工审核

---

## 3. 训练算法

### ModelTunnel v2 - 高效预训练策略搜索

> [!note] 两方面改进
> 1. **改进的性能指标**: 使用 ScalingBench 替代 LM 损失
>    - 解决 emergent abilities 导致的预测失效问题
>    - 建立 ScalingBench 损失与下游性能的关系
> 2. **搜索有效性验证**: 系统性验证 μP + 超参数搜索的有效性

### Chunk-wise Rollout - 负载均衡强化学习

> [!warning] 问题
> 长 CoT RL 训练中，不同轨迹长度差异导致 GPU 负载不均衡

> [!success] 解决方案
> - 限制每个 rollout 阶段的**最大输出 token 预算**
> - 在后续迭代中**恢复未完成轨迹**的生成
> - **稳定化技术**:
>   - KL 损失
>   - Dual-clip
>   - Chunk 级重要性采样
>   - Garble filter

### BitCPM4 - 三值 LLM

> [!tip] 量化感知训练
> - **两阶段框架**: 用预训练高精度模型初始化量化阶段
> - 使用 **10× 更少的训练 token** 达到与现有 QAT 方法相当的性能
> - 为极端资源受限设备提供实用方案

---

## 4. 推理系统

### CPM.cu - CUDA 推理框架

> [!abstract] 核心组件
> - 静态内存管理
> - Kernel 融合
> - 高效投机采样实现
> - **FR-Spec**: 改进投机草稿速度
> - **P-GPTQ**: 前缀感知的后训练量化
> - **SpecMQuant**: 投机采样与量化的结合

### ArkInfer - 跨平台部署

> [!note] 支持的推理框架
> - NeuroPilot (MediaTek)
> - Genie (Qualcomm)
> - RK-LLM (Rockchip)
> - TensorRTLLM (NVIDIA)
> - llama.cpp (CPU)

---

## 训练流程

> [!abstract] 完整训练流程
> 1. **预训练**: 8.3T 高质量 token
>    - WSD 学习率调度器
>    - 7T token warmup + 稳定阶段
>    - 1.3T token 退火阶段
> 2. **长上下文预训练**: 4K → 128K
> 3. **SFT 后训练**: 指令跟随能力
> 4. **MiniCPM4.1 混合推理模型**:
>    - 长 CoT 数据 SFT
>    - 数学和编码任务的 RL

## 应用

### MiniCPM4-Survey: 可信综述生成
> 生成长序列、高连贯性和逻辑性的综述

### MiniCPM4-MCP: 工具使用
> 支持 Model Context Protocol，调用复杂函数获取外部资源

## 效率评估

> [!success] 端侧 GPU 推理速度（128K 上下文）
> - **Jetson AGX Orin (64G)**: 相比 Llama-3-8B 等模型有显著速度提升
> - **RTX 4090 (24G)**: 在 prefilling 和 decoding 上均表现优异

## 个人思考

> [!tip] 研究启示
> 1. **InfLLM v2 的稀疏注意力设计**非常精巧，不引入额外参数同时实现 81% 稀疏度
> 2. **UltraClean 的高效验证策略**是亮点，利用接近训练完成的模型评估数据质量
> 3. **Chunk-wise Rollout** 解决了长 CoT RL 训练中的负载不均衡问题
> 4. **BitCPM4** 为极端资源受限设备提供了实用方案
> 5. 四个维度的系统创新思路值得学习
