---
type: Note
related_to: "[[Audio]]"
status: Active
url: https://arxiv.org/abs/2412.10117
date: 2024-12-25
---

# CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models

> [!info] 论文信息
> - **作者**: Zhihao Du, Yuxuan Wang, Qian Chen, Xian Shi, Xiang Lv, Tianyu Zhao, Zhifu Gao, Yexin Yang, Changfeng Gao, Hui Wang, Fan Yu, Huadai Liu, Zhengyan Sheng, Yue Gu, Chong Deng, Wen Wang, Shiliang Zhang, Zhijie Yan, Jingren Zhou
> - **机构**: 阿里巴巴
> - **日期**: 2024-12-25
> - **arXiv**: [2412.10117](https://arxiv.org/abs/2412.10117)
> - **GitHub**: https://github.com/FunAudioLLM/CosyVoice
> - **Demo**: https://funaudiollm.github.io/cosyvoice2

## 摘要

CosyVoice 2 是改进的流式语音合成模型，整合了全面系统的优化。引入**有限标量量化 (FSQ)** 改善语音 token 的 codebook 利用率，简化模型架构以直接使用预训练 LLM 作为骨干，开发 **chunk-aware 因果流匹配模型**支持流式和非流式合成。

> [!tip] 核心成就
> - 达到**人类水平自然度**
> - **最小响应延迟**
> - 流式模式下**几乎无损**的合成质量

## 系统架构概览

> [!abstract] 三阶段解耦设计
> ```
> 输入文本 → BPE Tokenizer → Text-Speech LM (Qwen2.5-0.5B) → Speech Tokens
>                                                                ↓
> 参考音频 → Voice Encoder → Speaker Embedding → Flow Matching → Mel Spectrum → Vocoder → 波形
> ```

> [!note] 设计哲学
> 分离语音的语义和声学信息，独立建模。语音生成过程重新定义为**渐进语义解码过程**。

---

## 1. 文本 Tokenizer

> [!tip] 简化设计
> - 使用 **BPE 文本 tokenizer** 直接处理原始文本
> - **无需前端模型**进行 grapheme-to-phoneme (g2p) 转换
> - 简化数据预处理流程，支持端到端学习发音

> [!warning] 特殊处理
> - **掩码一对多 token**: 如果 BPE token 编码多个中文字符，会被掩码
> - 每个字符单独编码，防止发音过长
> - 其他语言（英文、日文、韩文）不特殊处理

---

## 2. 有限标量量化 (FSQ) 语音 Tokenizer

> [!important] 核心创新
> 在 SenseVoice-Large ASR 模型的 Encoder 中插入 FSQ 模块，替代传统的向量量化 (VQ)

### 架构

```
输入语音 → Encoder1 (6层 Transformer + RoPE) → FSQ → Encoder2 + ASR Decoder → 文本 token
```

### FSQ 量化过程

> [!abstract] 数学表示
> $$\bar{H} = \text{ROUND}(\text{Proj}_{down}(H))$$
> $$\hat{H} = \text{Proj}_{up}(\bar{H})$$
> - 中间表示投影到 **D 维低秩空间**
> - 每维量化到 **[-K, K]**
> - 通过 **(2K+1) 进制系统**计算 token 索引

> [!note] Token 索引计算
> $$\mu_i = \sum_{j=0}^{D-1} \bar{h}_{i,j} (2K+1)^j$$

> [!success] FSQ vs VQ 的优势
> - **更高的 codebook 利用率**
> - 捕获更多语音信息
> - **Token 率: 25 Hz**（每秒 25 个语音 token）

---

## 3. 统一的文本-语音语言模型

> [!important] 架构简化
> - **移除文本编码器**: Qwen2.5-0.5B 足够强大以对齐文本和语音 token
> - **移除说话人嵌入**: 避免信息泄露（包含说话人身份、语言和副语言信息，损害韵律自然度和跨语言能力）

### 统一流式/非流式设计

> [!tip] 核心思想
> 通过**序列构造方式的差异**实现流式和非流式统一

#### 非流式模式

> [!note] 序列格式
> ```
> S, 所有文本 token, T, 所有语音 token, E
> ```
> - **Ignore token**: 损失在交叉熵目标函数中被忽略

#### 流式模式

> [!abstract] 序列格式
> ```
> S, [N:M 混合文本语音], T, 剩余语音 token, E
> ```
> - **N=5, M=15**: 每 5 个文本 token 后跟 15 个语音 token
> - 如果下一个 token 是文本 token，模型预测 **filling token**
> - 文本 token 用完后，添加 T token 和剩余语音

### 推理场景

| 场景 | 流式/非流式 | 序列格式 |
|------|-----------|---------|
| ICL | 非流式 | S, prompt text, text, T, prompt speech |
| ICL | 流式 | S, mixed text speech, T, remaining speech |
| SFT | 非流式 | S, text, T |
| SFT | 流式 | S, mixed text speech, T, remaining speech |

---

## 4. Chunk-aware 因果流匹配模型

> [!important] 设计目标
> 在单个模型中支持流式和非流式合成

### 架构组件

| 组件 | 功能 |
|------|------|
| Causal Transformer Encoder | 因果编码 |
| Causal Conv-Transformer UNet | 因果卷积变换 |
| Causal Upsampling Transformer | 因果上采样 |
| Lookahead PreConv (size=4) | 前瞻预卷积 |

### 因果性设计

> [!note] 四种掩码模式
> - **非因果 mask**: 全局注意力
> - **全因果 mask**: 严格因果
> - **Chunk-M mask**: chunk 级因果
> - **Chunk-2M mask**: 扩展 chunk 因果

> [!tip] 流式合成
> - 语音 token 逐 chunk 生成
> - CFM 模型基于因果设计，支持增量生成
> - 流式和非流式使用**相同的模型参数**

---

## 5. 指令 TTS 能力升级

> [!success] 支持更多指令类型
> - **情感 (emotion)**
> - **口音 (accent)**
> - **角色风格 (role style)**
> - **细粒度控制**

> [!note] 统一设计
> 指令能力和 zero-shot 能力集成到**单一模型**中

---

## 训练细节

### 训练策略

> [!abstract] 训练流程
> 1. **语音 tokenizer 训练**（ASR 目标）
> 2. **文本-语音 LM 训练**（next-token prediction）
> 3. **CFM 模型训练**
> 4. **联合微调**

---

## 实验结果

> [!success] 核心指标
> - **合成质量**: 达到人类水平自然度
> - **流式质量**: 几乎无损的合成质量
> - **响应延迟**: 最小响应延迟
> - **双向流式**: 支持双向流式合成

---

## 与 CosyVoice 1 的改进

> [!note] 版本对比
> | 特性 | CosyVoice | CosyVoice 2 |
> |------|-----------|-------------|
> | 量化方式 | VQ | **FSQ** |
> | 文本编码器 | 有 | **移除** |
> | 说话人嵌入 | 有 | **移除** |
> | 流式支持 | 无 | **统一** |
> | LLM 骨干 | 自定义 | **Qwen2.5-0.5B** |

---

## 技术路线总结

> [!tip] 三阶段解耦设计
> 1. **离散 token 量化**: FSQ 替代 VQ，提高 codebook 利用率
> 2. **LLM 建模**: 统一流式/非流式，简化架构
> 3. **流匹配生成**: Chunk-aware CFM，支持增量生成

> [!important] 设计哲学
> - 分离语义和声学信息
> - 渐进语义解码
> - 架构简化与统一

---

## 个人思考

> [!tip] 研究启示
> 1. **FSQ 替代 VQ** 是关键改进，显著提高 codebook 利用率
> 2. **统一流式/非流式设计** 非常优雅，通过序列构造方式的差异实现
> 3. **移除文本编码器和说话人嵌入** 简化了架构，同时提升了性能
> 4. **Chunk-aware CFM** 为流式 NAR 模型提供了新思路
> 5. 技术路线清晰：离散 token 量化 → LLM 建模 → 流匹配生成，三阶段解耦设计便于优化和扩展
> 6. 为后续 CosyVoice 3 的大规模扩展奠定了坚实基础
