---
type: Note
related_to: "[[Audio]]"
status: Active
url: https://arxiv.org/abs/2412.10117
date: 2024-12-25
---

# CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models

## 基本信息

- **作者**: Zhihao Du, Yuxuan Wang, Qian Chen, Xian Shi, Xiang Lv, Tianyu Zhao, Zhifu Gao, Yexin Yang, Changfeng Gao, Hui Wang, Fan Yu, Huadai Liu, Zhengyan Sheng, Yue Gu, Chong Deng, Wen Wang, Shiliang Zhang, Zhijie Yan, Jingren Zhou
- **机构**: 阿里巴巴
- **时间**: 2024-12-25
- **arXiv**: [2412.10117](https://arxiv.org/abs/2412.10117)
- **GitHub**: https://github.com/FunAudioLLM/CosyVoice
- **Demo**: https://funaudiollm.github.io/cosyvoice2

## 摘要

CosyVoice 2 是改进的流式语音合成模型，整合了全面系统的优化。引入有限标量量化 (FSQ) 改善语音 token 的 codebook 利用率，简化模型架构以直接使用预训练 LLM 作为骨干，开发 chunk-aware 因果流匹配模型支持流式和非流式合成。在大规模多语言数据集上训练，达到人类水平自然度、最小响应延迟和几乎无损的流式合成质量。

## 核心创新

### 1. 有限标量量化 (FSQ) 语音 Tokenizer

**架构**:
- 在 SenseVoice-Large ASR 模型的 Encoder 中插入 FSQ 模块
- Encoder1（6 层 Transformer + RoPE）→ FSQ → Encoder2 + ASR Decoder

**FSQ 量化过程**:
$$\bar{H} = \text{ROUND}(\text{Proj}_{down}(H))$$
$$\hat{H} = \text{Proj}_{up}(\bar{H})$$

- 中间表示投影到 D 维低秩空间
- 每维量化到 [-K, K]
- 通过 (2K+1) 进制系统计算 token 索引

**优势**: 相比向量量化 (VQ)，FSQ 提高 codebook 利用率，捕获更多语音信息

**Token 率**: 25 Hz

### 2. 统一的文本-语音语言模型

**架构简化**:
- **移除文本编码器**: Qwen2.5-0.5B 足够强大以对齐文本和语音 token
- **移除说话人嵌入**: 避免信息泄露（包含说话人身份、语言和副语言信息）

**统一流式/非流式模式**:

#### 非流式模式
```
S, 所有文本 token, T, 所有语音 token, E
```
- Ignore token: 损失在交叉熵目标函数中被忽略

#### 流式模式
```
S, [N:M 混合文本语音], T, 剩余语音 token, E
```
- N=5, M=15：每 5 个文本 token 后跟 15 个语音 token
- 如果下一个 token 是文本 token，模型预测 filling token
- 文本 token 用完后，添加 T token 和剩余语音

**推理场景**:
| 场景 | 流式/非流式 | 序列格式 |
|------|-----------|---------|
| ICL | 非流式 | S, prompt text, text, T, prompt speech |
| ICL | 流式 | S, mixed text speech, T, remaining speech |
| SFT | 非流式 | S, text, T |
| SFT | 流式 | S, mixed text speech, T, remaining speech |

### 3. Chunk-aware 因果流匹配模型

**设计目标**: 在单个模型中支持流式和非流式合成

**架构组件**:
- Causal Transformer Encoder
- Causal Conv-Transformer UNet
- Causal Upsampling Transformer
- Lookahead PreConv (size=4)

**因果性设计**:
- 非因果 mask: 全局注意力
- 全因果 mask: 严格因果
- Chunk-M mask: chunk 级因果
- Chunk-2M mask: 扩展 chunk 因果

**流式合成**:
- 语音 token 逐 chunk 生成
- CFM 模型基于因果设计，支持增量生成
- 流式和非流式使用相同的模型参数

### 4. 指令 TTS 能力升级

支持更多指令类型：
- 情感 (emotion)
- 口音 (accent)
- 角色风格 (role style)
- 细粒度控制

指令能力和 zero-shot 能力集成到单一模型中

## 训练细节

### 数据
- 大规模多语言数据集
- 覆盖中文、英文、日文、韩文等

### 训练策略
1. 语音 tokenizer 训练（ASR 目标）
2. 文本-语音 LM 训练（next-token prediction）
3. CFM 模型训练
4. 联合微调

## 实验结果

### 合成质量
- 达到人类水平自然度
- 流式模式下几乎无损的合成质量

### 响应延迟
- 最小响应延迟
- 支持双向流式合成

### 与 CosyVoice 1 的改进
| 特性 | CosyVoice | CosyVoice 2 |
|------|-----------|-------------|
| 量化方式 | VQ | FSQ |
| 文本编码器 | 有 | 移除 |
| 说话人嵌入 | 有 | 移除 |
| 流式支持 | 无 | 统一 |
| LLM 骨干 | 自定义 | Qwen2.5-0.5B |

## 架构图解

```
输入文本 → BPE Tokenizer → Text-Speech LM (Qwen2.5-0.5B) → Speech Tokens
                                                              ↓
参考音频 → Voice Encoder → Speaker Embedding → Flow Matching → Mel Spectrum → Vocoder → 波形
```

## 个人思考

CosyVoice 2 是 CosyVoice 系列的重要里程碑：

1. **FSQ 替代 VQ** 是关键改进，显著提高 codebook 利用率
2. **统一流式/非流式设计** 非常优雅，通过序列构造方式的差异实现
3. **移除文本编码器和说话人嵌入** 简化了架构，同时提升了性能
4. **Chunk-aware CFM** 为流式 NAR 模型提供了新思路

技术路线清晰：离散 token 量化 → LLM 建模 → 流匹配生成，三阶段解耦设计便于优化和扩展。

对后续 CosyVoice 3 的大规模扩展奠定了坚实基础。
