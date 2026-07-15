---
type: Note
related_to: "[[Audio]]"
status: Active
url: https://arxiv.org/abs/2505.17589
date: 2025-05-27
---

# CosyVoice 3: Towards In-the-wild Speech Generation via Scaling-up and Post-training

> [!info] 论文信息
> - **作者**: Zhihao Du, Changfeng Gao, Yuxuan Wang, Fan Yu, Tianyu Zhao, Hao Wang, Xiang Lv, Hui Wang, Chongjia Ni, Xian Shi, Keyu An, Guanrou Yang, Yabin Li, Yanni Chen, Zhifu Gao, Qian Chen, Yue Gu, Mengzhe Chen, Yafeng Chen, Shiliang Zhang, Wen Wang, Jieping Ye
> - **机构**: 阿里巴巴通义实验室
> - **日期**: 2025-05-27
> - **arXiv**: [2505.17589](https://arxiv.org/abs/2505.17589)
> - **Demo**: https://funaudiollm.github.io/cosyvoice3

## 摘要

CosyVoice 3 是面向 **in-the-wild 场景**的零样本多语言语音合成模型，在内容一致性、说话人相似度和韵律自然度上超越前代 CosyVoice 2。

> [!tip] 核心特性
> 1. **新型语音 tokenizer**: 基于监督多任务训练（ASR、LID、SER、AED、SA）
> 2. **可微分奖励模型**: DiffRO 后训练方法
> 3. **数据规模扩展**: 1 万小时 → **100 万小时**
> 4. **模型规模扩展**: 0.5B → **1.5B** 参数

## 与 CosyVoice 2 的演进关系

> [!note] 系列演进
> - **CosyVoice**: 基于监督离散语音 token 的多语言语音合成
> - **CosyVoice 2**: 引入 FSQ、统一流式/非流式、chunk-aware CFM
> - **CosyVoice 3**: 大规模 in-the-wild 应用，覆盖更多语言和场景

---

## 1. 监督多任务训练的语音 Tokenizer

> [!important] 架构设计
> - **基础模型**: MinMo（大规模语音理解模型，140 万小时语音训练）
> - **FSQ 插入位置**: Voice Encoder 中
> - **Voice Encoder1**: 12 层 Transformer + RoPE

### 多任务训练

> [!abstract] 训练任务（约 53 万小时数据）
> - **ASR**: 自动语音识别
> - **LID**: 语言识别
> - **SER**: 语音情感识别
> - **AED**: 音频事件检测
> - **SA**: 说话人分析

> [!success] 优势
> - Token 率：**25 Hz**（每秒 25 个语音 token）
> - 通过监督多任务学习，离散语音 token 能更好地捕获**副语言信息**（情感、发音风格等）

### FSQ 量化过程

$$\bar{H} = \text{ROUND}(\text{Proj}_{down}(H))$$
$$\hat{H} = \text{Proj}_{up}(\bar{H})$$

> [!note] Token 索引计算
> $$\mu_i = \sum_{j=0}^{D-1} \bar{h}_{i,j} (2K+1)^j$$
> 其中 D 是低秩空间维度，K 是量化边界

---

## 2. 可微分奖励优化 (DiffRO)

> [!warning] 传统 RL 方法的挑战
> 1. TTS 系统需要额外的下游 CFM 和 vocoder 模型
> 2. 下游处理后语音高度相似，难以区分正负反馈
> 3. 计算需求大

### DiffRO 方案

> [!tip] 核心思路
> **直接在 token 级优化**，而非依赖下游模型的输出

> [!abstract] 算法流程
> 1. 训练 ASR 类的 **Token2Text 模型**
> 2. 使用后验概率作为奖励 $R_{ASR}$
> 3. 用 **Gumbel-Softmax** 采样 LLM 预测的 token
> 4. 直接优化语音 token 以最大化奖励分数

**关键公式**:

$$R_{ASR}(Y) = \log P_{ASR}(\tilde{Y}_n = Y_n | Y_{1:n-1}; \tilde{\mu}_{1:T})$$

$$\pi^*_\theta = \max_{\pi_\theta} \mathbb{E}[R(Y)] - \beta D_{KL}[\pi_\theta(\mu|Y) \| \pi_{ref}(\mu|Y)]$$

> [!note] KL 散度计算
> 在**输出 token 级 logits** 上计算（而非序列级后验概率）
> $$D_{KL} = \sum_{t=1}^{T} \sum_{k=0}^{Q} P_{\pi_\theta}(\mu_t = k) \log \frac{P_{\pi_\theta}(\mu_t = k)}{P_{\pi_{ref}}(\mu_t = k)}$$

### 多任务奖励 (MTR)

> [!tip] 扩展奖励来源
> 除 Token2Text 外，还使用：
> - SER（语音情感识别）
> - MOS 分数预测
> - AED（音频事件检测）
> - 其他音频理解任务

$$R_{MTR}(Y, \{A_i\}_{i=1}^K) = \sum_{i=1}^K \log P_{task_i}(\tilde{A}_i = A_i | \tilde{\mu})$$

---

## 3. 发音修补 (Pronunciation Inpainting)

> [!important] 问题
> LLM-based TTS 主要使用 BPE 文本 tokenizer，缺乏发音可控性。多音字或罕见字的发音难以控制。

> [!success] 解决方案
> - 扩展 CosyVoice 3 以建模**词和音素的混合序列**
> - 扩展词汇表
> - 构建辅助训练集：
>   - 中文单音字替换为拼音
>   - 英文单音词替换为音素（CMU 发音词典）

---

## 4. 文本归一化自训练

> [!note] 三种数据构建方式
> 1. 规则 TN 模块 → CosyVoice 2 合成音频
> 2. Qwen 提示生成
> 3. 混合方式

> [!tip] 优势
> 减少对人工编写的大量规则的依赖，构建更统一的端到端 TTS 系统

---

## 5. 数据与模型规模扩展

### 数据规模扩展

> [!success] 从 1 万小时到 100 万小时
> | 版本 | 训练数据 | 语言 | 方言 |
> |------|---------|------|------|
> | CosyVoice 2 | 1 万小时 | 中英 | - |
> | CosyVoice 3 | **100 万小时** | **9 种语言** | **18 种中国方言/口音** |

### 模型规模扩展

> [!note] 参数量增长
> | 版本 | 参数量 | 多语言 benchmark 表现 |
> |------|--------|---------------------|
> | CosyVoice 2 | 0.5B | 基准 |
> | CosyVoice 3-0.5B | 0.5B | 提升 |
> | CosyVoice 3-1.5B | **1.5B** | **显著提升** |

---

## 训练流程

> [!abstract] 四阶段训练
> 1. **大规模预训练**: 通用语音合成能力
> 2. **后训练 (DiffRO)**: 超越训练数据的性能限制
> 3. **持续预训练**: 将指令可控性和多语言合成功能从 zero-shot 模型迁移到 SFT 模型
> 4. **多说话人微调**: 特定说话人优化

---

## CV3-Eval Benchmark

> [!note] 评估基准
> - 基于真实的 **in-the-wild 参考语音**
> - 数据来源：Common Voice, FLUERS, EmoBox, Web-crawled 数据
> - 覆盖范围：多种语言和方言、领域和环境、情感和风格

---

## 实验结果

### 内容一致性 (CER/WER)

> [!success] 全面超越竞争模型
> - 在 SEED (zh/en/hard) 和 CV3-Eval (ja/ko/de/es/fr/it/ru) 上全面领先
> - 相比 CosyVoice 2 显著降低错误率

### 说话人相似度

> [!note] WavLM 嵌入余弦相似度
> - 在多数 benchmark 上达到 **0.80+** 的相似度
> - 与 MaskGCT, F5TTS, Seed-TTS, FireRedTTS 等竞争

### 与竞争模型对比

> [!important] CosyVoice 3-1.5B 在多数指标上领先
> - **内容一致性**: 最低 CER/WER
> - **说话人相似度**: 最高余弦相似度
> - **多语言支持**: 9 种语言 + 18 种方言

---

## 个人思考

> [!tip] 研究启示
> 1. **监督多任务语音 tokenizer** 是重要创新，通过 ASR/LID/SER/AED/SA 联合训练，使离散 token 捕获更多副语言信息
> 2. **DiffRO** 解决了语音生成 RL 训练的核心难题，直接在 token 级优化而非依赖下游模型
> 3. **数据规模扩展** 到 100 万小时是工程上的重大突破
> 4. **发音修补** 功能对工业级 TTS 系统非常实用
> 5. 从 CosyVoice 2 到 3 的演进路线清晰：离散 token 利用优化 → 流式合成 → 大规模 in-the-wild 应用
