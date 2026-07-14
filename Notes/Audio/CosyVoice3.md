---
type: Note
related_to: "[[Audio]]"
status: Active
url: https://arxiv.org/abs/2505.17589
date: 2025-05-27
---

# CosyVoice 3: Towards In-the-wild Speech Generation via Scaling-up and Post-training

## 基本信息

- **作者**: Zhihao Du, Changfeng Gao, Yuxuan Wang, Fan Yu, Tianyu Zhao, Hao Wang, Xiang Lv, Hui Wang, Chongjia Ni, Xian Shi, Keyu An, Guanrou Yang, Yabin Li, Yanni Chen, Zhifu Gao, Qian Chen, Yue Gu, Mengzhe Chen, Yafeng Chen, Shiliang Zhang, Wen Wang, Jieping Ye
- **机构**: 阿里巴巴通义实验室
- **时间**: 2025-05-27
- **arXiv**: [2505.17589](https://arxiv.org/abs/2505.17589)
- **Demo**: https://funaudiollm.github.io/cosyvoice3

## 摘要

CosyVoice 3 是面向 in-the-wild 场景的零样本多语言语音合成模型，在内容一致性、说话人相似度和韵律自然度上超越前代 CosyVoice 2。核心特性包括：新型语音 tokenizer、可微分奖励模型后训练、数据规模扩展（1 万小时→100 万小时）和模型规模扩展（0.5B→1.5B）。

## 核心创新

### 1. 监督多任务训练的语音 Tokenizer

**基础架构**: 基于 MinMo（大规模语音理解模型），在 Voice Encoder 中插入 FSQ 模块

**多任务训练**（约 53 万小时数据）:
- **ASR**: 自动语音识别
- **LID**: 语言识别
- **SER**: 语音情感识别
- **AED**: 音频事件检测
- **SA**: 说话人分析

**Tokenizer 特性**:
- Token 率：25 Hz（每秒 25 个语音 token）
- 通过监督多任务学习，离散语音 token 能更好地捕获副语言信息（情感、发音风格等）

**与 CosyVoice 2 的区别**:
- CosyVoice 2: 基于 SenseVoice-Large ASR 模型
- CosyVoice 3: 基于 MinMo 多模态 LLM（140 万小时语音训练）

### 2. 可微分奖励优化 (DiffRO)

**问题**: 传统 RL 方法在语音生成中的挑战
- TTS 系统需要额外的下游 CFM 和 vocoder 模型
- 下游处理后语音高度相似，难以区分正负反馈

**DiffRO 方案**:
1. 训练 ASR 类的 Token2Text 模型
2. 使用后验概率作为奖励
3. 用 Gumbel-Softmax 采样 LLM 预测的 token
4. 直接优化语音 token 以最大化奖励分数（而非 RL 训练循环）

**关键公式**:
$$R_{ASR}(Y) = \log P_{ASR}(\tilde{Y}_n = Y_n | Y_{1:n-1}; \tilde{\mu}_{1:T})$$

$$\pi^*_\theta = \max_{\pi_\theta} \mathbb{E}[R(Y)] - \beta D_{KL}[\pi_\theta(\mu|Y) \| \pi_{ref}(\mu|Y)]$$

**多任务奖励 (MTR)**: 使用 SER、MOS 分数预测、AED 等下游任务进行多任务奖励建模

### 3. 发音修补 (Pronunciation Inpainting)

- 扩展 CosyVoice 3 以建模词和音素的混合序列
- 构建辅助训练集：中文单音字替换为拼音，英文单音词替换为音素（CMU 发音词典）
- 实现对发音的有效人工干预控制

### 4. 文本归一化自训练

- 利用 LLM 进行 TN 任务
- 三种数据构建方式：
  1. 规则 TN 模块 + CosyVoice 2 合成音频
  2. Qwen 提示生成
  3. 混合方式

### 5. 数据与模型规模扩展

**数据规模扩展**:
- CosyVoice 2: 1 万小时
- CosyVoice 3: 100 万小时
- 覆盖 9 种语言、18 种中国方言/口音
- 多种领域和文本格式

**模型规模扩展**:
- CosyVoice 2: 0.5B 参数
- CosyVoice 3: 1.5B 参数
- 在多语言 benchmark 上性能提升

## 训练流程

1. **大规模预训练**: 通用语音合成能力
2. **后训练 (DiffRO)**: 超越训练数据的性能限制
3. **持续预训练**: 将指令可控性和多语言合成功能从 zero-shot 模型迁移到 SFT 模型
4. **多说话人微调**: 特定说话人优化

## CV3-Eval Benchmark

- 基于真实的 in-the-wild 参考语音
- 数据来源：Common Voice, FLUERS, EmoBox, Web-crawled 数据
- 覆盖范围：多种语言和方言、领域和环境、情感和风格

## 实验结果

### 内容一致性 (CER/WER)
- 在 SEED (zh/en/hard) 和 CV3-Eval (ja/ko/de/es/fr/it/ru) 上全面超越
- 相比 CosyVoice 2 显著降低错误率

### 说话人相似度
- WavLM 嵌入的余弦相似度
- 在多数 benchmark 上达到 0.80+ 的相似度

### 与竞争模型对比
- MaskGCT, F5TTS, Seed-TTS, FireRedTTS, Qwen2.5-Omni-7B
- CosyVoice 3-1.5B 在多数指标上领先

## 个人思考

CosyVoice 3 的核心贡献：

1. **监督多任务语音 tokenizer** 是重要创新，通过 ASR/LID/SER/AED/SA 联合训练，使离散 token 捕获更多副语言信息
2. **DiffRO** 解决了语音生成 RL 训练的核心难题，直接在 token 级优化而非依赖下游模型
3. **数据规模扩展** 到 100 万小时是工程上的重大突破
4. **发音修补** 功能对工业级 TTS 系统非常实用

与 CosyVoice 2 的演进关系清晰：从离散 token 利用优化 → 流式合成 → 大规模 in-the-wild 应用。
