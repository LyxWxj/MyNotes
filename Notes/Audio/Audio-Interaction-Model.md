---
type: Note
related_to: "[[Audio]]"
status: Active
url: https://arxiv.org/abs/2606.05121
date: 2026-06-03
---

# Audio Interaction Model

> [!info] 论文信息
> - **作者**: Zhifei Xie, Zihang Liu, Ze An, Xiaobin Hu, Yue Liao, Ziyang Ma, Dongchao Yang, Mingbao Lin, Deheng Ye, Shuicheng Yan, Chunyan Miao
> - **机构**: NTU, NUS, CUHK
> - **日期**: 2026-06-03
> - **arXiv**: [2606.05121](https://arxiv.org/abs/2606.05121)
> - **项目主页**: https://xzf-thu.github.io/Audio-Interaction
> - **代码**: https://github.com/xzf-thu/Audio-Interaction
> - **数据集**: https://huggingface.co/datasets/zhifeixie/StreamAudio-2M

## 摘要

当前的大型音频语言模型 (LALMs) 是离线的，而流式音频模型各自只处理单一任务（如流式 ASR 或语音聊天）。本文提出了 **Audio Interaction Model**，通过一个始终在线的 **感知-决策-响应循环**（perceive-decide-respond loop），实时聆听声音、环境和指令，并即时反应。提出了 **SoundFlow** 框架，构建了 **StreamAudio-2M**（260 万条、30.2 万小时的流式语料库）和 **Proactive-Sound-Bench** 评估基准。

> [!tip] 核心思想
> 将音频的实时交互特性形式化为 Audio Interaction Model，统一离线任务执行和在线音频指令跟随于单一模型中。

## 动机

> [!warning] 现有方法的局限
> 1. **离线 LALMs 的问题**：遵循 y = f(x, A) 范式，需要完整音频才能响应，无法匹配音频的实时交互特性
> 2. **专用流式模型的不足**：每种能力需要从头训练自己的模型，且每个模型只处理狭窄的能力
> 3. **Moshi 等全流式系统的局限**：尽管对话能力强，但无法解释犹豫的停顿或识别咳嗽声

> [!important] 从 LALM 到 LAIM
> - **LALM (Large Audio Language Model)**: 离线音频语言模型
> - **LAIM (Large Audio Interaction Model)**: 大型音频交互模型，全合一框架，在单一交互模型中涵盖现有任务

## 核心挑战

> [!danger] 两大挑战
> - **C1: 基于理解的响应触发**：离线 LALMs 被动响应完整音频，而交互模型必须在每个 chunk 基于语义理解决定是否响应。监督信号稀疏且时间模糊，没有现成语料库将连续音频与正确时间的干预线索配对。
> - **C2: chunk 推理下的实时上下文连续性**：音频必须按固定长度 chunk 消费以满足低延迟要求，但 chunking 破坏了声学信号的时间连续性和跨 chunk 累积的长程上下文。

## 技术方案

### 1. Audio-Interaction 模型

> [!abstract] 模型定义
> 始终在线的音频交互模型，逐 chunk 消费音频，在每步做出基于理解的决策：
> $$(d_t, r_t) = f(a_{\leq t}, d_{<t}, r_{<t})$$
> - $a_t$: 当前音频 chunk（**400ms**）
> - $d_t \in \{<silent>, <response>\}$: 流式干预决策
> - $r_t$: 生成的响应

> [!note] 统一的能力范式
> - 传统音频能力（翻译、识别、对话）自然统一为指令
> - 从语音翻译到同声传译
> - 从语音对话到开放域音频讨论
> - 从音频理解到音频指令跟随
> - 甚至无需显式指令的主动辅助

### 2. SoundFlow 框架

端到端音频交互框架，包含三个组件：

#### 2.1 流式数据构建

> [!tip] 时频联合预处理 (TFJP)
> 使每个音频段更平滑、自然，包含以下步骤：
> 1. **静音裁剪 (silence_cut)**: 裁剪过多的内部静音
> 2. **噪声估计 (noise_profile)**: 从低能量区域估计背景噪声
> 3. **去噪 (denoise)**: 在频域去除噪声
> 4. **核心定位 (core_locate)**: 定位最密集的信息跨度
> 5. **边界规范化 (boundary_norm)**: 用半 chunk 对齐 δ=1/2 修剪边界
> 6. **频谱平滑 (spec_smooth)**: 短窗口频谱平滑

> [!important] 层次化音频事件选择
> 解决如何将离散 (audio, instruction, response) 片段组织成长的多轮音频流：
> - **场景规划**: 用 LLM 从随机匹配的音频注释规划完整高层场景
> - **事件细化**: 将主题细化为具体音频事件序列
> - **clip 置信**: 通过检索或生成获取最终音频
>   - 检索：搜索音频 clip 数据库，选择 top-3 最相关候选
>   - 生成：当检索不到合适 clip 时，调用音频生成模型合成

#### 2.2 流式训练

> [!abstract] 流式建模
> - 每个 chunk 400ms，预测特殊 token 决定继续监听或开始响应
> - $d_t, r_t = f_{det}(a_t, C_t)$
> - $r_t = \begin{cases} \emptyset, & d_t = <silent> \\ f_{resp}(a_t, C_t), & d_t = <response> \end{cases}$

> [!warning] 训练中的两种失败模式
> 1. **上下文保留不足**: 模型倾向于忽略早期上下文（长训练序列中噪声/语义空片段的普遍存在）
> 2. **误触发**: 模型倾向于对交互无关的声学事件响应

> [!success] 解决方案
> - **历史回顾训练**: 在序列后期插入关于前面内容的问题，显式鼓励长程上下文检索
> - **静音训练**: 加入大量经验证无需响应的静音音频，增强模型保持静默的能力

> [!note] 双损失多步流式转换
> $$\mathcal{L} = \frac{1}{N} \sum_{j=1}^{N} \left( -\log P_\theta(t_j | H_j) + \lambda \cdot (-\log P_\theta(s_j | H_j)) \right)$$
> - $t_j$: 目标文本 token
> - $s_j$: 目标流式控制 token
> - $\lambda$: 流式目标的相对权重

> [!abstract] 四阶段训练流程
> 1. **格式训练**: 用离线数据教模型目标序列格式和 `<Spe_token>` 的使用
> 2. **适配器训练**: 训练适配器映射 chunk 级声学表示到语言模型空间
> 3. **大规模流式监督训练**: 联合优化适配器和语言模型（音频理解、ASR、口语对话）
> 4. **指令跟随微调**: 训练复杂流式行为（连续辅助、感知感知干预、主动响应）

#### 2.3 异步推理

> [!tip] FIFO 调度
> - 编码器持续处理流式音频 chunk 并追加到时间有序队列
> - 解码器在 $r_{t-1} \in \{<eos>, <silent>\}$ 时触发
> - 消除推理停滞，恢复监听的首帧延迟降低 **4.5×**

### 3. StreamAudio-2M 数据集

> [!note] 数据集规模
> - **总量**: 260 万条，30.2 万小时，3-15 轮交互
> - **覆盖 7 大类 28 子任务**

| 类别 | 数量 | 占比 | 数据来源 |
|------|------|------|---------|
| Voice Chatting | 539k | 23.1% | MOSS, GammaCorpus |
| Streaming Instruction Following | 487k | 20.8% | UltraChat, Magpie-Pro |
| Streaming Audio Understanding | 382k | 16.4% | AudioSet, FMA |
| Streaming Translation | 357k | 15.3% | CoVoST 2, AISHELL |
| Real-time ASR | 270k | 11.6% | CommonVoice, GigaSpeech |
| Proactive Response | 171k | 7.3% | AudioSet, AudioX |
| Audio Agent | 130k | 5.5% | MOSS, AudioSet |

### 4. Proactive-Sound-Bench

> [!important] 评估主动流式响应
> - 644 个人工设计的声学事件
> - 6 个顶层类别，17 个子类别
> - **Single 层级**: 单事件决策
> - **Multiple 层级**: 拼接同类事件，探测在干扰下的持续干预能力

## 实验结果

> [!success] 三个增强维度
> 1. **[Enh.1] 保留音频理解**: MMAU 上达到 58.15（音频指令），与 7B 系统相当
> 2. **[Enh.2] 核心语音任务竞争力**: CoVoST2 上 en-zh/zh-en 提升 +15.72/+17.04 BLEU
> 3. **[Enh.3] 解锁离线 LALMs 无法实现的能力**

### Proactive-Sound-Bench 结果

> [!note] 主动响应能力
> | 模型 | Single Avg | Multi Avg |
> |------|-----------|----------|
> | Qwen2.5-Omni-7B | 58.2 | 32.1 |
> | MiniCPM-o-4.5 | 58.9 | 58.9 |
> | **Audio-Interaction** | **61.2** | **62.8** |

> [!tip] 关键观察
> - SALMs 在**早期解码器层**将离散 chunk 统一为连续表示
> - GPT Layer 0 将连续性比率从 0.25 提升到 0.80（1.0 表示无缝连续）

## 与现有工作的对比

| 特性 | 离线 LALMs | 流式对话模型 | Audio-Interaction |
|------|-----------|-------------|-------------------|
| 输入模式 | 完整音频 | chunk 级 | chunk 级 |
| 任务范围 | 广泛 | 单一 | 统一 |
| 响应时机 | 被动 | 轮次结束 | 主动决策 |
| 非语音理解 | ✓ | ✗ | ✓ |

## 个人思考

> [!tip] 研究启示
> 1. **统一范式的价值**: 将感知-决策-响应循环形式化，使模型可以主动决定何时响应
> 2. **数据构建方法论**: 层次化事件选择 + TFJP 预处理的流式数据构建方法值得借鉴
> 3. **双损失训练**: 标准 LM 损失 + 流式控制 token 专用损失的联合优化策略
> 4. **异步推理设计**: FIFO 调度消除推理停滞，工程价值高
