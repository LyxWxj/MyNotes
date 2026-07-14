---
type: Note
related_to: "[[Audio]]"
status: Active
url: https://arxiv.org/abs/2606.05121
date: 2026-06-03
---

# Audio Interaction Model

## 基本信息

- **作者**: Zhifei Xie, Zihang Liu, Ze An, Xiaobin Hu, Yue Liao, Ziyang Ma, Dongchao Yang, Mingbao Lin, Deheng Ye, Shuicheng Yan, Chunyan Miao
- **机构**: NTU, NUS, CUHK
- **时间**: 2026-06-03
- **arXiv**: [2606.05121](https://arxiv.org/abs/2606.05121)
- **项目主页**: https://xzf-thu.github.io/Audio-Interaction
- **代码**: https://github.com/xzf-thu/Audio-Interaction
- **数据集**: https://huggingface.co/datasets/zhifeixie/StreamAudio-2M

## 摘要

当前的大型音频语言模型 (LALMs) 是离线的，而流式音频模型各自只处理单一任务（如流式 ASR 或语音聊天）。本文提出了 **Audio Interaction Model**，通过一个始终在线的感知-决策-响应循环（perceive-decide-respond loop），实时聆听声音、环境和指令，并即时反应。提出了 **SoundFlow** 框架，构建了 **StreamAudio-2M**（260 万条、30.2 万小时的流式语料库）和 **Proactive-Sound-Bench** 评估基准。

## 核心动机

1. **音频是天然的交互模态**: 不同于文本（压缩为符号）和图像（静态快照），音频是连续的、实时的通道
2. **现有 LALMs 的局限**: 遵循离线输入-输出范式 y = f(x, A)，无法匹配音频的实时和交互特性
3. **专用流式模型的不足**: 每种能力需要从头训练自己的模型，且每个模型只处理狭窄的能力

## 核心挑战

- **C1: 基于理解的响应触发**: 离线 LALMs 被动响应完整音频，而交互模型必须在每个 chunk 基于语义理解决定是否响应
- **C2: chunk 推理下的实时上下文连续性**: 音频必须按固定长度 chunk 消费以满足低延迟要求，但 chunking 破坏了时间连续性

## 技术方案

### 1. Audio-Interaction 模型

始终在线的音频交互模型，逐 chunk 消费音频，在每步做出基于理解的决策：

$$(d_t, r_t) = f(a_{\leq t}, d_{<t}, r_{<t})$$

- $a_t$: 当前音频 chunk（400ms）
- $d_t \in \{<silent>, <response>\}$: 流式干预决策
- $r_t$: 生成的响应

### 2. SoundFlow 框架

端到端音频交互框架，包含三个组件：

#### 2.1 流式数据构建

- **时频联合预处理 (TFJP)**: 使每个音频段更平滑、自然，包括静音裁剪、噪声去除、核心定位、边界规范化和频谱平滑
- **层次化音频事件选择**: 
  - 场景规划：用 LLM 规划高层场景
  - 事件细化：将主题细化为具体音频事件序列
  - clip 置信：通过检索或生成获取最终音频

#### 2.2 流式训练

- **流式建模**: 每个 chunk 400ms，预测特殊 token 决定继续监听或开始响应
- **上下文记忆与感知感知静音训练**:
  - 历史回顾训练：在序列后期插入关于前面内容的问题
  - 大量静音音频训练以减少误触发
- **双损失多步流式转换**: 标准语言建模目标 + 流式控制 token 专用目标

**四阶段训练流程**:
1. 格式训练：用离线数据教模型目标序列格式
2. 适配器训练：训练适配器映射 chunk 级声学表示
3. 大规模流式监督训练：联合优化核心能力
4. 指令跟随微调：训练复杂流式行为

#### 2.3 异步推理

FIFO 调度的异步推理方案：
- 编码器持续处理流式音频 chunk 并追加到时间有序队列
- 解码器在 $r_{t-1} \in \{<eos>, <silent>\}$ 时触发
- 消除推理停滞，恢复监听的首帧延迟降低 4.5×

### 3. StreamAudio-2M 数据集

- **规模**: 260 万条，30.2 万小时，3-15 轮交互
- **覆盖 7 大类 28 子任务**:
  - Voice Chatting (539k, 23.1%)
  - Streaming Instruction Following (487k, 20.8%)
  - Streaming Audio Understanding (382k, 16.4%)
  - Streaming Translation (357k, 15.3%)
  - Real-time ASR (270k, 11.6%)
  - Proactive Response (171k, 7.3%)
  - Audio Agent (130k, 5.5%)

### 4. Proactive-Sound-Bench

- 644 个人工设计的声学事件
- 6 个顶层类别，17 个子类别
- Single 和 Multiple 两个层级

## 实验结果

### 主要发现

1. **[Enh.1] 保留音频理解**: MMAU 上达到 58.15（音频指令），与 7B 系统相当
2. **[Enh.2] 核心语音任务竞争力**: CoVoST2 上 en-zh/zh-en 提升 +15.72/+17.04 BLEU
3. **[Enh.3] 解锁离线 LALMs 无法实现的能力**:
   - 对口语指令的鲁棒性
   - 选择性主动响应：Proactive-Sound-Bench 上 Single 61.2，Multi 62.8
   - 流式拼接下的能力稳定性

### 关键观察

- SALMs 在早期解码器层将离散 chunk 统一为连续表示（GPT Layer 0 将连续性比率从 0.25 提升到 0.80）

## 与现有工作的区别

| 特性 | 离线 LALMs | 流式对话模型 | Audio-Interaction |
|------|-----------|-------------|-------------------|
| 输入模式 | 完整音频 | chunk 级 | chunk 级 |
| 任务范围 | 广泛 | 单一 | 统一 |
| 响应时机 | 被动 | 轮次结束 | 主动决策 |
| 非语音理解 | ✓ | ✗ | ✓ |

## 个人思考

这篇工作非常有前瞻性，将音频的实时交互特性形式化为 Audio Interaction Model。核心创新在于：
1. 统一了离线和流式能力于单一模型
2. 通过 perceive-decide-respond 循环实现主动干预
3. 构建了大规模流式训练数据集

对我的研究启示：流式音频模型的训练数据构建方法（层次化事件选择、TFJP 预处理）值得借鉴。
