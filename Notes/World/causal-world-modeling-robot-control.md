---
type: Note
related_to: "[[world-model]]"
status: Active
tags:
  - world-model
  - causal
  - asynchronous
  - robot-control
  - diffusion
---

# Causal World Modeling for Robot Control

## 元信息

| 字段 | 内容 |
|------|------|
| **作者** | Lin Li, Qihang Zhang, Yiming Luo, Shuai Yang, Ruilin Wang, Fei Han, Mingrui Yu, Zelin Gao, Nan Xue, Xing Zhu, Yujun Shen, Yinghao Xu |
| **日期** | 2026-01 |
| **arXiv** | [2601.21998](https://arxiv.org/abs/2601.21998) |
| **URL** | https://arxiv.org/abs/2601.21998 |
| **Zotero Key** | Q7JFQVEB |
| **项目主页** | https://technology.robbyant.com/lingbot-va |
| **代码** | https://github.com/robbyant/lingbot-va |

## 摘要 (Abstract)

Video world modeling 结合 vision-language pre-training，构成了机器人学习的一种全新且独立的基础范式。Video world models 通过理解动作与视觉变化之间的因果关系，能够想象近期未来。本文提出 **LingBot-VA**，一个自回归扩散框架，联合学习帧预测和策略执行。

## 核心贡献 (Key Contributions)

1. **提出 LingBot-VA 框架** — 自回归扩散模型，联合学习视频帧预测与机器人策略执行
2. **共享潜在空间 (Shared Latent Space)** — 使用 Mixture-of-Transformers (MoT) 架构整合视觉和动作 tokens
3. **闭环展开机制 (Closed-loop Rollout)** — 持续获取环境反馈与 ground-truth 观测
4. **异步推理流水线 (Asynchronous Inference Pipeline)** — 并行化动作预测和电机执行，提升控制效率

## 方法 (Method)

### 整体架构

- **Mixture-of-Transformers (MoT)**: 视觉 tokens 和动作 tokens 共享潜在空间，通过 MoT 架构进行联合建模
- **自回归扩散 (Autoregressive Diffusion)**: 逐帧预测未来视觉状态，同时输出对应的控制动作
- **闭环展开**: 在推理过程中，模型可以接收真实环境的反馈观测，实现闭环控制
- **异步推理**: 动作预测与电机执行并行进行，减少延迟

### 设计理念

论文认为 video world modeling 是一种独立于 vision-language pre-training 的机器人学习基础。关键洞察是：理解动作与视觉变化之间的因果关系是机器人控制的核心。

## 实验结果 (Results)

- 在**仿真基准测试和真实世界场景**中均进行了评估
- 在**长时域操作任务 (long-horizon manipulation)** 中表现出色
- 展示了**后训练阶段的数据效率 (data efficiency in post-training)**
- 对**新配置具有强泛化能力 (strong generalizability to novel configurations)**

## 与其他工作的关系 (Related Work)

- **Video World Models**: 与 video generation/prediction 相关工作（如 Sora 类模型）的区别在于明确建模因果关系
- **Vision-Language Pre-training**: 本文提出 video world modeling 是与 VLP 并行的独立范式
- **Diffusion for Robotics**: 将扩散模型应用于机器人控制，但加入了因果结构
- **Closed-loop Control**: 与开环规划方法的区别在于持续接收环境反馈

## 个人笔记 (Notes)

- **异步推理**是一个很实用的工程创新，解决了 world model 推理延迟与实时控制之间的矛盾
- MoT 架构将视觉和动作统一在共享空间中，这种设计思路与 JEPA 系列工作有相似之处（都是在表示空间中进行预测）
- 闭环展开机制使得模型可以从自己的预测误差中修正，类似于 MPC (Model Predictive Control) 的思路
- 作为 2026 年的工作，代表了 world model 在机器人控制领域的最新进展
