---
type: Note
related_to: "[[world-model]]"
status: Active
tags:
  - world-model
  - scalable
  - compositional
  - robot-data-synthesis
  - teleoperation
---

# RoboDream: Compositional World Models for Scalable Robot Data Synthesis

## 元信息

| 字段 | 内容 |
|------|------|
| **作者** | Junjie Ye, Rong Xue, Basile Van Hoorick, Runhao Li, Harshitha Rajaprakash, Pavel Tokmakov, Muhammad Zubair Irshad, Vitor Guizilini, Yue Wang |
| **日期** | 2026-06 |
| **arXiv** | [2606.02577](https://arxiv.org/abs/2606.02577) |
| **URL** | https://arxiv.org/abs/2606.02577 |
| **Zotero Key** | 2RGCQXQ6 |
| **项目主页** | https://junjieye.com/RoboDream/ |

## 摘要 (Abstract)

现实世界通过遥操作收集数据的成本极高且耗时。本文提出一个**可泛化的、以 embodiment 为中心的 world model**，能够生成**照片级真实的演示数据**，包括新物体、新场景和新视角。该方法将生成过程锚定到渲染的机器人运动上，同时以显式的场景和物体先验为条件，有效地将轨迹执行与环境合成解耦。

## 核心贡献 (Key Contributions)

1. **组合式 World Model** — 将轨迹执行与环境合成解耦，实现可组合的数据生成
2. **检索与重生 (Retrieval and Rebirth)** — 将现有轨迹复用到全新上下文中，无需新的运动数据
3. **无道具遥操作 (Prop-free Teleoperation)** — 操作者操纵空气，模型随后幻觉出目标物体和场景，消除重置时间

## 方法 (Method)

### 解耦设计

该方法将数据生成的两个方面解耦：

1. **轨迹执行 (Trajectory Execution)** — 锚定到渲染的机器人运动，确保运动的物理合理性
2. **环境合成 (Environment Synthesis)** — 以场景和物体先验为条件，生成多样化的视觉环境

### 检索与重生 (Retrieval and Rebirth)

- 从已有数据中检索轨迹
- 将这些轨迹"重生"到全新的上下文中（新物体、新场景、新视角）
- 无需收集新的运动数据，大幅降低数据获取成本

### 无道具遥操作 (Prop-free Teleoperation)

- 操作者只需在空中做出操纵动作
- World model 随后在视频中幻觉出目标物体和场景
- 消除了传统遥操作中的重置时间，大幅提升数据收集效率

## 实验结果 (Results)

- 在真实世界实验中验证
- 生成的数据**一致性地提升下游策略性能**
- **显著减少真实世界数据需求**，覆盖多种操作任务
- 支持新物体、新场景和新视角的泛化

## 与其他工作的关系 (Related Work)

- **Video Generation for Robotics**: 与 Genie、UniSim 等工作相关，但强调组合性和可扩展性
- **Teleoperation Data Collection**: 解决传统遥操作数据收集成本高的问题
- **Sim-to-Real**: 与仿真到真实迁移相关，但本文通过生成模型实现
- **Data Augmentation for Robotics**: 本质上是一种高级的数据增强方法

## 个人笔记 (Notes)

- **解耦设计**是一个关键洞察：将运动轨迹与视觉外观分离，使得数据生成具有组合性
- **无道具遥操作**是一个非常实用的创新，解决了机器人数据收集中的一个真实痛点
- 这个工作本质上是在用 world model 做数据增广，但增广的质量和多样性远超传统方法
- 对于解决机器人学习中的数据瓶颈问题非常有价值
- 与 Causal World Modeling (LingBot-VA) 形成对比：一个关注控制，一个关注数据
