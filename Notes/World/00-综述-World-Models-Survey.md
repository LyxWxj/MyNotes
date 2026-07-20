# World Models: A Comprehensive Survey (2026)

> **论文**: World Models: A Comprehensive Survey of Architectures, Methodologies, Reasoning Paradigms, and Applications
> **作者**: Arif Hassan Zidan, Yi Pan, Hanqi Jiang, Wei Zhang 等 26 人
> **来源**: arXiv 2606.00133 | 147 页 | CC BY 4.0
> **分类**: cs.LG, cs.ET

---

## 一、核心定义

### 什么是世界模型？

世界模型是**学习环境结构和动力学的内部模拟器**（internal simulators that learn the structure and dynamics of an environment）。

**形式化定义**：参数化预测系统，近似环境动力学：

$$p_\theta(s_{t+1}, o_{t+1}, r_t | s_t, a_t)$$

在部分可观测设置中，模型必须从历史 $h_t = (o_{\leq t}, a_{<t})$ 推断潜在信念状态。

### 三个区分性特征

| 特征 | 说明 |
|------|------|
| **动作条件化** (Action-conditioning) | 预测特定动作下的演化，支持反事实推理 |
| **多步展开** (Multi-step rollout) | 可自回归地生成任意长度轨迹 |
| **决策效用** (Utility for decision-making) | 预测用于策略优化、规划、数据增强或安全验证 |

### 与 Model-Free RL 的关键区别

| 维度 | 世界模型 | Model-Free RL |
|------|---------|---------------|
| 学习内容 | 预测结构 | 策略/价值函数 |
| 规划 vs 直接执行 | 在线规划或潜在想象 | 直接前向传播 |
| 样本效率 | 通过预测/想象复用经验 | 需要大量交互 |
| 模型偏差 | 长程展开可能漂移 | 避免此失败模式 |
| 反事实推理 | 原生支持 | 不提供显式模拟器 |
| 透明度 | 可检查重建/想象轨迹 | 知识隐式编码 |

---

## 二、四维分类框架

### 维度 1：架构 (Architecture)

#### 1.1 按表示分类

| 类型 | 代表 | 优势 | 劣势 |
|------|------|------|------|
| **观测空间（像素级）** | DIAMOND, GameNGen | 无信息丢失 | 高维、计算昂贵 |
| **连续潜在** | RSSM (Dreamer 系列) | 紧凑、平滑插值 | 可能丢失细节 |
| **离散 Token** | IRIS, GAIA-1 | 支持 AR Transformer | 码本容量有限 |
| **联合嵌入（表示空间）** | JEPA, MuZero | 避免像素重建瓶颈 | 需要好的表示学习 |
| **结构化/对象中心** | C-SWM, Slot Attention | 组合泛化 | O(N²) 扩展性差 |
| **3D/占据** | OccWorld, Copilot4D | 几何感知 | 内存/计算立方增长 |

#### 1.2 按动力学分类

- **确定性动力学**：固定转移函数，简单但模糊不确定未来
- **随机动力学**：概率转移，捕捉歧义和多模态
- **隐式生成动力学**：GAN/Flow，无需显式概率评估
- **表示空间预测动力学**：JEPA 风格，在嵌入空间预测
- **记忆增强动力学**：显式记忆机制处理长程依赖

#### 1.3 按模态分类

视觉 / 语言 / 3D 几何 / 本体感觉 / 多模态融合

#### 1.4 按学习范式分类

- 自监督/无监督（重建、对比、JEPA）
- 在线 MBRL（想象策略学习、搜索推理）
- 离线/批量学习
- 基础模型范式（预训练→微调）
- 监督/模仿学习
- 混合多阶段

#### 1.5 按下游用途分类

RL/规划、自动驾驶、机器人、医疗、视频生成、语言推理

---

### 维度 2：方法论家族 (Methodological Families)

#### 2.1 状态空间与循环潜在世界模型

核心：**RSSM** (Recurrent State-Space Model)

- 确定性循环状态 $h_t$ + 随机潜在 $z_t$
- Dreamer 系列的基石
- 离散类别潜在（DreamerV2）优于连续高斯

#### 2.2 Transformer 世界模型

- 自回归序列预测
- IRIS（next-token prediction）、GAIA-1（9B 参数驾驶模型）
- 利用 Transformer 的缩放特性和注意力机制

#### 2.3 扩散世界模型

- 迭代去噪生成未来状态
- DIAMOND、Copilot4D
- 高质量生成但计算成本更高

#### 2.4 物理信息与结构化世界模型

- 融合已知物理定律、守恒约束
- 可微物理引擎、哈密顿/拉格朗日神经网络

#### 2.5 语言增强与多模态世界模型

- 语言条件潜在动力学
- LLM 作为世界模型
- 文本条件视频生成作为世界模拟（Sora、Cosmos）
- 视觉-语言联合嵌入
- 统一多模态架构

---

### 维度 3：推理策略 (Reasoning Strategies)

#### 3.1 基于想象的规划

| 模式 | 说明 | 代表 |
|------|------|------|
| **学习时想象** (Background Planning) | 用想象轨迹改进策略 | Dreamer |
| **决策时想象** (Forward Search) | 推理时在线规划 | MuZero MCTS, PlaNet |
| **潜在想象的跨域优势** | 效率、抽象、可迁移 | TD-MPC |

**核心挑战**：误差复合（compounding errors）、目标不匹配（objective mismatch）

#### 3.2 策略学习

在想象展开中训练 Actor-Critic（Dreamer）、学习策略网络（MuZero）

#### 3.3 反事实推理

- **溯因-动作-预测** 管道
- 用结构因果模型隔离决策效应
- Schema Networks, Woulda-Coulda-Shoulda

#### 3.4 不确定性下规划

- 概率集成（PETS）
- 分布鲁棒规划
- Monte Carlo Dropout

---

### 维度 4：应用领域

| 领域 | 代表工作 |
|------|---------|
| 机器人 | Dreamer, TD-MPC, GWM, π₀ |
| 自动驾驶 | GAIA-1, OccWorld, Vista |
| 视频预测 | Sora, Cosmos, Genie |
| 多模态智能体 | Voyager, Smallville |
| RL/游戏 | MuZero, IRIS, GameNGen |
| 科学模拟 | GraphCast, NeuralGCM |
| 医疗 | 疾病进展建模 |
| 教育 | 学生状态追踪 |
| 金融 | 信念建模、反事实策略 |

---

## 三、关键组件

### POMDP 框架下的四模块

1. **编码器** (Encoder): $z_t = q_\phi(z_t | o_{\leq t}, a_{<t})$
   - CNN 确定性 / VAE 后验随机性
   - Ha & Schmidhuber: 64×64 图像 → 32 维潜在向量

2. **动力学模型** (Dynamics Model): $\tilde{z}_{t+1} = p_\theta(\tilde{z}_{t+1} | z_t, a_t)$
   - **世界模型的核心**
   - RNN-based: RSSM（确定性循环 + 随机组件）
   - Transformer-based: 自回归序列预测
   - Diffusion-based: 迭代去噪

3. **奖励预测器** (Reward Predictor): $\tilde{r}_t = p_\psi(r_t | z_t)$
   - MuZero 证明：完全在抽象空间中预测奖励/价值/策略即可超人表现

4. **解码器** (Decoder, 可选): 从潜在状态重建观测
   - MuZero 完全摒弃
   - JEPA 在表示空间预测，无需解码器

---

## 四、潜在空间的角色

### 为什么需要潜在空间？

原始观测高维（256×256 RGB = 196,608 维）。潜在空间预测**压缩观测为紧凑表示** $z_t$，保留决策相关信息，过滤感知噪声。

### 确定性 vs 随机

- **RSSM 混合设计**：确定性循环状态 $h_t$ + 随机组件 $z_t$
- DreamerV2: 离散类别潜在优于连续高斯

### 连续 vs 离散 Tokenization

- **IRIS**: VQ-VAE 将帧转为离散 token，Transformer 做 next-token prediction
- **STORM**: 混合策略——连续潜在 + Transformer 动力学

### JEPA（联合嵌入预测架构）

- LeCun 提出：在嵌入空间直接预测未来表示，而非重建观测
- 避免像素重建瓶颈
- I-JEPA（图像）、V-JEPA（视频）、V-JEPA 2（大规模视频理解）
- MuZero 也可视为 JEPA：潜在动力学仅优化奖励和价值预测

### 开放挑战

1. 潜在空间坍塌
2. 表示漂移
3. 解纠缠
4. 可扩展性

---

## 五、里程碑系统时间线

| 年份 | 里程碑 |
|------|--------|
| 1974 | Minsky 框架系统理论 |
| 1990 | Sutton Dyna 架构 |
| 2018 | Ha & Schmidhuber 神经世界模型 |
| 2019 | DreamerV1 |
| 2020 | MuZero 超人表现 |
| 2021 | DreamerV2 离散潜在 |
| 2022 | LeCun JEPA 提案, DreamerV3 |
| 2023 | LLM 中的世界知识 |
| 2024 | Sora, UniSim, Cosmos, Genie |
| 2025-2026 | 交互式世界模型爆发 |

---

## 六、核心挑战

1. **误差复合**：多步预测误差累积
2. **Sim-to-Real 迁移**：学习/模拟与真实环境的差距
3. **评估标准碎片化**：缺乏统一基准
4. **潜在空间坍塌**：不同观测映射到相同编码
5. **表示漂移**：训练中潜在空间偏移
6. **模型偏差/利用**：策略利用学习动力学的不准确性
7. **目标不匹配**：模型似然优化不保证任务性能
8. **可扩展性**：保持紧凑但有表达力的表示
9. **非平稳性**：环境在被建模时发生变化（尤其金融）
10. **反事实不可辨识性**：从观测数据做因果推断的根本限制

---

## 七、未来方向

1. **统一多模态世界模型**：整合视觉、语言、本体感觉等
2. **基础规模交互式模拟器**：大规模预训练世界模型
3. **安全关键领域安全部署**
4. **想象链 (Chain-of-Imagination) 推理**：用接地的时空想象替代语言思维链
5. **因果结构化世界模型**
6. **跨域泛化**

---

## 八、关键参考系统

| 系统 | 类型 | 关键特性 |
|------|------|---------|
| Dreamer 系列 | MBRL | 潜在想象中学习策略 |
| MuZero | 规划 | 无需环境规则的超人表现 |
| Sora | 视频生成 | 大规模视频世界模拟 |
| Cosmos | 基础模型 | 开源世界基础模型平台 |
| Genie | 交互式 | 分钟级交互式世界生成 |
| JEPA 系列 | 自监督 | 表示空间预测 |

---

## 九、与你研究方向的关联

作为 DiT 推理系统优化方向的研究者，本综述中与你最相关的部分：

1. **扩散世界模型**（3.3 节）：DiT 作为世界模型骨干的架构设计
2. **推理加速**：KV Cache 管理、并行策略、蒸馏
3. **块扩散/半自回归**：Inferix 的 block-diffusion 范式
4. **Cosmos 平台**：DiT 架构的 AdaLN-LoRA、序列并行等优化
5. **交互式世界模型**：实时推理需求驱动的效率优化
