---
type: Note
related_to: "[[world-model]]"
status: Active
tags:
  - world-model
  - autonomous-intelligence
  - JEPA
  - representation-learning
  - energy-based-model
  - foundational
---

# A Path Towards Autonomous Machine Intelligence

## 元信息

| 字段 | 内容 |
|------|------|
| **作者** | Yann LeCun |
| **日期** | 2022-06 |
| **URL** | https://openreview.net/forum?id=BZ5a1r-kVsf |
| **Zotero Key** | CFIXWCTV |
| **类型** | 白皮书 (White Paper) |

## 摘要 (Abstract)

Yann LeCun 的奠基性白皮书，提出了自主机器智能的整体架构蓝图。核心提出 **Joint Embedding Predictive Architecture (JEPA)** 作为学习世界模型的关键框架。JEPA 学习从输入的一部分预测另一部分的**抽象表示（嵌入）**，而非预测原始像素。

## 核心贡献 (Key Contributions)

1. **提出 JEPA 框架** — Joint Embedding Predictive Architecture，一种在表示空间中进行预测的架构
2. **自主智能架构蓝图** — 提出了包含多个模块的完整自主智能系统架构
3. **批评生成式方法** — 论证了自回归 LLM 和生成式方法在实现真正理解和规划方面的根本局限
4. **能量模型视角** — 提出 energy-based models 和 JEPA 是更有前景的路径

## 方法 (Method)

### JEPA 核心思想

- **不在像素空间预测**: 与 GPT、diffusion model 等生成式方法不同
- **在表示空间预测**: 预测未来的抽象表示，忽略不可预测的细节
- **优势**: 能够聚焦于相关信息，避免在无意义的细节上浪费容量

### 自主智能架构模块

LeCun 提出的完整架构包含以下模块：

| 模块 | 功能 |
|------|------|
| **World Model** | 学习预测环境动态 |
| **Configurator** | 配置其他模块的行为 |
| **Perception** | 感知输入 |
| **Cost/Energy** | 评估状态的好坏 |
| **Actor** | 产生动作 |
| **Short-term Memory** | 短期记忆 |

### 关键论点

- 自回归 LLM 本质上是逐 token 预测，无法进行真正的规划和推理
- 生成式模型在像素级重建上浪费了大量容量
- JEPA 通过在抽象表示空间中预测，能够学习到更有意义的世界模型

## 实验结果 (Results)

- 作为白皮书，本文主要是概念性和框架性的
- 后续的 I-JEPA 和 V-JEPA 论文验证了 JEPA 框架的有效性

## 与其他工作的关系 (Related Work)

- **I-JEPA**: JEPA 框架在图像领域的具体实现 (Assran et al., 2023)
- **V-JEPA**: JEPA 框架在视频领域的具体实现 (Bardes et al., 2024)
- **自回归 LLM**: 本文批评的对象，如 GPT 系列
- **Diffusion Models**: 另一种生成式方法，本文认为不是通向自主智能的最佳路径
- **Energy-Based Models**: 本文推荐的理论框架

## 个人笔记 (Notes)

- 这是一篇**极具影响力的白皮书**，奠定了 JEPA 系列工作的理论基础
- **核心洞察**: 在表示空间中预测比在像素空间中预测更高效、更有意义
- LeCun 对自回归 LLM 的批评在当时颇具争议，但 JEPA 系列的后续成功证明了其观点的合理性
- **架构蓝图**虽然抽象，但为后续研究提供了清晰的方向
- 这篇论文应该是理解所有 JEPA 相关工作的必读文献
- 与当前 world model 研究的关系：JEPA 提供了一种非生成式的 world model 建模范式
- 值得注意的是，2022 年的这篇白皮书预见了 2023-2024 年 JEPA 系列工作的成功
