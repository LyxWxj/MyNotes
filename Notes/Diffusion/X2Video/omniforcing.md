---
type: Note
related_to: "[[X2Video]]"
status: Active
url: https://arxiv.org/abs/2603.11647
date: 2026-03-13
---

# OmniForcing: Unleashing Real-time Joint Audio-Visual Generation

> [!info] 论文信息
> - **作者**: Yaofeng Su, Yuming Li, Zeyue Xue, Jie Huang, Siming Fu, Haoran Li, Ying Li, Zezhong Qian, Haoyang Huang, Nan Duan
> - **机构**: JD Explore Academy, 复旦大学, 北京大学, 香港大学
> - **日期**: 2026-03-13
> - **arXiv**: [2603.11647](https://arxiv.org/abs/2603.11647)
> - **项目主页**: https://omniforcing.com

## 摘要

近期的联合音频-视觉扩散模型实现了卓越的生成质量，但由于双向注意力依赖导致高延迟，阻碍了实时应用。OmniForcing 是**首个将离线双流双向扩散模型蒸馏为高保真流式自回归生成器的框架**。通过引入非对称块因果对齐、Audio Sink Token 机制和联合 Self-Forcing 蒸馏范式，在单 GPU 上实现 ~25 FPS 的流式生成，同时保持与双向教师模型相当的多模态同步和视觉质量。

> [!tip] 核心贡献
> - 首个将双向音视频扩散模型蒸馏为实时流式生成器的框架
> - 解决双流架构因果蒸馏中的训练不稳定问题
> - 单 GPU 上实现 ~25 FPS，TTFC 仅 ~0.7s（教师模型需 ~197s）

## 动机与挑战

### 问题背景

> [!warning] 现有方法的局限
> 1. **级联流水线**（先视频后音频或反之）：割裂联合分布，阻碍连续流式生成
> 2. **仅视频自回归方法**（CausVid, Self-Forcing）：局限于视觉域，直接扩展到双流架构会导致训练崩溃

### 核心挑战

> [!danger] 三大挑战
> - **挑战1：频率不对称性**：视频 3 FPS vs 音频 25 FPS，严格的逐帧因果掩码会导致破坏性的特征截断和时间错位
> - **挑战2：条件分布偏移**：将双向预训练知识转换为因果域时，信息缺失导致 Softmax 坍塌和梯度爆炸
> - **挑战3：曝光偏差**：长序列展开中的预测误差累积会被放大为跨模态去同步

## 方法详解

### 整体架构

> [!abstract] 三阶段蒸馏流水线
> ```
> Stage I: 双向 DMD → 少步快速去噪能力
> Stage II: 因果 ODE 回归 → 适应因果掩码
> Stage III: 联合 Self-Forcing → 缓解曝光偏差
> ```

基于 LTX-2（14B 视频流 + 5B 音频流）教师模型，将联合分布因子化为 K+1 个同步块：

$$p(V, A | c) = p(B_0 | c) \prod_{k=1}^{K} p(B_k | B_{<k}, c)$$

### 1. 非对称块因果对齐

> [!important] 物理时间对齐设计
> - **宏块对齐**：1 秒时间窗口完美封装 3 个视频 latent + 25 个音频 latent
> - **全局前缀 Token (B₀)**：包含初始 V₀ 和 A₀，在物理时间 t≈0s 处自然锚定
> - B₀ 内注意力无条件双向，对所有未来 token 全局可见（类似 LLM 的 system prompt）

**四路非对称因果掩码**：

| 注意力类型 | 掩码规则 |
|-----------|---------|
| 视频自注意力 V→V | τ_v(k) ≤ τ_v(q) |
| 音频自注意力 A→A | τ_a(k) ≤ τ_a(q) |
| 视频→音频 V→A | τ_a(k) ≤ τ_v(q) |
| 音频→视频 A→V | τ_v(k) ≤ τ_a(q) |

### 2. Audio Sink Token 与 Identity RoPE

> [!danger] 问题根源
> 音频每个块仅 25 个 token，早期块的可见历史极短，Softmax 分布退化为近 one-hot 向量（熵趋近于零），导致梯度方差爆炸（‖∇L‖ → ∞），fp16/bf16 精度下产生 NaN 损失。

> [!success] 解决方案
> - **Audio Sink Token**：在音频序列前预置 S=16 个可学习的 Sink Token，永久锚定在全局前缀 B₀ 内
> - **作用**：将早期音频 token 的注意力分母从 i 扩展到 i+S，打破极端的 Softmax 坍塌
> - **Identity RoPE 约束**：对 Sink Token 强制 cos(θ_sink)=1, sin(θ_sink)=0，使其成为位置无关的语义锚点

### 3. 联合 Self-Forcing 蒸馏

> [!note] Stage III: 因果 DMD + 联合 Self-Forcing
> - 模型在训练时自回归展开序列（而非依赖 ground-truth 历史）
> - 视频和音频流动态适应彼此的预测漂移
> - 确保严格的跨模态同步

$$\mathcal{L}_{SF} = \sum_{k=1}^{K} \mathbb{E}_{\hat{B}_{<k}} \left[ \nabla_\theta \text{KL} \left( G_\theta(z_k | \text{KV}_{<k}, c) \| R_\phi(z_k | c) \right) \right]$$

### 4. 模态独立滚动 KV-Cache

> [!tip] 推理优化
> - 14B 视频分支和 5B 音频分支在 Transformer 层内解耦
> - 视频自注意力和音频自注意力无数据依赖，仅在跨模态注意力边界同步
> - 每步上下文复杂度降至 O(L)
> - 自然支持非对称张量并行

## 实验结果

### 推理效率

> [!success] 效率对比
> | 指标 | OmniForcing | LTX-2 教师 |
> |------|------------|-----------|
> | 5s 480p 视频 | ~5.7s | ~197s |
> | TTFC | ~0.7s | ~197s |
> | FPS | ~25 | - |
> | 加速比 | **~35×** | 基准 |

### 生成质量（JavisBench）

> [!note] 质量指标对比
> | 模型 | FVD↓ | FAD↓ | CLIP↑ | AV-IB↑ | DeSync↓ |
> |------|------|------|-------|--------|---------|
> | LTX-2 (教师) | 125.4 | 4.6 | 0.318 | 0.318 | 0.384 |
> | **OmniForcing** | **137.2** | **5.7** | **0.322** | **0.269** | **0.392** |
> | JavisDiT++ | 141.5 | 5.5 | 0.316 | 0.198 | 0.832 |

### VBench 视觉保真度

> [!important] 蒸馏学生超越教师
> | 指标 | LTX-2 | OmniForcing |
> |------|-------|------------|
> | Aesthetic Quality | 0.569 | **0.595** (+0.026) |
> | Imaging Quality | 0.574 | **0.594** (+0.020) |
> | Subject Consistency | 0.945 | **0.955** (+0.010) |

### 消融实验

> [!warning] Sink Token 数量影响
> - S ≥ 4：稳定收敛，视觉质量正常
> - S ≤ 2：NaN 梯度，Softmax 坍塌
> - Identity RoPE vs 标准 RoPE：Identity RoPE 损失 0.081 vs 0.402

> [!note] 替代稳定化方案对比
> - **QK-Norm**：收敛但损失高（0.232），对比度被抑制
> - **Tanh-Gated Attention**：损失停滞在 1.258，注意力被破坏
> - **Audio Sink Token + Identity RoPE**：最佳稳定性-质量权衡

## 技术亮点总结

> [!tip] 关键创新点
> 1. **非对称块因果对齐**：基于物理时间的宏块设计，自然桥接 25:3 的非整数频率比
> 2. **Audio Sink Token + Identity RoPE**：解决音频流的 Softmax 坍塌问题
> 3. **联合 Self-Forcing 蒸馏**：视频和音频流协同适应跨模态误差累积
> 4. **模态独立滚动 KV-Cache**：实现真正的单 GPU 实时推理

## 与相关工作的对比

| 方法 | 类型 | 流式 | FPS | 多模态同步 |
|------|------|------|-----|-----------|
| CausVid | 仅视频 | ✓ | ~9.4 | - |
| Self-Forcing | 仅视频 | ✓ | - | - |
| Causal-Forcing | 仅视频 | ✓ | - | - |
| **OmniForcing** | **音视频联合** | **✓** | **~25** | **✓** |

## 个人思考

> [!tip] 研究启示
> 1. **物理时间对齐**的设计非常优雅，利用 VAE 的 stride 特性自然地解决了非整数频率比问题
> 2. **Audio Sink Token** 机制从 LLM 的 attention sink 现象获得灵感，迁移应用到多模态扩散模型中
> 3. **模态独立 KV-Cache** 的设计使得单 GPU 实时推理成为可能，具有很强的工程价值
> 4. 三阶段蒸馏流水线的设计思路（少步能力 → 因果适应 → 曝光偏差缓解）值得借鉴
