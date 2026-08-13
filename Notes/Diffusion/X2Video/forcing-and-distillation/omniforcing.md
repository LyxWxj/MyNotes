---
type: Note
related_to:
  - "[[X2Video]]"
  - "[[self-forcing]]"
  - "[[causal-forcing]]"
  - "[[causal-video-model]]"
  - "[[rolling-forcing]]"
status: Active
url: https://arxiv.org/abs/2603.11647
date: 2026-03-13
---

# OmniForcing: Unleashing Real-time Joint Audio-Visual Generation

> [!info] 论文信息
> - **作者**: Yaofeng Su, Yuming Li, Zeyue Xue, Jie Huang, Siming Fu, Haoran Li, Ying Li, Zezhong Qian, Haoyang Huang, Nan Duan
> - **机构**: JD Explore Academy, 复旦大学, 北京大学, 香港大学
> - **版本**: v1: 2026-03-12 / v2: 2026-03-13（本文阅读版本，14 页）
> - **arXiv**: [2603.11647](https://arxiv.org/abs/2603.11647)
> - **项目主页**: https://omniforcing.com

## 摘要

近期联合音频 - 视觉扩散模型（LTX-2、Veo 3）实现了卓越的生成质量，但依赖双向全序列注意力导致高延迟，阻碍实时应用。OmniForcing 是**首个将离线双流双向扩散模型蒸馏为高保真流式自回归生成器的框架**。通过引入非对称块因果对齐（Asymmetric Block-Causal Alignment）、零截断 Global Prefix、Audio Sink Token（带 Identity RoPE）机制和联合 Self-Forcing 蒸馏范式，在单 GPU 上实现 ~25 FPS 的流式生成，同时保持与双向教师（LTX-2）相当的多模态同步和视觉质量。

> [!tip] 核心贡献
> 1. 首个将双向音视频扩散模型蒸馏为实时流式生成器的框架
> 2. 非对称块因果对齐 + 零截断 Global Prefix：解决 25:3 非整数频率比与多模态同步漂移
> 3. Audio Sink Token + Identity RoPE：解决双流因果化中音频流的 Softmax 坍塌与梯度爆炸
> 4. 联合 Self-Forcing 蒸馏：让视频/音频流动态自校正跨模态误差累积
> 5. 模态独立滚动 KV-Cache：单 GPU ~25 FPS，TTFC ~0.7s（教师需 ~197s，约 35× 加速）

## 动机与挑战

### 问题背景

> [!warning] 双向联合模型的延迟瓶颈
> - LTX-2 等联合音视频 DiT 依赖双向全序列注意力：生成单帧需要模型同时看到整条物理时间线
> - 复杂度随序列长度与模态维度二次增长 → 巨大 TTFC 延迟，无法用于交互、实时或流式场景

> [!note] 两条既有路线的局限
> 1. **级联流水线**：T2V+V2A（FoleyGen、Diff-Foley、FoleyCrafter、MMAudio）或 T2A+A2V（TPoS 等）→ 割裂联合时间分布，难以做细粒度跨模态同步（如视觉动作对突发声学事件的即时反应）；次级模态必须等主模态成形，从根上阻塞连续流式
> 2. **仅视频因果自回归**：CausVid（~9.4 FPS）、Self-Forcing、Causal-Forcing、Rolling-Forcing → 仅限视觉域；直接移植到双流架构会因模态间极端时间不对称引发训练不稳定（稀疏模态信息缺失）

### 核心挑战

> [!danger] 三大挑战
> - **挑战 1：频率不对称性**：视频 latent 3 FPS vs 音频 latent 25 FPS，25:3 非整数比 → 严格逐帧因果掩码导致破坏性特征截断和时间错位
> - **挑战 2：条件分布偏移**：双向预训练知识转入因果域时，后验从全局知情退化为截断因果：p(xi | x1:N, c) → p(xi | x1:i, c)；信息缺失对音频流尤甚 → Softmax 坍塌与梯度爆炸
> - **挑战 3：曝光偏差**：长序列自回归展开中预测误差累积，被放大为跨模态去同步

## 方法详解

### 问题定义

将联合分布按物理秒分解为 K+1 个同步块 {B₀, …, B_K}（K = 生成的物理秒数）：

$$p(V, A | c) = p(B_0 | c) \prod_{k=1}^{K} p(B_k | B_{<k}, c)$$

### 1. 非对称块因果对齐

#### 宏块对齐与 VAE stride 的数学契合

> [!important] 物理时间对齐设计
> - **宏块**：ΔT = 1s 窗口完美封装 3 个视频 latent + 25 个音频 latent，无分数余数
> - 因果 VAE 时间压缩的首帧 stride = 1，后续帧用全感受野 stride（视频 8，音频 4）→ 整个 latent 序列长度严格满足：
>
>   $$N_v = 1 + K \cdot f_v, \quad N_a = 1 + K \cdot f_a \quad (f_v = 3, f_a = 25)$$
>
> - 常数项 "+1" 即 t≈0 处的初始 latent V₀、A₀ → 架构上无法塞进 1s 标准块 → 显式合并为 **Global Prefix B₀**

> [!note] Global Prefix（零截断设计）
> - B₀ 内注意力**无条件双向**，对所有未来 token 全局可见（类似 LLM 的 system prompt，不受因果衰减影响）
> - 既是 " 零截断 " 的完美对齐，也是长序列自回归生成中稳健的跨模态语义锚点
> - B₀ 中 token 的块索引 τ = 0，对任意 query 天然满足 τ(k) ≤ τ(q) → 数学上保证全局可见

#### Token 密度不对称

| 模态 | 每 latent 帧 token 数 | 每 1s 块 token 数 |
|------|----------------------|-------------------|
| 视频 | patchify 成 Hv·Wv 个 token | 3 × 384 = 1152（本文配置） |
| 音频 | mel 频段折叠进通道维 → 1 个 token | 25 |

#### 四路非对称因果掩码（块内双向、块间严格因果）

前缀之外的块索引：τv(q) = 1 + ⌊(q − HvWv) / (3·HvWv)⌋，τa(q) = 1 + ⌊(q − 1) / 25⌋

| 注意力类型 | 掩码规则 |
|-----------|---------|
| 视频自注意力 V→V | τ_v(k) ≤ τ_v(q) |
| 音频自注意力 A→A | τ_a(k) ≤ τ_a(q) |
| 视频→音频 V→A | τ_a(k) ≤ τ_v(q) |
| 音频→视频 A→V | τ_v(k) ≤ τ_a(q) |

- 块间严格因果防止未来信息泄漏，块内允许双向流动
- 尽管 token 密度严重不对称，两种模态的时间感受野仍在物理块边界同步扩展

### 2. 三阶段蒸馏流水线

> [!abstract] 总体思路
> 先把 " 少步去噪能力 " 与 " 因果生成范式 " 解耦，再顺序注入：**双向 DMD → 因果 ODE 回归 → 联合 Self-Forcing**

**Stage I：双向 DMD** — 保持原全局注意力感受野，把预训练模型蒸馏为少步双向学生：

$$\mathcal{L}_{Bi-DMD} = \lambda_v \mathcal{L}_{DMD}^v + \lambda_a \mathcal{L}_{DMD}^a$$

为后续因果迁移提供高质量、易回归的教师 ODE 轨迹。

**Stage II：因果 ODE 回归** — 换上块因果掩码，回归 Stage I 教师 vφ 的流匹配速度场：

$$\mathcal{L}_{ODE} = \mathbb{E}_{t, \mathbf{x}_t} \left[ \lambda_v \| v_\theta^v(\mathbf{x}_t, c) - v_\phi^v(\mathbf{x}_t, c) \|_2^2 + \lambda_a \| v_\theta^a(\mathbf{x}_t, c) - v_\phi^a(\mathbf{x}_t, c) \|_2^2 \right]$$

其中 x_t = [V_t, A_t] 为联合噪声 latent，t 为流匹配时间。目标是修正权重对因果掩码的适应不良，让模型仅凭因果历史做出有效去噪预测。

### 3. Audio Sink Token 与 Identity RoPE

> [!danger] 问题根源：Softmax 坍塌
> - 音频每块仅 25 个 token；早期块的可见历史极短（新块第一个 token 只能注意自己和前面几个 token）
> - 归一化分母极小 → Softmax 退化为近 one-hot（熵趋近零）；饱和区指数非线性把微小 logit 扰动急剧放大 → 梯度方差爆炸（‖∇L‖ → ∞），bf16/fp16 下产生 NaN
> - 视频因空间 patchify 每块有数百 token（3×384），不受此问题困扰 → 不稳定主要来自音频流

> [!success] 解决方案
> - 在音频序列前预置 S = 16 个可学习 Sink Token，永久锚定在全局前缀 B₀ 内
> - 数学上把早期音频 token 的注意力分母从 i 扩展到 i+S，恢复注意力熵，打破 Softmax 坍塌；物理上充当 " 软全局记忆缓冲 "，吸收异常 logit 扰动
> - 灵感来源：LLM 的 attention sink（StreamingLLM）与视觉模型的 register tokens（DINOv2）
> - **Identity RoPE 约束**：cos(θ_sink) = 1，sin(θ_sink) = 0 → 旋转变换解析退化为恒等映射（RoPE(x) = x）→ Sink Token 成为位置无关的语义锚点，避免标准 RoPE 注入虚假的物理时间偏差

### 4. 联合 Self-Forcing 蒸馏（Stage III）

> [!note] 因果 DMD + 联合 Self-Forcing
> - 训练时模型自回归展开**自己的预测** B̂k（而非 ground-truth 历史）；其 KV 以无噪声方式计算并追加到滚动缓存
> - 冻结双向教师 Rφ，用因果 DMD（含 backward simulation，DMD2 风格）评估生成轨迹：
>
> $$\mathcal{L}_{SF} = \sum_{k=1}^{K} \mathbb{E}_{\hat{\mathcal{B}}_{<k}} \left[ \nabla_\theta \text{KL} \left( G_\theta(z_k | \text{KV}_{<k}, c) \| R_\phi(z_k | c) \right) \right]$$
>
> - 耦合展开迫使视频与音频流动态适应彼此的预测漂移 → 保证严格的跨模态同步

### 5. 模态独立滚动 KV-Cache 与非对称并行推理

> [!tip] 推理优化
> - 14B 视频分支与 5B 音频分支在 Transformer 层内解耦：各自独立 FFN 子层，视频/音频自注意力之间无数据依赖，仅在跨模态注意力边界（A2V、V2A）短暂同步
> - 模态独立滚动 KV-Cache：每步上下文复杂度降至 O(L)（L = 缓存窗口内 latent 帧数）
> - 天然支持**非对称张量并行**：把更多算力分给更重的视频流，是多 GPU 扩展的实际路径

## 训练配置

> [!note] 实现细节
> - 32 GPU，bf16，global batch 32，lr 2e-5
> - Stage I（双向 DMD）：2000 步；Stage II（因果 ODE 回归）：3000 步；Stage III（Self-Forcing DMD）：2000 步；Stage I/III 使用 backward simulation
> - S = 16 Audio Sink Token + Identity RoPE，从 Stage II 起启用
> - CFG 引导：wv = 3（视频），wa = 5（音频）
> - 数据：Mixkit 视频片段 + Open-Sora-Plan 字幕 + AudioCaps 音频字幕；所有字幕由 Gemma 3 12B 重写为音视频连贯的描述（蒸馏自预训练教师，相对紧凑的数据集即可）

## 评估协议

- **JavisBench 四维**：AV-Quality（FVD、FAD）；Text-Consistency（TV-IB、TA-IB via ImageBind；CLIP；CLAP）；AV-Consistency（AV-IB、AVHScore）；AV-Synchrony（JavisScore、DeSync）
- **TTFC 定义**：生成并解码 Global Prefix（B₀）+ 第一个流式块（B₁）的墙钟时间；此后生成与解码并发，实现无中断流式播放
- **VBench**：官方支持对用户提供视频逐条评分，用于蒸馏保真度分析

## 实验结果

### 推理效率

> [!success] 效率对比
>
> | 指标 | OmniForcing | LTX-2 教师 |
> |------|------------|-----------|
> | 5s 480p 音视频 | ~5.7s | ~197s |
> | TTFC | ~0.7s | ~197s |
> | FPS | ~25 | - |
> | 加速比 | **~35×** | 基准 |

### JavisBench 全量对比

> [!note] 完整结果（↑ 越高越好，↓ 越低越好）
>
> | 模型 | 规模 | FVD↓ | FAD↓ | CLIP↑ | CLAP↑ | AV-IB↑ | AVHScore↑ | DeSync↓ | Runtime↓ |
> |------|------|------|------|-------|-------|--------|-----------|---------|----------|
> | TempoTokens (T2A+A2V) | 1.3B | 539.8 | - | 0.205 | - | 0.139 | 0.122 | 1.532 | 20s |
> | TPoS (T2A+A2V) | 1.0B | 839.7 | - | 0.229 | - | 0.124 | 0.129 | 1.493 | 19s |
> | ReWaS (T2V+V2A) | 0.6B | - | 9.4 | - | 0.280 | 0.110 | 0.104 | 1.071 | 17s |
> | Seeing&Hearing (T2V+V2A) | 0.4B | - | 7.6 | - | 0.263 | 0.160 | 0.143 | 1.099 | 25s |
> | FoleyCrafter (T2V+V2A) | 1.2B | - | 9.1 | - | 0.383 | 0.193 | 0.186 | 0.952 | 16s |
> | MMAudio (T2V+V2A) | 0.1B | - | 6.1 | - | 0.407 | 0.198 | 0.182 | 0.849 | 15s |
> | MM-Diffusion (T2AV) | 0.4B | 2311.9 | 27.5 | 0.181 | 0.079 | 0.119 | 0.109 | 0.875 | 9s |
> | JavisDiT (T2AV) | 3.1B | 204.1 | 7.2 | 0.302 | 0.391 | 0.197 | 0.179 | 1.039 | 30s |
> | UniVerse-1 (T2AV) | 6.4B | 194.2 | 8.7 | 0.309 | 0.245 | 0.104 | 0.098 | 0.929 | 13s |
> | JavisDiT++ (T2AV) | 2.1B | 141.5 | 5.5 | 0.316 | 0.424 | 0.198 | 0.184 | 0.832 | 10s |
> | LTX-2（双向教师） | 19B | 125.4 | 4.6 | 0.318 | 0.442 | 0.318 | 0.298 | 0.384 | 197s |
> | **OmniForcing** | 19B | 137.2 | 5.7 | **0.322** | 0.401 | 0.269 | 0.254 | 0.392 | **5.7s** |

- FVD/FAD 仅次于教师，超过所有其他基线；**CLIP 全场最佳**（0.322，超教师 0.318）
- TV-IB 0.287 排第二（教师 0.290）；TA-IB 0.162 与 CLAP 0.401 略低于教师但紧咬 JavisDiT++（0.164 / 0.424）；JavisScore 0.208
- AV-IB 0.269 / AVHScore 0.254 均排第二，远超非教师基线（最近者 0.198）
- DeSync 0.392 紧贴教师 0.384，远优于 JavisDiT++ 0.832；与教师的小幅差距归因于因果感受野替代双向全序列注意力——这是换取流式能力的固有代价

### VBench 视觉保真度

> [!important] 蒸馏学生超越教师（每帧质量指标）
>
> | 指标 | LTX-2 | OmniForcing |
> |------|-------|-------------|
> | Aesthetic Quality | 0.569 | **0.595** (+0.026) |
> | Imaging Quality | 0.574 | **0.594** (+0.020) |
> | Motion Smoothness | 0.993 | **0.995** |
> | Subject Consistency | 0.945 | **0.955** (+0.010) |
> | Temporal Flickering | 0.988 | **0.989** |

- 与既往 DMD 蒸馏工作一致：学生可在单样本质量指标上反超教师

### 消融实验（Stage II，3k 步，一步去噪损失 @ σ=0.5）

> [!warning] Sink Token 数量影响（均配 Identity RoPE）
>
> | S | 收敛 | 最大梯度范数 | 可视化 | Loss (σ=0.5) |
> |---|------|-------------|--------|--------------|
> | 24 | 稳定 | 9.15 | 正常 | 0.110 |
> | 16 | 稳定 | 9.23 | 正常 | **0.081** |
> | 8 | 稳定 | 21.95 | 正常 | 0.129 |
> | 4 | 稳定 | 49.71 | 正常 | 0.141 |
> | ≤2 | NaN | ∞ | 噪声 | -（Softmax 坍塌） |

> [!note] 替代稳定化方案对比（S=16 基线）
>
> | 配置 | 收敛 | Loss (σ=0.5) | 结论 |
> |------|------|--------------|------|
> | S=16 + Identity RoPE | 稳定 | **0.081** | 最佳稳定性 - 质量权衡 |
> | S=16 + 递增 RoPE | 稳定 | 0.402 | 位置偏差，输出带噪 |
> | QK-Norm | 稳定 | 0.232 | 过度归一化压制注意力对比度 |
> | Tanh-Gated Attention | 平台期 | 1.258 | tanh 饱和抑制梯度，注意力被破坏，出现块状伪影 |
> | 无稳定器 | NaN | - | Softmax 坍塌 |

## 与相关工作的对比

| 方法 | 模态 | 流式 | FPS | 备注 |
|------|------|------|-----|------|
| CausVid | 仅视频 | ✓ | ~9.4 | 首个流式扩散范式（非对称 DMD 蒸馏） |
| Self-Forcing | 仅视频 | ✓ | - | 解决暴露偏差（训练时自回归展开） |
| Causal-Forcing | 仅视频 | ✓ | - | 严格因果一致性，帧级注入性 |
| Rolling-Forcing | 仅视频 | ✓ | - | 分钟级长上下文生成 |
| **OmniForcing** | **音视频联合** | **✓** | **~25** | 首个双流因果蒸馏，单 GPU 实时 |

## 局限与开放问题

> [!warning] 未验证的边界
> - 仅基于 LTX-2 验证；对其他教师架构（单流 token 融合、非 RoPE 位置编码、不同参数量）的迁移性未知
> - 设计依赖 25:3 latent 帧率与特定 VAE stride；对其他（含非整数）频率比的鲁棒性未评估
> - 宏块固定 1s，未消融块时长（如 0.25s / 0.5s）在延迟 - 稳定性 - 同步之间的权衡
> - 论文展望：非对称张量并行扩展、更高分辨率与更长序列、以及向多模态多流同步任务（如多传感器融合、时间语言接地）的通用化

## 个人思考

> [!tip] 研究启示
> 1. **物理时间对齐**设计优雅：利用 VAE stride 特性（首帧 stride=1）把 " 架构必然性 " 变成 " 设计优势 "（零截断 Global Prefix），自然解决 25:3 非整数频率比
> 2. **Audio Sink Token** 把 LLM 的 attention sink 现象迁移到多模态扩散模型：通过扩大注意力分母恢复熵，从根上抑制梯度爆炸——比 QK-Norm / Tanh-Gated 等通用稳定器更对症
> 3. **模态独立 KV-Cache + 层内解耦**使单 GPU 实时推理成为可能，工程价值高；非对称张量并行是未来扩展的自然路径
> 4. 三阶段蒸馏流水线（少步能力 → 因果适应 → 曝光偏差缓解）与 [[causal-video-model]] 中 " 注入性 " 原则一脉相承：先用双向 DMD 提供易回归的教师轨迹，再因果化，最后用 Self-Forcing 修复分布
> 5. 消融实验是方法可信度的关键：每个设计选择（S≥4、Identity RoPE、替代稳定器对比）都有定量证据支撑
