---
type: Note
status: Active
related_to:
  - "[[attention-evolution-overview]]"
  - "[[linear-attention-rwkv-mamba]]"
  - "[[kda]]"
  - "[[sparse-attention-architectures]]"
---

# 线性注意力与混合架构补充

调研截至 2026-11-26。[[linear-attention-rwkv-mamba]] 与 [[kda]] 已经介绍 RWKV、Mamba-1/2 和 Kimi Delta Attention。本篇补齐 **Gated DeltaNet 的规模化落地、Lightning Attention、Mamba 混合骨干**，以及近期针对循环记忆的结构改动。

## 从外积累加到可写的关联记忆

令 $S_t\in\mathbb R^{d_k\times d_v}$ 为每头状态，读取写为 $o_t=S_t^\top q_t$。下表略去对输出的归一化、门控以及实际模型中的短卷积。

| 机制 | 状态更新 | 区别 |
|---|---|---|
| RetNet | $S_t=\gamma S_{t-1}+k_t v_t^\top$ | 以多尺度衰减分别组织不同头的记忆，可并行、chunkwise 或递推求值。表内省略位置相位。 [RetNet](https://arxiv.org/abs/2307.08621) |
| Lightning Attention-2 | 指数因果掩码对应 $S_t=\gamma S_{t-1}+k_t v_t^\top$ | 块内做因果矩阵乘，块间带衰减地传递状态，面向 GPU I/O 与矩阵单元。 [Lightning Attention-2](https://arxiv.org/abs/2401.04658) |
| Gated Linear Attention（GLA） | $S_t=\operatorname{Diag}(a_t)S_{t-1}+k_t v_t^\top$ | 输入决定各 key 通道的遗忘速率，写入仍采用外积累加。 [GLA](https://arxiv.org/abs/2312.06635) |
| Gated DeltaNet（GDN） | $\bar S_t=\alpha_tS_{t-1}$，$S_t=\bar S_t+\beta_t k_t(v_t-\bar S_t^\top k_t)^\top$ | 每头标量遗忘加 delta rule。相似的 key 可以定向修正旧关联。 [GDN](https://arxiv.org/abs/2412.06464) |
| Kimi Delta Attention（KDA） | 令 GDN 的 $\bar S_t=\operatorname{Diag}(a_t)S_{t-1}$ | 在 delta rule 前逐 key 通道遗忘。详见 [[kda]]。 [Kimi Linear](https://arxiv.org/abs/2510.26692) |

GDN 的过程可以直接按读写理解。先衰减状态，再沿 $k_t$ 读出当前记忆，最后根据目标 $v_t$ 与旧读出的差写回。衰减门决定整头的保留速度，$\beta_t$ 控制这次定向修正的强度，$k_t$ 决定修正方向。Qwen 的实现还有短因果卷积、Q/K 的 L2 归一化和输出门控。训练及长 prefill 可在固定大小的 chunk 内构造三角更新，使用 WY 形式把状态转移改写为矩阵运算。decode 则多用单步递推。 [GDN §2–3](https://arxiv.org/abs/2412.06464) · [Qwen3.8-Next §2.1.1](https://arxiv.org/html/2608.30320)

## 公开模型如何混合不同层

| 代表模型 | 序列混合方式 | 全模型的历史开销 |
|---|---|---|
| Qwen3-Next、Qwen3.5 | GDN 与带输出门的全注意力交错，典型配置为 3:1 | GDN 层维护矩阵状态和短卷积状态，全注意力层保留逐 token KV。 [Qwen3-Next 配置](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/configuration_qwen3_next.py) · [Qwen3.5 配置](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/configuration_qwen3_5.py) · [Qwen3.5 实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py) |
| Qwen3.8-Flash-Next | 36 个 GDN 层与 12 个 QSA 层。在继续预训练阶段用 QSA 替换原全注意力层 | QSA 层保留原始 token KV，以压缩索引择块。Qwen3.8-Omni-Flash 的语言主干沿用这项设计，详见 [[sparse-attention-architectures]]。 [Qwen3.8-Next §2.1](https://arxiv.org/html/2608.30320) · [FlashNext-RL §1](https://arxiv.org/html/2610.28998) · [Qwen3.8-Omni 技术报告](https://arxiv.org/html/2609.25611) |
| MiniMax-01、MiniMax-M1 | 以 Lightning Attention 为主要算子。MiniMax-01 每七个 Lightning 层接一个 softmax 层，M1 从 MiniMax-Text-01 继续训练 | 线性层持续更新固定状态，全局层仍随序列长度增长。MiniMax-M3 使用 MSA 走块稀疏路线。 [MiniMax-01](https://arxiv.org/html/2501.08313v1) · [MiniMax-M1](https://arxiv.org/html/2506.13585v1) · [MiniMax-M3](https://github.com/MiniMax-AI/MiniMax-M3) |
| Kimi Linear | KDA 与 MLA 混合 | KDA 逐通道衰减，MLA 层存低秩 token KV，提供逐 token 全局检索。 [Kimi Linear](https://arxiv.org/abs/2510.26692) |
| Nemotron 3 Super / Ultra | 大量 Mamba-2 层加少量全局 attention 层 | SSM 状态固定，注意力层的 token KV 继续增长。Mamba-2 属于 SSM，此处因其长序列计算与部署性质一并比较。 [Nemotron 3 Super](https://arxiv.org/html/2604.12374) · [Nemotron 3 Ultra](https://arxiv.org/html/2606.15007) |
| Falcon-H1 | SSM 与 attention 在 block 内并行混合 | 同一层同时承担循环状态与逐 token 检索开销，可对照上面按层交错的布局。 [Falcon-H1](https://arxiv.org/abs/2507.22448) |

## 循环记忆近期的改动

- **Gated DeltaNet-2** 将原本共享的标量写入门拆为 key 侧擦除门 $b_t$ 和 value 侧写入门 $w_t$。逐通道衰减之后，以 $(b_t\odot k_t)$ 读取旧状态，再用 $k_t$ 把 $(w_t\odot v_t)$ 与旧读出之差写回。两门都固定为同一标量的广播时回到 KDA。论文给出与 WY 分块兼容的实现。 [Gated DeltaNet-2 §3](https://arxiv.org/html/2605.22791)
- **Integral DeltaNet** 在快速更新的 delta state 外加入一个积分状态，保留前者的加权历史。每头仍使用与上下文长度无关的状态容量，固定状态的常数项增加。 [Integral DeltaNet](https://arxiv.org/html/2611.32063)
- **RWKV-7、Mamba-3** 分别代表另外两条状态演进。RWKV-7 引入带输入依赖的状态转移与值更新。Mamba-3 用指数梯形离散化、复数状态和 MIMO SSM，分别扩展递推、状态跟踪能力和每次读写状态所能利用的计算。两者可与 delta rule 路线并读。 [RWKV-7](https://arxiv.org/abs/2503.14456) · [Mamba-3](https://arxiv.org/html/2603.15569v1)
- **视频 DiT** 也有按注意力范围混合的设计。SLA 用稀疏 softmax 处理最重要的交互，并用线性分支近似其余有贡献的交互。[[vdn]] 对应的 Video DeltaNet 保留局部双向 softmax 和首尾帧锚点，远端帧则用前后双向循环状态，按帧联合求解 delta 更新。 [SLA](https://arxiv.org/abs/2509.24006) · [Video DeltaNet](https://arxiv.org/html/2609.20744)

## 长上下文 serving 的比较口径

- **复杂度按层核算。** 固定头维和 chunk 大小时，纯循环层沿序列长度的工作量线性，单步状态大小恒定。若混合 $L_g$ 个全注意力层，仍有 $O(L_gN^2)$ 的注意力 prefill 项和 $O(L_gN)$ 的全局 KV 项。QSA 版本应将前一项换成相应的索引与稀疏核心开销。
- **缓存容量与状态访问量分别测。** 循环层通常每步读写一份包含多个头的 $d_k\times d_v$ 状态，并保留短卷积缓冲。短上下文时，矩阵状态可以比几页小 KV 更大。连续批处理应按每请求实际 dtype、头数与状态布局估算显存。
- **前缀复用要对齐两类状态。** 对纯 KV 层，可以按可用的 token 页恢复。循环层要恢复某个前缀边界上的状态及卷积尾部，或从更早的快照重放。可复用前缀粒度、状态快照数量和 speculative decode 的回滚由此成为系统成本。 [Tail-Replay](https://arxiv.org/abs/2608.30310) · [TreeWY](https://arxiv.org/abs/2608.20961)
