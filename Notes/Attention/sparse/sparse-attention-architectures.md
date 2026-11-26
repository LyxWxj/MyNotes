---
type: Note
status: Active
related_to:
  - "[[attention-evolution-overview]]"
  - "[[nsa-dsa]]"
  - "[[Notes/Attention/sparse/csa-hca]]"
  - "[[linear-attention-hybrids]]"
  - "[[compressed-attention-architectures]]"
---

# 稀疏注意力架构补充

调研截至 2026-11-26，以因果语言模型及多模态模型的语言主干为主。已有的 NSA、DSA、CSA 见上方关联笔记。新增的公开模型路线主要是 **局部与全局层混合**、**GQA 组级块稀疏（MiniMax MSA）**、**循环层与微块稀疏层混合（Qwen3.8-Flash-Next）**。

## 固定模式：局部窗口与全局层混合

窗口层对位置 $t$ 只读取最近 $W$ 个因果可见位置，单层 prefill 的注意力工作量为 $O(NW)$，decode 的驻留 KV 可以只保留窗口。设窗口层与全局层数量分别为 $L_s,L_g$，相同每条成本下，驻留条目数为 $L_s\min(N,W)+L_g N$，prefill 的注意力工作量为 $O(L_sNW+L_gN^2)$。每个全局层都还是一次全历史检索。跨层传递让堆叠的窗口层扩大感受野，但中间表示会限制对远端细节的直接访问。

| 模型 | 局部与全局注意力的部署 | 一手来源 |
|---|---|---|
| Gemma 4 | E2B 为 4:1，其余为 5:1。E2B/E4B 还跨层共享 KV，较大版本的全局层共享 K/V 投影 | [Gemma 4 Technical Report §2](https://arxiv.org/html/2607.02770) |
| gpt-oss | 128-token 带状窗口与全注意力交替，使用 GQA | [gpt-oss Model Card](https://arxiv.org/html/2508.10925v1) |
| MiMo-V2.5-Pro | 60 个 SWA 层、10 个全局层，窗口为 128。系统实现分别管理 SWA 与全局 KV 的池及前缀复用 | [MiMo-V2.5 部署报告 §2–3](https://arxiv.org/html/2607.13095) |
| Step-3.5-Flash | 每三个 SWA 层配置一个全注意力层 | [Step-3.5-Flash 官方仓库](https://github.com/stepfun-ai/Step-3.5-Flash) |

Longformer 的局部窗口加全局 token、BigBird 的局部加随机及全局连接，是同样使用固定结构的早期参照。上述生成模型通过不同层的模式切换引入全局连接。 [Longformer](https://arxiv.org/abs/2004.05150) · [BigBird](https://arxiv.org/abs/2007.14062)

## 内容路由：MoBA、MSA、QSA 与 InfLLM-V2

四者在选中块内保留原始 token KV 并执行因果 softmax。设块大小为 $B$，每个 query 选择 $k$ 个块，以固定头维数计，核心注意力的 prefill 为 $O(NkB)$，主 KV cache 随 $N$ 线性增长。索引成本取决于对**块摘要**还是对**原始 token**打分，及不同头能否共享打分或选择结果。

| 机制 | 索引打分与选择 | 落地证据 |
|---|---|---|
| MoBA | 用当前 query 给历史块的平均 key 打分，强制加入当前块并在块内使用因果掩码。无需额外的索引投影，router 复用主注意力的 Q/K。扫块摘要约 $O(N^2/B)$ | 月之暗面论文模型及开源实现。 [论文 §2](https://arxiv.org/html/2502.13189v1) · [MoBA](https://github.com/MoonshotAI/MoBA) |
| MiniMax Sparse Attention（MSA） | 每个 GQA 组一个索引 query 头，组间共用一个索引 key 头。先逐 token 打分，块内 max pooling，再分组 Top-K。固定保留 query 所在块 | MiniMax-M3 采用 MSA。 [MSA §3–4](https://arxiv.org/html/2606.13392v1) · [MiniMax-M3](https://github.com/MiniMax-AI/MiniMax-M3) |
| Qwen Sparse Attention（QSA） | 索引 key 先在不重叠微块上做均值池化，再做 RMSNorm 和 partial RoPE。多头索引分数经 ReLU 后求和，每个 query 得到一份块列表。主分支展开为原始 token，保留当前未完成块中可见的 token | Qwen3.8-Flash-Next 的 48 层中，36 层为 Gated DeltaNet，12 层为 QSA。Qwen3.8-Omni-Flash 的 Thinker 也采用 GDN + QSA。 [架构报告 §2.1](https://arxiv.org/html/2608.30320) · [FlashNext-RL §1](https://arxiv.org/html/2610.28998) · [Omni 报告](https://arxiv.org/html/2609.25611) |
| InfLLM-V2 | GQA 组内共用无额外参数的多阶段池化索引。初始块、近期块和检索块取并集后统一做 softmax。稠密与稀疏模式共享 KV 表示，可随长度切换 | MiniCPM4.1-8B。旧笔记中的 MiniCPM 介绍见 [[Notes/Audio/MiniCPM4]]。 [InfLLM-V2 §3](https://arxiv.org/html/2509.24663) |

**MSA 的代价集中在索引扫描与稀疏 kernel。** 报告以 $B=128,k=16$ 为例，每组每 query 最多读 2,048 个主 KV token。索引先对历史 token 逐个打分，prefill 仍有 $O(N^2)$ 的轻量项，decode 索引仍随 $N$ 增长。训练先用 dense 教师的注意力分布对齐索引器，再切到稀疏训练。prefill kernel 用选块结果构建反向索引，以 KV block 为外循环捞取相关 query，提升块内复用。报告的 1M 长度结果使用 109B 实验骨干与同头配置 GQA 对照。 [MSA §3.2–5.4](https://arxiv.org/html/2606.13392v1)

**QSA 降低索引长度。** Qwen3.8-Flash-Next 每个索引块含 4 个 token，最多选 512 块。在微块大小为 $B$ 时，扫描全部可见微块会带来 $O(N^2/B)$ 的 prefill 索引项，因此 $B$ 固定时仍然是二次增长。教师用 max pooling 把 token 级注意力分布聚合到块级，索引 key 侧用均值聚合。QSA 压的是 indexer K，主 KV 仍按原始 token 驻留。 [Qwen3.8-Next §2.1.2](https://arxiv.org/html/2608.30320) · [FlashNext-RL §1](https://arxiv.org/html/2610.28998)

## 2026 年下半年值得跟踪的架构和索引

- **LongCat Sparse Attention** 已用于开源的 LongCat-Flash-Lite-Sparse。它把 sink 与近期窗口直接加入注意力范围，索引器检索两者之间的历史。组内首层生成索引供后续层复用，并用块级粗筛加 token 级细排降低评分成本。可与 CSA2 对照，分别看哪些层复用了 Top-K、候选如何生成。 [LongCat Sparse Attention §3、§6](https://arxiv.org/html/2608.01662)
- **HySparse → HySparse2（小米）**。HySparse 的全注意力层给随后的稀疏层共享全局 KV 和块级选择。HySparse2 将共享扩为两级。外层将 self-decoder 的隐状态投影成对应 cross-decoder 全局层的 KV，因此大段 prefill 可跳过 cross-decoder。内层的稀疏层复用同组全注意力层的 KV 和选择结果，改用 token 级选择并强制纳入近期窗口。HySparse2 论文在 80B-A3B 实验模型上验证。 [HySparse](https://arxiv.org/abs/2602.03560) · [HySparse2 §3–4](https://arxiv.org/html/2609.26368)
- **Latent Sparse Attention（LSA）**。在学习出的索引空间聚类，再用 query 检索簇。Flat 扫全部簇中心，Tree 只沿入选分支向下搜索。论文有蒸馏改造与原生训练两条路线，可用来对照 QSA 的相邻微块和 CSA2 的分层候选池。10M 长度上的速度数字包含单 attention 层结果。 [LSA §3–4](https://arxiv.org/html/2610.42147)
- **LongCat Recall Sparse Attention（LRSA）**。沿时间分桶，对桶内 key 做容量均衡分组并构建可重用的页。先检索候选组，再对成员打分细排。论文集成 SGLang，处理索引增量构建和跨请求的前缀复用，适合作为 DSA 全前缀索引的系统对照。 [LRSA §3–5](https://arxiv.org/html/2611.27233)

## 给现有 checkpoint 加稀疏策略

- **Prefill**。MInference 按头使用 A-shape、Vertical-Slash 或 Block-Sparse 三类模式。FlexPrefill 随输入选择模式和计算预算。XAttention 用块内反对角线采样给块打分。三者主要削减长 prompt prefill 的注意力计算。 [MInference](https://arxiv.org/abs/2407.02490) · [FlexPrefill](https://arxiv.org/abs/2502.20766) · [XAttention](https://arxiv.org/abs/2503.16428)
- **Decode**。Quest 用 KV 页内每通道的最小值和最大值形成依赖 query 的相关性上界，按页挑选读入的 KV。长期 KV 仍可保留以供后续 query 检索。 [Quest](https://arxiv.org/abs/2406.10774)

## 视频 DiT 中的稀疏模式

**Sliding Tile Attention** 将局部三维窗口对齐到 GPU tile。**VSA** 先池化时空 tile 做粗粒度检索，再在选中的 tile 内计算 token 级注意力，可端到端训练。**Prism** 在视频与音频联合生成中根据局部特征变化和音视频耦合动态分配块形状，再进行块选择。这些方法的主要成本来自单次去噪中的时空 token 交互，不能直接用因果语言模型的 decode KV 账本衡量。 [Sliding Tile Attention](https://arxiv.org/abs/2502.04507) · [VSA](https://arxiv.org/abs/2505.13389) · [Prism](https://arxiv.org/html/2610.05416)
