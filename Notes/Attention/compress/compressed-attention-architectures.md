---
type: Note
status: Active
related_to:
  - "[[attention-evolution-overview]]"
  - "[[mla]]"
  - "[[Notes/Attention/compress/csa-hca]]"
  - "[[sparse-attention-architectures]]"
---

# 压缩注意力与 KV 共享架构补充

调研截至 2026-11-26。[[mla]] 已覆盖每 token 的低秩表示，[[Notes/Attention/compress/csa-hca]] 已覆盖 DeepSeek-V4 的跨位置压缩。本篇补上 **DeepSeek-V4.1-Flash 的 CSA2 与 CED**、**跨层 KV 共享**，以及 **学习如何合并历史位置**的代表方法。

## KV 账本

设第 $l$ 个 KV 生产层保留 $m_l$ 条历史记录，每条含 $c_l$ 个缓存元素，按 $b_l$ 字节存储。主缓存可按 $\sum_l m_l c_l b_l$ 估算，再加索引 key、量化元数据、窗口、对齐和块分配开销。几种设计可以同时生效。

| 压缩轴 | 架构实例 | 直接节省的部分 |
|---|---|---|
| 条目宽度 | GQA、MLA、Gemma 4 共享 K/V 投影 | 每条 KV 的元素数 $c_l$ |
| 历史条目数 | CSA/HCA、Activation Beacon、DMC | 每个生产层留存的条目数 $m_l$ |
| 生产层数 | Cross-Layer Attention、YOCO、CSA2、Gemma 4 E2B/E4B | 需要产生并驻留独立 KV 的层数 |
| 每元素存储 | KIVI、DeepSeek-V4.1-Flash 的 FP4 global KV | 每元素字节数 $b_l$；还要计入 scale 等元数据。 [KIVI](https://arxiv.org/abs/2402.02750) · [DeepSeek-V4.1-Flash §2.4.4](https://arxiv.org/html/2609.19969) |

只复用 Top-K 索引会减少评分和选择计算，主 KV 的驻留条目数保持原样。每 query 只读少量候选时，尚未读取的 KV 也可能继续驻留以服务未来 query。[[sparse-attention-architectures]] 中的 MSA 和 QSA 是直接的对照。

## DeepSeek-V4.1-Flash：CSA2 与 CED

**CSA2 改了压缩器、KV 与索引的生产关系。** 压缩历史段改用不重叠分组，去掉压缩器内的绝对位置嵌入，indexer K 由主 KV 投影得到。主注意力继续使用选中的全局条目和本层的 SWA KV，压缩率为 1 时可直接使用未合并的主 KV。各层按下表配置，主 Q 和本层 SWA KV 都在当前层计算。 [DeepSeek-V4.1-Flash §2.3](https://arxiv.org/html/2609.19969)

| CSA2 模式 | 全局主 KV 和 indexer K | Top-K |
|---|---|---|
| Full | 生成自己的主 KV，由主 KV 投影 indexer K | 本层索引器打分并选出 |
| Reindex | 复用前面 Full 层的主 KV 和 indexer K | 用本层的索引 query 在共享 cache 上重新选择 |
| Reuse | 复用前面 Full 层的主 KV | 复用与这份 KV 对应的最近一次选择 |

**分层索引器减少重复扫描。** CED 的 decoder 中，第一个 Full 层扫描所有因果可见的全局条目，并另外以块内最大索引分数选出一批候选块。后续 Reindex 层在候选池内各自选 Top-K，Reuse 层沿用结果。固定池大小可约束后续 Reindex 层的单步打分量，第一个 Full 层仍需扫全历史。 [DeepSeek-V4.1-Flash §2.3.2](https://arxiv.org/html/2609.19969)

**CED 减少长 prompt 的 prefill 层数。** 40 层主干分成 20 层因果 encoder 和 20 层 decoder。decoder 中负责生产全局 KV 的层从 encoder 末层隐状态做各自的投影。大部分 prompt token 因而可在 encoder 末尾完成全局缓存构建，decoder 的本地 SWA 状态通过尾部 bounded replay 补齐。报告区分了 HBM 驻留的全局 KV（890 B/token，约为 V4-Flash 的 1/4）和 SSD/host 上用于复用的持久 KV（约 1/8）。这些数字包含 CSA2、FP4 及相应的缓存策略。 [DeepSeek-V4.1-Flash §2.2、§2.4.4、§3.2](https://arxiv.org/html/2609.19969)

## 跨层与同条目共享

| 架构 | 数据依赖关系 | 对 prefill 的影响 |
|---|---|---|
| Cross-Layer Attention（CLA） | 一组相邻层共享生产层生成的 K/V。各个消费层保留自己的 query | 少生成、少存储 KV，消费层仍用自身隐状态执行 attention。 [CLA](https://arxiv.org/abs/2405.12981) |
| YOCO | 前部 self-decoder 用 gated retention 或 SWA 处理序列并构建全局 K/V，后部 cross-decoder 层复用这一份全局 cache | 全局 KV 构建可在进入 cross-decoder 前完成，生成首个输出所需的最后位置仍需经过 cross-decoder。 [YOCO §2–3](https://arxiv.org/html/2405.05254) |
| Gemma 4 E2B/E4B | 后部层按局部或全局类型复用前面同类型生产层的 cache | 共享后的层继续计算自己的查询。 [Gemma 4 §2](https://arxiv.org/html/2607.02770) · [Transformers 配置](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/configuration_gemma4.py) |

Gemma 4 12B、26B-A4B、31B 的全局层还展示了条目内部的共享。K 与 V 来自同一个投影，K 走独立的归一化权重与 partial RoPE。用 31B 的每头 512 维举例，缓存完整 V 与 K 的 128 个旋转维度，一个条目占 $512+128=640$ 个数，较独立存放 K/V 的 1,024 个数减少 37.5%。未旋转部分可通过权重折叠到 Q 来读共享区域，节省取决于跑模型的引擎是否采用相应 cache 布局。 [Gemma 4 §2](https://arxiv.org/html/2607.02770) · [Gemma 4 紧凑全局 KV 实现报告](https://arxiv.org/html/2611.23035)

## 学习把多个位置合成更少条目

| 方法 | 写入和读取 | 历史驻留量 |
|---|---|---|
| Compressive Transformer | 保留当前段与近期细粒度激活，将溢出的旧激活压进较粗的 memory 队列。attention 可同时读取原始与压缩激活，训练可用 attention reconstruction 辅助损失 | 细粒度与压缩队列均可设固定容量，进一步溢出时淘汰更早内容。 [Compressive Transformer §2](https://arxiv.org/html/1911.05507) |
| Activation Beacon | 一段 $l$ 个原始 token 后部署 $k$ 个 beacon token，通过专门的 attention mask 与训练把上下文写入 beacon 的 K/V。后续读取历史 beacon 与当前原始窗口 | 按 $l/k$ 压缩历史，随历史段数增长至设定的窗口预算。 [Activation Beacon §2](https://arxiv.org/html/2401.03462v2) |
| Dynamic Memory Compression（DMC） | 每层每头预测 `append` 或 `merge`。`merge` 将当前 K/V 加权并入最后一个 cache 条目，得到可变长度的连续分段。softmax 读取压缩后的条目 | 条目数等于分段数。通过继续预训练学习目标平均压缩比，分段可随输入变化。 [DMC §3](https://arxiv.org/html/2403.09636v2) |
| SlimKV | 在 beacon 压缩的基础上为 beacon 学习低秩 K/V，对不同层分配不同 rank。对 beacon key 去除 RoPE 后在 latent 空间直接计算，保留 raw token 的原有位置机制 | 同时减少条目数和 beacon 条目宽度。 [SlimKV §3](https://arxiv.org/html/2610.02953) |

DMC 展示了自适应分段的核心。若当前条目的 K 或 V 为 $x_*$、累积权重为 $z$，新输入为 $x_t$、权重为 $\omega_t$，合并后的条目为 $(zx_*+\omega_t x_t)/(z+\omega_t)$，随后令 $z\leftarrow z+\omega_t$。固定比率下，读取全部压缩条目的单步 attention 工作量仍与压缩后长度成正比。继续训练时增加的压缩约束，以及部署时不同头上长短不齐的序列，是它相对于固定块压缩的主要实现差异。 [DMC §3](https://arxiv.org/html/2403.09636v2)

Gisting 用少量 gist token 保存提示词的语义，适合重复使用的指令前缀。SnapKV 依据观察窗口的注意力投票保留代表性 prompt 位置，PyramidKV 再把保留预算按层做金字塔式分配。前者把前缀变成可复用的摘要，后二者保留部分原始条目，分别适合作为序列合并的对照。 [Gisting](https://arxiv.org/abs/2304.08467) · [SnapKV](https://arxiv.org/abs/2404.14469) · [PyramidKV](https://arxiv.org/abs/2406.02069)
