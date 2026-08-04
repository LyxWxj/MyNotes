---
type: Note
status: Active
related_to:
  - "[[attention-evolution-overview]]"
  - "[[mla]]"
  - "[[csa-hca]]"
---

# NSA 与 DSA:从块级稀疏到令牌级稀疏

原生 softmax 注意力要求每个 query 看全部 key,长上下文下 $O(n^2)$ 不可承受。稀疏注意力的思路是:**让模型自己学会"该看哪些位置"**。DeepSeek 的两代工作给出两条路径:

- **NSA**(Native Sparse Attention, DeepSeek-NSA):压缩 + 块级选择 + 滑动窗口三路并行,门控融合。
- **DSA**(DeepSeek Sparse Attention, DeepSeek-V3.2):Lightning Indexer 打分,token 级 top-k 选择,核心注意力是 MLA 的 MQA 模式。

两者都是**可训练的稀疏注意力**(不是靠手工规则),且训练与推理使用同一套稀疏结构。

## 1. NSA:原生稀疏注意力

### 1.1 动机

人的阅读不是均匀扫描:重要段落精读、次要段落略读。NSA 把每个 query 的注意力分成三条通路:

1. **压缩分支**:把若干 token 压缩成一个粗粒度 KV 条目,用少量块覆盖全局信息(花小成本扫一遍)。
2. **选择分支**:用粗粒度打分选出最相关的 top-n 块,在块内做细粒度注意力(精读)。
3. **滑动窗口分支**:最近 $w$ 个 token 全量注意力(保底局部依赖)。

### 1.2 数学原理(教学简化)

设块大小 $m$,第 $i$ 个块为 $\{h_{mi}, \dots, h_{m(i+1)-1}\}$。压缩分支把块压成一个粗粒度 key/value:

$$\tilde{k}_i^{c} = \frac{1}{m}\sum_{j=0}^{m-1} h_{mi+j} W^{cK},\qquad
\tilde{v}_i^{c} = \frac{1}{m}\sum_{j=0}^{m-1} h_{mi+j} W^{cV}$$

(论文中用可学习的压缩网络,教学版用平均池化 + 投影。)

选择分支:query 用粗粒度 key 打分,取 top-n 块,再对选中块的原始 token 做标准注意力:

$$S_t = \mathrm{Top}\text{-}n\!\left(\Big\{\tilde{q}_t^\top \tilde{k}_i^{c}\Big\}_{i < \lfloor t/m \rfloor}\right),\qquad
o_t^{\mathrm{sel}} = \mathrm{Attn}\big(q_t,\; \{k_j, v_j\}_{j \in \cup S_t}\big)$$

窗口分支直接取最近 $w$ 个 token:

$$o_t^{w} = \mathrm{Attn}\big(q_t,\; \{k_j, v_j\}_{j=t-w}^{t-1}\big)$$

三路输出用输入相关的门控融合:

$$u_t = g_t^{(1)} \odot o_t^{c} + g_t^{(2)} \odot o_t^{\mathrm{sel}} + g_t^{(3)} \odot o_t^{w},\qquad
g_t = \mathrm{Softmax}\big(x_t W^g\big) \in \mathbb{R}^3$$

复杂度:选择打分只发生在 query 与 $\approx n/m$ 个块之间,核心注意力只看 $n \cdot m + w$ 个 token,总成本从 $O(n^2)$ 降到约 $O(n \cdot n/m)$(块数 × query 数)。

![[nsa-fig2.png|DeepSeek-NSA 论文 Figure 2:NSA 架构总览(压缩 / 选择 / 滑动窗口三路)]]

### 1.3 PyTorch 教学实现

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NativeSparseAttention(nn.Module):
    """NSA 教学简化:压缩 + 块级 top-n 选择 + 滑动窗口,三路门控融合。
    简化点:块压缩用平均池化;选择打分用整头 query 均值;因果性用掩码保证。"""

    def __init__(self, d_model, n_heads, head_dim, block_size=8,
                 n_sel_blocks=4, n_window=16, d_comp=32):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, head_dim
        self.block_size, self.n_sel_blocks, self.n_window = block_size, n_sel_blocks, n_window

        self.w_q = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.w_k = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.w_v = nn.Linear(d_model, n_heads * head_dim, bias=False)
        # 压缩分支:块级粗粒度 KV
        self.w_ck = nn.Linear(d_model, d_comp, bias=False)
        self.w_cv = nn.Linear(d_model, d_comp, bias=False)
        self.w_comp_o = nn.Linear(d_comp, head_dim, bias=False)   # 压缩分支输出投影
        self.w_gate = nn.Linear(d_model, 3, bias=False)           # 三路门控
        self.w_o = nn.Linear(n_heads * head_dim, d_model, bias=False)

    def _compress(self, x: torch.Tensor):
        """平均池化 + 投影,得到块级 KV:(B, nb, d_comp)"""
        B, L, D = x.shape
        m = self.block_size
        nb = L // m
        xb = x[:, :nb * m].view(B, nb, m, D).mean(dim=2)     # (B, nb, D)
        return self.w_ck(xb), self.w_cv(xb)

    def _attn_selected(self, q, k, v, sel):
        """按选中的块 gather 原始 KV 做因果注意力。
        q/k/v: (B, H, L, hd);sel: (B, L, ns) 块号"""
        B, H, L, hd = q.shape
        m, ns = self.block_size, sel.shape[-1]
        offs = torch.arange(m, device=q.device)
        idx = (sel.unsqueeze(-1) * m + offs).reshape(B, L, ns * m)     # (B, L, ns*m)
        k5 = k.transpose(1, 2)[:, :, :, None, :].expand(B, L, H, ns * m, hd)
        v5 = v.transpose(1, 2)[:, :, :, None, :].expand(B, L, H, ns * m, hd)
        k_sel = k5.gather(2, idx[:, :, None, :, None].expand_as(k5))  # (B, L, H, ns*m, hd)
        v_sel = v5.gather(2, idx[:, :, None, :, None].expand_as(v5))
        qh = q.transpose(1, 2)                                        # (B, L, H, hd)
        scores = torch.einsum("blhd,bljhd->blhj", qh, k_sel) / math.sqrt(hd)
        causal = idx < torch.arange(L, device=q.device)[:, None]      # (B, L, ns*m)
        scores = scores.masked_fill(~causal[:, None], float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        return torch.einsum("blhj,bljhd->blhd", attn, v_sel).transpose(1, 2)  # (B, H, L, hd)

    def _attn_window(self, q, k, v, w):
        """最近 w 个 token 的因果注意力"""
        B, H, L, hd = q.shape
        kw, vw = k[:, :, -w:], v[:, :, -w:]
        scores = torch.einsum("bhld,bhjd->bhlj", q, kw) / math.sqrt(hd)
        causal = (torch.arange(w, device=q.device) + (L - w)) < \
            torch.arange(L, device=q.device)[:, None]
        scores = scores.masked_fill(~causal[None, None], float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        return torch.einsum("bhlj,bhjd->bhld", attn, vw)

    def _attn_compressed(self, q, ck, cv):
        """压缩分支:query 对块级 KV 的注意力"""
        B, H, L, hd = q.shape
        nb = ck.shape[1]
        scores = torch.einsum("bhld,bld->bhl", q, ck) / math.sqrt(ck.shape[-1])  # (B, H, L, nb)
        s_idx = torch.arange(nb, device=q.device).view(1, 1, 1, nb)
        t_block = (torch.arange(L, device=q.device) // self.block_size).view(1, 1, L, 1)
        scores = scores.masked_fill(~(s_idx < t_block), float("-inf"))
        attn = torch.softmax(scores, dim=-1)                       # (B, H, L, nb)
        o = torch.einsum("bhlj,bjd->bhld", attn, cv)               # (B, H, L, d_comp)
        return self.w_comp_o(o)                                    # -> (B, H, L, hd)

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape
        q = self.w_q(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        ck, cv = self._compress(x)

        # 选择分支:块级打分 -> top-n 块(仅限 query 所在块之前的块)
        nb = ck.shape[1]
        s_idx = torch.arange(nb, device=x.device).view(1, 1, nb)
        t_block = (torch.arange(L, device=x.device) // self.block_size).view(1, L, 1)
        q_pool = q.mean(dim=1)                                     # (B, L, hd) 教学简化
        block_scores = q_pool @ ck.transpose(1, 2) / math.sqrt(ck.shape[-1])
        block_scores = block_scores.masked_fill(~(s_idx < t_block), float("-inf"))
        k_sel = min(self.n_sel_blocks, nb)
        sel = block_scores.topk(k_sel, dim=-1).indices                # (B, L, ns)
        o_sel = self._attn_selected(q, k, v, sel)

        o_win = self._attn_window(q, k, v, self.n_window)
        o_comp = self._attn_compressed(q, ck, cv)

        g = torch.softmax(self.w_gate(x), dim=-1)                    # (B, L, 3)
        o = (g[:, :, 0][:, None, :, None] * o_sel +
             g[:, :, 1][:, None, :, None] * o_win +
             g[:, :, 2][:, None, :, None] * o_comp)                 # (B, H, L, hd)
        o = o.transpose(1, 2).reshape(B, L, -1)
        return self.w_o(o)


if __name__ == "__main__":
    torch.manual_seed(0)
    m = NativeSparseAttention(64, n_heads=4, head_dim=16,
                              block_size=4, n_sel_blocks=2, n_window=8, d_comp=16)
    x = torch.randn(2, 24, 64)
    y = m(x)
    assert y.shape == x.shape
    print("NSA forward OK:", y.shape)
```

## 2. DSA:DeepSeek Sparse Attention

### 2.1 动机

NSA 的选择粒度是**块**,一个 query 要么精读整块要么跳过整块。DSA 把粒度细化到 **token 级**:用一个极轻量的"闪电索引器"(Lightning Indexer)给每个 token 打分,每个 query 只保留 top-k 个 token 的 KV 条目。

### 2.2 数学原理(DeepSeek-V3.2-Exp)

**Lightning Indexer**。对 query token $h_t \in \mathbb{R}^d$ 与前面的 token $h_s$,索引分数为:

$$I_{t,s} = \sum_{j=1}^{H^I} w_{t,j}^{I} \cdot \mathrm{ReLU}\!\left(q_{t,j}^{I\top} k_s^{I}\right)$$

其中 $H^I$ 是索引器头数;$q_{t,j}^I \in \mathbb{R}^{d^I}$ 与标量 $w_{t,j}^I$ 由 $h_t$ 派生,$k_s^I \in \mathbb{R}^{d^I}$ 由 $h_s$ 派生。选择 ReLU 是为了吞吐(FP8 下便宜);索引器维度小、头数少,打分矩阵可以近似认为"免费"。

**token 级 top-k 选择与核心注意力**:

$$u_t = \mathrm{Attn}\!\left(h_t,\; \big\{c_s \;\big|\; I_{t,s} \in \mathrm{Top}\text{-}k(I_{t,:})\big\}\right)$$

$c_s$ 是 MLA 低秩压缩后的 KV 条目(DeepSeek-V3.2 中 MLA 以 MQA 模式运行,所有头共享同一份 KV),因此核心注意力是"稀疏化的 MLA-MQA"。

![[dsa-fig2.png|DeepSeek-V3.2 论文 Figure 2:MLA 实例化下的 DSA 注意力架构(Lightning Indexer + top-k 选择)]]

**两阶段训练**:

阶段一(密集预热,约 1000 步):冻结除索引器外的全部参数,用稠密注意力作为教师对齐:

$$\mathcal{L}^{I} = \sum_{t} D_{\mathrm{KL}}\!\left(p_{t,:} \;\|\; \mathrm{Softmax}(I_{t,:})\right),\qquad p_{t,:} = \text{稠密 MLA 的注意力权重}$$

学习率 $10^{-3}$。

阶段二(稀疏训练,约 15000 步):只对被选中的集合 $S_t = \{s \mid I_{t,s} \in \mathrm{Top}\text{-}k(I_{t,:})\}$ 计算 KL 损失,索引器输入 detach;主模型只受语言模型损失:

$$\mathcal{L}^{I,\mathrm{sparse}} = \sum_{t} D_{\mathrm{KL}}\!\left(p_{t,S_t} \;\|\; \mathrm{Softmax}(I_{t,S_t})\right)$$

学习率 $7.3 \times 10^{-6}$。这样索引器学会"模仿稠密注意力",而主模型学会"在稀疏集合上仍然表现良好"。

### 2.3 PyTorch 教学实现

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LightningIndexer(nn.Module):
    """DSA 闪电索引器:I_{t,s} = Σ_j w_{t,j} · ReLU(q_{t,j}·k_s)"""

    def __init__(self, d_model, n_heads=4, d_indexer=32):
        super().__init__()
        self.n_heads = n_heads
        self.w_q = nn.Linear(d_model, n_heads * d_indexer, bias=False)
        self.w_k = nn.Linear(d_model, n_heads * d_indexer, bias=False)
        self.w_w = nn.Linear(d_model, n_heads, bias=False)

    def forward(self, h_q: torch.Tensor, h_k: torch.Tensor):
        """h_q: (B, Lq, d);h_k: (B, Lk, d) -> I: (B, Lq, Lk)"""
        B, Lq, _ = h_q.shape
        d = self.w_q.out_features // self.n_heads
        q = self.w_q(h_q).view(B, Lq, self.n_heads, d)
        k = self.w_k(h_k).view(B, h_k.shape[1], self.n_heads, d)
        w = self.w_w(h_q)                                        # (B, Lq, H^I)
        relu = F.relu(torch.einsum("blhd,bjhd->blhj", q, k))     # (B, Lq, Lk, H^I)
        return torch.einsum("bljh,blh->blj", relu, w)            # (B, Lq, Lk)


class DSAAttention(nn.Module):
    """DSA 教学简化:索引器 top-k 选择 + MQA 核心注意力"""

    def __init__(self, d_model, n_heads, head_dim, indexer, top_k=16):
        super().__init__()
        self.n_heads, self.head_dim, self.top_k = n_heads, head_dim, top_k
        self.indexer = indexer
        self.w_q = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.w_k = nn.Linear(d_model, head_dim, bias=False)      # MQA:单 KV 头
        self.w_v = nn.Linear(d_model, head_dim, bias=False)
        self.w_o = nn.Linear(n_heads * head_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape
        q = self.w_q(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(B, L, 1, self.head_dim)
        v = self.w_v(x).view(B, L, 1, self.head_dim)

        I = self.indexer(x, x)                                   # (B, L, L)
        causal = torch.arange(L, device=x.device)[None, :, None] > \
            torch.arange(L, device=x.device)[None, None, :]      # t > s
        I = I.masked_fill(~causal, float("-inf"))
        k_eff = max(1, min(self.top_k, L - 1))
        sel = I.topk(k_eff, dim=-1).indices                      # (B, L, k) token 索引

        qh = q.transpose(1, 2)                                   # (B, L, H, hd)
        k4 = k.transpose(1, 2)[:, :, None].expand(B, L, self.n_heads, L, self.head_dim)
        v4 = v.transpose(1, 2)[:, :, None].expand(B, L, self.n_heads, L, self.head_dim)
        idx = sel[:, :, None, :, None].expand(B, L, self.n_heads, k_eff, self.head_dim)
        k_sel = k4.gather(2, idx)                                # (B, L, H, k, hd)
        v_sel = v4.gather(2, idx)
        q5 = qh[:, :, :, None, :].expand(B, L, self.n_heads, k_eff, self.head_dim)
        scores = torch.einsum("blhd,bljhd->blhj", q5, k_sel) / math.sqrt(self.head_dim)
        valid = sel < torch.arange(L, device=x.device)[:, None]   # 兜底因果掩码(早期位置不足 k)
        scores = scores.masked_fill(~valid[:, None], float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        o = torch.einsum("blhj,bljhd->blhd", attn, v_sel)        # (B, L, H, hd)
        o = o.reshape(B, L, -1)
        return self.w_o(o)


def indexer_kl_loss(I: torch.Tensor, p_dense: torch.Tensor, top_k=None) -> torch.Tensor:
    """索引器 KL 损失。
    I: 索引器 logits (B, L, L);p_dense: 稠密注意力权重 (B, L, L);
    top_k 不为 None 时进入稀疏阶段:只统计选中集合 S_t。"""
    logp = torch.log_softmax(I, dim=-1)
    if top_k is not None:
        idx = I.topk(top_k, dim=-1).indices
        mask = torch.zeros_like(I, dtype=torch.bool).scatter(-1, idx, True)
        logp = logp.masked_fill(~mask, 0.0)
        p_dense = p_dense.masked_fill(~mask, 0.0)
    return (p_dense * (p_dense.clamp_min(1e-8).log() - logp)).sum(dim=-1).mean()


if __name__ == "__main__":
    torch.manual_seed(0)
    idx = LightningIndexer(64, n_heads=4, d_indexer=16)
    m = DSAAttention(64, n_heads=4, head_dim=16, indexer=idx, top_k=8)
    x = torch.randn(2, 20, 64)
    y = m(x)
    assert y.shape == x.shape
    # 模拟 KL 损失调用
    I = torch.randn(2, 20, 20)
    p = torch.softmax(torch.randn(2, 20, 20), dim=-1)
    loss_dense = indexer_kl_loss(I, p)
    loss_sparse = indexer_kl_loss(I, p, top_k=8)
    print("DSA forward OK:", y.shape, "| KL loss:", loss_dense.item(), loss_sparse.item())
```

## 3. NSA vs DSA 对比

| 维度 | NSA | DSA |
|---|---|---|
| 选择粒度 | 块(block) | token |
| 检索方式 | query 与块级压缩 key 打分,top-n 块 | Lightning Indexer 打分,top-k token |
| 打分器 | 粗粒度 query/key 投影 | 多头 ReLU + 可学习权重 $w^I$(低维、FP8) |
| 非稀疏通路 | 压缩分支 + 滑动窗口(三路) | 滑动窗口(部分实现)/ 无额外压缩分支 |
| 核心注意力 | 选中块内的原始 token | MLA 的 MQA 模式(共享低秩 KV) |
| 训练 | 端到端 + 三路门控 | 密集 KL 预热 → 稀疏 KL + LM 损失 |
| 代表模型 | DeepSeek-NSA(原型) | DeepSeek-V3.2 / V3.2-Exp |

DSA 之后,DeepSeek-V4 的 [[csa-hca]] 进一步在稀疏选择**之前**增加 KV 压缩,把"索引器检索的条目"从原始 token 换成压缩块。

## 4. 参考

- DeepSeek-AI, *NSA: Native Sparse Attention*, arXiv:2502.11089
- DeepSeek-AI, *DeepSeek-V3.2-Exp: Boosting Long-Context Efficiency with DeepSeek Sparse Attention*(DSA 公式 1-2、两阶段训练)
- DeepSeek-AI, *DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models*, arXiv:2512.02556
