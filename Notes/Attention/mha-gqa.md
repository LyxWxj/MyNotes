---
type: Note
status: Active
related_to:
  - "[[attention-evolution-overview]]"
  - "[[mla]]"
---

# MHA 与 GQA:多头注意力与分组查询注意力

多头注意力(Multi-Head Attention, MHA)是 Transformer 的注意力基础形态;多查询注意力(Multi-Query Attention, MQA)与分组查询注意力(Grouped-Query Attention, GQA)则沿着"共享 KV 头"的方向压缩推理时的 KV cache。三者共享同一套 softmax 注意力数学,区别只在于 **KV 头的数量**。

## 1. 背景与动机

- MHA 每个注意力头独立投影 Q/K/V,表达能力最强,但 KV cache 与头数成正比。
- 解码阶段是 **memory-bound**:每生成一个 token 都要重读全部 KV cache,因此 KV cache 大小直接决定解码吞吐。
- MQA 让所有 query 头共享一份 K/V(单 KV 头),KV cache 缩到 $1/n_h$,但质量损失明显。
- GQA 在两者之间取折中:$n_{kv}$ 个 KV 头,每个被连续的 $n_h/n_{kv}$ 个 query 头共享。当 $n_{kv}=1$ 时退化为 MQA,当 $n_{kv}=n_h$ 时退化为 MHA。

## 2. 数学原理

### 2.1 MHA

设输入 $x_t \in \mathbb{R}^d$,$d = n_h \cdot d_h$。MHA 先把输入投影成 $n_h$ 组 query/key/value:

$$q_{t,i} = x_t W^Q_i,\quad k_{t,i} = x_t W^K_i,\quad v_{t,i} = x_t W^V_i,\qquad W^Q_i, W^K_i, W^V_i \in \mathbb{R}^{d \times d_h}$$

第 $i$ 个头对位置 $t$ 的输出是因果 softmax 注意力:

$$o_{t,i} = \sum_{j=1}^{t} \mathrm{Softmax}_j\!\left(\frac{q_{t,i}^\top k_{j,i}}{\sqrt{d_h}}\right) v_{j,i}$$

最后拼接所有头并投影回模型宽度:

$$O_t = \mathrm{Concat}(o_{t,1}, o_{t,2}, \dots, o_{t,n_h})\, W^O,\qquad W^O \in \mathbb{R}^{d \times d}$$

$\sqrt{d_h}$ 缩放防止点积进入 softmax 饱和区。RoPE(旋转位置编码)在 $q,k$ 上按位置施加:

$$\theta_i = 10000^{-2i/d_h},\qquad
\mathrm{RoPE}(q, t) = \left[ q_{2i}\cos(t\theta_i) - q_{2i+1}\sin(t\theta_i),\; q_{2i}\sin(t\theta_i) + q_{2i+1}\cos(t\theta_i) \right]_{i=0}^{d_h/2-1}$$

![[transformer-fig2.png|Transformer 论文 Figure 2:缩放点积注意力(左)与多头注意力(右)]]

### 2.2 MQA

MQA 只有 1 个 KV 头(记 $k_j, v_j \in \mathbb{R}^{d_h}$),所有 query 头共享:

$$o_{t,i} = \sum_{j=1}^{t} \mathrm{Softmax}_j\!\left(\frac{q_{t,i}^\top k_j}{\sqrt{d_h}}\right) v_j$$

### 2.3 GQA

GQA 有 $n_{kv}$ 个 KV 头,第 $i$ 个 query 头使用第 $\lfloor i \cdot n_{kv} / n_h \rfloor$ 个 KV 头:

$$o_{t,i} = \sum_{j=1}^{t} \mathrm{Softmax}_j\!\left(\frac{q_{t,i}^\top k_{j, \lfloor i \cdot n_{kv} / n_h \rfloor}}{\sqrt{d_h}}\right) v_{j, \lfloor i \cdot n_{kv} / n_h \rfloor}$$

![[gqa-fig2.png|GQA 论文 Figure 2:分组查询方法总览(MHA → GQA → MQA 的头共享关系)]]

### 2.4 KV cache 对比(每层、每 token)

| 架构 | KV 头数 | 缓存大小 | 相对 MHA |
|---|---|---|---|
| MHA | $n_h$ | $2 n_h d_h = 2d$ | $1\times$ |
| GQA | $n_{kv}$ | $2 n_{kv} d_h = 2d \cdot n_{kv}/n_h$ | $n_{kv}/n_h$ |
| MQA | 1 | $2 d_h$ | $1/n_h$ |

## 3. PyTorch 教学实现

下面的 `GroupedQueryAttention` 用一个 `n_kv_heads` 参数同时覆盖三种架构:$n_{kv}=n_h$ 是 MHA,$n_{kv}=1$ 是 MQA,其余是 GQA。代码自包含,并附带 RoPE 工具函数(后续 MLA、CSA 笔记会复用同一思路)。

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """把 (..., D) 按相邻维度对做 90° 旋转,用于 RoPE"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def precompute_rope_cache(seq_len: int, dim: int, theta: float = 10000.0):
    """预计算 cos/sin 缓存:(L, D)"""
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, inv_freq)          # (L, D/2)
    emb = torch.cat((freqs, freqs), dim=-1)   # (L, D)
    return emb.cos(), emb.sin()


def apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """q/k: (B, H, L, D);cos/sin: (1, 1, L, D)"""
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k


class GroupedQueryAttention(nn.Module):
    """n_kv_heads == n_heads -> MHA;n_kv_heads == 1 -> MQA;其余 -> GQA"""

    def __init__(self, d_model, n_heads, n_kv_heads, head_dim=None, dropout=0.0):
        super().__init__()
        self.d_model, self.n_heads, self.n_kv_heads = d_model, n_heads, n_kv_heads
        self.head_dim = head_dim or d_model // n_heads
        assert n_heads % n_kv_heads == 0
        self.qkv_proj = nn.Linear(
            d_model, (n_heads + 2 * n_kv_heads) * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, rope: bool = False):
        """x: (B, L, d_model) -> (B, L, d_model)"""
        B, L, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(
            [self.n_heads * self.head_dim,
             self.n_kv_heads * self.head_dim,
             self.n_kv_heads * self.head_dim],
            dim=-1,
        )
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)       # (B, H, L, D)
        k = k.view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)   # (B, H_kv, L, D)
        v = v.view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if rope:
            cos, sin = precompute_rope_cache(L, self.head_dim)
            q, k = apply_rotary(q, k, cos[None, None], sin[None, None])

        # GQA:把每个 KV 头复制给组内的 query 头(repeat_interleave 保持分组顺序)
        groups = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)

        attn = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout)   # (B, H, L, D)
        out = attn.transpose(1, 2).reshape(B, L, -1)
        return self.o_proj(out)


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 16, 128)
    for n_kv in (1, 2, 8):                       # MQA / GQA / MHA
        m = GroupedQueryAttention(128, n_heads=8, n_kv_heads=n_kv, head_dim=16)
        y = m(x, rope=True)
        assert y.shape == x.shape
        print(f"n_kv={n_kv}: {y.shape} OK")
```

## 4. 要点与对比

- **实现技巧**:把 Q/K/V 拼在一次投影里,再按 head 拆分,是主流实现方式;`repeat_interleave` 保证分组顺序正确。
- **训练与推理不对称**:GQA 的省内存只在推理(以及训练时对激活的处理)体现;训练 FLOPs 与 MHA 几乎相同。
- **KV cache 之外的开销**:GQA 还需配合位置编码(如 RoPE)与缓存管理,才能支撑 vLLM 这类推理引擎的连续批处理。
- **后续演进**:GQA 仍是"每 token 一份 KV";[[mla]] 把 KV 压缩进低秩潜变量,把缓存从 $2 n_{kv} d_h$ 进一步降到 $d_c + d_h^R$。

## 5. 参考

- Vaswani et al., *Attention Is All You Need*, arXiv:1706.03762
- Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need* (MQA), arXiv:1911.02150
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, arXiv:2305.13245
