---
type: Note
status: Active
related_to:
  - "[[attention-evolution-overview]]"
  - "[[mla]]"
  - "[[nsa-dsa]]"
---

# CSA 与 HCA:压缩稀疏注意力与重度压缩注意力(DeepSeek-V4)

DeepSeek-V4 系列(arXiv:2606.19348)把两条思路合流:**先压缩 KV,再做稀疏/全量注意力**。CSA(Compressed Sparse Attention)与 HCA(Heavily Compressed Attention)分别负责"压缩 + 稀疏"与"重度压缩"两条通路,在层间混合部署。在 BF16 GQA8(head_dim=128)基线下,V4 在 1M 上下文时 KV cache 只有基线的 **约 2%**;V4-Flash 相对 V3.2 仅 10% FLOPs、7% KV cache,V4-Pro 为 27% FLOPs、10% KV cache。

## 1. 背景与动机

- [[nsa-dsa]] 的 DSA 已经把注意力"看哪些 token"稀疏化,但**每个 token 仍要存一份 KV 条目**。
- CSA/HCA 的想法:先把每 $m$ 个 token 的 KV 压缩成 1 个条目($1/m$ 压缩率),再在压缩条目上做稀疏选择(CSA)或全量注意力(HCA)。KV cache 与注意力 FLOPs 同时下降。
- CSA 用**双流重叠压缩**弥补纯分块压缩的信息损失:HCA 用更大的压缩率 $m' \gg m$ 换取极致压缩,放弃稀疏。

## 2. 数学原理

记号:输入 $H \in \mathbb{R}^{n \times d}$,$c$ 为头维度,$m$ 为 CSA 压缩块大小,$m' \gg m$ 为 HCA 压缩块大小。

### 2.1 CSA 压缩:双流重叠压缩(论文公式 9-12)

CSA 维护两条 KV 流及其压缩权重:

$$C^{a} = H W^{aKV},\quad C^{b} = H W^{bKV},\qquad
Z^{a} = H W^{aZ},\quad Z^{b} = H W^{bZ},\qquad W \in \mathbb{R}^{d \times c}$$

第 $i$ 个压缩条目把 **a 流第 $i$ 块 $[mi, m(i{+}1))$** 与 **b 流上一块 $[m(i{-}1), mi)$** 联合加权(权重来自各自 $Z$ 与可学习位置偏置 $B^a, B^b \in \mathbb{R}^{m \times c}$,共 $2m$ 个元素联合 softmax):

$$\big[S^a_{mi:m(i+1)-1};\; S^b_{m(i-1):mi-1}\big] =
\mathrm{Softmax}_{row}\!\Big(\big[Z^a_{mi:m(i+1)-1} + B^a;\; Z^b_{m(i-1):mi-1} + B^b\big]\Big)$$

$$C^{\mathrm{Comp}}_i = \sum_{j=mi}^{m(i+1)-1} S^a_j \odot C^a_j
+ \sum_{j=m(i-1)}^{mi-1} S^b_j \odot C^b_j \in \mathbb{R}^{c}$$

当 $i=0$ 时,b 流的 $Z$ 以 $-\infty$ 填充、$C$ 以 0 填充。由于 $C_i^{Comp}$ 的 b 流与 $C_{i-1}^{Comp}$ 的 a 流索引重叠(都是 $[m(i{-}1), mi)$),整体实际压缩率是 $1/m$(不是 $1/2m$)。

### 2.2 CSA 索引器(论文公式 13-17)

索引器 key 用同样的双流压缩得到 $K^{IComp} \in \mathbb{R}^{n/m \times c^I}$。query 侧走低秩:

$$c_t^Q = h_t W^{DQ},\qquad
\big[q_{t,1}^{I}; \dots; q_{t,n_h^I}^{I}\big] = c_t^Q W^{IUQ},\qquad
\big[w_{t,1}^{I}; \dots; w_{t,n_h^I}^{I}\big] = h_t W^{w}$$

索引分数(与 DSA 同型,但作用在压缩块上,$s < \lfloor t/m \rfloor$):

$$I_{t,s} = \sum_{h=1}^{n_h^I} w_{t,h}^{I} \cdot \mathrm{ReLU}\!\left(q_{t,h}^{I} \cdot K^{\mathrm{IComp}}_s\right)$$

选择:

$$\mathcal{C}^{\mathrm{SprsComp}}_t = \left\{ C^{\mathrm{Comp}}_s \;\big|\; I_{t,s} \in \mathrm{Top}\text{-}k(I_{t,:}) \right\}$$

### 2.3 CSA 核心注意力:MQA + 分组输出(公式 18-19)

query 同样低秩生成,$W^{UQ} \in \mathbb{R}^{d_c \times c \cdot n_h}$,然后对选中的压缩 KV 做 **MQA**(KV 是同一向量,同时充当 key 与 value):

$$q_t = c_t^Q W^{UQ},\qquad
o_{t,i} = \mathrm{CoreAttn}\!\left(\mathrm{query}=q_{t,i},\; \mathrm{key}=\mathcal{C}^{\mathrm{SprsComp}}_t,\; \mathrm{value}=\mathcal{C}^{\mathrm{SprsComp}}_t\right)$$

输出投影分组进行:$n_h$ 个头分成 $g$ 组,每组输出 $o^G \in \mathbb{R}^{c n_h/g}$ 先压到 $d_g < c n_h/g$,拼接后再投影回 $d$:

$$\hat{o}_t = \left[ o^{G'}_{t,1}; \dots; o^{G'}_{t,g} \right] W^O,\qquad
o^{G'}_{t,i} \in \mathbb{R}^{d_g}$$

### 2.4 HCA(论文公式 20-26)

HCA 不做重叠、不做稀疏,压缩率 $m' \gg m$:

$$C = H W^{KV},\quad Z = H W^{Z},\qquad
S_{m'i:m'(i+1)-1} = \mathrm{Softmax}_{row}\!\left(Z_{m'i:m'(i+1)-1} + B\right)$$

$$C^{\mathrm{Comp}}_i = \sum_{j=m'i}^{m'(i+1)-1} S_j \odot C_j$$

其余与 CSA 相同:低秩 query($c_t^Q = h_t W^{DQ}$)、MQA 核心注意力($o_{t,i} = \mathrm{CoreAttn}(q_{t,i}, C^{Comp}, C^{Comp})$)、分组输出。没有索引器与 top-k。

### 2.5 通用细节(论文 2.3.3)

- **Q/KV 归一化**:核心注意力前对每个 query 头和唯一 KV 头做 RMSNorm,替代 QK-Clip。
- **部分 RoPE**:RoPE 只施加在向量最后 64 维;由于 KV 同时作 key/value,输出 $o_{t,i}$ 会携带绝对位置信息,因此对输出最后 64 维按位置 $-i$ 反向旋转,使输出携带相对位置。
- **滑动窗口分支**:额外产生最近 $n_{win}$ 个未压缩 token 的 KV,与压缩 KV 一起参与核心注意力,保证块内信息可访问与局部依赖。
- **Attention Sink**(公式 27):每头一个可学习 sink logit $z'_h$,加入 softmax 分母:

$$s_{h,i,j} = \frac{\mathrm{Exp}(z_{h,i,j})}{\sum_{k}\mathrm{Exp}(z_{h,i,k}) + \mathrm{Exp}(z'_{h})}$$

- **低精度**:KV 混合存储(BF16 给 RoPE 维,FP8 给其余维);索引器用 FP4 计算。

## 3. PyTorch 教学实现

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def rms_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def compress_single(C: torch.Tensor, Z: torch.Tensor, B: torch.Tensor, m: int):
    """HCA 式单流压缩:C/Z: (B, n, c) -> (B, n//m, c)。
    块内 m 个元素按 softmax 权重加权求和(权重逐通道)。"""
    B_, n, c = C.shape
    n_comp = n // m
    C = C[:, :n_comp * m].view(B_, n_comp, m, c)
    Z = Z[:, :n_comp * m].view(B_, n_comp, m, c)
    S = torch.softmax(Z + B, dim=2)              # (B, n_comp, m, c)
    return (S * C).sum(dim=2)                    # (B, n_comp, c)


def compress_overlapped(Ca, Cb, Za, Zb, Ba, Bb, m: int):
    """CSA 双流重叠压缩:块 i = a 流 [mi, m(i+1)) + b 流 [m(i-1), mi),
    2m 个元素联合 softmax;i=0 时 b 流以 -inf/0 填充。"""
    B_, n, c = Ca.shape
    n_comp = n // m
    Ca_b = Ca[:, :n_comp * m].view(B_, n_comp, m, c)
    Za_b = Za[:, :n_comp * m].view(B_, n_comp, m, c)
    # b 流整体右移 m 个位置:第 i 块 = 原始 [m(i-1), mi)
    Cb_shift = F.pad(Cb[:, :(n_comp - 1) * m], (0, 0, m, 0))
    Zb_shift = F.pad(Zb[:, :(n_comp - 1) * m], (0, 0, m, 0))
    Cb_b = Cb_shift.view(B_, n_comp, m, c)
    Zb_b = Zb_shift.view(B_, n_comp, m, c)
    Zb_b[:, 0] = float("-inf")                   # i=0:b 流填充
    Cb_b[:, 0] = 0.0
    logits = torch.cat([Za_b + Ba, Zb_b + Bb], dim=2)   # (B, n_comp, 2m, c)
    S = torch.softmax(logits, dim=2)
    Sa, Sb = S[..., :m, :], S[..., m:, :]
    return (Sa * Ca_b).sum(dim=2) + (Sb * Cb_b).sum(dim=2)


class CSAAttention(nn.Module):
    """CSA 教学简化:双流重叠压缩 + 索引器 top-k + MQA 核心注意力 + 分组输出。
    简化点:未实现部分 RoPE 与 attention sink(见 2.5);窗口与压缩 KV 拼接后统一 softmax。"""

    def __init__(self, d_model, n_heads, head_dim, m=8, top_k=16, n_window=8,
                 d_comp=32, n_idx_heads=4, d_idx=16, n_groups=2, d_g=32):
        super().__init__()
        self.m, self.top_k, self.n_window = m, top_k, n_window
        self.n_heads, self.head_dim = n_heads, head_dim
        self.n_idx_heads, self.d_idx = n_idx_heads, d_idx
        self.n_groups, self.d_g = n_groups, d_g

        # 双流 KV 及压缩权重
        self.w_akv = nn.Linear(d_model, head_dim, bias=False)
        self.w_bkv = nn.Linear(d_model, head_dim, bias=False)
        self.w_az = nn.Linear(d_model, head_dim, bias=False)
        self.w_bz = nn.Linear(d_model, head_dim, bias=False)
        self.b_a = nn.Parameter(torch.randn(m, head_dim) * 0.1)
        self.b_b = nn.Parameter(torch.randn(m, head_dim) * 0.1)
        # 索引器 key 压缩(教学简化:单流;论文中按双流压缩)
        self.w_ik = nn.Linear(d_model, d_idx, bias=False)
        self.w_iz = nn.Linear(d_model, d_idx, bias=False)
        self.b_i = nn.Parameter(torch.randn(m, d_idx) * 0.1)
        # 索引器
        self.w_iq = nn.Linear(d_comp, n_idx_heads * d_idx, bias=False)
        self.w_iw = nn.Linear(d_model, n_idx_heads, bias=False)
        # query 低秩 + 窗口 KV
        self.w_dq = nn.Linear(d_model, d_comp, bias=False)
        self.w_uq = nn.Linear(d_comp, n_heads * head_dim, bias=False)
        self.w_kw = nn.Linear(d_model, head_dim, bias=False)
        self.w_vw = nn.Linear(d_model, head_dim, bias=False)
        # 分组输出
        self.w_g = nn.Linear(n_heads * head_dim // n_groups, d_g, bias=False)
        self.w_o = nn.Linear(d_g * n_groups, d_model, bias=False)

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape
        m = self.m
        C = compress_overlapped(self.w_akv(x), self.w_bkv(x),
                                self.w_az(x), self.w_bz(x),
                                self.b_a, self.b_b, m)               # (B, nb, c)
        K_comp = compress_single(self.w_ik(x), self.w_iz(x), self.b_i, m)  # (B, nb, d_idx)
        C = rms_norm(C, dim=-1)
        nb = C.shape[1]

        c_q = self.w_dq(x)                                           # (B, L, d_comp)
        q = rms_norm(self.w_uq(c_q).view(B, L, self.n_heads, self.head_dim), dim=-1)
        qh = q.transpose(1, 2)                                       # (B, L, H, c)

        # 索引器打分:I_{t,s} = Σ_h w^I_{t,h} · ReLU(q^I_{t,h} · K_comp_s)
        qI = self.w_iq(c_q).view(B, L, self.n_idx_heads, self.d_idx)
        relu = F.relu(torch.einsum("blhd,bjd->blhj", qI, K_comp))    # (B, L, nb, H^I)
        wI = self.w_iw(x)                                            # (B, L, H^I)
        I = torch.einsum("bljh,blh->blj", relu, wI)                  # (B, L, nb)
        s_idx = torch.arange(nb, device=x.device).view(1, 1, nb)
        t_block = (torch.arange(L, device=x.device) // m).view(1, L, 1)
        I = I.masked_fill(~(s_idx < t_block), float("-inf"))         # 只选过去的块
        k_eff = min(self.top_k, nb)
        sel = I.topk(k_eff, dim=-1).indices                          # (B, L, k)

        # 核心注意力:选中的压缩 KV + 滑动窗口 KV,统一 softmax(MQA)
        C5 = C[:, None, None, :, :].expand(B, L, self.n_heads, nb, self.head_dim)
        Csel = C5.gather(3, sel[:, :, None, :, None].expand(
            B, L, self.n_heads, k_eff, self.head_dim))               # (B, L, H, k, c)
        kw = self.w_kw(x)[:, -self.n_window:]                        # (B, w, c)
        vw = self.w_vw(x)[:, -self.n_window:]
        kw5 = kw[:, None, None].expand(B, L, self.n_heads, self.n_window, self.head_dim)
        vw5 = vw[:, None, None].expand(B, L, self.n_heads, self.n_window, self.head_dim)
        k_cat = torch.cat([Csel, kw5], dim=3)                        # (B, L, H, k+w, c)
        v_cat = torch.cat([Csel, vw5], dim=3)
        scores = torch.einsum("blhd,bljhd->blhj", qh, k_cat) / math.sqrt(self.head_dim)

        # 因果掩码:压缩部分要求所选块 < t//m;窗口部分要求 (L-w)+j < t
        w_causal = (torch.arange(self.n_window, device=x.device) + (L - self.n_window)) < \
            torch.arange(L, device=x.device)[:, None]                # (L, w)
        full_causal = torch.ones(B, L, k_eff + self.n_window, dtype=torch.bool, device=x.device)
        full_causal[:, :, :k_eff] = (sel < t_block).expand(B, L, k_eff)
        full_causal[:, :, -self.n_window:] = w_causal[None].expand(B, L, self.n_window)
        scores = scores.masked_fill(~full_causal[:, None], float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        o = torch.einsum("blhj,bljhd->blhd", attn, v_cat)            # (B, L, H, c)

        # 分组输出:每组先降维再拼接
        o = o.reshape(B, L, self.n_groups, -1)
        o = self.w_g(o)                                              # (B, L, g, d_g)
        return self.w_o(o.reshape(B, L, -1))


class HCAttention(nn.Module):
    """HCA 教学简化:单流重度压缩 + 全量 MQA + 分组输出,无索引器、无稀疏。"""

    def __init__(self, d_model, n_heads, head_dim, m=32, n_window=8,
                 d_comp=32, n_groups=2, d_g=32):
        super().__init__()
        self.m, self.n_window = m, n_window
        self.n_heads, self.head_dim = n_heads, head_dim
        self.n_groups, self.d_g = n_groups, d_g

        self.w_kv = nn.Linear(d_model, head_dim, bias=False)
        self.w_z = nn.Linear(d_model, head_dim, bias=False)
        self.b = nn.Parameter(torch.randn(m, head_dim) * 0.1)
        self.w_dq = nn.Linear(d_model, d_comp, bias=False)
        self.w_uq = nn.Linear(d_comp, n_heads * head_dim, bias=False)
        self.w_kw = nn.Linear(d_model, head_dim, bias=False)
        self.w_vw = nn.Linear(d_model, head_dim, bias=False)
        self.w_g = nn.Linear(n_heads * head_dim // n_groups, d_g, bias=False)
        self.w_o = nn.Linear(d_g * n_groups, d_model, bias=False)

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape
        m = self.m
        C = rms_norm(compress_single(self.w_kv(x), self.w_z(x), self.b, m), dim=-1)
        nb = C.shape[1]
        c_q = self.w_dq(x)
        q = rms_norm(self.w_uq(c_q).view(B, L, self.n_heads, self.head_dim), dim=-1)
        qh = q.transpose(1, 2)

        C5 = C[:, None, None, :, :].expand(B, L, self.n_heads, nb, self.head_dim)
        kw = self.w_kw(x)[:, -self.n_window:]
        vw = self.w_vw(x)[:, -self.n_window:]
        kw5 = kw[:, None, None].expand(B, L, self.n_heads, self.n_window, self.head_dim)
        vw5 = vw[:, None, None].expand(B, L, self.n_heads, self.n_window, self.head_dim)
        k_cat = torch.cat([C5, kw5], dim=3)
        v_cat = torch.cat([C5, vw5], dim=3)
        scores = torch.einsum("blhd,bljhd->blhj", qh, k_cat) / math.sqrt(self.head_dim)

        w_causal = (torch.arange(self.n_window, device=x.device) + (L - self.n_window)) < \
            torch.arange(L, device=x.device)[:, None]
        full_causal = torch.ones(B, L, nb + self.n_window, dtype=torch.bool, device=x.device)
        s_idx = torch.arange(nb, device=x.device).view(1, 1, nb)
        t_block = (torch.arange(L, device=x.device) // m).view(1, L, 1)
        full_causal[:, :, :nb] = (s_idx < t_block).expand(B, L, nb)
        full_causal[:, :, -self.n_window:] = w_causal[None].expand(B, L, self.n_window)
        scores = scores.masked_fill(~full_causal[:, None], float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        o = torch.einsum("blhj,bljhd->blhd", attn, v_cat)

        o = o.reshape(B, L, self.n_groups, -1)
        return self.w_o(self.w_g(o).reshape(B, L, -1))


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 32, 64)
    csa = CSAAttention(64, n_heads=4, head_dim=16, m=8, top_k=8, n_window=8,
                       d_comp=32, n_idx_heads=4, d_idx=16, n_groups=2, d_g=32)
    hca = HCAttention(64, n_heads=4, head_dim=16, m=16, n_window=8,
                      d_comp=32, n_groups=2, d_g=32)
    for name, mod in [("CSA", csa), ("HCA", hca)]:
        y = mod(x)
        assert y.shape == x.shape
        print(name, "forward OK:", y.shape)
```

## 4. CSA vs HCA vs DSA

| 维度 | DSA | CSA | HCA |
|---|---|---|---|
| KV 粒度 | 每 token 一份(MLA 潜变量) | 每 $m$ token 压缩成 1 条 | 每 $m'$ token 压缩成 1 条($m' \gg m$) |
| 压缩方式 | 无 | 双流重叠、$1/m$ | 单流、$1/m'$ |
| 稀疏选择 | token 级 top-k | 压缩块级 top-k | 无(全部压缩块) |
| 核心注意力 | MQA(MLA) | MQA(KV 同向量) | MQA(KV 同向量) |
| 额外分支 | 窗口(可选) | 窗口 $n_{win}$ | 窗口 $n_{win}$ |
| 每 token KV cache | $d_c + d_h^R$ | ~$c/m$ + 窗口 | ~$c/m'$ + 窗口 |
| 代表模型 | V3.2 | V4-Flash/Pro(混合) | V4-Flash/Pro(混合) |

## 5. 参考

- DeepSeek-AI, *DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence*, arXiv:2606.19348(2.3 节公式 9-27)
- DeepSeek-V3.2(DSA)与 DeepSeek-V3.2-Exp(索引器细节)
