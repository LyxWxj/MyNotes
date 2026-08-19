---
type: Note
status: Active
related_to:
  - "[[attention-evolution-overview]]"
  - "[[mha-gqa]]"
  - "[[nsa-dsa]]"
---

# MLA:多头潜在注意力(Multi-head Latent Attention)

MLA 是 DeepSeek-V2 提出的注意力架构,核心思想:**把 key 和 value 联合压缩进一个低秩潜在向量**,让推理 KV cache 与训练激活内存同时大幅缩小。DeepSeek-V2/V3、Kimi K2 等模型都采用了 MLA。

## 1. 背景与动机

- GQA 把 KV 头数从 $n_h$ 降到 $n_{kv}$,但仍是"每 token 缓存 $2 n_{kv} d_h$ 个原始值"。
- MLA 更进一步:假设 KV 信息集中在低维子空间,先用 $W^{DKV} \in \mathbb{R}^{d \times d_c}$ 把 $h_t \in \mathbb{R}^d$ 压成 $c_t^{KV} \in \mathbb{R}^{d_c}$($d_c \ll n_h d_h$),推理时**只缓存 $c_t^{KV}$**。
- 训练时还有第二个红利:query 也走低秩($c_t^Q = h_t W^{DQ}$),注意力的中间激活(每个头一份的 $q, k, v$)由每 token 一份的低维潜变量生成,激活内存显著下降。
- 位置信息单独处理:解耦 RoPE 维度 $d_h^R$,避免低秩压缩破坏 RoPE 的相对位置性质。

## 2. 数学原理(DeepSeek-V2,公式 9-15)

### 2.1 KV 低秩联合压缩

对第 $t$ 个 token 的隐藏状态 $h_t \in \mathbb{R}^d$:

$$c_t^{KV} = W^{DKV} h_t \in \mathbb{R}^{d_c}$$

$$k_t^{C} = W^{UK} c_t^{KV},\qquad v_t^{C} = W^{UV} c_t^{KV}$$

其中 $W^{DKV} \in \mathbb{R}^{d \times d_c}$,$W^{UK}, W^{UV} \in \mathbb{R}^{d_c \times n_h d_h}$。注意 $k_t^C, v_t^C$ 是所有头**共享**的(推理时是单份 KV,即 MQA 形态)。

![[mla-fig2.png|DeepSeek-V2 论文 Figure 2:MLA 架构(低秩 KV 压缩 + 解耦 RoPE,推理只缓存 c^KV 与 k^R)]]

### 2.2 query 低秩

$$c_t^Q = W^{DQ} h_t \in \mathbb{R}^{d_c},\qquad q_t^C = W^{UQ} c_t^Q$$

其中 $W^{UQ} \in \mathbb{R}^{d_c \times n_h d_h}$,$q_t^C$ 按头拆成  $\{q_{t,1}^C, \dots, q_{t,n_h}^C\}$,每头 $d_h$ 维。

### 2.3 解耦 RoPE

低秩压缩会破坏 RoPE 的旋转结构,因此 MLA 额外准备一份低维的解耦 RoPE 分量(每头 $d_h^R$ 维,DeepSeek-V2 中 $d_h^R = 64$):

$$\big[q_{t,1}^{R}; \dots; q_{t,n_h}^{R}\big] = \mathrm{RoPE}\big(W^{QR} c_t^Q\big),\qquad
k_t^{R} = \mathrm{RoPE}\big(W^{KR} h_t\big) \in \mathbb{R}^{d_h^R}$$

最终 query/key 是"内容分量 + 位置分量"的拼接:

$$q_{t,i} = \big[q_{t,i}^{C};\, q_{t,i}^{R}\big] \in \mathbb{R}^{d_h + d_h^R},\qquad
k_t = \big[k_t^{C};\, k_t^{R}\big] \in \mathbb{R}^{d_h + d_h^R}$$

### 2.4 注意力输出

$$o_{t,i} = \sum_{j=1}^{t} \mathrm{Softmax}_j\!\left(\frac{q_{t,i}^\top k_j}{\sqrt{d_h + d_h^R}}\right) v_j^{C},\qquad
o_t = W^O \big[o_{t,1}; \dots; o_{t,n_h}\big]$$

### 2.5 推理缓存与权重吸收

推理时每层每 token 只需缓存:

$$\mathrm{cache}_t = \big[\, c_t^{KV} \in \mathbb{R}^{d_c};\; k_t^{R} \in \mathbb{R}^{d_h^R} \,\big],\qquad |\mathrm{cache}_t| = d_c + d_h^R$$

DeepSeek-V2(307B)中 $d_c = 512$、$d_h^R = 64$,即每 token 每层 576 个值;而相同规模 MHA 是 $2 \times 5120 = 10240$,压缩约 **18 倍**。

两个权重吸收技巧(数学恒等变形,零损失):

1. **$W^{UK}$ 吸收进 $W^{UQ}$**:$q_{t,i}^\top k_j^{C} = q_{t,i}^\top W^{UK} c_j^{KV} = \big(W^{UK\top} q_{t,i}\big)^\top c_j^{KV}$,推理时先用 $W^{UK\top}$ 处理 query 头,打分只接触缓存的 $c_j^{KV}$,不必恢复完整 key。
2. **$W^{UV}$ 吸收进 $W^O$**:$o_t = W^O \big[v_1^{C} \cdots\big] = W^{O'}\big[c_1^{KV} \cdots\big]$,$W^{UV}$ 与 $W^O$ 合并成 $W^{O'} \in \mathbb{R}^{d \times d_c n_h}$。

## 3. PyTorch 教学实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class MultiHeadLatentAttention(nn.Module):
    """MLA 教学简化:低秩 KV 联合压缩 + query 低秩 + 解耦 RoPE。
    真实实现会做权重吸收(见 2.5):吸收是精确恒等变形,训练与推理均可用
    (训练省激活内存,推理免恢复完整 key);这里为清晰起见直接展开计算。"""

    def __init__(self, d_model, n_heads, d_comp=64, d_rope=16, head_dim=32):
        super().__init__()
        self.d_model, self.n_heads = d_model, n_heads
        self.d_comp, self.d_rope, self.head_dim = d_comp, d_rope, head_dim

        # KV 低秩联合压缩
        self.w_dkv = nn.Linear(d_model, d_comp, bias=False)      # h_t -> c_t^KV
        self.w_uk = nn.Linear(d_comp, head_dim, bias=False)      # c_t^KV -> k_t^C (MQA 单头)
        self.w_uv = nn.Linear(d_comp, head_dim, bias=False)      # c_t^KV -> v_t^C
        # query 低秩
        self.w_dq = nn.Linear(d_model, d_comp, bias=False)       # h_t -> c_t^Q
        self.w_uq = nn.Linear(d_comp, n_heads * head_dim, bias=False)
        # 解耦 RoPE
        self.w_qr = nn.Linear(d_comp, n_heads * d_rope, bias=False)
        self.w_kr = nn.Linear(d_model, d_rope, bias=False)       # 单 KV 头
        self.w_o = nn.Linear(n_heads * head_dim, d_model, bias=False)

    def project_kv(self, x: torch.Tensor, cos=None, sin=None):
        """返回 (c_t^KV, k_t^R):推理时只需缓存这两个量,合计 d_comp + d_rope"""
        c_kv = self.w_dkv(x)                                     # (B, L, d_comp)
        k_r = self.w_kr(x)                                       # (B, L, d_rope)
        if cos is not None:
            k_r = k_r * cos + rotate_half(k_r) * sin
        return c_kv, k_r

    def forward(self, x: torch.Tensor, cos=None, sin=None):
        """x: (B, L, d);cos/sin: (L, d_rope) 或 None(与 mha-gqa 的缓存形状一致)"""
        B, L, _ = x.shape
        c_kv, k_r = self.project_kv(x, cos, sin)

        # key/value:由共享潜变量生成,天然是 MQA 形态
        k_c = self.w_uk(c_kv)   # (B, L, d_h)
        v_c = self.w_uv(c_kv)   # (B, L, d_h)

        # query:低秩 + 解耦 RoPE
        c_q = self.w_dq(x)      # (B, L, d_comp)
        q_c = self.w_uq(c_q).view(B, L, self.n_heads, self.head_dim)
        q_r = self.w_qr(c_q).view(B, L, self.n_heads, self.d_rope)
        if cos is not None:
            q_r = q_r * cos + rotate_half(q_r) * sin

        q = torch.cat([q_c, q_r], dim=-1).transpose(1, 2)        # (B, H, L, d_h+d_rope)
        k = torch.cat([k_c, k_r], dim=-1)                        # (B, L, d_h+d_rope)
        v = v_c

        # 广播到所有头(MQA),缩放因子用拼接后的总维度
        k = k.unsqueeze(1).expand(B, self.n_heads, L, -1)
        v = v.unsqueeze(1).expand(B, self.n_heads, L, -1)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.transpose(1, 2).reshape(B, L, -1)
        return self.w_o(o)


if __name__ == "__main__":
    torch.manual_seed(0)
    L, d_model, n_heads = 32, 128, 8
    mla = MultiHeadLatentAttention(d_model, n_heads, d_comp=64, d_rope=16, head_dim=32)
    x = torch.randn(2, L, d_model)
    y = mla(x)                                          # 不启用 RoPE 也能跑
    assert y.shape == (2, L, d_model)
    c_kv, k_r = mla.project_kv(x)
    print("output:", y.shape)
    print("KV cache per token per layer:", c_kv.shape[-1] + k_r.shape[-1])
```

### 权重吸收的等价性验证

```python
torch.manual_seed(0)
q_head = torch.randn(8, 32)     # 单个 query 头 (B', d_h)
c = torch.randn(8, 64)          # 缓存的 c^KV (B', d_comp)
w_uk = torch.randn(64, 32)      # W^UK

scores_naive = q_head @ w_uk.T @ c.T                       # qᵀ(W^UK c) 直接展开
scores_abs = (w_uk.T @ q_head.T).T @ c.T                   # (W^UKᵀ q)ᵀ c 吸收后
assert torch.allclose(scores_naive, scores_abs, atol=1e-5)
print("absorption equivalence: OK")
```

## 4. 要点与对比

- **MLA vs GQA**:GQA 缓存的是 KV 头投影后的原始向量($2 n_{kv} d_h$);MLA 缓存的是低维潜变量($d_c + d_h^R$),且 $d_c$ 可以做到比 $n_{kv} d_h$ 更小。
- **MLA 的 MQA 形态**:压缩后的 $k^C, v^C$ 是所有 query 头共享的单份,这与 [[nsa-dsa]] 中 DSA 的核心注意力模式一致(DeepSeek-V3.2 说明 MLA 有 MHA/MQA 两种模式,DSA 使用 MQA 模式):

![[dsa-fig7.png|DeepSeek-V3.2 论文 Figure 7:MLA 的 MHA 与 MQA 两种模式]]
- **解耦 RoPE 的必要性**:若把 RoPE 施加在低秩向量上,旋转维数被压缩,位置分辨率丢失;独立 $d_h^R$ 维让位置编码保持完整旋转结构。
- **训练激活内存**:query 低秩意味着注意力打分可以先用低维潜变量算(配合吸收技巧),训练时不必展开每头的完整 K/V 激活。吸收在训练时同样成立(前向、反向都是恒等变形),并非推理专属;代码里不写吸收只是为了对照 §2 公式、便于理解。
- **局限**:MLA 仍是 $O(n^2)$ 的精确注意力,计算量没有下降;长上下文的计算瓶颈由 [[nsa-dsa]] 的稀疏化和 [[csa-hca]] 的压缩来解决。

## 5. 参考

- DeepSeek-AI, *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*, arXiv:2405.04434(公式 9-15)
- DeepSeek-V3 技术报告(MLA 的工程化细节)与 DeepSeek-V3.2-Exp 报告(MLA 的 MQA 模式)
