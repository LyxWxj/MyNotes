---
type: Note
status: Active
related_to:
  - "[[attention-evolution-overview]]"
  - "[[linear-attention-rwkv-mamba]]"
---

# KDA:Kimi Delta Attention(Kimi Linear)

KDA 是 Kimi Linear(月之暗面,arXiv:2510.26692)提出的线性注意力架构,属于"delta rule 联想记忆"家族:它用一个固定大小的关联记忆矩阵 $S_t$ 递推地记录历史,训练复杂度 $O(n)$、推理状态 $O(1)$。相比前作 Gated DeltaNet,KDA 的关键升级是把**标量遗忘门换成逐通道对角衰减**,让记忆的每个通道独立决定遗忘速率。

## 1. 背景与动机

- 线性注意力(如 [[linear-attention-rwkv-mamba]])用 $\phi(k_t) v_t^\top$ 的外积累加记忆,但简单累加会被无关信息污染,无法"覆盖"已学到的关联。
- Delta rule 记忆:写入新关联 $k_t v_t^\top$ 前,先沿 $k_t$ 方向擦除旧状态 $(I - k_t k_t^\top) S_{t-1}$,实现联想记忆(Hecht-Nielsen 的 Oja/Delta rule,Linear Transformer 的 fast weight 视角)。
- Gated DeltaNet(GDN)给 delta rule 加上标量遗忘门 $\alpha_t$:

$$S_t = \alpha_t \big(I - \beta_t k_t k_t^\top\big) S_{t-1} + \beta_t k_t v_t^\top$$

- KDA 的洞察:标量 $\alpha_t$ 让所有通道以相同速率遗忘,太粗糙。改为**对角矩阵** $\mathrm{Diag}(\alpha_t)$,$\alpha_t \in [0,1]^{d_k}$ 逐通道可学习,记忆的"选择性遗忘"能力大幅提升。

## 2. 数学原理

### 2.1 核心递推(论文公式 1)

$$S_t = \big(I - \beta_t k_t k_t^\top\big)\, \mathrm{Diag}(\alpha_t)\, S_{t-1} + \beta_t k_t v_t^\top \in \mathbb{R}^{d_k \times d_v},\qquad
o_t = S_t^\top q_t$$

其中 $S$ 是关联记忆状态,$q_t, k_t \in \mathbb{R}^{d_k}$(L2 归一化),$v_t \in \mathbb{R}^{d_v}$。直观分解:

1. **遗忘**:$\mathrm{Diag}(\alpha_t) S_{t-1}$ —— 每个记忆通道按自己的速率 $\alpha_{t,i}$ 衰减;
2. **擦除**:$-\beta_t k_t k_t^\top (\mathrm{Diag}(\alpha_t) S_{t-1})$ —— 把状态中"沿 $k_t$ 方向"的分量删掉,为新写入腾空间;
3. **写入**:$+\beta_t k_t v_t^\top$ —— 存入当前关联;$\beta_t \in [0,1]$ 是标量写入/擦除门控。

### 2.2 神经参数化(论文公式 5-8)

$$q_t = \mathrm{L2Norm}\big(\mathrm{Swish}(\mathrm{ShortConv}(W_q x_t))\big),\qquad
k_t = \mathrm{L2Norm}\big(\mathrm{Swish}(\mathrm{ShortConv}(W_k x_t))\big)$$

$$v_t = \mathrm{Swish}(\mathrm{ShortConv}(W_v x_t))$$

$$\alpha_t = f\big(W_\alpha^{\uparrow} W_\alpha^{\downarrow} x_t\big) \in [0,1]^{d_k},\qquad
\beta_t = \sigma\big(W_\beta x_t\big) \in [0,1]$$

要点:L2Norm 保证特征值稳定;短卷积(short conv)把相邻 token 信息注入 q/k/v;$\alpha$ 用低秩投影($W_\alpha^{\downarrow}, W_\alpha^{\uparrow}$,秩为头维度)+ 衰减函数 $f$ 映射到 $[0,1]$;$\beta$ 用 sigmoid。

### 2.3 输出门控

$$o_t = W_o\Big(\sigma\big(W_g^{\uparrow} W_g^{\downarrow} x_t\big) \odot \mathrm{RMSNorm}\big(\mathrm{KDA}(q_t, k_t, v_t, \alpha_t, \beta_t)\big)\Big)$$

### 2.4 因果顺序与并行化

递推是 **write-then-read**:$S_t$ 先吸收当前 token 的 $k_t, v_t$,再被 $q_t$ 读取。$q_t, k_t, v_t$ 都只依赖 $x_t$ 本身,因此不破坏因果。推理内核常在 log 域实现(如 llama.cpp 的注释形式 $h_t = e^{g_t} h_{t-1} + k_t^\top \beta_t(v_t - h_{t-1}^\top k_t)$,擦除作用在旧状态上,与论文顺序略有实现差异)。

训练时用 **chunkwise 并行**:把 rank-1 变换打包成 WY 表示(论文公式 2-4),块内并行、块间递推,避免逐 token 串行:

$$P_r[t] = \mathrm{Diag}(\gamma^r) - \sum_i \mathrm{Diag}\big(\gamma^{i \to r}\big) k^i w^{i\top},\qquad
H_r[t] = \sum_i \mathrm{Diag}\big(\gamma^{i \to r}\big) k^i u^{i\top}$$

其中 $\gamma^{i \to r}$ 是从位置 $i$ 到 $r$ 的累积对角衰减。教学代码用最简单的逐 token 循环,便于理解;生产实现用 selective scan / chunked kernel。

### 2.5 混合部署与收益

Kimi Linear 在层间按 **KDA:MLA = 3:1** 交替(KDA 层无位置编码),$d_k = d_v = 128$。相比纯 MLA:KV cache 最高减少 **75%**,解码吞吐最高 **6 倍**;同时保留 MLA 层的精确注意力以保证质量。

## 3. PyTorch 教学实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ShortConv(nn.Module):
    """因果短卷积(depthwise):把相邻 token 信息注入 q/k/v"""

    def __init__(self, dim, kernel_size=4):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size, groups=dim, padding=kernel_size - 1)

    def forward(self, x):                      # (B, L, D)
        B, L, D = x.shape
        y = self.conv(x.transpose(1, 2))       # (B, D, L + k - 1)
        return y[..., :L].transpose(1, 2)      # 截断保持因果


class KDAHead(nn.Module):
    """KDA 单头(教学简化,逐 token 循环)。
    核心:S_t = (I - β k kᵀ) Diag(α) S_{t-1} + β k vᵀ;o_t = S_tᵀ q_t
    q/k 已 L2 归一化;α ∈ [0,1]^{d_k};β ∈ [0,1]。
    因果性:k_t, v_t, q_t 均来自同一 x_t,读取更新后的状态不破坏因果。"""

    def __init__(self, d_model, d_k=64, d_v=64, conv_kernel=4):
        super().__init__()
        self.d_k, self.d_v = d_k, d_v
        self.w_q = nn.Linear(d_model, d_k, bias=False)
        self.w_k = nn.Linear(d_model, d_k, bias=False)
        self.w_v = nn.Linear(d_model, d_v, bias=False)
        self.short_conv = ShortConv(d_k, conv_kernel)
        # 逐通道衰减:低秩投影 + sigmoid 限定 [0,1]
        self.w_alpha_down = nn.Linear(d_model, d_k, bias=False)
        self.w_alpha_up = nn.Linear(d_k, d_k, bias=False)
        self.w_beta = nn.Linear(d_model, 1, bias=False)     # 标量写入门控

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape
        q = F.normalize(self.short_conv(F.silu(self.w_q(x))), dim=-1)
        k = F.normalize(self.short_conv(F.silu(self.w_k(x))), dim=-1)
        v = F.silu(self.w_v(x))
        alpha = torch.sigmoid(self.w_alpha_up(F.silu(self.w_alpha_down(x))))  # (B, L, d_k)
        beta = torch.sigmoid(self.w_beta(x))                                  # (B, L, 1)

        state = torch.zeros(B, self.d_k, self.d_v, device=x.device)
        outs = []
        for t in range(L):
            # 1) 逐通道遗忘:Diag(α_t) S_{t-1}
            state = state * alpha[:, t, :, None]                     # (B, d_k, d_v)
            # 2) 擦除:沿 k_t 方向删除状态分量
            state = state - beta[:, t, :, None] * (
                k[:, t, :, None] @ (k[:, t, None, :] @ state))
            # 3) 写入:β_t k_t v_tᵀ
            state = state + beta[:, t, :, None] * (
                k[:, t, :, None] @ v[:, t, None, :])
            # 4) 读取:o_t = S_tᵀ q_t
            outs.append(torch.einsum("bdv,bd->bv", state, q[:, t]))  # (B, d_v)
        return torch.stack(outs, dim=1)                              # (B, L, d_v)


class KDAAttention(nn.Module):
    """多头 KDA + 输出门控(教学版)。
    o_t = W_o(σ(W_g↑ W_g↓ x_t) ⊙ RMSNorm(多头 KDA 输出))"""

    def __init__(self, d_model, n_heads, d_k=64, d_v=64, conv_kernel=4):
        super().__init__()
        self.heads = nn.ModuleList(
            [KDAHead(d_model, d_k, d_v, conv_kernel) for _ in range(n_heads)])
        self.w_g_down = nn.Linear(d_model, d_model, bias=False)
        self.w_g_up = nn.Linear(d_model, n_heads * d_v, bias=False)
        self.w_o = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, x: torch.Tensor):
        o = torch.cat([h(x) for h in self.heads], dim=-1)          # (B, L, H*d_v)
        o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + 1e-6)  # RMSNorm
        g = torch.sigmoid(self.w_g_up(F.silu(self.w_g_down(x))))
        return self.w_o(o * g)


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 16, 64)
    head = KDAHead(64, d_k=32, d_v=32)
    out = head(x)
    assert out.shape == (2, 16, 32), out.shape
    attn = KDAAttention(64, n_heads=4, d_k=32, d_v=32)
    y = attn(x)
    assert y.shape == x.shape, y.shape
    print("KDAHead OK:", out.shape, "| KDAAttention OK:", y.shape)
```

### 与 Gated DeltaNet 的退化关系

把 $\mathrm{Diag}(\alpha_t)$ 换成标量 $\alpha_t I$、去掉短卷积与门控,就退化为 Gated DeltaNet;再令 $\alpha_t = 1$、$\beta_t = 1$,进一步退化为经典 delta rule 线性注意力。教学代码中只需把 `state * alpha[:, t, :, None]` 改成 `state * alpha[:, t, None, None]` 即可观察退化行为。

## 4. 线性注意力家族对比

| 架构 | 更新规则 | 遗忘机制 | 状态大小 | 记忆能力 |
|---|---|---|---|---|
| 线性注意力(Katharopoulos) | $S \mathrel{+}= k v^\top$ | 无(无限累加) | $d_k \times d_v$ | 弱(被旧信息污染) |
| RWKV | $S \mathrel{*}= e^{-w} + k v^\top$ | 逐通道指数衰减 | ~$4d$ | 中 |
| Mamba(SSM) | $h = \bar{A} h + \bar{B} x$ | 状态矩阵 $\bar{A}$ | $d \times N$ | 中 |
| Gated DeltaNet | $S = \alpha(I - \beta kk^\top) S + \beta k v^\top$ | 标量 $\alpha$ | $d_k \times d_v$ | 强(可擦除) |
| **KDA** | $S = (I - \beta kk^\top)\mathrm{Diag}(\alpha) S + \beta k v^\top$ | **逐通道** $\alpha$ | $d_k \times d_v$ | 最强(细粒度遗忘 + 擦除) |

## 5. 参考

- Kimi Team, *Kimi Linear: An Expressive, Efficient Attention Architecture*, arXiv:2510.26692
- Yang et al., *Gated Delta Networks: Improving Mamba2 with Delta Rule*, arXiv:2412.06464
- 开源参考实现:hwilner/kimi-delta-attention(llama.cpp 内核的 Python 对照)
