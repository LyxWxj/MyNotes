---
type: Note
status: Active
related_to:
  - "[[attention-evolution-overview]]"
  - "[[kda]]"
---

# 线性注意力:从 RWKV 到 Mamba 的常值状态化之路

softmax 注意力的问题在于 $O(n^2)$。线性注意力家族换个角度:**把"注意力权重"变成"固定大小的循环状态"**,训练时复杂度 $O(n)$、推理时状态 $O(1)$,从根本上摆脱 KV cache 随上下文增长的问题。本文覆盖三个代表性架构:RWKV(指数衰减 WKV)、Mamba(选择性 SSM)、Gated DeltaNet(作为 [[kda]] 的前身)。

## 1. 线性注意力的统一视角

### 1.1 Kernel 形式

用核函数 $\phi(\cdot)$ 近似 softmax 注意力,并利用矩阵结合律把求和变成状态累加:

$$o_t = \sum_{j \le t} \frac{\phi(q_t)^\top \phi(k_j)}{\sum_{j' \le t} \phi(q_t)^\top \phi(k_{j'})} v_j
\;\;\Longrightarrow\;\;
S_t = S_{t-1} + \phi(k_t) v_t^\top,\qquad o_t = \phi(q_t)^\top S_t$$

这就是 Linear Transformer / Performer 的思路:状态 $S \in \mathbb{R}^{d_k \times d_v}$ 大小与序列长度无关。

### 1.2 从"软注意力"到"门控状态机"

纯累加 $S_t = S_{t-1} + k_t v_t^\top$ 的问题是旧信息永不消失,而且新写入会覆盖/污染旧关联。于是演化出三条改进:

- **指数衰减**:$S_t = S_{t-1} \odot e^{-w} + k_t v_t^\top$(RWKV,逐通道遗忘);
- **选择性状态转移**:$h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t$(Mamba,输入决定遗忘与写入);
- **delta rule 擦除**:$S_t = \alpha_t(I - \beta_t k_t k_t^\top) S_{t-1} + \beta_t k_t v_t^\top$(Gated DeltaNet → [[kda]])。

## 2. RWKV:token 级指数衰减注意力

### 2.1 数学原理

RWKV 用逐通道衰减向量 $w \in \mathbb{R}^d$ 定义"类注意力"权重:第 $i$ 个历史 token 对当前 token 的权重随距离指数衰减 $e^{-(t-1-i)w}$,当前 token 自己额外有一个可学习 bonus $u$:

$$\mathrm{wkv}_t = \frac{\sum_{i=1}^{t-1} e^{-(t-1-i)w + k_i}\, v_i + e^{u + k_t}\, v_t}
{\sum_{i=1}^{t-1} e^{-(t-1-i)w + k_i} + e^{u + k_t}}$$

分子分母都可以写成 O(1) 递推($\odot$ 为逐元素乘):

$$a_t = a_{t-1} \odot e^{-w} + e^{k_t + u} \odot v_t,\qquad
b_t = b_{t-1} \odot e^{-w} + e^{k_t + u},\qquad
\mathrm{wkv}_t = a_t \oslash b_t$$

> 注:闭式里 $u$ 只加在当前 token;而递推形式把 $u$ 也带进历史项。两者只差一个逐通道常数偏移,等价于对 key 做重参数化 $k'_i = k_i + u$。官方 CUDA kernel 采用的就是下面的递推形式,教学代码与其一致。

![[rwkv-fig2.png|RWKV 论文 Figure 2:RWKV block 内的时间混合 / 通道混合元素]]

![[rwkv-fig8.png|RWKV 论文 Figure 8:time-mixing block 的 RNN 视角(逐 token 递推)]]

RWKV-4 的 Time Mixing 还会对输入做"时间移位"混合(time shift:$\tilde{x}_t = \mu x_t + (1-\mu) x_{t-1}$),并给 WKV 输出套一个 $\sigma(r_t)$ 门控:

$$r_t = W_r(\mu x_t + (1-\mu) x_{t-1}),\qquad
o_t = \sigma(r_t) \odot \mathrm{wkv}_t$$

RWKV-6 / Eagle 进一步把 $w$ 变成输入依赖的逐 token 值,接近下面 Mamba 的"选择性"。

### 2.2 PyTorch 教学实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class RWKVTimeMix(nn.Module):
    """RWKV-4 Time Mixing(教学简化):token 级指数衰减注意力 WKV。
    递推:num = num * exp(-w) + exp(k + u) * v;den 同理;wkv = num / den"""

    def __init__(self, d_model, shift=0.5):
        super().__init__()
        self.time_shift = shift
        self.w_r = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.w = nn.Parameter(torch.randn(d_model) * 0.1)   # 逐通道衰减(log 域)
        self.u = nn.Parameter(torch.randn(d_model) * 0.1)   # 当前 token bonus

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape
        # time shift:x_t 与 x_{t-1} 的线性混合
        x_shift = torch.cat([x[:, :1] * 0.0, x[:, :-1]], dim=1)
        xm = x * (1 - self.time_shift) + x_shift * self.time_shift
        r = torch.sigmoid(self.w_r(xm))
        k = self.w_k(xm)
        v = self.w_v(xm)

        decay = torch.exp(-torch.exp(self.w))          # exp(-w) ∈ (0,1),逐通道
        num = torch.zeros(B, D, device=x.device)
        den = torch.zeros(B, D, device=x.device)
        outs = []
        for t in range(L):
            cur = torch.exp(k[:, t] + self.u)          # 当前 token 的贡献
            num = num * decay + cur * v[:, t]
            den = den * decay + cur
            outs.append(num / (den + 1e-6))
        wkv = torch.stack(outs, dim=1)
        return self.w_o(r * wkv)


class RWKVChannelMix(nn.Module):
    """RWKV-4 Channel Mixing(FFN):ReLU² + 门控"""

    def __init__(self, d_model, d_ff, shift=0.5):
        super().__init__()
        self.time_shift = shift
        self.w_r = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_ff, bias=False)
        self.w_v = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor):
        x_shift = torch.cat([x[:, :1] * 0.0, x[:, :-1]], dim=1)
        xm = x * (1 - self.time_shift) + x_shift * self.time_shift
        r = torch.sigmoid(self.w_r(xm))
        k = self.w_k(xm)
        return r * self.w_v(torch.square(F.relu(k)))


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 16, 32)
    tm = RWKVTimeMix(32)
    y = tm(x)
    cm = RWKVChannelMix(32, 64)
    z = cm(x)
    assert y.shape == x.shape and z.shape == x.shape
    print("RWKV TimeMix OK:", y.shape, "| ChannelMix OK:", z.shape)
```

## 3. Mamba:选择性状态空间模型

### 3.1 数学原理

SSM 把序列建模为线性连续系统:

$$\dot{h}(t) = A h(t) + B x(t),\qquad y(t) = C h(t) + D x(t)$$

离散化(零阶保持,ZOH):

$$\bar{A} = e^{\Delta A},\qquad \bar{B} = (\Delta A)^{-1}\big(e^{\Delta A} - I\big)\Delta B \approx \Delta B$$

得到循环形式与卷积形式两种等价视图:

$$h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t,\qquad y_t = C_t h_t$$

$$y = x \circledast \bar{K},\qquad \bar{K} = \big(C\bar{B},\, C\bar{A}\bar{B},\, \dots,\, C\bar{A}^{L-1}\bar{B}\big)$$

训练用卷积/并行扫描,推理用循环。**选择性机制**(Mamba-1)让 $\Delta_t, B_t, C_t$ 都依赖输入 $x_t$,相当于"注意力权重"随内容变化:

$$y_t = \sum_{i \le t} C_t \bar{A}_t \cdots \bar{A}_{i+1} \bar{B}_i x_i
\;\;\longleftrightarrow\;\;
\sum_{i \le t} \mathrm{Softmax}_i\big(q_t^\top k_i\big) v_i$$

对应关系:$\bar{A}$ 是"衰减/遗忘",$\bar{B}$ 是"写入",$C$ 是"读取"。Mamba-2(SSD)进一步证明,当 $\bar{A}$ 取特殊结构时,SSM 等价于一种带掩码的线性注意力,从而可以直接套用注意力矩阵的分解技巧。

![[mamba-fig3.png|Mamba 论文 Figure 3:简化块设计(H3 + Gated MLP 组合成 Mamba 块)]]

### 3.2 PyTorch 教学实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSM(nn.Module):
    """Mamba-1 教学简化:单层选择性 SSM。
    A 取可学习的对角负矩阵;Δ/B/C 由输入生成(选择性);循环逐 token 计算。
    真实实现用 selective scan / chunked 并行,这里为了可读性用循环。"""

    def __init__(self, d_model, d_state=16, d_inner=None):
        super().__init__()
        self.d_inner = d_inner or 2 * d_model
        self.d_state = d_state
        self.log_a = nn.Parameter(torch.randn(d_state))          # A = -exp(log_a)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)   # x, z
        self.conv = nn.Conv1d(self.d_inner, self.d_inner, 4,
                              groups=self.d_inner, padding=3)             # 因果短卷积
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)  # Δ, B, C
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        x_conv = F.silu(self.conv(x_in.transpose(1, 2))[..., :L].transpose(1, 2))

        params = self.x_proj(x_conv)                             # (B, L, 2N+1)
        log_dt, B_s, C_s = params.split([1, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(log_dt)                                  # (B, L, 1)
        A = -torch.exp(self.log_a)                               # (N,)
        A_bar = torch.exp(dt * A.view(1, 1, -1))                 # (B, L, N)
        B_bar = dt * B_s                                         # (B, L, N)

        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device)
        outs = []
        for t in range(L):
            h = h * A_bar[:, t, None, :] \
                + torch.einsum("bd,bn->bdn", x_conv[:, t], B_bar[:, t])
            outs.append(torch.einsum("bdn,bn->bd", h, C_s[:, t]))
        y = torch.stack(outs, dim=1)
        return self.out_proj(y * F.silu(z))


if __name__ == "__main__":
    torch.manual_seed(0)
    m = SelectiveSSM(32, d_state=8, d_inner=32)
    x = torch.randn(2, 16, 32)
    y = m(x)
    assert y.shape == x.shape
    print("SelectiveSSM OK:", y.shape)
```

## 4. Gated DeltaNet:向联想记忆过渡

Gated DeltaNet 把 Mamba 的"矩阵状态转移"换成"delta rule 联想记忆 + 标量门控":

$$S_t = \alpha_t\big(I - \beta_t k_t k_t^\top\big) S_{t-1} + \beta_t k_t v_t^\top,\qquad o_t = S_t^\top q_t$$

其中 $\alpha_t$ 是标量遗忘门,$\beta_t$ 是标量写入门,$k_t$ 决定"擦除/写入的方向"。它是 [[kda]] 的直接前身 —— KDA 把 $\alpha_t$ 从标量升级为逐通道对角矩阵。

## 5. 三种线性架构对比

| 维度 | RWKV | Mamba | Gated DeltaNet |
|---|---|---|---|
| 状态 | 分子分母两组向量(~$4d$) | $d \times N$ 矩阵 | $d_k \times d_v$ 矩阵 |
| 遗忘 | 逐通道 $e^{-w}$ | 矩阵 $\bar{A}_t$(输入依赖) | 标量 $\alpha_t$ |
| 写入 | $k v^\top$(不可擦除) | $\bar{B}_t x_t$ | $\beta k v^\top$(先擦除) |
| 读取 | $q_t$(除法归一) | $C_t$ | $S_t^\top q_t$ |
| 位置信息 | time shift + 指数衰减 | 隐式(选择性) | 无(需配合 RoPE 变体) |
| 代表模型 | RWKV-4/5/6、Eagle | Mamba-1/2、Jamba | ——(被 KDA 取代) |

共同点:训练 $O(n)$、推理状态 $O(1)$、适合无限上下文;代价是**精确的"指哪打哪"检索能力弱于 softmax 注意力**,因此实践中常用混合架构(如 Jamba = Mamba + attention,Kimi Linear = KDA + MLA)。

## 6. 参考

- Peng et al., *RWKV: Reinventing RNNs for the Transformer Era*, arXiv:2305.13048
- Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, arXiv:2312.00752
- Dao & Gu, *Transformers are SSMs: Generalized Models and Efficient Algorithms* (Mamba-2), arXiv:2405.21060
- Yang et al., *Gated Delta Networks*, arXiv:2412.06464
