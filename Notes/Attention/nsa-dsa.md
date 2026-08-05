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

两者都是**可训练的稀疏注意力**(不是靠手工规则),且训练与推理使用同一套稀疏结构。下文所有公式、超参与训练设置均逐条对照原始论文/报告,不再做教学简化。

## 1. NSA:原生稀疏注意力

### 1.1 动机与设计原则

人的阅读不是均匀扫描:重要段落精读、次要段落略读。NSA 把每个 query 的注意力分成三条通路:

1. **压缩分支**:把若干 token 压缩成一个粗粒度 KV 条目,用少量块覆盖全局信息(花小成本扫一遍)。
2. **选择分支**:用粗粒度打分选出最相关的 top-n 块,在块内做细粒度注意力(精读)。
3. **滑动窗口分支**:最近 $w$ 个 token 全量注意力(保底局部依赖)。

两条硬件层面的约束直接塑造了 NSA 的设计(论文 §3.1、§3.4):

- **算术强度不对称**:训练与 prefill 阶段是计算受限(compute-bound),decoding 是访存受限(memory-bound,每步要重读整个 KV cache)。所以优化目标分别是"减少训练/prefill 的计算量"和"减少 decode 的 KV cache 访存量"。
- **GQA 组的 KV 共享**:若每个头独立选择自己的 KV 块(如 Quest),在 GQA/MQA 下同组 query 头会选择不同的块,导致 KV cache 访存量退化为各组选择的并集,内存稀疏性被破坏。因此 NSA 的选择必须**在 GQA 组内保持一致**——这是后面式 10 的来源。

### 1.2 总体框架(论文式 3-6)

对 query $\mathbf{q}_t$,NSA 用两个映射函数把历史 KV 重映射成信息密度更高、规模更小的表示:

$$\tilde{K}_t = f_K(\mathbf{q}_t, \mathbf{k}_{:t}, \mathbf{v}_{:t}),\qquad
\tilde{V}_t = f_V(\mathbf{q}_t, \mathbf{k}_{:t}, \mathbf{v}_{:t}),\qquad
\mathbf{o}^*_t = \mathrm{Attn}\!\left(\mathbf{q}_t, \tilde{K}_t, \tilde{V}_t\right)$$

共有三种映射策略 $\mathcal{C} = \{\text{cmp}, \text{slc}, \text{win}\}$(压缩、选择、滑动窗口),最终输出是三条通路输出的**加权和**:

$$\mathbf{o}^*_t = \sum_{c \in \mathcal{C}} g_t^{c} \cdot \mathrm{Attn}\!\left(\mathbf{q}_t, \tilde{K}_t^{c}, \tilde{V}_t^{c}\right)$$

其中 $g_t^{c} \in [0,1]$ 是门控分数,**由输入特征经 MLP + sigmoid 得到**(论文原文:"derived from input features via an MLP and sigmoid activation";不是 softmax 归一化,三路门控彼此独立)。设 $N_t = \sum_{c \in \mathcal{C}} \mathrm{size}[\tilde{K}_t^c]$ 为每个 query 实际接触的 KV 条目数,NSA 通过保证 $N_t \ll t$ 维持高稀疏率。注意式中的 $\mathrm{Attn}$ 与标准定义一致(论文式 2):$\mathrm{Attn}(\mathbf{q}_t, \mathbf{k}_{:t}, \mathbf{v}_{:t}) = \sum_{i=1}^{t} \frac{\alpha_{t,i}\mathbf{v}_i}{\sum_j \alpha_{t,j}}$,其中 $\alpha_{t,i} = e^{\mathbf{q}_t^\top \mathbf{k}_i / \sqrt{d_k}}$。

### 1.3 压缩分支(论文式 7)

按块长 $l$、滑动步长 $d$ 把 key 序列切成(可重叠的)块,每块经可学习压缩网络 $\varphi$ 压成**一个**粗粒度 key;value 同理:

$$\tilde{K}_t^{\text{cmp}} = f_K^{\text{cmp}}(\mathbf{k}_{:t}) = \left\{\, \varphi\!\left(\mathbf{k}_{id+1 : id+l}\right) \;\middle|\; 0 \le i \le \left\lfloor \tfrac{t-l}{d} \right\rfloor \,\right\}$$

细节:

- $\varphi$ 是**可学习 MLP,带块内位置编码**(intra-block position encoding),把块内 $l$ 个向量映射到单个压缩 key。论文未固定 $\varphi$ 的具体结构。
- **步长 $d < l$**:压缩块相互重叠,论文明确说明采用 $d < l$ 是为了"缓解信息碎片化"(mitigate information fragmentation)。
- 论文实验配置:$l = 32$,$d = 16$。
- 因果性体现在索引上界:块 $i$ 覆盖 $[id+1, id+l]$,query $t$ 只能看到满足 $id + l \le t$ 的块,即 $i \le \lfloor (t-l)/d \rfloor$。
- 压缩分支的注意力输出 $\mathrm{Attn}(\mathbf{q}_t, \tilde K_t^{\text{cmp}}, \tilde V_t^{\text{cmp}})$ 是对**粗粒度 value** 的加权和,维度与选择/窗口分支的输出保持一致(论文实验中 value 维 $d_v = 128$)。

### 1.4 选择分支(论文式 8-12)

**重要性分数直接复用压缩分支的注意力分数**,不引入额外打分器(这是 NSA 与 Quest/InfLLM 式"另设重要性网络 + 辅助损失"的关键区别):

$$\mathbf{p}_t^{\text{cmp}} = \mathrm{Softmax}\!\left(\mathbf{q}_t^\top \tilde{K}_t^{\text{cmp}}\right)$$

其中 $\mathbf{p}_t^{\text{cmp}} \in \mathbb{R}^{\lfloor (t-l)/d \rfloor + 1}$。

**块方案一致时的特例**:若压缩块与选择块完全同构($l' = l = d$),则 $\mathbf{p}_t^{\text{slc}} = \mathbf{p}_t^{\text{cmp}}$ 直接作为选择块重要性。

**块方案不同时**(论文配置即如此:$l = 32$、$d = 16$、$l' = 64$):给定 $l \le l'$、$d \mid l$、$d \mid l'$,把与选择块 $j$ 空间对齐/邻近的压缩块分数聚合:

$$\mathbf{p}_t^{\text{slc}}[j] = \sum_{m=0}^{\frac{l'}{d}-1} \sum_{n=0}^{\frac{l}{d}-1} \mathbf{p}_t^{\text{cmp}}\!\left[\tfrac{l'}{d}j - m - n\right]$$

(重叠的压缩块会被重复计入;该求和把选择块 $j$ 附近的压缩注意力分数累加为块级重要性。)

**GQA/MQA 组内一致选择**(论文式 10):同一 GQA 组内的 $H$ 个 query 头共享 KV cache,为保证解码时 KV 只加载一次,选择分数先跨组内头求和:

$$\mathbf{p}_t^{\text{slc}\prime} = \sum_{h=1}^{H} \mathbf{p}_t^{\text{slc},(h)}$$

**top-n 块选择**(论文式 11-12):

$$\mathcal{I}_t = \left\{\, i \;\middle|\; \mathrm{rank}\!\left(\mathbf{p}_t^{\text{slc}\prime}[i]\right) \le n \,\right\},\qquad
\tilde{K}_t^{\text{slc}} = \mathrm{Cat}\!\left[\left\{\, \mathbf{k}_{il'+1 : (i+1)l'} \;\middle|\; i \in \mathcal{I}_t \,\right\}\right]$$

其中 $\mathrm{rank}(\cdot)$ 为降序排名(rank = 1 为最高分),$\mathcal{I}_t$ 是选中的块索引集合,$\tilde K_t^{\text{slc}} \in \mathbb{R}^{d_k \times n l'}$,value 同理。选择分支的注意力只对选中块内的**原始 token** 进行(细粒度)。

论文实验配置:选择块大小 $l' = 64$,选择块数 $n = 16$(其中**固定激活第 1 个块和最近 2 个局部块**),滑动窗口 $w = 512$。

### 1.5 滑动窗口分支与门控融合(论文 §3.3.3)

窗口分支维护最近 $w$ 个 token:$\tilde K_t^{\text{win}} = \mathbf{k}_{t-w:t}$、$\tilde V_t^{\text{win}} = \mathbf{v}_{t-w:t}$。论文引入它的动机是:**局部模式收敛更快、容易主导学习**,若不做隔离,压缩与选择分支会被"捷径化"(shortcut),学不到长程结构;独立窗口分支把局部上下文单独处理,让另外两条通路专注各自的特征。

两个容易被教学版忽略的细节:

1. **三个分支使用相互独立的 key/value 投影**(论文原文:"we provide independent keys and values for three branches"),以最小代价防止分支间的捷径学习与梯度干扰。
2. 门控分数 $g_t^{c}$ 是 **MLP + sigmoid 的输出**(见 1.2),三路输出按式 5 逐元素加权求和,再经过最终的输出投影 $W^O$。

### 1.6 复杂度与解码成本

**每 query 的 KV 条目数**(论文式 6):

$$N_t = \underbrace{\left\lfloor \tfrac{t-l}{d} \right\rfloor + 1}_{\text{压缩}} + \underbrace{n \cdot l'}_{\text{选择}} + \underbrace{w}_{\text{窗口}}$$

代入论文配置($d=16, n=16, l'=64, w=512$):32k 序列上平均每 query ≈ $L/(2d) + nl' + w = 1024 + 1024 + 512 = 2560$ 个条目(LongBench 对比实验的稀疏预算正是取这个平均值);总计算量约 $\sum_t (t/d) \approx L^2/(2d) + L(nl'+w)$——注意压缩分支仍是二次项(常数 $1/2d$),"线性复杂度"的表述仅对"每 query 条目数"成立。

**解码每步的 KV 访存量**(论文 §5.2):每步最多加载 $\lfloor (s-l)/d \rfloor$ 个压缩 token + $n l'$ 个选中 token + $w$ 个窗口 token,其中 $s$ 为已缓存长度:

| 上下文长度 | 8k | 16k | 32k | 64k |
|---|---|---|---|---|
| Full Attention 访存(等价 token 数) | 8192 | 16384 | 32768 | 65536 |
| NSA 访存 | 2048 | 2560 | 3584 | 5632 |
| 期望加速比 | 4× | 6.4× | 9.1× | 11.6× |

训练(prefill)侧,Triton 实现的 NSA 相比 Triton FlashAttention-2,64k 上下文下前向最快 **9.0×**、反向 **6.0×**(论文 §5.1)。

### 1.7 硬件对齐的 kernel 设计(论文 §3.4)

- 压缩与滑动窗口的注意力**直接兼容 FlashAttention-2 类 kernel**;专门优化的是稀疏选择注意力。
- 关键设计:不按 FlashAttention 的"时间连续 query 块"加载,而是**按 GQA 组加载**:内循环中把同一组内所有头的 query $Q \in \mathbb{R}^{[h, d_k]}$ 与其共享的稀疏块索引 $\mathcal{I}_t$ 一起装入 SRAM(组中心数据加载)。
- **共享 KV 抓取**:内循环按 $\mathcal{I}_t$ 顺序加载连续 KV 块 $K \in \mathbb{R}^{[B_k, d_k]}$、$V \in \mathbb{R}^{[B_k, d_v]}$ 到 SRAM,kernel 块大小 $B_k$ 整除 $l'$,消除冗余 KV 搬运。
- **外循环上网格**:不同 query 块的内循环长度(∝ 选中块数 $n$)几乎相同,因此把 query/输出循环放到 Triton 的 grid 调度器上,简化调度并均衡 SM 负载。
- 块式连续访问最大化 Tensor Core 利用率(对齐合并访存),配合精细的循环调度消除冗余 KV 传输。

### 1.8 实验配置(论文 §4.1,用于理解"细节"的上下文)

- 主干:27B 总参 / 3B 激活,30 层,hidden 2560;GQA:4 组 × 16 头 = 64 头,每头 $d_q = d_k = 192$、$d_v = 128$;DeepSeekMoE:72 路由专家 + 2 共享专家,top-6,首层 MoE 换成 SwiGLU MLP。
- NSA 超参:$l = 32$、$d = 16$、$l' = 64$、$n = 16$、$w = 512$(见 1.3-1.5)。
- 预训练 270B tokens(8k 长度文本),随后在 32k 文本上用 YaRN 继续训练 + SFT 做长上下文适配。
- 推理对比基线(H2O、InfLLM、Quest、Exact-Top)统一稀疏预算 2560 tokens/query(32k 平均),并按 StreamLLM 惯例含前导 128 + 局部 512 tokens。
- 长上下文:64k needle-in-a-haystack 全位置完美检索;LongBench 平均 0.469,超 Full Attention(+0.032)与 Exact-Top(+0.046)。
- 推理能力:用 DeepSeek-R1 蒸馏的 10B tokens、32k 数学推理轨迹做 SFT,AIME24 上 8k/16k 生成上限分别 +0.075/+0.054(NSA-R vs Full Attention-R)。

### 1.9 参考实现(与论文公式逐条对应)

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """非交错 RoPE(MLA 与 DSA 索引器均用这种形式)"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


# ---------------------------------------------------------------- NSA


class BlockCompressor(nn.Module):
    """φ:把块内 l 个 key/value 向量压成 1 个粗粒度向量(式 7)。
    论文只限定"可学习 MLP + 块内位置编码",未固定具体结构;
    这里实现为:块内位置编码 + 两层 MLP + 平均池化。"""

    def __init__(self, d_in: int, block_len: int):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(block_len, d_in) * 0.02)
        self.mlp = nn.Sequential(nn.Linear(d_in, 4 * d_in), nn.SiLU(),
                                 nn.Linear(4 * d_in, d_in))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, nb, l, d_in) -> (B, H, nb, d_in)
        return self.mlp(x + self.pos).mean(dim=-2)


class NativeSparseAttention(nn.Module):
    """NSA(论文 §3),与公式逐条对应:

    - 压缩分支:块长 l、滑动步长 d < l(重叠块),φ 为可学习压缩网络(式 7)
    - 选择分支:重要性 = 压缩注意力分数按空间重叠求和(式 9)+ GQA 组内跨头求和
      (式 10),top-n 块(式 11-12);第 1 块与最近 2 块固定激活
    - 窗口分支:最近 w 个 token;三条通路 K/V 相互独立(§3.3.3)
    - 门控 = MLP + sigmoid(式 5,非 softmax)
    """

    def __init__(self, d_model: int, n_heads: int, d_k: int, d_v: int,
                 group_size: int = 1, block_len: int = 32, stride: int = 16,
                 sel_block_len: int = 64, n_sel_blocks: int = 16,
                 n_window: int = 512):
        super().__init__()
        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v
        self.group_size = group_size                     # GQA 组内 query 头数
        self.block_len, self.stride = block_len, stride
        self.sel_block_len, self.n_sel_blocks = sel_block_len, n_sel_blocks
        self.n_window = n_window
        assert stride <= block_len <= sel_block_len
        assert sel_block_len % stride == 0 and block_len % stride == 0
        assert n_heads % group_size == 0

        self.w_q = nn.Linear(d_model, n_heads * d_k, bias=False)
        # 三条通路各自的 K/V(论文 §3.3.3:独立投影,防捷径学习)
        self.w_sk = nn.Linear(d_model, n_heads * d_k, bias=False)   # 选择 K
        self.w_sv = nn.Linear(d_model, n_heads * d_v, bias=False)   # 选择 V
        self.w_wk = nn.Linear(d_model, n_heads * d_k, bias=False)   # 窗口 K
        self.w_wv = nn.Linear(d_model, n_heads * d_v, bias=False)   # 窗口 V
        self.w_ck = nn.Linear(d_model, n_heads * d_k, bias=False)   # 压缩 K
        self.w_cv = nn.Linear(d_model, n_heads * d_v, bias=False)   # 压缩 V
        self.compress_k = BlockCompressor(d_k, block_len)
        self.compress_v = BlockCompressor(d_v, block_len)

        self.gate = nn.Sequential(                       # 式 5:MLP + sigmoid
            nn.Linear(d_model, d_model), nn.SiLU(),
            nn.Linear(d_model, 3), nn.Sigmoid())
        self.w_o = nn.Linear(n_heads * d_v, d_model, bias=False)

    # ---- 压缩分支(式 7)----
    def _compress(self, x: torch.Tensor, proj: nn.Module,
                  compressor: BlockCompressor) -> torch.Tensor:
        B, L, _ = x.shape
        l, d = self.block_len, self.stride
        k = proj(x).view(B, L, self.n_heads, -1).transpose(1, 2)   # (B, H, L, d)
        nb = (L - l) // d + 1                          # 块 i 覆盖 [i*d, i*d+l)
        off = torch.arange(l, device=x.device)
        idx = torch.arange(nb, device=x.device)[:, None] * d + off  # (nb, l)
        return compressor(k[:, :, idx])                # (B, H, nb, d)

    def _attn_compressed(self, q: torch.Tensor, ck: torch.Tensor,
                         cv: torch.Tensor):
        """q: (B,H,L,d_k);ck/cv: (B,H,nb,d_k)/(B,H,nb,d_v)"""
        B, H, L, _ = q.shape
        nb = ck.shape[2]
        scores = torch.einsum("bhld,bhjd->bhlj", q, ck) / math.sqrt(self.d_k)
        i = torch.arange(nb, device=q.device)
        t = torch.arange(L, device=q.device)
        causal = (i[:, None] * self.stride + self.block_len) <= t[None, :]  # (nb, L)
        scores = scores.masked_fill(~causal.T[None, None], float("-inf"))
        p = torch.softmax(scores, dim=-1)              # (B, H, L, nb) = 式 8
        p = p.masked_fill(p != p, 0.0)                 # t < l 时无合法块,置零
        o = torch.einsum("bhlj,bhjd->bhld", p, cv)     # (B, H, L, d_v)
        return o, p

    # ---- 选择分支(式 9-12)----
    def _sel_importance(self, p_cmp: torch.Tensor) -> torch.Tensor:
        B, H, L, nb_c = p_cmp.shape
        l, d, l2 = self.block_len, self.stride, self.sel_block_len
        g = self.group_size
        p = p_cmp.view(B, H // g, g, L, nb_c).sum(dim=2)      # 式 10:组内头求和
        m = torch.arange(l2 // d, device=p_cmp.device)[:, None]
        n = torch.arange(l // d, device=p_cmp.device)[None, :]
        off = (m + n).reshape(-1)                        # 0 .. l'/d + l/d - 2
        j = torch.arange(L // l2, device=p_cmp.device)
        idx = (l2 // d) * j[:, None] - off[None, :]      # (nb_s, K) = 式 9 索引
        valid = (idx >= 0) & (idx < nb_c)
        pj = p[:, :, :, idx.clamp(min=0)] * valid.to(p_cmp.dtype)
        return pj.sum(dim=-1)                            # (B, G, L, nb_s)

    def _select_blocks(self, imp: torch.Tensor) -> torch.Tensor:
        """imp: (B, G, L, nb_s) -> 选中块号 (B, G, L, ns)"""
        B, G, L, nb = imp.shape
        ns = min(self.n_sel_blocks, nb)
        t = torch.arange(L, device=imp.device)
        valid = (torch.arange(nb, device=imp.device)[None, :]
                 * self.sel_block_len) < t[:, None]     # (L, nb):未来块不可选
        imp = imp.masked_fill(~valid[:, None], float("-inf"))
        fixed = sorted({0, nb - 2, nb - 1} - {-1, -2})   # 首块 + 最近 2 块固定激活
        is_fixed = torch.zeros(nb, dtype=torch.bool, device=imp.device)
        is_fixed[fixed] = True
        imp = torch.where(is_fixed[None, None, None] & valid[:, None],
                          torch.full_like(imp, float("inf")), imp)
        sel = imp.topk(ns, dim=-1).indices              # (B, G, L, ns)
        return sel.sort(dim=-1).values

    def _attn_selected(self, q, ks, vs, sel):
        """q/ks: (B,H,L,d_k);vs: (B,H,L,d_v);sel: (B,G,L,ns)"""
        B, H, L, _ = q.shape
        g, l2, ns = self.group_size, self.sel_block_len, sel.shape[-1]
        off = torch.arange(l2, device=q.device)
        idx = (sel.unsqueeze(-1) * l2 + off).reshape(B, H // g, L, ns * l2)
        idx = idx.repeat_interleave(g, dim=1)            # 组内头共享同一组块
        k_sel = ks.gather(2, idx[..., None].expand(B, H, L, ns * l2, self.d_k))
        v_sel = vs.gather(2, idx[..., None].expand(B, H, L, ns * l2, self.d_v))
        scores = torch.einsum("bhld,bhjd->bhlj", q, k_sel) / math.sqrt(self.d_k)
        causal = idx < torch.arange(L, device=q.device)[None, None, :, None]
        scores = scores.masked_fill(~causal, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = attn.masked_fill(attn != attn, 0.0)       # 空集合(t=0 等)置零
        return torch.einsum("bhlj,bhjd->bhld", attn, v_sel)

    # ---- 窗口分支(§3.3.3)----
    def _attn_window(self, q, kw, vw):
        B, H, L, _ = q.shape
        w = min(self.n_window, L)
        kw_, vw_ = kw[:, :, -w:], vw[:, :, -w:]          # 最近 w 个 token
        scores = torch.einsum("bhld,bhjd->bhlj", q, kw_) / math.sqrt(self.d_k)
        s = torch.arange(L - w, L, device=q.device)[None, :]
        t = torch.arange(L, device=q.device)[:, None]
        scores = scores.masked_fill(~(s < t)[None, None], float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = attn.masked_fill(attn != attn, 0.0)
        return torch.einsum("bhlj,bhjd->bhld", attn, vw_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        assert L >= self.block_len and L >= self.sel_block_len
        q = self.w_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        ks = self.w_sk(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        vs = self.w_sv(x).view(B, L, self.n_heads, self.d_v).transpose(1, 2)
        kw = self.w_wk(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        vw = self.w_wv(x).view(B, L, self.n_heads, self.d_v).transpose(1, 2)
        ck = self._compress(x, self.w_ck, self.compress_k)   # (B, H, nb, d_k)
        cv = self._compress(x, self.w_cv, self.compress_v)   # (B, H, nb, d_v)
        o_cmp, p_cmp = self._attn_compressed(q, ck, cv)
        imp = self._sel_importance(p_cmp)                    # (B, G, L, nb_s)
        sel = self._select_blocks(imp)                       # (B, G, L, ns)
        o_sel = self._attn_selected(q, ks, vs, sel)
        o_win = self._attn_window(q, kw, vw)
        g = self.gate(x)                                     # (B, L, 3),sigmoid
        o = (g[:, :, 0][:, None, :, None] * o_cmp +
             g[:, :, 1][:, None, :, None] * o_sel +
             g[:, :, 2][:, None, :, None] * o_win)           # 式 5
        o = o.transpose(1, 2).reshape(B, L, -1)
        return self.w_o(o)
```

## 2. DSA:DeepSeek Sparse Attention

### 2.1 动机

NSA 的选择粒度是**块**:一个 query 要么精读整块要么跳过整块。DSA 把粒度细化到 **token 级**,原型由两个组件构成(论文 §2.1):

1. **Lightning Indexer(闪电索引器)**:一个极轻量的打分器,给每个 (query, 历史 token) 对打索引分;
2. **细粒度 token 选择机制**:每个 query 只保留 top-k 个 token 的 KV 条目,再在其上做注意力。

### 2.2 Lightning Indexer(论文式 1)

对 query token $h_t \in \mathbb{R}^d$ 与它前面的 token $h_s \in \mathbb{R}^d$,索引分数为:

$$I_{t,s} = \sum_{j=1}^{H^I} w^{I}_{t,j} \cdot \mathrm{ReLU}\!\left(\mathbf{q}^{I\top}_{t,j}\, \mathbf{k}^{I}_{s}\right)$$

- $H^I$ 是索引器头数;**$q^I_{t,j} \in \mathbb{R}^{d^I}$ 与标量权重 $w^I_{t,j} \in \mathbb{R}$ 由 query token $h_t$ 派生**;$k^I_s \in \mathbb{R}^{d^I}$ 由前面的 token $h_s$ 派生。
- **选 ReLU 是为了吞吐**:打分量很大(每个 query 对所有历史 token 打分),ReLU 在 FP8 下便宜且无溢出问题。
- **索引器很小**:头数少、维度低,整个打分矩阵可以近似认为"免费";官方实现 $H^I = 64$、$d^I = 128$(其中 64 维内容分量 + 64 维解耦 RoPE 分量,论文中"索引器头"的维度构成见下方实现细节)。

**官方实现的对齐细节**(DeepSeek-V3.2-Exp 推理代码 `Indexer`):

- 索引器 query 由 **MLA 的查询潜变量**派生:$q^I = W^{IqB}\big(\mathrm{RMSNorm}(W^{IqA}(h_t))\big)$——即与主注意力共用 $c_t^Q$,不额外从 $h_t$ 全量投影,进一步降低开销;
- 索引器 key 由隐藏态派生:$k^I_s = \mathrm{LayerNorm}\big(W^{Ik}\, h_s\big)$;
- 每个索引头内部再拆成 RoPE 分量与无位置分量,RoPE **不交错**(非 interleaved);
- $w^I_t = W^{Iw}(h_t)$ 再乘 $1/\sqrt{H^I}$ 缩放;
- 部署时对 $q^I, k^I$ 先做 **Hadamard 旋转**(`rotate_activation`,利于量化),再按块做 **FP8 量化**(`float8_e4m3fn`),$k^I$ 的 FP8 cache 附带每块 scale cache,打分用专用 `fp8_index` kernel 完成。

### 2.3 top-k 选择与核心注意力(论文式 2)

$$u_t = \mathrm{Attn}\!\left(h_t,\; \left\{\, c_s \;\middle|\; I_{t,s} \in \mathrm{Top}\text{-}k(I_{t,:}) \,\right\}\right)$$

- **$c_s$ 是 MLA 的低秩潜变量条目**:DSA 实例化在 MLA 之上(DeepSeek-V3.2 与 V3.2-Exp 相比 V3.1-Terminus 唯一的架构改动),因此"被选择的 KV 条目"不是原始 key/value,而是每 token 一份的 $c^{KV}$(外加解耦 RoPE 的 $k^R$,见 [[mla]])。
- **MLA 的 MQA 模式**:论文明确指出"在内核层面,每个 KV 条目必须被多个 query 共享才能高效"(引 NSA/Yuan et al. 2025),因此 DSA 用 MLA 的 MQA 模式——同一个潜向量 $c_s^{KV}$ 被该 query token 的所有注意力头共享;缓存里仍只有一份潜变量,但每个头保留自己的 $W^{UK}, W^{UV}$ 上投影(MLA 的 MHA/MQA 两种模式差异见 V3.2 论文附录 A)。
- **top-k 值**:$k = 2048$(官方配置 `index_topk = 2048`;位置不足 2048 时取 `min(2048, t)`)。
- **选择以"掩码"而非"gather"实现**:先算出全量分数矩阵,再用索引器 top-k 结果构造掩码(top-k 位置置 0,其余 $-\infty$)加到注意力分数上。prefill 阶段用**重建完整 K/V 的 masked-MHA 模式**模拟 DSA(短序列下更高效),decode 阶段在 **MQA 吸收式打分**上加同一掩码(见 2.5)。

### 2.4 两阶段训练(论文式 3-4)

从已扩展到 128K 上下文的 DeepSeek-V3.1-Terminus 基座继续预训练,两阶段数据分布与 V3.1-Terminus 的 128K 长上下文扩展数据完全一致。

**阶段一:密集预热(dense warm-up)**。保持稠密注意力,冻结除索引器外的全部参数。目标分布 $p_{t,:}$ 的构造方式很具体:**先把主注意力分数跨所有头求和,再沿序列维度做 L1 归一化**,得到 $p_{t,:} \in \mathbb{R}^{t}$;索引器损失为 KL 散度:

$$\mathcal{L}^{I} = \sum_{t} \mathbb{D}_{\mathrm{KL}}\!\left(p_{t,:} \;\|\; \mathrm{Softmax}(I_{t,:})\right)$$

学习率 $10^{-3}$,只训 **1000 步**,每步 16 条 128K 序列,合计 **2.1B tokens**。

**阶段二:稀疏训练(sparse training)**。引入 top-k 选择,优化所有参数,使主模型适应 DSA 的稀疏模式。索引器仍与主注意力分布对齐,但只在选中集合 $\mathcal{S}_t = \{ s \mid I_{t,s} \in \mathrm{Top}\text{-}k(I_{t,:}) \}$ 上计算:

$$\mathcal{L}^{I} = \sum_{t} \mathbb{D}_{\mathrm{KL}}\!\left(p_{t,\mathcal{S}_t} \;\|\; \mathrm{Softmax}(I_{t,\mathcal{S}_t})\right)$$

关键训练信号隔离(论文原文明确):**索引器输入从计算图中 detach**,索引器只从 $\mathcal{L}^{I}$ 拿梯度,主模型只从语言建模损失拿梯度。学习率 $7.3 \times 10^{-6}$,每个 query 选 $k = 2048$ 个 KV 条目,训 **15000 步**,每步 480 条 128K 序列,合计 **943.7B tokens**。

后续 post-training(专家蒸馏 + GRPO 混合 RL)同样使用稀疏注意力,且流程、算法、数据与 V3.1-Terminus 完全一致,以便严格评估 DSA 的影响。

### 2.5 复杂度与推理实现

- 主注意力核心复杂度从 $O(L^2)$ 降到 $O(Lk)$($k \ll L$);
- 索引器仍是 $O(L^2)$,但计算量远小于主注意力(低维 $d^I = 128$、$H^I = 64$、FP8、且 q 复用已算好的 $c^Q$),配合优化实现可在长上下文下获得显著端到端加速;
- decode 时 KV cache 与 MLA 完全一致(每 token $c^{KV} + k^R$),索引器对缓存打分后,**只有被选中的条目参与加载与注意力计算**;分数计算沿用 MLA 的权重吸收(见 [[mla]] §2.5):$q_{t,i}^{C\top} k_s^C = (W^{UK\top} q_{t,i}^C)^\top c_s^{KV}$,打分直接打在缓存的潜变量上;
- 短序列 prefill 使用 masked-MHA 模式(重建全量 K/V 后加 top-k 掩码),短上下文下比纯稀疏路径更高效。

### 2.6 参考实现(与论文公式逐条对应)

```python
# ---------------------------------------------------------------- DSA


class LightningIndexer(nn.Module):
    """式 1:I_{t,s} = Σ_j w^I_{t,j} · ReLU(q^I_{t,j} · k^I_s)

    与 DeepSeek-V3.2-Exp 官方推理代码对齐:
    - q^I 由 MLA 查询潜变量 c^Q 派生(wq_a -> RMSNorm 后直接 wq_b 映射)
    - k^I 由 h_s 投影 + LayerNorm 得到
    - 每个索引头 = (d_rope 维 RoPE 分量,非交错) + (d_nope 维内容分量)
    - w^I = W^{Iw}(h_t) · (H^I)^{-1/2}
    - 部署时的 Hadamard 旋转 + 块级 FP8 量化在此省略(属数值实现细节)
    """

    def __init__(self, d_model: int, c_q_dim: int, n_heads: int = 64,
                 head_dim: int = 128, d_rope: int = 64):
        super().__init__()
        self.n_heads, self.head_dim, self.d_rope = n_heads, head_dim, d_rope
        self.wq_b = nn.Linear(c_q_dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(d_model, head_dim, bias=False)
        self.k_norm = nn.LayerNorm(head_dim)
        self.weights_proj = nn.Linear(d_model, n_heads, bias=False)

    def forward(self, c_q: torch.Tensor, h_k: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """c_q: (B, L, c_q_dim);h_k: (B, L, d_model);cos/sin: (L, d_rope)
        返回 I: (B, L, L)"""
        B, L, _ = c_q.shape
        q = self.wq_b(c_q).view(B, L, self.n_heads, self.head_dim)
        q_pe, q_nope = torch.split(
            q, [self.d_rope, self.head_dim - self.d_rope], dim=-1)
        q_pe = q_pe * cos[None, :, None, :] + rotate_half(q_pe) * sin[None, :, None, :]
        q = torch.cat([q_pe, q_nope], dim=-1)                   # (B, L, H^I, d^I)

        k = self.k_norm(self.wk(h_k))                           # (B, L, d^I)
        k_pe, k_nope = torch.split(
            k, [self.d_rope, self.head_dim - self.d_rope], dim=-1)
        k_pe = k_pe * cos[None, :, :] + rotate_half(k_pe) * sin[None, :, :]
        k = torch.cat([k_pe, k_nope], dim=-1)

        relu = F.relu(torch.einsum("blhd,bjd->blhj", q, k))     # (B, L, L, H^I)
        w = self.weights_proj(h_k) / math.sqrt(self.n_heads)    # (B, L, H^I)
        return torch.einsum("bljh,blh->blj", relu, w)           # (B, L, L)


class DSAAttention(nn.Module):
    """DSA 核心注意力:MLA 的 MQA 模式 + 索引器 top-k 掩码。

    - KV 条目 = 共享潜变量 c^{KV}(每 token 一份)+ 解耦 RoPE 的 k^R
    - 打分走权重吸收:q_nope 先乘本头 W^{UK},直接与 c^{KV} 点积
      (等价于恢复完整 key,见 [[mla]] §2.5)
    - 输出侧把 W^{UV} 吸收进 wkv_b 后半段:先对 c^{KV} 加权再上投影
    - 选择实现为"掩码"而非 gather:top-k 位置置 0,其余 -inf
      (prefill 用重建全 K/V 的 masked-MHA 模拟;decode 用上式)
    """

    def __init__(self, d_model: int, n_heads: int, d_nope: int, d_rope: int,
                 d_v: int, kv_rank: int, indexer: nn.Module, top_k: int = 2048):
        super().__init__()
        self.n_heads, self.d_v = n_heads, d_v
        self.d_nope, self.d_rope, self.kv_rank = d_nope, d_rope, kv_rank
        self.top_k = top_k
        self.indexer = indexer
        self.wq_a = nn.Linear(d_model, kv_rank, bias=False)     # h_t -> c^Q
        self.q_norm = RMSNorm(kv_rank)
        self.wq_b = nn.Linear(kv_rank, n_heads * d_nope, bias=False)
        self.wq_r = nn.Linear(kv_rank, n_heads * d_rope, bias=False)
        self.wkv_a = nn.Linear(d_model, kv_rank + d_rope, bias=False)
        self.kv_norm = RMSNorm(kv_rank)
        # 每头自己的 W^{UK}|W^{UV} 上投影(MQA 共享的是潜变量,不是投影)
        self.wkv_b = nn.Linear(kv_rank, n_heads * (d_nope + d_v), bias=False)
        self.w_o = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor,
                sin: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        c_q = self.q_norm(self.wq_a(x))                         # (B, L, kv_rank)
        q_nope = self.wq_b(c_q).view(B, L, self.n_heads, self.d_nope)
        q_pe = self.wq_r(c_q).view(B, L, self.n_heads, self.d_rope)
        q_pe = q_pe * cos[None, :, None, :] + rotate_half(q_pe) * sin[None, :, None, :]

        kv = self.wkv_a(x)
        c_kv, k_pe = torch.split(kv, [self.kv_rank, self.d_rope], dim=-1)
        c_kv = self.kv_norm(c_kv)
        k_pe = k_pe * cos[None, :, :] + rotate_half(k_pe) * sin[None, :, :]

        # 吸收式打分(式 2 的内容分量部分)
        wkv_b = self.wkv_b.weight.view(self.n_heads, -1, self.kv_rank)
        w_uk = wkv_b[:, :self.d_nope]                           # (H, d_nope, kv_rank)
        q_abs = torch.einsum("blhd,hdc->blhc", q_nope, w_uk)    # (B, L, H, kv_rank)
        scores = (torch.einsum("blhc,btc->blht", q_abs, c_kv) +
                  torch.einsum("blhr,btr->blht", q_pe, k_pe))
        scores = scores / math.sqrt(self.d_nope + self.d_rope)

        # 索引器 top-k 掩码(论文式 2;官方实现:掩码而非 gather)
        I = self.indexer(c_q, x, cos, sin)                      # (B, L, L)
        causal = torch.arange(L, device=x.device)[None, :, None] > \
            torch.arange(L, device=x.device)[None, None, :]     # t > s
        I = I.masked_fill(~causal, float("-inf"))
        k_eff = max(1, min(self.top_k, L - 1))
        topk = I.topk(k_eff, dim=-1).indices                    # (B, L, k)
        mask = torch.full_like(I, float("-inf")).scatter_(-1, topk, 0.0)
        mask = mask.masked_fill(~causal, float("-inf"))
        scores = scores + mask[:, None]                         # 广播到各头

        attn = torch.softmax(scores, dim=-1)
        attn = attn.masked_fill(attn != attn, 0.0)              # 空集合(t=0)置零
        o = torch.einsum("blht,btc->blhc", attn, c_kv)          # 先对潜变量加权
        w_uv = wkv_b[:, self.d_nope:]                           # (H, d_v, kv_rank)
        o = torch.einsum("blhc,hdc->blhd", o, w_uv)             # W^UV 吸收进输出
        o = o.reshape(B, L, -1)
        return self.w_o(o)


def indexer_kl_loss(I: torch.Tensor, p_dense_heads: torch.Tensor,
                    top_k=None) -> torch.Tensor:
    """式 3/4。目标分布 = 主注意力权重跨头求和后沿序列 L1 归一化;
    top_k 不为 None 时只在选中集合 S_t 上计算(式 4)。"""
    p = p_dense_heads.sum(dim=1)                                # (B, L, L)
    p = p / p.sum(dim=-1, keepdim=True)                         # L1 归一化
    logp = torch.log_softmax(I, dim=-1)
    if top_k is not None:
        idx = I.topk(top_k, dim=-1).indices
        keep = torch.zeros_like(I, dtype=torch.bool).scatter(-1, idx, True)
        logp = logp.masked_fill(~keep, 0.0)
        p = p.masked_fill(~keep, 0.0)
    return (p * (p.clamp_min(1e-8).log() - logp)).sum(dim=-1).mean()


# 两阶段训练示意(论文 §2.1.1):
# 阶段一:冻结除 indexer 外全部参数,主注意力用稠密权重,
#   loss = indexer_kl_loss(I, p_dense)                       # 式 3,LR 1e-3,1000 步
# 阶段二:全参数可训,主注意力只算 top-k 集合,
#   I = indexer(c_q.detach(), x.detach(), ...)               # 索引器输入 detach
#   loss = lm_loss + indexer_kl_loss(I, p_sparse, top_k)     # 式 4,LR 7.3e-6,15000 步
# 主模型梯度只来自 lm_loss;索引器梯度只来自 KL 损失。


if __name__ == "__main__":
    torch.manual_seed(0)

    # NSA(缩放参数;论文配置 l=32, d=16, l'=64, n=16, w=512)
    nsa = NativeSparseAttention(64, n_heads=4, d_k=16, d_v=16, group_size=2,
                                block_len=4, stride=2, sel_block_len=8,
                                n_sel_blocks=2, n_window=8)
    x = torch.randn(2, 24, 64)
    y = nsa(x)
    assert y.shape == x.shape
    print("NSA forward OK:", y.shape)

    # DSA(缩放参数;论文配置 H^I=64, d^I=128, k=2048)
    indexer = LightningIndexer(64, c_q_dim=16, n_heads=4, head_dim=16, d_rope=8)
    m = DSAAttention(64, n_heads=4, d_nope=8, d_rope=8, d_v=16, kv_rank=16,
                     indexer=indexer, top_k=6)
    L = 20
    cos, sin = torch.randn(L, 8), torch.randn(L, 8)
    x = torch.randn(2, L, 64)
    y = m(x, cos, sin)
    assert y.shape == x.shape
    I = torch.randn(2, L, L)
    p = torch.softmax(torch.randn(2, 4, L, L), dim=-1)
    print("DSA forward OK:", y.shape,
          "| KL:", indexer_kl_loss(I, p).item(),
          indexer_kl_loss(I, p, top_k=6).item())
```

## 3. NSA vs DSA 对比

| 维度 | NSA | DSA |
|---|---|---|
| 选择粒度 | 块(压缩块 32、选择块 64,滑动步长 16) | token |
| 检索方式 | 压缩注意力分数聚合(式 9)+ GQA 组内跨头求和(式 10)→ top-n 块 | Lightning Indexer(式 1)→ top-k token |
| 打分器 | 复用压缩分支注意力分数,**零额外参数** | 多头 ReLU 索引器($H^I=64$、$d^I=128$、FP8) |
| 非稀疏通路 | 压缩分支(全局粗粒度)+ 滑动窗口($w=512$) | 无窗口、无压缩分支,纯 top-k |
| 核心注意力 | 选中块内的原始 token,组内头共享同一组块 | MLA 的 MQA 模式(每头保留自己的 $W^{UK}/W^{UV}$,共享潜变量缓存) |
| 每 query 条目 | $\approx t/d + nl' + w$(32k 平均 2560) | $k = 2048$(固定) |
| 训练 | 端到端预训练,门控可学习(MLP+sigmoid) | 密集 KL 预热(2.1B tokens)→ 稀疏 KL + LM 分离优化(943.7B tokens) |
| 代表模型 | DeepSeek-NSA 27B(原型) | DeepSeek-V3.2-Exp / V3.2 |

DSA 之后,DeepSeek-V4 的 [[csa-hca]] 进一步在稀疏选择**之前**增加 KV 压缩,把"索引器检索的条目"从原始 token 换成压缩块。

## 4. 参考

- DeepSeek-AI, *NSA: Native Sparse Attention*, arXiv:2502.11089(公式 3-12、kernel 设计 §3.4、实验 §4-5)
- DeepSeek-AI, *DeepSeek-V3.2-Exp: Boosting Long-Context Efficiency with DeepSeek Sparse Attention*(DSA 公式 1-4、两阶段训练;GitHub `deepseek-ai/DeepSeek-V3.2-Exp`,报告与推理代码)
- DeepSeek-AI, *DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models*, arXiv:2512.02556(DSA 与 V3.2-Exp 完全同构;复杂度与 masked-MHA prefill 说明见 §2.3)
