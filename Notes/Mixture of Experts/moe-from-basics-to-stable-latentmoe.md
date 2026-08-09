---
type: Note
status: Active
related_to:
  - "[[collective-communication-basics]]"
---

# 从 MoE、Expert Parallel 到 LatentMoE 与 Stable LatentMoE

> 面向 AI Infra 初学者的渐进式笔记。核心阅读材料是 *LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts*（arXiv:2601.18089v1）和 *Kimi K3: Open Frontier Intelligence*（arXiv:2607.24653v1）。

## 0. 先建立全局图景

一句话版本：**MoE 把 Transformer 中每个 token 都经过的一个大 FFN，换成许多 FFN expert；router 为每个 token 只选其中少数几个。**

```text
普通 Transformer block
x -> Norm -> Attention -> + -> Norm -> Dense FFN -> + -> output

MoE Transformer block
x -> Norm -> Attention -> + -> Norm -> Sparse MoE -> + -> output
                                              |
                         router -> Top-k experts -> 加权合并
```

由此得到三个重要事实：

1. MoE 通常替换 **FFN/MLP 子层**，而不是 Attention 子层。
2. 总参数可以很大，但每个 token 只激活少量 expert，因此 activated parameters 和 FLOPs 小得多。
3. expert 分散到多张 GPU 后，token 必须被发往 expert 所在 GPU，所以 MoE 的系统核心是 **all-to-all + grouped GEMM + 负载均衡**。

后文按下面的依赖关系展开：

```text
Dense FFN
  -> 单卡 Sparse MoE（理解 router / dispatch / combine）
    -> Expert Parallel（把 dispatch / combine 变成跨卡 all-to-all）
      -> LatentMoE（先压缩 token，再通信和计算）
        -> Stable LatentMoE（解决超多 expert 下的数值与路由稳定性）
```

## 1. 为什么 MoE 通常放在 FFN 的位置

### 1.1 Transformer block 中的两个主要子层

以 Pre-Norm Transformer 为例：

$$h'=h+\operatorname{Attention}(\operatorname{Norm}(h))$$

$$h''=h'+\operatorname{FFN}(\operatorname{Norm}(h'))$$

Attention 负责 token 之间的信息混合；FFN 对每个 token 独立做通道变换。以 SwiGLU 为例：

$$\operatorname{FFN}(x)=W_2\left[\operatorname{SiLU}(W_gx)\odot W_ux\right]$$

输入输出都是 $d$ 维，中间维度是 $m$。不同 token 之间没有依赖，因此可以自然地说：“这个 token 用 expert 3，另一个 token 用 expert 17。”

### 1.2 为什么替换 FFN 很自然

- **条件计算容易**：每个 expert 本身就是一个 FFN，输入输出形状一致。
- **扩参数但不同比例增 FLOPs**：有 $N$ 个 expert，但每个 token 只算 Top-$K$ 个，$K\ll N$。
- **容易独立分片**：expert 间没有参数依赖，可以把不同 expert 放到不同 GPU。
- **不会直接破坏序列混合**：Attention 仍让 token 交流，MoE 只改变每个 token 的通道变换。

工程中不一定每层都使用 MoE，可以每隔若干层替换一次。Kimi K3 的架构中，每个 attention layer 后都配一个 Stable LatentMoE FFN。

## 2. 标准 Sparse MoE：先只看数学

设输入 $x\in\mathbb{R}^d$，共有 $N$ 个 expert，每个 token 激活 $K$ 个。

### 2.1 Router

router 通常是一个很小的线性层：

$$s=W_rx\in\mathbb{R}^N,\qquad p=\operatorname{softmax}(s)$$

取概率最大的 $K$ 个 expert：

$$\mathcal{T}_K(x)=\operatorname{TopK}(p)$$

标准 MoE 输出为：

$$\operatorname{MoE}(x)=\sum_{i\in\mathcal{T}_K(x)}\widetilde p_iE_i(x)$$

其中 $\widetilde p_i$ 通常是在选中的 $K$ 项上重新归一化的权重。

### 2.2 一个 token 到底发生了什么

假设 $N=4,K=2$，router 给出：

```text
expert       E0    E1    E2    E3
probability  .05   .60   .10   .25
```

Top-2 是 E1、E3。token 被复制两份，分别进入 E1 和 E3，最后按 $0.60/(0.60+0.25)$ 与 $0.25/(0.60+0.25)$ 加权相加。这里的“复制”会让实际 expert-token 数从 $T$ 变为 $T\times K$。

### 2.3 容易混淆的三种规模

| 名称 | 含义 | 主要影响 |
|---|---|---|
| 总参数 | 所有 expert 参数之和 | checkpoint、部署显存 |
| 激活参数 | 一个 token 实际走过的参数 | 每 token FLOPs |
| 常驻参数/rank | 某张 GPU 持有的 expert 参数 | HBM 容量、权重读取 |

“1T 参数 MoE”不意味着每个 token 都计算 1T 参数；但权重仍需存放在集群中，低 batch decode 时还可能受读取 expert 权重的 HBM 带宽限制。

## 3. 可运行的 PyTorch 教学实现

下面的代码保留最清楚的语义：先 flatten token，Top-K 路由，再逐 expert 收集 token，最后 `index_add_` 合并。它适合学习和单元测试，**不适合生产性能**。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUExpert(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class SparseMoE(nn.Module):

    def __init__(self, d_model: int, d_ff: int,
                 n_experts: int, top_k: int = 2):
        super().__init__()
        assert 1 <= top_k <= n_experts
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([
            SwiGLUExpert(d_model, d_ff) for _ in range(n_experts)
        ])

    def forward(self, x: torch.Tensor):
        # x: [batch, seq, d_model]
        shape = x.shape
        tokens = x.reshape(-1, shape[-1])                 # [T, d]

        logits = self.router(tokens)                      # [T, N]
        probs = logits.softmax(dim=-1)
        weights, expert_ids = probs.topk(self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        # 每行 route 对应 (token_id, expert_id, weight)。
        token_ids = torch.arange(tokens.size(0), device=x.device)
        token_ids = token_ids[:, None].expand(-1, self.top_k).reshape(-1)
        flat_expert_ids = expert_ids.reshape(-1)          # [T*K]
        flat_weights = weights.reshape(-1)                # [T*K]

        out = torch.zeros_like(tokens)
        for expert_id, expert in enumerate(self.experts):
            mask = flat_expert_ids == expert_id
            selected_token_ids = token_ids[mask]
            if selected_token_ids.numel() == 0:
                continue
            expert_in = tokens.index_select(0, selected_token_ids)
            expert_out = expert(expert_in)
            expert_out = expert_out * flat_weights[mask, None]
            # 同一个 token 的 K 个 expert 输出在这里相加。
            out.index_add_(0, selected_token_ids, expert_out)

        # 简化的 Switch-style 辅助均衡项；真实训练还会考虑实现所用定义。
        importance = probs.mean(dim=0)                    # router 概率份额
        load = F.one_hot(expert_ids, self.n_experts).float()
        load = load.mean(dim=(0, 1))                      # 实际选择份额
        aux_loss = self.n_experts * torch.sum(importance * load)
        return out.view(shape), aux_loss


if __name__ == "__main__":
    torch.manual_seed(0)
    moe = SparseMoE(d_model=64, d_ff=128, n_experts=8, top_k=2)
    x = torch.randn(2, 16, 64, requires_grad=True)
    y, aux = moe(x)
    loss = y.square().mean() + 0.01 * aux
    loss.backward()
    assert y.shape == x.shape
    print(y.shape, aux.item())
```

读代码时抓住四个动词：

1. **route**：算 `expert_ids` 和 `weights`；
2. **dispatch**：按 expert 收集并排列 token；
3. **expert compute**：同一 expert 的 token 做一批 GEMM；
4. **combine**：按原 token id 加权累加并恢复形状。

生产实现会把 Python expert 循环换成 grouped GEMM，把布尔 mask 换成排序/直方图/前缀和 kernel，并预分配通信 buffer。

## 4. Router 为什么需要负载均衡

如果所有 token 都偏爱 E0：

- E0 所在 GPU 成为 straggler，其他 GPU 等待；
- E0 的临时激活和通信 buffer 可能溢出；
- 冷门 expert 收不到足够梯度，变成 dying expert；
- collective 的完成时间由最慢 rank 决定，平均负载好看也没有用。

常见办法有：辅助均衡 loss、每 expert capacity 与 token drop、expert-choice routing、以及不把额外 loss 加进模型目标的 auxiliary-loss-free bias。后者在 Top-K 排名分数上给 expert 加 bias：过热 expert 降 bias，过冷 expert 升 bias。

需要分清：均衡 router 的选择是一项**模型算法问题**；把已经产生的 expert-token 工作放到 GPU 上并让物理 rank 同时完成，是一项**系统调度问题**。

## 5. Expert Parallel：从单卡 dispatch 到跨卡 dispatch

### 5.1 参数怎样放置

设 $N=8$ 个 expert、EP size $R=4$，最简单的静态映射是：

```text
rank 0: E0 E1    rank 1: E2 E3    rank 2: E4 E5    rank 3: E6 E7
owner(expert_id) = expert_id // (N / R)
```

每个 EP rank 一开始都有自己的一批 token，但 token 的 Top-K expert 可能在任意 rank。Attention/共享 expert 往往在 EP ranks 上复制，routed experts 才按 EP 分片。

### 5.2 前向的完整数据流

```text
各 rank 本地 token
  -> router + Top-K
  -> 展开为 T*K 条 route，附带 token_id / expert_id / weight / source_rank
  -> 按 destination_rank 排列
  -> All-to-All dispatch
  -> 每个 rank 按 local_expert 分组
  -> grouped GEMM
  -> All-to-All combine（沿原路返回）
  -> source rank 按 token_id 加权 scatter-add
```

以 rank 0 的 token A 为例，若选 E1（rank 0）和 E6（rank 3），一份留本地，一份发给 rank 3。rank 3 算完 E6 后把结果发回 rank 0，rank 0 才能合并出 A 的最终输出。

因此一个 MoE 层通常有两次 all-to-all：

- dispatch：发送 expert 输入；
- combine：返回 expert 输出。

这正是 [[collective-communication-basics|集合通信基础]] 中 all-to-all 的典型应用。

### 5.3 对应 PyTorch distributed 的伪代码

```python
def expert_parallel_forward(x, router, local_experts, ep_group):
    # 1. 本地路由，产生 T*K 条 route。
    score = softmax(router(x), dim=-1)
    weight, expert_id = topk(score, K)
    routes = expand_routes(x, expert_id, weight)
    routes.dst_rank = owner(routes.expert_id)

    # 2. 按目的 rank 排序，并交换每个 peer 的变长 split size。
    send = sort_by(routes, keys=(dst_rank, expert_id))
    send_splits = histogram(send.dst_rank, R)
    recv_splits = all_to_all_counts(send_splits, ep_group)

    # metadata 至少要能在返回时找到 source token 和 route weight。
    recv_x = torch.empty((sum(recv_splits), x.size(-1)),
                         device=x.device, dtype=x.dtype)
    dist.all_to_all_single(
        recv_x,
        send.x,
        output_split_sizes=recv_splits,
        input_split_sizes=send_splits,
        group=ep_group,
    )
    recv_meta = all_to_all_metadata(send.meta, send_splits, recv_splits)

    # 3. 接收 buffer 已按 expert 分组，执行 local grouped GEMM。
    recv_y = grouped_expert_gemm(recv_x, recv_meta.local_expert_id,
                                 local_experts)

    # 4. 反向 all-to-all，结果回到 route 的 source rank。
    returned_y = torch.empty_like(send.x)
    dist.all_to_all_single(
        returned_y,
        recv_y,
        output_split_sizes=send_splits,
        input_split_sizes=recv_splits,
        group=ep_group,
    )

    # 5. 撤销排序，对同一 source token 的 K 条 route 加权求和。
    return scatter_add(returned_y * send.weight, send.source_token_id)
```

真实实现还必须处理：

- 变长 `send_splits/recv_splits` 和 capacity；
- token 排列的逆置索引；
- 前向与反向的 autograd collective；
- expert 权重梯度只属于 owner rank；
- EP 与 DP/TP/PP/CP process group 的正交组合；
- 通信 stream 与 shared expert / GEMM 的 overlap；
- 跨节点时尽量让高频 EP 留在高带宽互联域内。

### 5.4 反向传播

反向基本是前向的镜像：输出梯度先按 route dispatch 到 expert owner，计算 expert 参数梯度和输入梯度，再把输入梯度发回 token source，最后对 Top-K 分支求和。router 的梯度来自 mixture weight；Top-K 的离散索引本身通常不求导。

### 5.5 性能直觉

标准 MoE 每 rank 的通信量近似与 $T K d / R$ 成正比。即使 GEMM 很快，all-to-all 仍可能占主导；小 batch decode 时，每个 expert token 很少，GEMM 又会退化成读取大量权重的 memory-bound 工作。

## 6. LatentMoE：压缩 routed path，而不是压缩整个模型

### 6.1 论文的问题意识

LatentMoE 论文同时优化两个指标：

- accuracy per FLOP：计算相同，谁更准；
- accuracy per parameter：读取/存储相同参数，谁更准。

后者对低延迟 decode 尤其重要，因为小 expert batch 经常不是算力不足，而是 HBM 不断加载 expert 权重。论文用 Qwen3-235B-A22B 和 GB200 做 roofline 示例，估算每 expert 约需 1418 tokens 才进入 compute-bound，而交互式服务通常只有数百。

论文还指出吞吐场景中 EP all-to-all 可能成为主要成本。减少 expert 中间维度 $m$ 不会减小通信 token，减少 $K$ 又会损失每 token 的非线性容量，于是最值得动的是 routed hidden width。

### 6.2 架构

只把 routed 分支从模型宽度 $d$ 压到 latent width $\ell$：

$$z=W_\downarrow x\in\mathbb{R}^{\ell}$$

$$u=\sum_{i\in\mathcal{T}}p_iE_i(z),\qquad y_\text{routed}=W_\uparrow u$$

完整输出还加上在原始 $d$ 维工作的 shared experts：

$$y=W_\uparrow\left(\sum_{i\in\mathcal{T}}p_iE_i(W_\downarrow x;\ell)\right)+\sum_jE_j^\text{shared}(x;d)$$

关键细节：

- router 仍直接看完整的 $x\in\mathbb{R}^d$；
- **先 down-project，再 dispatch**，所以网络发送的是 $\ell$ 维 token；
- routed expert 的输入输出宽度是 $\ell$，但中间非线性宽度 $m$ 保持不变；
- full-width shared experts 保留通用能力，routed experts 学专门变换。

### 6.3 两个版本

令压缩率 $\alpha=d/\ell$，标准 MoE 有 $N$ 个 experts、Top-$K$。

| 版本 | routed experts | 激活数 | 目标 |
|---|---:|---:|---|
| 标准 MoE | $N$ | $K$ | baseline |
| $\ell$-MoE-eff | $\alpha N$ | $K$ | 接近精度，降低推理成本 |
| $\ell$-MoE-acc | $\alpha N$ | $\alpha K$ | 近似相同推理成本，提高精度 |

直觉是：每个 routed expert 约缩小 $\alpha$ 倍，所以可把 expert 总数放大 $\alpha$ 倍而保持总 routed 参数近似不变。`acc` 版本再把 Top-K 放大 $\alpha$ 倍；每条 route 只传 $\ell=d/\alpha$ 维，因此 $\alpha K\ell=Kd$，通信量近似不变。

同时选择组合数从 $\binom{N}{K}$ 变为 $\binom{\alpha N}{\alpha K}$，专家组合空间显著扩大。论文实验发现压缩率不超过 4 时质量可保持，并推荐 accuracy 版本作为 accuracy/inference-cost Pareto 方案；这不是任意模型上都成立的理论保证，$\ell$ 仍不能低于任务所需的有效特征秩。

### 6.4 从标准 MoE 代码改成 LatentMoE

```python
class LatentMoE(nn.Module):
    def __init__(self, d_model, d_latent, d_ff,
                 n_routed, top_k, n_shared=1):
        super().__init__()
        self.down_proj = nn.Linear(d_model, d_latent, bias=False)
        self.up_proj = nn.Linear(d_latent, d_model, bias=False)
        self.router = nn.Linear(d_model, n_routed, bias=False)  # 看完整 x
        self.n_routed, self.top_k = n_routed, top_k
        self.routed = nn.ModuleList([
            SwiGLUExpert(d_latent, d_ff) for _ in range(n_routed)
        ])
        self.shared = nn.ModuleList([
            SwiGLUExpert(d_model, d_ff) for _ in range(n_shared)
        ])

    def forward(self, x):
        shape = x.shape
        full_x = x.reshape(-1, shape[-1])
        z = self.down_proj(full_x)                # EP dispatch 的是 z

        probs = self.router(full_x).softmax(-1)   # router 看完整 x
        weights, expert_ids = probs.topk(self.top_k, dim=-1)
        weights = weights / weights.sum(-1, keepdim=True)
        token_ids = torch.arange(z.size(0), device=z.device)
        token_ids = token_ids[:, None].expand(-1, self.top_k).reshape(-1)
        flat_ids, flat_w = expert_ids.reshape(-1), weights.reshape(-1)

        latent_out = torch.zeros_like(z)
        for expert_id, expert in enumerate(self.routed):
            mask = flat_ids == expert_id
            ids = token_ids[mask]
            if ids.numel():
                value = expert(z.index_select(0, ids)) * flat_w[mask, None]
                latent_out.index_add_(0, ids, value)

        shared_y = sum(expert(x) for expert in self.shared)
        routed_y = self.up_proj(latent_out).view(shape)
        return shared_y + routed_y
```

这里没有直接复用普通 `SparseMoE`，因为那会让 router 错误地看 `z`，或者产生 full-width router 与 latent 输入的维度冲突。正确工程接口应拆成 `route(x)`、`dispatch(z, routes)`、`experts(…)`、`combine(…)` 四步。架构图容易画对，软件边界却很容易写错。

## 7. Stable LatentMoE：Kimi K3 为什么还要再加“Stable”

Kimi K3 把 LatentMoE 推到 2.78T 总参数、104.2B 激活参数：$d=7168$、$\ell=3584$、896 routed experts、Top-16、2 个 full-width shared experts、每 expert 中间维度 3072。

此时原始 LatentMoE 暴露两个问题：

1. `down projection -> gated expert 的多分支矩阵乘 -> up projection` 接近连续四次矩阵乘，在 2.8T 规模出现 routed branch 内部激活爆炸；
2. 近千 expert 下，固定步长的无辅助损失 bias 更新会在“响应太慢”和“负载振荡”之间难以调节。

Stable LatentMoE 用三个组件处理它们。

### 7.1 Aggregation 后、up projection 前加 RMSNorm

$$u=\sum_{i\in\mathcal{T}_k(x)}p_iE_i^\text{routed}(W_\downarrow x)$$

$$y=\sum_jE_j^\text{shared}(x)+W_\uparrow\operatorname{RMSNorm}(u)$$

不同 expert 与 mixture weight 会让 $u$ 的尺度变化。RMSNorm 在升回 $d$ 维前固定 routed branch 的尺度，降低它与 full-width shared branch 相加时的敏感性。报告称它不仅稳定训练，也持续改善验证 loss 和下游指标。

### 7.2 SiTU-GLU：给乘法两边都加平滑上界

SwiGLU 的 gate 和 up 两个乘法因子都可能无界，大坐标相乘会制造 outlier，并增加低精度 overflow 风险。SiTU-GLU 定义：

$$\left[\beta_1\tanh\left(\frac{W_gx}{\beta_1}\right)\odot\sigma(W_gx)\right]
\odot\left[\beta_2\tanh\left(\frac{W_ux}{\beta_2}\right)\right]$$

Kimi K3 使用 $\beta_1=4,\beta_2=25$，所以每个输出坐标绝对值不超过 $100$。在零点附近，$\beta\tanh(z/\beta)=z+O(z^3/\beta^2)$，因此局部仍像 SwiGLU；相比 hard clamp，平滑饱和保留了更连续的梯度。

### 7.3 Quantile Balancing（QB）

router 先算 sigmoid score：

$$s_i=\sigma(W_rx_i)$$

选择时加入 expert bias，但混合权重不含 bias：

$$\mathcal T_i=\operatorname{TopK}(s_i+b),\qquad
p_{i,j}=\frac{s_{i,j}}{\sum_{r\in\mathcal T_i}s_{i,r}}$$

因此 $b$ 只控制“谁被选中”，不直接改变 expert 输出混合比例和 router 的梯度目标。

对 $m$ 个 token、$n$ 个 expert、Top-$k$，每 expert 目标负载是：

$$q=mk/n$$

QB 在当前步临时取 Top-$(k+1)$。第 $k+1$ 个 biased score 是 token $i$ 的入选门槛 $\alpha_i$。expert $j$ 要进入该 token，需满足 $s_{i,j}+b_j>\alpha_i$。于是查看所有 token 的 margin $s_{i,j}-\alpha_i$，选一个分位数阈值，恰好让 $q$ 个 margin 超过它：

$$\widehat b_j^{(t+1)}=-\operatorname{quantile}_{1-k/n}(s_{:,j}-\alpha^{(t)})$$

最后对所有 bias 减去均值；公共平移不改变 Top-K。新 bias 只在**下一 training step** 生效，推理时冻结。

固定步长法只知道“expert 太热还是太冷”；QB 直接估算“bias 应移动多少才到目标分位点”，所以不需要类似 learning rate 的步长超参。

全局 batch 有数百万 token，不能 gather 全部 $m\times n$ margins。Kimi K3 为每个 expert 维护数百至 1000 个 bin 的 histogram，各 rank 本地累计计数，step 末一次 integer all-reduce 合并，再从累计计数读分位数。通信规模是 $O(nB)$，与 token 数 $m$ 无关。

## 8. QB 与 MoonEP：算法均衡后为什么还需要系统均衡

QB 让训练 step 的全局 expert 负载接近目标，但一个 micro-batch、一个 layer、一次实际执行中仍可能不均，而且 expert 的静态 owner 映射会把热点集中到某个 rank。Kimi K3 的 MoonEP 在物理执行层再做一次规划。

### 8.1 MoonEP 的做法

- 根据当前 micro-batch 和 layer 的 router 输出在线规划；
- 把热点 expert 临时复制成 **redundant expert** 到缺工作量的 rank，并预取权重；
- 重新决定 token route 的执行目的地，使每 rank 恰收 $S\times K$ 个 expert-token；
- 反向把 redundant copy 的梯度先放本地 reduce buffer，计算后归约回 home rank；
- 每 rank 预留最多 $E/R$ 个 redundant expert slots，可保证总能找到严格均衡方案。

```text
逻辑 expert 归属不变：梯度最终回 home rank
物理执行位置可变：热点 expert 可在本 step 临时复制执行
```

### 8.2 严格均衡带来的系统收益

1. 每 rank 固定接收 $S\times K$ token，所有 rank 计算量相同；
2. 通信 buffer 固定为 $S\times K$，不需为最坏不均衡准备 $S\times K\times R$；
3. 所有层的计算 shape 静态已知，避免 CPU 每层同步 GPU token count 后再 launch；
4. fused permute/unpermute 直接把 token 写入远端 expert-grouped 位置，计算直接读取通信 buffer view，减少中间 copy；
5. rank 内不同 expert 仍可能不均，所以 grouped GEMM 还使用 workload-aware SM 调度；shared experts 放另一 CUDA stream 做 overlap。

MoonEP 的抽象伪代码是：

```python
routes = router(microbatch)
plan = gpu_plan(routes, target_tokens_per_rank=S*K,
                max_redundant_experts_per_rank=E/R)
async_prefetch(plan.redundant_expert_weights)
recv = fused_permute_all_to_all(routes, plan)   # 直接落到 grouped layout
y = workload_aware_grouped_gemm(recv)
out = fused_all_to_all_unpermute(y, plan)

# backward
grad_redundant = backward_on_redundant_copies(...)
reduce_to_home_expert(grad_redundant, plan)
```

## 9. Stable LatentMoE 的 serving 优化

Kimi K3 报告还给出几项与架构直接对应的 kernel 设计：

- 把 latent down-projection 与 router 合成一次 GEMM；
- latent weight 跨 rank 分片，把 output all-gather 融入 GEMM epilogue；
- latent 通信与 shared expert 计算 overlap；
- 小 batch routed expert GEMM 是权重流式读取的 memory-bound 场景，使用 token-centric kernel：warp 负责一个输出 neuron，lane teams 分摊不同 experts，最后 warp reduction；
- 权重提前做离线 layout permutation，降低运行时 dequantization 开销。

这些优化不是架构公式的附属细节。LatentMoE 把矩阵做窄、把 Top-K 做大后，传统面向大 tile 的 GEMM kernel 未必自然高效；论文在 95B 模型实测中也观察到不同并发下吞吐并非全面胜过标准 MoE，说明“理论通信/FLOPs 更少”不自动等于“kernel 已经更快”。

## 10. 一次完整前向：把所有层次串起来

```text
1. x [T,d]
2. full-width router: score = sigmoid(x @ W_r)
3. Top-K(score + frozen/current bias)，weight 只由原 score 归一化
4. z = x @ W_down^T                         [T,l]
5. MoonEP 根据 routes 规划 redundant experts 和目的 rank
6. all-to-all dispatch：发送 z，不发送 d 维 x
7. local SiTU-GLU routed experts + grouped GEMM
8. all-to-all combine，按 route weight 聚合为 u [T,l]
9. routed = W_up(RMSNorm(u))                 [T,d]
10. shared = sum(full-width shared experts(x))
11. y = shared + routed，再进入 block residual
12. step 末用全局 histogram all-reduce 更新下一步 QB bias
```

如果能解释每一步的 tensor shape、参数在哪张卡、是否通信、通信量与 $d$ 还是 $\ell$ 成正比，就已经掌握了 MoE infra 的主干。

## 11. 常见误区与排查清单

- **“Top-2 就是先跑第一名，不够再跑第二名”**：不是；两个 expert 都执行，输出加权相加。
- **“总参数大，所以每 token FLOPs 一样大”**：错；要看 activated experts。
- **“All-to-All 是同步梯度”**：MoE 前向的 all-to-all 搬的是 activation/token；梯度同步是另一条数据流。
- **“全局 expert 数越多，通信一定越多”**：通信首先取决于 route 数 $TK$ 和每 route token 宽度；$N$ 还会影响调度、metadata 和常驻参数。
- **“LatentMoE 的 router 看 latent z”**：原论文 router 看 full-width $x$；dispatch 和 expert compute 才在 latent space。
- **“RMSNorm 放在 down projection 前”**：Stable LatentMoE 的新增关键 Norm 是 expert 聚合后、up projection 前。
- **“QB 保证每次 kernel 完全均衡”**：QB 是跨全局 training step 的路由 bias 更新；MoonEP 才对当前执行做 rank-level perfect balance。
- **“负载均衡只看平均 token 数”**：还要看最慢 rank、rank 内 expert skew、buffer 峰值、token drop、通信热点。

建议打印/监控：每 expert token histogram、每 rank token 总量、最大/平均负载比、dropped token、dispatch/combine 时间、grouped GEMM 时间、通信 overlap 比例、expert activation max/RMS、router entropy。

## 12. 自测题

1. $T=1024,N=64,K=2,R=8$，均匀路由时每 expert 和每 rank 平均各处理多少 expert-token？
2. 为什么把 $m$ 减半会减少 expert FLOPs，却不会让 dispatch activation 减半？
3. LatentMoE 取 $d=4096,\ell=1024$，若 Top-K 从 8 增到 32，为什么 routed activation 通信量近似不变？
4. 为什么 shared expert 通常复制，而 routed experts 才做 EP？
5. QB 已经均衡，为何某个 micro-batch 仍可能出现 rank straggler？

答案：1）总 route 数 2048，每 expert 32，每 rank 256；2）网络发送的是 $d$ 维 token，和 FFN 中间维 $m$ 无关；3）$8\times4096=32\times1024$；4）shared path 每 token 必走，复制可避免额外 route，routed 参数规模大且稀疏，分片收益高；5）QB 面向全局 step 的统计目标，局部采样和静态 expert placement 仍可能偏斜。

## 参考材料

- Venmugil Elango et al., *LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts*, arXiv:2601.18089v1。重点：`sections/design_choices.tex`、`sections/latentmoe.tex`、`sections/results.tex`。
- Kimi Team, *Kimi K3: Open Frontier Intelligence*, arXiv:2607.24653v1。重点：`2-model-architecture.tex` 的 Stable LatentMoE、`5-infrastructure.tex` 的 MoonEP/serving kernels、`appendix.tex` 的 QB 与 MoonEP 推导。
- 本 vault 的 [[collective-communication-basics|集合通信基础（原语 / 算法 / 拓扑）]]。
