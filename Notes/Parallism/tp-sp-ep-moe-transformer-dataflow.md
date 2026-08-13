---
type: Note
related_to:
  - "[[tensor-and-sequence-parallelism]]"
  - "[[expert-parallelism-and-moe]]"
status: Active
---

# TP=2、SP=2、EP=4 的单 Transformer + MoE 数据流

## 6. Transformer Block 的完整数据流

以下只描述一个 Pre-LN Transformer block 的前向过程。忽略 batch 维度，记 `L=S/2`，`d=H/h`；`I` 是 SwiGLU 的中间维度，`E` 是专家总数，`K` 是 top-k。激活元素占 `b` 字节（例如 BF16/FP16 时 `b=2`）。下文的通信量均指前向的**有效张量载荷**；具体 collective 算法可能额外引入分块、对齐、padding 或协议开销。

设备网格与进程组固定为：

```text
             TP rank 0    TP rank 1
SP rank 0       GPU0         GPU1
SP rank 1       GPU2         GPU3

TP groups: [GPU0, GPU1]、[GPU2, GPU3]
SP groups: [GPU0, GPU2]、[GPU1, GPU3]
EP group:  [GPU0, GPU1, GPU2, GPU3]
MoE owner: GPU0（token 0:L）、GPU2（token L:S）
```

### 6.1 Block 输入：`(S,H)` 的二维切片

全局逻辑输入为 `X.shape=(S,H)`，没有任何单卡持有它的完整副本。block 边界上的物理激活是同时按 sequence 和 hidden 切分的四块：

```text
GPU0: X[0:L,   0:H/2], shape=(L,H/2)  # SP=0, TP=0
GPU1: X[0:L, H/2:H],  shape=(L,H/2)  # SP=0, TP=1
GPU2: X[L:S,   0:H/2], shape=(L,H/2)  # SP=1, TP=0
GPU3: X[L:S, H/2:H],  shape=(L,H/2)  # SP=1, TP=1
```

本节选择 GPU0、GPU2 为 MoE token owner。也就是说，Attention 内同一 token 的两个 hidden shard 分别在两个 TP rank 上；进入 MoE 前才把它们收集到 owner，避免同一 token 被重复路由。

### 6.2 第一个 Norm：hidden shard 上计算，TP 内归约统计量

所有 GPU 都保存完整的 Norm 参数：`gamma.shape=(H,)`，LayerNorm 时还有 `beta.shape=(H,)`。但 GPU 只将参数对应的 hidden slice 作用于本地激活，例如 GPU0 使用 `gamma[0:H/2]`，GPU1 使用 `gamma[H/2:H]`。

以 LayerNorm 为例，各 GPU 先从自己的 `(L,H/2)` 激活计算每个 token 的局部统计量：

```text
GPU0: sum_0, sqsum_0 over X[0:L,   0:H/2], shape=(L,)
GPU1: sum_1, sqsum_1 over X[0:L, H/2:H],  shape=(L,)
GPU2: sum_2, sqsum_2 over X[L:S,   0:H/2], shape=(L,)
GPU3: sum_3, sqsum_3 over X[L:S, H/2:H],  shape=(L,)
```

通信就在这里发生：

```text
[GPU0,GPU1]: all-reduce(sum, sqsum)，两个输入各为 (L,)
[GPU2,GPU3]: all-reduce(sum, sqsum)，两个输入各为 (L,)
```

每张参与卡的逻辑归约载荷为 `2L` 个标量，即 `2Lb` 字节；两卡实现中可直接互换这两个 `(L,)` 向量。归约后，每张卡都有其 `L` 个 token 的完整 `H` 维均值/方差，并在本地得到：

```text
GPU0: U0 = Norm(X[0:L,   0:H/2]), shape=(L,H/2)
GPU1: U1 = Norm(X[0:L, H/2:H]),  shape=(L,H/2)
GPU2: U2 = Norm(X[L:S,   0:H/2]), shape=(L,H/2)
GPU3: U3 = Norm(X[L:S, H/2:H]),  shape=(L,H/2)
```

RMSNorm 只需归约平方和，因此相同位置的载荷由 `2L` 个标量降为 `L` 个标量。

### 6.3 QKV Column-Parallel：先收集完整 hidden，再产生本卡 heads

Q/K/V 权重按输出列切分，并沿 SP 轴复制：

```text
GPU0、GPU2:
  Wq_0, Wk_0, Wv_0.shape=(H,H/2)  # 对应 head 0:h/2
GPU1、GPU3:
  Wq_1, Wk_1, Wv_1.shape=(H,H/2)  # 对应 head h/2:h
```

列并行 GEMM 需要完整输入 hidden，因此先在每个 TP group 内 all-gather：

```text
GPU0 -> GPU1: U0, shape=(L,H/2)  # X[0:L,   0:H/2] 的 Norm 输出
GPU1 -> GPU0: U1, shape=(L,H/2)  # X[0:L, H/2:H] 的 Norm 输出
GPU0/GPU1: U_01=concat(U0,U1), shape=(L,H)  # 逻辑切片 X[0:L,:]

GPU2 -> GPU3: U2, shape=(L,H/2)  # X[L:S,   0:H/2] 的 Norm 输出
GPU3 -> GPU2: U3, shape=(L,H/2)  # X[L:S, H/2:H] 的 Norm 输出
GPU2/GPU3: U_23=concat(U2,U3), shape=(L,H)  # 逻辑切片 X[L:S,:]
```

每张 GPU 发送并接收 `(L,H/2)`，即单卡发送量和接收量均为 `LHb/2`；两个 TP group 合计双向载荷为 `2LHb`。all-gather 后的 `(L,H)` 只作本地投影，不作为 block 的稳定激活布局。

```text
GPU0: Q0=U_01@Wq_0, K0=U_01@Wk_0, V0=U_01@Wv_0, each shape=(L,H/2)
      = logical token [0:L]、head [0:h/2]，reshape 后 each (L,h/2,d)
GPU1: Q1,K1,V1, each shape=(L,H/2)
      = logical token [0:L]、head [h/2:h]，reshape 后 each (L,h/2,d)
GPU2: Q2,K2,V2, each shape=(L,H/2)
      = logical token [L:S]、head [0:h/2]，reshape 后 each (L,h/2,d)
GPU3: Q3,K3,V3, each shape=(L,H/2)
      = logical token [L:S]、head [h/2:h]，reshape 后 each (L,h/2,d)
```

QKV GEMM 本身没有通信。每卡的 fused QKV 临时输出是 `(L,3H/2)`，拆分后 Q、K、V 均是 `(L,H/2)`。

### 6.4 SP K/V 交换：每个 query 留在本地，K/V 获得全序列上下文

GPU0/GPU2 持有同一半 heads `[0:h/2]`，GPU1/GPU3 持有另一半 heads `[h/2:h]`。每个 GPU 的 Q 不移动；只在同一 TP rank 的 SP group 内交换 K、V：

```text
GPU0 -> GPU2: K0,V0, each shape=(L,h/2,d)  # token [0:L],   head [0:h/2]
GPU2 -> GPU0: K2,V2, each shape=(L,h/2,d)  # token [L:S],   head [0:h/2]

GPU1 -> GPU3: K1,V1, each shape=(L,h/2,d)  # token [0:L],   head [h/2:h]
GPU3 -> GPU1: K3,V3, each shape=(L,h/2,d)  # token [L:S],   head [h/2:h]
```

对每张 GPU，K 和 V 合计发送 `(L,H)` 个元素、接收 `(L,H)` 个元素，即各 `LHb` 字节；四卡合计双向有效载荷为 `4LHb`。若使用 all-gather，交换后每卡逻辑上持有：

```text
GPU0: Q0=(L,h/2,d), token [0:L], head [0:h/2]
      K/V=concat(K0/K2, V0/V2), each (S,h/2,d), token [0:S], head [0:h/2]
GPU1: Q1=(L,h/2,d), token [0:L], head [h/2:h]
      K/V=concat(K1/K3, V1/V3), each (S,h/2,d), token [0:S], head [h/2:h]
GPU2/GPU3: 同理，Q 的 token 切片为 [L:S]
```

若用 ring attention，收到的远端 `(L,h/2,d)` K/V block 可立刻与本地 Q 计算，不必长期物化完整 `(S,h/2,d)`；通信张量和通信量不变。

### 6.5 本地 Attention：输出仍是本地 token 的半数 heads

每张卡在本地执行 attention。以 GPU0 为例：

```text
GPU0: score0 = Q0 @ K0_all^T / sqrt(d), shape=(h/2,L,S)
      A0 = softmax(score0) @ V0_all,      shape=(L,h/2,d)=(L,H/2)
      A0 对应逻辑 X 的 token [0:L]、head [0:h/2]
```

GPU1 得到 token `[0:L]`、head `[h/2:h]` 的 `A1.shape=(L,H/2)`；GPU2、GPU3 对 token `[L:S]` 得到 `A2`、`A3`。本步骤没有通信。

### 6.6 Wo Row-Parallel：TP 内合并半数 heads 的 partial output

输出投影 `Wo.shape=(H,H)` 按输入行切分，并沿 SP 轴复制：

```text
GPU0、GPU2: Wo_0.shape=(H/2,H)  # 消费 head [0:h/2] 的 A0/A2
GPU1、GPU3: Wo_1.shape=(H/2,H)  # 消费 head [h/2:h] 的 A1/A3
```

每卡先在本地生成完整输出维度的 partial sum：

```text
GPU0: P0=A0@Wo_0, shape=(L,H), token [0:L]
GPU1: P1=A1@Wo_1, shape=(L,H), token [0:L]
GPU2: P2=A2@Wo_0, shape=(L,H), token [L:S]
GPU3: P3=A3@Wo_1, shape=(L,H), token [L:S]
```

通信紧随其后：

```text
[GPU0,GPU1]: all-reduce(P0,P1) -> Y_01=P0+P1, shape=(L,H) on GPU0 and GPU1
[GPU2,GPU3]: all-reduce(P2,P3) -> Y_23=P2+P3, shape=(L,H) on GPU2 and GPU3
```

每个 all-reduce 的输入/输出张量均为 `(L,H)`；在两卡的直接交换实现中，每张参与 GPU 发送并接收 `LHb` 字节，两个 TP group 合计双向有效载荷为 `4LHb`。随后完整输出按 hidden 维重新取 shard：

```text
GPU0: Y_01[:,0:H/2],   shape=(L,H/2), logical token [0:L], hidden [0:H/2]
GPU1: Y_01[:,H/2:H],   shape=(L,H/2), logical token [0:L], hidden [H/2:H]
GPU2: Y_23[:,0:H/2],   shape=(L,H/2), logical token [L:S], hidden [0:H/2]
GPU3: Y_23[:,H/2:H],   shape=(L,H/2), logical token [L:S], hidden [H/2:H]
```

### 6.7 Attention residual，并将完整 token hidden 收集到 MoE owner

Residual add 先在 shard 所在 GPU 上进行：

```text
GPU0: R0_lo=X[0:L,   0:H/2] + Y_01[:,0:H/2], shape=(L,H/2)
GPU1: R0_hi=X[0:L, H/2:H] + Y_01[:,H/2:H], shape=(L,H/2)
GPU2: R1_lo=X[L:S,   0:H/2] + Y_23[:,0:H/2], shape=(L,H/2)
GPU3: R1_hi=X[L:S, H/2:H] + Y_23[:,H/2:H], shape=(L,H/2)
```

MoE 不做 TP，因此立即收集到 owner，而不是让两个 TP rank 都执行 Router：

```text
GPU1 -> GPU0: R0_hi, shape=(L,H/2)
GPU0: R0_full=concat(R0_lo,R0_hi), shape=(L,H), logical token [0:L], hidden [0:H]

GPU3 -> GPU2: R1_hi, shape=(L,H/2)
GPU2: R1_full=concat(R1_lo,R1_hi), shape=(L,H), logical token [L:S], hidden [0:H]
```

这是两个 TP group 内的单播（或仅 root 收集的 gather）：GPU1/GPU3 分别发送 `LHb/2`，GPU0/GPU2 分别接收 `LHb/2`，总有效载荷为 `LHb`。GPU1、GPU3 不再持有本轮 MoE source activation。

### 6.8 第二个 Norm 与 Router：完整 token hidden 只在 owner 上路由

GPU0、GPU2 都保存完整的第二个 Norm 参数 `gamma_2.shape=(H,)`（LayerNorm 时 `beta_2.shape=(H,)`），并各自在本地处理完整输入：

```text
GPU0: V0=Norm(R0_full), shape=(L,H), logical token [0:L]
GPU2: V1=Norm(R1_full), shape=(L,H), logical token [L:S]
```

没有通信，因为每个 token 的完整 hidden 已在 owner 上。Router 参数 `W_router.shape=(H,E)` 在四张 GPU 完整复制，但只有两个 owner 使用它：

```text
GPU0: logits0=V0@W_router, shape=(L,E), token [0:L]
GPU2: logits1=V1@W_router, shape=(L,E), token [L:S]

GPU0/GPU2: expert_id.shape=(L,K), router_weight.shape=(L,K)
```

top-k 沿每个 token 的 `E` 个 logits 独立执行，没有通信。每条 route 同时保存 `source_owner`、`source_token_index`、`expert_id` 和 `router_weight`，使专家输出可以精确返回原 token。

### 6.9 EP dispatch：完整 `(H,)` token 向量送往所属专家设备

第 `e` 个专家的参数均完整地位于唯一设备上：

```text
GPUd: expert e in [d*E/4, (d+1)*E/4)
      W_gate^e.shape=(H,I), W_up^e.shape=(H,I), W_down^e.shape=(I,H)
```

GPU0、GPU2 将各自 `V0/V1` 的每个 token 按其 `K` 条 route 复制并按目标 GPU 分桶。设 `n_{src->d}` 是 source owner `src` 发往 GPU `d` 的 route 数：

```text
GPU0 -> GPUd: send_hidden_0d.shape=(n_{0->d},H)
GPU2 -> GPUd: send_hidden_2d.shape=(n_{2->d},H)
```

每行对应一个完整 token hidden `(H,)` 和一条 route；旁路元数据包含同样长度的整数 index/expert id 与 `router_weight.shape=(n,)`。EP group 做 all-to-all：GPU0/GPU2 按上述分桶发送；GPU1/GPU3 没有 source token，但仍以零长度或 padding buffer 参与该 collective。

GPU `d` 收到的激活是按本地 expert 分组前的 ragged buffer：

```text
GPUd: U_d.shape=(N_d,H)
N_d=n_{0->d}+n_{2->d}
```

未发生 token dropping 时，两个 owner 总共发出 `K*S` 条 route，即有效激活载荷 `KSHb`；发往本 GPU 的 route 不经过互连，实际网络量为 `H*b` 乘以跨设备 route 数。接收端 `U_d` 不再对应 `(S,H)` 的连续切片，而是来自 token `[0:L]` 和 `[L:S]` 的混合副本。

### 6.10 本地专家计算：每个专家只处理自己的 route 子集

GPU `d` 按 expert id 划分 `U_d`。对于其本地专家 `e`，令 `N_e` 为被路由到该专家的 token 副本数：

```text
GPUd: U_e.shape=(N_e,H)
GPUd: G_e=SiLU(U_e@W_gate^e), shape=(N_e,I)
GPUd: A_e=U_e@W_up^e,           shape=(N_e,I)
GPUd: Y_e=(G_e*A_e)@W_down^e,   shape=(N_e,H)
```

`U_e` 的每一行保留其原始 token 位置和 route 元数据；它可能来自全局逻辑序列的任意位置。权重和激活同在 GPU `d`，专家内部没有 TP 通信。

### 6.11 EP combine：专家输出按原 route 返回并按 token 加权求和

每个 GPU 依 route 元数据将专家结果按目标 owner 重分桶：

```text
GPUd -> GPU0: return_d0.shape=(m_{d->0},H)  # 结果属于 token [0:L]
GPUd -> GPU2: return_d2.shape=(m_{d->2},H)  # 结果属于 token [L:S]
```

EP group 做第二次 all-to-all。未发生 dropping 时，GPU0 和 GPU2 分别收到 `L*K` 行 `(H,)` 的 route 输出：

```text
GPU0: returned_routes_0.shape=(L*K,H), source token [0:L]
GPU2: returned_routes_1.shape=(L*K,H), source token [L:S]
```

这次的有效激活载荷同样是 `KSHb`，实际网络量同样扣除 source 与 destination 相同的 route；还会随行返回 token index 和 router weight 等元数据。owner 在本地按 `source_token_index` scatter-add 并乘 Router 权重：

```text
GPU0: M0[t]=sum_{j=1..K} router_weight[t,j] * Y_{expert_id[t,j]}[t], shape=(L,H)
GPU2: M1[t]=sum_{j=1..K} router_weight[t,j] * Y_{expert_id[t,j]}[t], shape=(L,H)
```

`M0`、`M1` 重新恢复为连续序列切片；GPU0 对应全局逻辑输出 `[0:L,:]`，GPU2 对应 `[L:S,:]`。

### 6.12 MoE residual 与下一 block 的 TP hidden-shard 分发

owner 本地完成第二个 residual：

```text
GPU0: Z0=R0_full+M0, shape=(L,H), logical output Z[0:L,:]
GPU2: Z1=R1_full+M1, shape=(L,H), logical output Z[L:S,:]
```

下一 block 的第 6.1 节需要四卡 `(L,H/2)` 布局。因此完整输出按 hidden 维分发：

```text
GPU0 保留 Z0[:,0:H/2], shape=(L,H/2)  = Z[0:L,   0:H/2]
GPU0 -> GPU1: Z0[:,H/2:H], shape=(L,H/2) = Z[0:L, H/2:H]

GPU2 保留 Z1[:,0:H/2], shape=(L,H/2)  = Z[L:S,   0:H/2]
GPU2 -> GPU3: Z1[:,H/2:H], shape=(L,H/2) = Z[L:S, H/2:H]
```

GPU0、GPU2 各发送 `LHb/2`，GPU1、GPU3 各接收 `LHb/2`，总有效载荷为 `LHb`。这只是完整结果的 hidden 维 scatter，不是 reduce-scatter；通信完成后四卡的激活精确回到第 6.1 节所列的 `(S,H)` 四个逻辑切片。

### 6.13 vLLM 的实际 SP-MoE 路径：布局由 Attention 与 MoE 的边界决定

上面的 `TP=2、SP=2、EP=4` 是为了显式展示三条并行轴而构造的二维例子。vLLM 当前的 MoE sequence parallelism 不使用该独立 `SP×TP` 网格；它复用同一个 TP group，在 Attention 与 MoE 之间切换两种稳定布局：

```text
Attention 前：每个 TP rank 都有 (T,H) 的完整 token 集，QKV 按 head/feature 做 TP。
MoE 前后：  每个 TP rank 有 (ceil(T/TP),H) 的 token shard，hidden 完整。
```

这不是所有 `TP>1` 的部署都会启用的通用路径。当前 vLLM 的开关还要求启用 EP、`DP>1`，并选择支持的 all-to-all 后端。源码为：

```python
@property
def use_sequence_parallel_moe(self) -> bool:
    return (
        self.all2all_backend
        in (
            "allgather_reducescatter",
            "deepep_high_throughput",
            "deepep_low_latency",
            "mori_high_throughput",
            "mori_low_latency",
            "nixl_ep",
        )
        and self.enable_expert_parallel
        and self.tensor_parallel_size > 1
        and self.data_parallel_size > 1
    )
```

来源：`vllm/config/parallel.py` 的 `ParallelConfig.use_sequence_parallel_moe`。

下面以一个 DP replica 内 `TP=2`、本轮 token 数为 `T` 为例。上一层 MoE 输出保留为 token 分片时：

```text
GPU0: hidden_states/residual, shape=(T/2,H), token [0:T/2]
GPU1: hidden_states/residual, shape=(T/2,H), token [T/2:T]
```

vLLM 用当前 token 数是否小于 `positions` 中的完整 token 数来判断该状态：

```python
full_num_tokens = positions.shape[0]
input_is_sequence_parallel = (
    self.use_sequence_parallel_moe
    and residual is not None
    and hidden_states.shape[0] != full_num_tokens
)
```

输入已是 token shard 时，`input_layernorm` 先在本地运行。这里每个 token 的 `(H,)` 已完整在一张 GPU 上，RMSNorm 不需要跨 TP 归约；随后才对 Norm 后的激活沿 token 维 all-gather，以恢复 Attention 所需的复制布局：

```python
if residual is None:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
else:
    hidden_states, residual = self.input_layernorm(hidden_states, residual)

if input_is_sequence_parallel:
    hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
    hidden_states = hidden_states[:full_num_tokens]
```

对 `TP=2`，该 all-gather 对应：GPU0、GPU1 互相发送 `(T/2,H)`；之后两卡的 `hidden_states` 都是 `(T,H)`，而 `residual` 仍保留为各自 `(T/2,H)`。这是 vLLM 为 QKV 和 Attention 恢复完整 query token 集的唯一原因。

Attention 的 `o_proj` 是 row-parallel。普通路径会立即对每卡 `(T,H)` partial output 做 TP all-reduce：

```python
output_parallel = self.quant_method.apply(self, input_parallel, bias_)

if self.reduce_results and self.tp_size > 1:
    output = tensor_model_parallel_all_reduce(output_parallel)
else:
    output = output_parallel
```

但 SP-MoE 路径在构造 Attention 时关闭该 all-reduce：

```python
self.self_attn = attn_cls(
    ...,
    reduce_results=not self.use_sequence_parallel_moe,
)
```

因此，GPU0/GPU1 分别保留自己计算的 partial output `P0/P1.shape=(T,H)`，随后 decoder layer 将常规的“all-reduce 后两卡复制 `(T,H)`”改成一次沿 token 维的 reduce-scatter：

```python
if self.use_sequence_parallel_moe:
    tp_world_size = get_tensor_model_parallel_world_size()
    sp_pad = (-hidden_states.shape[0]) % tp_world_size
    hidden_states = torch.nn.functional.pad(
        hidden_states, (0, 0, 0, sp_pad)
    )
    hidden_states = tensor_model_parallel_reduce_scatter(
        hidden_states, 0
    )
    if not input_is_sequence_parallel:
        residual = sequence_parallel_chunk(residual)
```

`TP=2` 时这一句的数学含义是：

```text
GPU0: P0[0:T/2,:] + P1[0:T/2,:] -> hidden_states.shape=(T/2,H)
GPU1: P0[T/2:T,:] + P1[T/2:T,:] -> hidden_states.shape=(T/2,H)
```

即归约和 token 分片一次完成。若 `residual` 原来还是复制的 `(T,H)`，`sequence_parallel_chunk` 仅取出本 rank 对应的 token slice，使它与 reduce-scatter 的结果对齐；该函数实现是本地 `torch.narrow` 和必要的 padding，不产生通信：

```python
chunk = y.shape[0] // tp_size
start = tp_rank * chunk
out = torch.narrow(y, 0, start, chunk)
return out.clone() if y is x else out
```

此时 Post-Attention Norm 和 MoE 的输入为 `(T/2,H)`，完整 hidden 位于唯一 TP rank，所以每个 token 只进行一次 Router 和一次 EP dispatch。decoder layer 将该状态明确传给 MoE，避免 MoE 再次切分：

```python
hidden_states, residual = self.post_attention_layernorm(
    hidden_states, residual
)
if self.use_sequence_parallel_moe:
    hidden_states = self.mlp(
        hidden_states,
        already_sequence_parallel=True,
    )
else:
    hidden_states = self.mlp(hidden_states)
```

对应的 MoE 前向代码说明了这个标志的含义。只有输入尚为复制布局时，MoE 才自行 chunk，并在结束时 all-gather 回复制布局：

```python
if self.is_sequence_parallel and not already_sequence_parallel:
    hidden_states = sequence_parallel_chunk(hidden_states)

final_hidden_states = self.experts(
    hidden_states=hidden_states, router_logits=hidden_states
)

if self.is_sequence_parallel and not already_sequence_parallel:
    final_hidden_states = tensor_model_parallel_all_gather(
        final_hidden_states, 0
    )
    final_hidden_states = final_hidden_states[:num_tokens]
```

在 `DeepseekV2DecoderLayer` 的路径中，传入的是 `already_sequence_parallel=True`，因此 MoE 输出保持 `(T/2,H)`，直接跨越 layer boundary。下一层重新从本节的 `all_gather -> Attention -> reduce_scatter -> MoE` 循环开始。

若未启用 SP-MoE，FusedMoE 的最终输出会走传统 TP all-reduce，恢复为每个 TP rank 都有 `(T,H)` 的复制布局：

```python
if (
    not self.moe_config.is_sequence_parallel
    and not self.moe_config.skip_final_all_reduce
    and (self.moe_config.tp_size > 1 or self.moe_config.ep_size > 1)
    and not output_is_reduced
):
    states = tensor_model_parallel_all_reduce(states)
```

来源分别为本地 vLLM 的 `vllm/model_executor/models/deepseek_v2.py`、`vllm/model_executor/layers/linear.py`、`vllm/model_executor/models/utils.py` 和 `vllm/model_executor/layers/fused_moe/runner/moe_runner.py`。
