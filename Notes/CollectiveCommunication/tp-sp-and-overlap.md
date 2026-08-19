---
type: Note
related_to: "[[collective-communication-index]]"
status: Active
---

# PyTorch NCCL：4 卡 TP=2、SP=2 与计算通信重叠

本文以 4 个 GPU 进程为例，说明如何创建张量并行（Tensor Parallel, TP）与序列并行（Sequence Parallel, SP）通信组、一个 Transformer 层的数据流，以及如何把矩阵乘与 NCCL 通信重叠起来。

> [!IMPORTANT]
> “SP=2”有两种常见含义，不能混用：
>
> 1. **独立二维 mesh**：TP 和 SP 是两个正交维度，4 卡可表示为 `TP=2 x SP=2`。本文的主例子采用这一含义。
> 2. **Megatron 风格 SP**：SP 附着在 TP 组上，SP size 等于 TP size，并没有再多乘一维 GPU。vLLM 当前 DeepSeek V3.2 MoE 的实现属于此类：其 `sp_all_gather` 和 `sp_reduce_scatter` 直接调用 TP group 的 collective。

## 1. 4 卡的二维 TP x SP 拓扑

令全局 rank 为 `rank = sp_rank * TP + tp_rank`：

```text
                 tp_rank=0       tp_rank=1
sp_rank=0         rank 0           rank 1
sp_rank=1         rank 2           rank 3

TP groups: [0, 1], [2, 3]
SP groups: [0, 2], [1, 3]
```

- **TP 组**：同一段 token，拆不同 attention heads / 权重分片；需要在层内交换或归约张量。
- **SP 组**：同一 TP 权重分片，拆不同 token 段；需要为 attention 构造跨段上下文，或做 token/head 转置。

下面的初始化示例使用 `torchrun` 的 `env://` 环境变量。所有 rank 必须以相同顺序调用 `new_group`，否则 NCCL communicator 的创建序列不一致，任务可能挂起。

```python
import os
import torch
import torch.distributed as dist


def init_tp_sp():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", init_method="env://")

    rank = dist.get_rank()
    assert dist.get_world_size() == 4

    tp_ranks = [[0, 1], [2, 3]]
    sp_ranks = [[0, 2], [1, 3]]

    # 所有 rank 都执行全部循环，且顺序完全相同。
    tp_groups = [dist.new_group(ranks=ranks, backend="nccl") for ranks in tp_ranks]
    sp_groups = [dist.new_group(ranks=ranks, backend="nccl") for ranks in sp_ranks]

    sp_rank, tp_rank = divmod(rank, 2)
    return tp_groups[sp_rank], sp_groups[tp_rank], tp_rank, sp_rank
```

启动命令：

```bash
torchrun --standalone --nproc_per_node=4 tp_sp_demo.py
```

## 2. 一个 Transformer block 的数据流

设 prefill 处理 `S=8192` 个 token、hidden size `H=4096`、32 个 attention heads、bf16。SP 将 token 沿序列维二分，因此每个 SP row 持有 `S/SP=4096` 个 token：

```text
rank 0 / rank 1: X0 = tokens [0:4096],    shape [4096, 4096]
rank 2 / rank 3: X1 = tokens [4096:8192], shape [4096, 4096]
```

在每一个 SP row 中，TP 再沿 attention head / 模型权重维度切分：

```text
TP rank 0: QKV heads [0:16]，O projection 的输入列 [0:2048]
TP rank 1: QKV heads [16:32]，O projection 的输入列 [2048:4096]
```

以最容易理解的“先 gather K/V”实现为例：

```python
# x: [S / SP, H]。同一 SP row 内的两个 TP rank 都有相同 token shard。
# wqkv_local: [3 * H / TP, H]，每个 TP rank 只持有自己的输出列。
qkv_local = torch.nn.functional.linear(x, wqkv_local)
q, k, v = split_local_qkv(qkv_local)
# q/k/v: [S / SP, num_heads / TP, head_dim]

# 每个 TP rank 的 attention heads 都需要完整序列的 K/V。
k_full = torch.empty((S, num_heads // 2, head_dim), dtype=k.dtype, device=k.device)
v_full = torch.empty_like(k_full)
dist.all_gather_into_tensor(k_full, k, group=sp_group)
dist.all_gather_into_tensor(v_full, v, group=sp_group)

attn_local = flash_attention(q, k_full, v_full)
# attn_local: [S / SP, H / TP]

# Row-parallel O projection：每个 TP rank 得到 H 维 partial result。
partial = torch.nn.functional.linear(attn_local, wo_local)
# partial: [S / SP, H]

# TP 归约后，每张卡都得到本 SP shard 的完整 hidden states。
dist.all_reduce(partial, op=dist.ReduceOp.SUM, group=tp_group)
h = residual + partial
```

MLP 的通信模式相同：column-parallel `W1` 和激活不通信，row-parallel `W2` 后执行一次 TP `all_reduce`。token shard 可保留在本地，直到模型输出阶段或某个需要全序列张量的算子。

| 阶段 | 通信组 | 原语 |
|---|---|---|
| Attention 跨 sequence K/V 上下文 | SP | `all_gather` |
| Attention O projection | TP | `all_reduce` |
| MLP W2 projection | TP | `all_reduce` |
| 最终 logits 或完整 sequence 输出 | SP | `all_gather` |

> [!NOTE]
> 对超长上下文，直接 all-gather 完整 K/V 会造成高峰值显存。真实实现通常会改为 Ring Attention 的分块 P2P 环传，或 Ulysses 的 `all_to_all` 进行 sequence/head 维度转置。

## 3. vLLM 中的 SP：TP 组内 token 分片

当前 vLLM 的 `sequence_parallel.py` 用 TP group 实现 SP：

- `sp_shard` 沿第 0 维按 TP rank 切 token，必要时 padding；
- `sp_all_gather` 在 TP group 上沿第 0 维恢复完整 token；
- `sp_reduce_scatter` 在 TP group 上沿第 0 维求和后重新分片。

DeepSeek V3.2 NVIDIA 实现的层内顺序为：

```text
local token shard
  -> RMSNorm
  -> SP all-gather
  -> attention（输出未归约）
  -> SP reduce-scatter
  -> RMSNorm
  -> MLP / MoE（保持本地 token shard）
```

该路径的目的不是建立独立二维 TP x SP mesh，而是让后续 Norm 和 MoE 在 local token shard 上计算，减少激活驻留。代码还明确限制：SP 暂不支持与 PP 一起使用。

相关源码：

- [[vLLM-omni/docs/vllm_distributed|vLLM 分布式说明]]
- `vllm/models/common/ops/sequence_parallel.py`
- `vllm/models/deepseek_v32/nvidia/model.py`

## 4. 计算与通信如何真正重叠

仅仅写 `async_op=True` 不保证性能重叠。若通信结果立刻被消费，后续计算仍会等待它。可重叠的基本条件是：通信和另一个计算块之间没有数据依赖，或能将张量按 token 维分块，形成流水。

以 row-parallel MLP 的 `W2` 为例。每个 TP rank 先计算一个 partial output，随后必须 all-reduce 才能进行 residual 和 RMSNorm。若把 token 切为多个 chunk：

```text
chunk 0: GEMM(0) -> all-reduce(0) -> RMSNorm(0)
chunk 1: GEMM(1) -> all-reduce(1) -> RMSNorm(1)
```

则 `all-reduce(0)` 可与 `GEMM(1)` 并发。示例只用 CUDA Event 约束 GEMM producer 与提交通信的 stream；真正消费 collective 输出前，调用 `Work.wait()`，由 PyTorch 建立当前计算 stream 对 ProcessGroupNCCL 内部通信 stream 的正确依赖。

```python
import torch
import torch.distributed as dist
import torch.nn.functional as F


def row_parallel_linear_overlap(
    x_local, w_local, residual, tp_group, chunk_tokens=1024
):
    compute_stream = torch.cuda.current_stream()
    comm_stream = torch.cuda.Stream(device=x_local.device)
    pending = []
    outputs = []

    for begin in range(0, x_local.shape[0], chunk_tokens):
        end = min(begin + chunk_tokens, x_local.shape[0])

        # 当前块的 local GEMM 在 compute stream 上运行。
        with torch.cuda.stream(compute_stream):
            partial = F.linear(x_local[begin:end], w_local)
            gemm_done = torch.cuda.Event()
            gemm_done.record(compute_stream)

        # NCCL 等待 GEMM 的输入 ready 后，异步提交 all-reduce。
        with torch.cuda.stream(comm_stream):
            comm_stream.wait_event(gemm_done)
            work = dist.all_reduce(partial, group=tp_group, async_op=True)
        pending.append((work, partial, residual[begin:end]))

        # 消费前一块：其 AR 与当前块 GEMM 在不同 stream 上重叠。
        if len(pending) > 1:
            old_work, old_partial, old_residual = pending[-2]
            with torch.cuda.stream(compute_stream):
                # work.wait() 将当前 stream 与 NCCL 的内部 stream 正确同步。
                old_work.wait()
                outputs.append(rmsnorm(old_residual + old_partial))

    # drain 最后一个通信块。
    work, partial, residual_chunk = pending[-1]
    with torch.cuda.stream(compute_stream):
        work.wait()
        outputs.append(rmsnorm(residual_chunk + partial))

    return torch.cat(outputs, dim=0)
```

对应的理想时间线：

```text
compute stream: GEMM(0) -- GEMM(1) -- Work(0).wait, Norm(0) -- GEMM(2) -- ...
NCCL stream:                  AR(0) ---------- AR(1) --------------- AR(2) -- ...
```

如果 `T_gemm`、`T_ar`、`T_norm` 分别是每个 chunk 的 GEMM、all-reduce、Norm 时间，串行执行约为：

```text
T_gemm + T_ar + T_norm
```

稳态流水后约为：

```text
max(T_gemm, T_ar) + T_norm
```

实际加速不一定达到该上界：GEMM 与 NCCL kernel 会竞争 SM、HBM、NVLink/PCIe DMA 等资源；chunk 太小会增大 launch 与通信延迟，太大又会降低流水粒度。应通过 Nsight Systems 检查 GPU stream 的实际空洞，而不是只看 Python 计时。

## 5. 与 vLLM DBO 的关系

vLLM 的 DBO（dual batch overlap）将一个请求批拆为多个 microbatch，并使用独立的 `comm_stream` 与 `compute_stream`。它通过 CUDA Event：

```text
compute stream record event -> comm stream wait event
comm stream record event    -> compute stream wait event
```

在不破坏数据依赖的地方，让一个 microbatch 的通信与另一个 microbatch 的计算重叠。vLLM 还可为特定通信后端保留部分 SM，降低通信和计算争抢执行资源的程度。它说明 overlap 是一套调度、stream、event 和资源划分的设计，并非给 collective 增加 `async_op=True` 就会自然获得的效果。

## 参考

- PyTorch Distributed 文档：https://docs.pytorch.org/docs/stable/distributed.html
- [[collective-communication-basics|集合通信基础（原语 / 算法 / 拓扑）]]
- [[nccl-tuning-and-debugging|NCCL 调优与问题排查]]
