---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm
---

# Paged Attention

> 历史文档，基于 vLLM 原始论文。当前代码可能已不同。

## 核心思想

vLLM 的 paged attention kernel 将 KV cache 分成固定大小的 block，kernel 通过精心设计的内存布局和访问方式实现高性能的 multi-head query attention。

## 关键概念

| 概念 | 说明 |
|------|------|
| **Sequence** | 一个客户端请求，每个序列只有一个 query token |
| **Context** | 序列中已生成的 token |
| **Vec** | 一起获取和计算的元素列表，Q/K 的 VEC_SIZE 保证 16 bytes/tick，V 的 V_VEC_SIZE 保证 16 bytes/tick |
| **Thread Group** | 小线程组（THREAD_GROUP_SIZE），共同处理一个 query token 和一个 key token 的部分数据 |
| **Block** | KV cache 分块，每块存 BLOCK_SIZE 个 token 的一个 head 数据 |
| **Warp** | 32 个线程同时执行，处理一个 query token 与一个 block 所有 key token 的计算 |
| **Thread Block** | 多个 warp 组成，处理一个 query token 与整个 context 的计算 |
| **Grid** | 形状 `(num_heads, num_seqs, max_num_partitions)` |

## 数据布局

### Query

`q_ptr` 指向 `[num_seqs, num_heads, head_size]` 的 query 数据。每个线程组获取一个 query token，每个线程处理部分元素。读入 shared memory `q_vecs`，相邻线程读相邻地址实现内存合并。

### Key

`k_ptr` 指向 `[num_blocks, num_kv_heads, head_size/x, block_size, x]` 的 key cache。每个 warp 通过多次迭代处理多个 block 的 key token。读入寄存器 `k_vecs`，相邻线程读相邻 vec 实现内存合并。

### Value

V 的内存布局与 K 不同：同列对应同一个 value token。每个线程从同一组 token 的同一 head 位置获取 V_VEC_SIZE 个元素，通过多次内循环遍历不同 head 位置。

## 计算流程

### 1. QK 计算

```text
q_vecs = ...  # 从 global memory 读入 shared memory
for each block:
    for each vec in block:
        k_vecs[i] = ...  # 从 global memory 读入寄存器
    qk = scale * dot(q_vecs, k_vecs)  # 含跨线程组归约
```

### 2. Softmax

三个子步骤：
- **qk_max**：warp 内 shuffle 归约 → warp 间共享内存归约 → 全 thread block 最大值
- **exp_sum**：`logits[i] = exp(qk - qk_max)`，然后全 thread block 归约求和
- **归一化**：`logits[i] *= 1 / exp_sum`，得到最终 softmax 结果

### 3. LV 计算（Value 乘法）

```text
for each block:          # 外循环：不同 block
    logits_vec = ...     # 从 logits 取 V_VEC_SIZE 个元素
    for each row:        # 内循环：不同 head 位置
        v_vec = ...      # 从 value cache 取 V_VEC_SIZE 个元素
        accs[i] += dot(logits_vec, v_vec)
```

### 4. 归约与输出

- warp 内 shuffle 归约 `accs`
- warp 间共享内存归约
- 每个线程将自己负责的 head 位置结果写入 global memory

## 一句话总结

Paged Attention 通过将 KV cache 分块、thread group 协作处理、warp 内/warp 间两级归约，在分页 KV cache 上实现了高效的 attention 计算，是 vLLM 的核心内核之一。
