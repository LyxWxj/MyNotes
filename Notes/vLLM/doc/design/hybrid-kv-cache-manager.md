---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm
_organized: true
---

# Hybrid KV Cache Manager

## 什么是 Hybrid 模型？

近年来很多 LLM 在同一个模型中混合了**多种注意力类型**：

| 类型 | 示例模型 |
|------|----------|
| 滑动窗口注意力 (SW) + 全注意力 (Full) | GPT-OSS, Gemma 2/3, Ministral, Cohere |
| Mamba + 全注意力 | Bamba, Jamba, Minimax |
| 局部块注意力 + 全注意力 | Llama 4 |

核心挑战：不同注意力类型的层需要**不同数量的 KV cache slot** — 全注意力层需要所有 token 的 slot，滑动窗口层只需要最近 `sliding_window_size` 个 token 的 slot。

## 基础概念

- **kv hidden size**：单层单 token 的 KV cache 字节数
- **block**：KV cache 内存被划分为等大的 block
- **block size**：一个 block 能容纳的 token 数
- **page size**：一个 block 的物理内存大小 = `num_layers x block_size x kv_hidden_size`

注意：这里的 `num_layers` 不是模型总层数，而是**一个 group 中的层数**。

## 内存分配策略

### 核心思路

所有层类型共用**一个内存池**，page size 统一。`KVCacheManager` 根据注意力类型给不同层分配不同数量的 block。

### Case 1：简单模型

全 attention + 滑动窗口，同 kv_hidden_size。page size = `kv_hidden_size x block_size`，每层独立分配 block。

### Case 2：有规律的层比例

当不同注意力类型的层数有**整数比**关系时，按比例分组减少分配次数：

- Gemma-2：1 sw : 1 full
- Llama 4：3 local : 1 full

例如 20 sw + 10 full，比例 2:1，分为 3 个 group，page size = `10 x kv_hidden_size x block_size`。分配结果只需计算一次，重复 10 次即可。

### Case 3：无规律的层比例

如 Gemma-3-27b：52 sw + 10 full，比例不整齐。策略是用**最小层类型数**（10）作为 group size：

- Group 0：10 full（full.0-9）
- Group 1：10 sw（sw.0-9）
- Group 2：10 sw（sw.10-19）
- ...
- Group 6：10 sw（sw.40-49）
- Group 7：2 sw + **8 padding**（填充层，浪费少量内存）

### Case 4：不同 kv_hidden_size（Mamba 模型）

Mamba 层的 state size 远大于 attention 层的 kv_hidden_size。解决方案：

1. 增大 attention 层的 `block_size`，直到 `block_size x kv_hidden_size_att >= state_size_mamba`
2. 将 Mamba state padding 到 `block_size x kv_hidden_size_att`
3. 再用 Case 3 的分组策略

缺点：可能导致 block_size 超过 400，过大。

### Case 5：KV Sharing

某些层（如 gemma-3n）复用其他层的 KV cache。`KVCacheManager` 跳过这些层，只分配需要 KV cache 的层，然后在 model runner 中将分配结果映射到共享层。

## Prefix Caching（前缀缓存）

假设 `block_size=1` 来简化说明。

### 全注意力

从**左到右**扫描，找到最长的连续缓存命中前缀，遇到 miss 即停止。

### 滑动窗口注意力

cache hit 条件更宽松：只要求最后 `sliding_window_size - 1` 个 token 被缓存。从**右到左**扫描，找到匹配即停止。

例如 `sliding_window_size=4`，15 token 的 prompt，缓存了 token [0,1,2,5,6,11,12,13]，最长 cache hit = 14（因为 token 11,12,13 被缓存），只需计算 [14]。

### 混合模型（Full + X）

算法：
1. **从左到右**扫描全注意力的最长 cache hit
2. **从右到左**扫描滑动窗口注意力的 cache hit（起点为全注意力的 cache hit 长度）
3. 两者**交集**即为最终 cache hit

这比"分别找所有可能前缀再取交集"更高效，因为可以提前退出。

**限制**：目前只支持恰好两种注意力类型（Full + X），不支持无 Full 层或超过 2 种类型。

**淘汰策略**：所有 group 共用一个 LRU 队列，block 被释放时（请求完成或超出滑动窗口）加入队列。

## 实现架构

三层结构：

```text
KVCacheManager          ← 调度器接口
    |
KVCacheCoordinator      ← 协调多个 group 的分配
    |-- KVCacheCoordinatorNoPrefixCache  ← 无 prefix cache
    |-- UnitaryKVCacheCoordinator        ← 单 group（无需交集）
    +-- HybridKVCacheCoordinator         ← 双 group（Full + X）
    |
SingleTypeKVCacheManager  ← 每个 group 一个实例
    |-- FullAttentionManager
    +-- SlidingWindowManager
```

## 物理内存布局

以 10 full + 20 sw（分为 3 group）为例：

- 物理内存分为 **10 个 buffer**（`KVCacheTensor` 0-9）
- 每个 buffer 被 **3 个 group 中各一层共享**
- 每个 buffer 按 `block_size x kv_hidden_size` 切片

```text
KVCacheTensor 0: full.0 | sw.0 | sw.10  共享
KVCacheTensor 1: full.1 | sw.1 | sw.11  共享
...
KVCacheTensor 9: full.9 | sw.9 | sw.19  共享
```

一个逻辑 "block" 映射到 10 个 buffer 中的 10 个物理切片。请求被分配 `block_id` 0-6 给 group 0，7-8 给 group 1，9-10 给 group 2，各层根据自己的 group 找到对应的物理位置。

## 一句话总结

Hybrid KV Cache Manager 通过**统一分组 + 统一 page size** 的设计，让一个内存池同时服务全注意力、滑动窗口、Mamba 等不同类型的层，并通过 group 间的 prefix cache 交集算法实现高效的前缀缓存复用。
