---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm
_organized: true
---

# Automatic Prefix Caching

## 核心思想

缓存已处理请求的 KV cache block，当新请求有相同前缀时复用这些 block。vLLM 采用**基于哈希**的方案，只缓存**完整的 block**。

## 哈希方案

每个 block 的哈希由以下组成：
- **Parent hash**：父 block 的哈希值
- **Block tokens**：本 block 的 token tuple
- **Extra hashes**：LoRA ID、多模态输入哈希、cache salt 等

```text
Block 1: hash(block_tokens)
Block 2: hash(parent_hash + block_tokens)
Block 3: hash(parent_hash + block_tokens + extra_hashes)
```

哈希算法选项：
- `sha256`（默认）：用 pickle 序列化
- `sha256_cbor`：用 cbor2 序列化，跨语言可复现
- `xxhash`：更快但非密码学安全
- `xxhash_cbor`：可复现 + 快速

## 多模态输入支持

图像 token 被替换为 placeholder token `<P>`，每个 block 的 extra hash 中加入图像哈希，确保不同图像的 placeholder 不会误命中。

## Cache Salt（安全隔离）

通过 `cache_salt` 注入第一个 block 的哈希，确保只有相同 salt 的请求才能复用缓存，防止时序攻击。

## 数据结构

```python
class KVCacheBlock:
    block_id: int          # 不可变
    block_hash: BlockHash  # block 满时赋值，驱逐时重置
    ref_cnt: int           # 引用计数
    prev_free_block / next_free_block  # 双向链表指针
```

设计要点：
- 初始化时预分配所有 KVCacheBlock（block pool），避免 Python 对象创建开销
- 双向链表指针直接嵌入 block，实现 O(1) 的中间元素移动

组件：Block Pool、Free Block Queue（双向链表）、Cache Blocks（hash → block_id）、Request Blocks（request_id → block_id）。

## 操作流程

### Block 分配（新请求）

1. `get_computed_blocks()`：哈希 prompt tokens，查找已缓存 block
2. `allocate_slots()`：
   - 计算所需新 block 数
   - "Touch" 已缓存 block（增加引用计数，从 free queue 移除）
   - 从 free queue 头部弹出新 block（如是 cached block 则驱逐）
   - 已满的 block 立即加入 cache

### Block 分配（运行中请求）

1. 计算所需新 block 数
2. 从 free queue 头部弹出新 block
3. 追加 token 到已有 block 和新 block，满时加入 cache

### 重复 Block

vLLM v1 的 block table 是 append-only 的，无法将 `[0, 3]` 改为 `[0, 1]`。因此同一内容可能被缓存两次，请求释放时消除重复。

### Free

请求完成时释放所有 block（ref_cnt = 0）。释放的 block 按**逆序**加入 free queue 尾部（最后的 block 哈希更多 token，更不可能被复用，应优先驱逐）。

### Eviction（LRU）

当 free queue 头部是 cached block 时驱逐：弹出 → 从 cache blocks 移除 → 清除 block hash。

## 完整示例

以 block_size=4，10 个 block 为例：

1. **Time 1**：空缓存，新请求 ABCDEF → 分配 4 block，3 个已满并缓存
2. **Time 2**：block 3 满，缓存并分配 block 4
3. **Time 3**：新请求前 10 token 相同 → 前 2 block（8 token）命中缓存
4. **Time 4**：请求 0 完成释放，block 2,3,4 按逆序加入 free queue
5. **Time 5**：请求 1 完成释放
6. **Time 6**：新请求前 12 token 相同 → 3 个 cached block 被 touch 并从 free queue 移除，分配 3 个 cached + 5 个新 block（含 1 个被驱逐的）

## 一句话总结

Prefix caching 通过哈希（parent hash + block tokens + extra hashes）唯一标识每个 KV cache block，配合双向链表 free queue 实现 LRU 驱逐，让相同前缀的请求自动复用已计算的 KV cache。
