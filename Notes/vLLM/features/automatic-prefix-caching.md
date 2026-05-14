---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm
---

# Automatic Prefix Caching

## 概述

Automatic Prefix Caching (APC) 是 vLLM 的一项优化特性，通过缓存相同前缀的 KV Cache 块来避免重复计算，显著提升多轮对话和共享系统提示场景下的推理效率。

## 工作原理

### 基于哈希的块匹配

- 每个 KV Cache 块通过哈希值唯一标识
- 哈希由三部分组成：**父块哈希 + 块内 token 序列 + 额外哈希信息**
- 哈希匹配意味着两个块的 token 内容完全一致，可复用 KV Cache

### 缓存淘汰策略

- 采用 **LRU (Least Recently Used)** 策略
- 当显存不足时，最久未使用的缓存块被淘汰

## 使用方式

### 在线服务

```bash
vllm serve <model> --enable-prefix-caching
```

### 离线推理

```python
llm = LLM(model="...", enable_prefix_caching=True)
```

## 典型场景

1. **多轮对话**：前几轮的 KV Cache 可以复用
2. **共享系统提示**：多个请求共享相同的 system prompt
3. **少样本学习**：相同的 few-shot examples 前缀

## 注意事项

- 对于随机前缀或每次不同的输入，APC 没有效果
- 缓存命中率取决于输入的重叠程度
- 启用 APC 会略微增加内存开销（存储哈希元数据）
