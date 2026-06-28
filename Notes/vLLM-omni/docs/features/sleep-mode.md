---
type: Note
related_to: "[[vllm-omni]]"
status: Active
url: https://github.com/vllm-project/vllm-omni/blob/main/docs/features/sleep_mode.md
---

# Sleep Mode（睡眠模式）

vLLM-Omni的**睡眠模式**允许临时释放模型使用的大部分GPU内存（如模型权重和键值缓存），**无需停止服务器或卸载Docker容器**。

此功能继承自[vLLM的睡眠模式](https://blog.vllm.ai/2025/10/26/sleep-mode.html)，为多模型服务提供零重载模型切换。

特别适用于**RLHF**、**训练**或**节省成本**场景，需要在推理工作负载之间释放GPU资源。

## Omni模型

Omni模型继承vLLM睡眠模式的功能：
- 支持Level 1和Level 2睡眠
- 允许释放和重置模型权重和KV缓存

## Diffusion模型扩展

为**扩散模型**添加了睡眠模式支持，此前缺乏此功能。在扩散管道中，当前仅卸载**模型权重内存**，因为这些模型通常不使用KV缓存。

**特性**：
- 扩散模型现在可以进入Level 1睡眠
- 管道状态（如噪声调度器、缓冲区）在唤醒后保持完整
- 适用于在图像生成或训练周期之间释放VRAM

## 启用睡眠模式

在`engine_args`中设置`enable_sleep_mode`为`True`：

```python
omni = Omni(model=..., enable_sleep_mode=True)
```

## 使用场景

| 场景 | 描述 |
|------|------|
| **RLHF** | 在训练和推理之间切换时释放GPU内存 |
| **训练** | 在训练周期之间释放资源 |
| **成本节省** | 在不使用时释放GPU资源 |
| **多模型服务** | 在不同模型之间快速切换 |

## 睡眠级别

| 级别 | 描述 | Omni模型 | Diffusion模型 |
|------|------|----------|---------------|
| **Level 1** | 释放模型权重 | ✅ | ✅ |
| **Level 2** | 释放模型权重和KV缓存 | ✅ | ❌（无KV缓存） |

## 相关链接

- [vLLM Sleep Mode博客](https://blog.vllm.ai/2025/10/26/sleep-mode.html)
