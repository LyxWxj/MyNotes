---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm
---

# torch.compile with Multimodal Encoders

## 概述

`torch.compile` 现可应用于多模态编码器（如 LLaMA 4、Qwen-VL 的视觉编码器），带来额外的性能提升。默认关闭，通过 `compile_mm_encoder: true` 启用。

Qwen2.5-VL 视觉 block 应用后观察到 ~4.5% 端到端性能提升。

## 启用方式

在 `@support_torch_compile` 装饰器上添加：
- `enable_if=should_torch_compile_mm_encoder`：受 `compile_mm_encoder` 配置控制
- `is_encoder=True`：标记为编码器组件，用于 compile range 集成，避免与 text backbone 缓存目录冲突

```python
@support_torch_compile(enable_if=should_torch_compile_mm_encoder, is_encoder=True)
class VisionEncoder(nn.Module):
    ...
```

## Compile Range

编码器的输入形状范围难以推断（不像 text backbone 有 max_batch_size）。`is_encoder=True` 告知 torch.compile 默认使用范围 `(1, MAX_INT)`。

## CUDAGraph

多模态编码器的 CUDAGraph 集成**尚未探索**，行为未定义。

## 常见问题排查

### Graph Breaks

原因：
- 动态图像大小 → 用 `dynamic_shapes_config` 处理
- 不可追踪操作（如 `to_list`）→ Dynamo 不支持
- 基于图像属性的条件处理

```bash
TORCH_LOGS="+dynamo" vllm serve <MODEL>
```

### 编译失败

1. 先禁用编译验证模型可用：
   ```bash
   VLLM_TORCH_COMPILE_LEVEL=0 vllm serve <model> --compilation-config='{"compile_mm_encoder":"false"}'
   ```
2. 开启调试日志：
   ```bash
   VLLM_LOGGING_LEVEL=DEBUG vllm serve <model> --compilation-config='{"compile_mm_encoder":"true"}'
   ```

## 一句话总结

torch.compile 对多模态编码器的支持通过 `@support_torch_compile` 装饰器的 `is_encoder=True` 参数实现，独立于 text backbone 编译，可带来额外的端到端性能提升，但 CUDAGraph 集成尚不成熟。
