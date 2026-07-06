---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm
_organized: true
---

# torch.compile Integration

## 概述

vLLM V1 默认启用 `torch.compile`，是框架的关键组件。所有编译在服务请求前完成，不会因请求触发新编译。

## 编译缓存

缓存目录考虑所有相关因素（配置、PyTorch 配置、模型 forward 函数等），保证缓存安全。可直接复制 `~/.cache/vllm/torch_compile_cache` 目录加速部署启动。

- 禁用：`VLLM_DISABLE_COMPILE_CACHE=1`
- 调试格式：`compile_cache_save_format=unpacked`

## Dynamic Shapes

三种模式：

| 模式 | 说明 |
|------|------|
| **BACKED**（默认） | 接受潜在不安全的 guard 丢弃以获得最大性能 |
| **UNBACKED** | 最强的无 guard 保证，最保守，可能错过优化 |
| **BACKED_SIZE_OBLIVIOUS** | 实验性，比 BACKED 安全，比 UNBACKED 性能好 |

配置方式：
```python
# Python
CompilationConfig(dynamic_shapes_config=DynamicShapesConfig(type=DynamicShapesType.UNBACKED))
# CLI
vllm serve model -cc.dynamic_shapes_config.type=unbacked
```

## 编译流程

### 1. Dynamo 图捕获

Dynamo 追踪模型的 `forward` 函数及其调用的所有函数（PyTorch nn.Module、vLLM 的通信/attention/激活函数等），生成：
- `transformed_code.py`：转换后的函数
- `computation_graph.py`：计算图

### 2. 计算图处理

- 输入：input_ids、position_ids（动态形状）、模型权重/buffer（静态形状）
- 唯一变化维度：batch size
- Attention 操作被封装为 `torch.ops.vllm.unified_attention_with_output` 自定义 op，Dynamo 不深入其内部
- 图按 `splitting_ops`（通常是 attention）切分为子模块：attention 前、attention 间、attention 后

### 3. Inductor 编译

每个子图由 Inductor 编译，生成优化的 kernel 代码。3 个唯一子图：
- 第一层 attention 前
- 中间层（attention 间）
- 最后一层 attention 后

可为特定 batch size 编译并开启 auto-tuning：
```bash
vllm serve model --compilation-config '{"compile_sizes": [1, 2, 4, 8]}'
```

## CUDAGraph Capture

V1 使用 **piecewise cudagraph**，与 piecewise compilation 对齐：
- 只捕获 attention 之间的计算图（含首尾）为 CUDA graph
- Attention 操作在 eager 模式运行（灵活处理复杂 KV cache 交互）
- 精细内存管理：attention kernel 排除在 cudagraph 外，其余模块和内存分配在 cudagraph 内
- Attention 的输出 tensor 作为输入传入（以便 cudagraph 管理中间 buffer）

可指定捕获大小：
```bash
vllm serve model --compilation-config '{"cudagraph_capture_sizes": [1, 2, 4, 8]}'
```

也支持 **Full CUDAGraph**（包含 attention），适用于 cudagraph 兼容的 attention backend，可提升小模型或 MoE 的 decode 速度。

## 一句话总结

torch.compile 在 vLLM 中通过 Dynamo 图捕获 → 图按 attention 切分 → Inductor 编译 → piecewise cudagraph 的流程，在请求服务前完成所有编译，通过缓存复用避免重复编译，实现高性能推理。
