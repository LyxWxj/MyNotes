---
type: Note
related_to: "[[vLLM-omni]]"
status: Active
url: https://github.com/vllm-project/vllm-omni/blob/main/docs/design/feature/cache_dit.md
---

# Cache-DiT

Cache-DiT是一个Diffusion Transformer（DiT）加速库，通过缓存去噪步骤之间的中间计算结果来加速推理。

## 核心原理

相邻去噪步骤通常会产生相似的中间特征，因此可以通过重用缓存结果来跳过冗余计算。

## 缓存策略

| 策略 | 描述 |
|------|------|
| **DBCache** | 动态块级缓存，基于残差差异选择性计算或缓存transformer块 |
| **TaylorSeer** | 基于校准的预测，使用泰勒展开估计块输出 |
| **SCM** | 动态步骤跳过，基于可配置策略 |

## 架构

vLLM-omni通过`CacheDiTBackend`类集成cache-dit：

| 方法/类 | 用途 |
|---------|------|
| `CacheDiTBackend` | 统一后端接口 |
| `enable_cache_for_dit()` | 将缓存应用到transformer |

### Cache-DiT核心API

| API | 描述 |
|-----|------|
| `BlockAdapter` | 将cache-dit应用到transformer的核心抽象 |
| `ForwardPattern` | 定义块前向签名模式：Pattern_0、Pattern_1、Pattern_2 |
| `ParamsModifier` | 每个transformer或块列表的缓存配置自定义 |
| `DBCacheConfig` | DBCache参数配置 |
| `refresh_context()` | 更新缓存上下文 |

## 标准模型自动支持

大多数DiT模型遵循标准模式（单transformer、单ModuleList块列表），无需代码修改：

```python
from vllm_omni import Omni

omni = Omni(
    model="Qwen/Qwen-Image",
    cache_backend="cache_dit",
    cache_config={
        "Fn_compute_blocks": 1,
        "Bn_compute_blocks": 0,
        "max_warmup_steps": 4,
    }
)
```

## 自定义架构

需要自定义处理的模型类型：
- **单/双transformer**：如Wan2.2
- **多块列表**：如LongCatImage（`transformer_blocks` + `single_transformer_blocks`）
- **特殊前向模式**：非标准块执行模式

### 示例：双transformer模型（Wan2.2）

```python
cache_dit.enable_cache(
    BlockAdapter(
        transformer=[pipeline.transformer, pipeline.transformer_2],
        blocks=[pipeline.transformer.blocks, pipeline.transformer_2.blocks],
        forward_pattern=[ForwardPattern.Pattern_2, ForwardPattern.Pattern_2],
        params_modifiers=[...],
    ),
    cache_config=db_cache_config,
)
```

### 注册自定义实现

在`vllm_omni/diffusion/cache/cache_dit_backend.py`中注册：

```python
CUSTOM_DIT_ENABLERS = {
    "Wan22Pipeline": enable_cache_for_wan22,
    "LongCatImagePipeline": enable_cache_for_longcat_image,
    "YourCustomPipeline": enable_cache_for_your_model,
}
```

## 测试验证

1. 缓存是否应用（检查日志）
2. 性能提升（预期1.5x-2x加速）
3. 图像质量（与`cache_backend=None`对比）

## 故障排除

### 缓存未应用
- 检查pipeline名称是否在`CUSTOM_DIT_ENABLERS`注册表中

### 质量下降
- 降低`residual_diff_threshold`（从0.24降至0.12-0.18）
- 增加`max_warmup_steps`（从4增至6-8）

## 参考实现

| 模型 | 模式 | 备注 |
|------|------|------|
| 标准DiT | 默认enabler | 单transformer，自动 |
| Wan2.2 | 单/双transformer | 自动检测模式 |
| LongCat | 多块列表 | 单transformer中两个块列表 |
| BAGEL | 全模态模型 | 复杂架构 |
