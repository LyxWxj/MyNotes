---
type: Note
related_to: "[[vLLM-omni]]"
status: Active
url: https://github.com/vllm-project/vllm-omni/blob/main/docs/design/feature/teacache.md
---

# TeaCache

## 概述

### 什么是TeaCache？

TeaCache通过在连续时间步相似时缓存transformer块计算来加速扩散推理，提供**1.5x-2.0x加速**且质量损失最小。

核心原理：调制输入（归一化和时间步调节后）在时间步之间逐渐变化。通过测量连续调制输入之间的L1距离并与阈值比较，TeaCache决定是执行完整的transformer块还是重用上一步的缓存残差。

vLLM-omni提供**基于钩子**的TeaCache系统，**无需修改模型代码**。钩子完全拦截transformer的前向传播并透明实现自适应缓存。

### 架构

| 组件 | 用途 | 位置 |
|------|------|------|
| `CacheContext` | 包含缓存模型特定信息的数据类 | `vllm_omni/diffusion/cache/teacache/context.py` |
| `EXTRACTOR_REGISTRY` | 将transformer类名映射到提取器函数 | `vllm_omni/diffusion/cache/teacache/extractors.py` |
| `TeaCacheConfig` | 包括阈值和多项式系数的配置 | `vllm_omni/diffusion/cache/teacache/config.py` |

钩子自动处理所有缓存逻辑：
- CFG感知状态管理（正/负分支的独立状态）
- CFG并行兼容性
- L1距离计算与多项式重缩放
- 残差缓存与重用

## 实现步骤

### 步骤1：模型特定预处理

```python
def extract_qwen_context(
    module: nn.Module,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_mask: torch.Tensor,
    timestep: torch.Tensor,
    img_shapes: torch.Tensor,
    txt_seq_lens: torch.Tensor,
    guidance: torch.Tensor | None = None,
    **kwargs: Any,
) -> CacheContext:
    # 预处理：嵌入输入
    hidden_states = module.img_in(hidden_states)
    timestep = timestep.to(device=hidden_states.device, dtype=hidden_states.dtype)
    encoder_hidden_states = module.txt_norm(encoder_hidden_states)
    encoder_hidden_states = module.txt_in(encoder_hidden_states)

    # 创建时间步嵌入
    temb = module.time_text_embed(timestep, hidden_states)

    # 计算位置嵌入
    image_rotary_emb = module.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)
```

### 步骤2：提取调制输入

从**第一个transformer块**提取调制输入，用于缓存决策：

```python
    block = module.transformer_blocks[0]
    img_mod_params = block.img_mod(temb)
    img_mod1, _ = img_mod_params.chunk(2, dim=-1)
    img_modulated, _ = block.img_norm1(hidden_states, img_mod1)
```

### 步骤3：定义transformer执行

```python
    def run_transformer_blocks():
        """执行所有Qwen transformer块。"""
        h = hidden_states
        e = encoder_hidden_states

        for block in module.transformer_blocks:
            e, h = block(
                hidden_states=h,
                encoder_hidden_states=e,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )
        return (h, e)  # 返回图像和文本隐藏状态
```

**返回格式**：
- 单流模型：`(hidden_states,)`
- 双流模型：`(hidden_states, encoder_hidden_states)`

### 步骤4：定义后处理

```python
    def postprocess(h):
        """应用Qwen特定输出后处理。"""
        h = module.norm_out(h, temb)
        output = module.proj_out(h)
        return Transformer2DModelOutput(sample=output)
```

### 步骤5：返回CacheContext

```python
    return CacheContext(
        modulated_input=img_modulated,
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,  # 单流为None
        temb=temb,
        run_transformer_blocks=run_transformer_blocks,
        postprocess=postprocess,
    )
```

**CacheContext字段**：

| 字段 | 类型 | 用途 |
|------|------|------|
| `modulated_input` | `torch.Tensor` | 用于缓存决策的张量（相似性比较） |
| `hidden_states` | `torch.Tensor` | 当前隐藏状态 |
| `encoder_hidden_states` | `torch.Tensor \| None` | 双流模型的编码器状态 |
| `temb` | `torch.Tensor` | 时间步嵌入张量 |
| `run_transformer_blocks` | `Callable[[], tuple]` | 执行transformer块 |
| `postprocess` | `Callable[[torch.Tensor], Any]` | 应用最终变换 |
| `extra_states` | `dict \| None` | 可选的额外模型特定状态字典 |

### 步骤6：注册提取器

在`vllm_omni/diffusion/cache/teacache/extractors.py`中添加：

```python
EXTRACTOR_REGISTRY: dict[str, Callable] = {
    "QwenImageTransformer2DModel": extract_qwen_context,
    "Bagel": extract_bagel_context,
    "YourModelTransformer2DModel": extract_your_model_context,  # 添加此处
}
```

**键**：使用transformer类名（`module.__class__.__name__`）

### 步骤7：添加模型系数

在`vllm_omni/diffusion/cache/teacache/config.py`中添加多项式重缩放系数：

```python
_MODEL_COEFFICIENTS = {
    "QwenImageTransformer2DModel": [
        -4.50000000e02,
        2.80000000e02,
        -4.50000000e01,
        3.20000000e00,
        -2.00000000e-02,
    ],
    "YourModelTransformer2DModel": [  # 添加模型系数
        # 5个多项式系数（初始可复用相似模型的系数）
    ],
}
```

## 自定义

### 系数估计

虽然可以从相似模型架构开始，但估计特定模型的自定义系数通常能提高TeaCache性能。

**为什么估计系数？**

多项式系数重缩放连续调制输入之间的L1距离，以更好地预测何时可以重用缓存残差。

| 方法 | 性能 | 工作量 |
|------|------|--------|
| 使用相似模型的默认值 | 接近最优5-10% | 低 |
| 估计自定义系数 | 最佳性能 | 中 |

#### 实现数据收集适配器

在`vllm_omni/diffusion/cache/teacache/coefficient_estimator.py`中添加适配器：

```python
class YourModelAdapter:
    @staticmethod
    def load_pipeline(model_path, device, dtype):
        from your_model_package import YourModelPipeline
        pipeline = YourModelPipeline.from_pretrained(model_path, torch_dtype=dtype)
        return pipeline.to(device)

    @staticmethod
    def get_transformer(pipeline):
        return pipeline.transformer, "YourTransformer2DModel"

    @staticmethod
    def install_hook(transformer, hook):
        from vllm_omni.diffusion.hooks import HookRegistry
        registry = HookRegistry.get_or_create(transformer)
        registry.register_hook(hook._HOOK_NAME, hook)

_MODEL_ADAPTERS["YourModel"] = YourModelAdapter
```

#### 收集数据和估计

```python
from vllm_omni.diffusion.cache.teacache.coefficient_estimator import TeaCacheCoefficientEstimator

estimator = TeaCacheCoefficientEstimator(
    model_path="/path/to/your/model",
    model_type="YourModel",
)

dataset = load_dataset("nateraw/parti-prompts", split="train")
prompts = dataset["Prompt"][:70]

for prompt in tqdm(prompts):
    estimator.collect_from_prompt(prompt=prompt, num_inference_steps=50)

coeffs = estimator.estimate(poly_order=4)
print(f"Estimated coefficients: {coeffs.tolist()}")
```

## 测试

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(
    model="your-model-name",
    cache_backend="tea_cache",
    cache_config={
        "rel_l1_thresh": 0.2,
        "coefficients": [1.33e6, -1.69e5, 7.95e3, -1.64e2, 1.26],
    }
)

images = omni.generate(
    "a beautiful landscape",
    OmniDiffusionSamplingParams(num_inference_steps=50),
)
```

**验证**：
1. 检查日志中的TeaCache初始化消息
2. 与基线比较性能（预期1.5x-2.0x加速）
3. 验证输出质量（缓存与未缓存输出应几乎相同）

## 故障排除

### "Unknown model type"
- 检查transformer类名是否在`EXTRACTOR_REGISTRY`中

### "Cannot find coefficients"
- 在`config.py`中添加系数或传递自定义系数

### 质量下降
- 降低`rel_l1_thresh`（尝试0.1-0.2）
- 估计模型特定系数

## 参考实现

| 模型 | 路径 | 模式 |
|------|------|------|
| Qwen-Image | `vllm_omni/diffusion/cache/teacache/extractors.py` | 双流 |
| Bagel | `vllm_omni/diffusion/cache/teacache/extractors.py` | 全模态模型 |
| TeaCache Core | `vllm_omni/diffusion/cache/teacache/` | 基础实现 |
| Coefficient Estimator | `vllm_omni/diffusion/cache/teacache/coefficient_estimator.py` | 估计工具 |

## 总结

1. 编写提取器 - 创建返回`CacheContext`的函数
2. 注册提取器 - 添加到`EXTRACTOR_REGISTRY`
3. 添加系数 - 添加多项式系数到`_MODEL_COEFFICIENTS`
4. 测试 - 使用`cache_backend="tea_cache"`验证
