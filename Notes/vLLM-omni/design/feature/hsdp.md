---
type: Note
related_to: "[[vLLM-omni]]"
status: Active
url: https://github.com/vllm-project/vllm-omni/blob/main/docs/design/feature/hsdp.md
---

# HSDP（混合分片数据并行）

## 概述

### 什么是HSDP？

HSDP（Hybrid Sharded Data Parallel）是一种内存优化技术，使用PyTorch的FSDP2在多个GPU上分片模型权重。与张量并行不同，HSDP：

- 在GPU上分片权重以减少每GPU内存使用
- 在前向传播期间按需收集权重
- 可单独使用或与其他并行（如序列并行）组合

这使得在有限内存的GPU上推理大型模型（如Wan2.2 14B）成为可能。

**重要约束**：
- HSDP不能与张量并行一起使用
- 对于独立HSDP，必须显式指定`hsdp_shard_size`

### 架构

HSDP实现依赖于：

1. **`_hsdp_shard_conditions`**：指定要分片的模块的模型属性
2. **`apply_hsdp_to_model`**：应用FSDP2分片的函数
3. **`HSDPInferenceConfig`**：HSDP的运行时配置

## 实现步骤

### 步骤1：识别要分片的模块

确定transformer中哪些模块应被分片：
- Transformer块（如`blocks.0`、`blocks.1`）
- 具有大量权重内存的大型子模块

### 步骤2：定义分片条件

添加`_hsdp_shard_conditions`到模型类：

```python
class MyTransformerModel(nn.Module):

    @staticmethod
    def _is_transformer_block(name: str, module) -> bool:
        """匹配transformer块用于HSDP分片"""
        return "blocks" in name and name.split(".")[-1].isdigit()

    _hsdp_shard_conditions = [_is_transformer_block]
```

**多条件示例**：

```python
class MyModel(nn.Module):

    @staticmethod
    def _is_transformer_block(name: str, module) -> bool:
        return "blocks" in name and name.split(".")[-1].isdigit()

    @staticmethod
    def _is_moe_expert(name: str, module) -> bool:
        return "experts" in name and name.split(".")[-1].isdigit()

    # 模块在任一条件返回True时被分片
    _hsdp_shard_conditions = [_is_transformer_block, _is_moe_expert]
```

## 测试

### Python API

```python
from vllm_omni import Omni
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

parallel_config = DiffusionParallelConfig(
    use_hsdp=True,
    hsdp_shard_size=8,  # 跨8个GPU分片
)
omni = Omni(model="your-model-name", parallel_config=parallel_config)

output = omni.generate(
    "a cup of coffee on the table",
    OmniDiffusionSamplingParams(num_inference_steps=50),
)
```

### 命令行

```bash
vllm serve Your-org/your-model --omni --port 8091 --use-hsdp
```

### 验证

1. 检查日志中的"HSDP Inference: replicate_size=..., shard_size=..."
2. 检查日志中的"Sharded N modules + root"
3. 验证内存使用按比例减少
4. 与禁用HSDP时比较生成输出质量

## 参考实现

| 模型 | 路径 | 备注 |
|------|------|------|
| Wan2.2 | `vllm_omni/diffusion/models/wan2_2/wan2_2_transformer.py` | 参考实现 |
| HSDP Core | `vllm_omni/diffusion/distributed/hsdp.py` | `apply_hsdp_to_model`、`shard_model` |
| HSDP Tests | `tests/diffusion/distributed/test_hsdp.py` | 单元测试 |
