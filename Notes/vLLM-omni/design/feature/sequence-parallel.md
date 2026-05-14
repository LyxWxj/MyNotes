---
type: Note
related_to: "[[vLLM-omni]]"
status: Active
url: https://github.com/vllm-project/vllm-omni/blob/main/docs/design/feature/sequence_parallel.md
---

# Sequence Parallel（序列并行）

## 概述

### 什么是Sequence Parallel？

**术语说明**：我们的"序列并行"（SP）对应[diffusers库](https://github.com/huggingface/diffusers)中的"上下文并行"（CP）。

扩散transformer处理长序列的图像块或视频帧。对于高分辨率生成，这些序列可能非常大。启用SP允许每个GPU只处理序列的一部分，注意力机制（Ulysses/Ring）透明处理跨GPU通信。

### 核心API

```python
from vllm_omni.diffusion.distributed.sp_plan import (
    SequenceParallelInput,   # 用于分片（分割）张量
    SequenceParallelOutput,  # 用于收集张量
)
from vllm_omni.diffusion.distributed.sp_sharding import sp_shard, sp_gather
```

| 方法/类 | 用途 | 行为 |
|---------|------|------|
| `SequenceParallelInput` | 在`_sp_plan`中声明输入分片 | 在模块输入自动分片张量 |
| `SequenceParallelOutput` | 在`_sp_plan`中声明输出收集 | 在模块输出自动收集张量 |
| `sp_shard()` | 手动张量分片 | 跨SP工作者分割张量 |
| `sp_gather()` | 手动张量收集 | 从所有工作者收集分片张量 |

## UAA模式（实验性）

`ulysses_mode="advanced_uaa"`启用实验性UAA（"Ulysses Anything Attention"）功能，允许Ulysses注意力处理任意序列长度和任意注意力头数。

### 设计摘要

1. **严格模式不变**：`ulysses_mode="strict"`保持原始快速路径
2. **UAA使用可变all-to-all分割大小**：在Ulysses Q/K/V交换之前，每个rank all-gather其本地序列长度
3. **UAA仅在Ulysses交换内填充头**：如果`head_cnt % ulysses_degree != 0`
4. **混合Ulysses + Ring仍有形状约束**：Ring注意力期望环组中的每个rank交换相同的后Ulysses序列形状
5. **小型标量收集保持在TorchDynamo跟踪之外**

### UAA vs `auto_pad`

| 特性 | `auto_pad` | `advanced_uaa` |
|------|-----------|----------------|
| 依赖注意力掩码 | ✅ | ❌ |
| 处理不可整除头数 | ❌ | ✅ |
| 兼容Ring注意力 | ❌ | ✅（有限制） |
| 状态 | 实验性 | 实验性 |

## 方法1：非侵入式`_sp_plan`（推荐）

`_sp_plan`机制允许**不修改`forward()`逻辑**实现SP。框架自动注册钩子在模块边界分片输入和收集输出。

**适用场景**：
- 标准transformer架构
- 张量操作在`nn.Module`边界发生
- 可预测的分片/收集模式

### 步骤1：理解模块边界

```python
class MyTransformer(nn.Module):
    def __init__(self):
        self.patch_embed = PatchEmbed()      # ← 边界1
        self.pos_embed = RoPE()              # ← 边界2
        self.blocks = nn.ModuleList([...])   # ← 边界3
        self.norm_out = LayerNorm()
        self.proj_out = Linear()             # ← 边界4

    def forward(self, x):
        x = self.patch_embed(x)              # ← 在此之前分片？
        pos = self.pos_embed(x)              # ← 分片RoPE输出？
        for block in self.blocks:
            x = block(x, pos)                # ← 块处理分片的x
        x = self.norm_out(x)
        output = self.proj_out(x)            # ← 在此之后收集？
        return output
```

### 步骤2：处理内联操作

如果`forward()`包含内联张量操作，**提取到子模块**：

```python
# ❌ 差：内联操作 - 钩子无法拦截
class ZImageTransformer(nn.Module):
    def forward(self, x, cap_feats):
        unified = torch.cat([x, cap_feats], dim=1)  # 内联操作！

# ✅ 好：提取到子模块
class UnifiedPrepare(nn.Module):
    def forward(self, x, cap_feats):
        return torch.cat([x, cap_feats], dim=1)

class ZImageTransformer(nn.Module):
    def __init__(self):
        self.unified_prepare = UnifiedPrepare()  # 现在是模块！
```

### 步骤3：编写`_sp_plan`

**模式1：在第一个块分片，在输出投影收集**

```python
class StandardTransformer(nn.Module):
    _sp_plan = {
        "blocks.0": {
            "hidden_states": SequenceParallelInput(split_dim=1, expected_dims=3),
        },
        "proj_out": SequenceParallelOutput(gather_dim=1, expected_dims=3),
    }
```

**模式2：单独分片RoPE嵌入**

```python
class TransformerWithRoPE(nn.Module):
    _sp_plan = {
        "rope": {
            0: SequenceParallelInput(split_dim=1, expected_dims=4, split_output=True),  # cos
            1: SequenceParallelInput(split_dim=1, expected_dims=4, split_output=True),  # sin
        },
        "blocks.0": {
            "hidden_states": SequenceParallelInput(split_dim=1, expected_dims=3),
        },
        "proj_out": SequenceParallelOutput(gather_dim=1, expected_dims=3),
    }
```

**模式3：双流注意力的RoPE分片**

```python
class DualStreamTransformer(nn.Module):
    _sp_plan = {
        "rope_preparer": {
            # 输出0, 1（文本）- 不分片（复制）
            # 输出2, 3（图像）- 分片
            2: SequenceParallelInput(split_dim=0, expected_dims=2, split_output=True),  # img_cos
            3: SequenceParallelInput(split_dim=0, expected_dims=2, split_output=True),  # img_sin
        },
        "transformer_blocks.0": {
            "hidden_states": SequenceParallelInput(split_dim=1, expected_dims=3),
        },
        "proj_out": SequenceParallelOutput(gather_dim=1, expected_dims=3),
    }
```

### API参考

**SequenceParallelInput参数**：

| 参数 | 类型 | 描述 |
|------|------|------|
| `split_dim` | int | 分割维度（通常`1`表示序列） |
| `expected_dims` | int \| None | 预期张量秩（可选） |
| `split_output` | bool | `False`：分片输入参数；`True`：分片输出张量 |
| `auto_pad` | bool | 序列不可被world_size整除时自动填充 |

**模块命名约定**：

| 键 | 含义 | Python等价 |
|----|------|------------|
| `""` | 根模型 | `model` |
| `"blocks.0"` | ModuleList第一个元素 | `model.blocks[0]` |
| `"blocks.*"` | ModuleList所有元素 | `for b in model.blocks` |

## 方法2：侵入式修改（复杂情况）

对于无法通过`_sp_plan`表达的动态分片逻辑，手动插入分片/收集调用：

```python
from vllm_omni.diffusion.distributed.sp_sharding import sp_shard, sp_gather

def forward(self, hidden_states, ...):
    if self.parallel_config.sequence_parallel_size > 1:
        hidden_states = sp_shard(hidden_states, dim=1)

    # ... 计算 ...

    if self.parallel_config.sequence_parallel_size > 1:
        output = sp_gather(output, dim=1)

    return output
```

## 测试

```bash
python text_to_image.py \
    --model Your-org/your-model \
    --ulysses-degree 2 \
    --ring-degree 2 \
    --output sp_test.png
```

### 验证

1. **正确性**：输出在所有`sp_size`值下应相同
2. **速度**：吞吐量应保持稳定或提高
3. **日志**：检查形状不匹配或通信错误

## 故障排除

### 形状不匹配错误

- **RoPE维度不匹配**：在`_sp_plan`中分片RoPE输出
- **序列长度不可被sp_size整除**：使用`ulysses_mode="advanced_uaa"`或`auto_pad=True`

### 内联操作未分片

- **问题**：操作在`forward()`中内联发生
- **解决方案**：提取到子模块

## 参考实现

| 模型 | 路径 | 模式 | 备注 |
|------|------|------|------|
| LongCat | `vllm_omni/diffusion/models/longcat_image/` | 双流 | 文本复制，图像分片 |
| Qwen-Image | `vllm_omni/diffusion/models/qwen_image/` | 双流 + 预处理 | auto_pad |
| Wan2.2 | `vllm_omni/diffusion/models/wan2_2/` | 双transformer + RoPE | 视频transformer |
| Z-Image | `vllm_omni/diffusion/models/z_image/` | 统一序列 | 连接输入 |

## 总结

1. 选择方法 - 标准情况用`_sp_plan`，复杂情况用侵入式修改
2. 识别分片边界 - 张量应在何处分割/收集？
3. 提取内联操作 - 将`torch.cat`、`pad_sequence`等移到子模块
4. 定义`_sp_plan` - 声明分片/收集点为类属性
5. 使用`auto_pad`处理可变长度
6. 测试 - 验证不同的`ulysses_degree`和`ring_degree`组合
