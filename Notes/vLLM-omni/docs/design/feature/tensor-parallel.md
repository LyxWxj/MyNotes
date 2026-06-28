---
type: Note
related_to: "[[vllm-omni]]"
status: Active
url: https://github.com/vllm-project/vllm-omni/blob/main/docs/design/feature/tensor_parallel.md
---

# Tensor Parallel（张量并行）

## 概述

### 什么是Tensor Parallel？

Tensor Parallel（TP）是一种模型并行技术，将**模型权重分片**到多个GPU上。每个GPU只持有模型参数的一部分，只计算每层输出的一部分。

扩散transformer包含大型注意力和MLP层。我们可以使用Tensor Parallel将模型维度分片到多个GPU上，使更大的模型能够放入内存，同时实现近乎线性的加速。

### 架构

Tensor Parallel实现依赖vLLM的Parallel Layers：

| 层类型 | 用途 | 权重分区 |
|--------|------|----------|
| `ColumnParallelLinear` | 第一个FFN层，分离的QKV | 列（输出维度） |
| `RowParallelLinear` | 第二个FFN层，注意力输出 | 行（输入维度） |
| `QKVParallelLinear` | 多头/分组查询注意力QKV | 自动处理头复制 |
| `ReplicatedLinear` | 不应分片的层 | 不分区（复制） |

## 实现步骤

### 步骤1：识别线性层

找到transformer中所有需要分片的`nn.Linear`层：
- 哪些层应列并行（权重按列分割）？
- 哪些层应行并行（权重按行分割）？

### 步骤2：用并行等效层替换线性层

**MLP块示例（上-下模式）**：

```python
class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        # 列并行：权重按列分割 [hidden_dim/N, dim]
        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, return_bias=False,
        )
        self.act = nn.GELU()

        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False,
            input_is_parallel=True,  # 输入已从w1分片
            return_bias=False,
        )

    def forward(self, x):
        # x: [batch, seq, dim]（在所有GPU上复制）
        # w1输出分片 [batch, seq, hidden_dim/N]
        x = self.w1(x)
        x = self.act(x)
        # w2通过all-reduce输出完整dim [batch, seq, dim]
        x = self.w2(x)
        return x
```

**注意力示例（QKV-输出模式）**：

```python
class YourModelAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        self.head_dim = dim // num_heads

        # 列并行：QKV权重按列分割
        # 每个GPU获得num_heads/N个头
        self.to_qkv = QKVParallelLinear(
            hidden_size=dim,
            head_size=self.head_dim,
            total_num_heads=num_heads,
            total_num_kv_heads=num_kv_heads,
            bias=False,
            return_bias=False,
        )

        # 行并行：输出权重按行分割
        self.to_out = RowParallelLinear(
            dim, dim, bias=False,
            input_is_parallel=True,  # 输入已从注意力分片
            return_bias=False,
        )

        self.attn = Attention(
            num_heads=self.to_qkv.num_heads,  # 每个GPU获得num_heads/N个头
            head_size=self.head_dim,
            softmax_scale=1.0 / (self.head_dim**0.5),
            causal=False,
            num_kv_heads=self.to_qkv.num_kv_heads,
        )

    def forward(self, x):
        # x: [batch, seq, dim]（复制）
        # to_qkv输出分片 [batch, seq, (q+k+v) * head_dim/N]
        qkv = self.to_qkv(x)
        q, k, v = qkv.split([...], dim=-1)
        # 注意力在每个GPU上独立计算
        out = self.attn(q, k, v)
        # to_out通过all-reduce输出完整dim
        out = self.to_out(out)
        return out
```

**关键点**：
- `ColumnParallelLinear` → `RowParallelLinear`是标准配对
- 当输入来自`ColumnParallelLinear`时，在`RowParallelLinear`上设置`input_is_parallel=True`
- 对注意力投影使用`QKVParallelLinear`（自动处理头复制）

### 步骤3：验证TP约束

为确保TP正确操作，以下维度**必须可被`tensor_parallel_size`整除**：

| 维度 | 原因 | 示例错误 |
|------|------|----------|
| `num_heads` | 头被QKVParallelLinear分片 | `num_heads=30, tp=4` ❌ |
| `num_kv_heads` | KV头被QKVParallelLinear分片 | `num_kv_heads=30, tp=4` ❌ |

## 测试

### Python API

```python
from vllm_omni import Omni
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

parallel_config = DiffusionParallelConfig(tensor_parallel_size=2)
omni = Omni(model="your-model-name", parallel_config=parallel_config)

output = omni.generate(
    "a cup of coffee on the table",
    OmniDiffusionSamplingParams(num_inference_steps=50),
)
```

### 命令行

```bash
python text_to_image.py \
    --model Your-org/your-model \
    --prompt "a cup of coffee on the table" \
    --tensor-parallel-size 2
```

### 验证

1. 检查日志中的`e2e_time_ms`以确认加速
2. 与禁用TP时比较生成图像质量
3. 验证内存使用按比例减少
4. 在PR中记录比较结果

## 故障排除

### TP未激活

**症状**：模型在单GPU上运行，无内存节省或加速。

**原因**：仍在使用`nn.Linear`。

**解决方案**：
```python
# ❌ 差
self.proj = nn.Linear(dim, dim)

# ✅ 好
self.proj = RowParallelLinear(dim, dim, input_is_parallel=True)
```

### 维度不匹配错误

**症状**：前向传播期间`RuntimeError: shape mismatch`。

**原因**：缺少`input_is_parallel=True`。

**解决方案**：
```python
self.w1 = ColumnParallelLinear(dim, hidden_dim, return_bias=False)
self.w2 = RowParallelLinear(
    hidden_dim, dim,
    input_is_parallel=True,  # 输入已从w1分片
    return_bias=False,
)
```

## 参考实现

| 模型 | 路径 | 模式 | 备注 |
|------|------|------|------|
| Z-Image | `vllm_omni/diffusion/models/z_image/z_image_transformer.py` | 标准TP | 完整实现 |
| FLUX | `vllm_omni/diffusion/models/flux/flux_transformer.py` | 双流 | 图像 + 文本流 |
| Qwen-Image | `vllm_omni/diffusion/models/qwen_image/qwen_image_transformer.py` | 标准TP | 带RoPE |
| TP Tests | `tests/e2e/offline_inference/test_zimage_parallelism.py` | E2E测试 | TP正确性和性能 |
| Constraint Tests | `tests/diffusion/models/z_image/test_zimage_tp_constraints.py` | 单元测试 | 验证逻辑 |

## 总结

1. 识别线性层 - 哪些层应被分片？
2. 用并行层替换 - 使用QKVParallelLinear、ColumnParallelLinear、RowParallelLinear
3. 验证TP约束 - 确保维度可被TP大小整除
4. 测试 - 使用`tensor_parallel_size=N`验证，检查内存、速度和质量
