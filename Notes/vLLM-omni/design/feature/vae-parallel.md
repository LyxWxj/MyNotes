---
type: Note
related_to: "[[vLLM-omni]]"
status: Active
url: https://github.com/vllm-project/vllm-omni/blob/main/docs/design/feature/vae_parallel.md
---

# VAE Patch Parallelism（VAE补丁并行）

## 概述

### 什么是VAE Patch Parallelism？

**VAE补丁并行**是一种用于**编码**和**解码**的加速技术。不是一次处理整个张量，而是：
- 将张量分割成多个空间块
- 分布到多个rank
- 并行编码/解码
- 合并以重建最终输出

这种方法：
- 将计算分布到多个设备
- 减少每设备峰值内存使用
- 加速编码/解码延迟

### 何时使用编码 vs 解码并行

| 操作 | 用例 | 示例 |
|------|------|------|
| **解码并行** | 文本到图像、文本到视频 | 潜变量 → 图像/视频 |
| **编码并行** | 图像到视频（I2V） | 图像 → 潜变量（用于条件） |

### 架构

引入**DistributedVaeExecutor**作为核心组件，负责分布式VAE编码/解码。

执行器是模型无关的，接受三个函数参数：
- `split`：将潜变量分割成块
- `exec`：解码单个块
- `merge`：将解码的块合并成最终输出

#### 执行流程

1. 调用`split(z)`生成TileTask列表和GridSpec
2. 使用基于工作量的平衡在rank之间分发任务
3. 每个rank执行`exec(task)`处理分配的块
4. 将解码的块结果收集到rank 0
5. Rank 0执行`merge(...)`
6. （可选）广播最终结果到所有rank

#### 为什么需要split/exec/merge？

潜变量张量不能任意分区。在解码期间：
- 每个输出像素可能依赖相邻像素
- 感受野是模型相关的

因此：
- 块必须包含重叠
- 合并必须执行混合以避免接缝

## 解码并行实现

### 步骤1：实现DistributedAutoencoderKLQwenImage

```python
class DistributedAutoencoderKLQwenImage(AutoencoderKLQwenImage, DistributedVaeMixin):
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        model = super().from_pretrained(*args, **kwargs)
        model.init_distributed()
        return model
```

**关键点**：
- 继承AutoencoderKLQwenImage和DistributedVaeMixin
- 加载权重后调用`init_distributed()`

### 步骤2：实现split/exec/merge

重用`AutoencoderKLQwenImage.tiled_decode`逻辑并分为三个阶段：

```python
class GridSpec:
    split_dims: tuple[int, ...]  # 被分割的张量维度
    grid_shape: tuple[int, ...]  # 块网格布局
    tile_spec: dict = field(default_factory=dict)  # 合并所需的元数据
    output_dtype: torch.dtype | None = None  # 最终输出dtype

class TileTask:
    tile_id: int  # 任务id
    grid_coord: tuple[int, ...]  # 块在网格中的位置
    tensor: torch.Tensor | list[torch.Tensor]  # 块张量
    workload: int | float = 1  # 用于负载均衡
```

**tile_split示例**：
```python
def tile_split(self, z):
    _, _, num_frames, height, width = z.shape
    # 计算块参数
    tiletask_list = []
    for i in range(0, height, tile_latent_stride_height):
        for j in range(0, width, tile_latent_stride_width):
            time_list = []
            for k in range(num_frames):
                tile = z[:, :, k:k+1, i:i+tile_latent_min_height, j:j+tile_latent_min_width]
                time_list.append(tile)
            tiletask_list.append(TileTask(...))
    grid_spec = GridSpec(...)
    return tiletask_list, grid_spec
```

**tile_exec示例**：
```python
def tile_exec(self, task):
    self.clear_cache()
    time = []
    for k in range(len(task.tensor)):
        tile = self.post_quant_conv(task.tensor[k])
        decoded = self.decoder(tile, feat_cache=self._feat_map, feat_idx=self._conv_idx)
        time.append(decoded)
    return torch.cat(time, dim=2)
```

**tile_merge示例**：
```python
def tile_merge(self, coord_tensor_map, grid_spec):
    grid_h, grid_w = grid_spec.grid_shape
    result_rows = []
    for i in range(grid_h):
        result_row = []
        for j in range(grid_w):
            tile = coord_tensor_map[(i, j)]
            if i > 0:
                tile = self.blend_v(coord_tensor_map[(i-1, j)], tile, blend_height)
            if j > 0:
                tile = self.blend_h(coord_tensor_map[(i, j-1)], tile, blend_width)
            result_row.append(tile[:, :, :, :stride_height, :stride_width])
        result_rows.append(torch.cat(result_row, dim=-1))
    return torch.cat(result_rows, dim=3)[..., :sample_height, :sample_width]
```

### 步骤3：重写tiled_decode

```python
def tiled_decode(self, z, return_dict=True):
    if not self.is_distributed_enabled():
        return super().tiled_decode(z, return_dict=return_dict)

    result = self.distributed_executor.execute(
        z,
        DistributedOperator(
            split=self.tile_split,
            exec=self.tile_exec,
            merge=self.tile_merge,
        ),
        broadcast_result=True,
    )
    if not return_dict:
        return (result,)
    return DecoderOutput(sample=result)
```

`broadcast_result`根据模型设置为True或False；启用时，rank 0以外的rank也会使用结果。

### 步骤4：修改管道

```python
class YourModelPipeline(nn.Module):
    def __init__(self, ...):
-       self.vae = AutoencoderKL.from_pretrained(...)
+       self.vae = DistributedAutoencoderKL.from_pretrained(...)
```

## 编码并行实现

对于需要VAE编码的模型（如图像到视频），也可以并行化编码操作。以**Wan2.2**为参考实现。

### 步骤1：实现encode_tile_split

```python
def encode_tile_split(self, x):
    # 与解码类似，将输入张量分割成块
    # 关键考虑：
    # - 补丁化处理：如果模型使用patch_size，相应缩放块参数
    # - 时间分块：视频VAE可能有时间压缩（如4x）
    ...
    return tiletask_list, grid_spec
```

### 步骤2：实现encode_tile_exec

```python
def encode_tile_exec(self, task):
    self.clear_cache()
    time = []
    for k, tile in enumerate(task.tensor):
        encoded = self.encoder(tile, feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
        encoded = self.quant_conv(encoded)
        time.append(encoded)
    return torch.cat(time, dim=2)
```

### 步骤3：实现encode_tile_merge

```python
def encode_tile_merge(self, coord_tensor_map, grid_spec):
    # 与解码合并类似，但使用编码特定的混合参数
    ...
    return enc
```

### 步骤4：重写tiled_encode

```python
def tiled_encode(self, x):
    # 注意：x已被父类的_encode()补丁化
    if not self.is_distributed_enabled():
        return super().tiled_encode(x)

    result = self.distributed_executor.execute(
        x,
        DistributedOperator(
            split=self.encode_tile_split,
            exec=self.encode_tile_exec,
            merge=self.encode_tile_merge,
        ),
        broadcast_result=True,  # 潜变量需要所有rank用于扩散
    )
    return result
```

### 编码 vs 解码并行对比

| 方面 | 解码并行 | 编码并行 |
|------|----------|----------|
| `broadcast_result` | 通常`False`（仅rank 0需要输出） | `True`（所有rank需要潜变量） |
| 补丁化 | 在合并中应用（反补丁化） | 由父类`_encode()`在`tiled_encode()`之前处理 |
| 时间分块 | 逐帧 | 基于块（如1 + 4n帧） |

## 测试

验证数值一致性：
- `vae_patch_parallel_size = 1`
- `vae_patch_parallel_size = N`
- `torch.allclose(output_1, output_n, atol=1e-5)`

测试要求：
- 固定随机种子
- 使用相同的分块策略

```python
m = Omni(
    model=model_name,
    vae_use_tiling=True,
    parallel_config=DiffusionParallelConfig(
        tensor_parallel_size=2,
        vae_patch_parallel_size=1,  # 或2
    ),
)
```

当`vae_patch_parallel_size`大于DiT world size时，会自动回退到使用DiT world size。

## 参考实现

| 模型 | 路径 | 解码并行 | 编码并行 |
|------|------|----------|----------|
| Z-Image | `vllm_omni/diffusion/distributed/autoencoders/autoencoder_kl.py` | ✅ | ❌ |
| Wan2.2 | `vllm_omni/diffusion/distributed/autoencoders/autoencoder_kl_wan.py` | ✅ | ✅ |
| Qwen-Image | `vllm_omni/diffusion/distributed/autoencoders/autoencoder_kl_qwenimage.py` | ✅ | ❌ |

## 总结

1. 实现分布式VAE - 继承基础VAE类和DistributedVaeMixin
2. 解码并行 - 将tiled_decode重构为tile_split/tile_exec/tile_merge
3. 编码并行（可选）- 为I2V模型实现encode_tile_split/encode_tile_exec/encode_tile_merge
4. 修改管道中的VAE模型 - 使用分布式版本
5. 测试 - 验证`vae_patch_parallel_size=1`与`N`的数值一致性
