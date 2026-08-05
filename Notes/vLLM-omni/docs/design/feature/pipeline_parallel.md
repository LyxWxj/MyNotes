---
type: Note
related_to: "[[vllm-omni]]"
status: Active
url: https://github.com/vllm-project/vllm-omni/blob/main/docs/design/feature/pipeline_parallel.md
---

# 扩散流水线并行（Pipeline Parallel）

本文说明如何为扩散管道添加流水线并行（PP）支持，以 Wan2.2 文生视频（T2V）与图生视频（I2V）管道为参考实现。

## 实现检查清单

添加 PP 支持需要：

1. ✅ **继承 mixin**：在管道类中把 `PipelineParallelMixin` 放在 `CFGParallelMixin` 之前
2. ✅ **让 stage forward 可续算**：transformer forward 路径支持 `intermediate_tensors`
3. ✅ **返回正确对象类型**：非最后 PP rank 返回 `IntermediateTensors`，最后 rank 返回最终模型输出
4. ✅ **使用标准去噪循环**：调用 `predict_noise_maybe_with_cfg()` 与 `scheduler_step_maybe_with_cfg()`
5. ✅ **测试一致性**：PP 结果与单 GPU 基线对比

## 概述

### 什么是流水线并行？

流水线并行把去噪 transformer 拆成多个顺序 stage，每个 stage 放在不同 rank 上。每个 PP rank 只持有层的一个切片，而非整个 DiT。

每个去噪 step 的流程：
1. Rank 0 用当前 latents 启动 forward
2. 每个中间 rank 接收上一 rank 的 hidden states，运行本地层切片，把中间张量转发给下游
3. 最后 PP rank 产出最终噪声预测
4. 最后 PP rank 应用调度器 step，并把更新后的 latents 发回 rank 0 供下一个 timestep 使用

这降低单 rank 模型显存，使更大的扩散 transformer 能跨多 GPU 运行。它还可以与 CFG-Parallel 组合：每条 PP 流水线承载一个 CFG 分支。

### 架构

vLLM-Omni 提供 `PipelineParallelMixin` 封装扩散管道的 PP 通信模式。

| 方法 | 用途 | 自动行为 |
|------|------|---------|
| `diffuse()` | 去噪循环边界 | 被 `PipelineParallelMixin` 包装，退出时冲刷未完成的异步 PP 发送 |
| `predict_noise_maybe_with_cfg()` | 带 PP 支持的噪声预测 | 非最后 PP rank 运行部分 forward，需要时与 CFG 逻辑组合 |
| `scheduler_step_maybe_with_cfg()` | 带 PP 同步的调度器 step | 在最后 PP rank 上运行调度器，把更新后的 latents 返回 rank 0 |
| `_sync_pp_send()` | 冲刷未完成异步发送 | 在后续集合通信或 decode 前等待未完成的 `isend` 句柄 |

`PipelineParallelMixin` 是有意设计的管道级抽象。模型专属的 `predict_noise()` 仍定义本地 stage 如何执行。

### 工作原理

`predict_noise_maybe_with_cfg()` 自动在以下模式间切换：

- **PP 关闭**（`pipeline_parallel_size == 1`）：回退到 `CFGParallelMixin.predict_noise_maybe_with_cfg()`
- **仅 PP**（`pipeline_parallel_size > 1`，`cfg_parallel_size == 1`）：
  - Rank 0 以输入 latents 启动
  - 中间 rank 接收 `intermediate_tensors`，运行本地层区间，异步发送给下游
  - 最后 rank 返回最终噪声预测
  - 该模式启用 CFG 时（顺序 CFG），正/负两个分支走同一条 PP 流水线，每个去噪 step 通信量翻倍。有 `cfg_parallel_size > 1` 时优先用 PP + CFG-Parallel 避免该开销
- **PP + CFG-Parallel**（`pipeline_parallel_size > 1`，`cfg_parallel_size > 1`）：
  - 每条 PP 流水线承载一个 CFG 分支
  - 最后 PP rank 在 CFG 组内做 all-gather
  - CFG 组合在每个 CFG 组的最后 PP rank 上进行，与非 PP 的 CFG-parallel 行为一致

`scheduler_step_maybe_with_cfg()` 保持去噪循环一致：
- **PP 关闭**：回退到 `scheduler_step_maybe_with_cfg()`
- **PP 启用**：
  - 只有最后 PP rank 拥有 `noise_pred` 并运行调度器 step
  - 结果 latents 发回 rank 0
  - Rank 0 收到一个 `AsyncLatents` 包装，仅在张量下次被消费时解析

这种异步设计避免去噪 step 之间的不必要阻塞。

管道类定义时，`PipelineParallelMixin` 包装其 `diffuse()` 方法，并在 `diffuse()` 返回或抛异常后的 `finally` 块中调用 `_sync_pp_send()`。这让模型代码无需显式 PP 清理，同时保证最后的非阻塞 PP 发送在 decode 或后续集合通信前完成。

## 分步实现

### 第 1 步：继承 `PipelineParallelMixin` 与 `CFGParallelMixin`

`PipelineParallelMixin` **要求** `CFGParallelMixin`，且必须在类 MRO 中位于其之前。这在类定义时通过 `__init_subclass__` 强制：继承 `PipelineParallelMixin` 但没有 `CFGParallelMixin`，或把 `CFGParallelMixin` 放在前面，import 时会立即抛 `TypeError`。

`PipelineParallelMixin` 把噪声预测、CFG 组合与调度器 step 委托给 `CFGParallelMixin`，后者提供 `predict_noise()`、`predict_noise_maybe_with_cfg()`、`scheduler_step_maybe_with_cfg()` 与 `combine_cfg_noise()`。

**示例**：

```python
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.pipeline_parallel import PipelineParallelMixin
import torch.nn as nn


class YourPipeline(nn.Module, PipelineParallelMixin, CFGParallelMixin):
    ...
```

顺序很重要：`PipelineParallelMixin` 必须列在 `CFGParallelMixin` 之前，使 `predict_noise_maybe_with_cfg()` 与 `scheduler_step_maybe_with_cfg()` 解析到 PP 感知包装器，而其 `super()` 调用在 PP 关闭或最后 PP stage 之后委托给 CFG 实现。

### 第 2 步：让模型 forward 与 `predict_noise()` 感知 PP

PP mixin 把 `intermediate_tensors` 注入普通的 `predict_noise()` 调用。模型 forward 路径必须支持两种输入：
- 来自 rank 0 的普通输入，通常以 `hidden_states` 或 `x` 传入
- 来自上游 PP rank 的 `intermediate_tensors`

标准模型 forward 模式：
1. 若存在 `intermediate_tensors`，从中读取本地 hidden state
2. 只运行本 rank 的层切片
3. 非最后 PP rank 返回 `IntermediateTensors(...)`
4. 最后 PP rank 返回最终模型输出。`CFGParallelMixin.predict_noise()` 对常见管道已遵循该契约

**最小示例**：

```python
from vllm.sequence import IntermediateTensors
from vllm_omni.diffusion.distributed.parallel_state import get_pp_group


def forward(self, hidden_states=None, intermediate_tensors=None, **kwargs):
    if intermediate_tensors is not None:
        hidden_states = intermediate_tensors["hidden_states"]

    for i in range(self.start_layer, self.end_layer):
        hidden_states = self.layers[i](hidden_states)

    pp_group = get_pp_group()
    if not pp_group.is_last_rank:
        return IntermediateTensors({"hidden_states": hidden_states})
    return (hidden_states,)
```

### 第 3 步：切分 transformer 层

每个 PP rank 上的本地模块必须只暴露该 rank 的层切片。参考测试与 vLLM 模型实现中通常用 `make_layers(...)` 等 vLLM 工具完成，缺失区间用 `PPMissingLayer` 填充。除了切分层本身，模型作者还应接入 `make_empty_intermediate_tensors_factory(...)`（用于中间张量分配）与 `is_pp_missing_parameter(...)`（用于 PP 感知的权重加载）。

为 PP 准备 transformer，按顺序实现以下部分：

#### 3.1 跨 PP rank 切分 transformer 层

每个 PP rank 只拥有本地层区间，通常暴露为 `[start_layer, end_layer)`。实践中通常用 `make_layers(...)` 构建本地层并用 `PPMissingLayer` 填充缺失区间。

目标是：
- 每个 PP rank 知道自己的 `[start_layer, end_layer)` 区间
- 非本地层不在该 rank 执行
- forward 可从传入的 `intermediate_tensors` 续算

默认情况下层通过 `get_pp_indices()` 均匀分布到各 PP rank。层数不能被 PP size 整除时，剩余层分配给中间分区以均衡计算与显存。可用 `VLLM_PP_LAYER_PARTITION` 环境变量覆盖，指定每 rank 精确层数：

```bash
# 示例：40 层分到 4 个 PP rank，分配 8 / 12 / 12 / 8 层
export VLLM_PP_LAYER_PARTITION=8,12,12,8
```

值必须是逗号分隔的整数列表，长度等于 `pipeline_parallel_size`，且总和等于 transformer 总层数。

#### 3.2 暴露 `make_empty_intermediate_tensors`

transformer 模块应暴露 `self.make_empty_intermediate_tensors`，通常用 `make_empty_intermediate_tensors_factory(...)` 创建。

这很重要：PP rank 需要一致的方式分配带预期 key 与隐藏维度的占位 `IntermediateTensors`。

**示例**：

```python
from vllm.model_executor.models.utils import make_empty_intermediate_tensors_factory

self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
    ["hidden_states"],
    inner_dim,
)
```

对 Wan2.2，PP stage 间的中间载荷是存在 `"hidden_states"` key 下的 token 序列，因此 factory 用该 key 与 transformer 隐藏大小创建。

#### 3.3 非最后 PP rank 返回 `IntermediateTensors`

模型 `forward()` 或自定义 `predict_noise()` 实现应在非首 rank 消费 `intermediate_tensors`，在非最后 rank 返回 `IntermediateTensors(...)`。

这样每个 PP stage 都能从上游 hidden states 续算，并把本地结果传给下一 stage。

#### 3.4 在 `load_weights()` 中跳过非本地权重

模型被 PP 切分后，checkpoint 中很多参数属于当前 rank 不存在的层。`load_weights()` 必须用 `is_pp_missing_parameter(...)` 跳过缺失 PP stage 的参数。

否则权重加载会失败，或错误地把张量加载进 `PPMissingLayer` 占位符。

Wan2.2 transformer 是最佳参考：它在加载 remapped 与 fused 参数前都使用 `is_pp_missing_parameter(...)`。

如果模型有多个 transformer 变体，只要每个选中的 transformer 遵循相同契约，PP 仍然有效。

### 第 4 步：使用标准去噪契约

vLLM-Omni 扩散管道已通过 `diffuse()`、`predict_noise_maybe_with_cfg()` 与 `scheduler_step_maybe_with_cfg()` 路由去噪。`pipeline_parallel_size > 1` 时 `PipelineParallelMixin` 覆写这些标准辅助函数，因此模型集成不应添加单独的 PP 专属辅助命名或手动循环后同步。

## 测试

使用 `pipeline_parallel_size > 1` 的离线推理脚本：

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
--model=Wan-AI/Wan2.2-TI2V-5B-Diffusers \
--width=1280 \
--height=704 \
--guidance-scale=5.0 \
--prompt="Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage" \
--output=t2v_5B_pp2.mp4 \
--pipeline-parallel-size=2
```

PP + CFG-Parallel 组合：

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
--model=Wan-AI/Wan2.2-TI2V-5B-Diffusers \
--width=1280 \
--height=704 \
--guidance-scale=5.0 \
--prompt="Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage" \
--output=t2v_5B_pp2_cfg2.mp4 \
--pipeline-parallel-size=2 \
--cfg-parallel-size=2
```

**验证**：
1. 运行在 PP stage 边界不挂起
2. 输出质量在正常数值波动范围内与非 PP 基线一致
3. 每 GPU 峰值显存相比单 rank 模型下降
4. decode 前无未完成的通信错误

## 故障排查

### 问题：import 时 `TypeError` — 缺少 `CFGParallelMixin`

**症状**：import 继承 `PipelineParallelMixin` 的管道时抛出：

```
TypeError: YourPipeline inherits PipelineParallelMixin but not CFGParallelMixin.
```

或：

```
TypeError: YourPipeline must inherit PipelineParallelMixin before CFGParallelMixin ...
```

**原因**：`PipelineParallelMixin` 通过 `__init_subclass__` 强制子类同时继承 `CFGParallelMixin`，且 `PipelineParallelMixin` 必须在 MRO 中排在最前。

**解决**：把 `CFGParallelMixin` 加在 `PipelineParallelMixin` 之后：

```python
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.pipeline_parallel import PipelineParallelMixin


class YourPipeline(nn.Module, PipelineParallelMixin, CFGParallelMixin):
    ...
```

### 问题：非最后 PP rank 调用 `predict_noise` 崩溃

**症状**：首个与最后 PP rank 之外的 rank 出现形状错误或缺失输入错误。

**原因**：模型 forward 路径假设直接输入张量，忽略 `intermediate_tensors`。

**解决**：更新 transformer `forward()` 或自定义 `predict_noise()` 路径，在存在 `intermediate_tensors` 时从中加载 hidden states。

### 问题：PP 输出与单 GPU 基线不一致

**症状**：PP 运行完成但数值结果不一致。

**原因与解决**：
- **本地层切分错误**：确认每个 rank 只运行自己的 `[start_layer, end_layer)` 切片
- **非最后 rank 返回普通张量而非 `IntermediateTensors`**：在最后 PP stage 之前返回 `IntermediateTensors({...})`
- **CFG 分支接线错误**：启用 CFG 时，确认正/负 kwargs 与非 PP 路径完全一致

## 参考实现

代码库中的完整示例：

| 组件 | 路径 | 说明 |
|------|------|------|
| `PipelineParallelMixin` | `vllm_omni/diffusion/distributed/pipeline_parallel.py` | 核心 PP 通信与调度器辅助 |
| `CFGParallelMixin` | `vllm_omni/diffusion/distributed/cfg_parallel.py` | 默认 `predict_noise()` 元组归一化与 CFG 辅助回退 |
| Wan2.2 transformer | `vllm_omni/diffusion/models/wan2_2/wan2_2_transformer.py` | 层切分、`IntermediateTensors`、`make_empty_intermediate_tensors` 与 PP 感知权重加载的参考 |
| Wan2.2 T2V 管道 | `vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py` | 文生视频的 PP + CFG 集成参考 |
| Wan2.2 I2V 管道 | `vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2_i2v.py` | 图生视频的 PP + CFG 集成参考 |
| PP 测试 | `tests/diffusion/distributed/test_pipeline_parallel.py` | 基线一致性与异步通信测试 |
