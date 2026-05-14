---
type: Note
related_to: "[[vLLM-omni]]"
status: Active
url: https://github.com/vllm-project/vllm-omni/blob/main/docs/design/feature/cfg_parallel.md
---

# CFG-Parallel（无分类器引导并行）

## 概述

### 什么是CFG-Parallel？

在标准无分类器引导（CFG）中，每个扩散步骤需要两次transformer前向传播：
1. **正向/条件**：由文本提示引导
2. **负向/无条件**：通常使用空提示或负提示

CFG-Parallel通过将两次前向传播分布在不同GPU rank上同时执行来消除这一瓶颈。

### 核心API

| 方法 | 用途 | 自动行为 |
|------|------|----------|
| `predict_noise_maybe_with_cfg()` | 使用CFG预测噪声 | 检测并行模式，分发计算，收集结果 |
| `scheduler_step_maybe_with_cfg()` | 步进调度器 | 所有rank本地步进（无需广播） |
| `combine_cfg_noise()` | 合并正/负预测 | 应用CFG公式，可选归一化 |
| `predict_noise()` | 前向传播包装器 | 可重写以自定义transformer调用 |
| `cfg_normalize_function()` | 归一化CFG输出 | 可重写以自定义归一化 |

## 执行模式

### CFG-Parallel模式（`cfg_world_size > 1`）
- Rank 0计算正向提示预测
- Rank 1计算负向提示预测
- 通过`all_gather()`收集结果
- 所有rank本地计算CFG合并（确定性，结果相同）

### 顺序模式（`cfg_world_size == 1`）
- 单rank计算正向和负向预测
- 直接使用CFG公式合并

## 实现步骤

### 步骤1：继承`CFGParallelMixin`

```python
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
import torch.nn as nn

class YourModelPipeline(nn.Module, CFGParallelMixin):
    def diffuse(self, ...) -> torch.Tensor:
        for i, t in enumerate(timesteps):
            positive_kwargs = {...}
            negative_kwargs = {...} if do_true_cfg else None

            noise_pred = self.predict_noise_maybe_with_cfg(
                do_true_cfg=do_true_cfg,
                true_cfg_scale=true_cfg_scale,
                positive_kwargs=positive_kwargs,
                negative_kwargs=negative_kwargs,
            )

            latents = self.scheduler_step_maybe_with_cfg(
                noise_pred, t, latents, do_true_cfg
            )
        return latents
```

### 步骤2：调用`diffuse`

```python
class YourModelPipeline(nn.Module, CFGParallelMixin):
    def forward(self, ...):
        latents = self.diffuse(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            latents=latents,
            timesteps=timesteps,
            do_true_cfg=do_true_cfg,
            true_cfg_scale=guidance_scale,
        )
```

## 自定义

### 重写`predict_noise()`

适用于transformer需要自定义预测函数的情况（如Wan2.2的双transformer）：

```python
class Wan22Pipeline(nn.Module, CFGParallelMixin):
    def predict_noise(self, current_model=None, **kwargs):
        if current_model is None:
            current_model = self.transformer
        return current_model(**kwargs)[0]
```

### 重写`cfg_normalize_function()`

适用于模型有自定义归一化函数的情况（如LongCat Image）：

```python
class LongCatImagePipeline(nn.Module, CFGParallelMixin):
    def cfg_normalize_function(self, noise_pred, comb_pred, cfg_renorm_min=0.0):
        cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
        noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
        scale = (cond_norm / (noise_norm + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
        return comb_pred * scale
```

### 重写`combine_cfg_noise()`

适用于多输出模型（如视频+音频），对不同输出应用不同CFG逻辑：

```python
class MyVideoAudioPipeline(nn.Module, CFGParallelMixin):
    def combine_cfg_noise(self, positive_noise_pred, negative_noise_pred, scale, normalize):
        (video_pos, audio_pos) = positive_noise_pred
        (video_neg, audio_neg) = negative_noise_pred
        video_combined = super().combine_cfg_noise(video_pos, video_neg, scale, normalize)
        return (video_combined, audio_pos)  # 音频只使用正向，无CFG
```

## 测试

```bash
python text_to_image.py \
    --model Your-org/your-model \
    --prompt "a cup of coffee on the table" \
    --negative-prompt "ugly, unclear" \
    --cfg-scale 4.0 \
    --cfg-parallel-size 2
```

## 参考实现

| 模型 | 路径 | 模式 |
|------|------|------|
| Qwen-Image | `vllm_omni/diffusion/models/qwen_image/cfg_parallel.py` | Mixin |
| Wan2.2 | `vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py` | Mixin |
| CFGParallelMixin | `vllm_omni/diffusion/distributed/cfg_parallel.py` | 基础实现 |
