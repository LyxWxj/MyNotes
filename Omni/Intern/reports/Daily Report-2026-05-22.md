# 娄雨轩 Daily Report — 2026-05-22

## pipeline_multi_device.py 端到端调试

### 目标

排查 `pipeline_multi_device.py` 输出噪点的原因。该脚本手动组装了 vllm-omni transformer + diffusers scheduler 的 denoising loop，用于在多卡环境下复现 `gen_image.py`（diffusers 标准 pipeline）的行为。需要使其输出与参考 pipeline 一致，从而验证 vllm-omni transformer 在多卡场景下的正确性。

---

### 调试方法

采用**逐层比对**策略：利用 `ActivationDumper` 工具类对两个版本的 pipeline 分别 dump 每步的中间张量（latents、prompt_embeds、transformer 输出、VAE 输出），通过比对 shape 和统计量（mean、std、abs_max）定位第一个出现偏差的环节。

---

### 排查过程

#### 第 1 步：比对 latents 统计量，定位偏差点

运行参考版 `gen_image.py` 和多卡版 `pipeline_multi_device.py`，将 dump 结果汇总比对。

**发现：**
- 共有 30 个 key（每步的 `latent_step_XXX`），shape 均为 `[1, 4096, 256]`
- **step 0-6 完全一致**（mean、std、abs_max 差异 < 0.001）
- **step 7 开始出现显著偏差**（std 从 0.8316 vs 0.9358 开始分裂）

**结论：** 偏差不是逐步累积的浮点误差，而是从一开始就存在但被掩盖的系统性差异。

---

#### 第 2 步：检查初始 latents 是否一致

进一步检查发现，两个版本的初始 latents 从第一步就不同（max diff = 0.078），只是在前 6 步中差异较小、未超过阈值。

**关键发现 — shape 不匹配：**

| 版本 | 初始 shape | pack 后 |
|------|-----------|---------|
| 参考版 `gen_image.py` | `[1, 64, 128, 128]` | `[1, 4096, 256]` |
| 多卡版 `pipeline_multi_device.py` | `[1, 16, 128, 128]` | `[1, 4096, 64]` |

**根因：** 多卡版使用了 `num_channels_latents = transformer.in_channels // 4 = 64 // 4 = 16`。但 `in_channels=64` 是 transformer 的 packed 特征维度（16 raw channels × 4 spatial patches），不是 VAE 的 latent 通道数。`// 4` 是多余的。

**修复：** 去掉 `// 4`，使用 `num_channels_latents = transformer.in_channels = 64`。

**但这个修复本身是错误的**（见第 5 步的纠正）。

---

#### 第 3 步：修复 transformer 输入格式

修复 shape 后运行报错：

```
RuntimeError: aclnnAddmm failed, error code is 161002
AclNN_Parameter_Error(EZ1001): The k-axis of the two inputs are different.
```

错误发生在 `self.img_in(hidden_states)`，`img_in` 是 `Linear(64, hidden_size)`，期望输入最后一维是 64。

**根因：** 多卡版传入的 latents 是 4D `[1, 64, 128, 128]`（最后一维=128），而 vllm-omni transformer 期望 3D 输入 `[B, seq_len, channels]`（最后一维=64）。参考 pipeline 内部在 `prepare_latents` 中已经 pack 为 3D 格式。

**修复：** 在传入 transformer 前做 permute+reshape：
```python
# [1, 64, 128, 128] → [1, 128, 128, 64] → [1, 16384, 64]
latents = latents.permute(0, 2, 3, 1).reshape(1, lh * lw, num_channels_latents)
```

---

#### 第 4 步：修复 timesteps 对齐

修复输入格式后能跑通 30 步，但输出仍是噪点。比对 timesteps 发现：

| step | 参考版 | 多卡版 |
|------|-------|-------|
| 1 | 1000.0000 | 1000.0000 |
| 2 | 988.2627 | 997.5013 |
| 3 | 975.9653 | 994.8318 |

**根因：** scheduler 的 `set_timesteps` 通过 `mu` 参数控制 timestep 分布，而 `mu` 依赖 `image_seq_len`。参考 pipeline 用 packed seq_len=4096，多卡版用 unpacked 16384，导致 `mu` 不同、timestep schedule 不同。

**修复：** 使用 `image_seq_len = (lh // 2) * (lw // 2) = 4096`（packed）计算 mu。

**同时发现：** 参考 pipeline 直接将 packed latents `[1, 4096, 64]` 传入 transformer，不需要 unpack 再传入。transformer 内部的 `image_rope_prepare` 会处理 packed 格式。因此 denoising loop 中不需要 unpack/re-pack 操作，直接传 packed latents 即可。

---

#### 第 5 步：修正 `num_channels_latents` 和 VAE 通道数

第 2 步中去掉 `// 4` 的修复导致了新问题：初始 latents `[1, 64, 128, 128]` pack 后是 `[1, 4096, 256]`，但 transformer 的 `img_in` 期望最后一维是 64（`in_channels`），不是 256。

**重新分析 pipeline 源码：**

```python
# pipeline prepare_latents:
shape = (batch_size, 1, num_channels_latents, height, width)
latents = randn_tensor(shape, ...)
latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)

# pipeline __call__ 中:
num_channels_latents = self.transformer.config.in_channels // 4  # = 64 // 4 = 16
```

**结论：** `num_channels_latents` 确实是 `in_channels // 4 = 16`。初始 latents 是 `[1, 1, 16, 128, 128]`，pack 后是 `[1, 4096, 64]`（64 = 16 × 4）。transformer 的 `img_in(64)` 正好匹配。

**修复：** 改回 `num_channels_latents = in_channels // 4 = 16`。同时 `img_shapes` 应该是 `[(1, 64, 64)]`（packed 空间维度），不是 `(1, 128, 128)`。

**VAE 通道数：** unpack `[1, 4096, 64]` → `[1, 16, 1, 128, 128]`（64 ÷ 4 = 16），与 VAE 的 `post_quant_conv(16, 16)` 匹配。VAE 前需要用 `vae.config.z_dim=16` 做 mean/std 归一化：

```python
latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1)
latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1)
latents = latents / latents_std + latents_mean
```

---

#### 第 6 步（最终根因）：timestep 缩放

修复以上所有问题后，输出仍是噪点。timesteps 已完全对齐（与参考版逐值一致），shapes 也正确。问题只能在 transformer forward 本身。

**最终发现 — 检查 pipeline 源码：**

```python
# pipeline __call__ 行 237:
timestep=timestep / 1000
```

pipeline 在调用 transformer 时将 timestep 除以 1000。scheduler 给出的 timestep 范围是 [20, 1000]（`Timesteps(scale=1000)`），但 transformer 期望的输入范围是 [0.02, 1.0]。

**多卡版之前直接传 `t`（如 1000.0），而参考 pipeline 传 `t/1000`（如 1.0）。** transformer 收到了比预期大 1000 倍的 timestep，导致噪声预测完全错误。

**修复：**
```python
# 修复前:
timestep=t.unsqueeze(0).to(dtype=latents.dtype, device=device_tf)

# 修复后:
timestep=(t / 1000).unsqueeze(0).to(dtype=latents.dtype, device=device_tf)
```

**修复后输出图片与参考 pipeline 一致。**

---

### 最终修复方案汇总

| # | 问题 | 错误代码 | 修复代码 | 排查手段 |
|---|------|---------|---------|---------|
| 1 | latent 通道数 | `num_channels_latents = in_channels // 4` | `= in_channels // 4`（即 16） | 比对初始 latents shape |
| 2 | transformer 输入格式 | 4D `[1, 64, 128, 128]` | packed 3D `[1, 4096, 64]` 直接传入 | NPU FA 报错 |
| 3 | img_shapes | `[(1, 128, 128)]` | `[(1, 64, 64)]`（packed 空间维度） | 检查 pipeline 源码 |
| 4 | image_seq_len | 16384（unpacked） | 4096（packed） | 比对 timesteps |
| 5 | VAE 归一化 | 无或错误 | `z_dim=16` 的 mean/std | 检查 pipeline 源码 |
| 6 | **timestep 缩放** | `t`（范围 [20, 1000]） | `t / 1000`（范围 [0.02, 1.0]） | 检查 pipeline 源码行 237 |

---

### 经验总结

1. **不要假设 API 行为**：scheduler 给出的 timestep 和 transformer 期望的 timestep 不在同一尺度，需要检查 pipeline 的实际调用方式。
2. **检查 pipeline 源码是最终手段**：很多隐含行为（如 `/1000` 缩放、`in_channels // 4`）不会在文档中说明，必须阅读源码。
3. **逐层比对是有效的调试方法**：通过 dump 中间张量并比对统计量，可以快速定位第一个偏差环节，避免盲目猜测。
4. **pack/unpack 是 QwenImage 的核心操作**：`in_channels=64` 是 packed 特征维度，实际 latent 通道数是 16。transformer、scheduler 都在 packed 空间工作，只有 VAE 需要 unpack 回原始空间。

---

### 当前状态

- [x] 所有 6 个问题已修复
- [x] **验证通过：** `pipeline_multi_device.py` 输出图片与 `gen_image.py` 一致
- [x] **验证通过：** `pipeline_qwen_image_multi_instance.py`（vllm-omni 原生多卡 pipeline）修复 timestep/1000 后，输出正确图像

### 移植到 vllm-omni 的修改

`pipeline_qwen_image_multi_instance.py` 中 `serve_diffusion_loop` 的 denoising loop，只需修改 1 处：

```python
# 改前:
timestep = t.expand(latents.shape[0]).to(
    device=latents.device, dtype=latents.dtype
)

# 改后:
timestep = (t / 1000.0).expand(latents.shape[0]).to(
    device=latents.device, dtype=latents.dtype
)
```

`multi_instance_scheduler.py` 无需修改，所有逻辑已正确。

