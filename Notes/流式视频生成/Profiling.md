# 视频/音频生成模型 Profiling 报告

> 测试日期：2026-06-15
> 测试环境：Ascend NPU 910B 64GB × 8, aarch64, CANN 8.5.1, Python 3.11

---

## 1. Wan2.2-TI2V-5B-Diffusers

> 框架：vllm-omni | 类型：双向 Diffusion
> 测试日期：2026-06-17

### 1.1 测试配置

| 项目 | 配置 |
|------|------|
| 模型 | Wan2.2-TI2V-5B-Diffusers |
| 任务 | Image-to-Video |
| 输入图片 | i2v_input.JPG |
| Prompt | "A Cat" |
| 分辨率 / 帧数 | 832×480, 120 帧 |
| 推理步数 | 4 |
| 并行配置 | ulysses_degree=8 (8 NPU) |

### 1.2 端到端性能

| 指标 | 值 |
|------|-----|
| 端到端耗时 | 21.73 s |
| Pipeline Forward | 20.11 s |
| 每步去噪耗时 | ~1.23 s |
| 每秒生成帧数 | ~5.52 fps (120帧/21.73s) |

### 1.3 瓶颈

```
vae.decode    ██████████████████████████████████████████████████████  61.9%
denoising     █████████████████                                       18.2%
vae.encode    ████████                                                13.8%
text_encoder  ██                                                       4.1%
其他          █                                                        2.2%
```

---

## 2. Qwen3-Omni-30B-A3B-Instruct

> 框架：vllm-omni v0.18.0 | 类型：LLM 多阶段 (Thinker → Talker → Code2Wav)

### 2.1 测试配置

| 项目 | 配置 |
|------|------|
| 模型 | Qwen3-Omni-30B-A3B-Instruct |
| 任务 | Text → Text + Audio |
| Prompt | "What is the capital of France? Answer in one sentence." |
| Stage 数量 | 3 |
| 并行配置 | Thinker TP=2, Talker TP=1, Code2Wav TP=1 |

### 2.2 各 Stage 耗时

| Stage | 名称 | 耗时 (ms) | 耗时 (s) | 占比 |
|-------|------|-----------|----------|------|
| Stage 0 | Thinker | 811.76 | 0.81 | 1.8% |
| Stage 1 | Talker | 7,759.30 | 7.76 | 17.2% |
| Stage 2 | Code2Wav | 35,051.32 | 35.05 | 77.6% |
| Stage 间传输 | - | 0.00 | 0.00 | 0.0% |
| **E2E 总计** | | **45,089.83** | **45.09** | 100% |

### 2.3 端到端性能

| 指标 | 值 |
|------|-----|
| 端到端耗时 | 45.09 s |
| 总 token 数 | 101 |
| 吞吐量 | 2.23 tokens/s |
| 文本输出 | "The capital of France is Paris." |
| 音频输出 | 24kHz WAV, ~122 KB |

### 2.4 瓶颈

```
Code2Wav   ████████████████████████████████████████████████████████████  77.6%
Talker     ██████████████                                               17.2%
Thinker    ██                                                           1.8%
```

---

## 3. CausalForcingWan2.1-T2V-1.3B

> 框架：flashdreams (NVIDIA) | 类型：自回归流式视频生成

### 3.1 测试配置

| 项目 | 配置 |
|------|------|
| 模型 | FastVideo/CausalForcingWan2.1-T2V-1.3B |
| 任务 | Text-to-Video（自回归流式） |
| Prompt | "A cat walking on the grass, sunny day, cinematic lighting." |
| 分辨率 | 832×480 (latent 104×60) |
| 配置 | len_t=1, window_size_t=6, sampling_steps=4, bf16 |
| AR 步数 | 21 |
| 并行策略 | 单卡 NPU |

### 3.2 各 AR Step 组件耗时 (ms)

| AR | encode | diffuse | decode | finalize | total |
|----|--------|---------|--------|----------|-------|
| 0 (warmup) | 0.3 | 694.5 | 7,739.1 | 203.2 | 8,638.6 |
| 1 | 0.1 | 643.0 | 155.8 | 321.1 | 1,120.5 |
| 2 | 0.1 | 653.9 | 157.0 | 355.4 | 1,166.8 |
| 3 | 0.1 | 663.3 | 156.7 | 331.7 | 1,152.3 |
| 4-20 (稳态) | ~0.1 | 383 – 440 | ~154 | 188 – 214 | 727 – 810 |

### 3.3 稳态平均 (AR 4-20)

| 组件 | 平均耗时 (ms) | 占比 |
|------|-------------|------|
| encode | ~0.1 | 0.0% |
| diffuse (DiT) | ~395 | 53.2% |
| decode (VAE) | ~154 | 20.8% |
| finalize (KV cache) | ~193 | 26.0% |
| **total** | **~742** | 100% |

### 3.4 端到端性能

| 指标 | 值 |
|------|-----|
| 总帧数 | 21 |
| 总生成时间 | ~15 s |
| 平均每 AR step | 742 ms |
| 平均每帧 | ~714 ms |
| 显存峰值 | ~20 GB |
| text_encoder | 1,930 ms |

---

## 4. Wan2.2-I2V-A14B-INT8 (vLLM-Omni)

> 框架：vllm-omni | 类型：双向 Diffusion (INT8 量化)
> 测试日期：2026-06-17

### 4.1 测试配置

| 项目 | 配置 |
|------|------|
| 模型 | Wan2.2-I2V-A14B-Diffusers_INT8 |
| 任务 | Image-to-Video |
| 输入图片 | i2v_input.JPG |
| Prompt | "A Cat" |
| 分辨率 / 帧数 | 832×480, 120 帧 |
| 推理步数 | 4 |
| CFG 引导 | guidance_scale=5.0, guidance_scale_high=6.0 |
| 并行配置 | ulysses_degree=4, tensor_parallel_size=2 (8 NPU) |
| 量化方式 | INT8 block-wise (128×128) |

### 4.2 Pipeline 各阶段耗时

| 阶段 | 耗时 (s) | 占比 |
|------|----------|------|
| text_encoder.forward (×16) | ~0.88 | 1.6% |
| vae.encode | ~5.93 | 11.0% |
| denoising (4 steps, ~8.68s/step) | ~34.72 | 64.5% |
| vae.decode | ~10.18 | 18.9% |
| 后处理 | ~0.46 | 0.9% |
| **Pipeline Forward** | **~51.60** | 100% |

### 4.3 DiffusionEngine 汇总

```
DiffusionEngine.step breakdown:
  preprocess       =    29.50 ms
  add_req_and_wait = 53314.16 ms  (扩散推理 + VAE 编解码)
  postprocess      =   464.44 ms
  total            = 53809.00 ms   (53.81 s)
```

### 4.4 端到端性能

| 指标 | 值 |
|------|-----|
| 端到端耗时 | 53.82 s |
| Pipeline Forward | 51.60 s |
| 每步去噪耗时 | ~8.68 s |
| 每秒生成帧数 | ~2.23 fps (120 帧/53.8s) |

### 4.5 瓶颈

```
denoising     ████████████████████████████████████████████████████  64.5%
vae.decode    ██████████████                                        18.9%
vae.encode    ███████                                               11.0%
text_encoder  █                                                      1.6%
其他          █                                                      0.9%
```

---

## 5. 模型对比总览

| 维度 | TI2V-5B (120帧) | I2V-A14B-INT8 (120帧) | Qwen3-Omni-30B | CausalForcingWan2.1-1.3B |
|------|-----------------|----------------------|----------------|--------------------------|
| 框架 | vllm-omni | vllm-omni | vllm-omni | flashdreams |
| 模型类型 | 双向 Diffusion | 双向 Diffusion (INT8) | LLM 多阶段 | 自回归流式 Diffusion |
| 参数量 | 5B | 14B (INT8 量化) | 30B (3B active) | 1.3B |
| 任务 | Image→Video | Image→Video | Text→Text+Audio | Text→Video |
| 分辨率 | 832×480 | 832×480 | - | 832×480 |
| 帧数 | 120 | 120 | - | 21 |
| 推理步数 | 4 | 4 | - | 4/AR |
| 并行策略 | ulysses=8 (8 NPU) | ulysses=4, TP=2 (8 NPU) | TP=2 (Thinker) | 单 NPU |
| 端到端耗时 | 21.73 s | 53.82 s | 45.09 s | ~15 s |
| 每秒帧数 | ~5.52 fps | ~2.23 fps | - | ~1.40 fps |
| 瓶颈 | VAE 解码 (62%) | DiT 去噪 (65%) | Code2Wav (78%) | DiT (53%) |

### 5.1 各模型瓶颈对比

| 模型 | 瓶颈组件 | 占比 |
|------|----------|------|
| I2V-A14B-INT8 (vllm-omni) | DiT 去噪 | 64.5% |
| TI2V-5B (vllm-omni) | VAE 解码 | 61.9% |
| Qwen3-Omni-30B | 音频波形生成 (Code2Wav) | 77.6% |
| CausalForcingWan2.1 (flashdreams) | DiT 去噪推理 | 53.2% |
