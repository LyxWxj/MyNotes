---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm
---

# Multimodal Inputs

## 概述

vLLM 支持向多模态模型传递图像、视频、音频等多模态输入，兼容离线推理和 OpenAI 兼容的在线服务。

## 离线推理

### 数据格式

- `prompt`：遵循 HuggingFace 文档格式
- `multi_modal_data`：字典，键为模态类型（`image`、`video`、`audio`）

### 图像输入

```python
outputs = llm.generate({
    "prompt": "USER: <image>\nWhat is this?\nASSISTANT:",
    "multi_modal_data": {"image": pil_image},
})
```

- 支持单图和多图输入
- 多图传入列表：`{"image": [img1, img2]}`
- `llm.chat()` 支持 `image_url`、`image_pil`、`image_embeds` 格式
- RGBA 图像自动转 RGB，可通过 `rgba_background_color` 自定义背景色

### 视频输入

```python
mm_data = {"video": video_frames}  # NumPy 数组或 torch.Tensor 列表
```

### 音频输入

```python
mm_data = {"audio": (audio_array, sampling_rate)}
```

- 支持长音频分块：`split_audio()` 在静音处分割
- 自动声道归一化（Whisper、Qwen2-Audio 等模型）

### Embedding 输入

直接传入预计算的 embedding tensor，需启用 `enable_mm_embeds=True`：

```python
llm = LLM(model="...", enable_mm_embeds=True)
mm_data = {"image": image_embeds}  # shape: (..., hidden_size)
```

### 缓存输入

通过 `multi_modal_uuids` 提供稳定 ID，避免重复哈希：

```python
outputs = llm.generate({
    "prompt": prompt,
    "multi_modal_data": {"image": [img_a, img_b]},
    "multi_modal_uuids": {"image": ["sku-1234-a", None]},
})
```

UUID 命中缓存时可跳过发送媒体数据。

## 在线服务

### OpenAI Chat Completions API

- 图像：`image_url` 类型（支持 URL 和本地文件路径）
- 视频：`video_url` 类型
- 音频：`input_audio`（base64）或 `audio_url` 类型
- Embedding：`image_embeds` 类型，需 `--enable-mm-embeds` 启用

### 安全配置

```bash
vllm serve <model> --allowed-media-domains example.com --allowed-local-media-path /path
```

- `VLLM_MEDIA_URL_ALLOW_REDIRECTS=0` 禁止重定向
- `VLLM_IMAGE_FETCH_TIMEOUT` / `VLLM_VIDEO_FETCH_TIMEOUT` / `VLLM_AUDIO_FETCH_TIMEOUT` 控制超时

### 视频帧恢复

`--media-io-kwargs '{"video": {"frame_recovery": true}}'` 启用损坏视频的帧恢复。
