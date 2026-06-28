# vllm-omni 全模型调查报告

> 调查范围：`vllm-omni/examples/offline_inference/`、`vllm-omni/vllm_omni/diffusion/models/`、`vllm-omni/vllm_omni/model_executor/models/`

---

## 一、总览：按任务类型分类

### 🖼️ 图像生成模型 (Text-to-Image / Image-to-Image)

| 模型 | 示例脚本 | 输入 | 输出 | 核心组件 |
|------|---------|------|------|---------|
| **BAGEL** | `bagel/end2end.py` | 文本 + 可选参考图 | 图片/PNG | BAGEL MoT Transformer, VAE, SigLIP ViT, Qwen2 tokenizer |
| **FLUX** | `text_to_image/text_to_image.py` | 文本(+负提示) | 图片/PNG | FluxTransformer2D, CLIP+T5 双文本编码器, AutoencoderKL VAE, FlowMatchEuler Scheduler |
| **FLUX Kontext** | `image_to_image/image_edit.py` | 文本 + 参考图 | 图片/PNG | FluxKontextTransformer2D, CLIP+T5, AutoencoderKL VAE |
| **FLUX2** | — | 文本 + 可选参考图 | 图片/PNG | Flux2Transformer2D, Mistral3 文本编码器, PixtralProcessor, AutoencoderKLFlux2 VAE |
| **FLUX2 Klein** | — | 文本(+负提示) + 可选参考图 | 图片/PNG | Flux2Transformer2D, Qwen3 文本编码器, AutoencoderKLFlux2 VAE |
| **GLM-Image** | `glm_image/end2end.py` | 文本 + 可选源图 | 图片/PNG | AR阶段(GlmImageForConditionalGeneration) + DiT阶段(GlmImageTransformer2D + VAE), T5 glyph encoder |
| **Helios** | `helios/end2end.py` | 文本 + 可选图/视频 | 视频/MP4 | HeliosTransformer3D, UMT5, AutoencoderKLWan VAE, HeliosScheduler |
| **HunyuanImage-3** | `hunyuan_image3/image_to_text.py` | 文本(+系统提示) | 图片/PNG | HunyuanImage3Model (LLM+Diffusion), AutoencoderKLConv3D VAE, Siglip2 ViT + LightProjector |
| **LongCat-Image** | `image_to_image/image_edit.py` | 文本(含引号文字渲染) | 图片/PNG | LongCatImageTransformer2D, Qwen2.5-VL 文本编码器, AutoencoderKL VAE |
| **MammothModa2** | `mammothmodal2_preview/` | 文本 + 图片 | 图片/PNG | AR阶段(Qwen2.5-VL + MoE LLM) + DiT阶段(Transformer2D + VAE) |
| **NextStep-1.1** | `text_to_image/text_to_image.py` | 文本 | 图片/PNG | NextStepModel (LLM + flow matching head), AutoencoderKL VAE |
| **OmniGen2** | `image_to_image/image_edit.py` | 文本 + 可选参考图 | 图片/PNG | OmniGen2Transformer2D, Qwen2.5-VL, AutoencoderKL VAE |
| **Ovis-Image** | `text_to_image/text_to_image.py` | 文本(+负提示) | 图片/PNG | OvisImageTransformer2D, Qwen3 文本编码器, AutoencoderKL VAE |
| **Qwen-Image** | `text_to_image/text_to_image.py` | 文本 | 图片/PNG | QwenImageTransformer2D, Qwen2.5-VL 文本编码器, DistributedAutoencoderKLQwenImage VAE |
| **Qwen-Image-Edit** | `image_to_image/image_edit.py` | 文本 + 参考图 | 图片/PNG | QwenImageTransformer2D, Qwen2.5-VL, AutoencoderKLQwenImage VAE |
| **SD3.5** | `text_to_image/text_to_image.py` | 文本(+负提示) | 图片/PNG | SD3Transformer2D, CLIP+T5 双文本编码器, DistributedAutoencoderKL VAE |
| **Z-Image** | `text_to_image/text_to_image.py` | 文本(+负提示) | 图片/PNG | ZImageTransformer2D, AutoModel 文本编码器, DistributedAutoencoderKL VAE |

### 🎬 视频生成模型 (Text/Image-to-Video)

| 模型 | 示例脚本 | 输入 | 输出 | 核心组件 |
|------|---------|------|------|---------|
| **Wan2.2 T2V** | `text_to_video/text_to_video.py` | 文本 | 视频/MP4 | WanTransformer3D, UMT5, AutoencoderKLWan VAE, FlowUniPC Scheduler |
| **Wan2.2 I2V** | `image_to_video/image_to_video.py` | 文本 + 参考图 | 视频/MP4 | WanTransformer3D, UMT5, CLIPVisionModel, AutoencoderKLWan VAE |
| **Wan2.2 TI2V** | `image_to_video/profile_test.py` | 文本 + 可选参考图 | 视频/MP4 | WanTransformer3D, UMT5, AutoencoderKLWan VAE |
| **HunyuanVideo-1.5** | `text_to_video/text_to_video.py` | 文本(+负提示) | 视频/MP4 | HunyuanVideo15Transformer3D, Qwen2.5-VL + T5 双编码器, AutoencoderKLHunyuanVideo15 VAE |
| **HunyuanVideo-1.5 I2V** | `image_to_video/image_to_video.py` | 文本 + 参考图 | 视频/MP4 | HunyuanVideo15Transformer3D, Qwen2.5-VL + T5, SiglipVisionModel, VAE |
| **LTX2** | `text_to_video/text_to_video.py` | 文本 | 视频+可选音频 | LTX2VideoTransformer3D, Gemma3 文本编码器, Video VAE + Audio VAE + Vocoder |
| **LTX2 I2V** | `image_to_video/image_to_video.py` | 文本 + 参考图 | 视频+可选音频 | 同LTX2，继承其组件 |
| **DreamID-Omni** | `x_to_video_audio/x_to_video_audio.py` | 图片 + 音频 + 文本 | 视频+音频 | FusionModel, Wan2.2 VAE, MMAudio VAE, T5, FlowUniPC/DPM++/Euler Scheduler |

### 🔊 音频生成模型 (Text-to-Audio / Text-to-Speech)

| 模型 | 示例脚本 | 输入 | 输出 | 核心组件 |
|------|---------|------|------|---------|
| **Stable Audio** | `text_to_audio/text_to_audio.py` | 文本(+负提示) + 时长 | 音频/WAV 44.1kHz | StableAudioDiTModel, T5, AutoencoderOobleck VAE, CosineDPMSolver Scheduler |
| **CosyVoice3** | `cosyvoice3/verify_e2e_cosyvoice.py` | 文本 + 参考音频 | 音频/WAV 22kHz | Qwen2 LLM (Talker) + DiT flow matching + HiFiGAN vocoder (Code2Wav) |
| **Fish Speech S2 Pro** | `fish_speech/end2end.py` | 文本 + 可选参考音频 | 音频/WAV 44.1kHz | Qwen3 LLM (SlowAR) + FastAR (4层transformer) + DAC codec decoder |
| **MiMo-Audio** | `mimo_audio/end2end.py` | 文本 + 音频(多模式) | 音频/WAV 24kHz 或 文本 | Qwen2 LLM + local_transformer (RVQ) + MiMoAudioTokenizer |
| **Qwen2.5-Omni** | `qwen2_5_omni/end2end.py` | 文本/图片/视频/音频(多模态) | 文本 + 音频/WAV 24kHz | Thinker(ViT+Audio+Qwen2) + Talker(proj+Qwen2+codec) + Code2Wav(DiT+BigVGAN) |
| **Qwen3-Omni** | `qwen3_omni/end2end.py` | 文本/图片/视频/音频(多模态) | 文本 + 音频/WAV 24kHz | Thinker(MoE LLM+ViT+Audio) + Talker(proj+MoE+code_predictor) + Code2Wav(embedding+transformer+decoder) |
| **Qwen3-TTS** | `qwen3_tts/end2end.py` | 文本 + 可选说话人/参考音频 | 音频/WAV | Qwen3 LLM + code_predictor + ECAPA-TDNN speaker encoder + Qwen3TTSTokenizer |
| **Voxtral TTS** | `voxtral_tts/end2end.py` | 文本 + 可选参考音频/声音名 | 音频/WAV 24kHz | Text LM + Acoustic Transformer + Audio Tokenizer (RVQ encoder/decoder) |

---

## 二、示例脚本详细说明

| # | 脚本路径 | 模型 | 任务类型 | 输入 | 输出 | 所需组件 |
|---|---------|------|---------|------|------|---------|
| 1 | `bagel/end2end.py` | ByteDance-Seed/BAGEL-7B-MoT | T2I, I2I, I2T, T2T | 文本提示；可选图片 | 图片(PNG) 或 文本 | BAGEL MoT 模型, 扩散 Pipeline, OmniLLM 编排器 |
| 2 | `cosyvoice3/verify_e2e_cosyvoice.py` | FunAudioLLM/Fun-CosyVoice3-0.5B | TTS + 声音克隆 | 文本 + 参考音频(wav) | 音频(wav, 22050Hz) | CosyVoice3, Qwen tokenizer, GPT阶段, S2Mel阶段, Vocoder, mel filters |
| 3 | `custom_pipeline/image_to_image/custom_pipeline.py` | Qwen/Qwen-Image-Edit | 图像编辑(自定义Pipeline) | 图片 + 文本 | 编辑后图片 + trajectory latents | QwenImageEditPipeline, VAE, DiT, 自定义Pipeline类 |
| 4 | `custom_pipeline/image_to_image/image_edit.py` | Qwen/Qwen-Image-Edit | 图像编辑 | 一张或多张图片 + 文本 | 编辑后图片(PNG) | Qwen-Image-Edit 扩散Pipeline, VAE, DiT |
| 5 | `fish_speech/end2end.py` | fishaudio/s2-pro | TTS + 声音克隆 | 文本；可选参考音频+参考文本 | 音频(wav) | Fish Speech S2 Pro, AutoTokenizer, DAC codec |
| 6 | `glm_image/end2end.py` | zai-org/GLM-Image | T2I, I2I(编辑) | 文本；可选源图 | 图片(PNG) | GLM-Image (AR阶段 + DiT扩散阶段 + VAE) |
| 7 | `helios/end2end.py` | BestWishYsh/Helios-Base | T2V, I2V, V2V | 文本；可选图片或视频 | 视频(MP4) | Helios DiT, VAE, 可选金字塔多阶段去噪, CFG-Zero*, DMD |
| 8 | `hunyuan_image3/image_to_text.py` | tencent/HunyuanImage-3.0-Instruct | I2T (VQA) | 图片 + 文本 | 文本 | HunyuanImage-3.0-Instruct (backbone: Hunyuan-A13B-Instruct) |
| 9 | `image_to_image/image_edit.py` | Qwen/Qwen-Image-Edit / OmniGen2 | 图像编辑 | 一张或多张图片 + 文本 | 编辑后图片(PNG) | Qwen-Image-Edit / OmniGen2 扩散Pipeline, VAE, DiT |
| 10 | `image_to_video/image_to_video.py` | Wan2.2-I2V-A14B / LTX2 / HunyuanVideo-1.5 | I2V | 图片 + 文本 | 视频(MP4), 可选音频 | Wan2.2/LTX2/HunyuanVideo-1.5 扩散模型, VAE, CLIP/SigLIP 图像编码器 |
| 11 | `image_to_video/profile_test.py` | Wan2.2-TI2V-5B | I2V(性能测试) | 图片 + 文本 | 视频(MP4) | Wan2.2-TI2V-5B 扩散模型, VAE |
| 12 | `mammothmodal2_preview/run_mammothmoda2_image_summarize.py` | MammothModa2-Preview | I2T(摘要) | 图片 + 问题文本 | 文本 | MammothModa2 AR模型, Qwen2.5-VL backbone |
| 13 | `mammothmodal2_preview/run_mammothmoda2_t2i.py` | MammothModa2-Preview | T2I | 文本 | 图片(PNG) | MammothModa2 (AR阶段 + DiT扩散阶段 + VAE) |
| 14 | `mimo_audio/end2end.py` | XiaomiMiMo/MiMo-Audio-7B-Instruct | TTS, 音频理解, 对话, STT | 文本和/或音频 | 音频(wav, 24kHz) 或 文本 | MiMo-Audio (thinker + code2wav) |
| 15 | `qwen2_5_omni/end2end.py` | Qwen/Qwen2.5-Omni-7B | 多模态理解 + 语音生成 | 文本, 图片, 视频, 音频 | 文本 + 音频(wav, 24kHz) | Qwen2.5-Omni (thinker + talker + code2wav) |
| 16 | `qwen3_omni/end2end.py` | Qwen/Qwen3-Omni-30B-A3B-Instruct | 多模态理解 + 语音生成 | 文本, 音频, 图片, 视频 | 文本 + 音频(wav, 24kHz) | Qwen3-Omni (thinker + talker + code2wav) |
| 17 | `qwen3_omni/end2end_async_chunk.py` | Qwen/Qwen3-Omni-30B-A3B-Instruct | 多模态理解 + 语音生成(异步) | 文本, 音频, 图片, 视频 | 文本 + 音频(wav, 24kHz) | Qwen3-Omni, AsyncOmni 编排器 |
| 18 | `qwen3_tts/end2end.py` | Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice | TTS (CustomVoice/VoiceDesign/声音克隆) | 文本 + instruct/说话人信息；可选参考音频 | 音频(wav) | Qwen3-TTS (talker + code2wav), Qwen3TTSTokenizer |
| 19 | `text_to_audio/text_to_audio.py` | stabilityai/stable-audio-open-1.0 | T2A | 文本 | 音频(WAV, 44.1kHz) | Stable Audio Open 扩散模型 |
| 20 | `text_to_image/text_to_image.py` | Qwen/Qwen-Image / FLUX / HunyuanImage 等 | T2I | 文本 | 图片(PNG) | 扩散模型(DiT), VAE, 可选LoRA/量化/缓存 |
| 21 | `text_to_video/text_to_video.py` | Wan2.2-T2V-A14B / HunyuanVideo-1.5 | T2V | 文本 | 视频(MP4), 可选音频 | Wan2.2/HunyuanVideo 扩散模型, VAE |
| 22 | `voxtral_tts/end2end.py` | mistralai/Voxtral-4B-TTS-2603 | TTS + 声音克隆/声音选择 | 文本 + 参考音频(或声音名) | 音频(wav, 24kHz) | Voxtral TTS, MistralTokenizer |
| 23 | `x_to_video_audio/x_to_video_audio.py` | DreamID-Omni | 图+音→视频+音 | 图片 + 音频片段 + 文本 | 视频+音频(MP4) | DreamID-Omni Fusion模型, Wan2.2-TI2V-5B, MMAudio |

---

## 三、多阶段流水线架构模式

大多数模型采用**多阶段流水线**设计：

```
┌─────────────────────────────────────────────────────────────────┐
│                        输入层                                    │
│  文本 ──→ Tokenizer ──→ Text Encoder                            │
│  图片 ──→ VAE Encoder ──→ Latent                                │
│  音频 ──→ Audio Encoder / Codec Encoder                         │
├─────────────────────────────────────────────────────────────────┤
│                     理解/推理层 (Thinker)                         │
│  LLM Backbone (Qwen2/3, MoE) + Vision Encoder + Audio Encoder  │
│  输出: hidden states / text logits                               │
├─────────────────────────────────────────────────────────────────┤
│                     生成层 (Talker/DiT/AR)                       │
│  方案A: Talker (LLM → codec tokens → Code2Wav → waveform)      │
│  方案B: DiT Transformer (noise → denoise → VAE decode → image) │
│  方案C: AR (autoregressive token generation → DiT refine)      │
├─────────────────────────────────────────────────────────────────┤
│                        输出层                                    │
│  图片: VAE Decoder → PIL Image                                  │
│  视频: Video VAE Decoder → numpy frames → MP4                   │
│  音频: Codec Decoder / Vocoder → waveform → WAV                 │
└─────────────────────────────────────────────────────────────────┘
```

### 典型多阶段模型

| 模型 | 阶段1 | 阶段2 | 阶段3 |
|------|-------|-------|-------|
| **Qwen2.5-Omni** | Thinker (多模态理解) | Talker (语音token生成) | Code2Wav (波形合成) |
| **Qwen3-Omni** | Thinker (MoE多模态理解) | Talker (RVQ token生成) | Code2Wav (波形合成) |
| **MiMo-Audio** | fused_thinker_talker (理解+语音) | code2wav (波形合成) | — |
| **CosyVoice3** | Talker (LLM→语音token) | Code2Wav (DiT+HiFiGAN) | — |
| **Fish Speech** | SlowAR (文本→语义token) | FastAR (残差codebook) | DAC Decoder (波形合成) |
| **GLM-Image** | AR (先验token生成) | DiT (扩散+VAE解码) | — |
| **MammothModa2** | AR (Qwen2.5-VL理解+视觉token) | DiT (扩散+VAE解码) | — |
| **BAGEL** | OmniLLM (多模态理解) | DiT (扩散+VAE解码) | — |

---

## 四、关键组件汇总

### 文本编码器

| 编码器 | 使用模型 |
|--------|---------|
| **T5EncoderModel / UMT5EncoderModel** | FLUX, SD3, Wan2.2, Helios, GLM-Image, HunyuanVideo, Stable Audio |
| **CLIPTextModel / CLIPTextModelWithProjection** | FLUX, SD3 |
| **Qwen2.5-VL (ForConditionalGeneration)** | Qwen-Image, LongCat-Image, OmniGen2, MammothModa2, HunyuanVideo |
| **Qwen3 (ForCausalLM / Model)** | FLUX2 Klein, Ovis-Image, Fish Speech |
| **Mistral3 (ForConditionalGeneration)** | FLUX2 |
| **Gemma3 (ForConditionalGeneration)** | LTX2 |
| **AutoModel / AutoTokenizer** | Z-Image, Wan2.2, NextStep |

### 视觉编码器 (用于 I2V / 多模态理解)

| 编码器 | 使用模型 |
|--------|---------|
| **CLIPVisionModel** | Wan2.2 I2V |
| **SiglipVisionModel** | HunyuanVideo I2V |
| **Siglip2VisionModel + LightProjector** | HunyuanImage3 |
| **Qwen2.5-VisionTransformer** | Qwen2.5-Omni, Qwen3-Omni |
| **Qwen3Omni_VisionTransformer** | Qwen3-Omni |
| **GlmImageVisionModel** | GLM-Image |

### VAE / Autoencoder

| VAE 类型 | 用途 | 使用模型 |
|----------|------|---------|
| **AutoencoderKL** | 标准图像 VAE | FLUX, SD3, LongCat, OmniGen2, Ovis, MammothModa2, NextStep |
| **DistributedAutoencoderKL** | 分布式图像 VAE | SD3, Z-Image |
| **AutoencoderKLQwenImage** | Qwen-Image 专用 | Qwen-Image, Qwen-Image-Edit |
| **DistributedAutoencoderKLQwenImage** | 分布式 Qwen-Image VAE | Qwen-Image |
| **AutoencoderKLFlux2** | FLUX2 专用 | FLUX2, FLUX2 Klein |
| **AutoencoderKLConv3D** | 3D VAE | HunyuanImage3 |
| **AutoencoderKLWan / DistributedAutoencoderKLWan** | 视频 VAE (含时序压缩) | Wan2.2 |
| **AutoencoderKLHunyuanVideo15** | HunyuanVideo 视频 VAE | HunyuanVideo-1.5 |
| **AutoencoderKLLTX2Video / AutoencoderKLLTX2Audio** | LTX2 视频+音频 VAE | LTX2 |
| **AutoencoderOobleck** | 音频 VAE | Stable Audio |
| **GlmImageVQVAE** | VQ-VAE | GLM-Image |
| **AutoEncoder (Bagel)** | Bagel 专用 VAE | BAGEL |

### Scheduler (调度器)

| Scheduler | 使用模型 |
|-----------|---------|
| **FlowMatchEulerDiscreteScheduler** | FLUX, SD3, HunyuanImage3, HunyuanVideo, Qwen-Image, LongCat, OmniGen2, Ovis, Z-Image, FLUX2, MammothModa2, NextStep |
| **FlowUniPCMultistepScheduler** | Wan2.2, Helios, DreamID-Omni |
| **CosineDPMSolverMultistepScheduler** | Stable Audio |
| **HeliosScheduler** | Helios (自定义) |

### Codec / Vocoder (音频编解码器)

| 组件 | 使用模型 | 采样率 |
|------|---------|--------|
| **HiFiGAN (CausalHiFTGenerator)** | CosyVoice3 | 22050 Hz |
| **BigVGAN** | Qwen2.5-Omni, Qwen3-Omni | 24000 Hz |
| **DAC codec** | Fish Speech | 44100 Hz |
| **MiMoAudioTokenizer** | MiMo-Audio | 24000 Hz |
| **Qwen3TTSTokenizer (SpeechTokenizer)** | Qwen3-TTS | — |
| **VoxtralTTSAudioTokenizer (RVQ)** | Voxtral TTS | 24000 Hz |
| **LTX2Vocoder** | LTX2 | — |

### Speaker Encoder (说话人编码器)

| 组件 | 使用模型 |
|------|---------|
| **ECAPA-TDNN** | Qwen2.5-Omni, Qwen3-TTS |
| **Speaker Embedding Affine** | CosyVoice3 |

---

## 五、model_executor/models 模型注册表

`registry.py` 定义了 `_OMNI_MODELS` 字典，将架构名映射到 `(module_folder, module_file, class_name)` 元组，与 vLLM 的 `_VLLM_MODELS` 合并形成 `OmniModelRegistry`。

| 架构名 | 模块文件 | 主类 |
|--------|---------|------|
| Bagel | `bagel/bagel.py` | `OmniBagelForConditionalGeneration` |
| CosyVoice3 | `cosyvoice3/cosyvoice3.py` | `CosyVoice3Model` |
| FishSpeech | `fish_speech/fish_speech_slow_ar.py` | `FishSpeechSlowARForConditionalGeneration` |
| FishSpeechDACDecoder | `fish_speech/fish_speech_dac_decoder.py` | `FishSpeechDACDecoder` |
| GlmImage | `glm_image/glm_image_ar.py` | `GlmImageForConditionalGeneration` |
| HunyuanImage3 | `hunyuan_image3/hunyuan_image3.py` | `HunyuanImage3ForConditionalGeneration` |
| MammothModa2 | `mammoth_moda2/mammoth_moda2.py` | `MammothModa2ForConditionalGeneration` |
| MiMoAudio | `mimo_audio/mimo_audio.py` | `MiMoAudioForConditionalGeneration` |
| MiMoAudioLLM | `mimo_audio/mimo_audio_llm.py` | `MiMoAudioLLMForConditionalGeneration` |
| MiMoAudioCode2Wav | `mimo_audio/mimo_audio_code2wav.py` | `MiMoAudioToken2WavForConditionalGenerationVLLM` |
| Qwen2_5Omni | `qwen2_5_omni/qwen2_5_omni.py` | `Qwen2_5OmniForConditionalGeneration` |
| Qwen3OmniMoe | `qwen3_omni/qwen3_omni.py` | `Qwen3OmniMoeForConditionalGeneration` |
| Qwen3TTS | `qwen3_tts/qwen3_tts_talker.py` | `Qwen3TTSTalkerForConditionalGeneration` |
| Qwen3TTSCode2Wav | `qwen3_tts/qwen3_tts_code2wav.py` | `Qwen3TTSCode2Wav` |
| VoxtralTTS | `voxtral_tts/voxtral_tts.py` | `VoxtralTTSForConditionalGeneration` |

---

## 六、目录结构对照

```
vllm-omni/
├── examples/offline_inference/     ← 用户调用入口 (示例脚本)
│   ├── bagel/                      ← BAGEL 多模态 (T2I/I2I/I2T/T2T)
│   ├── cosyvoice3/                 ← CosyVoice3 TTS + 声音克隆
│   ├── custom_pipeline/            ← 自定义Pipeline示例
│   ├── fish_speech/                ← Fish Speech TTS + 声音克隆
│   ├── glm_image/                  ← GLM-Image (AR+DiT, T2I/I2I)
│   ├── helios/                     ← Helios 视频生成 (T2V/I2V/V2V)
│   ├── hunyuan_image3/             ← HunyuanImage3 图像理解 (I2T)
│   ├── image_to_image/             ← 图像编辑 (OmniGen2/Qwen-Edit)
│   ├── image_to_video/             ← 图生视频 (Wan/LTX2/HunyuanVideo)
│   ├── mammothmodal2_preview/      ← MammothModa2 理解+生成
│   ├── mimo_audio/                 ← MiMo-Audio 多功能音频
│   ├── qwen2_5_omni/              ← Qwen2.5-Omni 多模态+语音
│   ├── qwen3_omni/                ← Qwen3-Omni 多模态+语音
│   ├── qwen3_tts/                 ← Qwen3-TTS 语音合成
│   ├── text_to_audio/              ← Stable Audio 文生音频
│   ├── text_to_image/              ← 文生图 (FLUX/SD3/Qwen-Image等)
│   ├── text_to_video/              ← 文生视频 (Wan/HunyuanVideo)
│   ├── voxtral_tts/               ← Voxtral TTS
│   └── x_to_video_audio/          ← DreamID-Omni 图+音→视频+音
│
├── vllm_omni/diffusion/models/     ← 扩散模型实现 (Pipeline + Transformer)
│   ├── bagel/                      ← Bagel Transformer + Pipeline
│   ├── cosyvoice3_audio/           ← CosyVoice3 DiT
│   ├── dreamid_omni/              ← DreamID-Omni Fusion + Pipeline
│   ├── flux/                       ← FLUX Transformer + Pipeline + Kontext
│   ├── flux2/                      ← FLUX2 Transformer + Pipeline
│   ├── flux2_klein/               ← FLUX2 Klein (Qwen3 encoder)
│   ├── glm_image/                 ← GLM-Image Transformer + Pipeline
│   ├── helios/                    ← Helios Transformer + Scheduler + Pipeline
│   ├── hunyuan_image_3/           ← HunyuanImage3 Transformer + VAE + Pipeline
│   ├── hunyuan_video/             ← HunyuanVideo Transformer + Pipeline (T2V + I2V)
│   ├── longcat_image/             ← LongCat Transformer + Pipeline (T2I + Edit)
│   ├── ltx2/                      ← LTX2 Transformer + Pipeline (T2V + I2V)
│   ├── mammoth_moda2/             ← MammothModa2 DiT + Pipeline
│   ├── nextstep_1_1/              ← NextStep LLM + Flow Head + Pipeline
│   ├── omnigen2/                  ← OmniGen2 Transformer + Pipeline
│   ├── ovis_image/                ← Ovis Transformer + Pipeline
│   ├── qwen_image/                ← Qwen-Image Transformer + Pipeline (T2I + Edit + Layered)
│   ├── sd3/                       ← SD3 Transformer + Pipeline
│   ├── stable_audio/              ← Stable Audio DiT + Pipeline
│   ├── wan2_2/                    ← Wan2.2 Transformer + Pipeline (T2V + I2V + TI2V)
│   ├── z_image/                   ← Z-Image Transformer + Pipeline
│   ├── t5_encoder/                ← T5 文本编码器
│   └── schedulers/                ← 共享调度器 (FlowUniPC 等)
│
└── vllm_omni/model_executor/models/ ← vLLM 模型执行器 (LLM + Talker + Code2Wav)
    ├── bagel/                      ← OmniBagelForConditionalGeneration
    ├── cosyvoice3/                ← CosyVoice3 (Talker + Code2Wav)
    ├── fish_speech/               ← FishSpeech (SlowAR + FastAR + DAC Decoder)
    ├── glm_image/                 ← GlmImageForConditionalGeneration
    ├── hunyuan_image3/            ← HunyuanImage3ForConditionalGeneration
    ├── mammoth_moda2/             ← MammothModa2 (AR + DiT)
    ├── mimo_audio/                ← MiMoAudio (LLM + Code2Wav)
    ├── qwen2_5_omni/             ← Qwen2.5-Omni (Thinker + Talker + Code2Wav)
    ├── qwen3_omni/               ← Qwen3-Omni (Thinker + Talker + Code2Wav)
    ├── qwen3_tts/                ← Qwen3-TTS (Talker + Code2Wav)
    ├── voxtral_tts/              ← VoxtralTTS (AudioGeneration + AudioTokenizer)
    ├── registry.py                ← 模型注册表 (OmniModelRegistry)
    └── output_templates.py        ← 输出模板
```

---

## 七、关键发现

1. **统一架构模式**: 所有模型遵循 `Tokenizer → Encoder → Backbone → Decoder → Output` 的统一模式，通过 `OmniOutput` 在阶段间传递数据。

2. **三阶段流水线** (Omni类模型): Thinker(理解) → Talker(语音token生成) → Code2Wav(波形合成)，被 Qwen2.5-Omni、Qwen3-Omni、MiMo-Audio 等采用。

3. **两阶段流水线** (生成类模型): AR阶段(先验生成) → DiT阶段(扩散细化)，被 GLM-Image、MammothModa2、BAGEL 等采用。

4. **Flow Matching 主导**: 绝大多数扩散模型使用 `FlowMatchEulerDiscreteScheduler` 或 `FlowUniPCMultistepScheduler`，而非传统的 DDPM/DDIM。

5. **LLM 作为文本编码器趋势**: 新模型越来越多地使用 Qwen2.5-VL/Qwen3/Mistral3 等 LLM 替代传统 CLIP/T5 作为文本编码器。

6. **RVQ 多层量化**: Qwen3-Omni、MiMo-Audio、Voxtral TTS 等新模型采用多层 RVQ (Residual Vector Quantization) 提升音频质量。

7. **CUDA Graph 加速**: 多个 TTS 模型 (MiMo-Audio, Qwen3-TTS, Voxtral TTS) 在 Code2Wav 阶段使用 CUDA Graph 加速推理。
