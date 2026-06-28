# MiniCPM-o 4.5 模型分析

> [!info] 分析目标
> 本文档针对**流式语音生成**任务需求，深入分析 MiniCPM-o 4.5 在以下维度的实现现状：
> 1. **Duplex 双工系统** - s2t + tts 并行处理能力
> 2. **Session 级别上下文管理** - 会话状态维护与复用
> 3. **Audio Chunk 调度** - context-chunk vs request-chunk 机制
> 4. **双 Token 调度** - t token（文本）和 z token（audio latent）的合并策略

---

## 1. 模型架构概述

> [!important] 核心架构
> MiniCPM-o 4.5 采用**串行两阶段流水线**设计，**不是真正的 Duplex 系统**

```
┌─────────────────────────────────────────────────────────────┐
│                    Stage 0: Thinker (思考者)                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐ │
│  │ 图像编码器 │  │ 视频编码器 │  │ 音频编码器 │  │ 3D Resampler│ │
│  └──────────┘  └──────────┘  └──────────┘  └─────────────┘ │
│                           ↓                                 │
│  ┌─────────────────────────────────────────────────────────┐│
│  │            Qwen2 LLM (多模态理解 + 文本生成)            ││
│  └─────────────────────────────────────────────────────────┘│
│                           ↓                                 │
│              输出: t token (文本) + hidden states            │
└─────────────────────────────────────────────────────────────┘
                           ↓ llm2tts()
┌─────────────────────────────────────────────────────────────┐
│                    Stage 1: Talker (说话者)                  │
│  ┌─────────────────────────────────────────────────────────┐│
│  │        MiniCPMTTS (离散音频token生成)                   ││
│  └─────────────────────────────────────────────────────────┘│
│                           ↓                                 │
│  ┌─────────────────────────────────────────────────────────┐│
│  │        Token2Wav (音频token → 波形)                     ││
│  └─────────────────────────────────────────────────────────┘│
│                           ↓                                 │
│              输出: z token (audio codec) → 波形              │
└─────────────────────────────────────────────────────────────┘
```

### 1.1 两阶段串行执行

> [!warning] 关键限制
> Thinker 和 Talker **必须串行执行**，不支持并行处理

```python
# minicpmo_4_5_omni.py:86-117
if self.model_stage == "llm":
    # Stage 0: Thinker - 多模态理解 + 文本生成
    self.thinker = init_vllm_registered_model(...)
    self.talker = None  # Talker 不参与此阶段

elif self.model_stage == "tts":
    # Stage 1: Talker - 语音合成
    self.thinker = None  # Thinker 不参与此阶段
    self.talker = init_vllm_registered_model(...)
```

**执行流程**：
1. Stage 0 (Thinker): 接收多模态输入 → 生成 t token + hidden states
2. llm2tts(): 提取 TTS 区域，构建 additional_information
3. Stage 1 (Talker): 接收 hidden states → 生成 z token → 合成波形

**对你任务的启示**：
> ⚠️ **MiniCPM-o 4.5 不是 Duplex 系统**
> 如果要实现真正的 Duplex（同时收听和说话），需要：
> 1. 新增 s2t 模块（实时语音识别）
> 2. 设计并行执行框架（Thinker/Talker/s2t 同时运行）
> 3. 实现 Session 状态机（Listening → Thinking → Speaking）

---

## 2. Session 级别上下文管理

> [!note] 核心机制
> 通过 `additional_information` 实现**单次请求级别**的上下文传递

### 2.1 上下文传递链

```python
# minicpmo_4_5_omni.py:278-281
# Thinker 输出
return OmniOutput(
    text_hidden_states=text_hidden_states,
    multimodal_outputs={"latent": text_hidden_states},
)
```

```python
# stage_input_processors/minicpmo_4_5_omni.py:110-115
# llm2tts() 桥接函数
additional_information = {
    "prompt_embeds": prompt_hidden,           # prompt 隐藏状态
    "prompt_token_ids": list(prompt_token_ids),  # prompt token IDs
    "llm_output_token_ids": list(llm_output_ids),  # 生成的 token IDs
    "llm_output_text": [thinker_text],        # 生成的文本
    "tts_token_ids": tts_token_ids_slice,     # TTS 区域 tokens
    "tts_hidden_states": tts_hidden_slice,    # TTS 区域隐藏状态
}
```

```python
# minicpmo_4_5_omni_tts.py:398-403
# Talker 接收
tts_token_ids = additional_information.get("tts_token_ids")
tts_hidden_states = additional_information.get("tts_hidden_states")
tts_text = additional_information.get("llm_output_text", [""])
```

### 2.2 上下文信息类型

> [!abstract] 传递的上下文信息

| 信息类型 | 内容 | 用途 |
|---------|------|------|
| `prompt_embeds` | prompt 隐藏状态 | 说话人嵌入提取 |
| `prompt_token_ids` | prompt token IDs | 定位 spk_bos/spk_eos |
| `llm_output_token_ids` | 生成的 token IDs | 解码 TTS 文本 |
| `llm_output_text` | 生成的文本 | TTS 内容提取 |
| `tts_token_ids` | TTS 区域 tokens | Talker 输入 |
| `tts_hidden_states` | TTS 区域隐藏状态 | 语义嵌入 |

### 2.3 Session 管理现状

> [!warning] 关键限制
> 当前实现是**单次请求级别**，**不是 Session 级别**

**缺失的 Session 功能**：
- ❌ Session 状态缓存（跨请求复用）
- ❌ KVCache 的 Session 级管理
- ❌ Session 生命周期管理（创建、复用、清理）
- ❌ 多阶段状态流转（Listening → Thinking → Speaking）

**对你任务的启示**：
> 需要设计 Session-aware 的上下文管理：
> 1. **Session 状态缓存**：维护会话级别的 KVCache
> 2. **上下文复用**：避免每个 chunk 重复计算
> 3. **生命周期管理**：Session 创建、chunk 间复用、结束清理
> 4. **状态机设计**：管理多阶段状态流转

---

## 3. Audio Chunk 调度机制

> [!success] 关键发现
> MiniCPM-o 4.5 的 TTS 模块已经实现了**流式 chunk 处理**

### 3.1 Chunk 处理配置

```python
# minicpmo_4_5_omni_tts.py:262-264
STREAM_THRESHOLD = int(os.environ.get("MINICPMO45_TTS_STREAM_THRESHOLD", "2500"))  # ~100s @ 25Hz
CHUNK_SIZE = int(os.environ.get("MINICPMO45_TTS_STREAM_CHUNK", "50"))  # ~2s per chunk
MIN_TAIL = 6  # must exceed flow.pre_lookahead_len (typically 3)
```

**参数说明**：
| 参数 | 默认值 | 含义 |
|------|--------|------|
| `STREAM_THRESHOLD` | 2500 tokens (~100s) | 短音频一次性处理的阈值 |
| `CHUNK_SIZE` | 50 tokens (~2s) | 流式处理的 chunk 粒度 |
| `MIN_TAIL` | 6 tokens | 最小尾部长度，避免过小 chunk |

### 3.2 Chunk 处理逻辑

```python
# minicpmo_4_5_omni_tts.py:266-309
if num_tokens <= STREAM_THRESHOLD:
    # 短音频：一次性处理（< 100s）
    wav_bytes = self.audio_tokenizer(token_list, prompt_wav_path)
else:
    # 长音频：流式 chunk 处理（>= 100s）
    boundaries = []
    i = 0
    while i < num_tokens:
        end = min(i + CHUNK_SIZE, num_tokens)
        # 避免过小的尾部 chunk
        if 0 < num_tokens - end < MIN_TAIL:
            end = num_tokens
        boundaries.append((i, end))
        i = end
    
    # 流式生成
    stream_cache, hift_cache_dict = self.audio_tokenizer.set_stream_cache(prompt_wav_path)
    
    pieces = []
    for idx, (s, e) in enumerate(boundaries):
        is_last = idx == len(boundaries) - 1
        wav_np = self.audio_tokenizer.stream(
            token_list[s:e],
            prompt_wav_path,
            last_chunk=is_last,
            return_waveform=True,
        )
        pieces.append(np.asarray(wav_np).reshape(-1))
    
    waveform = np.concatenate(pieces, axis=0).astype(np.float32)
```

### 3.3 Chunk 类型对比

> [!important] 与任务需求的对比

| Chunk 类型 | MiniCPM-o 4.5 现状 | 任务需求 | 差距 |
|-----------|-------------------|---------|------|
| **TTS 内部 chunk** | ✅ 50 tokens (~2s) | - | 已实现 |
| **request-chunk** | ❌ 无 | 200ms~1s 增量包 | 需要新增 |
| **context-chunk** | ❌ 无 | 10s 音频块 + 上下文 | 需要新增 |
| **滑动窗口** | ❌ 无 | 长序列音频处理 | 需要新增 |

### 3.4 流式处理优化

```python
# minicpmo_4_5_omni_tts.py:256-261
# 长音频 OOM 优化
# 对于长输出，一次性 vocoder 路径会运行完整的 O(N^2) 自注意力
# 当 N 超过几千时会在 24GB 卡上 OOM
# 切换到分块/流式 vocoder，将 flow attention 缓存截断到 prompt_len + 100 步
```

**对你任务的启示**：
> MiniCPM-o 4.5 的 chunk 是 **TTS 内部的**，不是**请求级别**的
> 需要设计：
> 1. **request-chunk (200ms~1s)**：请求级别的增量语音包
> 2. **context-chunk (10s)**：包含上下文信息的音频块
> 3. **滑动窗口**：处理长序列音频的机制
> 4. **chunk 调度器**：管理 chunk 的到达、处理、输出

---

## 4. 双 Token 调度

> [!note] 核心概念
> MiniCPM-o 4.5 中存在两种 token：
> - **t token**: 文本 token（LLM 输出）
> - **z token**: Audio latent token（TTS codec token）

### 4.1 Token 类型定义

| Token 类型 | 来源 | 用途 | 格式 |
|-----------|------|------|------|
| **t token** | Thinker (LLM) | 文本表示 | 文本 token IDs |
| **z token** | Talker (TTS) | 音频表示 | Audio codec token IDs |
| **hidden states** | Thinker (LLM) | 语义桥梁 | 连续向量 |

### 4.2 Token 处理流程

```python
# minicpmo_4_5_omni_tts.py:192-196
# t token 和 z token 的合并
llm_embeds = tts.emb_text(tts_token_ids.to(device))  # t token 嵌入
hidden_embeds = tts.projector_semantic(tts_hidden_states.to(device=device, dtype=dtype))  # z token 嵌入

if getattr(tts.config, "normalize_projected_hidden", False):
    hidden_embeds = F.normalize(hidden_embeds, p=2, dim=-1)

tts_embeds = llm_embeds + hidden_embeds  # t + z 合并
```

**Token 转换链**：
```
Thinker 输出
    ↓
t token (文本) + hidden states (语义)
    ↓
llm2tts() 提取 TTS 区域
    ↓
Talker 处理:
  - emb_text(t_token) → 文本嵌入 (llm_embeds)
  - projector_semantic(hidden) → 语义嵌入 (hidden_embeds)
  - tts_embeds = llm_embeds + hidden_embeds  # t + z 合并
    ↓
MiniCPMTTS.generate() → z token (audio codec token)
    ↓
Token2Wav(z_token) → 波形
```

### 4.3 双 Token 调度现状

> [!warning] 关键限制
> 当前实现是**串行处理**：先生成 t token，再生成 z token

```python
# minicpmo_4_5_omni.py:258-265
# Stage 0: Thinker 生成 t token
thinker_output = self.thinker(
    input_ids=thinker_input_ids,
    positions=thinker_positions,
    intermediate_tensors=intermediate_tensors,
    inputs_embeds=thinker_inputs_embeds,
    **kwargs,
)

# Stage 1: Talker 生成 z token（必须等 Thinker 完成）
talker_result = self.talker(
    input_ids=input_ids,
    positions=positions,
    inputs_embeds=inputs_embeds,
    additional_information=talker_info,
)
```

**调度问题**：
- ❌ t token 和 z token **串行生成**，无重叠
- ❌ 无统一调度器管理两种 token
- ❌ GPU 利用率低（Thinker 和 Talker 交替空闲）

### 4.4 mt1z 调度需求

> [!important] 任务目标
> 实现 **mt1z 调度**：统一管理 t token 和 z token，优化流水线

**mt1z 调度特性**：
1. **混合调度**：t token 和 z token 在同一调度器中管理
2. **流水线重叠**：Thinker 生成 t token 的同时，Talker 处理前一批 z token
3. **资源优化**：减少 GPU 空闲时间，提高利用率
4. **延迟优化**：通过重叠减少端到端延迟

**对你任务的启示**：
> 需要设计：
> 1. **统一调度器**：同时管理 t token 和 z token
> 2. **流水线重叠**：Thinker 和 Talker 的并行执行
> 3. **tz 合并策略**：优化两种 token 的合并时机
> 4. **资源分配**：避免 GPU 争用

---

## 5. Async CPU Zero-overhead

> [!tip] 优化机会
> MiniCPM-o 4.5 的 CPU 开销主要在以下环节

### 5.1 CPU 开销分析

| 环节 | CPU 开销 | 原因 | 优化方向 |
|------|---------|------|---------|
| **Token2Wav 声码器** | 高 | 10步迭代、flow attention | GPU 加速 |
| **音频后处理** | 中 | numpy 拼接、类型转换 | 异步化 |
| **torchaudio 操作** | 中 | 文件 I/O、格式转换 | GPU 实现 |
| **Tokenizer** | 低 | 文本分词 | 已优化 |

### 5.2 Token2Wav CPU 开销

```python
# minicpmo_4_5_omni_tts.py:156
# Token2Wav 初始化（CPU 密集型）
self.audio_tokenizer = _Token2wav(token2wav_dir, float16=False, n_timesteps=10)
```

**问题**：
- Token2Wav 使用 CPU 进行 flow attention 计算
- 10步迭代，每步都是 O(N^2) 复杂度
- 长音频时 CPU 成为瓶颈

### 5.3 音频后处理 CPU 开销

```python
# minicpmo_4_5_omni_tts.py:303-304
# 音频拼接（CPU 密集型）
pieces.append(np.asarray(wav_np).reshape(-1))
waveform = np.concatenate(pieces, axis=0).astype(np.float32)
```

**问题**：
- numpy 操作在 CPU 上执行
- 多次内存拷贝
- 类型转换开销

### 5.4 torchaudio 优化

```python
# minicpmo_4_5_omni_tts.py:39-67
# 使用 soundfile 替代 torchaudio（兼容性优化）
def _patched_load(uri, *args, **kwargs):
    try:
        return _orig_load(uri, *args, **kwargs)
    except Exception:
        import numpy as _np
        import soundfile as _sf
        data, sr = _sf.read(uri, dtype="float32", always_2d=True)
        wav = torch.from_numpy(_np.ascontiguousarray(data.T))
        return wav, sr
```

**对你任务的启示**：
> 需要实现 **Async CPU Zero-overhead**：
> 1. **Token2Wav GPU 加速**：将 flow attention 移到 GPU
> 2. **异步后处理**：音频拼接、类型转换异步化
> 3. **零拷贝**：减少内存拷贝
> 4. **流水线重叠**：CPU 和 GPU 操作重叠执行

---

## 6. 针对任务的改进方案

> [!idea] 基于 MiniCPM-o 4.5 的改进设计

### 6.1 Duplex 系统扩展

```python
class MiniCPMO45Duplex(nn.Module):
    """扩展 MiniCPM-o 4.5 为 Duplex 系统"""
    
    def __init__(self):
        # 新增 s2t 模块
        self.s2t = WhisperSTTModule()
        
        # 原有模块
        self.thinker = MiniCPMO45Thinker()
        self.talker = MiniCPMO45Talker()
        
        # 状态机
        self.state_machine = DuplexStateMachine()
        
        # 并行执行器
        self.executor = ThreadPoolExecutor(max_workers=3)
    
    def forward(self, audio_stream):
        """并行处理：s2t 和 thinker/talker 同时运行"""
        
        # 状态机管理
        state = self.state_machine.get_state()
        
        if state == "listening":
            # 并行：s2t 识别 + thinker 准备
            s2t_future = self.executor.submit(self.s2t, audio_stream)
            # ... 其他并行任务
            
        elif state == "thinking":
            # Thinker 处理
            thinker_output = self.thinker(...)
            
        elif state == "speaking":
            # Talker 生成语音
            waveform = self.talker(...)
```

### 6.2 Session-aware Chunk 调度器

```python
class SessionAwareChunkScheduler:
    """Session 级别的 Chunk 调度器"""
    
    def __init__(self):
        self.session_cache = {}  # Session 级缓存
        self.context_chunks = {}  # context-chunk 管理
        self.request_queue = Queue()  # request-chunk 队列
    
    def schedule(self, request_chunk):
        """调度 request-chunk"""
        
        # 1. 识别 session
        session_id = request_chunk.session_id
        
        # 2. 获取 context-chunk (10s)
        context = self.get_context_chunk(session_id)
        
        # 3. 调度 request-chunk (200ms~1s)
        return self.schedule_with_context(request_chunk, context)
    
    def get_context_chunk(self, session_id):
        """获取 context-chunk（10s 音频块 + 上下文）"""
        
        if session_id not in self.context_chunks:
            # 创建新的 context-chunk
            self.context_chunks[session_id] = {
                "audio": [],  # 10s 音频
                "asr_text": [],  # ASR 文本
                "ttff": None,  # 首音频时间
            }
        
        return self.context_chunks[session_id]
    
    def update_context(self, session_id, new_audio, asr_text):
        """更新 context-chunk（滑动窗口）"""
        
        context = self.context_chunks[session_id]
        
        # 滑动窗口：保留最近 10s
        context["audio"].append(new_audio)
        if len(context["audio"]) > 10:  # 10s 窗口
            context["audio"].pop(0)
        
        context["asr_text"].append(asr_text)
        if len(context["asr_text"]) > 10:
            context["asr_text"].pop(0)
```

### 6.3 mt1z 统一调度器

```python
class MT1ZScheduler:
    """t token 和 z token 的统一调度器"""
    
    def __init__(self):
        self.t_token_queue = Queue()  # 文本 token 队列
        self.z_token_queue = Queue()  # 音频 token 队列
        self.token_buffer = []  # token 缓冲区
    
    def schedule(self):
        """统一调度 t token 和 z token"""
        
        while True:
            # 获取 t token
            if not self.t_token_queue.empty():
                t_token = self.t_token_queue.get()
                self.process_t_token(t_token)
            
            # 获取 z token
            if not self.z_token_queue.empty():
                z_token = self.z_token_queue.get()
                self.process_z_token(z_token)
            
            # tz 合并优化
            self.merge_tokens()
    
    def process_t_token(self, t_token):
        """处理 t token（文本）"""
        # 发送给 Talker 生成 z token
        pass
    
    def process_z_token(self, z_token):
        """处理 z token（音频）"""
        # 发送给 Token2Wav 生成波形
        pass
    
    def merge_tokens(self):
        """tz 合并：优化两种 token 的处理"""
        
        # 检查是否有足够的 t token 和 z token
        if len(self.token_buffer) >= 2:
            t_token, z_token = self.token_buffer.pop(0), self.token_buffer.pop(0)
            
            # 合并处理
            merged = self.merge_t_z(t_token, z_token)
            
            # 发送给下游
            self.send_to_downstream(merged)
    
    def merge_t_z(self, t_token, z_token):
        """合并 t token 和 z token"""
        
        # 策略 1: 简单拼接
        # return torch.cat([t_token, z_token])
        
        # 策略 2: 加权融合
        # return t_token * 0.5 + z_token * 0.5
        
        # 策略 3: 注意力融合
        # return attention_merge(t_token, z_token)
        
        pass
```

### 6.4 Async CPU Zero-overhead 优化

```python
class AsyncCPUOptimizer:
    """Async CPU Zero-overhead 优化器"""
    
    def __init__(self):
        self.gpu_stream = torch.cuda.Stream()
        self.cpu_executor = ThreadPoolExecutor(max_workers=4)
    
    async def process_audio_async(self, audio_tokens):
        """异步处理音频"""
        
        # 1. GPU 上运行 Token2Wav
        with torch.cuda.stream(self.gpu_stream):
            waveform_gpu = self.token2wav_gpu(audio_tokens)
        
        # 2. CPU 上异步处理后处理
        waveform_cpu = await self.cpu_executor.submit(
            self.post_process_cpu, waveform_gpu
        )
        
        # 3. 零拷贝返回
        return waveform_cpu
    
    def token2wav_gpu(self, audio_tokens):
        """GPU 上的 Token2Wav"""
        # 将 flow attention 移到 GPU
        # 使用 CUDA kernel 实现
        pass
    
    def post_process_cpu(self, waveform_gpu):
        """CPU 上的后处理（异步）"""
        # 异步 numpy 操作
        # 减少内存拷贝
        pass
```

---

## 7. 代码结构分析

> [!note] 关键文件

```
vllm_omni/model_executor/models/minicpmo_4_5/
├── minicpmo_4_5_omni.py          # 主入口 (385行)
│   ├── MiniCPMO45OmniForConditionalGeneration
│   ├── 两阶段调度逻辑
│   └── 上下文传递机制
│
├── minicpmo_4_5_omni_llm.py      # Thinker 实现 (4408行)
│   ├── 多模态编码器
│   ├── Qwen2 LLM
│   └── 文本生成
│
├── minicpmo_4_5_omni_tts.py      # Talker 实现 (457行)
│   ├── MiniCPMTTS
│   ├── Token2Wav 声码器
│   └── 流式 chunk 处理
│
└── pipeline.py                    # 流水线配置 (78行)
    └── StagePipelineConfig

vllm_omni/model_executor/stage_input_processors/
└── minicpmo_4_5_omni.py          # Stage 桥接 (137行)
    └── llm2tts() 函数
```

### 7.1 关键代码位置

| 功能 | 文件 | 行号 |
|------|------|------|
| 两阶段调度 | `minicpmo_4_5_omni.py` | 86-117 |
| 上下文传递 | `minicpmo_4_5_omni.py` | 278-281 |
| llm2tts 桥接 | `stage_input_processors/minicpmo_4_5_omni.py` | 19-136 |
| t+z token 合并 | `minicpmo_4_5_omni_tts.py` | 192-196 |
| 流式 chunk 处理 | `minicpmo_4_5_omni_tts.py` | 262-309 |
| Token2Wav 初始化 | `minicpmo_4_5_omni_tts.py` | 156 |
| 音频后处理 | `minicpmo_4_5_omni_tts.py` | 303-304 |

---

## 8. 部署配置

> [!example] GPU 配置

| 配置 | GPU | Thinker | Talker | 适用场景 |
|------|-----|---------|--------|---------|
| `minicpmo_4_5.yaml` | 2 | GPU 0 (70%) | GPU 1 (75%) | 默认 |
| `minicpmo_4_5_3gpu.yaml` | 3 | GPU 0,1 (TP=2) | GPU 2 | 更大 batch |
| `minicpmo_4_5_8x4090.yaml` | 8 | GPU 0-3 (TP=4) | GPU 4 | 消费级卡 |

**对你任务的启示**：
> Duplex 系统需要额外的 GPU 资源：
> - s2t 模块：1 GPU
> - Thinker：1-2 GPU
> - Talker：1 GPU
> - 总计：3-4 GPU

---

## 9. 总结与任务映射

> [!summary] 关键结论

| 任务维度 | MiniCPM-o 4.5 现状 | 需要改进 | 优先级 |
|---------|-------------------|---------|--------|
| **Duplex 双工** | ❌ 串行两阶段 | +s2t模块、并行框架、状态机 | 高 |
| **Session 管理** | ⚠️ 单次请求级别 | +Session缓存、KVCache复用、生命周期管理 | 高 |
| **Audio Chunk** | ⚠️ TTS内部chunk | +请求级chunk、context-chunk、滑动窗口 | 中 |
| **双 Token 调度** | ⚠️ 串行处理 | +统一调度器、tz合并、流水线重叠 | 中 |
| **Async CPU** | ⚠️ 部分CPU操作 | +GPU加速、异步化、零拷贝 | 低 |

### 9.1 研究路径建议

> [!tip] 基于 MiniCPM-o 4.5 的研究路径

**阶段一：理解现有实现**
1. 阅读 `minicpmo_4_5_omni.py` 理解两阶段调度
2. 阅读 `stage_input_processors/minicpmo_4_5_omni.py` 理解上下文传递
3. 阅读 `minicpmo_4_5_omni_tts.py` 理解 chunk 处理和 t+z 合并

**阶段二：设计改进方案**
1. 设计 Duplex 系统架构（s2t + Thinker + Talker 并行）
2. 设计 Session-aware 调度器（context-chunk + request-chunk）
3. 设计 mt1z 统一调度器（t token + z token 合并）

**阶段三：实现与验证**
1. 实现 Duplex 系统原型
2. 实现 Session-aware Chunk 调度器
3. 实现 mt1z 统一调度器
4. 性能测试与优化

---

## 10. 参考资源

> [!link] 相关资源

### 代码
- `vllm_omni/model_executor/models/minicpmo_4_5/` - MiniCPM-o 4.5 实现
- `vllm_omni/core/sched/omni_ar_scheduler.py` - AR 调度器
- `vllm_omni/engine/orchestrator.py` - 编排层

### 论文
- [Duplex TTS 论文](https://arxiv.org/pdf/2602.02204) - 双工语音合成理论基础
- [Omni RFC #3745](https://github.com/vllm-project/vllm-omni/issues/3745) - Duplex 系统设计讨论

### 文档
- [MiniCPM-o 4.5 部署指南](../../recipes/OpenBMB/VoxCPM2.md)
- [Gradio Demo](../../examples/online_serving/minicpmo/)

---

*最后更新: 2026-06-02*
*标签: #vLLM-omni #MiniCPM #Duplex #Session管理 #AudioChunk #双Token调度*
