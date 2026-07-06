---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm/blob/main/docs/serving/openai_compatible_server.md
---

# OpenAI-Compatible Server（OpenAI兼容服务器）

vLLM提供一个HTTP服务器，实现OpenAI的[Completions API](https://platform.openai.com/docs/api-reference/completions)、[Chat API](https://platform.openai.com/docs/api-reference/chat)等。

## 启动服务器

```bash
vllm serve NousResearch/Meta-Llama-3-8B-Instruct \
  --dtype auto \
  --api-key token-abc123
```

## 客户端调用

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",
)

completion = client.chat.completions.create(
    model="NousResearch/Meta-Llama-3-8B-Instruct",
    messages=[
        {"role": "user", "content": "Hello!"},
    ],
)

print(completion.choices[0].message)
```

> **提示**：vLLM支持OpenAI不支持的参数（如`top_k`），可通过`extra_body`参数传递：`extra_body={"top_k": 50}`

> **重要**：默认情况下，服务器会应用HuggingFace模型仓库中的`generation_config.json`（如果存在）。要禁用此行为，请在启动服务器时传递`--generation-config vllm`。

## 支持的API

### OpenAI兼容API

| API | 端点 | 适用模型 |
|-----|------|----------|
| **Completions API** | `/v1/completions` | 文本生成模型 |
| **Responses API** | `/v1/responses` | 文本生成模型 |
| **Chat Completions API** | `/v1/chat/completions` | 带聊天模板的文本生成模型 |
| **Embeddings API** | `/v1/embeddings` | 嵌入模型 |
| **Transcriptions API** | `/v1/audio/transcriptions` | ASR模型 |
| **Translation API** | `/v1/audio/translations` | ASR模型 |
| **Realtime API** | `/v1/realtime` | ASR模型 |

### 自定义API

| API | 端点 | 适用模型 |
|-----|------|----------|
| **Tokenizer API** | `/tokenize`, `/detokenize` | 带tokenizer的模型 |
| **Pooling API** | `/pooling` | 池化模型 |
| **Classification API** | `/classify` | 分类模型 |
| **Cohere Embed API** | `/v2/embed` | 嵌入模型 |
| **Score API** | `/score`, `/v1/score` | 评分模型 |
| **Generative Scoring API** | `/generative_scoring` | CausalLM模型 |
| **Rerank API** | `/rerank`, `/v1/rerank`, `/v2/rerank` | 重排模型 |

## Chat Template

为了让语言模型支持聊天协议，vLLM要求模型在其tokenizer配置中包含聊天模板。聊天模板是一个Jinja2模板，指定如何编码角色、消息和其他聊天特定token。

如果模型没有提供聊天模板，可以手动指定：

```bash
vllm serve <model> --chat-template ./path-to-chat-template.jinja
```

### 内容格式

vLLM支持两种聊天模板内容格式：
- `"string"`：字符串格式，如`"Hello world"`
- `"openai"`：OpenAI模式的字典列表，如`[{"type": "text", "text": "Hello world!"}]`

可通过`--chat-template-content-format` CLI参数覆盖使用的格式。

## 额外参数

vLLM支持一组不属于OpenAI API的参数，可通过`extra_body`传递：

```python
completion = client.chat.completions.create(
    model="NousResearch/Meta-Llama-3-8B-Instruct",
    messages=[
        {"role": "user", "content": "Classify this sentiment: vLLM is wonderful!"},
    ],
    extra_body={
        "structured_outputs": {"choice": ["positive", "negative"]},
    },
)
```

## 额外HTTP头

目前仅支持`X-Request-Id` HTTP请求头，可通过`--enable-request-id-headers`启用。

## 离线API文档

FastAPI `/docs`端点默认需要互联网连接。要在离线环境中启用离线访问，使用`--enable-offline-docs`标志：

```bash
vllm serve NousResearch/Meta-Llama-3-8B-Instruct --enable-offline-docs
```

## Realtime API

Realtime API提供基于WebSocket的流式音频转录，允许在录音时进行实时语音转文本。

### 音频格式

音频必须以16kHz采样率、单声道的base64编码PCM16格式发送。

### 协议概述

1. 客户端连接到`ws://host/v1/realtime`
2. 服务器发送`session.created`事件
3. 客户端可选发送`session.update`配置模型/参数
4. 客户端在准备好时发送`input_audio_buffer.commit`
5. 客户端发送带有base64 PCM16块的`input_audio_buffer.append`事件
6. 服务器发送`transcription.delta`事件（增量文本）
7. 服务器发送`transcription.done`（最终文本 + 使用统计）
8. 重复步骤5进行下一段语音

### 事件类型

**客户端 → 服务器**：

| 事件 | 描述 |
|------|------|
| `input_audio_buffer.append` | 发送base64编码的音频块 |
| `input_audio_buffer.commit` | 触发转录处理或结束 |
| `session.update` | 配置会话 |

**服务器 → 客户端**：

| 事件 | 描述 |
|------|------|
| `session.created` | 连接建立，包含会话ID和时间戳 |
| `transcription.delta` | 增量转录文本 |
| `transcription.done` | 最终转录和使用统计 |
| `error` | 错误通知 |

## Generative Scoring API

`/generative_scoring`端点使用CausalLM模型计算指定token ID作为下一个token的概率。

### 工作原理

1. **提示构建**：对于每个item，构建`prompt = query + item`
2. **前向传播**：在每个提示上运行模型获取下一个token的logits
3. **概率提取**：提取指定`label_token_ids`的logprobs
4. **Softmax归一化**：仅对label token应用softmax
5. **分数**：返回第一个label token的归一化概率

### 示例

```bash
curl -X POST http://localhost:8000/generative_scoring \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "query": "Is this city the capital of France?",
    "items": ["Paris", "London", "Berlin"],
    "label_token_ids": [9454, 2753]
  }'
```

## Ray Serve LLM

Ray Serve LLM支持可扩展的、生产级的vLLM引擎服务。

**关键能力**：
- 暴露OpenAI兼容的HTTP API和Pythonic API
- 从单GPU扩展到多节点集群，无需代码更改
- 通过Ray仪表板和指标提供可观测性和自动扩展策略

## 相关链接

- [Ray Serve LLM文档](https://docs.ray.io/en/latest/serve/llm/index.html)
