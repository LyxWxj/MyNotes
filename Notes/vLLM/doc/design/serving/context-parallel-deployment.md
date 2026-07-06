---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm/blob/main/docs/serving/context_parallel_deployment.md
---

# Context Parallel Deployment（上下文并行部署）

上下文并行主要解决长上下文请求的服务问题。由于预填充和解码呈现完全不同的特性并具有不同的SLO（服务级别目标），需要分别实现上下文并行。

**主要考虑**：
- **长上下文预填充**：通过在查询token之间摊销预填充的计算时间来控制TTFT（首token时间）
- **长上下文解码**：需要更多KV缓存空间以增加批处理大小（从而提高吞吐量）

## 预填充上下文并行

在预填充期间，对于具有`T`个新token的长请求，需要为这些新token计算query/key/value张量。假设有`N`个GPU，可以将请求分成`N`个块，每个GPU计算一个块的query/key/value张量。

### 两种策略

| 策略 | 适用场景 | 描述 |
|------|----------|------|
| **部分查询，完整key/value** | 请求token长度适中 | 从所有GPU收集key/value张量，让每个GPU计算其块对应的查询token的注意力输出 |
| **部分查询，部分key/value** | 请求token长度过长 | 每个GPU只计算一个块的query/key/value张量，使用[ring-attention](http://arxiv.org/abs/2310.01889)等技术逐块发送/接收key/value张量 |

两种方法都在积极开发中。

## 解码上下文并行

由于解码的自回归特性，每个解码步骤需要计算少量查询token相对于分页KV缓存中存储的大量key/value token。解码上下文并行的核心是如何在GPU之间分片KV缓存。

对于具有`H`个kv-head的模型，上下文中有`T`个token的请求需要在KV缓存中存储`H * T`个key/value张量。

### 三种场景

1. **单GPU可容纳**：如果一个GPU可以容纳所有内容且性能足够好，则不需要并行化
2. **沿H维度分片**：如果一个GPU无法容纳，或想在KV缓存中容纳更多请求，可以首先沿`H`维度分片KV缓存（即张量并行分片），只需在命令行添加`-tp <num_gpus>`
3. **沿T维度分片**：由于`H`有限（由模型架构决定），当继续增加张量并行大小时，每个GPU的KV缓存将被复制`tp_size / H`次。需要添加解码上下文并行以进一步沿`T`维度分片KV缓存，只需在命令行添加`-dcp <size>`

> **注意**：`size`不会增加需要启动的GPU数量，只是减少KV缓存重复。dcp大小应在`[1, tp_size/H]`范围内。

### KV缓存增长

kv缓存在解码期间可以增长，需要仔细实现分片策略。使用交错策略沿`T`维度分片KV缓存，以便未来token的KV缓存可以自然沿`T`维度分片。

### 案例研究

| 模型 | 配置 | 问题 | 解决方案 |
|------|------|------|----------|
| **DeepSeek-R1** | `-tp 8`，MLA启用时1个kv-head | 8x KV缓存重复 | 添加`-dcp 8`减少重复 |
| **Kimi-K2** | `-tp 16` | 16x KV缓存重复 | 添加`-dcp 16`完全消除重复（更多通信开销）或`-dcp 8`减少到2x |
| **Qwen3-235B-A22B** | `-tp 8`，4个kv-heads | 2x KV缓存重复 | 添加`-dcp 2`消除重复 |

**简而言之**：对于解码上下文并行，尝试增加`-tp`大小直到获得满意的性能，然后添加`-dcp`以减少KV缓存重复。

## 支持情况

解码上下文并行在vLLM中支持，适用于MLA和GQA模型。一些注意力后端还支持解码上下文并行与MTP（多token预测）的组合，以进一步加速解码阶段。

## 技术讨论

主要讨论发生在[vLLM Slack](https://slack.vllm.ai/)的`#sig-context-parallel`频道中。
