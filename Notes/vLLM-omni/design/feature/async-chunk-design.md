---
type: Note
related_to: "[[vLLM-omni]]"
status: Active
url: https://github.com/vllm-project/vllm-omni/blob/main/docs/design/feature/async_chunk_design.md
---

# Async Chunk Design

`async_chunk`特性支持在多阶段管道中异步、分块处理数据（如Qwen3-Omni的Thinker → Talker → Code2Wav阶段）。

## 核心概念

### 分块大小定义

- **预填充阶段**：`chunk_size = num_scheduled_tokens`
- **解码阶段**：`chunk_size = num_scheduled_tokens = 1`（逐token流式）

### Qwen3-Omni分块策略

| 阶段 | 分块策略 |
|------|----------|
| Thinker → Talker | 每个解码步骤（chunk_size=1） |
| Talker → Code2Wav | 累积到`codec_chunk_frames`（默认25）后发送 |
| Code2Wav | 流式解码 |

## 性能优势

- **降低延迟**：下一阶段可立即开始处理
- **流式支持**：支持音频生成的流式输出
- **IO-计算重叠**：块检索与其他请求计算异步进行
- **非阻塞调度器**：等待块的请求不会阻塞整个调度器

### 性能数据（H800 GPU）

启用`async_chunk`后：
- TTFP（首音频时间）：并发1时降低约92%（6.5s→0.52s）
- E2E延迟：并发1时降低约6%，并发10时降低约17%
- RTF（实时因子）：并发1时提升约8%（0.24→0.22）

## 架构组件

1. **OmniConnector**：仅用于阶段间数据传输
   - 传输专用API：`put()`和`get()`
   - 无请求特定状态

2. **Transfer Adapter Layer**：
   - `OmniTransferAdapterBase`：基础类，包含后台recv_loop和save_loop线程
   - `OmniChunkTransferAdapter`：块特定实现，管理完整的块生命周期

3. **Stage Input Processors**：自定义函数，处理阶段输出为块

4. **Schedulers**：
   - `OmniARScheduler`：自回归阶段
   - `OmniGenerationScheduler`：生成阶段

5. **Model Runners**：
   - `OmniGPUModelRunner`：AR阶段块处理
   - `GPUGenerationModelRunner`：生成阶段块处理

6. **Request Status**：新增`RequestStatus.WAITING_FOR_CHUNK`状态

## 配置

### 阶段配置

```yaml
async_chunk: true
stage_args:
  - stage_id: 0
    engine_args:
      custom_process_next_stage_input_func: vllm_omni.model_executor.stage_input_processors.qwen3_omni.thinker2talker_async_chunk
  - stage_id: 1
    engine_args:
      custom_process_next_stage_input_func: vllm_omni.model_executor.stage_input_processors.qwen3_omni.talker2code2wav_async_chunk
```

### Code2Wav批处理配置

```yaml
stage_args:
  - stage_id: 2
    runtime:
      devices: "1"
    engine_args:
      model_stage: code2wav
      max_num_seqs: 64
```

## 相关文件

- `vllm_omni/model_executor/stage_input_processors/qwen3_omni.py`
- `vllm_omni/distributed/omni_connectors/transfer_adapter/`
- `vllm_omni/core/sched/omni_ar_scheduler.py`
- `vllm_omni/worker/gpu_model_runner.py`
- `vllm_omni/model_executor/models/qwen3_omni/`
