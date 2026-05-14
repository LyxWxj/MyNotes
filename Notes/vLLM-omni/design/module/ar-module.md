---
type: Note
related_to: "[[vLLM-omni]]"
status: Active
url: https://github.com/vllm-project/vllm-omni/blob/main/docs/design/module/ar_module.md
---

# AutoRegressive (AR) Module

## 概述

vLLM-Omni中的AR模块处理自回归生成阶段，主要用于多阶段模型（如Qwen2.5-Omni、Qwen3-Omni、BAGEL等）中的文本、思维链（COT）和音频潜变量token生成阶段。

AR模块扩展vLLM核心组件以支持：
- **多模态输入/输出**：处理图像、视频和音频以及文本
- **直接嵌入转移**：通过序列化有效负载在管道阶段之间传递预计算的提示嵌入
- **附加信息流**：在管道中携带每请求元数据（张量、列表）
- **隐藏状态暴露**：暴露每请求隐藏表示供下游阶段使用
- **基本生成器支持**：支持一些基本异构架构（如卷积、LSTM等）

## 与vLLM的关系

AR模块通过继承建立在vLLM主框架之上，扩展核心类同时保持与vLLM调度、批处理、KV缓存管理和执行机制的兼容性。

### 继承层次

#### Scheduler
```
VLLMScheduler
├── OmniARScheduler
└── OmniGenerationScheduler
```

#### Worker
```
GPUWorker
├── GPUARWorker
└── GPUGenerationWorker
```

#### ModelRunner
```
GPUModelRunner
└── OmniGPUModelRunner
    ├── GPUARModelRunner
    └── GPUGenerationModelRunner
```

#### InputProcessor/OutputProcessor
```
InputProcessor (stage-0使用上游vllm.v1.engine.input_processor.InputProcessor)

VLLMOutputProcessor
└── MultimodalOutputProcessor
```

### 关键扩展

- **Scheduler**：`OmniARScheduler`扩展`vllm.v1.core.sched.scheduler.Scheduler`以使用omni特定有效负载丰富调度请求
- **Worker**：`GPUARWorker`扩展`vllm.v1.worker.gpu_worker.Worker`以初始化AR特定模型运行器
- **ModelRunner**：`GPUARModelRunner`扩展`OmniGPUModelRunner`以暴露隐藏状态并处理多模态输出
- **InputProcessor**：Stage-0使用上游`InputProcessor`；`AsyncOmniEngine`在构建`OmniEngineCoreRequest`时恢复omni特定有效负载
- **OutputProcessor**：`MultimodalOutputProcessor`扩展`vllm.v1.engine.output_processor.OutputProcessor`以路由和累积多模态输出

## 调度器设计

### 请求流程

```
InputProcessor (stage-0 in AsyncOmniEngine)
    → EngineCoreRequest (升级为OmniEngineCoreRequest)
    → OmniARScheduler
    → OmniNewRequestData
    → GPUARWorker
    → SchedulerOutput
    → GPUARModelRunner
    → Model Forward Pass
    → OmniModelRunnerOutput
    → OmniARScheduler
    → MultimodalOutputProcessor
    → RequestOutput
    → Client/Downstream Stage
```

### OmniARScheduler

扩展基础vLLM调度器，专注于使用omni特定有效负载丰富调度请求。

#### 修改的API：`schedule()`

```python
def schedule(self) -> SchedulerOutput:
    scheduler_output = super().schedule()
    # 将基础NewRequestData条目重包装为OmniNewRequestData
    new_list = []
    for nr in scheduler_output.scheduled_new_reqs:
        request = self.requests.get(nr.req_id)
        omni_nr = OmniNewRequestData(
            req_id=nr.req_id,
            prompt_token_ids=nr.prompt_token_ids,
            prompt_embeds=getattr(request, "prompt_embeds", None),
            additional_information=getattr(request, "additional_information", None),
        )
        new_list.append(omni_nr)
    scheduler_output.scheduled_new_reqs = new_list
    return scheduler_output
```

### OmniGenerationScheduler

为在单个步骤中处理所有输入token的基本异构架构实现快速路径调度策略。

#### 修改的API：`schedule()`

```python
def schedule(self) -> SchedulerOutput:
    # 快速路径：一次分配所有输入token
    while self.waiting and token_budget > 0:
        request = self.waiting.peek_request()
        required_tokens = max(getattr(request, "num_prompt_tokens", 0), 1)
        if required_tokens > token_budget:
            break  # 回退到默认调度
        # 分配并调度...
```

#### 修改的API：`update_from_output()`

在一步后立即将请求标记为完成，因为生成模型在单个前向传播中完成：

```python
def update_from_output(self, ...) -> dict[int, EngineCoreOutputs]:
    # ...
    request.status = RequestStatus.FINISHED_STOPPED
    kv_transfer_params = self._free_request(request)
    # ...
```

## Worker和ModelRunner设计

### GPUARWorker

初始化AR特定模型运行器，同时保持标准设备初始化：

```python
class GPUARWorker(GPUWorker):
    def init_device(self):
        # ... 标准设备初始化 ...
        self.model_runner = GPUARModelRunner(self.vllm_config, self.device)
```

### GPUARModelRunner

遵循vLLM的两阶段执行/采样流程，同时暴露隐藏状态和多模态输出。

#### 两阶段执行

**阶段1：`execute_model()`** - 运行前向传播并存储状态：
- 从隐藏状态计算logits
- 存储包含隐藏状态、logits和多模态输出的`ExecuteModelState`
- 返回`None`以延迟采样

**阶段2：`sample_tokens()`** - 采样token并构建输出：
- 从`execute_model()`检索存储状态
- 使用logits采样token
- 提取每请求隐藏状态和多模态输出
- 构建包含`pooler_output`（包含隐藏状态）的`OmniModelRunnerOutput`

```python
def sample_tokens(self, grammar_output) -> OmniModelRunnerOutput:
    # 检索存储状态
    hidden_states, multimodal_outputs = self.execute_model_state

    # 采样token
    sampler_output = self._sample(logits, spec_decode_metadata)

    # 提取每请求隐藏状态
    pooler_output = []
    for rid in req_ids:
        hidden_slice = hidden_states_cpu[start:end]
        payload = {"hidden": hidden_slice}
        # 添加多模态输出（如果存在）
        pooler_output.append(payload)

    return OmniModelRunnerOutput(
        pooler_output=pooler_output,
    )
```

### GPUGenerationModelRunner

为基本异构架构实现简化的单阶段执行：
- 无logits计算或token采样
- 在模型实现中直接从前向传播生成
- 在前向传播后立即通过`pooler_output`返回输出

### OmniGPUModelRunner

为AR和Generation运行器提供共享功能。

#### 提示嵌入覆盖

在预填充期间，将自定义`prompt_embeds`从请求状态覆盖到`inputs_embeds`：

```python
def _collect_additional_information_for_prefill(self, num_scheduled_tokens_np):
    for req_index, req_id in enumerate(self.input_batch.req_ids):
        req_state = self.requests[req_id]
        pe_cpu = getattr(req_state, "prompt_embeds_cpu", None)
        if pe_cpu is not None:
            src = pe_cpu[num_computed_tokens:num_computed_tokens + overlay_len]
            self.inputs_embeds[start_offset:start_offset + overlay_len].copy_(src)
```

#### 附加信息处理

解码和管理`additional_information`有效负载：
- 解码序列化有效负载 → 请求状态中的CPU张量
- 通过`runtime_additional_information` kwarg将运行时信息传递给模型
- 通过`postprocess()`钩子处理模型提供的更新
- 将更新合并回请求状态

#### M-RoPE位置初始化

对于使用M-RoPE的多模态模型（如Qwen2-VL），从多模态特征元数据（图像网格、视频网格、音频特征）计算位置编码。

## 输入/输出处理

### 处理管道

```
Client → AsyncOmniEngine
    → InputProcessor.process_inputs()
    → EngineCoreRequest
    → _upgrade_to_omni_request()
    → serialize_additional_information()
    → OmniNewRequestData (with payloads)
    → ModelRunner
        → Decode payloads → CPU tensors
        → Overlay prompt_embeds on inputs_embeds
        → Forward pass with runtime_additional_information
        → Extract hidden states + multimodal outputs
    → OmniModelRunnerOutput (pooler_output)
    → MultimodalOutputProcessor
        → Route by output_type
        → Accumulate tensors in OmniRequestState
        → Consolidate tensor lists
    → RequestOutput (with multimodal_output)
    → Client
```

### Stage-0输入处理

Stage-0现在直接使用上游`InputProcessor`，`AsyncOmniEngine`在恢复omni特定有效负载的同时将请求升级为`OmniEngineCoreRequest`。

```python
request = self.input_processor.process_inputs(
    request_id=request_id,
    prompt=prompt,
    params=params,
    supported_tasks=self.supported_tasks,
)
request = _upgrade_to_omni_request(request, prompt)
```

### MultimodalOutputProcessor

按模态类型路由输出并累积多模态张量。

#### 输出路由

按`output_type`属性路由`EngineCoreOutput`：
- `"text"`：标准文本生成路径
- `"image"`、`"audio"`、`"latents"`：从`pooling_output`或`multimodal_outputs`提取
- 回退：基于`pooling_output`存在的启发式

#### 张量累积

`OmniRequestState`跨多个步骤累积多模态张量：

```python
def add_multimodal_tensor(self, payload, mm_type):
    # 将有效负载规范化为字典
    incoming = {mm_type or "hidden": payload}

    # 累积：将张量转换为列表以进行延迟连接
    if isinstance(v, torch.Tensor) and isinstance(existing, torch.Tensor):
        self.mm_accumulated[k] = [existing, v]  # 列表累积
```

在最终输出之前，通过连接合并张量列表：

```python
def _consolidate_multimodal_tensors(self):
    for k, v in self.mm_accumulated.items():
        if isinstance(v, list) and isinstance(v[0], torch.Tensor):
            self.mm_accumulated[k] = torch.cat(v, dim=0)  # 连接
```

合并的张量附加到`RequestOutput.multimodal_output`供下游阶段或客户端使用。

## 总结

### 关键设计模式

1. **继承优于组合**：扩展vLLM类以保持与现有调度、批处理和执行机制的兼容性
2. **有效负载序列化**：使用序列化的`additional_information`有效负载与提示嵌入交接实现高效的阶段间转移
3. **两阶段执行**：为AR模型维护vLLM的执行/采样分离，同时为生成模型支持单阶段执行
4. **多模态路由**：按`output_type`路由输出并增量累积张量以支持流式

### 与vLLM的差异

- **有效负载支持**：序列化的附加信息和提示嵌入实现管道阶段之间的直接转移
- **多模态处理**：扩展的输入/输出处理器支持图像、音频和其他模态以及文本
- **隐藏状态暴露**：AR模型运行器通过`pooler_output`暴露每请求隐藏状态供下游使用
- **生成调度器**：为在一步中完成的基本异构架构提供快速路径调度

AR模块与vLLM现有基础设施无缝集成，同时为多阶段、多模态生成管道添加必要的扩展。
