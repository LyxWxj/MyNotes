---
type: Note
related_to: "[[vLLM-omni]]"
status: Active
url: https://github.com/vllm-project/vllm-omni/blob/main/docs/design/feature/diffusion_step_execution.md
---

# Diffusion Step Execution

本文档描述vLLM-Omni的逐步扩散执行契约，供模型作者和贡献者实现`step_execution=True`支持。

## 当前支持范围

`step_execution`不是通用扩散开关，仅适用于实现分段有状态契约的管道。

| 管道 | 示例模型 | 支持 |
|------|----------|------|
| `QwenImagePipeline` | Qwen/Qwen-Image | ✅ |
| 其他扩散管道 | QwenImageEdit、GLM-Image、Wan、Flux等 | ❌ |

### 当前限制

- `StepScheduler`仅调度`batch_size=1`
- `step_mode`不支持`cache_backend`
- 请求模式额外功能（如KV传输）尚未接入

## 执行契约

逐步模式由四个管道方法驱动：

| 方法 | 用途 |
|------|------|
| `prepare_encode(state)` | 一次性请求准备 |
| `denoise_step(state)` | 计算当前步骤的噪声预测 |
| `step_scheduler(state, noise_pred)` | 修改潜变量并推进步骤状态 |
| `post_decode(state)` | 去噪完成后解码最终输出 |

### 状态管理

状态存储在`vllm_omni/diffusion/worker/utils.py`的`DiffusionRequestState`中：
- 存储请求范围的张量
- 使用`state.extra`存储模型特定字段

### 执行流程

1. `prepare_encode()`：新请求运行一次
2. `denoise_step()`：每个调度器tick运行
3. `step_scheduler()`：修改`state.latents`并推进`state.step_index`
4. `post_decode()`：`state.denoise_completed`为true后运行一次

## 推荐拆分方式

| 请求级阶段 | 逐步方法 | 内容 |
|-----------|----------|------|
| 输入验证、提示编码、潜变量初始化、时间步准备 | `prepare_encode()` | 每个请求只执行一次 |
| Transformer前向/噪声预测 | `denoise_step()` | 当前时间步的纯去噪计算 |
| `scheduler.step(...)`和`step_index += 1` | `step_scheduler()` | 仅潜变量/状态修改 |
| VAE解码/后处理 | `post_decode()` | 仅最终解码 |

## 新管道规则

1. 不要在`self.scheduler`上保留请求范围的调度器状态，在`prepare_encode()`期间复制到`state.scheduler`
2. 不要在`denoise_step()`中修改`state.step_index`，仅`step_scheduler()`应推进步骤
3. 不要在`denoise_step()`或`step_scheduler()`中解码部分输出
4. 将条件潜变量、掩码或编辑特定张量存储在`state`或`state.extra`中
5. 通过共享`forward()`使用的辅助路径保留CFG行为
6. 保持`post_decode()`等同于`forward()`的尾部

## 验证清单

在标记管道为`supports_step_execution = True`之前验证：

- [ ] 逐步输出与相同种子和采样参数的请求级输出匹配
- [ ] 跨并发请求的每请求调度器状态隔离
- [ ] 去噪期间中止不会泄漏缓存状态
- [ ] `RunnerOutput`报告的`step_index`与调度器进度匹配
- [ ] 如果请求级管道支持，CFG并行和非CFG路径都能工作

## 相关文件

- 契约：`vllm_omni/diffusion/models/interface.py`
- 状态：`vllm_omni/diffusion/worker/utils.py`
- Runner循环：`vllm_omni/diffusion/worker/diffusion_model_runner.py`
- 调度器传输：`vllm_omni/diffusion/sched/interface.py`
- 参考管道：`vllm_omni/diffusion/models/qwen_image/pipeline_qwen_image.py`
