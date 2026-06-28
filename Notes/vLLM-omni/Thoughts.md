# vllm-omni 分离式推理系统分析

## 一、系统概述

vllm-omni 的分离式推理（Disaggregated DiT Inference）将 diffusion pipeline 拆分为独立的 stage，每个 stage 可以运行在不同的 NPU 上，由 Orchestrator 统一调度。

RFC: https://github.com/vllm-project/vllm-omni/issues/4590

Config RFC: https://github.com/vllm-project/vllm-omni/issues/4021

---

## 二、Stage 抽象的四层架构

```
PipelineConfig (模型拓扑，编译时确定，不可变)
  └─ StagePipelineConfig (单 stage 的拓扑定义)
       └─ StageConfig (合并 deploy YAML 后的运行时配置)
            └─ StagePool (运行时，管理 replicas + 路由)
```

### 2.1 PipelineConfig

定义在 `vllm_omni/config/stage_config.py:233`，是 `frozen=True` 的 dataclass：

```python
@dataclass(frozen=True)
class PipelineConfig:
    model_type: str                    # "qwen_image"
    model_arch: str = ""
    stages: tuple[StagePipelineConfig, ...] = ()
    hf_architectures: tuple[str, ...] = ()
    diffusers_class_name: str | None = None
```

各模型在 `pipeline_registry.py` 中注册一行，实际定义在 `model_executor/models/<model>/pipeline.py`。

### 2.2 StagePipelineConfig

定义在 `vllm_omni/config/stage_config.py:200`：

```python
@dataclass(frozen=True)
class StagePipelineConfig:
    stage_id: int
    model_stage: str                           # "encode" / "denoise" / "decode"
    execution_type: StageExecutionType         # DIFFUSION / DIFFUSION_SUBMODULE / LLM_AR / LLM_GENERATION
    input_sources: tuple[int, ...] = ()        # DAG 的入边，如 (0,) 或 (0, 1)
    final_output: bool = False                 # 是否是最终输出 stage
    final_output_type: str | None = None       # "image" / "audio" / "text"
    custom_process_input_func: str | None      # stage 间数据转换函数名
    requires_multimodal_data: bool = False
    owns_tokenizer: bool = False
```

`execution_type` 决定了 worker 类型：

- `DIFFUSION` → 完整 DiffusionEngine（含 scheduler + transformer，18 步迭代）
- `DIFFUSION_SUBMODULE` → 轻量 SubModuleRunner（仅 encode 或 decode，单次 forward）
- `LLM_AR` / `LLM_GENERATION` → LLM 自回归/生成

### 2.3 StageConfig

`merge_pipeline_deploy()` (`stage_config.py:903`) 将 `PipelineConfig`（拓扑）与 `DeployConfig`（deploy YAML）合并：

```
PipelineConfig (拓扑)  +  DeployConfig (deploy/qwen_image.yaml)  →  StageConfig[]
```

`DeployConfig` 来自 `deploy/<model>.yaml`，包含用户可调的部署参数（devices, num_replicas, max_num_seqs 等）。

### 2.4 StagePool

`vllm_omni/engine/stage_pool.py`，运行时核心抽象，每个 stage 有一个实例：

```python
class StagePool:
    stage_id: int
    clients: list[StagePoolClient | None]  # 每个 replica 一个 client
    _request_bindings: dict[str, int]      # request_id → replica_id 的亲和性
```

路由机制：

- 非分布式：`select_replica_id()` → round-robin over live replicas
- 分布式：`pick()` → hub.get_replicas_for_stage() → LoadBalancer.select() → affinity 记录

---

## 三、两条并行的执行路径

### 3.1 完整路径（denoise stage）

```
StageDiffusionClient → StageDiffusionProc → DiffusionEngine → DiffusionWorker → DiffusionModelRunner
```

支持：TeaCache、torch.compile、LoRA、CPU offload、step-wise 执行、KV transfer、sleep/wake、batch 调度。

- `StageDiffusionClient` (`engine/stage_client.py`)：head-side ZMQ client
- `StageDiffusionProc` (`diffusion/stage_diffusion_proc.py`, ~796 行)：子进程入口，ZMQ 事件循环，创建 DiffusionEngine
- `DiffusionWorker` (`diffusion/worker/diffusion_worker.py`, ~1087 行)：GPU worker，分布式环境、LoRA、sleep/wake
- `DiffusionModelRunner` (`diffusion/worker/diffusion_model_runner.py`, ~528 行)：完整 model runner，支持 step-wise denoise

### 3.2 轻量路径（encode/decode stage）

```
StageSubModuleClient → StageSubModuleProc → _SubModuleEngine → SubModuleWorker → DiffusionSubmoduleRunner
```

只有基础的 forward 能力，无 cache、无 compile、无 step-wise。

- `StageSubModuleClient` (`diffusion/stage_submodule_client.py`, ~55 行)：继承 StageDiffusionClient，只支持单 prompt
- `StageSubModuleProc` (`diffusion/stage_submodule_proc.py`, ~168 行)：继承 StageDiffusionProc，用 SubModuleWorker 替代 DiffusionEngine
- `SubModuleWorker` (`diffusion/worker/submodule_worker.py`, ~198 行)：单 GPU (world_size=1)，无 LoRA/sleep/compile
- `DiffusionSubmoduleRunner` (`diffusion/worker/diffusion_submodule_runner.py`, ~116 行)：单次 forward，调用 `pipeline.execute_encode()` 或 `pipeline.execute_decode()`

### 3.3 关键差异

| 能力 | DiffusionWorker/Runner | SubModuleWorker/Runner |
|---|---|---|
| 分布式环境初始化 | ✅ | ✅ |
| 模型加载 | ✅ + offload + compile + cache | ✅ 基础加载 |
| LoRA 管理 | ✅ | ❌ |
| Sleep/Wake | ✅ | ❌ |
| Step-wise 执行 | ✅ | ❌ |
| TeaCache | ✅ | ❌ |
| torch.compile | ✅ | ❌ |
| 进程模型 | MultiprocExecutor (多进程) | ThreadPoolExecutor (同进程) |

---

## 四、QwenImage 3-Stage Pipeline 详解

### 4.1 YAML 配置

```yaml
# Stage 0 — encode (submodule)
- stage_id: 0
  stage_type: diffusion
  worker_type: submodule
  runtime:
    devices: "0"
  engine_args:
    model_class_name: QwenImagePipeline
    model_stage: encode

# Stage 1 — denoise (完整路径)
- stage_id: 1
  stage_type: diffusion
  # 无 worker_type → 走完整路径
  engine_args:
    model_stage: denoise
  engine_input_source: [0]
  custom_process_input_func: encode_to_denoise

# Stage 2 — decode (submodule)
- stage_id: 2
  stage_type: diffusion
  worker_type: submodule
  engine_args:
    model_stage: decode
  engine_input_source: [1]
  custom_process_input_func: denoise_to_decode
  final_output: true
  final_output_type: image
```

### 4.2 模型创建流程（Startup）

```
YAML config
  → StageConfigFactory.create_from_model()
    → initialize_diffusion_stage() [stage_init_utils.py:1088]
      → build_diffusion_config() → OmniDiffusionConfig
      → worker_type == "submodule"?
          ├─ Yes → StageSubModuleClient.__init__()
          │         → spawn_submodule_proc() → multiprocessing.Process
          │           → StageSubModuleProc.initialize()
          │             → SubModuleWorker.__init__()
          │               → init_device() (单GPU, world_size=1)
          │               → DiffusionSubmoduleRunner.__init__()
          │             → SubModuleWorker.load_model()
          │               → DiffusersPipelineLoader.load_model()
          │                 → initialize_model(od_config) (registry lookup)
          │                 → self.load_weights(model) (加载 safetensors)
          │             → _SubModuleEngine(worker, od_config, executor)
          │           → handshake_socket.send({"status": "READY"})
          │           → asyncio.run(run_loop())
          → complete_submodule_handshake() (主进程收到 READY)
          → self._initialize_client() (ZMQ 连接建立)

          └─ No  → StageDiffusionClient (完整路径，类似但更复杂)
```

### 4.3 请求处理流程（Runtime）

```
用户请求 → Orchestrator._run_request()
  │
  ├─ StagePool[0].submit_initial(prompt)
  │   → StageSubModuleClient.add_request_async()
  │     → ZMQ PUSH → 子进程 run_loop() ZMQ PULL
  │       → _SubModuleEngine.step(request)
  │         → ThreadPoolExecutor → SubModuleWorker.execute_submodule()
  │           → DiffusionSubmoduleRunner.execute_model()
  │             → pipeline.execute_encode([req])
  │               → text_encoder + tokenizer
  │               → 返回 {context, context_mask, latents, timesteps, ...}
  │             → DiffusionOutput(output=None, multimodal_output=payload)
  │         → _to_request_output() → OmniRequestOutput
  │       → ZMQ PUSH → 主进程
  │
  ├─ Orchestrator._orchestration_loop() poll 到 stage 0 输出
  │   → _handle_processed_outputs(stage_id=0)
  │     → _forward_to_next_stage(req_id, src_stage_id=0, output)
  │
  ├─ encode_to_denoise(output, prompt)
  │   → 从 output.multimodal_output 提取 context/latents/timesteps
  │   → 包装为 OmniTokensPrompt.additional_information
  │
  ├─ StagePool[1].submit_initial(diffusion_prompt)
  │   → StageDiffusionClient → DiffusionEngine
  │     → pipeline.forward(req)  ← 18步去噪循环
  │       → 每步调用 DiT Transformer
  │     → pipeline.post_intermediate_output()
  │       → 导出 latents
  │
  ├─ Orchestrator poll 到 stage 1 输出
  │   → denoise_to_decode(output, prompt)
  │     → 从 output.multimodal_output 提取 latents
  │
  └─ StagePool[2].submit_initial(decode_prompt)
      → StageSubModuleClient → pipeline.execute_decode([req])
        → VAE decoder → PIL.Image
        → DiffusionOutput(output=image, multimodal_output=payload)
      → final_output=True → 返回给用户
```

### 4.4 Stage 间数据传递

Stage 之间不直接交互，全部通过 Orchestrator 中转：

```
Stage N 输出 → DiffusionOutput.multimodal_output (dict)
  → _to_request_output() → OmniRequestOutput.multimodal_output
    → Orchestrator._forward_to_next_stage()
      → custom_process_input_func(output, prompt)
        → 转换为 OmniTokensPrompt.additional_information (dict)
          → StagePool[N+1].submit_initial(prompt)
            → Stage N+1 从 additional_information 读取数据
```

`additional_information` 就是 stage 间的 " 协议 "——上游写什么 key，下游读什么 key，由 `custom_process_input_func` 负责转换。

---

## 五、Orchestrator 调度机制

### 5.1 调度循环

`_orchestration_loop()` (`orchestrator.py:569`) 是主循环：

```python
while not self._shutdown_event.is_set():
    for stage_id in range(self.num_stages):
        pool = self.stage_pools[stage_id]
        for replica_id in pool.live_replica_ids():
            if pool.stage_type == "diffusion":
                output = pool.poll_diffusion_output(replica_id)  # 非阻塞
                await self._handle_processed_outputs(stage_id, replica_id, [output])
            else:
                raw_outputs = await pool.poll_llm_raw_output(replica_id, timeout_s=0.001)
                await self._handle_processed_outputs(stage_id, replica_id, raw_output)
```

- Diffusion stage：一次性产出（整张图片），非阻塞 poll
- LLM stage：流式产出（逐 token），带 1ms 超时 poll

### 5.2 请求路由

`_forward_to_next_stage()` (`orchestrator.py:918`) 是 stage 间路由的核心：

```python
async def _forward_to_next_stage(self, req_id, src_stage_id, output, req_state):
    next_logical = src_stage_id + 1  # ← 硬编码！只能传给下一个 stage
    next_pool = self.stage_pools[next_logical]
    
    if next_pool.stage_type == "diffusion":
        # 调用 custom_process_input_func 转换数据
        diffusion_prompt = next_client.custom_process_input_func(source_outputs, prompt, ...)
        await next_pool.submit_initial(req_id, req_state, diffusion_prompt)
    else:
        # LLM stage 的处理
        next_inputs = next_client.process_engine_inputs(source_outputs, prompt, ...)
        for next_input in next_inputs:
            await next_pool.submit_initial(req_id, req_state, request)
```

---

## 六、存在的问题

### 6.1 OmniDiffusionConfig 滥用（期待 #4021 改进）

`OmniDiffusionConfig` (`diffusion/data.py:404`) 有 60+ 个字段，是一个 "god object"。

SubModuleWorker + DiffusionSubmoduleRunner 实际只需要 ~12 个字段：

```python
# 实际需要的字段
stage_id, model, model_stage, dtype,
trust_remote_code, revision, diffusion_load_format,
enable_cpu_offload, enable_layerwise_offload,
distributed_executor_backend, master_port
```

但传入的是完整的 60+ 字段的 `OmniDiffusionConfig`，包含大量无关字段（LoRA、cache、compile、sleep、VAE slicing 等）。

建议：为 submodule 路径创建轻量 `SubModuleConfig`，只保留必要字段。

### 6.2 DiffusionSubmoduleRunner 硬编码 stage 判断

```python
# diffusion_submodule_runner.py:112-115
if stage == "decode":
    return DiffusionOutput(output=payload.get("image"), multimodal_output=payload)
return DiffusionOutput(output=None, multimodal_output=payload)
```

用 `stage == "decode"` 硬编码来决定是否填充 `output`，而不是检查 `StageConfig.final_output`。

原因：Runner 层看不到 `final_output`（它在 StagePipelineConfig 中），只能看到 `od_config.model_stage`。

建议：Runner 总是填充 `output`，由 Orchestrator 层根据 `final_output` 决定是否返回给用户。

### 6.3 Orchestrator 硬编码线性链路由

```python
# orchestrator.py:929
next_logical = src_stage_id + 1  # ← 永远是 +1，不支持 DAG
```

所有 `input_sources` 检查都是线性的：

```python
# orchestrator.py:815
def _next_stage_already_submitted(self, stage_id, req_state):
    return (stage_id + 1) in req_state.stage_submit_ts
```

当前所有 23 个模型都使用线性链（`input_sources = ()` 或 `(N,)`），没有任何模型使用多父节点（`(N, M)`）。

### 6.4 final_stage_id 是单个 int

```python
# orchestrator.py:125
final_stage_id: int = -1
```

对于有多个终节点的 DAG（如 `ming_flash_omni` 的 thinker→text + talker→audio），需要改为 `terminal_stage_ids: set[int]`。

### 6.5 custom_process_input_func 只支持单父输出

```python
def encode_to_denoise(source_outputs, prompt, ...):
    mm = source_outputs[0].multimodal_output  # 只有一个父节点
```

不支持 fan-in（多个父节点的输出合并）。

### 6.6 DiffusionOutput.output 是单一字段

```python
@dataclass
class DiffusionOutput:
    output: torch.Tensor | dict | None = None  # 单一输出
    final_output_type: str | None = None       # 单一类型
```

对于同时输出音频 + 视频 + 图像的模型，需要用 dict 包装或多个 final stage 来绕过。

### 6.7 req_id 和 req_state.request_id 重复

```python
async def _forward_to_next_stage(self, req_id, src_stage_id, output, req_state):
    # req_id 和 req_state.request_id 永远相同
```

创建时 `req_state.request_id = request_id`，存储时 `self.request_states[request_id] = req_state`，两者永远一致。是 convenience pattern，不算 bug。

---

## 七、DAG Parallel 改进方案

### 7.1 目标

支持真正的 DAG 拓扑，而非仅线性链。适用于：

- 多模态模型（多个 encoder/decoder）
- Fan-in（多个 stage 输出汇聚到一个 stage）
- Fan-out（一个 stage 输出分发到多个 stage）

### 7.2 需要改动的组件

| 组件 | 当前问题 | 改动 |
|---|---|---|
| `Orchestrator._forward_to_next_stage` | `next_logical = src_stage_id + 1` | 从 `input_sources` 构建 successor 映射 |
| `Orchestrator._next_stage_already_submitted` | 只检查 `stage_id + 1` | 检查所有 successor |
| `OrchestratorRequestState` | `final_stage_id: int` | 改为 `terminal_stage_ids: set[int]` |
| `OrchestratorRequestState` | 无缓存 | 增加 `pending_outputs: dict[int, Any]` |
| `custom_process_input_func` | 单父输出 | 支持多父输出列表 |
| `PipelineConfig.validate()` | 已支持 DAG | 无需改动 |

### 7.3 实现思路

```python
# 1. 构建 successor 映射
def _build_successors(pipeline_config):
    successors = {}
    for stage in pipeline_config.stages:
        successors.setdefault(stage.stage_id, [])
        for src in stage.input_sources:
            successors.setdefault(src, []).append(stage.stage_id)
    return successors

# 2. 改 _forward_to_next_stage
async def _forward_to_next_stage(self, req_id, src_stage_id, output, req_state):
    next_stages = self._successors.get(src_stage_id, [])
    req_state.pending_outputs[src_stage_id] = output

    for next_logical in next_stages:
        input_sources = self.stage_pools[next_logical].input_sources
        ready = all(s in req_state.pending_outputs for s in input_sources)
        if not ready:
            continue

        all_outputs = [req_state.pending_outputs[s] for s in input_sources]
        prompt = next_client.custom_process_input_func(all_outputs, prompt, ...)
        await next_pool.submit_initial(req_id, req_state, prompt)
```

---

## 八、性能基准（NPU 8x 910B1, QwenImage）

| 配置 | 吞吐 | 相对提升 |
|---|---|---|
| 单实例，无 TeaCache | 0.05 req/s | baseline |
| 单实例 + TeaCache | 0.10 req/s | 2x |
| DP7 + TeaCache (3-stage) | 0.72 req/s | 14.4x |
| 8 实例 + TeaCache (nginx) | 0.60 req/s | 12x |

TeaCache 阈值：`rel_l1_thresh=0.2`（denoise stage），配置：`tea_cache_threshold: 0.3`。

---

## 九、相关文件索引

| 文件 | 用途 |
|---|---|
| `vllm_omni/config/stage_config.py` | Stage 配置系统（PipelineConfig, StageConfig, StageDeployConfig） |
| `vllm_omni/config/pipeline_registry.py` | Pipeline 注册表（model_type → module 映射） |
| `vllm_omni/engine/orchestrator.py` | Orchestrator（调度循环、stage 间路由） |
| `vllm_omni/engine/stage_pool.py` | StagePool（replica 管理、请求路由） |
| `vllm_omni/engine/stage_client.py` | StageClient 协议定义 |
| `vllm_omni/engine/stage_init_utils.py` | Stage 初始化（选择 submodule 或完整路径） |
| `vllm_omni/diffusion/stage_submodule_client.py` | SubModule head-side client |
| `vllm_omni/diffusion/stage_submodule_proc.py` | SubModule 子进程入口 |
| `vllm_omni/diffusion/stage_diffusion_proc.py` | 完整 Diffusion 子进程入口 |
| `vllm_omni/diffusion/worker/submodule_worker.py` | SubModule GPU worker |
| `vllm_omni/diffusion/worker/diffusion_worker.py` | 完整 Diffusion GPU worker |
| `vllm_omni/diffusion/worker/diffusion_submodule_runner.py` | SubModule model runner |
| `vllm_omni/diffusion/worker/diffusion_model_runner.py` | 完整 Diffusion model runner |
| `vllm_omni/diffusion/data.py` | OmniDiffusionConfig, DiffusionOutput |
| `vllm_omni/model_executor/stage_input_processors/qwen_image.py` | Stage 间数据转换函数 |
| `vllm_omni/model_executor/stage_configs/qwen_image_3stage.yaml` | 3-stage 部署配置 |
| `vllm_omni/model_executor/models/qwen_image/pipeline_qwen_image.py` | QwenImage Pipeline 实现 |

---

## 十、两种启动方式的完整流程

### 10.1 情况 1：YAML 配置文件启动（3-stage）

启动命令：
```bash
python -m vllm_omni.entrypoints.cli.main serve --omni \
    --model ~/models/Qwen-Image \
    --stage-configs-path vllm_omni/model_executor/stage_configs/qwen_image_3stage.yaml
```

AsyncOmniEngine 初始化链路：
```
AsyncOmniEngine.__init__()
  ├─ _resolve_stage_configs()
  │    └─ load_stage_configs_from_yaml("qwen_image_3stage.yaml")
  │         → self.stage_configs = [stage0, stage1, stage2]
  │
  └─ _bootstrap_orchestrator()  ← 后台守护线程
       ├─ _initialize_stages()
       │    ├─ compute_replica_layout() → 设备分配
       │    ├─ _initialize_stage_replicas()
       │    │    ├─ Stage 0 (encode, worker_type="submodule"):
       │    │    │    → StageSubModuleClient → spawn_submodule_proc()
       │    │    │      → SubModuleWorker → DiffusionSubmoduleRunner
       │    │    │        → QwenImagePipeline(stage="encode")
       │    │    │          → 加载: scheduler + text_encoder + tokenizer
       │    │    │
       │    │    ├─ Stage 1 (denoise, worker_type=None):
       │    │    │    → StageDiffusionClient → spawn_diffusion_proc()
       │    │    │      → DiffusionWorker → DiffusionModelRunner
       │    │    │        → QwenImagePipeline(stage="denoise")
       │    │    │          → 加载: scheduler + transformer
       │    │    │
       │    │    └─ Stage 2 (decode, worker_type="submodule"):
       │    │         → StageSubModuleClient → spawn_submodule_proc()
       │    │           → SubModuleWorker → DiffusionSubmoduleRunner
       │    │             → QwenImagePipeline(stage="decode")
       │    │               → 加载: vae
       │    │
       │    └─ _assemble_stage_pools()
       │         → self.stage_pools = [StagePool(0), StagePool(1), StagePool(2)]
       │
       └─ Orchestrator(stage_pools).run()  ← 主事件循环
```

进程模型：
```
主进程 (AsyncOmniEngine + Orchestrator)
  ├─ 子进程 1: StageSubModuleProc (Stage 0 - encode)
  │    └─ QwenImagePipeline(stage="encode")
  │         ├─ tokenizer + text_encoder + scheduler
  │
  ├─ 子进程 2: StageDiffusionProc (Stage 1 - denoise)
  │    └─ QwenImagePipeline(stage="denoise")
  │         ├─ transformer (DiT) + scheduler
  │
  └─ 子进程 3: StageSubModuleProc (Stage 2 - decode)
       └─ QwenImagePipeline(stage="decode")
            └─ vae
```

### 10.2 情况 2：`vllm serve --omni`（无 YAML）

启动命令：
```bash
vllm serve --omni --model ~/models/Qwen-Image
```

启动链路（与情况 1 的区别仅在 `_resolve_stage_configs`）：
```
AsyncOmniEngine.__init__()
  ├─ _resolve_stage_configs()
  │    ├─ load_stage_configs_from_model()
  │    │    └─ StageConfigFactory.create_from_model()
  │    │         ├─ _auto_detect_model_type() → model_type="qwen_image"
  │    │         ├─ 在 pipeline_registry 中找到 → PipelineConfig
  │    │         └─ merge_pipeline_deploy(PipelineConfig, DeployConfig)
  │    │              → 3 个 StageConfig (encode, denoise, decode)
  │    │
  │    └─ 如果 pipeline_registry 中没有 → fallback:
  │         _create_default_diffusion_stage_cfg()
  │           → 1 个 StageConfig (stage_id=0, model_stage="diffusion")
  │
  └─ _bootstrap_orchestrator()  ← 同情况 1
```

**关键区别**：如果 `qwen_image` 在 `pipeline_registry` 中注册了，会自动创建 3 个 stage（同 YAML）。如果没有注册，fallback 为单 stage。

单 stage fallback 的进程模型：
```
子进程 1: StageDiffusionProc (Stage 0 - diffusion)
  └─ QwenImagePipeline(stage="diffusion")
       ├─ tokenizer + text_encoder + transformer + scheduler + vae  ← 全部组件
```

### 10.3 两种情况对比

| | YAML 3-stage | 无 YAML (单 stage fallback) |
|---|---|---|
| 配置来源 | 显式 YAML 文件 | pipeline_registry 或 fallback |
| 进程数 | 3 个子进程 | 1 个子进程 |
| 模型实例 | 3 个 QwenImagePipeline（各加载部分组件） | 1 个（加载全部组件） |
| GPU 分布 | encode(NPU0), denoise(NPU1-7), decode(NPU0) | 全部在同一 NPU |
| 并发能力 | encode/decode 和 denoise 可并行 | 串行执行 |
| 适用场景 | 高吞吐、多 GPU | 简单部署、单 GPU |

---

## 十一、QwenImagePipeline 的 _STAGE_COMPONENTS 划分

### 11.1 组件定义

```python
# pipeline_qwen_image.py:257
class QwenImagePipeline(nn.Module, ...):
    _STAGE_COMPONENTS: ClassVar[dict[str, set[str]]] = {
        "diffusion": {"scheduler", "text_encoder", "tokenizer", "vae", "transformer"},
        "encode":    {"scheduler", "text_encoder", "tokenizer"},
        "denoise":   {"scheduler", "transformer"},
        "decode":    {"vae"},
    }
```

### 11.2 按需加载逻辑

```python
def __init__(self, *, od_config, prefix=""):
    self.stage = getattr(od_config, "model_stage", None) or "diffusion"
    owned_components = self._STAGE_COMPONENTS[self.stage]
    self._init_model(owned_components)

def _init_model(self, owned_components: set[str]) -> None:
    model = self.od_config.model  # ~/models/Qwen-Image

    self.scheduler = (
        FlowMatchEulerDiscreteScheduler.from_pretrained(model, subfolder="scheduler")
        if "scheduler" in owned_components else None
    )
    self.text_encoder = (
        Qwen2_5_VLForConditionalGeneration.from_pretrained(model, subfolder="text_encoder")
        if "text_encoder" in owned_components else None
    )
    if self.text_encoder is not None:
        del self.text_encoder.model.visual  # 删除不需要的 vision tower 节省显存
        self.text_encoder = self.text_encoder.to(self.device)
    self.vae = (
        DistributedAutoencoderKLQwenImage.from_pretrained(model, subfolder="vae").to(self.device)
        if "vae" in owned_components else None
    )
    self.tokenizer = (
        Qwen2Tokenizer.from_pretrained(model, subfolder="tokenizer")
        if "tokenizer" in owned_components else None
    )
    self.transformer = (
        QwenImageTransformer2DModel(od_config=od_config, ...)
        if "transformer" in owned_components else None
    )
```

### 11.3 各 stage 的执行方法

encode stage (`execute_encode`):
```python
def execute_encode(self, requests):
    for req in requests:
        tokens = self.tokenizer(prompt)
        context = self.text_encoder(tokens)
        latents = torch.randn(...)
        timesteps = self.scheduler.get_timesteps(...)
        return {context, context_mask, latents, timesteps, height, width, ...}
```

denoise stage (`pipeline.forward`):
```python
def forward(self, req):
    for t in timesteps:
        noise_pred = self.transformer(latents, t, context)
        latents = self.scheduler.step(noise_pred, t, latents)
    return {latents, height, width}
```

decode stage (`execute_decode`):
```python
def execute_decode(self, requests):
    for req in requests:
        image = self.vae.decode(latents)
        return {image}
```

### 11.4 QwenImage 的其他 pipeline 变体

```
vllm_omni/diffusion/models/qwen_image/
├── pipeline_qwen_image.py              ← 文生图 (T2I)
├── pipeline_qwen_image_edit.py         ← 图像编辑 (edit)
├── pipeline_qwen_image_edit_plus.py    ← 图像编辑增强版 (edit+)
├── pipeline_qwen_image_layered.py      ← 分层生成 (layered)
├── autoencoder_kl_qwenimage.py         ← VAE
├── qwen_image_transformer.py           ← DiT Transformer
└── cfg_parallel.py                     ← CFG 并行
```

---

## 十二、思考：Stage 分离的通用化与多 Encoder/Decoder 支持

### 12.1 问题：Stage 分离是否必须依赖 `_STAGE_COMPONENTS`？

当前的 stage 分离**强依赖**每个 Pipeline 自己定义 `_STAGE_COMPONENTS`。如果没有定义，就无法实现自定义的流水线划分。

**现状**：每个模型自己定义组件映射，没有统一的接口或协议。

### 12.2 方案 A：公共 Mixin 类

可以设计一个通用的 Mixin，让所有 Diffusion 模型自动支持 stage 划分：

```python
class DiffusionStageMixin:
    """通用的 Diffusion stage 分离 Mixin"""
    COMPONENTS: ClassVar[dict[str, set[str]]] = {}
    
    def get_stage_components(self, stage: str) -> set[str]:
        return self.COMPONENTS.get(stage, set())
    
    def execute_encode(self, requests): ...
    def execute_decode(self, requests): ...
```

**问题**：每个模型的组件名称、数量、交互方式都不同（QwenImage 用 text_encoder，其他模型可能用 clip_encoder），很难用一个通用 Mixin 覆盖所有情况。

### 12.3 方案 B：定义协议/接口

更实际的方案是定义一个**协议**，而不是 Mixin：

```python
class StageComponentProvider(Protocol):
    """每个 Pipeline 需要实现的接口"""
    def get_stage_components(self, stage: str) -> set[str]: ...
    def execute_encode(self, requests: list) -> list[dict]: ...
    def execute_decode(self, requests: list) -> list[dict]: ...
```

**优点**：
- 不强制继承，duck typing 即可
- 每个模型可以有自己的组件命名和交互方式
- Orchestrator 只需要调用接口方法，不需要知道具体实现

### 12.4 问题：多 Encoder 场景（TI2I）

考虑一个支持 Text+Image-to-Image (TI2I) 的模型：

```
T2I 任务:  text_encoder → denoise → decode
TI2I 任务: text_encoder + image_encoder → denoise → decode
```

#### 方案 1：多 Encoder 拆分为不同 stage

```
Stage 0: text_encoder (encode_text)
Stage 1: image_encoder (encode_image)  ← 只在 TI2I 任务时激活
Stage 2: denoise (需要收集 stage 0 和 stage 1 的输出)
Stage 3: decode
```

**问题**：
- Orchestrator 的 `_forward_to_next_stage` 硬编码 `next_logical = src_stage_id + 1`，无法支持 fan-in
- 需要重写为 DAG 遍历
- T2I 任务时 stage 1 为空，需要条件跳过

#### 方案 2：所有 Encoder 放在一个 stage（fused encode）

```
Stage 0: text_encoder + image_encoder (fused encode)
Stage 1: denoise
Stage 2: decode
```

**优点**：
- 不需要改 Orchestrator
- 内部可以自己决定调用哪些 encoder

**缺点**：
- 无法将不同 encoder 部署到不同设备
- 无法对不同 encoder 使用不同的并行方案

#### 方案 3：多 Encoder 作为 DAG 的并行前驱节点

```
         ┌─ text_encoder (stage 0) ─┐
         │                          ├→ fusion (stage 2) → denoise (stage 3) → decode (stage 4)
         └─ image_encoder (stage 1) ┘
```

**需要的改动**：
- Orchestrator 支持 DAG 拓扑（`_forward_to_next_stage` 改为 successor 映射）
- `pending_outputs` 缓存多父节点输出
- `custom_process_input_func` 支持多输入
- 条件执行：T2I 任务跳过 image_encoder

### 12.5 问题：多 Encoder 的部署配置

#### 方案 A：统一部署（fused）

```yaml
- stage_id: 0
  worker_type: submodule
  runtime:
    devices: "0"
  engine_args:
    model_stage: encode  # 内部包含 text_encoder + image_encoder
```

#### 方案 B：分别部署（独立 config）

```yaml
- stage_id: 0
  worker_type: submodule
  runtime:
    devices: "0"
  engine_args:
    model_stage: encode_text

- stage_id: 1
  worker_type: submodule
  runtime:
    devices: "1"  # 可以部署到不同设备
  engine_args:
    model_stage: encode_image
    tensor_parallel_size: 2  # 可以用不同的并行方案
```

#### 方案 C：混合部署（fused + 独立）

```yaml
# text_encoder 和 image_encoder 没有数据依赖，可以并行
- stage_id: 0
  worker_type: submodule
  runtime:
    devices: "0"
  engine_args:
    model_stage: encode_text

- stage_id: 1
  worker_type: submodule
  runtime:
    devices: "1"
  engine_args:
    model_stage: encode_image

# denoise 需要两个 encoder 的输出
- stage_id: 2
  runtime:
    devices: "2,3,4,5,6,7"
  engine_args:
    model_stage: denoise
  engine_input_source: [0, 1]  # ← fan-in: 依赖 stage 0 和 stage 1
```

### 12.6 问题：Denoise Stage 如何根据不同任务收集不同 Encoder 的输出

T2I 任务只需要 text_encoder 输出，TI2I 任务需要 text_encoder + image_encoder 输出。

#### 方案 1：Denoise 内部判断

```python
# custom_process_input_func
def encode_to_denoise(source_outputs, prompt, ...):
    text_output = source_outputs[0]  # 总是有
    image_output = source_outputs[1] if len(source_outputs) > 1 else None
    
    info = {"context": text_output.multimodal_output["context"], ...}
    if image_output is not None:
        info["image_latents"] = image_output.multimodal_output["image_latents"]
    return OmniTokensPrompt(additional_information=info)
```

#### 方案 2：Orchestrator 层面的条件路由

```python
# Orchestrator 根据请求类型决定跳过哪些 stage
if not has_image_input(req):
    skip_stage(1)  # 跳过 image_encoder
```

#### 方案 3：空输出占位

```python
# image_encoder 对 T2I 任务返回空输出
def execute_encode_image(self, requests):
    if not has_image_input(requests[0]):
        return [{"image_latents": None}]  # 空占位
    # ... 正常编码
```

### 12.7 问题：多 Decoders 的并行

类似多 Encoders，多个 Decoders 也可以并行：

```
denoise → ┌─ video_decoder (stage 3) ─┐
          │                           ├→ mux (stage 5) → 最终输出
          └─ audio_decoder (stage 4) ┘
```

**关键点**：
- video_decoder 和 audio_decoder 没有数据依赖，可以并行执行
- 需要一个 mux stage 来合并输出
- Orchestrator 需要支持 fan-out（一个 stage 输出到多个 stage）和 fan-in（多个 stage 输出合并到一个 stage）

### 12.8 总结：需要的架构改动

| 需求 | 当前支持 | 需要改动 |
|---|---|---|
| 单模型的 stage 划分 | ✅ `_STAGE_COMPONENTS` | 无需改动 |
| 通用化 stage 划分 | ❌ 每个模型自己定义 | 定义 `StageComponentProvider` 协议 |
| 多 Encoder fan-in | ❌ Orchestrator 硬编码线性链 | DAG 拓扑 + pending_outputs |
| 条件跳过 stage | ❌ | Orchestrator 支持条件路由 |
| 多 Decoder fan-out | ❌ | DAG 拓扑 |
| 多 Decoder 并行 | ❌ | DAG 拓扑 + 并行调度 |
| 独立部署配置 | ✅ 每个 stage 独立 YAML | 无需改动 |
| fused encode/decode | ✅ 单 stage 内部处理 | 无需改动 |

**最核心的改动**：Orchestrator 从线性链改为 DAG 拓扑，这是所有多 Encoder/Decoder 场景的前提。
