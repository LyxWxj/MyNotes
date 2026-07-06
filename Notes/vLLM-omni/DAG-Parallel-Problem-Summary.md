# DAG Parallel 问题总结与解决方案

## 目录

- [问题 1：Orchestrator 硬编码线性链路由](#问题-1orchestrator-硬编码线性链路由)
- [问题 2：SupportsComponentDiscovery 的复用](#问题-2supportscomponentdiscovery-的复用)
- [问题 3：Decoder 的热身问题](#问题-3decoder-的热身问题)
- [问题 4：Denoise 的热身问题](#问题-4denoise-的热身问题)
- [问题 5：Weight Loader 检查失败](#问题-5weight-loader-检查失败)
- [问题 6：模型注册表名称不匹配](#问题-6模型注册表名称不匹配)
- [问题 7：transformer_2 误加载](#问题-7transformer_2-误加载)
- [总结](#总结)

---

## 问题 1：Orchestrator 硬编码线性链路由

### 问题描述

```python
# orchestrator.py:929
next_logical = src_stage_id + 1  # 永远只能传给下一个 stage
```

Orchestrator 的 `_forward_to_next_stage` 硬编码 `next_logical = src_stage_id + 1`，无法支持 DAG 拓扑（fan-in/fan-out）。

### 解决方案

**1. StagePool 添加 `input_sources`**

文件：`vllm_omni/engine/stage_pool.py`

```python
class StagePool:
    def __init__(self, ..., input_sources: list[int] | None = None):
        self.input_sources: list[int] = input_sources or []
```

**2. Orchestrator 构建 successor 映射**

文件：`vllm_omni/engine/orchestrator.py`

```python
# Orchestrator.__init__ 中
self._successors: dict[int, list[int]] = {i: [] for i in range(self.num_stages)}
for pool in stage_pools:
    for src in pool.input_sources:
        self._successors.setdefault(src, []).append(pool.stage_id)
```

**3. OrchestratorRequestState 添加 `pending_outputs`**

```python
@dataclass
class OrchestratorRequestState:
    # ... 其他字段 ...
    pending_outputs: dict[int, Any] = field(default_factory=dict)  # 缓冲已完成 stage 的输出
```

**4. `_forward_to_next_stage` 改为 DAG 遍历**

```python
async def _forward_to_next_stage(self, req_id, src_stage_id, output, req_state):
    # 缓冲当前 stage 的输出
    req_state.pending_outputs[src_stage_id] = output
    
    # 遍历所有 successor
    for next_logical in self._successors.get(src_stage_id, []):
        input_sources = self.stage_pools[next_logical].input_sources
        
        # Fan-in：检查所有 input_sources 是否就绪
        if input_sources:
            ready = all(s in req_state.pending_outputs for s in input_sources)
            if not ready:
                continue  # 等待其他 predecessor
            source_outputs = [req_state.pending_outputs[s] for s in input_sources]
        else:
            source_outputs = [output]
        
        await self._submit_to_successor(...)
```

**5. `_next_stage_already_submitted` 支持 DAG**

```python
def _next_stage_already_submitted(self, stage_id, req_state):
    successors = self._successors.get(stage_id, [])
    if not successors:
        return False
    return all(s in req_state.stage_submit_ts for s in successors)
```

**6. `_route_output` 改用 `has_successors` 判断**

```python
has_successors = bool(self._successors.get(stage_id, []))
# 替代原来的 stage_id < req_state.final_stage_id
```

**7. async_omni_engine.py 传递 `input_sources`**

文件：`vllm_omni/engine/async_omni_engine.py`

```python
StagePool(
    plan.stage_idx,
    clients,
    output_processor=output_processor,
    stage_vllm_config=stage_vllm_config,
    input_sources=list(getattr(first_client, "engine_input_source", None) or []),
)
```

---

## 问题 2：SupportsComponentDiscovery 的复用

### 问题描述

每个模型需要定义 `_STAGE_COMPONENTS` 字典来声明每个 stage 需要哪些组件，这是模型特定的，无法复用。

### 解决方案

扩展 `SupportsComponentDiscovery` 协议，支持两种声明方式：

文件：`vllm_omni/diffusion/models/interface.py`

```python
@runtime_checkable
class SupportsComponentDiscovery(Protocol):
    # 粗粒度（向后兼容，供 offload 系统使用）
    _dit_modules: ClassVar[list[str]]
    _encoder_modules: ClassVar[list[str]]
    _vae_modules: ClassVar[list[str]]
    _scheduler_modules: ClassVar[list[str]] = []
    _tokenizer_modules: ClassVar[list[str]] = []
    
    # 细粒度（新增，供 stage 分离使用）
    _component_registry: ClassVar[dict[str, set[str]] | None] = None
    _default_stage_layout: ClassVar[dict[str, list[str]] | None] = None
    
    @classmethod
    def get_stage_components(cls, stage: str) -> set[str]:
        """Return the set of module attribute names required for *stage*."""
        # Path 1: fine-grained registry
        if cls._component_registry is not None and cls._default_stage_layout is not None:
            if stage == "diffusion":
                return set().union(*cls._component_registry.values())
            group_names = cls._default_stage_layout.get(stage)
            if group_names is None:
                raise ValueError(f"Unknown stage {stage!r}")
            result: set[str] = set()
            for name in group_names:
                modules = cls._component_registry.get(name)
                if modules is None:
                    raise ValueError(f"Component group {name!r} not in _component_registry")
                result.update(modules)
            return result
        
        # Path 2: coarse-grained auto-derive (backward compatible)
        if stage == "diffusion":
            return set(
                cls._encoder_modules + cls._dit_modules + cls._vae_modules
                + cls._scheduler_modules + cls._tokenizer_modules
            )
        mapping = {
            "encode": cls._encoder_modules + cls._tokenizer_modules + cls._scheduler_modules,
            "denoise": cls._dit_modules + cls._scheduler_modules,
            "decode": cls._vae_modules,
        }
        return set(mapping.get(stage, []))
```

**QwenImagePipeline 使用细粒度**：

文件：`vllm_omni/diffusion/models/qwen_image/pipeline_qwen_image.py`

```python
class QwenImagePipeline(..., SupportsComponentDiscovery):
    _component_registry: ClassVar[dict[str, set[str]]] = {
        "text_encoder":  {"tokenizer", "text_encoder"},
        "transformer":   {"transformer"},
        "scheduler":     {"scheduler"},
        "vae_decoder":   {"vae"},
    }
    _default_stage_layout: ClassVar[dict[str, list[str]]] = {
        "encode":  ["text_encoder", "scheduler"],
        "denoise": ["transformer", "scheduler"],
        "decode":  ["vae_decoder"],
    }
```

**Wan22I2VPipeline 使用细粒度（双编码器）**：

文件：`vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2_i2v.py`

```python
class Wan22I2VPipeline(..., SupportsComponentDiscovery):
    _component_registry: ClassVar[dict[str, set[str]]] = {
        "text_encoder":  {"tokenizer", "text_encoder"},
        "image_encoder": {"image_processor", "image_encoder"},
        "transformer":   {"transformer", "transformer_2"},
        "scheduler":     {"scheduler"},
        "vae_decoder":   {"vae"},
    }
    _default_stage_layout: ClassVar[dict[str, list[str]]] = {
        "encode_text":  ["text_encoder", "scheduler"],
        "encode_image": ["image_encoder"],
        "denoise":      ["transformer", "scheduler"],
        "decode":       ["vae_decoder"],
    }
```

### 不同粒度的 Stage 划分方案

通过 `_component_registry` 和 `_default_stage_layout` 的组合，可以实现不同粒度的 stage 划分。以下以多模态模型（TextEncoder + VaeEncoder + AudioEncoder → DiT → VaeDecoder + AudioDecoder）为例：

#### 方案 1：Fused Encoders + Fused Decoders（最粗粒度）

```
Stage 0 (Fused Encode): TextEncoder + VaeEncoder + AudioEncoder + Scheduler
    ↓
Stage 1 (Denoise): Transformer + Scheduler
    ↓
Stage 2 (Fused Decode): VaeDecoder + AudioDecoder
```

```python
_component_registry = {
    "text_encoder":   {"tokenizer", "text_encoder"},
    "vae_encoder":    {"vae_encoder"},
    "audio_encoder":  {"audio_encoder"},
    "transformer":    {"transformer"},
    "scheduler":      {"scheduler"},
    "vae_decoder":    {"vae"},
    "audio_decoder":  {"audio_decoder"},
}
_default_stage_layout = {
    "encode":  ["text_encoder", "vae_encoder", "audio_encoder", "scheduler"],
    "denoise": ["transformer", "scheduler"],
    "decode":  ["vae_decoder", "audio_decoder"],
}
```

#### 方案 2：Fused Encoders + Disaggregated Decoders

```
Stage 0 (Fused Encode): TextEncoder + VaeEncoder + AudioEncoder + Scheduler
    ↓
Stage 1 (Denoise): Transformer + Scheduler
    ↓
    ├→ Stage 2 (VaeDecode): VaeDecoder
    └→ Stage 3 (AudioDecode): AudioDecoder
```

```python
_component_registry = {
    "text_encoder":   {"tokenizer", "text_encoder"},
    "vae_encoder":    {"vae_encoder"},
    "audio_encoder":  {"audio_encoder"},
    "transformer":    {"transformer"},
    "scheduler":      {"scheduler"},
    "vae_decoder":    {"vae"},
    "audio_decoder":  {"audio_decoder"},
}
_default_stage_layout = {
    "encode":        ["text_encoder", "vae_encoder", "audio_encoder", "scheduler"],
    "denoise":       ["transformer", "scheduler"],
    "decode_vae":    ["vae_decoder"],
    "decode_audio":  ["audio_decoder"],
}
```

#### 方案 3：Disaggregated Encoders + Fused Decoders

```
Stage 0 (TextEncode): TextEncoder + Tokenizer + Scheduler
    ↓
    ├→ Stage 1 (VaeEncode): VaeEncoder
    └→ Stage 2 (AudioEncode): AudioEncoder
         ↓
    Stage 3 (Denoise): Transformer + Scheduler  ← fan-in，等待 Stage 0,1,2 全部完成
         ↓
    Stage 4 (Fused Decode): VaeDecoder + AudioDecoder
```

```python
_component_registry = {
    "text_encoder":   {"tokenizer", "text_encoder"},
    "vae_encoder":    {"vae_encoder"},
    "audio_encoder":  {"audio_encoder"},
    "transformer":    {"transformer"},
    "scheduler":      {"scheduler"},
    "vae_decoder":    {"vae"},
    "audio_decoder":  {"audio_decoder"},
}
_default_stage_layout = {
    "encode_text":  ["text_encoder", "scheduler"],
    "encode_vae":   ["vae_encoder"],
    "encode_audio": ["audio_encoder"],
    "denoise":      ["transformer", "scheduler"],
    "decode":       ["vae_decoder", "audio_decoder"],
}
```

#### 方案 4：Disaggregated Encoders + Disaggregated Decoders（最细粒度）

```
Stage 0 (TextEncode): TextEncoder + Tokenizer + Scheduler
    ↓
    ├→ Stage 1 (VaeEncode): VaeEncoder
    └→ Stage 2 (AudioEncode): AudioEncoder
         ↓
    Stage 3 (Denoise): Transformer + Scheduler  ← fan-in，等待 Stage 0,1,2 全部完成
         ↓
         ├→ Stage 4 (VaeDecode): VaeDecoder
         └→ Stage 5 (AudioDecode): AudioDecoder
```

```python
_component_registry = {
    "text_encoder":   {"tokenizer", "text_encoder"},
    "vae_encoder":    {"vae_encoder"},
    "audio_encoder":  {"audio_encoder"},
    "transformer":    {"transformer"},
    "scheduler":      {"scheduler"},
    "vae_decoder":    {"vae"},
    "audio_decoder":  {"audio_decoder"},
}
_default_stage_layout = {
    "encode_text":  ["text_encoder", "scheduler"],
    "encode_vae":   ["vae_encoder"],
    "encode_audio": ["audio_encoder"],
    "denoise":      ["transformer", "scheduler"],
    "decode_vae":   ["vae_decoder"],
    "decode_audio": ["audio_decoder"],
}
```

---

## 问题 3：Decoder 的热身问题

### 问题描述

第一次运行 `vae.decode()` 非常慢（~20s），第二次正常（~0.5s）。原因是 CUDA warmup（context 初始化、CuDNN benchmarking、显存分配）。

### 解决方案

在 `DiffusionSubmoduleRunner.load_model()` 末尾添加 warmup：

文件：`vllm_omni/diffusion/worker/diffusion_submodule_runner.py`

```python
def load_model(self, ...):
    # ... 原有加载逻辑 ...
    self._warmup()  # 添加 warmup

def _warmup(self) -> None:
    """Warmup CUDA kernels to avoid first-call latency."""
    stage = getattr(self.od_config, "model_stage", None)
    if stage != "decode" or self.pipeline is None:
        return
    if not hasattr(self.pipeline, "vae") or self.pipeline.vae is None:
        return

    logger.info("DiffusionSubmoduleRunner[decode]: warming up VAE decode...")
    t0 = time.perf_counter()
    try:
        with torch.inference_mode():
            dummy = torch.randn(
                1, self.pipeline.vae.config.z_dim, 1, 16, 16,
                device=self.device,
                dtype=self.pipeline.vae.dtype,
            )
            self.pipeline.vae.decode(dummy, return_dict=False)
        logger.info(
            "DiffusionSubmoduleRunner[decode]: warmup done in %.3fs",
            time.perf_counter() - t0,
        )
    except Exception:
        logger.warning(
            "DiffusionSubmoduleRunner[decode]: warmup failed (%.3fs), "
            "first real request may be slow",
            time.perf_counter() - t0,
            exc_info=True,
        )
```

---

## 问题 4：Denoise 的热身问题

### 问题描述

denoise stage 使用完整 `DiffusionEngine`，其 `_dummy_run()` 尝试运行完整 pipeline，但 disaggregated 模式下 `vae`、`text_encoder` 等组件是 None，导致失败。

错误信息：

```
RuntimeError: Dummy run failed: 'NoneType' object has no attribute 'to'
RuntimeError: Dummy run failed: Image is required for I2V generation.
```

### 当前的临时解决方案

在 pipeline 中实现 `build_dummy_run_request()` classmethod：

文件：`vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2_i2v.py`

```python
@classmethod
def build_dummy_run_request(cls, od_config, *, height, width, num_inference_steps):
    stage = getattr(od_config, "model_stage", None) or "diffusion"
    
    # 非 denoise stage 跳过 dummy run
    if stage != "diffusion" and stage != "denoise":
        return None
    
    # 单 stage 模式：使用默认 dummy request
    if stage == "diffusion":
        return OmniDiffusionRequest(
            prompts=[{"prompt": "dummy run"}],
            request_id=DUMMY_DIFFUSION_REQUEST_ID,
            sampling_params=OmniDiffusionSamplingParams(...),
        )
    
    # Denoise stage：使用预构建的虚拟 tensor
    latent_condition = torch.randn(...)
    first_frame_mask = torch.zeros(...)
    first_frame_mask[:, :, 1:, :, :] = 1.0
    
    return OmniDiffusionRequest(
        prompts=[OmniTokensPrompt(additional_information={
            "prompt_embeds": torch.zeros(...),
            "latents": torch.randn(...),
            "timesteps": torch.linspace(...),
            "latent_condition": latent_condition,
            "first_frame_mask": first_frame_mask,
            "expand_timesteps": True,
        })],
        request_id=DUMMY_DIFFUSION_REQUEST_ID,
        sampling_params=OmniDiffusionSamplingParams(...),
    )
```

### 问题

这个方案需要每个模型自己实现 `build_dummy_run_request()`，而且需要构造虚拟 tensor，逻辑复杂且容易出错。

### 更好的设计方案（未实现）

在**更高层次的调度器**（Orchestrator 或 StagePool）上进行热身：

```
Orchestrator / StagePool 层热身：
├─ 不需要知道模型内部结构
├─ 不需要构造虚拟 tensor
├─ 可以统一处理所有模型
└─ 热身逻辑与业务逻辑分离

Pipeline / Runner 层热身：
├─ 需要知道模型内部结构
├─ 需要构造虚拟 tensor
├─ 每个模型都要实现
└─ 热身逻辑与业务逻辑耦合
```

实现思路：

```python
class Orchestrator:
    async def _warmup_stage(self, stage_id: int):
        """在 Orchestrator 层进行 stage 热身。"""
        pool = self.stage_pools[stage_id]
        
        # 对于 submodule stage（encode/decode），直接跳过
        # 因为 submodule 路径的 warmup 已经在 DiffusionSubmoduleRunner 中处理
        
        # 对于 denoise stage（完整 DiffusionEngine），
        # 发送一个真实的 dummy request，而不是构造虚拟 tensor
        dummy_request = self._create_dummy_request(stage_id)
        if dummy_request:
            await pool.submit_initial("warmup", dummy_request)
            # 等待 warmup 完成
```

优点：

1. 不需要每个模型实现 `build_dummy_run_request()`
2. 不需要构造虚拟 tensor
3. 热身逻辑与业务逻辑分离
4. 可以统一处理所有模型

---

## 问题 5：Weight Loader 检查失败

### 问题描述

```
ValueError: Following weights were not initialized from checkpoint: 
{'text_encoder.encoder.block.5.layer.0.layer_norm.weight', ...}
```

encode_image stage 不加载 text_encoder，但 weight loader 检查发现 text_encoder 权重没有被加载。

### 解决方案

在 pipeline 的 `__init__` 中设置 `weights_loaded_by_model_init` 标志：

文件：`vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2_i2v.py`

```python
# 当 stage 不需要 transformer 时（如 encode_image），
# 告诉 weight loader 权重已在模型初始化时加载，跳过严格检查
self.weights_loaded_by_model_init = not owns_transformer
```

当 `owns_transformer = False` 时（encode_text、encode_image、decode stage），weight loader 会跳过检查，因为这些 stage 的权重是通过 `from_pretrained` 加载的，不是通过 `load_weights`。

---

## 问题 6：模型注册表名称不匹配

### 问题描述

```
ValueError: Model class Wan22I2VPipeline not found in diffusion model registry.
```

YAML 中的 `model_class_name` 用的是实际 Python 类名，但注册表使用的是不同的 key。

### 解决方案

注册表中的映射：

```python
# registry.py
"WanImageToVideoPipeline":    # ← 注册表 key（YAML 中应该用这个）
    "wan2_2",
    "pipeline_wan2_2_i2v",
    "Wan22I2VPipeline",        # ← 实际 Python 类名
```

YAML 中应该使用注册表的 key：

```yaml
# 错误
model_class_name: Wan22I2VPipeline

# 正确
model_class_name: WanImageToVideoPipeline
```

---

## 问题 7：transformer_2 误加载

### 问题描述

```
RuntimeError: Cannot find any model weights with `/home/lyx1/models/Wan2.2-TI2V-5B-Diffusers/`
```

`Wan2.2-TI2V-5B-Diffusers` 的 `model_index.json` 中 `transformer_2` 存在但值是 `[null, null]`，代码只检查了 key 是否存在，没有检查值是否为 null。

### 解决方案

文件：`vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2_i2v.py`

```python
# 原来的代码（错误）
self.has_transformer_2 = "transformer_2" in model_index

# 修复后的代码
self.has_transformer_2 = (
    "transformer_2" in model_index
    and model_index["transformer_2"][0] is not None
)
```

---

## 总结

| 问题 | 状态 | 解决方案 | 涉及文件 |
|---|---|---|---|
| Orchestrator 线性链路由 | ✅ 已解决 | DAG 拓扑 + successor 映射 + pending_outputs | orchestrator.py, stage_pool.py, async_omni_engine.py |
| SupportsComponentDiscovery 复用 | ✅ 已解决 | 细粒度 `_component_registry` + `_default_stage_layout` | interface.py, pipeline_qwen_image.py, pipeline_wan2_2_i2v.py |
| Decoder 热身 | ✅ 已解决 | `DiffusionSubmoduleRunner._warmup()` | diffusion_submodule_runner.py |
| Denoise 热身 | ⚠️ 临时方案 | `build_dummy_run_request()` + 虚拟 tensor | pipeline_wan2_2_i2v.py |
| Denoise 热身（理想方案） | ❌ 未实现 | Orchestrator 层统一热身 | - |
| Weight Loader 检查失败 | ✅ 已解决 | `weights_loaded_by_model_init` 标志 | pipeline_wan2_2_i2v.py |
| 模型注册表名称不匹配 | ✅ 已解决 | 使用注册表 key 而非类名 | YAML 配置 |
| transformer_2 误加载 | ✅ 已解决 | 检查值是否为 null | pipeline_wan2_2_i2v.py |

---

## 开发过程中的思考与困惑

### 1. 关于 PR #3076 的复用

PR #3076 已经合并，它为 offload 场景设计了 `SupportsComponentDiscovery` 组件感知基类。我的思路是扩展这个协议，复用组件感知的能力，再加上组件拓扑信息（`_component_registry` + `_default_stage_layout`），这样：

- Offload 系统可以知道哪些模块是什么角色（encoder/dit/decoder）
- Stage 分离系统可以知道每个 stage 需要哪些模块
- 两者共享同一套组件声明，不需要重复定义

### 2. Orchestrator 改动的风险

Orchestrator 的改动在 Qwen 模型上测试没问题，但改 Orchestrator 这个组件非常危险，因为：

- 它是所有模型共享的核心调度器
- 改动可能影响所有模型的行为
- 线性链的假设渗透在很多地方（`_forward_to_next_stage`、`_next_stage_already_submitted`、`_route_output`）

encoder 分离感觉真的需要改 Orchestrator 的 stage 传送逻辑，但这个改动风险很高。

### 3. Denoise 热身的困境

在 Wan 模型上遇到的问题：DiffusionRunner 不持有所有组件（vae、text_encoder 是 None），所以没办法用传统方式 warmup。

改了很久改不清楚这一块。当前的 `build_dummy_run_request()` 方案是临时的，需要每个模型自己构造虚拟 tensor，逻辑复杂且容易出错。

### 4. SubModuleRunner 的热身问题

原来的 SubModuleRunner 也需要 warmup，不 warmup 的话第一次 decode 可能会非常慢（~20s）。但这个方案感觉不太好，因为：

- 分组件的话每个模型都要重新写自己的 warmup 逻辑
- 热身逻辑与业务逻辑耦合
- 不同模型的 warmup 需求不同

### 5. 理想的热身方案

还是觉得应该在更高层次的调度器上进行热身（Orchestrator 或 StagePool 层），这样：

- 不需要知道模型内部结构
- 不需要构造虚拟 tensor
- 可以统一处理所有模型
- 热身逻辑与业务逻辑分离

但这个方案需要修改 Orchestrator，又回到了 " 改 Orchestrator 风险很高 " 的问题。

### 6. 总结

- **已解决**：Orchestrator DAG 拓扑、SupportsComponentDiscovery 扩展、Decoder warmup、Weight Loader 检查
- **临时方案**：Denoise warmup（每个模型自己实现 `build_dummy_run_request()`）
- **未解决**：如何在不修改 Orchestrator 的情况下统一处理所有模型的 warmup
- **核心矛盾**：组件分离需要改 Orchestrator，但改 Orchestrator 风险很高

---

## 相关文件索引

| 文件 | 用途 |
|---|---|
| `vllm_omni/engine/orchestrator.py` | Orchestrator（调度循环、DAG 路由） |
| `vllm_omni/engine/stage_pool.py` | StagePool（replica 管理、input_sources） |
| `vllm_omni/engine/async_omni_engine.py` | AsyncOmniEngine（stage 初始化） |
| `vllm_omni/diffusion/models/interface.py` | SupportsComponentDiscovery 协议 |
| `vllm_omni/diffusion/models/qwen_image/pipeline_qwen_image.py` | QwenImagePipeline |
| `vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2_i2v.py` | Wan22I2VPipeline |
| `vllm_omni/diffusion/worker/diffusion_submodule_runner.py` | Submodule Runner（decode warmup） |
| `vllm_omni/diffusion/diffusion_engine.py` | DiffusionEngine（dummy run） |
| `vllm_omni/diffusion/registry.py` | 模型注册表 |
| `vllm_omni/model_executor/stage_configs/wan22_i2v_dag.yaml` | Wan2.2 I2V DAG 配置 |
| `vllm_omni/model_executor/stage_input_processors/wan22_i2v.py` | Wan2.2 I2V stage processor |
