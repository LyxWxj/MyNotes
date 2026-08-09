# vLLM Diffusion 可复用 vLLM-Omni 特性

## 1. 结论摘要

| 方向  | 最高收益项                                                         | 前置条件                       |
| --- | ------------------------------------------------------------- | -------------------------- |
| 传输  | 大张量改 POSIX SHM / OmniConnector，去掉 ZMQ+pickle 整包拷贝             | 同机部署（当前脚本全部 127.0.0.1）     |
| 传输  | 批量 decode RPC、`max_steps` 分块 denoise 接线（已实现未启用）               | 纯接线，改动小                    |
| 缓存  | PromptEmbedCache 开关透传（重复 prompt 跳过 text encoder）              | 加 CLI 参数                   |
| 缓存  | cache_dit（DBCache+SCM+TaylorSeer）接入分离场景，替代/补充 TeaCache        | 放开 disagg 白名单 + 泛化 refresh |
| 计算  | denoiser 连续批处理（`max_num_seqs>1`）、DiT PP、分布式 layerwise offload | 架构级改造                      |
| 调度  | 流式/渐进输出、least-queue-length、运行期健康管理                            | API/编排层                    |

---

## 2. 现状基线（已利用的上游能力）

- 引擎层：`DiffusionEngine` / `DiffusionWorker` / `DiffusionModelRunner`（`vllm_diffusion/engine.py`）
- 并行：USP / CFG-parallel / HSDP / VAE patch-PP（通过 CLI 参数提供 `--usp/--cfg-parallel-size/--use-hsdp/--vae-patch-parallel-size`）
- 分离场景：`step_execution=True` 角色 Pipeline、ZMQ Head↔Worker、TeaCache（仅 `tea_cache` + 命中率统计）、stage 间 `empty_cache`、自实现 sleep/wake、批量 encode RPC、多副本 Round-Robin

---

## 3. 传输优化（未利用）

### 3.1 大张量 SHM 传输（P0，收益最大）

- **现状**：`vllm_diffusion/disaggregation/codec.py` 将 `DiffusionRequestState` 全部张量搬到 CPU 后整包 pickle；`transport.py` 走 `tcp://` ZMQ REQ/REP。每次角色交接 = D2H → pickle → TCP → unpickle → H2D，Wan 视频单次 handoff 数 MB~几十 MB（latents + prompt_embeds + scheduler 状态）。
- **上游方案**：`vllm_omni/diffusion/ipc.py` 已实现 >1MB 张量写 POSIX SHM、队列只传 metadata 句柄的机制（支持 side-stream 非阻塞 D2H + bf16→fp32 打包），用于 omni 的 Worker↔Scheduler 传输。
- **建议**：同机部署（当前脚本全是 127.0.0.1）把大张量替换为 SHM 句柄，仅小对象走 ZMQ；跨机时再升级到 3.2 的 connector。

### 3.2 OmniConnector 体系（RDMA / Mooncake / NPU P2P）

- **上游**：`vllm_omni/distributed/omni_connectors/` 提供统一 `put/get`：单机 `SharedMemoryConnector`（默认自动选）、跨机 `MooncakeStoreConnector`（TCP）/ `MooncakeTransferEngineConnector`（RDMA）、`MoriTransferEngineConnector`、Ascend NPU P2P 的 `YuanrongTransferEngineConnector`（`memory_pool_device: "npu"`）。设计文档：`docs/design/feature/disaggregated_inference.md`、`docs/design/feature/omni_connectors/*.md`。
- **现状**：`docs/modules.md` 能力矩阵中 "RDMA / Mooncake" 标注 ⏳ 后续，尚未接入，但目前支持的模型（Qwen-Image, Wan22 暂时不需要 KVCache 传输）。
- **建议**：数据面换 connector（控制面保留 ZMQ）；单机先落 SharedMemory，跨机按硬件选 Mooncake/Yuanrong TE。

### 3.3 异步 D2H + 后台打包（async diffusion output）

- **上游**：`docs/design/feature/async_diffusion_output.md` — D2H 拷贝与 SHM 打包移到后台线程/side stream，消除 " 上一个请求输出搬运 " 造成的 GPU 气泡；`step_execution=False` 时自动启用。
- **现状**：分离 Worker 强制 `step_execution=True`（`vllm_diffusion/disaggregation/worker_runtime.py:137`），每次 handoff 的同步 D2H/pickle 都是气泡。
- **建议**：在 RoleWorker 的 encode/denoise/decode 出口实现同款后台 D2H + 打包。

### 3.4 已实现但 Head 未接线的能力

- **分块 denoise**：`worker_runtime.py:386` 支持 `max_steps`，codec 已传递 `chunk_index/chunk_num_steps`，但 orchestrator 从不传 `max_steps`。启用后可实现 "denoiser 分块 + decoder 与下一块并行 "（对标 omni `async_chunk`，见 `docs/design/feature/async_chunk.md`）。

### 3.5 负载均衡与健康管理

- **上游**：`vllm_omni/distributed/omni_coordinator/load_balancer.py` 提供 `round-robin / random / least-queue-length` 策略；`omni_coordinator` / `stage_pool` / `membership_controller` 提供副本注册、健康检查、生命周期管理。
- **现状**：仅 `RoundRobinPool` + 启动 ping + 超时重连，无运行期健康检查、失败剔除/重启。
- **建议**：长视频任务（秒级~分钟级）下 least-queue-length 明显优于轮询；随后补运行期健康管理。

---

## 4. 缓存优化（未利用）

### 4.1 Cache-DiT vs TeaCache

上游：

- `vllm-omni` 已内置 cache-dit 集成：`vllm_omni/diffusion/cache/cachedit/`（`CacheDiTBackend`、`enable_cache_for_dit`、Wan2.2 自定义 enabler `enable_cache_for_wan22`，含高/低噪双 DiT 拆分与 S2V）；cache selector 已支持 `cache_dit`（`vllm_omni/diffusion/cache/selector.py`）。

#### 4.1.1 算法机制对比

| 维度 | TeaCache（omni 实现） | Cache-DiT（DBCache 为主） |
|------|----------------------|--------------------------|
| 缓存粒度 | **整网二值**：要么完整计算整个 transformer，要么复用上一 step 的输出/residual | **逐 block 独立决策**：每个 transformer block 按自身输出 L1 diff 决定算/跳；粒度可达块级（`(L-Fn-Bn)/L` 的跳过比） |
| 决策信号 | 代理信号：timestep 调制输入（modulated input）的 rel-L1，经模型专属多项式系数 rescale 映射到输出误差 | 直接信号：真正被复用的 block 输出 `‖h_t^(l) − h_(t−1)^(l)‖₁` 与阈值 τ 比较 |
| 误差累积控制 | 无内置机制（只有单阈值） | 双块机制：前 `F_n` 块每步必算（稳定 diff 参考）+ 后 `B_n` 块每步必算（auto-scaler 吸收误差）；另有 warmup、`max_continuous_cached_steps` 防漂移、`force_refresh_step_hint` 强制刷新 |
| Step 级调度 | 无 | **SCM（Steps Computation Mask）**：早期 step 全算、后期 step 缓存；28-step 预设 slow/medium/fast/ultra，支持 dynamic 模式按 residual 轨迹自适应 |
| 外推校准 | 无 | **TaylorSeer**（多项式基 Taylor 展开预测下一 step 特征，预缓存提前介入，配 `B_n=0` 效果最佳）；**DMD**（指数基/线性动力系统 SVD 拟合，长外推不发散） |
| 块内剪枝 | 无 | **DBPrune**：块内子层按 L1 距离剪枝（24%~60%） |
| CFG | 正/负分支独立 state，CFG-parallel 兼容 | 独立 CFG state + 可配置 CFG 计算顺序（Wan/Qwen 用 `enable_separate_cfg=True`） |
| 模型接入 | extractor 机制，模型无关 | BlockAdapter + ForwardPattern（0~5）+ PatchFunctor，40+ pipeline 系列；Wan2.2 双 DiT 可对 `transformer`/`transformer_2` 分别设参 |
| 调参面 | 单阈值 `rel_l1_thresh`（+ 可选系数） | Fn/Bn/τ/warmup/max_cached_steps/max_continuous_cached_steps/SCM/TaylorSeer order… |
| 决策开销 | 极低（只需比较一个调制输入张量） | 较高（需保存并比较每 block 隐藏态，消耗显存/带宽；F_n+B_n 块每步必算） |

#### 4.1.2 基准数据（FLUX.1-dev, 50 steps, DrawBench, L20）

来源：`cache-dit/bench/cache/README.md` 与 `docs/benchmark/HYBRID_CACHE.md`。注意：除 DBCache 外，其余方法（含 TeaCache）数据引自 FoCa 论文（arxiv:2508.16211），为跨论文对照，同机同条件复测前仅作量级参考。

| 方法 | TFLOPs↓ | 加速↑ | ImageReward↑ | CLIP Score↑ |
|------|--------|-------|-------------|-------------|
| 无缓存基线 | 3726.87 | 1.00× | 0.9898 | 32.404 |
| **TeaCache(l=1.2)** | 669.27 | **5.56×** | **0.7394** | **31.704** |
| DBCache(F) | 651.90 | 5.72× | 0.9271 | 32.552 |
| DBCache(F)+TaylorSeer | 651.90 | 5.72× | 0.9526 | 32.568 |
| **DBCache(U)+TaylorSeer** | 505.47 | **7.37×** | 0.8645 | **32.719** |
| DBCache(S) 保守 | 1400.08 | 2.66× | 1.0065 | 32.838 |
| DBCache(S)+TaylorSeer | 1153.05 | 3.23× | 1.0221 | 32.819 |

1. **同加速比下质量显著更高**：5.6~5.7× 档，ImageReward 0.927 vs 0.739（+25%），CLIP 32.55 vs 31.70。
2. **更高速档仍优于 TeaCache 的 " 极速档 "**：DBCache(U)+TS 7.37× 的 ImageReward（0.8645）与 CLIP（32.719）均高于 TeaCache 5.56× 档。
3. **保守档质量反超无缓存基线**（1.0065 vs 0.9898），说明适度缓存有正则化效果，可作为高保真默认档。
4. **蒸馏少步模型鲁棒**：Qwen-Image-Lightning 4 步下 F16B16 仍保持 ImageReward 1.2614（基线 1.2630）、PSNR 34.8。

#### 4.1.3 Cache-DiT 相对 TeaCache 的优势（总结）

1. **粒度与信号更直接**：block 级决策用 " 真正被复用的输出 " 做差，TeaCache 用时间嵌入代理信号做整网二值决策；block 级决策天然允许 " 部分重算 "，高阈值下质量劣化更慢。
2. **误差累积有闭环控制**：`F_n`/`B_n` 双块 + warmup + `max_continuous_cached_steps` + 强制刷新点，TeaCache 除阈值外没有任何防漂移机制。
3. **可组合的混合缓存栈**：块级（DBCache）× step 级（SCM）× 外推校准（TaylorSeer/DMD）× 块内剪枝（DBPrune），可自由叠加；TeaCache 是单一机制。
4. **速度 - 质量 Pareto 前沿更优**：同速更高质、更高速不劣质（见 4.1.2）。
5. **与当前框架契合度高**：omni 已封装 `CacheDiTBackend`，且 Wan2.2 enabler 已支持 `transformer`/`transformer_2` 双 DiT 高/低噪拆分——正好对齐 vllm-diffusion 的 Wan 双 DiT 分离部署；Qwen-Image 单 transformer 走自动 `enable_cache_for_dit` 即可。

#### 4.1.4 客观劣势 / 代价（集成前需评估）

- **状态与计算开销**：需缓存并比较每个 block 的隐藏态（显存/带宽占用），且 `F_n`+`B_n` 块每步必算，整网完全跳过的上限受 `(L−Fn−Bn)/L` 限制；TeaCache 决策开销可忽略。
- **配置面大**：Fn/Bn/τ/warmup/MC/SCM/TaylorSeer 等参数需要按模型调优（cache-dit 提供默认 F8B0/W8 组合）。
- **集成复杂度**：依赖 `cache_dit` 第三方包（BlockAdapter/ForwardPattern/PatchFunctor）；分离场景需在 RoleWorker 侧做与 tea_cache 同等的 per-request `refresh_context` + 统计改造（`cache_dit.summary()`）。
- **基准口径**：上述对比中 TeaCache 数据引自 FoCa 论文，建议接入后用统一 prompt 集（DrawBench/自建）在本机复测 ImageReward/CLIP/PSNR。

#### 4.1.5 当前框架落地路径

1. 泛化 `vllm_diffusion/disaggregation/cache.py`：目前仅放行 `tea_cache`（`backend_name != "tea_cache"` 直接 raise），需改为按 backend 分派 refresh 接口（cache_dit 为 `cache_dit.refresh_context(transformer, num_inference_steps=…)`）与统计接口（`cache_dit.summary()`），并保持 per-request 隔离。
2. CLI：`--cache-backend` 帮助文案已过时（`cli.py:107` 写的是 "none, tea_cache, deep_cache"），更新为 `none | tea_cache | cache_dit | mag_cache | step_cache`，并透传 cache-dit 参数（omni `DiffusionCacheConfig` 已含 `Fn_compute_blocks / Bn_compute_blocks / residual_diff_threshold / max_warmup_steps / max_continuous_cached_steps / enable_taylorseer / taylorseer_order / scm_steps_mask_policy` 等，`vllm_omni/diffusion/data.py:382`）。
3. 先验证 monolithic（omni `DiffusionModelRunner` 已按 selector 自动安装 `CacheDiTBackend`），再接入分离 denoiser。
4. Wan2.2 双 DiT 分离直接用 omni 的 `enable_cache_for_wan22`（内部已按高/低噪拆分 step），注意其与 `denoise_phase` 角色切分的交互。
5. 建立统一评测：同 prompt 集 × 同 seed，记录 ImageReward / CLIP / PSNR / TFLOPs / 端到端延迟，与 TeaCache 对比。

### 4.2 PromptEmbedCache（P0，未暴露）

- **上游**：`vllm_omni/diffusion/cache/prompt_embed_cache.py` — LRU 缓存 `encode_prompt` 输出（key 由可哈希参数构造，遇 tensor/PIL 自动绕过）；`DiffusionModelRunner` 加载后自动 wrap（`vllm_omni/diffusion/worker/diffusion_model_runner.py:309`），配置项 `enable_prompt_embed_cache` / `prompt_embed_cache_size`（默认 32）。
- **现状**：CLI 未暴露开关（默认关闭）。同 prompt 多 seed（benchmark、GRPO 类重复请求）是 text encoder 的典型浪费场景；分离部署下 encoder 是独立进程，打开即生效。
- **建议**：CLI 加 `--enable-prompt-embed-cache` / `--prompt-embed-cache-size`，mono 与 encoder role 同时透传。

### 4.3 其它 cache backend（部分适用）

- `mag_cache`：Flux/Flux2 系策略（`vllm_omni/diffusion/cache/magcache/`），当前 Qwen/Wan 不适用。
- `step_cache`：DreamZero 系 velocity step skip，当前 Qwen/Wan 不适用。
- TeaCache 参数面：目前只透传 `rel_l1_thresh`，omni 还有模型系数表（Qwen-Image 已内置）与更多配置项（`vllm_omni/diffusion/cache/teacache/config.py`）。
- KV/前缀缓存、session state manager（`vllm_omni/experimental/world_models`）、AR KV 传输（`distributed/omni_connectors/kv_transfer_manager.py`）：纯 diffusion 无 AR 阶段，暂不适用；若未来做视频续帧/世界模型再接入。

---

## 5. 计算与调度优化（未利用）

| 特性 | 上游位置 | 现状 | 收益/场景 |
|------|---------|------|----------|
| denoiser 连续批处理 | `docs/design/feature/diffusion_continuous_batching.md` | 分离 Worker 强制 `max_num_seqs=1` | 同形状/同 CFG 多请求共享 `denoise_step`，DiT 是计算大头，吞吐提升空间最大 |
| 流式/渐进输出 | `DiffusionEngine.step_streaming`（`vllm_omni/diffusion/diffusion_engine.py:304`） | API 仅同步 mp4 + 异步任务轮询 | 逐 step 吐中间结果，长视频 TTFP 体验 |
| DiT 流水并行 | `docs/design/feature/pipeline_parallel.md`；`pipeline_parallel_size`（`vllm_omni/diffusion/data.py:149`） | CLI 未暴露（只有 VAE patch-PP） | 8 卡以上把单 DiT 按层切 PP，降显存、放大 batch |
| Sleep Mode level 2 | `docs/features/sleep_mode.md`；`DiffusionWorker.sleep(level)`（`vllm_omni/diffusion/worker/diffusion_worker.py:565`） | 自实现 `disagg_sleep/wake` 仅逐参数 CPU 搬移 | VMM 页回收可释放 95%+ 显存，共卡多角色切换更省 |
| Diffusion LoRA | `vllm_omni/diffusion/lora/`；`lora_path` | CLI 未暴露 | 多用户个性化生成 |
| MoE Expert Parallel | `docs/design/feature/expert_parallel.md`；`enable_expert_parallel` | CLI 未暴露 | MoE 模型（HunyuanImage3.0 参考）；Wan/Qwen 无 MoE 则不适用 |
| regional compile | `vllm_omni/diffusion/compile.py`；`diffusion_compile_granularity` | CLI 未暴露 | torch.compile 重复块，Ascend 需验证算子支持 |
| Ray executor | `distributed_executor_backend="ray"` | 仅 mp executor | 多机部署 |
| ComfyUI 入口 | `docs/features/comfyui.md` | 无 | 生态接入 |

## 6. 参考文件

- 当前框架：`vllm_diffusion/disaggregation/transport.py`、`codec.py`、`cache.py`、`orchestrator.py`、`worker_runtime.py`、`worker_ext.py`、`entrypoints/cli.py`、`docs/modules.md`
- vllm-omni：`vllm_omni/diffusion/ipc.py`、`vllm_omni/diffusion/cache/`（`teacache/`、`cachedit/`、`prompt_embed_cache.py`、`selector.py`）、`vllm_omni/distributed/omni_connectors/`、`vllm_omni/distributed/omni_coordinator/load_balancer.py`、`vllm_omni/diffusion/diffusion_engine.py`、`docs/design/feature/`（`async_chunk.md`、`async_diffusion_output.md`、`diffusion_continuous_batching.md`、`cache_dit.md`、`disaggregated_inference.md`、`pipeline_parallel.md`）
- cache-dit：`docs/user_guide/DBCACHE_DESIGN.md`、`docs/user_guide/CACHE_API.md`、`docs/papers/paper.tex`、`docs/benchmark/HYBRID_CACHE.md`、`bench/cache/README.md`

## 7. NVIDIA 已启用、Ascend NPU 可启用但未利用的特性

> 本清单聚焦 "NVIDIA 上已在用、Ascend NPU 上基础设施已具备或官方支持、但当前框架（含 omni diffusion 路径）尚未利用 " 的能力。标注均基于本地代码/文档核实。

### 7.1 ACLGraph（CUDA Graph → Ascend ACL Graph，重点）

- **NVIDIA 侧**：vLLM 系 AR/生成引擎使用 CUDA Graph 捕获（`vllm.compilation.cuda_graph`）；diffusion 侧在 NVIDIA 上通过 `torch.compile` 选项 `{"triton.cudagraphs": True}` 启用 CUDA Graph（cache-dit `docs/user_guide/CUDA_GRAPH.md`，compile 之上再降 kernel launch 开销，与 FP8 组合更稳定）。
- **Ascend 侧**：vllm-omni NPU platform 已实现 `get_graph_wrapper_cls()` → `vllm_ascend.compilation.acl_graph.ACLGraphWrapper`，以及 `set_forward_context(…, aclgraph_runtime_mode=…)`（`vllm_omni/platforms/npu/platform.py:235`），但**只用于 AR/generation 引擎**（`npu_ar_model_runner.py`、`npu_model_runner.py`）；`DiffusionModelRunner` 没有任何图捕获路径，且 NPU 上 `supports_torch_inductor() == False` 直接跳过编译。

### 7.2 torch.compile（regional / full）

- **NVIDIA**：`enforce_eager=False` 时 `DiffusionModelRunner` 自动 `pipeline.setup_compile()` 或 `regionally_compile(transformer/transformer_2)`（`vllm_omni/diffusion/worker/diffusion_model_runner.py:271-289`）。
- **Ascend**：`supports_torch_inductor()` 返回 False，被显式跳过并告警；Ascend 等价物是 TorchAIR/ACL Graph 图模式，未接线。

### 7.3 FP8 动态量化

- **NVIDIA**：diffusion 支持动态 FP8（e4m3）量化（`quantization_config`、`force_cutlass_fp8`，`vllm_omni/diffusion/data.py:791-800`），并有 HSDP/FSDP2 兼容层（`vllm_omni/diffusion/quantization/hsdp_fp8.py`）。
- **Ascend**：A3（910C）硬件支持 FP8；vllm-ascend 对 LLM 已有 FP8 路径，但 omni 的 NPU 侧 quant 目录仅 `kv_quant_npu.py`（AR KV 量化），**diffusion FP8 未接线**。

### 7.4 Sleep Mode（level 1 / level 2）

- **NVIDIA**：VMM 页回收释放 95%+ 显存（`docs/features/sleep_mode.md`，含 ACK 协议）。
- **Ascend**：官方文档声明 NPU 支持（Ascend memory scavenging）。
- **现状**：vllm-diffusion-origin 自实现 `disagg_sleep/disagg_wake`（逐参数 CPU 搬移，`worker_ext.py`），未使用 omni 的 `DiffusionWorker.sleep(level)` / `enable_sleep_mode`。

### 7.5 性能剖析（enable_diffusion_pipeline_profiler）

- **NVIDIA**：`DiffusionPipelineProfilerMixin` + torch profiler 对 pipeline 各阶段计时。
- **Ascend**：omni 已实现 `NPUTorchProfilerWrapper`（`vllm_omni/platforms/npu/profiler.py`，含 NPU profiler activities），`get_profiler_cls()` 已接入；但框架 CLI 未暴露 `enable_diffusion_pipeline_profiler`。

### 7.6 注意力后端选择

- **NVIDIA**：多后端可选（flash / flash3 / sage / sage3 / flashinfer / cudnn / trtllm / ring，`vllm_omni/diffusion/attention/backends/registry.py`）。
- **Ascend**：omni NPU 默认用 mindiesd 的 FLASH_ATTN，否则回退 SDPA（`vllm_omni/platforms/npu/platform.py:122-155`）；cache-dit 另提供 NPU 优化后端 `_native_npu` 与 ring 并行用 `_npu_fia`（`cache-dit/docs/user_guide/ASCEND_NPU.md`）。
- **现状**：框架未暴露 `diffusion_attention_backend` 选择。

### 7.7 Offload 系列（layerwise / 分布式 layerwise / pin_memory）

- **NVIDIA**：`enable_layerwise_offload`、`enable_distributed_layerwise_offload`（DP 分片 + H2D/AllGather 重叠）、`pin_cpu_memory`。
- **Ascend**：CPU offload 机制平台无关（权重搬 CPU + pinned memory），NPU 同样适用。
- **现状**：框架 CLI 只透传 cpu/layerwise；分离 Worker 默认强制关闭所有 offload（`worker_runtime.py:83-86`）；DLO 与 `pin_cpu_memory` 未暴露。
