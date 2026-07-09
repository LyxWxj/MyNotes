# Awesome-DiT-Inference Papers Reading TODO

## 📙 Sampling

- [x] [DDPM] Denoising Diffusion Probabilistic Models — [Paper](https://arxiv.org/abs/2006.11239) | [Code](https://github.com/hojonathanho/diffusion)
- [x] [DDIM] DENOISING DIFFUSION IMPLICIT MODELS — [Paper](https://arxiv.org/abs/2010.02502) | Code: N/A
- [x] [PNDM] PSEUDO NUMERICAL METHODS FOR DIFFUSION MODELS ON MANIFOLDS — [Paper](https://arxiv.org/abs/2202.09778) | [Code](https://github.com/luping-liu/PNDM)
- [x] [DPM-Solver] DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps — [Paper](https://arxiv.org/abs/2206.00927) | [Code](https://github.com/LuChengTHU/dpm-solver)
- [x] [DPM-Solver++] DPM-SOLVER++: FAST SOLVER FOR GUIDED SAMPLING OF DIFFUSION PROBABILISTIC MODELS — [Paper](https://arxiv.org/abs/2211.01095) | [Code](https://github.com/LuChengTHU/dpm-solver)
- [x] [DPM-Solver-v3] DPM-Solver-v3: Improved Diffusion ODE Solver with Empirical Model Statistics — [Paper](https://arxiv.org/abs/2310.13268) | [Code](https://github.com/thu-ml/DPM-Solver-v3)
- [x] [Parallel Sampling] Parallel Sampling of Diffusion Models — [Paper](https://papers.nips.cc/paper_files/paper/2023/file/0d1986a61e30e5fa408c81216a616e20-Paper-Conference.pdf) | [Code](https://github.com/AndyShih12/paradigms)
- [x] [SAMPLER SCHEDULER] SAMPLER SCHEDULER FOR DIFFUSION MODELS — [Paper](https://arxiv.org/abs/2311.06845) | Code: N/A
- [x] [Parallel Sampling] Accelerating Parallel Sampling of Diffusion Models — [Paper](https://arxiv.org/abs/2402.09970) | [Code](https://github.com/TZW1998/ParaTAA-Diffusion)
- [x] [YONOS] You Only Need One Step: Fast Super-Resolution with Stable Diffusion via Scale Distillation — [Paper](https://arxiv.org/abs/2401.17258) | Code: N/A
- [x] [S^2-DM] S^2-DMs: Skip-Step Diffusion Models — [Paper](https://arxiv.org/abs/2401.01520) | Code: N/A
- [x] [StepSaver] StepSaver: Predicting Minimum Denoising Steps for Diffusion Model Image Generation — [Paper](https://arxiv.org/abs/2408.02054) | Code: N/A
- [x] [DC-Solver] DC-Solver: Improving Predictor-Corrector Diffusion Sampler via Dynamic Compensation — [Paper](https://arxiv.org/abs/2409.03755v1) | [Code](https://github.com/wl-zhao/DC-Solver)

## 📙 Caching

- [ ] [Cache-Enabled Sparse Diffusion] Accelerating Text-to-Image Editing via Cache-Enabled Sparse Diffusion Inference — [Paper](https://arxiv.org/abs/2305.17423) | Code: N/A
- [ ] [DeepCache] DeepCache: Accelerating Diffusion Models for Free — [Paper](https://arxiv.org/abs/2312.00858) | [Code](https://github.com/horseee/DeepCache)
- [ ] [Block Caching] Cache Me if You Can: Accelerating Diffusion Models through Block Caching — [Paper](https://arxiv.org/abs/2312.03209) | Code: N/A
- [ ] [Approximate Caching] Approximate Caching for Efficiently Serving Diffusion Models — [Paper](https://arxiv.org/abs/2312.04429) | Code: N/A
- [ ] [Layer Caching] Learning-to-Cache: Accelerating Diffusion Transformer via Layer Caching — [Paper](https://arxiv.org/abs/2406.01733) | [Code](https://github.com/horseee/learning-to-cache/)
- [ ] [ElasticCache-LVLM] Efficient Inference of Vision Instruction-Following Models with Elastic Cache — [Paper](https://arxiv.org/abs/2407.18121) | [Code](https://github.com/liuzuyan/ElasticCache)
- [ ] [Fast-Forward Caching (FORA)] FORA: Fast-Forward Caching in Diffusion Transformer Acceleration — [Paper](https://arxiv.org/abs/2407.01425) | [Code](https://github.com/prathebaselva/FORA)
- [ ] [Faster I2V Generation] Faster Image2Video Generation: A Closer Look at CLIP Image Embedding's Impact on Spatio-Temporal Cross-Attentions — [Paper](https://arxiv.org/abs/2407.19205) | Code: N/A
- [ ] [T-GATE v1] Cross-Attention Makes Inference Cumbersome in Text-to-Image Diffusion Models — [Paper](https://arxiv.org/abs/2404.02747v1) | [Code](https://github.com/HaozheLiu-ST/T-GATE)
- [ ] [T-GATE v2] Faster Diffusion via Temporal Attention Decomposition — [Paper](https://arxiv.org/abs/2404.02747v2) | [Code](https://github.com/HaozheLiu-ST/T-GATE)
- [ ] [DiTFastAttn] DiTFastAttn: Attention Compression for Diffusion Transformer Models — [Paper](https://arxiv.org/abs/2406.08552) | [Code](https://github.com/thu-nics/DiTFastAttn)
- [ ] [∆-DiT] ∆-DiT: A Training-Free Acceleration Method Tailored for Diffusion Transformers — [Paper](https://arxiv.org/abs/2406.01125) | Code: N/A
- [ ] [TokenCache] Token Caching for Diffusion Transformer Acceleration — [Paper](https://arxiv.org/abs/2409.18523) | Code: N/A
- [ ] [AdaCache] Adaptive Caching for Faster Video Generation with Diffusion Transformers — [Paper](https://adacache-dit.github.io/clarity/adacache_meta.pdf) | [Code](https://github.com/AdaCache-DiT/AdaCache)
- [x] [TeaCache] Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model — [Paper](https://arxiv.org/abs/2411.19108) | [Code](https://github.com/LiewFeng/TeaCache)
- [ ] [LazyDiT] LazyDiT: Lazy Learning for the Acceleration of Diffusion Transformers — [Paper](https://arxiv.org/abs/2412.12444) | Code: N/A
- [ ] [Ca2-VDM] Ca2-VDM: Efficient Autoregressive Video Diffusion Model with Causal Generation and Cache Sharing — [Paper](https://arxiv.org/abs/2411.16375) | [Code](https://github.com/Dawn-LX/CausalCache-VDM/)
- [ ] [SmoothCache] SmoothCache: A Universal Inference Acceleration Technique for Diffusion Transformers — [Paper](https://arxiv.org/abs/2411.10510) | [Code](https://github.com/Roblox/SmoothCache)
- [ ] [FasterCache] FASTERCACHE: TRAINING-FREE VIDEO DIFFUSION MODEL ACCELERATION WITH HIGH QUALITY — [Paper](https://arxiv.org/abs/2410.19355) | [Code](https://github.com/Vchitect/FasterCache)
- [ ] [ToCa] ToCa: Accelerating Diffusion Transformers with Token-wise Feature Caching — [Paper](https://arxiv.org/abs/2410.05317) | [Code](https://github.com/Shenyi-Z/ToCa)
- [ ] [SkipCache] Accelerating Vision Diffusion Transformers with Skip Branches — [Paper](https://arxiv.org/abs/2411.17616) | [Code](https://github.com/OpenSparseLLMs/Skip-DiT)
- [ ] [DuCa] Accelerating Diffusion Transformers with Dual Feature Caching — [Paper](https://arxiv.org/abs/2412.18911) | [Code](https://github.com/Shenyi-Z/DuCa)
- [ ] [FBCache] Fastest HunyuanVideo Inference with Context Parallelism and First Block Cache on NVIDIA L20 GPUs — [Paper](https://github.com/chengzeyi/ParaAttention/blob/main/doc/fastest_hunyuan_video.md) | [Code](https://github.com/chengzeyi/ParaAttention)
- [ ] [FlexCache] FlexCache: Flexible Approximate Cache System for Video Diffusion — [Paper](https://arxiv.org/abs/2501.04012) | Code: N/A
- [ ] [Token Pruning (DaTo)] Token Pruning for Caching Better: 9× Acceleration on Stable Diffusion for Free — [Paper](https://arxiv.org/abs/2501.00375) | [Code](https://github.com/EvelynZhang-epiclab/DaTo)
- [ ] [AB-Cache] AB-Cache: Training-Free Acceleration of Diffusion Models via Adams-Bashforth Cached Feature Reuse — [Paper](https://arxiv.org/abs/2504.10540) | Code: N/A
- [ ] [DiTFastAttnV2] DiTFastAttnV2: Head-wise Attention Compression for Multi-Modality Diffusion Transformers — [Paper](https://arxiv.org/abs/2503.22796) | [Code](https://github.com/thu-nics/DiTFastAttn)
- [ ] [TaylorSeers] From Reusing to Forecasting: Accelerating Diffusion Models with TaylorSeers — [Paper](https://arxiv.org/abs/2503.06923) | [Code](https://github.com/Shenyi-Z/TaylorSeer)
- [ ] [Increment-Calibrated Cache] Accelerating Diffusion Transformer via Increment-Calibrated Caching with Channel-Aware Singular Value Decomposition — [Paper](https://arxiv.org/abs/2505.05829) | [Code](https://github.com/ccccczzy/icc)
- [ ] [FastCache] FastCache: Fast Caching for Diffusion Transformer Through Learnable Linear Approximation — [Paper](https://arxiv.org/abs/2505.20353) | [Code](https://github.com/NoakLiu/FastCache-xDiT)
- [ ] [DBCache] DBCache: Dual Block Caching for Diffusion Transformers — [Paper](https://github.com/vipshop/cache-dit) | [Code](https://github.com/vipshop/cache-dit)
- [ ] [DBPrune] DBPrune: Dynamic Block Prune with Residual Caching — [Paper](https://github.com/vipshop/cache-dit) | [Code](https://github.com/vipshop/cache-dit)
- [ ] [BACache] Block-wise Adaptive Caching for Accelerating Diffusion Policy — [Paper](https://arxiv.org/abs/2506.13456) | Code: N/A

## 📙 Parallelism

- [ ] [DistriFusion] DistriFusion: Distributed Parallel Inference for High-Resolution Diffusion Models — [Paper](https://arxiv.org/abs/2402.19481) | [Code](https://github.com/mit-han-lab/distrifuser)
- [ ] [PipeFusion] PipeFusion: Displaced Patch Pipeline Parallelism for Inference of Diffusion Transformer Models — [Paper](https://arxiv.org/abs/2405.14430) | [Code](https://github.com/xdit-project/xDiT)
- [ ] [AsyncDiff] AsyncDiff: Parallelizing Diffusion Models by Asynchronous Denoising — [Paper](https://arxiv.org/abs/2406.06911) | [Code](https://github.com/czg1225/AsyncDiff)
- [ ] [TensorRT-LLM SDXL] SDXL Distributed Inference with TensorRT-LLM and synchronous comm — [Paper](https://arxiv.org/abs/2402.19481) | [Code](https://github.com/NVIDIA/TensorRT-LLM/pull/1514)
- [ ] [Video-Infinity] Video-Infinity: Distributed Long Video Generation — [Paper](https://arxiv.org/abs/2406.16260) | [Code](https://github.com/Yuanshi9815/Video-Infinity)
- [ ] [FIFO-Diffusion] FIFO-Diffusion: Generating Infinite Videos from Text without Training — [Paper](https://arxiv.org/abs/2405.11473) | [Code](https://github.com/jjihwan/FIFO-Diffusion_public)
- [ ] [ParaAttention] Context parallel attention that accelerates DiT model inference with dynamic caching — [Paper](https://github.com/chengzeyi/ParaAttention) | [Code](https://github.com/chengzeyi/ParaAttention)
- [ ] [Cache-DiT] A PyTorch-native Inference Engine with Hybrid Cache Acceleration and Parallelism for DiTs — [Paper](https://github.com/vipshop/cache-dit) | [Code](https://github.com/vipshop/cache-dit)

## 📙 Quantization

- [ ] [Transfusion] Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model — [Paper](https://www.arxiv.org/abs/2408.11039) | Code: N/A
- [ ] [MixDQ] MixDQ: Memory-Efficient Few-Step Text-to-Image Diffusion Models with Metric-Decoupled Mixed Precision Quantization — [Paper](https://arxiv.org/abs/2405.17873) | [Code](https://github.com/thu-nics/MixDQ)
- [ ] [ViDiT-Q] ViDiT-Q: Efficient and Accurate Quantization of Diffusion Transformers for Image and Video Generation — [Paper](https://arxiv.org/abs/2406.02540) | [Code](https://github.com/thu-nics/ViDiT-Q)
- [ ] [VQ4DiT] VQ4DiT: Efficient Post-Training Vector Quantization for Diffusion Transformers — [Paper](https://arxiv.org/abs/2408.17131) | Code: N/A
- [ ] [LBQ] Low-Bitwidth Floating Point Quantization for Efficient High-Quality Diffusion Models — [Paper](https://arxiv.org/abs/2408.06995) | Code: N/A
- [ ] [EE-Diffusion] A Simple Early Exiting Framework for Accelerated Sampling in Diffusion Models — [Paper](https://arxiv.org/abs/2408.05927) | [Code](https://github.com/taehong-moon/ee-diffusion)
- [ ] [TFM-PTQ] Temporal Feature Matters: A Framework for Diffusion Model Quantization — [Paper](https://arxiv.org/abs/2407.19547) | Code: N/A
- [ ] [Diffusion-RWKV] Diffusion-RWKV: Scaling RWKV-Like Architectures for Diffusion Models — [Paper](https://arxiv.org/abs/2404.04478) | [Code](https://github.com/feizc/Diffusion-RWKV)
- [ ] [LinFusion] LINFUSION: 1 GPU, 1 MINUTE, 16K IMAGE — [Paper](https://arxiv.org/abs/2409.02097) | [Code](https://github.com/Huage001/LinFusion)
- [ ] [SVDQuant] SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models — [Paper](https://arxiv.org/abs/2411.05007) | [Code](https://github.com/mit-han-lab/nunchaku)

## 📙 Attention

- [ ] [SageAttention] SAGEATTENTION: ACCURATE 8-BIT ATTENTION FOR PLUG-AND-PLAY INFERENCE ACCELERATION — [Paper](https://arxiv.org/abs/2410.02367) | [Code](https://github.com/thu-ml/SageAttention)
- [ ] [SageAttention-2] SageAttention2: Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization — [Paper](https://arxiv.org/abs/2411.10958) | [Code](https://github.com/thu-ml/SageAttention)
- [ ] [SpargeAttention] SpargeAttn: Accurate Sparse Attention Accelerating Any Model Inference — [Paper](https://arxiv.org/abs/2502.18137) | [Code](https://github.com/thu-ml/SpargeAttn)
- [ ] [SageAttention-3] SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-bit Training — [Paper](https://arxiv.org/abs/2505.11594) | [Code](https://github.com/thu-ml/SageAttention)
- [ ] [DraftAttention] DraftAttention: Fast Video Diffusion via Low-Resolution Attention Guidance — [Paper](https://arxiv.org/abs/2505.14708) | [Code](https://github.com/shawnricecake/draft-attention)

---


