# Awesome-DiT-Inference Papers Reading TODO

## 📙 Sampling

- [x] [DDPM] Denoising Diffusion Probabilistic Models — [Paper](https://arxiv.org/abs/2006.11239) | [Code](https://github.com/hojonathanho/diffusion)
- [x] [DDIM] DENOISING DIFFUSION IMPLICIT MODELS — [Paper](https://arxiv.org/abs/2010.02502) | Code: N/A
- [x] [PNDM] PSEUDO NUMERICAL METHODS FOR DIFFUSION MODELS ON MANIFOLDS — [Paper](https://arxiv.org/abs/2202.09778) | [Code](https://github.com/luping-liu/PNDM)
- [x] [DPM-Solver] DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps — [Paper](https://arxiv.org/abs/2206.00927) | [Code](https://github.com/LuChengTHU/dpm-solver)
- [x] [DPM-Solver++] DPM-SOLVER++: FAST SOLVER FOR GUIDED SAMPLING OF DIFFUSION PROBABILISTIC MODELS — [Paper](https://arxiv.org/abs/2211.01095) | [Code](https://github.com/LuChengTHU/dpm-solver)
- [x] [DPM-Solver-v3] DPM-Solver-v3: Improved Diffusion ODE Solver with Empirical Model Statistics — [Paper](https://arxiv.org/abs/2310.13268) | [Code](https://github.com/thu-ml/DPM-Solver-v3)
- [ ] [Parallel Sampling] Parallel Sampling of Diffusion Models — [Paper](https://papers.nips.cc/paper_files/paper/2023/file/0d1986a61e30e5fa408c81216a616e20-Paper-Conference.pdf) | [Code](https://github.com/AndyShih12/paradigms)
- [ ] [SAMPLER SCHEDULER] SAMPLER SCHEDULER FOR DIFFUSION MODELS — [Paper](https://arxiv.org/abs/2311.06845) | Code: N/A
- [ ] [Parallel Sampling] Accelerating Parallel Sampling of Diffusion Models — [Paper](https://arxiv.org/abs/2402.09970) | [Code](https://github.com/TZW1998/ParaTAA-Diffusion)
- [ ] [YONOS] You Only Need One Step: Fast Super-Resolution with Stable Diffusion via Scale Distillation — [Paper](https://arxiv.org/abs/2401.17258) | Code: N/A
- [ ] [S^2-DM] S^2-DMs: Skip-Step Diffusion Models — [Paper](https://arxiv.org/abs/2401.01520) | Code: N/A
- [ ] [StepSaver] StepSaver: Predicting Minimum Denoising Steps for Diffusion Model Image Generation — [Paper](https://arxiv.org/abs/2408.02054) | Code: N/A
- [ ] [DC-Solver] DC-Solver: Improving Predictor-Corrector Diffusion Sampler via Dynamic Compensation — [Paper](https://arxiv.org/abs/2409.03755v1) | [Code](https://github.com/wl-zhao/DC-Solver)

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
- [ ] [TeaCache] Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model — [Paper](https://arxiv.org/abs/2411.19108) | [Code](https://github.com/LiewFeng/TeaCache)
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

# Awesome-LLM-Inference Papers Reading TODO

## 📖 Trending LLM/VLM Topics

- [ ] [OneComp] One-Line Revolution for Generative AI Model Compression — [Paper](https://arxiv.org/abs/2603.28845) | [Code](https://github.com/FujitsuResearch/OneCompression)
- [ ] [QEP] Quantization Error Propagation, NeurIPS 2025 — [Paper](https://openreview.net/pdf?id=a3l3K9khbL) | [Code](https://github.com/FujitsuResearch/OneCompression)
- [ ] [Open-Sora] Open-Sora: Democratizing Efficient Video Production for All — [Paper](https://github.com/hpcaitech/Open-Sora/blob/main/docs/zh_CN/README.md) | [Code](https://github.com/hpcaitech/Open-Sora)
- [ ] [Open-Sora Plan] Open-Sora Plan: reproduce Sora (Open AI T2V model) — [Paper](https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/main/docs/Report-v1.0.0.md) | [Code](https://github.com/PKU-YuanGroup/Open-Sora-Plan)
- [ ] [DeepSeek-V2] DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model — [Paper](https://arxiv.org/abs/2405.04434) | [Code](https://github.com/deepseek-ai/DeepSeek-V2)
- [ ] [YOCO] You Only Cache Once: Decoder-Decoder Architectures for Language Models — [Paper](https://arxiv.org/abs/2405.05254) | [Code](https://github.com/microsoft/unilm/tree/master/YOCO)
- [ ] [Mooncake] Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving — [Paper](https://github.com/kvcache-ai/Mooncake/blob/main/Mooncake-v3.pdf) | [Code](https://github.com/kvcache-ai/Mooncake)
- [ ] [FlashAttention-3] FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision — [Paper](https://tridao.me/publications/flash3/flash3.pdf) | [Code](https://github.com/Dao-AILab/flash-attention)
- [ ] [MInference 1.0] MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention — [Paper](https://arxiv.org/abs/2407.02490) | [Code](https://github.com/microsoft/MInference)
- [ ] [Star-Attention] Star Attention: Efficient LLM Inference over Long Sequences — [Paper](https://arxiv.org/abs/2411.17116) | [Code](https://github.com/NVIDIA/Star-Attention)
- [ ] [DeepSeek-V3] DeepSeek-V3 Technical Report — [Paper](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf) | [Code](https://github.com/deepseek-ai/DeepSeek-V3)
- [ ] [MiniMax-Text-01] MiniMax-01: Scaling Foundation Models with Lightning Attention — [Paper](https://filecdn.minimax.chat/_Arxiv_MiniMax_01_Report.pdf) | [Code](https://github.com/MiniMax-AI/MiniMax-01)
- [ ] [DeepSeek-R1] DeepSeek-R1 Technical Report — [Paper](https://arxiv.org/abs/2501.12948v1) | [Code](https://github.com/deepseek-ai/DeepSeek-R1)

## 📖 DeepSeek / Multi-head Latent Attention (MLA)

- [ ] [DeepSeek-NSA] Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention — [Paper](https://arxiv.org/abs/2502.11089) | Code: N/A
- [ ] [FlashMLA] DeepSeek FlashMLA — [Paper](https://github.com/deepseek-ai/FlashMLA) | [Code](https://github.com/deepseek-ai/FlashMLA)
- [ ] [DualPipe] DeepSeek DualPipe — [Paper](https://github.com/deepseek-ai/DualPipe) | [Code](https://github.com/deepseek-ai/DualPipe)
- [ ] [DeepEP] DeepSeek DeepEP — [Paper](https://github.com/deepseek-ai/DeepEP) | [Code](https://github.com/deepseek-ai/DeepEP)
- [ ] [DeepGEMM] DeepSeek DeepGEMM — [Paper](https://github.com/deepseek-ai/DeepGEMM) | [Code](https://github.com/deepseek-ai/DeepGEMM)
- [ ] [EPLB] DeepSeek EPLB — [Paper](https://github.com/deepseek-ai/EPLB) | [Code](https://github.com/deepseek-ai/EPLB)
- [ ] [3FS] DeepSeek 3FS — [Paper](https://github.com/deepseek-ai/3FS) | [Code](https://github.com/deepseek-ai/3FS)
- [ ] [推理系统] DeepSeek-V3 / R1 推理系统概览 — [Paper](https://zhuanlan.zhihu.com/p/27181462601) | Code: N/A
- [ ] [MHA2MLA] Towards Economical Inference: Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs — [Paper](https://arxiv.org/abs/2502.14837) | [Code](https://github.com/JT-Ushio/MHA2MLA)
- [ ] [TransMLA] TransMLA: Multi-head Latent Attention Is All You Need — [Paper](https://arxiv.org/abs/2502.07864) | [Code](https://github.com/fxmeng/TransMLA)
- [ ] [X-EcoMLA] X-EcoMLA: Upcycling Pre-Trained Attention into MLA for Efficient and Extreme KV Compression — [Paper](https://arxiv.org/abs/2503.11132) | Code: N/A

## 📖 Multi-GPUs / Multi-Nodes Parallelism

- [ ] [ZeRO] DeepSpeed-ZeRO: Memory Optimizations Toward Training Trillion Parameter Models — [Paper](https://arxiv.org/abs/1910.02054) | [Code](https://github.com/microsoft/DeepSpeed)
- [ ] [Megatron-LM (TP)] Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism — [Paper](https://arxiv.org/abs/1909.08053.pdf) | [Code](https://github.com/NVIDIA/Megatron-LM)
- [ ] [Megatron-LM (SP)] Megatron-LM: Reducing Activation Recomputation in Large Transformer Models — [Paper](https://arxiv.org/abs/2205.05198) | [Code](https://github.com/NVIDIA/Megatron-LM)
- [ ] [BPT] Blockwise Parallel Transformer for Large Context Models — [Paper](https://arxiv.org/abs/2305.19370) | [Code](https://github.com/lhao499/RingAttention)
- [ ] [Ring Attention] Ring Attention with Blockwise Transformers for Near-Infinite Context — [Paper](https://arxiv.org/abs/2310.01889.pdf) | [Code](https://github.com/lhao499/RingAttention)
- [ ] [Striped Attention] STRIPED ATTENTION: FASTER RING ATTENTION FOR CAUSAL TRANSFORMERS — [Paper](https://arxiv.org/abs/2311.09431.pdf) | [Code](https://github.com/exists-forall/striped_attention/)
- [ ] [DeepSpeed Ulysses] DEEPSPEED ULYSSES: SYSTEM OPTIMIZATIONS FOR ENABLING TRAINING OF EXTREME LONG SEQUENCE TRANSFORMER MODELS — [Paper](https://arxiv.org/abs/2309.14509) | [Code](https://github.com/microsoft/DeepSpeed)
- [ ] [Megatron-LM CP] Megatron-LM: Context parallelism overview — [Paper](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/context_parallel.html) | [Code](https://github.com/NVIDIA/Megatron-LM)
- [ ] [USP] Unified Sequence Parallel (USP) Attention for Long Context LLM Model Training and Inference — [Paper]() | [Code](https://github.com/feifeibear/long-context-attention)
- [ ] [Meta CP] Context Parallelism for Scalable Million-Token Inference — [Paper](https://arxiv.org/abs/2411.01783) | Code: N/A
- [ ] [Comm Compression] Communication Compression for Tensor Parallel LLM Inference — [Paper](https://arxiv.org/abs/2411.09510) | Code: N/A
- [ ] [TokenRing] TokenRing: An Efficient Parallelism Framework for Infinite-Context LLMs via Bidirectional Communication — [Paper](https://arxiv.org/abs/2412.20501) | [Code](https://github.com/ACA-Lab-SJTU/token-ring)
- [ ] [FSDP] PyTorch FSDP: Fully Sharded Data Parallel — [Paper](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) | Code: N/A

## 📖 Disaggregating Prefill and Decoding

- [ ] [DistServe] DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving — [Paper](https://arxiv.org/abs/2401.09670) | [Code](https://github.com/LLMServe/DistServe)
- [ ] [KVDirect] KVDirect: Distributed Disaggregated LLM Inference — [Paper](https://arxiv.org/abs/2501.14743) | Code: N/A
- [ ] [DeServe] DESERVE: TOWARDS AFFORDABLE OFFLINE LLM INFERENCE VIA DECENTRALIZATION — [Paper](https://arxiv.org/abs/2501.14784) | Code: N/A
- [ ] [MegaScale-Infer] MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism — [Paper](https://arxiv.org/abs/2504.02263) | Code: N/A

## 📖 LLM Algorithmic / Eval Survey

- [ ] [Evaluating LLMs] Evaluating Large Language Models: A Comprehensive Survey — [Paper](https://arxiv.org/abs/2310.19736.pdf) | [Code](https://github.com/tjunlp-lab/Awesome-LLMs-Evaluation-Papers)
- [ ] [Runtime Performance] Dissecting the Runtime Performance of the Training, Fine-tuning, and Inference of LLMs — [Paper](https://arxiv.org/abs/2311.03687.pdf) | Code: N/A
- [ ] [ChatGPT Anniversary] ChatGPT's One-year Anniversary: Are Open-Source LLMs Catching up? — [Paper](https://arxiv.org/abs/2311.16989.pdf) | Code: N/A
- [ ] [Algorithmic Survey] The Efficiency Spectrum of LLMs: An Algorithmic Survey — [Paper](https://arxiv.org/abs/2312.00678.pdf) | Code: N/A
- [ ] [Security and Privacy] A Survey on LLM Security and Privacy: The Good, the Bad, and the Ugly — [Paper](https://arxiv.org/abs/2312.02003.pdf) | Code: N/A
- [ ] [LLMCompass] A Hardware Evaluation Framework for LLM Inference — [Paper](https://arxiv.org/abs/2312.03134.pdf) | Code: N/A
- [ ] [Efficient LLMs Survey] Efficient Large Language Models: A Survey — [Paper](https://arxiv.org/abs/2312.03863.pdf) | [Code](https://github.com/AIoT-MLSys-Lab/Efficient-LLMs-Survey)
- [ ] [Serving Survey] Towards Efficient Generative LLM Serving: A Survey from Algorithms to Systems — [Paper](https://arxiv.org/abs/2312.15234.pdf) | Code: N/A
- [ ] [Understanding LLMs] Understanding LLMs: A Comprehensive Overview from Training to Inference — [Paper](https://arxiv.org/abs/2401.02038.pdf) | Code: N/A
- [ ] [LLM-Viewer] LLM Inference Unveiled: Survey and Roofline Model Insights — [Paper](https://arxiv.org/abs/2402.16363.pdf) | [Code](https://github.com/hahnyuan/LLM-Viewer)
- [ ] [Internal Consistency Survey] Internal Consistency and Self-Feedback in LLMs: A Survey — [Paper](https://arxiv.org/abs/2407.14507) | [Code](https://github.com/IAAR-Shanghai/ICSFSurvey)
- [ ] [Low-bit Survey] A Survey of Low-bit Large Language Models: Basics, Systems, and Algorithms — [Paper](https://arxiv.org/abs/2409.16694) | Code: N/A
- [ ] [LLM Inference Survey (Hardware)] LARGE LANGUAGE MODEL INFERENCE ACCELERATION: A COMPREHENSIVE HARDWARE PERSPECTIVE — [Paper](https://arxiv.org/abs/2410.04466) | Code: N/A

## 📖 LLM Train/Inference Framework/Design

- [ ] [FlexGen] High-Throughput Generative Inference of LLMs with a Single GPU — [Paper](https://arxiv.org/abs/2303.06865.pdf) | [Code](https://github.com/FMInference/FlexGen)
- [ ] [SpecInfer] Accelerating Generative LLM Serving with Speculative Inference and Token Tree Verification — [Paper](https://arxiv.org/abs/2305.09781.pdf) | [Code](https://github.com/flexflow/FlexFlow)
- [ ] [FastServe] Fast Distributed Inference Serving for Large Language Models — [Paper](https://arxiv.org/abs/2305.05920.pdf) | Code: N/A
- [ ] [vLLM / PagedAttention] Efficient Memory Management for LLM Serving with PagedAttention — [Paper](https://arxiv.org/abs/2309.06180.pdf) | [Code](https://github.com/vllm-project/vllm)
- [ ] [StreamingLLM] EFFICIENT STREAMING LANGUAGE MODELS WITH ATTENTION SINKS — [Paper](https://arxiv.org/abs/2309.17453.pdf) | [Code](https://github.com/mit-han-lab/streaming-llm)
- [ ] [Medusa] Medusa: Simple Framework for Accelerating LLM Generation with Multiple Decoding Heads — [Paper](https://sites.google.com/view/medusa-llm) | [Code](https://github.com/FasterDecoding/Medusa)
- [ ] [TensorRT-LLM] NVIDIA TensorRT LLM — [Paper](https://nvidia.github.io/TensorRT-LLM/) | [Code](https://github.com/NVIDIA/TensorRT-LLM)
- [ ] [DeepSpeed-FastGen] DeepSpeed-FastGen: High-throughput Text Generation for LLMs — [Paper](https://arxiv.org/abs/2401.08671.pdf) | [Code](https://github.com/microsoft/DeepSpeed)
- [ ] [SGLang] Efficiently Programming Large Language Models using SGLang — [Paper](https://arxiv.org/abs/2312.07104) | [Code](https://github.com/sgl-project/sglang)
- [ ] [PETALS] Distributed Inference and Fine-tuning of LLMs Over The Internet — [Paper](https://arxiv.org/abs/2312.08361.pdf) | [Code](https://github.com/bigscience-workshop/petals)
- [ ] [LightSeq] LightSeq: Sequence Level Parallelism for Distributed Training of Long Context Transformers — [Paper](https://arxiv.org/abs/2310.03294.pdf) | [Code](https://github.com/RulinShao/LightSeq)
- [ ] [PowerInfer] PowerInfer: Fast LLM Serving with a Consumer-grade GPU — [Paper](https://ipads.se.sjtu.edu.cn/_media/publications/powerinfer-20231219.pdf) | [Code](https://github.com/SJTU-IPADS/PowerInfer)
- [ ] [inferflow] INFERFLOW: AN EFFICIENT AND HIGHLY CONFIGURABLE INFERENCE ENGINE FOR LLMs — [Paper](https://arxiv.org/abs/2401.08294.pdf) | [Code](https://github.com/inferflow/inferflow)
- [ ] [LMDeploy] LMDeploy: toolkit for compressing, deploying, and serving LLMs — [Paper](https://lmdeploy.readthedocs.io/en/latest/) | [Code](https://github.com/InternLM/lmdeploy)
- [ ] [MLC-LLM] Universal LLM Deployment Engine with ML Compilation — [Paper](https://llm.mlc.ai/) | [Code](https://github.com/mlc-ai/mlc-llm)
- [ ] [LightLLM] LightLLM: Python-based LLM inference and serving framework — [Paper](https://github.com/ModelTC/lightllm) | [Code](https://github.com/ModelTC/lightllm)
- [ ] [llama.cpp] Inference of LLaMA model (and others) in pure C/C++ — [Paper](https://github.com/ggerganov/llama.cpp) | [Code](https://github.com/ggerganov/llama.cpp)
- [ ] [flashinfer] FlashInfer: Kernel Library for LLM Serving — [Paper](https://flashinfer.ai/2024/02/02/cascade-inference.html) | [Code](https://github.com/flashinfer-ai/flashinfer)
- [ ] [DynamoLLM] DynamoLLM: Designing LLM Inference Clusters for Performance and Energy Efficiency — [Paper](https://arxiv.org/abs/2408.00741) | Code: N/A
- [ ] [NanoFlow] NanoFlow: Towards Optimal LLM Serving Throughput — [Paper](https://arxiv.org/abs/2408.12757) | [Code](https://github.com/efeslab/Nanoflow)
- [ ] [Decentralized LLM] Decentralized LLM Inference over Edge Networks with Energy Harvesting — [Paper](https://arxiv.org/abs/2408.15907) | Code: N/A
- [ ] [SparseInfer] SparseInfer: Training-free Prediction of Activation Sparsity for Fast LLM Inference — [Paper](https://arxiv.org/abs/2411.12692) | Code: N/A
- [ ] [prima.cpp] PRIMA.CPP: Speeding Up 70B-Scale LLM Inference on Low-Resource Everyday Home Clusters — [Paper](https://arxiv.org/abs/2504.08791) | [Code](https://github.com/Lizonghang/prima.cpp)
- [ ] [DistFlow / siiRL] DistFlow: A Fully Distributed RL Framework for Scalable and Efficient LLM Post-Training — [Paper](https://arxiv.org/abs/2507.13833) | [Code](https://github.com/sii-research/siiRL)
- [ ] [ToolPipe] ToolPipe: 120+ Free Developer Tools REST API & MCP Server for AI Agents — [Paper](https://toolpipe.dev) | [Code](https://github.com/COSAI-Labs/toolpipe-mcp-server)

## 📖 Continuous / In-flight Batching

- [ ] [Orca / Continuous Batching] Orca: A Distributed Serving System for Transformer-Based Generative Models — [Paper](https://www.usenix.org/system/files/osdi22-yu.pdf) | Code: N/A
- [ ] [In-flight Batching] NVIDIA TensorRT LLM Batch Manager — [Paper](https://nvidia.github.io/TensorRT-LLM/batch_manager.html) | [Code](https://github.com/NVIDIA/TensorRT-LLM)
- [ ] [Splitwise] Splitwise: Efficient Generative LLM Inference Using Phase Splitting — [Paper](https://arxiv.org/abs/2311.18677.pdf) | Code: N/A
- [ ] [SpotServe] SpotServe: Serving Generative LLMs on Preemptible Instances — [Paper](https://arxiv.org/abs/2311.15566.pdf) | [Code](https://github.com/Hsword/SpotServe)
- [ ] [vAttention] vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention — [Paper](https://arxiv.org/abs/2405.04437) | [Code](https://github.com/microsoft/vattention)
- [ ] [vTensor] vTensor: Flexible Virtual Tensor Management for Efficient LLM Serving — [Paper](https://arxiv.org/abs/2407.15309) | [Code](https://github.com/intelligent-machine-learning/glake/tree/master/GLakeServe)
- [ ] [Auto Engine Tuning] Towards SLO-Optimized LLM Serving via Automatic Inference Engine Tuning — [Paper](https://arxiv.org/abs/2408.04323) | Code: N/A
- [ ] [SJF Scheduling] Efficient LLM Scheduling by Learning to Rank — [Paper](https://arxiv.org/abs/2408.15792) | Code: N/A
- [ ] [BatchLLM] BatchLLM: Optimizing Large Batched LLM Inference with Global Prefix Sharing and Throughput-oriented Token Batching — [Paper](https://arxiv.org/abs/2412.03594) | Code: N/A

## 📖 Weight/Activation Quantize/Compress

- [ ] [ZeroQuant] Efficient and Affordable Post-Training Quantization for Large-Scale Transformers — [Paper](https://arxiv.org/abs/2206.01861.pdf) | [Code](https://github.com/microsoft/DeepSpeed)
- [ ] [FP8 Quantization] FP8 Quantization: The Power of the Exponent — [Paper](https://arxiv.org/abs/2208.09225.pdf) | [Code](https://github.com/Qualcomm-AI-research/FP8-quantization)
- [ ] [LLM.int8()] 8-bit Matrix Multiplication for Transformers at Scale — [Paper](https://arxiv.org/abs/2208.07339.pdf) | [Code](https://github.com/timdettmers/bitsandbytes)
- [ ] [GPTQ] GPTQ: ACCURATE POST-TRAINING QUANTIZATION FOR GENERATIVE PRE-TRAINED TRANSFORMERS — [Paper](https://arxiv.org/abs/2210.17323.pdf) | [Code](https://github.com/IST-DASLab/gptq)
- [ ] [WINT8/4] Who Says Elephants Can't Run: Bringing Large Scale MoE Models into Cloud Scale Production — [Paper](https://arxiv.org/abs/2211.10017.pdf) | [Code](https://github.com/NVIDIA/FasterTransformer)
- [ ] [SmoothQuant] Accurate and Efficient Post-Training Quantization for LLMs — [Paper](https://arxiv.org/abs/2211.10438.pdf) | [Code](https://github.com/mit-han-lab/smoothquant)
- [ ] [ZeroQuant-V2] Exploring Post-training Quantization in LLMs from Comprehensive Study to Low Rank Compensation — [Paper](https://arxiv.org/abs/2303.08302.pdf) | [Code](https://github.com/microsoft/DeepSpeed)
- [ ] [AWQ] AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration — [Paper](https://browse.arxiv.org/abs/2306.00978.pdf) | [Code](https://github.com/mit-han-lab/llm-awq)
- [ ] [SpQR] SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression — [Paper](https://browse.arxiv.org/abs/2306.03078.pdf) | [Code](https://github.com/Vahe1994/SpQR)
- [ ] [SqueezeLLM] SQUEEZELLM: DENSE-AND-SPARSE QUANTIZATION — [Paper](https://arxiv.org/abs/2306.07629.pdf) | [Code](https://github.com/SqueezeAILab/SqueezeLLM)
- [ ] [ZeroQuant-FP] A Leap Forward in LLMs Post-Training W4A8 Quantization Using Floating-Point Formats — [Paper](https://arxiv.org/abs/2307.09782.pdf) | [Code](https://github.com/microsoft/DeepSpeed)
- [ ] [FP8-LM] FP8-LM: Training FP8 Large Language Models — [Paper](https://arxiv.org/abs/2310.18313.pdf) | [Code](https://github.com/Azure/MS-AMP)
- [ ] [LLM-Shearing] SHEARED LLAMA: ACCELERATING LANGUAGE MODEL PRE-TRAINING VIA STRUCTURED PRUNING — [Paper](https://arxiv.org/abs/2310.06694.pdf) | [Code](https://github.com/princeton-nlp/LLM-Shearing)
- [ ] [LLM-FP4] LLM-FP4: 4-Bit Floating-Point Quantized Transformers — [Paper](https://arxiv.org/abs/2310.16836.pdf) | [Code](https://github.com/nbasyl/LLM-FP4)
- [ ] [2-bit LLM] Enabling Fast 2-bit LLM on GPUs: Memory Alignment, Sparse Outlier, and Asynchronous Dequantization — [Paper](https://arxiv.org/abs/2311.16442.pdf) | Code: N/A
- [ ] [SmoothQuant+] SmoothQuant+: Accurate and Efficient 4-bit Post-Training Weight Quantization for LLM — [Paper](https://arxiv.org/abs/2312.03788.pdf) | [Code](https://github.com/Adlik/smoothquantplus)
- [ ] [OdysseyLLM W4A8] A Speed Odyssey for Deployable Quantization of LLMs — [Paper](https://arxiv.org/abs/2311.09550.pdf) | Code: N/A
- [ ] [SparQ Attention] SPARQ ATTENTION: BANDWIDTH-EFFICIENT LLM INFERENCE — [Paper](https://arxiv.org/abs/2312.04985.pdf) | Code: N/A
- [ ] [Agile-Quant] Agile-Quant: Activation-Guided Quantization for Faster Inference of LLMs on the Edge — [Paper](https://arxiv.org/abs/2312.05693.pdf) | Code: N/A
- [ ] [CBQ] CBQ: Cross-Block Quantization for Large Language Models — [Paper](https://arxiv.org/abs/2312.07950.pdf) | Code: N/A
- [ ] [QLLM] QLLM: ACCURATE AND EFFICIENT LOW-BITWIDTH QUANTIZATION FOR LLMs — [Paper](https://arxiv.org/abs/2310.08041.pdf) | Code: N/A
- [ ] [FP6-LLM] FP6-LLM: Efficiently Serving LLMs Through FP6-Centric Algorithm-System Co-Design — [Paper](https://arxiv.org/abs/2401.14112.pdf) | Code: N/A
- [ ] [QServe] QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving — [Paper](https://arxiv.org/abs/2405.04532) | [Code](https://github.com/mit-han-lab/qserve)
- [ ] [SpinQuant] SpinQuant: LLM Quantization with Learned Rotations — [Paper](https://arxiv.org/abs/2405.16406) | Code: N/A
- [ ] [I-LLM] I-LLM: Efficient Integer-Only Inference for Fully-Quantized Low-Bit LLMs — [Paper](https://arxiv.org/abs/2405.17849) | Code: N/A
- [ ] [OutlierTune] OutlierTune: Efficient Channel-Wise Quantization for LLMs — [Paper](https://arxiv.org/abs/2406.18832) | Code: N/A
- [ ] [GPTQT] GPTQT: Quantize Large Language Models Twice to Push the Efficiency — [Paper](https://arxiv.org/abs/2407.02891) | Code: N/A
- [ ] [ABQ-LLM] ABQ-LLM: Arbitrary-Bit Quantized Inference Acceleration for LLMs — [Paper](https://arxiv.org/abs/2408.08554) | [Code](https://github.com/bytedance/ABQ-LLM)
- [ ] [1-bit LLMs] Matmul or No Matmal in the Era of 1-bit LLMs — [Paper](https://arxiv.org/abs/2408.11939) | Code: N/A
- [ ] [Activation Sparsity / TEAL] TRAINING-FREE ACTIVATION SPARSITY IN LLMs — [Paper](https://arxiv.org/abs/2408.14690) | [Code](https://github.com/FasterDecoding/TEAL)
- [ ] [VPTQ] VPTQ: EXTREME LOW-BIT VECTOR POST-TRAINING QUANTIZATION FOR LLMs — [Paper](https://arxiv.org/abs/2409.17066) | [Code](https://github.com/microsoft/VPTQ)
- [ ] [BitNet a4.8] BitNet a4.8: 4-bit Activations for 1-bit LLMs — [Paper](https://arxiv.org/abs/2411.04965) | [Code](https://github.com/microsoft/unilm/tree/master/bitnet)
- [ ] [BitNet v2] BitNet v2: Native 4-bit Activations with Hadamard Transformation for 1-bit LLMs — [Paper](https://arxiv.org/abs/2504.18415) | [Code](https://github.com/microsoft/unilm/tree/master/bitnet)
- [ ] [GuidedQuant] GuidedQuant: LLM Quantization via Exploiting End Loss Guidance — [Paper](https://arxiv.org/abs/2505.07004) | [Code](https://github.com/snu-mllab/GuidedQuant)

## 📖 IO/FLOPs-Aware / Sparse Attention

- [ ] [Online Softmax] Online normalizer calculation for softmax — [Paper](https://arxiv.org/abs/1805.02867.pdf) | Code: N/A
- [ ] [MQA] Fast Transformer Decoding: One Write-Head is All You Need — [Paper](https://arxiv.org/abs/1911.02150.pdf) | Code: N/A
- [ ] [Reformer / Hash Attention] REFORMER: THE EFFICIENT TRANSFORMER — [Paper](https://arxiv.org/abs/2001.04451.pdf) | [Code](https://github.com/google/trax)
- [ ] [FlashAttention] Fast and Memory-Efficient Exact Attention with IO-Awareness — [Paper](https://arxiv.org/abs/2205.14135.pdf) | [Code](https://github.com/Dao-AILab/flash-attention)
- [ ] [GQA] GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints — [Paper](https://arxiv.org/abs/2305.13245.pdf) | [Code](https://github.com/google/flaxformer)
- [ ] [Sparse FlashAttention] Faster Causal Attention Over Large Sequences Through Sparse Flash Attention — [Paper](https://arxiv.org/abs/2306.01160.pdf) | [Code](https://github.com/epfml/dynamic-sparse-flash-attention)
- [ ] [FlashAttention-2] Faster Attention with Better Parallelism and Work Partitioning — [Paper](https://arxiv.org/abs/2307.08691.pdf) | [Code](https://github.com/Dao-AILab/flash-attention)
- [ ] [Flash-Decoding] Flash-Decoding for long-context inference — [Paper](https://crfm.stanford.edu/2023/10/12/flashdecoding.html) | [Code](https://github.com/Dao-AILab/flash-attention)
- [ ] [FlashDecoding++] FLASHDECODING++: FASTER LARGE LANGUAGE MODEL INFERENCE ON GPUS — [Paper](https://arxiv.org/abs/2311.01282.pdf) | Code: N/A
- [ ] [SparseGPT] SparseGPT: Massive Language Models Can be Accurately Pruned in One-Shot — [Paper](https://arxiv.org/abs/2301.00774.pdf) | [Code](https://github.com/IST-DASLab/sparsegpt)
- [ ] [GLA] Gated Linear Attention Transformers with Hardware-Efficient Training — [Paper](https://arxiv.org/abs/2312.06635.pdf) | [Code](https://github.com/berlino/gated_linear_attention)
- [ ] [SCCA] SCCA: Shifted Cross Chunk Attention — [Paper](https://arxiv.org/abs/2312.07305.pdf) | Code: N/A
- [ ] [FlashLLM] LLM in a flash: Efficient LLM Inference with Limited Memory — [Paper](https://arxiv.org/abs/2312.11514.pdf) | Code: N/A
- [ ] [CHAI] CHAI: Clustered Head Attention for Efficient LLM Inference — [Paper](https://arxiv.org/abs/2403.08058.pdf) | Code: N/A
- [ ] [DeFT] DeFT: Decoding with Flash Tree-Attention for Efficient Tree-structured LLM Inference — [Paper](https://arxiv.org/abs/2404.00242) | Code: N/A
- [ ] [MoA] MoA: Mixture of Sparse Attention for Automatic LLM Compression — [Paper](https://arxiv.org/abs/2406.14909) | [Code](https://github.com/thu-nics/MoA)
- [ ] [Shared Attention] Beyond KV Caching: Shared Attention for Efficient LLMs — [Paper](https://arxiv.org/abs/2407.12866) | [Code](https://github.com/metacarbon/shareAtt)
- [ ] [CHESS] CHESS: Optimizing LLM Inference via Channel-Wise Thresholding and Selective Sparsification — [Paper](https://arxiv.org/abs/2409.01366) | Code: N/A
- [ ] [INT-FlashAttention] INT-FLASHATTENTION: ENABLING FLASH ATTENTION FOR INT8 QUANTIZATION — [Paper](https://arxiv.org/abs/2409.16997) | [Code](https://github.com/INT-FlashAttention2024/INT-FlashAttention)
- [ ] [SageAttention] SAGEATTENTION: ACCURATE 8-BIT ATTENTION FOR PLUG-AND-PLAY INFERENCE ACCELERATION — [Paper](https://arxiv.org/abs/2410.02367) | [Code](https://github.com/thu-ml/SageAttention)
- [ ] [SageAttention-2] SageAttention2: Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization — [Paper](https://arxiv.org/abs/2411.10958) | [Code](https://github.com/thu-ml/SageAttention)
- [ ] [Squeezed Attention] SQUEEZED ATTENTION: Accelerating Long Context Length LLM Inference — [Paper](https://arxiv.org/abs/2411.09688) | [Code](https://github.com/SqueezeAILab/SqueezedAttention)
- [ ] [TurboAttention] TURBOATTENTION: EFFICIENT ATTENTION APPROXIMATION FOR HIGH THROUGHPUTS LLMS — [Paper](https://arxiv.org/abs/2412.08585) | Code: N/A
- [ ] [FFPA] FFPA: Faster Flash Prefill Attention with O(1) SRAM complexity for headdim > 256 — [Paper](https://github.com/xlite-dev/ffpa-attn) | [Code](https://github.com/xlite-dev/ffpa-attn)
- [ ] [SpargeAttention] SpargeAttn: Accurate Sparse Attention Accelerating Any Model Inference — [Paper](https://arxiv.org/abs/2502.18137) | [Code](https://github.com/thu-ml/SpargeAttn)
- [ ] [MMInference] MMInference: Accelerating Pre-filling for Long-Context VLMs via Modality-Aware Permutation Sparse Attention — [Paper](https://arxiv.org/abs/2504.16083) | [Code](https://github.com/microsoft/MInference/)
- [ ] [Sparse Frontier] The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs — [Paper](https://arxiv.org/abs/2504.17768) | [Code](https://github.com/PiotrNawrot/sparse-frontier)
- [ ] [Flex Attention] FLEX ATTENTION: A PROGRAMMING MODEL FOR GENERATING OPTIMIZED ATTENTION KERNELS — [Paper](https://arxiv.org/abs/2412.05496) | [Code](https://github.com/pytorch-labs/attention-gym)
- [ ] [SeerAttention] SeerAttention: Learning Intrinsic Sparse Attention in Your LLMs — [Paper](https://arxiv.org/abs/2410.13276) | [Code](https://github.com/microsoft/SeerAttention)
- [ ] [Slim Attention] Slim attention: cut your context memory in half without loss of accuracy — [Paper](https://arxiv.org/abs/2503.05840) | [Code](https://github.com/OpenMachine-ai/transformer-tricks)
- [ ] [SageAttention-3] SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-bit Training — [Paper](https://arxiv.org/abs/2505.11594) | [Code](https://github.com/thu-ml/SageAttention)
- [ ] [APE] APE: Faster and Longer Context-Augmented Generation via Adaptive Parallel Encoding — [Paper](https://arxiv.org/abs/2502.05431) | [Code](https://github.com/Infini-AI-Lab/APE)
- [ ] [Block-Attention] Block-Attention for Efficient Prefilling — [Paper](https://arxiv.org/abs/2409.15355) | [Code](https://github.com/TemporaryLoRA/Block-attention)

## 📖 KV Cache Scheduling/Quantize/Dropping

- [ ] [NexusQuant] NexusQuant: Training-Free KV Cache Compression via E8 Lattice Quantization — [Paper](https://github.com/nexusquant/nexusquant) | [Code](https://github.com/nexusquant/nexusquant)
- [ ] [LTP] Learned Token Pruning for Transformers — [Paper](https://arxiv.org/abs/2107.00910.pdf) | [Code](https://github.com/kssteven418/LTP)
- [ ] [Scissorhands] Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression — [Paper](https://arxiv.org/abs/2305.17118.pdf) | Code: N/A
- [ ] [H2O] H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs — [Paper](https://arxiv.org/abs/2306.14048.pdf) | [Code](https://github.com/FMInference/H2O)
- [ ] [SARATHI / Chunked Prefills] SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills — [Paper](https://arxiv.org/abs/2308.16369.pdf) | Code: N/A
- [ ] [TensorRT-LLM KV Cache FP8] NVIDIA TensorRT LLM — [Paper](https://nvidia.github.io/TensorRT-LLM/precision.html) | [Code](https://github.com/NVIDIA/TensorRT-LLM)
- [ ] [Adaptive KV Cache Compress] MODEL TELLS YOU WHAT TO DISCARD: ADAPTIVE KV CACHE COMPRESSION FOR LLMS — [Paper](https://arxiv.org/abs/2310.01801.pdf) | Code: N/A
- [ ] [CacheGen] CacheGen: Fast Context Loading for Language Model Applications — [Paper](https://arxiv.org/abs/2310.07240.pdf) | [Code](https://github.com/LMCache/LMCache)
- [ ] [KV Cache Compress with LoRA] Compressed Context Memory for Online Language Model Interaction — [Paper](https://arxiv.org/abs/2312.03414.pdf) | [Code](https://github.com/snu-mllab/Context-Memory)
- [ ] [DistKV-LLM / Infinite-LLM] Infinite-LLM: Efficient LLM Service for Long Context with DistAttention and Distributed KVCache — [Paper](https://arxiv.org/abs/2401.02669.pdf) | Code: N/A
- [ ] [Prompt Caching] Efficient Prompt Caching via Embedding Similarity — [Paper](https://arxiv.org/abs/2402.01173.pdf) | Code: N/A
- [ ] [Less] Get More with LESS: Synthesizing Recurrence with KV Cache Compression for Efficient LLM Inference — [Paper](https://arxiv.org/abs/2402.09398.pdf) | Code: N/A
- [ ] [MiKV] No Token Left Behind: Reliable KV Cache Compression via Importance-Aware Mixed Precision Quantization — [Paper](https://arxiv.org/abs/2402.18096.pdf) | Code: N/A
- [ ] [Hydragen] Hydragen: High-Throughput LLM Inference with Shared Prefixes — [Paper](https://arxiv.org/abs/2402.05099.pdf) | Code: N/A
- [ ] [ChunkAttention] ChunkAttention: Efficient Self-Attention with Prefix-Aware KV Cache and Two-Phase Partition — [Paper](https://arxiv.org/abs/2402.15220) | [Code](https://github.com/microsoft/chunk-attention)
- [ ] [QAQ] QAQ: Quality Adaptive Quantization for LLM KV Cache — [Paper](https://arxiv.org/abs/2403.04643.pdf) | [Code](https://github.com/ClubieDong/QAQ-KVCacheQuantization)
- [ ] [DMC] Dynamic Memory Compression: Retrofitting LLMs for Accelerated Inference — [Paper](https://arxiv.org/abs/2403.09636.pdf) | Code: N/A
- [ ] [Keyformer] Keyformer: KV Cache reduction through key tokens selection for Efficient Generative Inference — [Paper](https://arxiv.org/abs/2403.09054.pdf) | [Code](https://github.com/d-matrix-ai/keyformer-llm)
- [ ] [FASTDECODE] FASTDECODE: High-Throughput GPU-Efficient LLM Serving using Heterogeneous — [Paper](https://arxiv.org/abs/2403.11421.pdf) | Code: N/A
- [ ] [ALISA] ALISA: Accelerating LLM Inference via Sparsity-Aware KV Caching — [Paper](https://arxiv.org/abs/2403.17312.pdf) | Code: N/A
- [ ] [GEAR] GEAR: An Efficient KV Cache Compression Recipe for Near-Lossless Generative Inference of LLM — [Paper](https://arxiv.org/abs/2403.05527) | [Code](https://github.com/opengear-project/GEAR)
- [ ] [SqueezeAttention] SQUEEZEATTENTION: 2D Management of KV-Cache in LLM Inference via Layer-wise Optimal Budget — [Paper](https://arxiv.org/abs/2404.04793.pdf) | [Code](https://github.com/hetailang/SqueezeAttention)
- [ ] [SnapKV] SnapKV: LLM Knows What You are Looking for Before Generation — [Paper](https://arxiv.org/abs/2404.14469) | [Code](https://github.com/FasterDecoding/SnapKV)
- [ ] [KVCache-1Bit] KV Cache is 1 Bit Per Channel: Efficient LLM Inference with Coupled Quantization — [Paper](https://arxiv.org/abs/2405.03917) | Code: N/A
- [ ] [KV-Runahead] KV-Runahead: Scalable Causal LLM Inference by Parallel Key-Value Cache Generation — [Paper](https://arxiv.org/abs/2405.05329) | Code: N/A
- [ ] [ZipCache] ZipCache: Accurate and Efficient KV Cache Quantization with Salient Token Identification — [Paper](https://arxiv.org/abs/2405.14256) | Code: N/A
- [ ] [MiniCache] MiniCache: KV Cache Compression in Depth Dimension for LLMs — [Paper](https://arxiv.org/abs/2405.14366) | Code: N/A
- [ ] [CacheBlend] CacheBlend: Fast LLM Serving with Cached Knowledge Fusion — [Paper](https://arxiv.org/abs/2405.16444) | [Code](https://github.com/LMCache/LMCache)
- [ ] [CompressKV] Effectively Compress KV Heads for LLM — [Paper](https://arxiv.org/abs/2406.07056) | Code: N/A
- [ ] [MemServe] MemServe: Context Caching for Disaggregated LLM Serving with Elastic Memory Pool — [Paper](https://arxiv.org/abs/2406.17565) | Code: N/A
- [ ] [MLKV] MLKV: Multi-Layer Key-Value Heads for Memory Efficient Transformer Decoding — [Paper](https://arxiv.org/abs/2406.09297) | [Code](https://github.com/zaydzuhri/pythia-mlkv)
- [ ] [ThinK] ThinK: Thinner Key Cache by Query-Driven Pruning — [Paper](https://arxiv.org/abs/2407.21018) | Code: N/A
- [ ] [Palu] Palu: Compressing KV-Cache with Low-Rank Projection — [Paper](https://arxiv.org/abs/2407.21118) | [Code](https://github.com/shadowpa0327/Palu)
- [ ] [Zero-Delay QKV Compression] Zero-Delay QKV Compression for Mitigating KV Cache and Network Bottlenecks in LLM Inference — [Paper](https://arxiv.org/abs/2408.04107) | Code: N/A
- [ ] [AlignedKV] AlignedKV: Reducing Memory Access of KV-Cache with Precision-Aligned Quantization — [Paper](https://arxiv.org/abs/2409.16546) | [Code](https://github.com/AlignedQuant/AlignedKV)
- [ ] [LayerKV] Optimizing LLM Serving with Layer-wise KV Cache Management — [Paper](https://arxiv.org/abs/2410.00428) | Code: N/A
- [ ] [AdaKV] Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation for Efficient LLM Inference — [Paper](https://arxiv.org/abs/2407.11550) | [Code](https://github.com/FFY0/AdaKV)
- [ ] [KV Cache Recomputation] Efficient LLM Inference with I/O-Aware Partial KV Cache Recomputation — [Paper](https://arxiv.org/abs/2411.17089) | Code: N/A
- [ ] [ClusterKV] ClusterKV: Manipulating LLM KV Cache in Semantic Space for Recallable Compression — [Paper](https://arxiv.org/abs/2412.03213) | Code: N/A
- [ ] [DynamicKV] DynamicKV: Task-Aware Adaptive KV Cache Compression for Long Context LLMs — [Paper](https://arxiv.org/abs/2412.14838) | Code: N/A
- [ ] [DynamicLLaVA] Dynamic-LLaVA: Efficient MLLMs via Dynamic Vision-language Context Sparsification — [Paper](https://arxiv.org/abs/2412.00876) | [Code](https://github.com/Osilly/dynamic_llava)
- [ ] [CacheCraft] Cache-Craft: Managing Chunk-Caches for Efficient Retrieval-Augmented Generation — [Paper](https://www.arxiv.org/abs/2502.15734) | Code: N/A
- [ ] [KV Cache Prefetch] Accelerating LLM Inference Throughput via Asynchronous KV Cache Prefetching — [Paper](https://arxiv.org/abs/2504.06319) | Code: N/A
- [ ] [KVzip] KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction — [Paper](https://arxiv.org/abs/2505.23416) | [Code](https://github.com/snu-mllab/KVzip)
- [ ] [Inference-Time Hyper-Scaling] Inference-Time Hyper-Scaling with KV Cache Compression — [Paper](https://arxiv.org/abs/2506.05345) | Code: N/A
- [ ] [AVP] Agent Vector Protocol: Cross-Model KV-Cache Transfer via Vocabulary-Mediated Projection — [Paper](https://github.com/VectorArc/avp-spec) | [Code](https://github.com/VectorArc/avp-python)

## 📖 Prompt/Context/KV Compression

- [ ] [Selective-Context] Compressing Context to Enhance Inference Efficiency of LLMs — [Paper](https://arxiv.org/abs/2310.06201.pdf) | [Code](https://github.com/liyucheng09/Selective_Context)
- [ ] [AutoCompressor] Adapting Language Models to Compress Contexts — [Paper](https://arxiv.org/abs/2305.14788.pdf) | [Code](https://github.com/princeton-nlp/AutoCompressors)
- [ ] [LLMLingua] LLMLingua: Compressing Prompts for Accelerated Inference of LLMs — [Paper](https://arxiv.org/abs/2310.05736.pdf) | [Code](https://github.com/microsoft/LLMLingua)
- [ ] [LongLLMLingua] LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression — [Paper](https://arxiv.org/abs/2310.06839) | [Code](https://github.com/microsoft/LLMLingua)
- [ ] [LLMLingua-2] LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression — [Paper](https://arxiv.org/abs/2403.12968.pdf) | [Code](https://github.com/microsoft/LLMLingua)
- [ ] [500xCompressor] 500xCompressor: Generalized Prompt Compression for LLMs — [Paper](https://arxiv.org/abs/2408.03094) | Code: N/A
- [ ] [Eigen Attention] Eigen Attention: Attention in Low-Rank Space for KV Cache Compression — [Paper](https://arxiv.org/abs/2408.05646) | Code: N/A
- [ ] [Prompt Compression] Prompt Compression with Context-Aware Sentence Encoding for Fast and Improved LLM Inference — [Paper](https://arxiv.org/abs/2409.01227) | Code: N/A
- [ ] [Context Distillation] Efficient LLM Context Distillation — [Paper](https://arxiv.org/abs/2409.01930) | Code: N/A
- [ ] [CRITIPREFILL] CRITIPREFILL: A SEGMENT-WISE CRITICALITY BASED APPROACH FOR PREFILLING ACCELERATION IN LLMS — [Paper](https://arxiv.org/abs/2409.12490) | [Code](https://github.com/66RING/CritiPrefill)
- [ ] [KV-COMPRESS] PAGED KV-CACHE COMPRESSION WITH VARIABLE COMPRESSION RATES PER ATTENTION HEAD — [Paper](https://arxiv.org/abs/2410.00161) | [Code](https://github.com/IsaacRe/vllm-kvcompress)
- [ ] [LORC] Low-Rank Compression for LLMs KV Cache with a Progressive Compression Strategy — [Paper](https://arxiv.org/abs/2410.03111) | Code: N/A
- [ ] [KVTC] KV Cache Transform Coding for Compact Storage in LLM Inference — [Paper](https://arxiv.org/abs/2511.01815) | Code: N/A

## 📖 Long Context Attention / KV Cache Optimization

- [ ] [Landmark Attention] Random-Access Infinite Context Length for Transformers — [Paper](https://arxiv.org/abs/2305.16300.pdf) | [Code](https://github.com/epfml/landmark-attention/)
- [ ] [LightningAttention-1] TRANSNORMERLLM: A FASTER AND BETTER LARGE LANGUAGE MODEL WITH IMPROVED TRANSNORMER — [Paper](https://arxiv.org/abs/2307.14995.pdf) | [Code](https://github.com/OpenNLPLab/TransnormerLLM)
- [ ] [LightningAttention-2] Lightning Attention-2: A Free Lunch for Handling Unlimited Sequence Lengths in LLMs — [Paper](https://arxiv.org/abs/2401.04658.pdf) | [Code](https://github.com/OpenNLPLab/lightning-attention)
- [ ] [HyperAttention] HyperAttention: Long-context Attention in Near-Linear Time — [Paper](https://arxiv.org/abs/2310.05869.pdf) | [Code](https://github.com/insuhan/hyper-attn)
- [ ] [Streaming Attention] One Pass Streaming Algorithm for Super Long Token Attention Approximation — [Paper](https://arxiv.org/abs/2311.14652.pdf) | Code: N/A
- [ ] [Prompt Cache] PROMPT CACHE: MODULAR ATTENTION REUSE FOR LOW-LATENCY INFERENCE — [Paper](https://arxiv.org/abs/2311.04934.pdf) | Code: N/A
- [ ] [KVQuant] KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization — [Paper](https://browse.arxiv.org/abs/2401.18079.pdf) | [Code](https://github.com/SqueezeAILab/KVQuant/)
- [ ] [RelayAttention] RelayAttention for Efficient LLM Serving with Long System Prompts — [Paper](https://arxiv.org/abs/2402.14808.pdf) | Code: N/A
- [ ] [Infini-attention] Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention — [Paper](https://arxiv.org/abs/2404.07143.pdf) | Code: N/A
- [ ] [RAGCache] RAGCache: Efficient Knowledge Caching for Retrieval-Augmented Generation — [Paper](https://arxiv.org/abs/2404.12457.pdf) | Code: N/A
- [ ] [KCache] EFFICIENT LLM INFERENCE WITH KCACHE — [Paper](https://arxiv.org/abs/2404.18057) | Code: N/A
- [ ] [HOMER] Hierarchical Context Merging: Better Long Context Understanding for Pre-trained LLMs — [Paper](https://arxiv.org/abs/2404.10308) | [Code](https://github.com/alinlab/HOMER)
- [ ] [SKVQ] SKVQ: Sliding-window Key and Value Cache Quantization for LLMs — [Paper](https://arxiv.org/abs/2405.06219) | Code: N/A
- [ ] [CLA] Reducing Transformer Key-Value Cache Size with Cross-Layer Attention — [Paper](https://arxiv.org/abs/2405.12981) | Code: N/A
- [ ] [LOOK-M] LOOK-M: Look-Once Optimization in KV Cache for Efficient Multimodal Long-Context Inference — [Paper](https://arxiv.org/abs/2406.18139) | [Code](https://github.com/SUSTechBruce/LOOK-M)
- [ ] [InfiniGen] InfiniGen: Efficient Generative Inference of LLMs with Dynamic KV Cache Management — [Paper](https://arxiv.org/abs/2406.19707) | Code: N/A
- [ ] [Quest] Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference — [Paper](https://arxiv.org/abs/2406.10774) | [Code](https://github.com/mit-han-lab/Quest)
- [ ] [PQCache] PQCache: Product Quantization-based KVCache for Long Context LLM Inference — [Paper](https://arxiv.org/abs/2407.12820) | Code: N/A
- [ ] [SentenceVAE] SentenceVAE: Faster, Longer and More Accurate Inference with Next-sentence Prediction for LLMs — [Paper](https://arxiv.org/abs/2408.00655) | Code: N/A
- [ ] [InstInfer] InstInfer: In-Storage Attention Offloading for Cost-Effective Long-Context LLM Inference — [Paper](https://arxiv.org/abs/2409.04992) | Code: N/A
- [ ] [RetrievalAttention] RetrievalAttention: Accelerating Long-Context LLM Inference via Vector Retrieval — [Paper](https://arxiv.org/abs/2409.10516) | Code: N/A
- [ ] [ShadowKV] ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference — [Paper](https://arxiv.org/abs/2410.21465) | [Code](https://github.com/bytedance/ShadowKV)
- [ ] [REFORM] Compress, Gather, and Recompute: REFORMing Long-Context Processing in Transformers — [Paper](https://arxiv.org/abs/2506.01215) | Code: N/A

## 📖 Early-Exit / Intermediate Layer Decoding

- [ ] [DeeBERT] DeeBERT: Dynamic Early Exiting for Accelerating BERT Inference — [Paper](https://arxiv.org/abs/2004.12993.pdf) | Code: N/A
- [ ] [FastBERT] FastBERT: a Self-distilling BERT with Adaptive Inference Time — [Paper](https://aclanthology.org/2020.acl-main.537.pdf) | [Code](https://github.com/autoliuweijie/FastBERT)
- [ ] [BERxiT] BERxiT: Early Exiting for BERT with Better Fine-Tuning and Extension to Regression — [Paper](https://aclanthology.org/2021.eacl-main.8.pdf) | [Code](https://github.com/castorini/berxit)
- [ ] [SkipDecode] SkipDecode: Autoregressive Skip Decoding with Batching and Caching for Efficient LLM Inference — [Paper](https://arxiv.org/abs/2307.02628) | Code: N/A
- [ ] [LITE] Accelerating LLaMA Inference by Enabling Intermediate Layer Decoding via Instruction Tuning with LITE — [Paper](https://arxiv.org/abs/2310.18581v2.pdf) | Code: N/A
- [ ] [EE-LLM] EE-LLM: Large-Scale Training and Inference of Early-Exit LLMs with 3D Parallelism — [Paper](https://arxiv.org/abs/2312.04916.pdf) | [Code](https://github.com/pan-x-c/EE-LLM)
- [ ] [FREE] Fast and Robust Early-Exiting Framework for Autoregressive Language Models with Synchronized Parallel Decoding — [Paper](https://arxiv.org/abs/2310.05424.pdf) | [Code](https://github.com/raymin0223/fast_robust_early_exit)
- [ ] [EE-Tuning] EE-Tuning: An Economical yet Scalable Solution for Tuning Early-Exit LLMs — [Paper](https://arxiv.org/abs/2402.00518) | [Code](https://github.com/pan-x-c/EE-LLM)
- [ ] [Skip Attention] Attention Is All You Need But You Don't Need All Of It For Inference of LLMs — [Paper](https://arxiv.org/abs/2407.15516) | Code: N/A
- [ ] [KOALA] KOALA: Enhancing Speculative Decoding for LLM via Multi-Layer Draft Heads with Adversarial Learning — [Paper](https://arxiv.org/abs/2408.08146) | Code: N/A

## 📖 Parallel Decoding / Sampling

- [ ] [Blockwise Parallel Decoding] Blockwise Parallel Decoding for Deep Autoregressive Models — [Paper](https://arxiv.org/abs/1811.03115.pdf) | Code: N/A
- [ ] [Speculative Sampling (DeepMind)] Accelerating LLM Decoding with Speculative Sampling — [Paper](https://arxiv.org/abs/2302.01318.pdf) | Code: N/A
- [ ] [Speculative Decoding (Google)] Fast Inference from Transformers via Speculative Decoding — [Paper](https://arxiv.org/abs/2211.17192.pdf) | [Code](https://github.com/feifeibear/LLMSpeculativeSampling)
- [ ] [OSD] Online Speculative Decoding — [Paper](https://arxiv.org/abs/2310.07177.pdf) | Code: N/A
- [ ] [Cascade Speculative] Cascade Speculative Drafting for Even Faster LLM Inference — [Paper](https://arxiv.org/abs/2312.11462.pdf) | Code: N/A
- [ ] [LookaheadDecoding] Break the Sequential Dependency of LLM Inference Using LOOKAHEAD DECODING — [Paper](https://arxiv.org/abs/2402.02057.pdf) | [Code](https://github.com/hao-ai-lab/LookaheadDecoding)
- [ ] [Decoding Speculative Decoding] Decoding Speculative Decoding — [Paper](https://arxiv.org/abs/2402.01528.pdf) | [Code](https://github.com/uw-mad-dash/decoding-speculative-decoding)
- [ ] [TriForce] TriForce: Lossless Acceleration of Long Sequence Generation with Hierarchical Speculative Decoding — [Paper](https://arxiv.org/abs/2404.11912) | [Code](https://github.com/Infini-AI-Lab/TriForce)
- [ ] [Hidden Transfer] Parallel Decoding via Hidden Transfer for Lossless LLM Acceleration — [Paper](https://arxiv.org/abs/2404.12022.pdf) | Code: N/A
- [ ] [Instructive Decoding] INSTRUCTIVE DECODING: INSTRUCTION-TUNED LLMs ARE SELF-REFINER FROM NOISY INSTRUCTIONS — [Paper](https://openreview.net/pdf?id=LebzzClHYw) | [Code](https://github.com/joonkeekim/Instructive-Decoding)
- [ ] [S3D] S3D: A Simple and Cost-Effective Self-Speculative Decoding Scheme for Low-Memory GPUs — [Paper](https://arxiv.org/abs/2405.20314) | Code: N/A
- [ ] [Parallel Decoding (KAIST)] Exploring and Improving Drafts in Blockwise Parallel Decoding — [Paper](https://arxiv.org/abs/2404.09221) | Code: N/A
- [ ] [Multi-Token Speculative Decoding] Multi-Token Joint Speculative Decoding for Accelerating LLM Inference — [Paper](https://arxiv.org/abs/2404.09221) | Code: N/A
- [ ] [Token Recycling] Turning Trash into Treasure: Accelerating Inference of LLMs with Token Recycling — [Paper](https://arxiv.org/abs/2408.08696) | Code: N/A
- [ ] [PEARL] Parallel Speculative Decoding with Adaptive Draft Length — [Paper](https://arxiv.org/abs/2408.11850) | [Code](https://github.com/smart-lty/ParallelSpeculativeDecoding)
- [ ] [FocusLLM] FocusLLM: Scaling LLM's Context by Parallel Decoding — [Paper](https://arxiv.org/abs/2408.11745) | [Code](https://github.com/leezythu/FocusLLM)
- [ ] [MagicDec] MagicDec: Breaking the Latency-Throughput Tradeoff for Long Context Generation with Speculative Decoding — [Paper](https://arxiv.org/abs/2408.11049) | [Code](https://github.com/Infini-AI-Lab/MagicDec/)
- [ ] [Boosting Lossless Speculative Decoding] Boosting Lossless Speculative Decoding via Feature Sampling and Partial Alignment Distillation — [Paper](https://arxiv.org/abs/2408.15562) | Code: N/A
- [ ] [Hybrid Inference] Efficient Hybrid Inference for LLMs: Reward-Based Token Modelling with Selective Cloud Assistance — [Paper](https://arxiv.org/abs/2409.13757) | Code: N/A
- [ ] [PARALLELSPEC] PARALLELSPEC: PARALLEL DRAFTER FOR EFFICIENT SPECULATIVE DECODING — [Paper](https://arxiv.org/abs/2410.05589) | Code: N/A
- [ ] [Fast Best-of-N] Fast Best-of-N Decoding via Speculative Rejection — [Paper](https://arxiv.org/abs/2410.20290) | Code: N/A
- [ ] [Mamba Drafters] Mamba Drafters for Speculative Decoding — [Paper](https://arxiv.org/abs/2506.01206) | Code: N/A
- [ ] [STAND] Accelerated Test-Time Scaling with Model-Free Speculative Sampling — [Paper](https://arxiv.org/abs/2506.04708) | Code: N/A
- [ ] [MineDraft] MineDraft: A Framework for Batch Parallel Speculative Decoding — [Paper](https://arxiv.org/abs/2603.18016) | [Code](https://github.com/electron-shaders/MineDraft)

## 📖 Structured Prune / KD / Weight Sparse

- [ ] [FLAP] Fluctuation-based Adaptive Structured Pruning for LLMs — [Paper](https://arxiv.org/abs/2312.11983.pdf) | [Code](https://github.com/CASIA-IVA-Lab/FLAP)
- [ ] [LASER] The Truth is in There: Improving Reasoning in Language Models with Layer-Selective Rank Reduction — [Paper](https://arxiv.org/abs/2312.13558.pdf) | [Code](https://github.com/pratyushasharma/laser)
- [ ] [ADMM Pruning] Fast and Optimal Weight Update for Pruned LLMs — [Paper](https://arxiv.org/abs/2401.02938.pdf) | [Code](https://github.com/fmfi-compbio/admm-pruning)
- [ ] [FFSplit] FFSplit: Split Feed-Forward Network For Optimizing Accuracy-Efficiency Trade-off in Language Model Inference — [Paper](https://arxiv.org/abs/2401.04044.pdf) | Code: N/A
- [ ] [Simba] Sparsified State-Space Models are Efficient Highway Networks — [Paper](https://arxiv.org/abs/2505.20698) | [Code](https://github.com/woominsong/Simba)
- [ ] [SDMPrune] SDMPrune: Self-Distillation MLP Pruning for Efficient LLMs — [Paper](https://arxiv.org/abs/2506.11120) | [Code](https://github.com/visresearch/SDMPrune)
- [ ] [HFPrune] High-Fidelity Pruning for Large Language Models — [Paper](https://arxiv.org/abs/2603.08083) | [Code](https://github.com/visresearch/HFPrune)

## 📖 MoE LLM Inference

- [ ] [Mixtral Offloading] Fast Inference of Mixture-of-Experts Language Models with Offloading — [Paper](https://arxiv.org/abs/2312.17238.pdf) | [Code](https://github.com/dvmazur/mixtral-offloading)
- [ ] [MoE-Mamba] MoE-Mamba: Efficient Selective State Space Models with Mixture of Experts — [Paper](https://arxiv.org/abs/2401.04081.pdf) | Code: N/A
- [ ] [MoE Inference] Toward Inference-optimal Mixture-of-Expert LLMs — [Paper](https://arxiv.org/abs/2404.02852.pdf) | Code: N/A
- [ ] [MoE Survey] A Survey on Mixture of Experts — [Paper](https://arxiv.org/abs/2407.06204) | Code: N/A

## 📖 CPU / Single GPU / FPGA / NPU / Mobile

- [ ] [LLM CPU Inference] Efficient LLM Inference on CPUs — [Paper](https://arxiv.org/abs/2311.00502.pdf) | [Code](https://github.com/intel/intel-extension-for-transformers)
- [ ] [LinguaLinked] LinguaLinked: A Distributed LLM Inference System for Mobile Devices — [Paper](https://arxiv.org/abs/2312.00388.pdf) | Code: N/A
- [ ] [FlightLLM] FlightLLM: Efficient LLM Inference with a Complete Mapping Flow on FPGAs — [Paper](https://arxiv.org/abs/2401.03868.pdf) | Code: N/A
- [ ] [Transformer-Lite] Transformer-Lite: High-efficiency Deployment of LLMs on Mobile Phone GPUs — [Paper](https://arxiv.org/ftp/arxiv/papers/2403/2403.20041.pdf) | Code: N/A
- [ ] [xFasterTransformer] Inference Performance Optimization for LLMs on CPUs — [Paper](https://arxiv.org/abs/2407.07304) | [Code](https://github.com/intel/xFasterTransformer)
- [ ] [Inference Optimization on AI Accelerators] Inference Optimization of Foundation Models on AI Accelerators — [Paper](https://arxiv.org/abs/2407.09111) | Code: N/A
- [ ] [LLM Mobile Benchmarking] LLM Performance Benchmarking on Mobile Platforms — [Paper](https://arxiv.org/abs/2410.03613) | Code: N/A
- [ ] [FastAttention (NPU)] FastAttention: Extend FlashAttention2 to NPUs and Low-resource GPUs — [Paper](https://arxiv.org/abs/2410.16663) | Code: N/A
- [ ] [NITRO] NITRO: LLM INFERENCE ON INTEL LAPTOP NPUS — [Paper](https://arxiv.org/abs/2412.11053) | [Code](https://github.com/abdelfattah-lab/nitro)
- [ ] [Off Grid] On-device LLM + Vision + Image Gen for iOS & Android — [Paper](https://github.com/alichherawalla/off-grid-mobile) | [Code](https://github.com/alichherawalla/off-grid-mobile)
- [ ] [Grail-V/PSE] Non-bijunctive Attention Collapse via POWER8 vec_perm — [Paper](https://doi.org/10.5281/zenodo.14862410) | [Code](https://github.com/Scottcjn/ram-coffers)
- [ ] [llama-cpp-power8] POWER8 optimizations for llama.cpp — [Paper](https://github.com/Scottcjn/llama-cpp-power8) | [Code](https://github.com/Scottcjn/llama-cpp-power8)
- [ ] [RAM Coffers] NUMA-aware weight banking for LLM inference — [Paper](https://github.com/Scottcjn/ram-coffers) | [Code](https://github.com/Scottcjn/ram-coffers)

## 📖 Non Transformer Architecture

- [ ] [RWKV] RWKV: Reinventing RNNs for the Transformer Era — [Paper](https://arxiv.org/abs/2305.13048.pdf) | [Code](https://github.com/BlinkDL/RWKV-LM)
- [ ] [Mamba] Mamba: Linear-Time Sequence Modeling with Selective State Spaces — [Paper](https://arxiv.org/abs/2312.00752.pdf) | [Code](https://github.com/state-spaces/mamba)
- [ ] [RWKV-CLIP] RWKV-CLIP: A Robust Vision-Language Representation Learner — [Paper](https://arxiv.org/abs/2406.06973) | [Code](https://github.com/deepglint/RWKV-CLIP)
- [ ] [Kraken] Kraken: Inherently Parallel Transformers For Efficient Multi-Device Inference — [Paper](https://arxiv.org/abs/2408.07802) | Code: N/A
- [ ] [FLA] Flash Linear Attention: A Triton-Based Library for Hardware-Efficient Implementations of Linear Attention — [Paper](https://github.com/sustcsonglin/flash-linear-attention) | [Code](https://github.com/sustcsonglin/flash-linear-attention)

## 📖 GEMM / Tensor Cores / MMA / Parallel

- [ ] [Tensor Core Programmability] NVIDIA Tensor Core Programmability, Performance & Precision — [Paper](https://arxiv.org/abs/1803.04014.pdf) | Code: N/A
- [ ] [Intra-SM Parallelism] Exploiting Intra-SM Parallelism in GPUs via Persistent and Elastic Blocks — [Paper](https://mivenhan.github.io/publication/2021plasticine/2021plasticine.pdf) | Code: N/A
- [ ] [Dissecting Tensor Cores] Dissecting Tensor Cores via Microbenchmarks — [Paper](https://arxiv.org/abs/2206.02874.pdf) | [Code](https://github.com/sunlex0717/DissectingTensorCores)
- [ ] [FP8 Formats] FP8 FORMATS FOR DEEP LEARNING — [Paper](https://arxiv.org/abs/2209.05433.pdf) | Code: N/A
- [ ] [WMMA Extension] Reducing shared memory footprint to leverage high throughput on Tensor Cores — [Paper](https://arxiv.org/abs/2308.15152.pdf) | [Code](https://github.com/wmmae/wmma_extension)
- [ ] [cutlass/cute] Graphene: An IR for Optimized Tensor Computations on GPUs — [Paper](https://dl.acm.org/doi/pdf/10.1145/3582016.3582018) | [Code](https://github.com/NVIDIA/cutlass)
- [ ] [QUICK] QUICK: Quantization-aware Interleaving and Conflict-free Kernel for efficient LLM inference — [Paper](https://arxiv.org/abs/2402.10076.pdf) | [Code](https://github.com/SqueezeBits/QUICK)
- [ ] [TP-AWARE DEQUANTIZATION] TP-AWARE DEQUANTIZATION — [Paper](https://arxiv.org/abs/2402.04925.pdf) | Code: N/A
- [ ] [flute] Fast Matrix Multiplications for Lookup Table-Quantized LLMs — [Paper](https://arxiv.org/abs/2407.10960) | [Code](https://github.com/HanGuo97/flute)
- [ ] [LUT TENSOR CORE] LUT TENSOR CORE: Lookup Table Enables Efficient Low-Bit LLM Inference Acceleration — [Paper](https://arxiv.org/abs/2408.06003) | Code: N/A
- [ ] [MARLIN] MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on LLMs — [Paper](https://arxiv.org/abs/2408.11743) | [Code](https://github.com/IST-DASLab/marlin)
- [ ] [SpMM] High Performance Unstructured SpMM Computation Using Tensor Cores — [Paper](https://arxiv.org/abs/2408.11551) | Code: N/A
- [ ] [TEE on H100] Confidential Computing on nVIDIA H100 GPU: A Performance Benchmark Study — [Paper](https://arxiv.org/abs/2409.03992) | Code: N/A
- [ ] [HiFloat8] Ascend HiFloat8 Format for Deep Learning — [Paper](https://arxiv.org/abs/2409.16626) | Code: N/A
- [ ] [Efficient Arbitrary Precision] Efficient Arbitrary Precision Acceleration for LLMs on GPU Tensor Cores — [Paper](https://arxiv.org/abs/2409.17870) | Code: N/A
- [ ] [Tensor Product Acceleration] Acceleration of Tensor-Product Operations with Tensor Cores — [Paper](https://arxiv.org/abs/2407.09621) | Code: N/A
- [ ] [HADACORE] HADACORE: TENSOR CORE ACCELERATED HADAMARD TRANSFORM KERNEL — [Paper](https://arxiv.org/abs/2407.09621) | [Code](https://github.com/pytorch-labs/applied-ai)
- [ ] [FLASH-ATTENTION RNG] Reducing the Cost of Dropout in Flash-Attention by Hiding RNG with GEMM — [Paper](https://arxiv.org/abs/2410.07531) | Code: N/A
- [ ] [TRITONBENCH] TRITONBENCH: Benchmarking LLM Capabilities for Generating Triton Operators — [Paper](https://arxiv.org/abs/2502.14752) | [Code](https://github.com/thunlp/TritonBench)
- [ ] [Triton-distributed / TileLink] TileLink: Generating Efficient Compute-Communication Overlapping Kernels using Tile-Centric Primitives — [Paper](https://arxiv.org/abs/2503.20313) | [Code](https://github.com/ByteDance-Seed/Triton-distributed)

## 📖 VLM / Position Embed / Others

- [ ] [RoPE] ROFORMER: ENHANCED TRANSFORMER WITH ROTARY POSITION EMBEDDING — [Paper](https://arxiv.org/abs/2104.09864.pdf) | Code: N/A
- [ ] [ByteTransformer] A High-Performance Transformer Boosted for Variable-Length Inputs — [Paper](https://arxiv.org/abs/2210.03052.pdf) | [Code](https://github.com/bytedance/ByteTransformer)
- [ ] [Inf-MLLM] Inf-MLLM: Efficient Streaming Inference of Multimodal LLMs on a Single GPU — [Paper](https://arxiv.org/abs/2409.09086) | Code: N/A
- [ ] [VL-CACHE] VL-CACHE: SPARSITY AND MODALITY-AWARE KV CACHE COMPRESSION FOR VISION-LANGUAGE MODEL INFERENCE ACCELERATION — [Paper](https://arxiv.org/abs/2410.23317) | Code: N/A

## 📖 LLM Inference Applications

- [ ] [StoryRoute] Real-time LLM inference-powered GPS audio tour app — [Paper](https://storyroute.netlify.app) | [Code](https://github.com/samirasadov28-code/storyroute)
