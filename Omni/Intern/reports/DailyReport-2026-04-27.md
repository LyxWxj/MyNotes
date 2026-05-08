# 娄雨轩 Daily Report — 2026-04-27

## 今日工作

1. **文献精读**：精读了 Qwen-Image 和 Qwen3-Omni 的技术报告，以及 vLLM-Omni 论文。Qwen-Image 的推理链路是 VAE Encoder → Qwen2.5-VL → MMDiT → VAE Decoder，各组件参数规模从 54M 到 20B 不等，耗时分布差异很大。Qwen3-Omni 采用 Thinker-Talker 流水线，Thinker 负责融合多模态输入，Talker 负责生成文本和音频码流。vLLM-Omni 在这类 AR 模型上已经做了 Thinker-Talker 分离。

2. **源码分析**：研究了 vLLM-Omni 中 Qwen3-Omni 的 pipeline 实现。另外研究了 vLLM 中 offload-prefetch 的两种模式（prefetch 和 uva），以及distributed执行模块的实现。

3. **论文收集**：收集了 14 篇相关论文，涉及分离式推理（Splitwise、DistServe、Mooncake）、扩散模型并行（DistriFusion、PipeFusion、xDiT）、异构部署（HeteGen、SiPipe）和推理缓存（LMCache、IMPRESS、PRESERVE）几个方向，初步理出了文献综述的结构。

## 明日计划

- 阅读分布式推理相关文章（Splitwise、DistServe 等），理解 prefill-decode 分离和异构部署的方法论。
- 阅读缓存相关文章（LMCache、PRESERVE 等），了解现有缓存方案的层级设计和预取机制。
- 开始写 Introduction 和 Related Work 的初稿。
