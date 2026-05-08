# 娄雨轩 Daily Report — 2026-04-29

## 今日工作

1. **文献精读**：精读论文。
   - *FlashPS*：提出面向扩散图像编辑的 mask-aware 缓存和调度策略，在编辑场景下利用 mask 区域特征复用减少计算量。
   - *DiT-Serve*：面向 Diffusion Transformer 的高效服务引擎，针对 DiT 推理特点设计了专门的调度和缓存机制。
   - *TridentServe*：提出扩散流水线的 stage-level 服务系统，将扩散推理过程划分为多阶段进行精细化管理。
   - *GENSERVE*：实现异构扩散模型工作负载的高效协同服务，解决多种扩散模型混合部署时的资源分配问题。
   - *Towards Efficient Generative LLM Serving*：综述性论文，全面梳理了从算法到系统的生成式大模型服务优化技术。
   - *SwiftDiffusion*: 有界异步LoRA加载技术，为 CFG 设计的潜在并行 (Latent Parallelism)。

2. **源码分析**：
   - 研究了 **Cache-DiT** 的流水线实现源码，理解其基于扩散 Transformer 的缓存架构和调度逻辑。
   - 研究了 **vLLM-Omni** 的 Orchestrator 模块，掌握其在多模态推理中的任务编排与资源协调机制。
   - 研究了 **Omni-Connectors**，**Omni-Coordinator** 源码，了解各模态连接器的接口设计与数据流转方式和分布式环境下对多节点推理任务的协调与控制逻辑。

## 明日计划

- 整理论文阅读笔记，归纳各方案的核心思路与适用场景。
- 结合 vLLM-Omni 的 Orchestrator 和 Omni-Coordinator 理解整体推理框架的架构设计。
- 深入了解vLLM以及其他前沿引擎中的batching策略。
- 深入了解模型结构，参数量，激活量，计算量与显存大小之间的关系，收集相关论文。
