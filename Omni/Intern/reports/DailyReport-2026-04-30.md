# 娄雨轩 Daily Report — 2026-04-30

## 今日工作

1. **vllm-ascend 项目**：
   - 理解 vllm-ascend 的 platform 层设计，了解其如何将 vLLM 的 GPU-centric 抽象映射到 Ascend NPU 上。

2. **Ascend 卡量化知识学习**：
   - 了解了 Ascend NPU 上支持的低精度量化方案，包括 FP16、INT8、W8A16 等，以及各方案在精度与性能之间的权衡。
   - 学习了 Ascend 量化工具链的基本流程：校准（Calibration）→ 量化（Quantization）→ 部署（Deployment）。

3. **KVDirect 阅读**：
   - 阅读了 KVDirect 的设计思路，理解其如何通过绕过传统 KV cache 管理路径，直接在 GPU/NPU 之间传输 KV cache 数据，减少显存拷贝和通信开销。
   - 分析其在高吞吐服务场景下的适用性，以及在 Ascend NPU 上落地的可行性。

4. **系统调度机制学习**：
   - 阅读了 **In-flight Batching** 机制，理解 vLLM 如何在每次迭代中动态调度请求，通过 continuous batching 最大化 GPU 利用率。
   - 阅读了 **vLLM BlockManager** 源码，掌握其以逻辑块（Logical Blocks）管理 KV cache 显存的分配与回收逻辑，以及 block table 的维护方式。
   - 阅读了 **SequenceGroup** 的执行流程，理解从请求入队、调度决策、block allocation 到最终推理执行的全链路逻辑。

## 明日计划

- 熟悉 vllm-omni项目分支。
- 继续研究 Ascend NPU特性。
- 开始整理系统调度与显存管理的阅读笔记。
