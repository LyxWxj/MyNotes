# obsrepo

个人知识库，基于 [Tolaria](https://github.com/refactoringhq/tolaria) 构建的 Markdown 笔记仓库，同时兼容 [Obsidian](https://obsidian.md/) 编辑。

## 目录结构

```
obsrepo/
├── Notes/                          # 笔记主体
│   ├── Diffusion/                  # 扩散模型
│   │   ├── Diffusion/              #   基础理论（VAE、Stable Diffusion、蒸馏）
│   │   ├── Sampling/               #   采样算法（DDPM、DDIM、DPM-Solver、Flow Matching）
│   │   ├── SDE-ODE/                #   SDE/ODE 数学基础
│   │   ├── ServingSystem/          #   推理服务系统（DiT-Serve、GenServe、TetriServe 等）
│   │   ├── Caching/                #   缓存加速
│   │   └── X2Video/                #   视频生成（CogVideoX、Wan、Self-Forcing 等）
│   ├── LargeLanguageModel/         # 大语言模型（SFT/RLHF、数据、评估）
│   ├── ReinforceLearning/          # 强化学习（数学基础、实验记录）
│   ├── vLLM/                       # vLLM 推理框架笔记 & 设计文档
│   ├── vLLM-omni/                  # vLLM-Omni 多模态推理框架
│   ├── CPP/                        # C++（模板元编程）
│   ├── Tutorial/                   # 教程（卡尔曼滤波等）
│   ├── Awesome-DiT-Inference Papers Reading TODO.md
│   ├── Awesome-LLM-Inference Papers Reading TODO.md
│   └── Awesome-3D-Generative-Models-TODO.md
├── DS Panorama/                    # 数据结构与算法（代码随想录、王道数据结构）
├── blog/                           # 博客草稿（Attention 优化、CMake 等）
├── skills/                         # 自定义技能（PDF 文本提取）
├── 考研资料汇总.md                 # 考研资料整理
└── 关于复试.md                     # 复试相关
```

## 主要内容

### 🧠 大模型推理与服务

- **vLLM / vLLM-Omni**：LLM 和多模态模型的高性能推理框架笔记，包含架构设计、分布式部署、量化、投机解码等
- **Diffusion Serving System**：扩散模型推理服务系统（DiT-Serve、GenServe、TetriServe、TridentServe）

### 🎨 扩散模型

- **理论基础**：DDPM、DDIM、DPM-Solver、Flow Matching、SDE/ODE 统一视角
- **视频生成**：CogVideoX、Wan、Diffusion Forcing、Self-Forcing 等
- **缓存与加速**：推理阶段的缓存策略

### 🔊 流式生成

- **流式视频生成**：分离式 DiT 推理方案设计、FlashDreams 架构分析
- **流式语音生成**：MiniCPM-o 4.5 模型分析

### 📚 基础知识

- **强化学习**：数学基础、RLHF、实验记录
- **C++ 模板**：C++ Templates 第二版笔记
- **数据结构与算法**：代码随想录、王道数据结构

## 工具链

- **[Tolaria](https://github.com/refactoringhq/tolaria)**：Markdown 知识图谱引擎
- **[Obsidian](https://obsidian.md/)**：本地 Markdown 编辑器
- **Claude Code + Superpowers-ZH**：AI 辅助写作与代码开发
