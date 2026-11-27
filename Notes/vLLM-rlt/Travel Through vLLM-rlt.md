---
type: Note
---

# Travel Through vLLM-rlt

> **代码基准 Commit：** `ce200f0d2085fb041c66691245e29aef2a9ed714`（提交时间：`2026-11-20T18:26:24+08:00`）
>
> **笔记核对时间：** `2026-11-23 13:40:20 +0800`（Asia/Shanghai）
>
> **范围：** 当前本地检出；已跟踪文件相对上述 commit 无修改。本地还有未跟踪文件，见目录树下方说明。

## 仓库目录结构

以下按当前 commit 的已跟踪内容整理；目录注释描述主要用途，文件名只列阅读入口。

```text
vllm-rlt/
├── vllm_rlt/                  # 引擎 Python 包；__init__.py 导出公共 API
│   ├── attention/             # attention backend 选择、能力声明与元数据
│   ├── core/                  # Scheduler/policy、KVCacheManager 和 KV 容量规划
│   ├── engine/                # StageClient、StageWorker、StageEngine、connector、抢占
│   ├── entrypoints/           # 离线 LLM/CLI、单 stage/PD 服务入口、公共参数和 chat 模板
│   ├── kernels/               # Torch/Triton/Flash/Ascend attention、KV、routing 等算子
│   ├── layers/                # 模型共用的 normalization 和 rotary embedding
│   ├── models/                # Ouro、Nanbeige、Huginn、Dory 与 ModelConfig/加载分发
│   ├── pd/                    # prefill/decode coordinator、NIXL connector/transport 和配置
│   ├── serving/               # HTTP 协议、aiohttp server、请求 channel 和 engine 线程
│   ├── worker/                # Executor/ModelRuntime、runner、sampler、graphs、buffers、自推测
│   ├── config.py              # cache、scheduler、exit、execution、speculative 配置
│   ├── request.py             # Request/phase、执行描述和 RequestOutput
│   ├── sampling_params.py     # 每请求采样、循环深度及优先级参数
│   ├── device.py              # CPU/CUDA/NPU 设备解析
│   └── profiling*.py          # profiler 会话及 trace/artifact 处理
├── tests/                     # 引擎、scheduler、服务、模型、后端、设备等回归测试
│   └── reference/             # 独立的小模型和 Dory/Huginn 参考实现
├── benchmarks/                # 运行时、抢占、GSM8K 精度评估及相关后端
│   ├── fixtures/              # 固定 GSM8K 子集等 benchmark 输入
│   └── kernels/               # 独立 kernel benchmark
├── csrc/                      # 可选 PyTorch C++/CUDA 扩展与绑定
│   └── activation/            # SiLU-and-Mul CUDA 实现
├── docs/                      # 架构、scheduler、KV、服务、精度和 profiling 文档
│   ├── assets/                # 文档图片、性能图与其他静态资源
│   │   ├── inference/         # Ouro 推理示意图和动画
│   │   ├── performance/       # 已记录的 workload 性能图
│   │   └── runtime/           # runtime 图及生成脚本
│   ├── diagrams/              # stage/scheduler 等 PNG、SVG 和索引
│   └── recipes/               # Ascend、Ouro-Thinking 等特定运行方案
├── requirements/              # 可选后端的额外依赖约束
├── .github/                   # GitHub 贡献模板与 CI
│   ├── ISSUE_TEMPLATE/        # issue 分类模板
│   └── workflows/             # CPU 测试工作流
├── .agents/skills/            # 随仓库维护的专项 agent 工作流
│   ├── rlt-perf-opt/          # 性能分析与优化流程
│   ├── rlt-refactor/          # 推理运行时重构规则
│   └── vllm-rlt-review-pr/    # PR 审查规则
│       ├── agents/            # agent 配置
│       └── references/        # 审查证据、接口和测试规则
├── README.md                  # 项目概览、安装及模型支持
├── pyproject.toml             # Python 依赖、可选项、CLI、pytest/Ruff 配置
├── setup.py / MANIFEST.in     # 本地扩展构建和打包文件
└── CHANGELOG.md / CONTRIBUTING.md / LICENSE / NOTICE
```

## 文件索引

按阅读链路拆分后的章节如下：

- [Initialization](<Travel Through vLLM-rlt/01-initialization.md>)
- [Model、StageClient、StageWorker、StageEngine](<Travel Through vLLM-rlt/02-model-stage.md>)
- [Executor、Profiler、Attention Backend、Ascend NPU](<Travel Through vLLM-rlt/03-executor-and-backends.md>)
- [KVCache、KVCacheManager、Scheduler](<Travel Through vLLM-rlt/04-kv-cache-and-scheduler.md>)
- [SpeculativeRunner、ModelRunner](<Travel Through vLLM-rlt/05-speculative-and-model-runner.md>)
- [PreemptionManager、PDEngine、PDStageConnector](<Travel Through vLLM-rlt/06-preemption-and-pd.md>)
- [Runtime：请求加入、调度与计算](<Travel Through vLLM-rlt/07-runtime-admission-and-scheduling.md>)
- [Runtime：StageEngine.step、异步路径与 schedule](<Travel Through vLLM-rlt/08-runtime-step-and-schedule.md>)
- [Runtime：Executor.submit](<Travel Through vLLM-rlt/09-runtime-submit.md>)
- [Runtime：update 与状态推进](<Travel Through vLLM-rlt/10-runtime-update.md>)
- [Runtime：生命周期、验证与阅读入口](<Travel Through vLLM-rlt/11-runtime-lifecycle-and-validation.md>)
