---
type: Note
---

# Travel Through vLLM-Omni

> **代码基准 Commit：** `e5576969e12c878a9e4c0e9f5ea0a5b4884ce37c`（提交时间：`2026-11-27T01:41:07+08:00`）
>
> **笔记核对时间：** `2026-11-27 12:27:59 +0800`（Asia/Shanghai）
>
> **范围：** 当前本地检出；已跟踪文件相对上述 commit 无修改。本地还有未跟踪文件：`AGENT.md`、`artifacts/`、`docs/design/pipeline_parallelism/`、`feat/`、`refactor/`。它们保持原样，本文按上述 commit 的源码、设计页和测试入口整理。

## 仓库目录结构

以下按当前 commit 的已跟踪内容整理；目录注释描述主要用途，文件名只列阅读入口。

```text
vllm-omni-batching/
├── vllm_omni/
│   ├── config/                  # PipelineRegistry、StagePipelineConfig、deploy/stage 配置
│   ├── entrypoints/             # AsyncOmni、OpenAI、CLI、duplex 等用户入口
│   ├── engine/                  # OmniEngine、StageRuntime、StagePool、Orchestrator、消息协议
│   ├── core/sched/              # Omni AR Scheduler、输入协调、prefill/decode 和抢占
│   ├── core/prefix_cache/       # prefix cache manager/controller/block pool/runner adapter
│   ├── worker_v2/               # AR/generation runner、native data plane 和输出快照
│   ├── diffusion/               # DiffusionEngine、scheduler、executor、worker、runner
│   └── distributed/             # OmniConnector、KV transfer、coordinator 和 membership
├── tests/                       # engine、scheduler、worker、connector、diffusion、entrypoint 回归入口
├── docs/design/                 # architecture、module 和 feature 契约
├── .claude/skills/              # precheck/review 等仓库工作流
├── README.md                    # omni-modality、异构 pipeline、streaming、OpenAI API 总览
├── CONTRIBUTING.md              # 贡献边界和验证要求
└── pyproject.toml               # Python 包、pytest、lint 和构建配置
```

## 文件索引

按阅读链路拆分后的章节如下：

- [Initialization](<Travel Through vLLM-Omni/01-initialization.md>)
- [Entrypoint、消息和 Orchestrator](<Travel Through vLLM-Omni/02-entrypoint-and-orchestration.md>)
- [StageRuntime、StagePool 和 replica](<Travel Through vLLM-Omni/03-stage-runtime-and-replicas.md>)
- [AR Scheduler、KV 和 prefix cache](<Travel Through vLLM-Omni/04-ar-scheduler-and-cache.md>)
- [Diffusion runtime、batching 和 worker](<Travel Through vLLM-Omni/05-diffusion-runtime-and-batching.md>)
- [OmniConnector、async_chunk 和 full payload](<Travel Through vLLM-Omni/06-connector-and-async-chunk.md>)
- [Runtime：请求准入与跨 stage 路由](<Travel Through vLLM-Omni/07-runtime-admission-and-routing.md>)
- [Runtime：stage submit 与设备提交](<Travel Through vLLM-Omni/08-runtime-submit.md>)
- [Runtime：output update 与 materialization](<Travel Through vLLM-Omni/09-runtime-update.md>)
- [Runtime：取消、错误、shutdown 与资源生命周期](<Travel Through vLLM-Omni/10-runtime-lifecycle.md>)
- [验证入口与后续阅读路线](<Travel Through vLLM-Omni/11-validation-and-reading-guide.md>)

## 总体阅读方法

先从 `PipelineConfig` 确认逻辑 stage，再从 `OmniEngineBase` 和 `StageRuntime` 找到实际进程/设备 owner；随后沿 `AsyncOmni.generate()`、`Orchestrator` 和 `StagePool` 追一个请求。进入 stage 后，AR 路径看 `OmniARScheduler`，diffusion 路径看 `DiffusionEngine`。最后回到 `OmniConnector`、`omni_data_plane.py`、abort 和 shutdown，确认数据何时可见、何时可以复用。

