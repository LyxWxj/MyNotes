# LegoDiffusion: Micro-Serving Text-to-Image Diffusion Workflows

**arXiv:** 2604.08123
**Date:** 2026-04-09
**Authors:** Yang, Lingyun; Li, Suyi; Feng, Tianyu; et al.

---

## 1. 论文概述

LegoDiffusion 是一个微服务化（micro-serving）的扩散工作流推理系统，将 Text-to-Image 工作流分解为松耦合的模型执行节点，实现独立管理和调度。

**核心贡献**：
- 提出微服务化扩散工作流架构
- 实现模型级独立扩展、共享和并行化
- 请求率提升 **3×**，突发流量容忍度提升 **8×**

---

## 2. 背景与问题

### 2.1 Text-to-Image 工作流

典型工作流包含多个模型：

```
[Prompt] → [Text Encoder] → [CLIP] → [Diffusion Model] → [VAE] → [Image]
              ↓                ↓            ↓              ↓
           多个模型         可选模型      核心模型        后处理
```

**模型示例**：

| 模型 | 功能 | 参数量 | 特点 |
|------|------|--------|------|
| T5 | 文本编码 | 4.8B | 轻量 |
| CLIP | 图文对齐 | 400M | 轻量 |
| UNet/DiT | 扩散去噪 | 2-8B | 计算密集 |
| VAE | 图像解码 | 300M | 带宽受限 |
| ControlNet | 条件控制 | 1-2B | 可选 |
| Upscaler | 超分辨率 | 500M | 可选 |

### 2.2 整体式部署的问题

**Monolithic 部署**：所有模型打包在一起

**问题**：
1. **资源浪费**：轻量模型和密集模型使用相同资源
2. **无法共享**：相同模型在不同工作流中重复部署
3. **扩展粗糙**：只能整体扩展，无法按需扩展单个模型

---

### L1: Inefficient Scaling via Full Replication (低效的全复制扩展)

**问题描述：**

单体服务将整个工作流作为扩展单元，强制执行粗粒度的复制，而不管哪个组件是实际的瓶颈。

**详细分析：**

1. **扩展粒度问题**
   - 在负载激增时，基础扩散模型通常是唯一的瓶颈
   - 标准管线中，完整工作流占用空间通常是基础模型的 **1.7× 到 4×**
   - 系统必须复制整个工作流，而非仅扩展瓶颈组件

2. **性能影响**
   - 使用 Diffusers 的单体复制：增加高达 **80%** 的加载延迟
   - 浪费高达 **75%** 的 GPU 内存（相比仅扩展瓶颈组件）
   - vLLM-Omni：扩展整个 Flux-Dev 管线增加高达 **70%** 延迟
   - SGLang-Diffusion：扩展整个 Flux-Dev 管线增加高达 **75%** 延迟

3. **关键数据**
   ```
   工作流扩展 vs 基础模型扩展的加载时间对比：
   - SD3: 4.0s (工作流) vs 0.4s (基础DM)
   - Flux: 5.2s (工作流) vs 2.9s (基础DM)
   ```

---

### L2: Inability to Share Common Models (无法共享通用模型)

**问题描述：**

单体服务强制工作流实例之间的严格隔离，阻止模型共享。

**详细分析：**

1. **生产工作负载特征**
   - 阿里巴巴的跟踪分析显示：流行骨干（如 SDXL、SD3、Flux-Dev）出现在几乎所有工作流中
   - 前 5 个 ControlNets 服务 **95%** 的生成请求
   - 模型流行度呈现高度倾斜分布

2. **资源浪费**
   - 每个工作流实例必须维护独立的模型副本
   - 模型大小：**2-24 GiB**（FP16精度）
   - 冗余导致：
     - 内存多路复用失效
     - GPU 内存消耗过高
     - GPU 利用率低
     - 负载不平衡

3. **模型共享收益**
   ```
   模型共享对请求延迟的影响：
   - SD3: 3.0s (无共享) vs 1.2s (有共享)
   - Flux: 14.7s (无共享) vs 1.4s (有共享)
   
   内存占用减少：高达 60%
   延迟减少：高达 40%
   ```

---

### L3: Runtime Inefficiency (运行时低效)

**问题描述：**

将工作流封装为不透明黑盒，消除了系统级对内部模型依赖、数据流和执行逻辑的可见性。

**详细分析：**

1. **可见性缺失**
   - 系统无法观察内部模型依赖关系
   - 无法优化数据流和执行逻辑
   - 强制执行严格的工作流级资源分配

2. **异构性问题**
   - 工作流内的模型表现出异构算术强度
   - 不同模型有不同的延迟-吞吐权衡
   - 静态工作流配置天生次优

3. **并行度限制**
   - 单体系统通常强制固定度的模型并行
   - 无法适应动态工作负载
   - 无法适应 GPU 可用性波动
   - 导致显著性能下降

4. **自适应并行收益**
   ```
   不同并行配置的延迟分布：
   - Parallelism=1: 一致较高延迟，错过并行加速机会
   - Parallelism=2: 当后续请求等待可用 GPU 对时引入排队，产生阶梯状 CDF 曲线
   - Adaptive: 自动并行化调整，平均请求服务加速 1.3× 和 1.2×
   ```

---

### L4: System Fragility and Maintenance Overhead (系统脆弱性和维护开销)

**问题描述：**

单体服务违反模块化系统原则，增加维护开销，导致系统脆弱性。

**详细分析：**

1. **系统脆弱性**
   - 独立组件的紧耦合导致系统脆弱
   - 单个子组件故障级联为整个工作流故障
   - 缺乏故障隔离机制

2. **维护困难**
   - 调试复杂化：开发者必须检查整个单体以识别根本原因
   - 更新单个组件需要整个工作流的全面验证和协调
   - 不必要地延长开发和部署周期

3. **微服务化优势**
   - 提供更清晰的故障边界
   - 更干净的更新路径
   - 可以隔离修改、验证和调试单个模型
   - 降低维护开销，使 bug 更容易定位

---

## 3. 核心技术

### 3.1 微服务化架构

**核心思想**：将每个模型视为独立的微服务

```
传统方式:
┌─────────────────────────────────────┐
│         Monolithic Pipeline         │
│  [T5] → [CLIP] → [DiT] → [VAE]    │
└─────────────────────────────────────┘

微服务方式:
┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐
│ T5  │→│CLIP │→│ DiT │→│ VAE │
│Svc  │  │Svc  │  │Svc  │  │Svc  │
└─────┘  └─────┘  └─────┘  └─────┘
```

**优势**：
1. **独立扩展**：根据负载单独扩展每个模型
2. **模型共享**：相同模型可被多个工作流复用
3. **异构部署**：不同模型可部署在不同 GPU

### 3.2 模型级调度

**调度策略**：

```python
class MicroServiceScheduler:
    def schedule(self, workflow):
        # 解析工作流依赖
        dag = parse_workflow(workflow)
        
        # 为每个模型选择实例
        for model in dag.topological_sort():
            instance = self.select_instance(model)
            instance.submit(model.task)
            
    def select_instance(self, model):
        # 选择负载最低的实例
        instances = self.get_instances(model.name)
        return min(instances, key=lambda x: x.queue_length)
```

### 3.3 自适应模型并行

**问题**：不同模型需要不同的并行度

**解决方案**：每个模型独立选择并行策略

```
T5: 1 GPU (轻量)
CLIP: 1 GPU (轻量)
DiT: 4 GPU (计算密集)
VAE: 1 GPU (带宽受限)
```

**动态调整**：
```python
def adjust_parallelism(model, load):
    current_gpus = model.assigned_gpus
    target_gpus = calculate_optimal_gpus(model, load)
    
    if target_gpus > current_gpus:
        # 扩展
        model.scale_out(target_gpus - current_gpus)
    elif target_gpus < current_gpus:
        # 缩减
        model.scale_in(current_gpus - target_gpus)
```

---

## 4. 系统实现

### 4.1 架构

```
┌─────────────────────────────────────────────────────┐
│              LegoDiffusion Controller                │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐│
│  │  Workflow     │  │   Model      │  │  Resource  ││
│  │  Parser       │  │  Registry    │  │  Manager   ││
│  └──────────────┘  └──────────────┘  └────────────┘│
└─────────────────────────────────────────────────────┘
         ↓                    ↓                  ↓
   ┌──────────────────────────────────────────────────┐
│              Service Mesh / Scheduler                │
└──────────────────────────────────────────────────┘
         ↓                    ↓                  ↓
   ┌──────────┐        ┌──────────┐        ┌──────────┐
   │ T5 Svc   │        │ DiT Svc  │        │ VAE Svc  │
   │ Pool     │        │ Pool     │        │ Pool     │
   │ [1,2,3]  │        │ [1,2,3,4]│        │ [1,2]    │
   └──────────┘        └──────────┘        └──────────┘
```

### 4.2 工作流解析

```python
class WorkflowParser:
    def parse(self, workflow_config):
        # 构建 DAG
        dag = DAG()
        
        for step in workflow_config.steps:
            node = DAGNode(
                model=step.model,
                inputs=step.inputs,
                outputs=step.outputs
            )
            dag.add_node(node)
            
        # 添加依赖边
        for dependency in workflow_config.dependencies:
            dag.add_edge(dependency.from_node, dependency.to_node)
            
        return dag
```

### 4.3 模型共享

**共享机制**：
```python
class ModelRegistry:
    def __init__(self):
        self.shared_models = {}
        
    def get_or_create(self, model_name, config):
        if model_name in self.shared_models:
            # 复用已部署的模型
            return self.shared_models[model_name]
        else:
            # 部署新模型
            instance = deploy_model(model_name, config)
            self.shared_models[model_name] = instance
            return instance
```

---

## 5. 实验结果

### 5.1 实验设置

- **工作流**：Stable Diffusion + ControlNet + Upscaler
- **硬件**：8× A100
- **负载**：真实 trace + 突发负载

### 5.2 性能对比

| 方法 | 最大请求率 (req/s) | 突发容忍 | 资源利用率 |
|------|-------------------|----------|-----------|
| Monolithic | 5 | 2× | 45% |
| Static Split | 8 | 3× | 55% |
| **LegoDiffusion** | **15** | **16×** | **75%** |

### 5.3 模型共享效果

| 场景 | 无共享 | 有共享 | 节省 |
|------|--------|--------|------|
| 2 工作流共享 T5 | 2 T5 实例 | 1 T5 实例 | 50% |
| 3 工作流共享 CLIP | 3 CLIP 实例 | 1 CLIP 实例 | 67% |

---

## 6. 优势与局限

### 优势

1. **灵活扩展**：按需扩展单个模型
2. **资源共享**：相同模型可跨工作流复用
3. **异构友好**：不同模型可部署在不同 GPU

### 局限

1. **通信开销**：模型间需要数据传输
2. **调度复杂**：需要管理多个模型池
3. **延迟增加**：微服务调用有额外开销

---

## 7. 与其他架构的对比

| 架构 | 粒度 | 扩展性 | 资源效率 |
|------|------|--------|----------|
| Monolithic | 工作流级 | 低 | 低 |
| Stage-level | 阶段级 | 中 | 中 |
| **Micro-serving** | **模型级** | **高** | **高** |

---

## 8. 关键术语

| 术语 | 解释 |
|------|------|
| Micro-serving | 微服务化，每个模型独立部署 |
| Workflow DAG | 工作流有向无环图 |
| Model Sharing | 模型共享，跨工作流复用 |
| Adaptive Parallelism | 自适应并行，按需调整并行度 |
| Service Mesh | 服务网格，管理微服务通信 |

---

## 9. 相关论文

### Diffusion Serving 相关

1. **Understanding Diffusion Model Serving in Production: A Top-Down Analysis of Workload, Scheduling, and Resource Efficiency**
   - **会议/期刊:** ACM SoCC 2025
   - **作者:** Yanying Lin, Shuaipeng Wu, Shutian Luo, Hong Xu, Haiying Shen, Chong Ma, Min Shen, Le Chen, Chengzhong Xu, Lin Qu, Kejiang Ye
   - **摘要:** 对生产环境中扩散模型服务的工作负载、调度和资源效率进行自顶向下分析
   - **链接:** [论文链接待补充]

2. **Katz: Efficient Workflow Serving for Diffusion Models with Many Adapters**
   - **会议/期刊:** USENIX ATC 2025
   - **作者:** Suyi Li, Lingyun Yang, Xiaoxiao Jiang, Hanfeng Lu, Zhipeng Di, Weiyi Lu, Jiawei Chen, Kan Liu, Yinghao Yu, Tao Lan, Guodong Yang, Lin Qu, Liping Zhang, Wei Wang
   - **摘要:** 针对具有多个适配器的扩散模型的高效工作流服务系统
   - **链接:** [论文链接待补充]

### LLM Serving 相关

3. **AlpaServe: Statistical Multiplexing with Model Parallelism for Deep Learning Serving**
   - **会议/期刊:** EuroSys 2023 / OSDI 2023
   - **作者:** Zhuang Wang, Li Lyna Zhang, Yinmin Zhong, Zhihao Jia, Mao Yang
   - **DOI:** 10.1145/3600206.3600213
   - **arXiv:** 2302.11665
   - **摘要:** 结合统计复用和模型并行的深度学习服务系统
   - **链接:** [arXiv](https://arxiv.org/abs/2302.11665) | [DOI](https://doi.org/10.1145/3600206.3600213)

4. **Teola: Towards End-to-End Optimization of LLM-based Applications with Ayo**
   - **会议/期刊:** ASPLOS 2025
   - **作者:** Xin Tan, Yimin Jiang, Yitao Yang, Hong Xu
   - **arXiv:** 2407.00326
   - **摘要:** 面向LLM应用的端到端优化系统，包含Ayo项目
   - **链接:** [arXiv](https://arxiv.org/abs/2407.00326) | [GitHub](https://github.com/NetX-lab/Ayo)
