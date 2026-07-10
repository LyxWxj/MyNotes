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
