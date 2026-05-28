# 娄雨轩 Daily Report — 2026-05-10

关心怎么针对不同stage确定不同的硬件，突出多模态，
算法针对对模态做补充，
GPU和NPU的区别，突出异构的创新性

---

## 基于 Roofline 模型的多模态推理流水线异构硬件编排

基于多模态推理流水线各 stage 的算力特性、访存特征、显存占用三维度差异，通过 Roofline 模型驱动的三级硬件匹配机制实现最优硬件编排。

### 一、核心洞察：各 stage 的计算-访存特性存在本质差异

通过 Roofline 模型（以算术强度 AI = FLOPs / Bytes Accessed 为横轴、硬件峰值算力为上界的性能分析框架）对 DiT 文生图 pipeline 各 stage 进行分析，发现其计算模式截然不同：

**1. MMDiT Diffusion 生成 stage（计算密集型，Compute-Bound）**

MMDiT（Multi-Modal Diffusion Transformer）采用 Transformer 架构处理图文双模态 token 序列，其核心计算为大矩阵乘法（QKV 投影、Attention Score 矩阵乘、FFN 层的两层全连接）。以 20B 参数规模的 MMDiT 在 1024×1024 分辨率下推理为例：
- 单步去噪 FLOPs 约 40–60 GFLOPs，50 步总计约 2–3 TFLOPs；
- 隐藏维度 d=3072，序列长度 L≈4096（64×64 latent patch），矩阵乘法规模为 [L, d] × [d, d]，单次 GEMM 的算术强度可达 **150–500 FLOPs/byte**；
- 远超昇腾 NPU Da Vinci 架构 Cube Unit 的 Roofline 拐点（约 30–50 FLOPs/byte，FP16 精度）以及 NVIDIA GPU Tensor Core 的拐点（约 40–80 FLOPs/byte），处于 **计算密集区**。

这意味着 MMDiT stage 的性能瓶颈在于**矩阵乘法的峰值吞吐**（TFLOPS），而非内存带宽。该 stage 每步去噪需加载一次全部权重（~40GB FP16），但权重在 Transformer block 内被 QKV/Attention/FFN 三阶段复用，有效摊薄了访存开销。

**2. VAE-Decoder stage（访存密集型，Memory-Bandwidth-Bound）**

VAE Decoder 将低分辨率潜在表示（如 64×64×16 通道）逐级上采样至像素空间（如 1024×1024×3）。其计算结构以转置卷积（Transposed Convolution）和残差块（Residual Block）为主：
- 末级上采样阶段的特征图尺寸达 512×512×128 甚至更大，单层激活值读写量可达数百 MB；
- 转置卷积的算术强度仅约 2–10 FLOPs/byte——远低于硬件 Roofline 拐点，处于 访存密集区；
- 总 FLOPs 仅约 50–100 GFLOPs（远低于 MMDiT 的 2–3 TFLOPs），但因高分辨率特征图的反复读写，实际延迟占比可达 10%–20%。

这意味着 VAE-Decoder stage 的性能瓶颈在于内存带宽（TB/s），而非计算吞吐。选择高带宽硬件比选择高算力硬件更有效。

**3. Text Encoder / LLM 特征编码 stage（轻量计算型）**

Text Encoder（如 CLIP/T5）和 LLM 特征编码器（如 Qwen2.5-VL 7B）以自回归方式处理文本 token，单次推理 FLOPs 约 10–50 GFLOPs，算术强度中等（约 50–150 FLOPs/byte）。参数量 7B 对应约 14GB FP16 权重，但仅执行一次（非迭代），耗时占比不足 3%。

### 二、异构硬件算力特性对比：Da Vinci 架构 vs CUDA 架构

昇腾 NPU 与 NVIDIA GPU 在底层架构上存在本质差异，这直接决定了不同计算模式的效率：

| 维度 | 昇腾 NPU（Da Vinci 架构） | NVIDIA GPU（CUDA/Tensor Core 架构） |
|------|--------------------------|-------------------------------------|
| **核心计算单元** | Cube Unit（矩阵计算专用） + Vector Unit（向量/标量计算） | CUDA Core（通用并行） + Tensor Core（矩阵加速） |
| **矩阵乘法效率** | Cube Unit 针对 FP16 GEMM 深度优化，确定性执行时序，无 warp 调度开销 | Tensor Core 通过 warp 级协作执行 GEMM，需 CUDA 调度，灵活性更高 |
| **访存密集型效率** | Vector Unit 吞吐相对 Cube Unit 较低，对 element-wise/逐像素操作效率有限 | CUDA Core 通用性强，对不规则访存模式适应性好 |
| **峰值算力（FP16）** | 910B: ~320 TFLOPS; 910C: ~400–500+ TFLOPS | A100: 312 TFLOPS; H100: 990 TFLOPS |
| **HBM 带宽** | 910B: ~1.6 TB/s; 910C: ~2.0 TB/s | A100: 2.0 TB/s; H100: 3.35 TB/s |
| **算力/带宽比** | 910B: ~200 FLOPs/byte; 910C: ~250 FLOPs/byte | A100: ~156 FLOPs/byte; H100: ~295 FLOPs/byte |

关键差异解读：
- **昇腾 910C 的算力/带宽比（~250 FLOPs/byte）高于 A100（~156 FLOPs/byte）**，意味着 Da Vinci 架构在 Roofline 模型中更偏向"计算侧"——对于计算密集型任务（如 MMDiT 矩阵乘法），其 Cube Unit 能更充分地发挥算力优势，单位功耗下的矩阵计算效率更高；
- **NVIDIA H100 的 HBM 带宽（3.35 TB/s）显著高于 910C（~2.0 TB/s）**，在访存密集型任务上有天然优势，但受出口管制限制不可用于国内部署；
- **910B 的带宽（~1.6 TB/s）与 910C 的算力（~400+ TFLOPS）形成互补梯度**，适配不同计算模式的 stage。

### 三、三级硬件匹配机制

设有待部署的推理 stage 集合 $\mathcal{S} = \{s_1, s_2, \dots, s_n\}$（如 Text Encoder、LLM 特征编码、MMDiT Diffusion、VAE-Decoder），以及可用异构硬件池 $\mathcal{H} = \{h_1, h_2, \dots, h_m\}$（包含不同算力等级的 AI 加速器及通用处理器）。目标是求解 stage-to-hardware 映射 $\sigma: \mathcal{S} \to \mathcal{H}$，使系统吞吐量最大化。通过以下三级逐层递进的筛选与优化机制求解。

---

**第一级：显存可行性筛选（Feasibility Filter）**

**形式化定义：** 对每个 stage $s_i$，计算其峰值显存需求：

$$M_i^{peak} = M_i^{weight} + M_i^{act} + M_i^{cache}$$

其中 $M_i^{weight}$ 为模型权重占用（由参数量与精度决定），$M_i^{act}$ 为最大激活值占用（由输入分辨率与中间特征图尺寸决定），$M_i^{cache}$ 为调度器/缓存/中间变量占用。

**筛选条件：** 硬件 $h_j$ 对 stage $s_i$ 可行，当且仅当：

$$C_1:\quad M_j^{cap} \geq M_i^{peak} \cdot (1 + \delta)$$

其中 $M_j^{cap}$ 为硬件 $h_j$ 的显存容量，$\delta > 0$ 为安全余量系数（典型取 0.1–0.15，用于覆盖内存碎片化与框架开销）。

**输出：** 可行性矩阵 $\mathbf{F} \in \{0, 1\}^{n \times m}$，其中 $F_{ij} = 1$ 表示硬件 $h_j$ 满足 stage $s_i$ 的显存约束。不可行的 $(s_i, h_j)$ 组合在后续级别中被排除。

以多模态文生图 pipeline 为例：MMDiT stage 峰值显存约 44GB（28GB 权重 + 12GB 激活 + 4GB 缓存），仅大容量高端加速器可行；VAE-Decoder stage 仅需约 2.2GB，几乎所有加速器均可满足；Text Encoder stage 不足 1GB，CPU DRAM 即可容纳。

---

**第二级：计算模式匹配（Roofline-Driven Matching）**

第一级筛选后的可行硬件集合中，进一步根据各 stage 的计算瓶颈类型选择硬件。

**Roofline 模型回顾：** 硬件 $h_j$ 对 stage $s_i$ 的理论执行时间为：

$$T_{ij}^{comp} = \frac{FLOPS_i}{P_j^{peak}}, \qquad T_{ij}^{mem} = \frac{Bytes_i}{BW_j}$$

$$T_{ij} = \max\left(T_{ij}^{comp},\; T_{ij}^{mem}\right)$$

其中 $FLOPS_i$ 为 stage $s_i$ 的总浮点运算量，$P_j^{peak}$ 为硬件 $h_j$ 的峰值计算吞吐（如 FP16 TFLOPS），$Bytes_i$ 为 stage $s_i$ 的总内存访问量，$BW_j$ 为硬件 $h_j$ 的显存带宽。

stage $s_i$ 的算术强度定义为 $AI_i = FLOPS_i / Bytes_i$，硬件 $h_j$ 的 Roofline 拐点为 $AI_j^{ridge} = P_j^{peak} / BW_j$。当 $AI_i \gg AI_j^{ridge}$ 时，stage 处于计算密集区（$T_{ij} \approx T_{ij}^{comp}$）；当 $AI_i \ll AI_j^{ridge}$ 时，处于访存密集区（$T_{ij} \approx T_{ij}^{mem}$）。

**匹配准则：** 对第一级筛选出的每个可行组合 $(s_i, h_j)$，计算其**硬件利用效率**：

$$\eta_{ij} = \frac{\min\left(P_j^{peak} \cdot T_{ij},\; BW_j \cdot T_{ij}\right)}{\max\left(P_j^{peak} \cdot T_{ij},\; BW_j \cdot T_{ij}\right)} = \frac{\min\left(AI_i,\; AI_j^{ridge}\right)}{\max\left(AI_i,\; AI_j^{ridge}\right)}$$

$\eta_{ij} \in (0, 1]$ 衡量硬件 $h_j$ 的两种资源（算力与带宽）在 stage $s_i$ 上的均衡利用程度。$\eta_{ij} \to 1$ 表示两种资源均被充分利用；$\eta_{ij} \to 0$ 表示其中一种资源严重闲置。

**最优匹配求解：** 对每个 stage $s_i$，在可行性矩阵约束下求解：

$$h_j^* = \arg\min_{h_j:\; F_{ij}=1} \; T_{ij}$$

即在所有可行硬件中选择理论执行时间最短者。由于 $T_{ij} = \max(T_{ij}^{comp}, T_{ij}^{mem})$，该优化隐式地实现了以下匹配逻辑：

- **计算密集型 stage**（$AI_i \gg AI_j^{ridge}$）：$T_{ij} \approx FLOPS_i / P_j^{peak}$，最优解倾向于选择 $P_j^{peak}$ 最大的硬件——高算力硬件的计算资源被充分利用，而带宽资源虽然相对闲置，但由于瓶颈在计算侧，闲置带宽不构成浪费；
- **访存密集型 stage**（$AI_i \ll AI_j^{ridge}$）：$T_{ij} \approx Bytes_i / BW_j$，最优解倾向于选择 $BW_j$ 最大的硬件——高带宽硬件的内存通道被充分利用，而算力资源虽然相对闲置，但由于瓶颈在访存侧，闲置算力不构成浪费；
- **均衡型 stage**（$AI_i \approx AI_j^{ridge}$）：两种资源均接近满载，$\eta_{ij}$ 最大，硬件利用效率最高。

**关键约束——禁止跨区错配：** 为防止高端硬件被分配至访存密集型轻量 stage（导致算力严重浪费），引入**算力冗余比**约束：

$$C_2:\quad R_{ij} = \frac{P_j^{peak}}{FLOPS_i / T_i^{target}} \leq R_{max}$$

其中 $T_i^{target}$ 为 stage $s_i$ 的目标延迟上界，$R_{max}$ 为允许的最大冗余比（典型取 2.0–3.0）。当 $R_{ij} \gg 1$ 时，表示硬件算力远超 stage 需求，应降级至更低算力的硬件。

---

**第三级：流水线全局优化（Pipeline Throughput Maximization）**

前两级确定了每个 stage 的最优硬件选择后，第三级在全局视角下优化 stage 间的并行调度与资源分配，以系统吞吐量为最终优化目标。

**优化目标：** 设 stage 集合按依赖关系构成 DAG（有向无环图），每个 stage $s_i$ 分配至硬件 $h_{\sigma(i)}$，执行时间为 $T_i$。系统稳态吞吐量为：

$$\Theta = \min_{i \in \mathcal{S}} \frac{B_i}{T_i}$$

其中 $B_i$ 为 stage $s_i$ 的微批次大小。优化问题为：

$$\max_{\sigma, \{B_i\}} \; \Theta \qquad \text{s.t.} \quad \forall h_j:\; \sum_{i:\sigma(i)=j} M_i^{peak}(B_i) \leq M_j^{cap}$$

即在每块硬件的显存容量约束下，通过调整映射关系 $\sigma$ 和各 stage 的微批次大小 $B_i$，最大化系统吞吐量。

**关键优化策略：**

**（a）跨硬件流水线重叠：** 对于 DAG 中处于不同硬件上的相邻 stage $s_i \to s_k$（如 MMDiT → VAE-Decoder），当前请求的 stage $s_k$ 与下一请求的 stage $s_i$ 在不同硬件上并行执行。设 stage $s_i$ 的执行时间为 $T_i$，stage $s_k$ 的执行时间为 $T_k$，则流水线重叠后，系统的有效 stage 间等待时间为：

$$T_{wait} = \max(0,\; T_k - T_i)$$

当 $T_k \leq T_i$ 时（如 VAE-Decoder 延迟不超过 MMDiT 去噪延迟），stage $s_k$ 完全隐藏在 stage $s_i$ 的执行时间内，系统吞吐量仅由瓶颈 stage 决定。

**（b）同硬件多实例并行：** 对于计算密集型瓶颈 stage（如 MMDiT），在满足显存约束的条件下，将硬件 $h_j$ 划分为 $N_j$ 个独立实例（TP=1），每个实例处理不同请求。此时该 stage 的吞吐量提升为：

$$\Theta_i^{multi} = N_j \cdot \frac{B_i}{T_i}$$

实例数 $N_j$ 的上界由显存约束决定：$N_j \leq \lfloor M_j^{cap} / M_i^{peak}(B_i) \rfloor$。

**（c）负载均衡调度：** 当某 stage 存在多个同构实例时，采用轮询或最短队列优先策略分配请求，避免单实例过载导致的排队延迟。调度器维护各实例的忙闲状态，仅向空闲实例下发任务，确保：

$$\forall \text{ instance } k:\quad Q_k \leq Q_{max}$$

其中 $Q_k$ 为实例 $k$ 的当前队列深度，$Q_{max}$ 为允许的最大积压上限（用于背压控制）。

**（d）通信-计算重叠：** stage 间的数据传输（如 MMDiT 输出的 latent 张量传输至 VAE-Decoder）使用异步点对点通信。在传输进行的同时，发送端硬件可立即开始处理下一个请求的 stage $s_i$，实现通信延迟的完全隐藏。

---

**总结：** 三级匹配机制形成逐层递进的漏斗——第一级通过显存约束排除不可行组合，缩小搜索空间；第二级基于 Roofline 模型识别各 stage 的计算瓶颈类型，将 stage 匹配到在该瓶颈维度上具有最高利用率的硬件；第三级在全局视角下通过流水线重叠、多实例并行和负载均衡，将单 stage 的硬件效率优势转化为系统级吞吐量提升。

### 四、具体部署方案（以多模态文生图模型为例）

| Stage | 硬件分配 | 核心依据 |
|-------|---------|---------|
| **Text Encoder** | 通用处理器（CPU） | 参数量极小（~84M），仅执行一次文本预处理；CPU DRAM 容量充裕，无需占用 NPU 算力 |
| **LLM 特征编码** | 中端 NPU（如昇腾 310P，24GB） | 算术强度中等（~50–150 FLOPs/byte），接近 310P Roofline 拐点；14GB FP16 权重 + 2GB 激活值在 24GB 显存内充裕运行；释放高端 NPU 资源 |
| **MMDiT Diffusion 生成** | 高端 NPU（如昇腾 910C，64GB） | 算术强度极高（~150–500 FLOPs/byte），深度进入计算密集区；910C Cube Unit 的 ~400+ TFLOPS FP16 矩阵吞吐直接决定去噪延迟；44GB 峰值显存需 64GB 容量；Da Vinci 架构的确定性 GEMM 执行时序消除了 CUDA warp 调度的不确定性开销 |
| **VAE-Decoder** | 中高端 NPU（如昇腾 910B，32GB） | 算术强度极低（~2–10 FLOPs/byte），为典型访存密集型；910B 的 ~1.6 TB/s HBM 带宽可充分服务高分辨率特征图的反复读写；32GB 显存满足 2–4GB 的激活值缓存；与 910C 上的 MMDiT 形成硬件级流水线重叠，避免同一设备上的资源争抢 |

上述硬件型号及参数均为示例性说明，不构成具体限定。该分配方案的核心创新在于：**不是简单的"按显存大小分硬件"，而是基于 Roofline 模型识别各 stage 的计算瓶颈类型（计算密集 vs 访存密集），再匹配到在该瓶颈维度上具有最优性价比的异构硬件**——这使得昇腾 NPU Da Vinci 架构的 Cube Unit 矩阵计算优势与不同型号的带宽梯度得到最大化利用。
 