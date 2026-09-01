## 两个挑战（Two Challenges）

- **挑战 1：会话时长异构（Session duration heterogeneity）**。
- **挑战 2：用户需求的时间异构（Temporal user-demand heterogeneity）**。
![[intro_pictures_turboserve_hDzp.svg]]

## 三项基础能力（Three Things）

- **并发 Chunk 执行（Concurrent Chunk Execution）**：收集下一个块已准备好生成的会话，将同一 GPU 上的就绪会话组成合并的 chunk batch，调用一次模型，再把生成块和更新后的状态写回各自会话。
- **会话状态与生命周期（Session State and Lifecycle）**：执行（execution） -> 挂起（suspend，用户空闲） -> 终止（terminate，会话结束）。
- **GPU-CPU 状态卸载（GPU-CPU State Offloading）**：当会话变为空闲或必须释放 GPU 槽时，将持久状态从 GPU 复制到主机内存并释放 GPU 槽；重新活跃时，再把状态恢复到选定 GPU。这使会话生命周期与 GPU 驻留解耦，长期会话不必在空闲期间持续占用 GPU。

## 方法概览（Methods）

**联合迁移和自动扩缩容**: 评估一种结合会话**放置/迁移**和**GPU 自动扩缩容**的联合策略。该策略每 10 秒在各 GPU 间周期性地重新平衡活动会话，同时根据更新的 GPU 集在每次扩展决策周围额外执行重新平衡：向外扩展添加 GPU，然后重新分配活动会话以利用新容量，而向内扩展在释放不必要的 GPU 之前合并会话。

![[turboserve_overview_v3_hDzp.png]]

（i）会话放置，决定活动会话如何分配给 GPU；（ii）集群自动扩缩容，控制系统内配置的 GPU 数量。系统维护一个动态大小的 GPU 池，每个 GPU 能够在不超过每块延迟约束的情况下服务最多 K 个并发会话。

## 问题形式化（Formulation）

在每个事件 t，调度器解决一个**多目标优化问题**，旨在联合最小化 GPU 运营成本和每块服务延迟。

运营成本：$$\mathcal{C}(t)=c_{device}M(t)$$

延迟：

$$\mathcal{L(t)}=\max_{s_{i}\in\mathcal{S(t)}:\phi_{i}(t)\neq \emptyset}\mathcal{l}_{i}(t)$$

$\mathcal{l}_{i}(t)$表示来自会话 $s_i$ 的每 Chunk 延迟，它会随着每个 GPU 上会话共置的增加而增加。

最终我们要优化的问题是:

$$
\begin{aligned}
\arg \min_{M(t), \phi(t)} \quad & \mathcal{C}(t) + \lambda(t) \cdot \mathcal{L}(t), \\
\text{s.t.} \quad & |\{i : \phi_i(t) = g_j\}| \leq K, \quad \forall g_j \in \mathcal{G}(t), \\
& \alpha_i(t) = 1 \implies \phi_i(t) \neq \emptyset, \quad \forall s_i \in \mathcal{S}(t).
\end{aligned}
$$

第一个约束强制每个 GPU 的容量限制，确保没有 GPU 服务超过 K 个会话。第二个约束通过要求任何接收用户输入的会话必须在 GPU 上积极执行来保证服务响应能力。

## 优化方法（Optimization Method）

优化的方法由两个紧密耦合的组件组成：（i）**放置控制器**，更新分配 $\phi(t)$ 并在每个事件 t 提供负载反馈；（ii）**自动扩缩容控制器**，根据此反馈调整 GPU 预算 $M(t)$。它们共同形成一个闭环控制系统，将系统调节到期望的操作点。

放置控制器在每个事件 t 运行，并确定会话到当前配置的 GPU G(t) 的分配 $\phi(t)$。给定固定的 GPU 预算 M(t)，它近似求解放置优化问题：

$$
\mathcal{L}^*(M, t) = \arg \min_{φ(t) \text{feasible under}M(t)} \mathcal{L}(t)
$$

需要分配的会话集合定义为:

$$ U(t) = \{s_i \in S(t) : \alpha_i(t) = 1 \land \phi_i^{-}(t) = \emptyset\}. $$

对于每个 $s_i \in U(t)$，控制器评估所有可行的$GPU g_j ∈ \mathcal{G}(t)$，并选择能够最小化结果瓶颈延迟 L(t) 的分配。当多个分配产生相似的 L(t) 时，控制器使用固定的平局打破规则（例如，偏好负载较轻的 GPU）在可行 GPU 中进行选择。

**迁移感知的最小 - 最大重新平衡。** 在会话分配之后，控制器在每个事件 t 执行**迁移感知的最小 - 最大重新平衡**以减少瓶颈延迟 L(t)。

令 $n_j(t) = |\{i : \phi_i(t) = g_j\}|$ 表示分配给 GPU $g_j$ 的会话数，令 $g_{\max}(t) \in \arg\max_{g_j \in G(t)} \hat{\ell}_j(n_j(t))$ 表示达到最大每块延迟的 GPU。控制器考虑将会话 $s_i$（当前分配给 $g_{\max}(t)$）迁移到目标 GPU $g_{j'} \in G(t) \setminus \{g_{\max}(t)\}$，构建结果放置 $\phi'(t)$，并评估新的瓶颈延迟 $L'(t)$。迁移成本 $\kappa_i(t)$ 捕捉重新分配 $s_i$ 的开销，并使用 $\alpha$-$\beta$ 模型建模。每个候选移动的收益通过增益函数量化：

$\Gamma_{i,j'}(t) = L(t) - L'(t) - \eta \cdot \kappa_i(t)$ (4)

其中 $\eta > 0$ 控制延迟减少和迁移开销之间的权衡³。在每次迭代中，控制器选择使 $\Gamma_{i,j'}(t)$ 最大的移动 $(i_*, j'_*) = \arg\max_{i, j'} \Gamma_{i,j'}(t)$，并在 $\Gamma_{i_*, j'_*}(t) > 0$ 时应用它，然后重复直到没有候选移动产生正增益。

### 1. 为什么流式视频生成需要专门的服务系统

论文将它与普通离线视频生成和 LLM serving 区分开：

- 离线生成通常关心一次请求的总完成时间；流式生成则要求用户持续收到 chunk，因此更关心每个 chunk 的间隔和最坏值。
- LLM 的 KV cache 通常跟随一个较短请求；视频会话可能持续很久，并在 active/idle 之间反复切换，cache 和 prompt context 需要跨多个 chunk 保存。
- 用户在 idle 期间不需要 GPU 计算，但不能丢弃状态；下次 active 时必须从上次 frame 继续。

因此，单纯把所有会话固定在 GPU 上会造成两类问题：低峰期 GPU 空转，高峰期某些 GPU 上会话过多，导致这些会话的 chunk 延迟一起变差。

### 2. TurboServe 的整体闭环

论文的 workload detector 观察 arrival、departure、active 和 idle 事件，并在滑动窗口中计算近期需求和需求波动；随后四个模块形成闭环：

1. **工作负载检测器**：从事件流提取当前活跃数和波动性。
2. **放置控制器**：决定每个 session 在哪个 GPU，或者暂时挂起。
3. **自动扩缩容控制器**：根据放置后的负载反馈决定 GPU budget。
4. **会话管理器**：执行 chunk、挂起/恢复、迁移和终止，并把运行时测量反馈给控制器。

关键点是两个控制器不是独立的：放置决定每张 GPU 的真实负载，负载又决定是否扩缩容；扩缩容改变可用 GPU 集合后，还必须重新放置会话。

### 3. 两个控制参数怎样影响行为

论文中的 $\lambda(t)$ 是延迟相对于成本的权重，$\hat{\rho}(t)$ 是期望的 GPU 利用率：

- 工作负载变化剧烈时，提高 $\lambda$、降低 $\hat{\rho}$，多留一些空闲容量，避免新会话到来后立即超过延迟目标。
- 工作负载稳定时，降低 $\lambda$、提高 $\hat{\rho}$，让 GPU 多承载一些会话以降低成本。

论文把近期 activation volatility 分成少量等级，并为每个等级离线选择较合适的控制参数。这是一个控制策略，而不是重新求解完整的整数规划。

## 优化方法的直观解释

原公式可以先不看成“求一个很难的全局最优解”，而看成三个问题：

### 问题 A：要开几张 GPU？

假设当前有 $N$ 个活跃会话，一张 GPU 最多安全服务 $K$ 个会话，目标利用率是 $\hat{\rho}$。最简单的容量估计是：

$$
M_{tar}=\left\lceil\frac{N}{K\hat{\rho}}\right\rceil
$$

例如 $N=7$、$K=4$、$\hat{\rho}=0.9$，则需要 $\lceil 7/3.6\rceil=2$ 张 GPU。这里不是要求每张 GPU 恰好相同，而是保留约 10% 的 headroom。

### 问题 B：每个会话放哪张 GPU？

对新会话，尝试每一张尚未满的 GPU，选择放进去后最忙 GPU 的预测延迟最小的位置。对已有会话，先保留原位置，避免每个事件都全量搬迁。

### 问题 C：一次迁移值不值得？

把 session 从源 GPU 搬到目标 GPU 后，重新估算最忙 GPU 的延迟。如果延迟下降大于迁移成本的折算值，就执行迁移：

$$
\text{收益}=\text{迁移前瓶颈延迟}-\text{迁移后瓶颈延迟}-\eta\times\text{迁移成本}
$$

这就是 min-max：每轮只盯住当前最差的 GPU，并尝试把其中一个会话搬走，而不是枚举所有可能的全局排列。

## 代码中:

### 第一步：计算 GPU budget

入口是 turboserve/scheduling/scheduler.py 的 TurboServeScheduler.decide()，第一步调用 _autoscale_budget()：

1. 统计 active session 数 active_count；
2. 取 runtime 和 scheduler 中较小的 capacity_per_gpu；
3. 计算 hard_budget = ceil(active_count / capacity)；
4. 再计算 target_budget = ceil(active_count / (capacity * target_utilization))；
5. 取两者较大值，并限制在 min_gpus 和 max_gpus 内。

如果目标 budget 比当前 budget 大，返回 scale_out；如果更小，不会立即缩容，而是记录一个 scale-in deadline。当前配置 scale_in_hold_s=5 秒，5 秒内需求回升就取消缩容。

注意：当前的 autoscaling 是按 active_count 直接计算，代码中没有论文所说的滑窗 volatility 分类，也没有显式更新 $\lambda(t)$。

### 第二步：在固定 budget 下放置

_place_at_budget() 先建立 gpu_id -> session 列表：

1. 当前 GPU 仍在新 budget 范围内且没有超容量的 session，继续保留；
2. 新到达、刚从 idle 恢复或原 GPU 已不可用的 session，放入 needs_placement；
3. _least_load_place() 按负载最小、GPU id 最小的规则依次放置；
4. 放不下的 active session 保持 placement=None，并在 metadata 中计数 unplaced_active。

这一步对应论文里的 $\phi(t)$，但代码采用的是确定性的 least-load admission，而不是通用求解器。

### 第三步：模拟瓶颈延迟

代码不直接在调度线程上运行一次真实扩散，而是用 LatencyModel.sim_session_latency_ms() 做快速估计。估计值由以下因素组成：

$$
\text{latency}=
\text{base compute}
+\text{resolution/frame/prompt factor}
+\text{linear load penalty}
+\text{quadratic load penalty}
$$

当前配置中 base_chunk_latency_ms=180、load_penalty_ms=55、quadratic_load_penalty_ms=18。会话越多，线性排队项越大；接近容量上限时，二次项进一步放大压力。分辨率和帧数不同的 session 也可以得到不同的基础延迟。

### 第四步：执行 min-max rebalancing

_rebalance_minmax() 的代码流程可以简化成：

    重复若干轮：
        找出预测瓶颈 GPU source_gpu
        枚举 source_gpu 上的每个 session
        枚举其他未满的 target_gpu
        模拟搬迁后的 loads 和新瓶颈延迟
        gain = old_bottleneck - new_bottleneck
               - migration_eta * migration_cost
        选择 gain 最大的候选
        gain 超过 min_gain_ms 才真正更新 placement

代码还有几个工程上的 tie-break：优先降低负载 spread 和 variance，其次偏好状态更小的 session、较小的 GPU id 和 session id。这样相同收益时结果稳定、可复现。

当前 runtime_8gpu_process_nccl.yaml 的关键值是：

- 每 GPU 容量 4 个 session；
- target utilization 0.9；
- migration_eta 0.0；
- min_migration_gain_ms 5.0；
- 单次 scheduler 最多尝试 128 轮 rebalancing；
- 单个 replay cycle 实际最多执行 12 次物理迁移。

因此当前 demo 中迁移成本主要用于排序和记录，默认 eta=0 不会从 gain 中扣除迁移成本；这和论文公式允许 $\eta>0$ 的一般形式不同。

## 论文算法与当前代码的差异

这是理解结果时需要特别注意的地方：

1. **论文 Algorithm 1 的顺序**是先用旧 budget placement，得到 $\rho_{max}$，再决定 scale；scale-in 前重新合并，scale-out 后用新增 GPU 重新分配。
2. **当前 decide() 的顺序**是先 _autoscale_budget()，再以新 budget 调用一次 _place_at_budget()。它是一个简化实现，不是论文伪代码的逐行复刻。
3. **论文的负载反馈**以 placement 后的 $\rho_{max}$ 驱动控制；当前代码的 target budget 主要由 active_count 和固定 target_utilization 决定。
4. **论文的并发 chunk 执行**描述同一 GPU 将多个 ready session 合并成一个 batch；当前 LongLiveSessionEngine.step_one_block() 一次处理一个 session，worker 内部由 LocalRoundRobinStepScheduler 轮询。
5. **论文部署段落**描述 RDMA/NIXL 风格的 GPU 状态读取；当前仓库的 MultiGPUCoordinator 使用 process-isolated worker 和 process-NCCL P2P，迁移前会暂停源/目标 scheduler，在 chunk 完成后传输状态。

这些差异不影响 demo 验证调度思想，但不能把 demo 的 trace replay 结果直接当成论文生产部署和 batching 性能的复现。

## 迁移、挂起和状态恢复的简单流程

当 placement 从 GPU 0 变成 GPU 1 时，runtime 不会只修改一个整数：

1. coordinator 暂停源和目标 worker 的调度；
2. 源 worker 对 session 发起 suspend，等待正在执行的 block 完成；
3. 导出 session metadata、KV cache、cross-attention cache、output、frame 指针和 RNG 状态；
4. process_worker 通过 NCCL P2P 发送 tensor leaves；
5. 目标 worker 根据 skeleton/manifest 重建 SessionState，并恢复 packed cache view；
6. 删除源端 session，更新 session_to_worker；
7. 原 session 如果迁移前是 execution，则在目标 worker resume。

cache_pack.py 将分层、分散的 cache 复制到连续 buffer，再通过 view 重建原来的层级结构。这样传输的是少量大 buffer，而不是大量小 tensor。

## 论文实验中新增的结论

论文的 baseline 包括固定 GPU、round-robin 的 TurboServe_base，以及按 GPU 负载或显存的贪心放置。作者报告：

- TurboServe 最坏 chunk 延迟平均降低 37.5%，最高降低 51.6%；
- 在相同延迟约束下，GPU operating cost 平均降低 37.2%，最高降低 49.0%；
- 去掉 migration 后成本平均增加 15.0%；去掉 autoscaling 后平均增加 42.9%；
- 迁移端到端开销约 23--30 ms，占 chunk latency 的 2%--3%；
- 4--256 GPU 的调度实验中，64 GPU 以内调度时间低于 15 ms，相对 exhaustive oracle 的平均差距为 3.6%。

这些是论文作者基于生产 trace 的报告值，不是当前仓库的本地复现结果。

## 阅读代码时的几个问题

### 问题 1：为什么不直接做全局最优？

session 放置是组合优化，GPU 和 session 数增加后，枚举所有映射会迅速变慢。代码采用“保留旧 placement + least-load 放置 + 瓶颈 GPU 局部搜索”，用很小的调度时间换取接近均衡的结果。

### 问题 2：为什么迁移不能越多越好？

迁移本身需要 quiesce、分配目标 buffer、NCCL copy 和 owner 更新；如果只看当前延迟，会出现来回搬迁。min_gain_ms、migration_eta、单周期迁移上限和 scale-in hold 都是在抑制这种控制震荡。

### 问题 3：为什么 idle session 可以不占 GPU？

生成状态被保存在 SessionState 中，idle 时只需要保留 host/device 上的状态，不需要持续执行 denoise。恢复时重新把状态放回某个 worker，因此 GPU 资源可以给真正 active 的 session。

### 问题 4：当前代码是否已经实现论文所有能力？

没有。它已经实现可运行的 trace replay、预算计算、placement、局部迁移、session 生命周期和 process-NCCL 状态传输；但跨 session 合并 batch、生产集群物理 GPU 申请/回收、volatility 驱动的动态 $\lambda/\hat{\rho}$ 仍是论文设计或简化部分。

## 小结

把复杂数学翻译成代码，就是：

1. 用 active session 数估算需要几张 GPU；
2. 把新会话放到当前最空的可用 GPU；
3. 找出最慢的 GPU，试着搬走一个 session；
4. 只有延迟改善足够大才执行迁移；
5. idle 就挂起，active 再恢复；
6. 每个 trace 事件重复这个过程。

论文的贡献在于把这些动作组织成一个面向流式视频 session 的闭环；仓库的贡献则是提供了一个不依赖真实生产集群即可观察该闭环的实现骨架。
