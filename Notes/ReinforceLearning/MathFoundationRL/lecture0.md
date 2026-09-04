---
type: Note
related_to: "[[lecture1]]"
status: Active
---

# Overview of This Book

> [!info] 课程概览
> - 不仅是本课程的路线图，也是 RL 基础知识的全景图
> - 基础工具 + 算法与方法
> - 各部分的重要性与前后依赖关系

---

## Chapter 1: 基本概念

> [!note] Chapter 1 — 核心概念
> - 概念：**状态 (state)**、**动作 (action)**、**奖励 (reward)**、**回报 (return)**、**回合 (episode)**、**策略 (policy)** 等
> - Grid-world 示例（经典入门环境）
> - **马尔可夫决策过程 (MDP)** 的形式化定义
> - 这些基础概念将在后续章节中反复使用

> [!tip] 为什么从概念开始？
> RL 的所有算法都建立在这些基本概念之上。理解 state、action、reward 的数学定义，是后续学习 Bellman 方程、动态规划等工具的前提。

---

## Chapter 2: 状态价值与 Bellman 方程

> [!note] Chapter 2 — 状态价值
> - 引入核心概念：**状态价值函数 (state value)**
> $$
> V^\pi(s) = \mathbb{E}_\pi \left[ G_t \mid S_t = s \right]
> $$
> - 引入核心工具：**Bellman 方程**
> $$
> v_\pi = r_\pi + \gamma P_\pi v_\pi
> $$
> - **策略评估 (policy evaluation)**：给定策略 $\pi$，计算其状态价值
> - Bellman 方程是后续几乎所有算法的数学基础

> [!important] 关键理解
> Bellman 方程将价值函数分解为**即时奖励** + **折扣未来价值**，这是 RL 的核心递归结构。

---

## Chapter 3: 最优性与 Bellman 最优方程

> [!note] Chapter 3 — 最优策略
> - Bellman 方程的特殊形式
> - 引入两个关键概念：**最优策略 $\pi^*$** 和 **最优状态价值 $v_*$**
> - 核心工具：**Bellman 最优方程**
> $$
> v_*(s) = \max_a \sum_{s',r} p(s',r|s,a) \left[ r + \gamma v_*(s') \right]
> $$
> - 数学基础：
>   1) **不动点定理** — 保证 Bellman 最优方程有唯一解
>   2) **基本问题** — 如何求解这个方程
>   3) **求解算法** — 从理论到实践的桥梁
> - 最优性概念贯穿后续所有章节

> [!important] 与 Chapter 2 的关系
> Chapter 2 的 Bellman 方程是**给定策略下的评估**；Chapter 3 的 Bellman 最优方程是**寻找最优策略**。两者是 RL 的两大基石。

---

## Chapter 4: 动态规划

> [!note] Chapter 4 — 动态规划算法
> - 首批求解最优策略的算法
> - 三大算法：
>   1) **价值迭代 (Value Iteration)** — 直接迭代 Bellman 最优方程
>   2) **策略迭代 (Policy Iteration)** — 交替进行策略评估和策略改进
>   3) **截断策略迭代 (Truncated Policy Iteration)** — 两者的折中
> - **策略更新** 和 **价值更新** 的思想在后续算法中反复出现
> - 需要**环境模型**（已知状态转移概率和奖励函数）

> [!warning] 局限性
> 动态规划要求完整的环境模型（$p(s',r|s,a)$），这在实际问题中往往不可用。这正是后续章节引入无模型方法的动机。

---

## Chapter 5: 蒙特卡洛方法

> [!note] Chapter 5 — 从有模型到无模型
> - 填补空白：**如何在没有环境模型的情况下学习？**
> - 核心思想：用采样数据估计期望值（均值估计）
> $$
> \mathbb{E}\left[X\right] \approx \hat{x} = \frac{1}{n}\sum_{i=1}^n x_i
> $$
> - 首批**无模型 RL 算法**：
>   1) **MC Basic** — 最基础的蒙特卡洛方法
>   2) **MC Exploring Starts** — 保证每个状态-动作对被充分探索
>   3) **MC $\epsilon$-greedy** — 用 $\epsilon$-贪心策略平衡探索与利用

> [!tip] 核心突破
> 蒙特卡洛方法不需要知道环境的转移概率，只需要通过与环境交互采样来估计价值。这是 RL 从理论走向实际应用的关键一步。

---

## Chapter 6: 增量学习与随机梯度下降

> [!note] Chapter 6 — 从批量到增量
> - 填补空白：**如何从非增量方式过渡到增量方式？**
> - 核心问题：均值估计的增量更新
> - 关键算法：
>   1) **Robbins-Monro (RM) 算法** — 随机逼近的经典方法
>   2) **随机梯度下降 (SGD)** — 用单个样本近似梯度
>   3) **SGD、BGD、MBGD** 的对比：
>      - SGD：每次用 1 个样本更新
>      - BGD：每次用全部样本更新
>      - MBGD：每次用一个 mini-batch 更新
> - 增量学习和 SGD 是后续深度 RL 的基础

> [!important] 为什么需要增量方式？
> 批量方法需要存储所有数据，计算开销大；增量方法每收到一个样本就更新，更高效且适合在线学习场景。

---

## Chapter 7: 经典 TD 学习算法

> [!note] Chapter 7 — 时序差分学习
> - RL 中最经典的算法族
> - 核心算法：
>   1) **TD 学习状态价值** — 结合蒙特卡洛和动态规划的思想
>   2) **Sarsa** — TD 学习动作价值（on-policy）
>   3) **Q-learning** — TD 学习最优动作价值（off-policy）
> - **On-policy vs Off-policy** 的区别：
>   - On-policy（如 Sarsa）：学习的策略 = 行为策略
>   - Off-policy（如 Q-learning）：学习的策略 ≠ 行为策略
> - 统一视角：所有 TD 方法都可以看作 **Bellman 方程的采样近似**

> [!tip] Sarsa vs Q-learning
> - Sarsa 更新：$Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma Q(s',a') - Q(s,a) \right]$（用实际选择的 $a'$）
> - Q-learning 更新：$Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]$（用最优动作）

---

## Chapter 8: 函数近似与深度 RL

> [!note] Chapter 8 — 从表格到函数近似
> - 填补空白：**如何从表格表示过渡到函数表示？**
> - 当状态空间很大或连续时，表格方法不可行，需要函数近似
> - 核心方法：**价值函数近似 (VFA)**
> $$
> \min_\omega \; J(\omega) = \mathbb{E}\left[ \left( v_\pi(s) - \hat{v}(s, \omega) \right)^2 \right]
> $$
> - 算法：
>   1) 基于 VFA 的状态价值估计
>   2) **Sarsa with VFA**
>   3) **Q-learning with VFA**
>   4) **Deep Q-learning (DQN)** — 用神经网络作为函数近似器
> - 神经网络正式进入 RL 领域

> [!important] DQN 的突破
> DeepMind 2013 年的 DQN 论文证明了深度神经网络可以有效近似价值函数，在 Atari 游戏上达到人类水平，开启了深度 RL 的时代。

---

## Chapter 9: 策略梯度方法

> [!note] Chapter 9 — 从价值-based 到策略-based
> - 填补空白：**如何从价值-based 方法过渡到策略-based 方法？**
> - 核心思想：直接参数化策略 $\pi_\theta$，通过梯度上升优化
> - 优化指标：
> $$
> J(\theta) = \hat{v}_\pi, \quad \hat{r}_\pi
> $$
> - **策略梯度定理**：
> $$
> \nabla_\theta J(\theta) = \mathbb{E}\left[ \nabla_\theta \log \pi_\theta(A \mid S, \theta) \; q_\pi(S, A) \right]
> $$
> - 算法：**REINFORCE**
> $$
> \theta_{t+1} = \theta_t + \alpha \, \nabla_\theta \ln \pi(a_t \mid s_t, \theta_t) \, q_t(s_t, a_t)
> $$

> [!tip] 策略梯度的优势
> - 可以处理连续动作空间
> - 可以学习随机策略
> - 有更强的收敛性保证
> - 缺点：方差大，需要基线 (baseline) 技巧来降低方差

---

## Chapter 10: Actor-Critic 方法

> [!note] Chapter 10 — 结合价值与策略
> - 填补空白：**如何将策略-based 和价值-based 方法结合？**
> - 核心思想：同时维护**演员 (Actor)** 和 **评论家 (Critic)**
>   - Actor：策略网络 $\pi_\theta$
>   - Critic：价值网络 $q_w$ 或 $v_w$
> - 策略梯度的更新公式：
> $$
> \theta_{t+1} = \theta_t + \alpha \, \nabla_\theta \ln \pi(a_t \mid s_t, \theta_t) \, q_t(s_t, a_t)
> $$
> - 算法：
>   1) **QAC (Q-learning Actor-Critic)** — 最简单的 Actor-Critic
>   2) **A2C (Advantage Actor-Critic)** — 用优势函数降低方差
>   3) **Off-policy Actor-Critic** — 结合重要性采样 (importance sampling)
>   4) **DPG (Deterministic Policy Gradient)** — 确定性策略梯度

> [!important] Actor-Critic 的核心优势
> Actor-Critic 结合了策略梯度（处理连续动作、随机策略）和价值方法（低方差、样本高效）的优点，是现代深度 RL 算法（如 PPO、SAC）的基础框架。

---

> [!summary] 全书脉络
> ```mermaid
> graph LR
>     A[Ch1: 基本概念] --> B[Ch2: Bellman 方程]
>     B --> C[Ch3: Bellman 最优方程]
>     C --> D[Ch4: 动态规划]
>     D --> E[Ch5: 蒙特卡洛]
>     E --> F[Ch6: 增量学习/SGD]
>     F --> G[Ch7: TD 学习]
>     G --> H[Ch8: 函数近似/Deep RL]
>     H --> I[Ch9: 策略梯度]
>     I --> J[Ch10: Actor-Critic]
> ```
> 每一章都在填补前一章留下的空白，形成完整的学习路径。
