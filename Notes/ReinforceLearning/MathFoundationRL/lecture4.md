---
type: Note
related_to: "[[lecture3]]"
status: Active
---

# Lecture 4: Value Iteration、Policy Iteration 与 Truncated Iteration

> [!abstract] 本章主线
> 三种方法都在已知 MDP 模型 $p(r\mid s,a),p(s'\mid s,a)$ 的前提下，通过“价值更新 ↔ 策略更新”寻找最优策略。它们的核心区别是：**每轮对当前策略的价值评估做多少次迭代**。

## 预备：两个算子

对任意价值向量 $v$，定义动作的一步前瞻值

$$
q_v(s,a)=\sum_r p(r\mid s,a)r+\gamma\sum_{s'}p(s'\mid s,a)v(s').
$$

对每个状态选择最大动作得到贪心策略：

$$
a_v(s)\in\arg\max_a q_v(s,a).
$$

固定策略 $\pi$ 时，Bellman 算子为

$$
T_\pi v=r_\pi+\gamma P_\pi v;
$$

最优 Bellman 算子为

$$
Tv=\max_a q_v(\cdot,a)=\max_{\pi}(r_\pi+\gamma P_\pi v).
$$

最优价值满足 $v^*=Tv$（Lecture 3）。

## 1. Value iteration（值迭代）

从任意初始价值 $v_0$ 出发，每轮只做一次 Bellman 最优算子更新：

$$
v_{k+1}=Tv_k=\max_{\pi}(r_\pi+\gamma P_\pi v_k).
$$

实现上可拆成两步：

1. **策略更新（policy update）**：用 $v_k$ 计算 $q_k(s,a)$，选择贪心策略 $\pi_{k+1}$。
2. **价值更新（value update）**：$v_{k+1}=r_{\pi_{k+1}}+\gamma P_{\pi_{k+1}}v_k$，等价于 $v_{k+1}(s)=\max_a q_k(s,a)$。

由于 $T$ 是 $γ$-压缩映射，$v_k\to v^*$；最终从 $v^*$ 取贪心策略即可得到最优策略。

> [!note] 中间量的含义
> 有限步得到的 $v_k$ 通常**不是某条策略的状态价值**，因为它未必满足 $v_k=r_\pi+\gamma P_\pi v_k$；它只是逼近 $v^*$ 的迭代量。

## 2. Policy iteration（策略迭代）

从任意初始策略 $\pi_0$ 出发，每轮包含“完整评估 + 改进”：

### Policy evaluation（策略评估）

求当前策略的真实状态价值：

$$
v_{\pi_k}=r_{\pi_k}+\gamma P_{\pi_k}v_{\pi_k}.
$$

理论上可用 $(I-\gamma P_{\pi_k})^{-1}r_{\pi_k}$，实践中通常迭代

$$
v_{\pi_k}^{(j+1)}=r_{\pi_k}+\gamma P_{\pi_k}v_{\pi_k}^{(j)}
$$

直到收敛（或足够精确）。

### Policy improvement（策略改进）

对 $v_{\pi_k}$ 做一步前瞻并贪心选择：

$$
\pi_{k+1}=\arg\max_\pi(r_\pi+\gamma P_\pi v_{\pi_k}).
$$

策略改进定理保证 $v_{\pi_{k+1}}\ge v_{\pi_k}$（逐状态成立）；有限 MDP 中策略不断改进，最终达到最优策略。

> [!note] 与值迭代的关键差异
> 策略迭代中的 $v_{\pi_k}$ 是当前策略的真实状态价值；值迭代中的 $v_k$ 一般只是最优算子的中间近似。

## 3. Truncated policy iteration（截断策略迭代）

它保留策略迭代的“评估 → 改进”结构，但策略评估只执行有限的 $m$ 次迭代：

$$
v_k^{(0)}=v_{k-1},\qquad
v_k^{(j+1)}=r_{\pi_k}+\gamma P_{\pi_k}v_k^{(j)},\quad j=0,\ldots,m-1,
$$

然后令 $v_k=v_k^{(m)}$，再依据 $v_k$ 做贪心策略改进。

“截断”指不把策略评估一直运行到 $j=\infty$，所以有限 $m$ 时的 $v_k$ 只是 $v_{\pi_k}$ 的近似，不一定是真实状态价值。通常 $m$ 取少量迭代：比值迭代每轮多获得一些策略评估信息，又避免策略迭代的完整评估成本。

## 三者的统一关系

| 方法 | 每轮价值步骤 | 策略步骤 | 价值量是否为真实 $v_\pi$ | 典型特征 |
| --- | --- | --- | --- | --- |
| Value iteration | 对最优算子做 **1 次**更新 | 同轮取贪心策略 | 通常不是 | 单轮便宜，收敛可能较慢 |
| Truncated policy iteration | 对固定 $\pi_k$ 做 **有限 $m$ 次**评估 | 评估后贪心改进 | 通常不是（$m<\infty$） | 速度与单轮成本折中 |
| Policy iteration | 评估固定 $\pi_k$ 至收敛（$m\to\infty$） | 评估后贪心改进 | 是 | 外层轮数少，但单轮昂贵 |

在从相同初值开始比较时：

$$
\text{Value iteration }(m=1)\quad\subset\quad
\text{Truncated PI }(1<m<\infty)\quad\subset\quad
\text{Policy iteration }(m=\infty).
$$

因此，截断策略迭代不是完全不同的第四种思想，而是连接两端的统一框架。三者共同体现 **Generalized Policy Iteration（广义策略迭代，GPI）**：价值估计与策略改进交替进行。它们都依赖已知环境模型，属于动态规划方法；后续无模型强化学习算法通常保留这一“评估—改进”骨架，只是用采样数据替代精确模型计算。

## 一句话记忆

- **Value iteration**：不等当前策略评估收敛，每轮只向最优价值推进一步。
- **Policy iteration**：先把当前策略评估到（近似）完全准确，再改进策略。
- **Truncated iteration**：评估做几步就停，在“每轮便宜”和“外层收敛快”之间折中。

