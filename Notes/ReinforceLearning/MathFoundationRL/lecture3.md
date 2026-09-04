---
type: Note
related_to: "[[lecture4]]"
status: Active
---

# Lecture 3: 最优状态值与 Bellman 最优方程

> [!abstract] 本章主线
> Lecture 2 的 Bellman 方程回答的是：**给定策略 $\pi$ 后，它的状态价值是多少？**
> 本章进一步回答：**在所有策略中，哪个策略最好，以及如何求它？** 核心工具是 Bellman Optimality Equation（Bellman 最优方程，BOE）。

## 从策略评估到策略改进

对于给定策略 $\pi$，先用其状态价值计算每个动作的价值：

$$
q_\pi(s,a)=\sum_r p(r\mid s,a)r+\gamma\sum_{s'}p(s'\mid s,a)v_\pi(s').
$$

它由两部分组成：执行 $a$ 的**期望即时奖励**，以及到达下一状态后、继续遵从 $\pi$ 的**折扣未来价值**。若某个动作的 $q_\pi(s,a)$ 更大，改为选择它就有机会改善策略。最优方程将这个“选择最优动作”的步骤直接写入价值方程。

---

## 最优策略与最优状态值

### 策略优劣的定义

比较两条策略 $\pi_1,\pi_2$ 时，若对**每一个状态**都有

$$
v_{\pi_1}(s)\ge v_{\pi_2}(s),\qquad \forall s\in\mathcal S,
$$

则称 $\pi_1$ 不差于 $\pi_2$。这里是逐元素（statewise）比较，而不是只比较某个起点或状态分布下的平均回报。

> [!note] 最优策略（optimal policy）
> 策略 $\pi^*$ 是最优的，当且仅当它对任意其他策略 $\pi$ 都满足
>
> $$
> v_{\pi^*}(s)\ge v_\pi(s),\qquad \forall s\in\mathcal S.
> $$
>
> 此时 $v^*(s)\triangleq v_{\pi^*}(s)$ 称为**最优状态值**。

因此，$v^*(s)$ 也可理解为从 $s$ 出发、允许在所有可选策略中挑选时能达到的最大期望回报：

$$
v^*(s)=\max_\pi v_\pi(s).
$$

> [!important] 唯一性要分开说
> - 最优状态值 $v^*$ **唯一**。
> - 最优策略 $\pi^*$ **不一定唯一**：若多个动作有同样大的最优动作价值，选择任一个，或在它们之间随机化，都可能最优。
> - 但总存在一条**确定性贪心最优策略**。

---

## Bellman Equation 回顾：固定策略 $\pi$

Bellman 方程中策略已给定，未知量只有 $v_\pi$。

### 元素形式

定义策略下的期望即时奖励和状态转移概率：

$$
r_\pi(s)\triangleq\sum_a\pi(a\mid s)\sum_r p(r\mid s,a)r,
\qquad
p_\pi(s'\mid s)\triangleq\sum_a\pi(a\mid s)p(s'\mid s,a).
$$

则

$$
\boxed{
v_\pi(s)=r_\pi(s)+\gamma\sum_{s'}p_\pi(s'\mid s)v_\pi(s')
}
$$

等价地，将动作的随机选择显式保留：

$$
v_\pi(s)=\sum_a\pi(a\mid s)
\left[
\sum_r p(r\mid s,a)r+\gamma\sum_{s'}p(s'\mid s,a)v_\pi(s')
\right].
$$

### 矩阵形式

若 $|\mathcal S|=n$，令 $v_\pi,r_\pi\in\mathbb R^n$，并令 $[P_\pi]_{ij}=p_\pi(s_j\mid s_i)$。则

$$
\boxed{v_\pi=r_\pi+\gamma P_\pi v_\pi.}
$$

这是**线性**方程：策略固定，所以 $r_\pi,P_\pi$ 都是已知量。若模型已知，可写出闭式解

$$
v_\pi=(I-\gamma P_\pi)^{-1}r_\pi.
$$

实际中通常避免显式求逆，而使用迭代策略评估 $v_{k+1}=r_\pi+\gamma P_\pi v_k$。

---

## Bellman Optimality Equation（BOE）

最优时不再对给定 $\pi$ 做期望，而是在每个状态选择使长期回报最大的动作。

### 元素形式

先以最优状态值定义最优动作价值：

$$
q^*(s,a)
=\sum_r p(r\mid s,a)r
+\gamma\sum_{s'}p(s'\mid s,a)v^*(s').
$$

Bellman 最优方程为

$$
\boxed{
v^*(s)=\max_{a\in\mathcal A}
\left[
\sum_r p(r\mid s,a)r
+\gamma\sum_{s'}p(s'\mid s,a)v^*(s')
\right]
=\max_a q^*(s,a).
}
$$

也可先在策略空间上写最大化：

$$
v^*(s)=\max_{\pi(\cdot\mid s)}\sum_a\pi(a\mid s)q^*(s,a).
$$

因为 $\pi(\cdot\mid s)$ 是概率分布，右侧是各 $q^*(s,a)$ 的加权平均，最大值必能通过将全部概率放在最大 $q^*$ 的动作上取得。这就是确定性贪心策略存在的原因。

### 矩阵形式

对每个候选策略 $\pi$，仍定义 $r_\pi,P_\pi$。将所有状态的方程合并：

$$
\boxed{
v^*=\max_{\pi\in\Pi}\left(r_\pi+\gamma P_\pi v^*\right).
}
$$

此处 $\max$ 是**逐元素最大值**：第 $s$ 个分量只在状态 $s$ 的动作/策略选择上取最大。定义 Bellman 最优算子

$$
(Tv)(s)\triangleq\max_a\left[
\sum_r p(r\mid s,a)r+\gamma\sum_{s'}p(s'\mid s,a)v(s')
\right],
$$

则 BOE 简洁地写成固定点方程：

$$
\boxed{v^*=T(v^*).}
$$

> [!warning] 与普通 Bellman 方程的关键区别
> - 普通方程：$\pi$ 固定，$v_\pi=r_\pi+\gamma P_\pi v_\pi$，是线性的策略评估问题。
> - 最优方程：每个状态包含 $\max$，$v^*=T(v^*)$，一般是非线性的控制问题。

---

## 如何从 BOE 求解最优策略

### 1. 先求最优状态值：值迭代

从任意初始向量 $v_0$ 出发，反复应用 Bellman 最优算子：

$$
\boxed{
v_{k+1}(s)=\max_a\left[
\sum_r p(r\mid s,a)r+\gamma\sum_{s'}p(s'\mid s,a)v_k(s')
\right].
}
$$

矩阵记号为

$$
v_{k+1}=\max_{\pi\in\Pi}(r_\pi+\gamma P_\pi v_k).
$$

停止时可用 $\lVert v_{k+1}-v_k\rVert_\infty$ 小于阈值作为准则。该过程称为**值迭代**；下一章会给出其算法实现细节。

### 2. 再从 $v^*$ 提取策略：贪心

计算每个状态下的 $q^*(s,a)$，然后选择

$$
a^*(s)\in\arg\max_a q^*(s,a).
$$

一条确定性最优策略可写为

$$
\pi^*(a\mid s)=
\begin{cases}
1,& a=a^*(s),\\
0,& \text{otherwise}.
\end{cases}
$$

若 $\arg\max$ 中有多个动作，任选一个可得到确定性最优策略；在这些并列最优动作之间分配概率，也可以得到随机最优策略。

---

## 存在性与唯一性：固定点与压缩映射

本节只介绍证明所依赖的两个概念，而不展开完整证明。

### Fixed Point（固定点）

对于映射 $f:\mathbb R^n\to\mathbb R^n$，如果某个 $x^*$ 满足

$$
f(x^*)=x^*,
$$

则 $x^*$ 是 $f$ 的固定点。直觉上，输入 $x^*$ 后映射不再改变它。BOE 的 $v^*=T(v^*)$ 正是在说：最优状态值是 Bellman 最优算子的固定点。

### Contraction Mapping（压缩映射）

若存在常数 $c\in[0,1)$，使任意 $x,y$ 都满足

$$
\lVert f(x)-f(y)\rVert\le c\lVert x-y\rVert,
$$

则称 $f$ 为压缩映射。它会把任意两点之间的距离至少按固定比例缩小。

对于有限折扣 MDP，取最大范数 $\lVert x\rVert_\infty=\max_i|x_i|$，Bellman 最优算子满足

$$
\lVert T(v)-T(w)\rVert_\infty
\le\gamma\lVert v-w\rVert_\infty,
\qquad 0<\gamma<1.
$$

折扣因子 $\gamma$ 正是压缩系数。压缩映射定理给出三个直接结论：

1. **存在性**：$T$ 有固定点，因此 BOE 有解 $v^*$。
2. **唯一性**：该固定点唯一，因此最优状态值 $v^*$ 唯一。
3. **算法性**：从任意 $v_0$ 重复 $v_{k+1}=T(v_k)$，都会以几何速度收敛至 $v^*$。

> [!important] 结论的适用前提
> 这里的论证依赖折扣回报与 $0<\gamma<1$，并采用有限状态/动作的表格型 MDP 表述。函数逼近、平均回报或未折扣问题需要额外条件，不能直接照搬此结论。

---

## 影响最优策略的因素

从 BOE 可以直接看出，最优策略由以下环境和目标设定共同决定：

| 因素 | 在方程中的位置 | 典型影响 |
|---|---|---|
| 即时奖励 | $p(r\mid s,a)r$ | 改变“什么行为更有价值”；增大禁区惩罚会使绕开禁区更有吸引力。 |
| 折扣因子 | $\gamma$ | $\gamma$ 大时更重视远期收益，可能愿意承受眼前损失；$\gamma$ 小时更短视。$\gamma=0$ 时只选择即时奖励最大的动作。 |
| 转移模型 | $p(s'\mid s,a)$ | 环境动力学或风险改变时，同一动作通向好/坏状态的概率变化，最优选择也会变化。 |
| 奖励随机性 | $p(r\mid s,a)$ | 不仅奖励数值，奖励出现的概率也会影响期望即时回报。 |

### 奖励变换的一个重要例外

在 $\alpha>0$ 时，若把**所有**奖励统一变成 $r'=\alpha r+\beta$，最优策略不变；最优状态值变为

$$
v'^*=\alpha v^*+\frac{\beta}{1-\gamma}\mathbf 1.
$$

原因是正比例缩放和统一平移不会改变动作价值的相对大小。相反，若只修改某些状态、动作或奖励事件的回报，动作之间的相对排序可能改变，最优策略也可能随之改变。

### 为什么通常不会出现无意义绕路

即使每一步的普通移动奖励为 0，$0<\gamma<1$ 也会惩罚延迟获得的正回报：更长路径会让后续奖励多乘几次 $\gamma$。因此在其他条件相同且目标回报为正时，最优策略偏好更短的到达路径。给每一步统一加一个常数负奖励并不是必要条件，因为这属于上面的仿射奖励变换，不会改变最优策略。

---

## 小结

- Bellman 方程评估**给定**策略；Bellman 最优方程同时刻画最优状态值与最优策略。
- BOE 的核心操作是对动作价值取最大：$v^*(s)=\max_a q^*(s,a)$。
- 值迭代先求唯一的 $v^*$，再对 $q^*$ 贪心即可恢复一条最优策略。
- $T$ 是以 $\gamma$ 为压缩系数的压缩映射，因此存在唯一最优状态值，值迭代从任意初值收敛。
- 策略不一定唯一，但总存在确定性贪心最优策略。
