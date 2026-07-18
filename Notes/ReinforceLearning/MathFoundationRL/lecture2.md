# Bellman Equations

## Why return is important

The return is discounted sum of the rewards obtained along a trajectory.

How to calculate it?

Method1: by definition

Let $v_i$ denote the return obtained starting from $s_i$ (i = 1, 2, 3,4)

$$
\begin{aligned}
v_1 &= r_1 + \gamma r_2 + \gamma^2 r_3 +\cdots\\
v_2 &= r_2 + \gamma r_3 + \gamma^2 r_4 + \cdots\\
v_3 &= r_3 + \gamma r_4 + \gamma^2 r_5 + \cdots\\
v_4 &= r_4 + \gamma r_5 + \gamma^2 r_6 + \cdots
\end{aligned}
$$

Method2: by recursive relationship

$$
\begin{aligned}
v_1 &= r_1 + \gamma v_2\\
v_2 &= r_2 + \gamma v_3\\
v_3 &= r_3 + \gamma v_4\\
v_4 &= r_4 + \gamma v_5
\end{aligned}
$$

The returns rely on each other: Bootstrapping.

How to solve these equations?

Write in the following matrix-vector form:

$$
\underset{\displaystyle\mathbf{v}}{\begin{bmatrix}
v_1\\
v_2\\
v_3\\
v_4
\end{bmatrix}}
=
\underset{\displaystyle\mathbf{r}}{\begin{bmatrix}
r_1\\
r_2\\
r_3\\
r_4
\end{bmatrix}}
+
\gamma\,
\underset{\displaystyle\mathbf{P}}{\begin{bmatrix}
0 & 1 & 0 & 0\\
0 & 0 & 1 & 0\\
0 & 0 & 0 & 1\\
1 & 0 & 0 & 0
\end{bmatrix}}
\underset{\displaystyle\mathbf{v}}{\begin{bmatrix}
v_1\\
v_2\\
v_3\\
v_4
\end{bmatrix}}
$$

Which can be written as:

$$
\mathbf{v} = \mathbf{r} + \gamma \mathbf{P} \mathbf{v}
$$

This is the Bellman equation(for this specific deterministic problem)

- Though simple, it demonstrates the core idea: the value of one state relies on the values of other states.
- A matrix-vector form is more clear to see how to solve the state values.

## State Value

Consider the following single-step process:

$$
S_t \xrightarrow {A_{t}} R_{t+1}, S_{t+1}
$$

$$
\text{Note that} S_{t},A_{t},R_{t+1} \text{ are all random variables}
$$

This step is governed by the following probability distributions:

$$
\begin{aligned}
	&S_{t}\rightarrow A_{t} \, \text{is governed by } \pi (A_{t}=a|S_{t}=s) \\
	&S_{t},A_{t}\rightarrow R_{t+1} \, \text{is governed by  } p(R_{t+1}=r|S_{t},A_{t} = a)\\
	&S_{t},A_{t} \rightarrow S_{t+1} \text{is governed by } p\left( S_{t+1}=s'|S_{t}=s,A_{t}=a \right)
\end{aligned}

$$

Consider the following multi-step trajectory:

$$
S_{t}\xrightarrow{A_{t}} R_{t+1},S_{t+1}\xrightarrow{A_{t+1}} R_{t+2},S_{t+2}\xrightarrow{A_{t+2}} R_{t+3},\dots
$$

The discounted return is:

$$
G_{t} = R_{t+1} + \gamma R_{t+2}+\gamma^2R_{t+3} + \dots
$$

$G_t$ is also a random variable since $R_{t+1}$,$R_{t+2}$,$\dots$are random variables.
---

The expectation of $G_{t}$ is defined as the state-value function or simply state value:

$$
v_{\pi}(s) = \mathbb{E}[G_{t}|S_{t}=s]
$$

Q: What is the relationship between return and state value?

A: The state value is the mean of all possible returns that can be obtained starting from a state. If everything -$\pi(a|s),p(r|s,a),p(s'|s,a)$ is deterministic then the state value is the same as return.

 ![[state-value-different-trajectory.png]]

## Deriving The Bellman Equation

Consider a random trajectory:

$$
S_{t}\xrightarrow{A_{t}} R_{t+1},S_{t+1}\xrightarrow{A_{t+1}} R_{t+2},S_{t+2}\xrightarrow{A_{t+2}} R_{t+3},\dots
$$

the return $G_{t}$ can be written as

$$
\begin{aligned}
	G_{t}&= R_{t+1} + \gamma R_{t+2}+\gamma^2 R_{t+3} + \dots\\
	&=R_{t+1} + \gamma G_{t+1}
\end{aligned}
$$

state value:

$$
\begin{aligned}
v_{\pi}(s) &= \mathbb{E}[G_{t}|S_{t}=s]\\
&=\mathbb{E}[R_{t+1}+\gamma G_{t+1}|S_{t}=s]\\
&=\mathbb{E}(R_{t+1}|S_{t}=s) + \gamma \mathbb{E}(G_{t+1}|S_{t}=s)
\end{aligned}
$$

$\mathbb{E}[R_{t+1}|S_{t}=s]$:

- $S_{t},A_{t}\rightarrow R_{t+1} \, \text{is governed by  } p(R_{t+1}=r|S_{t}=s,A_{t} = a)$
- Policy: the probability of action a with a given state s: $\pi(a|s)$
- The average reward with  given s, a: $\mathbb{E}[R_{t+1}|S_{t}=s,a_{t}=a]=\sum_{r}p(r|s,a)r$
- $\mathbb{E}[R_{t+1}|S_{t}=s]=\sum_{a}E[R_{t+1}|S_{t}=s, A_{t}=a]\pi(a|s)=\sum_{a}\pi(a|s)\sum_{r}p(r|s,a)r$

$$
\begin{aligned}
	\mathbb{E}[R_{t+1}|S_{t}=s] &= \sum_{a}\pi(a|s)\mathbb{E}[R_{t+1}|S_{t}=s,A_{t}=a]\\
	&=\sum_{a}\pi(a|s)\sum_{r}p(r|s,a)r
\end{aligned}
$$

$$
\begin{aligned}
	\mathbb{E}[G_{t+1}|S_{t}=s]&=\sum_{s'}\mathbb{E}[G_{t+1}|S_{t}=s,S_{t+1}=s']p(s'|s)\\
	&=\sum_{s'}\mathbb{E}[G_{t+1}|S_{t+1}=s']p(s'|s)\\
	&=\sum_{s'}v_{\pi}(s')p(s'|s) \\
	&=\sum_{s'}v_{\pi}(s')\sum_{a} p(s'|s,a)\pi(a|s)
\end{aligned}
$$

- $\mathbb{E}[G_{t+1}|S_{t}=s])$ is the mean of future rewards
- $\mathbb{E}[G_{t+1}|S_{t} =s,S_{t+1}=s']=\mathbb{E}[G_{t+1}|S_{t+1}=s']$ due to memoryless Markov property.
Therefore, we have

$$
\begin{align*}
v_\pi(s) &= \mathbb{E}[R_{t+1} \mid S_t = s] + \gamma \mathbb{E}[G_{t+1} \mid S_t = s], \\
&= \underbrace{\sum_{a} \pi(a|s) \sum_{r} p(r|s,a)r}_{\text{mean of immediate rewards}} + \gamma \underbrace{\sum_{a} \pi(a|s) \sum_{s'} p(s'|s,a) v_\pi(s')}_{\text{mean of future rewards}}, \\
&= \sum_{a} \pi(a|s) \left[ \sum_{r} p(r|s,a)r + \gamma \sum_{s'} p(s'|s,a) v_\pi(s') \right], \quad \forall s \in \mathcal{S}.
\end{align*}
$$

Highlights:

- The above equation is called the Bellman equation, which characterizes the relationship among the state-value functions of different states.
- It consists of two terms : the immediate reward term and the future reward term.
- A set of equations:  every state has an equation like this.
- $v_{\pi}(s)$ and $v_{\pi}(s')$ are state values to be calculated. Need Bootstrapping.
- $\pi(a|s)$ is a given policy. Solving the equation is called policy evaluation.
- $p(r|s,a)$ and $p(s'|s,a)$ represent the dynamic model. What if the model is known or unknown?
![[Bellman-equation.png]]

![[Bellman-equation-2.png]]

Solve:

$$
\begin{aligned}
	v_{\pi}(s_{1}) = \frac{\gamma}{1-\gamma}\\
	v_{\pi}(s_{2}) = v_{\pi}{s_{3}} = v_{\pi}(s_{4})=\frac{1}{1-\gamma}
\end{aligned}
$$

![[Bellman-equation-excercise.png]]

$$
\begin{aligned}
	v_{\pi}(s_{1})&=0.5[1+\gamma v_{\pi}(s_{2})]+0.5[1+\gamma v_{\pi}(s_{3})]\\
	v_{\pi}(s_{2})&=1+\gamma v_{\pi}(s_{4})\\
	v_{\pi}(s_{3})&=1+\gamma v_{\pi}(s_{4})\\
	v_{\pi}(s_{4})&=1+\gamma v_{\pi}(s_{4})
\end{aligned}
$$

$\Rightarrow$

$$
\begin{aligned}
	v_{\pi}(s_{1})=-0.5+\frac{\gamma}{1-\gamma}\\
	v_{\pi}(s_{2})=v_{\pi}(s_{3})=v_{\pi}(s_{4}) = \frac{1}{1-\gamma}
\end{aligned}
$$

## Matrix-Vector Representation

Recall that:

$$
v_\pi(s) = \sum_{a} \pi(a|s) \left[ \sum_{r} p(r|s,a)r + \gamma \sum_{s'} p(s'|s,a)v_\pi(s') \right]
$$

Rewrite the Bellman equation as

$$
v_\pi(s) = r_\pi(s) + \gamma \sum_{s'} p_\pi(s'|s)v_\pi(s') \tag{1}
$$

where

$$
r_\pi(s) \triangleq \sum_{a} \pi(a|s) \sum_{r} p(r|s,a)r, \qquad p_\pi(s'|s) \triangleq \sum_{a} \pi(a|s)p(s'|s,a)
$$

Suppose the states could be indexed as \(s_i\) (\(i = 1, \dots, n\)).

For state \(s_i\), the Bellman equation is

$$
v_\pi(s_i) = r_\pi(s_i) + \gamma \sum_{s_j} p_\pi(s_j|s_i)v_\pi(s_j)
$$

Put all these equations for all the states together and rewrite to a matrix-vector form

$$
v_\pi = r_\pi + \gamma P_\pi v_\pi
$$

where

- *$v_\pi = [v_\pi(s_1), \dots, v_\pi(s_n)]^T \in \mathbb{R}^n$
* $r_\pi = [r_\pi(s_1), \dots, r_\pi(s_n)]^T \in \mathbb{R}^n$
* $P_\pi \in \mathbb{R}^{n \times n}$, where $[P_\pi]_{ij} = p_\pi(s_j|s_i)$, is the state transition matrix

 If there are four states, $v_\pi = r_\pi + \gamma P_\pi v_\pi$ can be written out as

$$

\begin{aligned}
\underbrace{\begin{bmatrix}
v_\pi(s_1) \\
v_\pi(s_2) \\
v_\pi(s_3) \\
v_\pi(s_4)
\end{bmatrix}}_{v_\pi}
&= \underbrace{\begin{bmatrix}
r_\pi(s_1) \\
r_\pi(s_2) \\
r_\pi(s_3) \\
r_\pi(s_4)
\end{bmatrix}}_{r_\pi}
+ \gamma
\underbrace{\begin{bmatrix}
p_\pi(s_1|s_1) & p_\pi(s_2|s_1) & p_\pi(s_3|s_1) & p_\pi(s_4|s_1) \\
p_\pi(s_1|s_2) & p_\pi(s_2|s_2) & p_\pi(s_3|s_2) & p_\pi(s_4|s_2) \\
p_\pi(s_1|s_3) & p_\pi(s_2|s_3) & p_\pi(s_3|s_3) & p_\pi(s_4|s_3) \\
p_\pi(s_1|s_4) & p_\pi(s_2|s_4) & p_\pi(s_3|s_4) & p_\pi(s_4|s_4)
\end{bmatrix}}_{P_\pi}
\underbrace{\begin{bmatrix}
v_\pi(s_1) \\
v_\pi(s_2) \\
v_\pi(s_3) \\
v_\pi(s_4)
\end{bmatrix}}_{v_\pi}
.
\end{aligned}
$$

Why to solve state values?

- Given a policy, finding out the corresponding state values is called policy evaluation ! It is a fundamental problem in RL. It is the foundation to find better policies.
- It is important to understand how to solve the Bellman equation.
The Bellman equation in matrix-vector form is

$$
v_{\pi} = r_{\pi} + \gamma P_{\pi}v_{\pi}
$$

- The *closed-form* solution is:

$$
v_{\pi}=(I-\gamma P_{\pi})^{-1}r_{\pi}
$$

In practice, we still need to use numerical tools to  calculate the matrix inverse.

Can we avoid the matrix inverse operation? Yes, by iterative algorithms.

- An iterative solution is:

$$
v_{k+1} = r_{\pi} + \gamma P_{\pi}v_{k}
$$

This algorithm leads to a sequence $v_{0}, v_{1}, v_{2}, \dots$. We can show that:

$$
v_{k}\to v_{\pi}=(I-\gamma P_{\pi})^{-1}r_{\pi}, k\to \infty 
$$

> [!ABSTRACT] 定义误差
> 将误差定义为 $\delta_k = v_k - v_\pi$。我们只需证明 $\delta_k \rightarrow 0$。

> [!IMPORTANT] 代入与推导
> 将 $v_{k+1} = \delta_{k+1} + v_\pi$ 和 $v_k = \delta_k + v_\pi$ 代入贝尔曼方程 $v_{k+1} = r_\pi + \gamma P_\pi v_k$ 中，得：
>
> $$
> \delta_{k+1} + v_\pi = r_\pi + \gamma P_\pi(\delta_k + v_\pi),
> \]
> 对其重新整理并利用 \(v_\pi = r_\pi + \gamma P_\pi v_\pi\) 消去同类项，可得：
> \[
> \delta_{k+1} = -v_\pi + r_\pi + \gamma P_\pi \delta_k + \gamma P_\pi v_\pi = \gamma P_\pi \delta_k.
> $$

> [!NOTE] 结果的递归展开
> 因此，可以得到：
>
> $$
> \delta_{k+1} = \gamma P_\pi \delta_k = \gamma^2 P_\pi^2 \delta_{k-1} = \cdots = \gamma^{k+1} P_\pi^{k+1} \delta_0.
> $$

> [!TIP] 收敛性论证
> 注意 $0 \leq P_\pi^k \leq 1$，这意味着对于任意 $k = 0, 1, 2, \dots$，$P_\pi^k$ 的每一项都不大于 1。这是因为 $P_\pi^k \mathbf{1} = \mathbf{1}$，其中 $\mathbf{1} = [1, \dots, 1]^T$。
>
> 另一方面，由于折扣因子 $\gamma < 1$，我们知道 $\gamma^k \to 0$。因此，当 $k \to \infty$ 时，必然有 $\delta_{k+1} = \gamma^{k+1} P_\pi^{k+1} \delta_0 \to 0$。
>
> 证明完毕。 $\square$

## Action value

From state value to action value:

-  State value: the average return the agent can get starting from a state.
-  Action value: the average return the agent can get starting from a state and taking an action.
Why do we care action value ? Because we want to know which action is better.
Definition:

$$
q_\pi(s, a) = \mathbb{E}[G_t \mid S_t = s, A_t = a]
$$

*   $q_\pi(s, a)$ is a function of the state-action pair \((s, a)\)
*   $q_\pi(s, a)$ depends on $\pi$

It follows from the properties of conditional expectation that

$$
\underbrace{\mathbb{E}[G_t \mid S_t = s]}_{v_\pi(s)} = \sum_{a} \underbrace{\mathbb{E}[G_t \mid S_t = s, A_t = a]}_{q_\pi(s, a)} \pi(a \mid s)
$$

Hence,

$$
\color{red}{v_\pi(s)} = \sum_{a} \pi(a \mid s) \color{red}{q_\pi(s, a)} \tag{2}
$$

Recall that the state value is given by

$$
v_\pi(s) = \sum_{a} \pi(a \mid s) \underbrace{\left[ \sum_{r} p(r \mid s, a)r + \gamma \sum_{s'} p(s' \mid s, a)v_\pi(s') \right]}_{\color{red}{q_\pi(s,a)}} \tag{3}
$$

By comparing $(2)$ and $(3)$, we have the action-value function as

$$
\color{red}{q_\pi(s,a)} = \sum_{r} p(r \mid s, a)r + \gamma \sum_{s'} p(s' \mid s, a)v_\pi(s') \tag{4}
$$

$(2)$ and $(4)$ are the two sides of the same coin:

*   $(2)$ shows how to obtain state values from action values.
*   $(4)$ shows how to obtain action values from state values.
