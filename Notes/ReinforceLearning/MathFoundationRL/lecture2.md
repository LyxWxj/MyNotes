# Bellman Equations
## Why return is important

The return is discounted sum of the rewards obtained along a trajectory.

How to calculate it?
Method1: by definition
Let $v_i$ denote the return obtainded starting from $s_i$ (i = 1, 2, 3,4)
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
