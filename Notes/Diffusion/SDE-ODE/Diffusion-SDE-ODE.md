---
type: Note
related_to:
  - "[[Diffusion]]"
  - "[[DDPM]]"
  - "[[DDIM]]"
status: Active
---

# Diffusion 中的 SDE 与 ODE 转换

---

## 1. 从离散到连续：扩散过程的 SDE 形式

### 1.1 离散马尔可夫链（DDPM 视角）

DDPM 定义前向加噪过程为：

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)$$

其中 $\beta_t$ 为预设的噪声调度。等价地：

$$x_t = \sqrt{1-\beta_t} \, x_{t-1} + \sqrt{\beta_t} \, \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I)$$

这是一个离散时间的马尔可夫链。

### 1.2 连续化：随机微分方程（SDE）

> [!note] 从离散到连续
> 当时间步长 $\Delta t \to 0$ 时，离散的马尔可夫链可以推广为连续时间的 SDE：
>
> $$dx = f(x, t) \, dt + g(t) \, dw$$
>
> 其中：
> - $f(x, t)$：**漂移系数**（drift），描述确定性的趋势
> - $g(t)$：**扩散系数**（diffusion），描述随机噪声的强度
> - $w$：标准 Wiener 过程（布朗运动）

Song et al. (Score SDE, 2021) 统一了多种扩散模型，提出三种等价的 SDE 形式：

#### VP-SDE（Variance Preserving）

> [!info] VP-SDE
> 对应 DDPM 的连续极限：
>
> $$dx = -\frac{1}{2}\beta(t) x \, dt + \sqrt{\beta(t)} \, dw$$
>
> - 漂移项将数据向原点收缩
> - 方差被噪声注入"保留"在一定范围内

#### VE-SDE（Variance Exploding）

> [!info] VE-SDE
> 对应 SMLD（Score Matching with Langevin Dynamics）：
>
> $$dx = \sqrt{\frac{d[\sigma^2(t)]}{dt}} \, dw$$
>
> - 没有漂移项，纯噪声注入
> - 方差随时间"爆炸"式增长

#### sub-VP-SDE

介于 VP 和 VE 之间，方差有更紧的界。

### 1.3 反向 SDE

> [!important] 反向 SDE
> 前向 SDE 将数据逐步加噪为纯噪声。反向过程（从噪声还原数据）由 Anderson (1982) 给出：
>
> $$dx = [f(x, t) - g(t)^2 \nabla_x \log p_t(x)] \, dt + g(t) \, d\bar{w}$$
>
> 其中：
> - $\nabla_x \log p_t(x)$ 是**得分函数**（score function），即数据分布的梯度
> - $d\bar{w}$ 是**反向**的 Wiener 过程
> - 得分函数由神经网络 $\epsilon_\theta(x_t, t)$ 学习得到

> [!tip] 关键洞察
> 只要知道得分函数（或等价地，噪声预测），就能从噪声反向采样得到数据。

---

## 2. 从 SDE 到 ODE 的转换

### 2.1 概率流 ODE（Probability Flow ODE）

> [!important] 概率流 ODE
> Song et al. 证明，对于任意前向 SDE：
>
> $$dx = f(x, t) \, dt + g(t) \, dw$$
>
> 存在一个**确定性的 ODE**，其边缘概率分布 $p_t(x)$ 与 SDE 完全相同：
>
> $$\frac{dx}{dt} = f(x, t) - \frac{1}{2} g(t)^2 \nabla_x \log p_t(x)$$
>
> 这就是**概率流 ODE**（Probability Flow ODE）。

#### 推导直觉

SDE 中的随机项 $g(t) \, dw$ 引入了扩散。如果我们精确地补偿这个扩散——通过一个确定性的"收缩"项 $-\frac{1}{2} g(t)^2 \nabla_x \log p_t(x)$——那么整体的概率演化就保持不变。

数学上，这源于 Fokker-Planck 方程。SDE 对应的 FPE 为：

$$\frac{\partial p_t}{\partial t} = -\nabla_x \cdot [f(x,t) p_t] + \frac{1}{2} g(t)^2 \Delta_x p_t$$

而 ODE 对应的连续性方程为：

$$\frac{\partial p_t}{\partial t} = -\nabla_x \cdot [v(x,t) p_t]$$

令两者相等，解出 $v(x,t)$ 即可得到 ODE 的漂移项。

### 2.2 VP-SDE 对应的 ODE

对于 VP-SDE，概率流 ODE 为：

$$\frac{dx}{dt} = -\frac{1}{2}\beta(t) x - \frac{1}{2}\beta(t) \nabla_x \log p_t(x)$$

将得分函数用噪声预测网络表示 $\nabla_x \log p_t(x) = -\frac{\epsilon_\theta(x_t, t)}{\sigma_t}$：

$$\frac{dx}{dt} = -\frac{1}{2}\beta(t) x + \frac{\beta(t)}{2\sigma_t} \epsilon_\theta(x_t, t)$$

> [!note] 与 DDIM 的联系
> 这就是 DDIM 论文中推导的 ODE，也是 DPM-Solver 求解的对象。

### 2.3 SDE vs ODE：采样的权衡

> [!example] SDE vs ODE 对比
> | 特性 | 反向 SDE | 概率流 ODE |
> |------|----------|------------|
> | 轨迹 | 随机（每次不同） | 确定性（相同输入→相同输出） |
> | 单步精度 | 较低（需多步平均） | 较高 |
> | 少步采样 | 需要噪声注入补偿 | 天然适合少步采样 |
> | 可逆性 | 不可逆 | 可逆（可编码/解码） |
> | 对数似然 | 难以计算 | 可精确计算（instantaneous change of variables） |
> | 采样质量 | 多步时略优 | 少步时显著优于 SDE |

> [!tip] 实践结论
> 少步采样（10-50 步）用 ODE（DDIM / DPM-Solver）；多步采样且追求多样性时可用 SDE。

---

## 3. 统一视角

> [!abstract] 统一视角
> Song et al. 的贡献在于将 VP、VE、sub-VP 等不同扩散模型统一到 SDE/ODE 框架下：
>
> ```
> 离散 DDPM ──(连续极限)──→ VP-SDE ──(概率流)──→ ODE
> 离散 SMLD ──(连续极限)──→ VE-SDE ──(概率流)──→ ODE
> ```
>
> 所有扩散模型都共享同一个核心思想：**前向加噪 + 反向去噪**，区别仅在于噪声调度策略和 SDE/ODE 的具体形式。

---

## 4. 参考

- Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021). Score-Based Generative Modeling through Stochastic Differential Equations. ICLR 2021.
- Anderson, B. D. (1982). Reverse-time diffusion equation models. Stochastic Processes and their Applications.
