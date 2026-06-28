---
type: Note
related_to:
  - "[[Diffusion]]"
  - "[[DPM-Solver]]"
  - "[[DDPM]]"
status: Active
url: https://arxiv.org/abs/2211.01095
code: https://github.com/LuChengTHU/dpm-solver
---

# DPM-Solver++: A Fast Solver for Guided Sampling of Diffusion Probabilistic Models

**作者**: Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, Jun Zhu (清华)
**发表**: NeurIPS 2022（DPM-Solver 的续作）

---

## 第一部分：DPM-Solver 有什么问题？

### 1.1 回顾 DPM-Solver

> [!warning] DPM-Solver 的两个麻烦
> DPM-Solver 是一个很厉害的扩散采样器，10 步就能生成高质量图片。但它有两个麻烦：

**麻烦 1：噪声预测的中间步问题**

DPM-Solver-2 每一步需要在**中间时间点** $s = t_n + h/2$ 处评估模型。但问题是：

- 训练时模型只见过 1000 个离散时间步（$t=0, 1, 2, ..., 999$）
- 中间步 $s$ 不在训练时间步里！
- 需要用**插值**来近似，引入额外误差

> [!example] 温度计类比
> 你有一个温度计，只能显示整数温度（20°C, 21°C, 22°C...）。有人问你"20.5°C 时温度计读数是多少？"你只能猜——这就是插值带来的误差。

**麻烦 2：Classifier-Free Guidance 让评估代价翻倍**

现代扩散模型（如 Stable Diffusion）使用 **Classifier-Free Guidance** 来控制生成内容。这意味着每次评估模型时，实际上要跑**两次**神经网络：

- 一次有条件（比如"一只猫"）
- 一次无条件（不给提示）

> [!warning] 开销很大！
> - DPM-Solver-2 每步需要 1 次额外评估 → 实际要多跑 2 次神经网络
> - DPM-Solver-3 每步需要 2 次额外评估 → 实际要多跑 4 次神经网络

---

## 第二部分：DPM-Solver++ 的两个改进

### 改进 1：从"预测噪声"变成"预测干净图片"

> [!important] 数据预测 vs 噪声预测
>
> **DPM-Solver 的做法（噪声预测）**：
> 扩散模型训练时，让它从带噪图片 $x_t$ 预测**噪声** $\epsilon_\theta(x_t, t)$。然后用"加噪公式"反推干净图片：
>
> $$x_0 \approx \frac{x_t - \sigma_t \epsilon_\theta(x_t, t)}{\alpha_t}$$
>
> 这个公式里有除法（除以 $\alpha_t$），在中间步计算时会产生复杂的系数。
>
> **DPM-Solver++ 的做法（数据预测）**：
> 让模型直接预测**干净图片** $x_\theta(x_t, t)$。好处：非线性项 $D_\theta(t)$ 变成了 $x_\theta$ 的**简单线性函数**：
>
> $$D_\theta(t) = -\frac{\dot{\sigma}_t}{\sigma_t} x_t + \frac{\dot{\sigma}_t \alpha_t}{\sigma_t} x_\theta(x_t, t)$$

> [!example] 类比
> 原来的方式：你问朋友"你家在哪？"，他说"往东走 3 个路口，然后往北走 2 个路口"——你需要在脑子里算出具体位置。
> 新的方式：朋友直接告诉你经纬度——更简单直接。

### 改进 2：从"单步法"变成"多步法"

> [!important] 多步法的核心思想
> 这是 DPM-Solver++ 最重要的改进。
>
> **单步法（DPM-Solver 的做法）**：DPM-Solver-2 的每一步需要在中间时间 $s$ **再评估一次模型**，引入额外开销。
>
> **多步法（DPM-Solver++ 的做法）**：**复用之前已经算过的模型评估结果**。比如二阶方法（k=2）：第 $n+1$ 步用 $D_\theta(t_n)$ 和 $D_\theta(t_{n-1})$（上一步已经算过）来构造高阶近似。
>
> **不需要额外评估！** 总评估次数 = 步数。

> [!example] 做菜类比
> - 单步法（DPM-Solver）：每次做菜都要重新尝味道 → 每步都要评估模型
> - 多步法（DPM-Solver++）：记住前几次尝过的味道，用记忆来调味 → 复用之前的评估结果

#### 具体公式

DPM-Solver++ 的更新公式：

$$x_{n+1} = \frac{\alpha_{n+1}}{\alpha_n} x_n + \alpha_{n+1} \sum_{j=0}^{k-1} \omega_j \cdot \tilde{D}_{n+1-j}$$

其中：
- $\tilde{D}_n = D_\theta(t_n) / \alpha_n$（第 $n$ 步的模型评估结果，已缩放）
- $\omega_j$ 是**权重系数**，由 Newton-Cotes 求积公式确定
- $k$ 是阶数（用前几步的信息）

> [!note] 阶数含义
> | 阶数 $k$ | 含义 | 需要几步历史 |
> |----------|------|-------------|
> | k=1 | 一阶，等价于 DDIM | 0 步历史 |
> | k=2 | 二阶，用前两步信息 | 1 步历史 |
> | k=3 | 三阶，用前三步信息 | 2 步历史 |

> [!info] Newton-Cotes 求积公式
> 就是"用已知点的值来估算积分面积"的方法。
> - 梯形公式（2 个点）：把面积当梯形算
> - Simpson 公式（3 个点）：把面积当抛物线下面积算
> 这些是数学里现成的工具，不需要自己推导。

---

## 第三部分：效率对比

### 3.1 模型评估次数

> [!example] 评估次数对比
> | 方法 | 每步评估次数 | 10步总评估 | 带 CFG 时实际前向次数 |
> |------|------------|-----------|---------------------|
> | DPM-Solver-1 | 1 | 10 | 20 |
> | DPM-Solver-2 | **2** | **11** | **22** |
> | DPM-Solver-3 | **3** | **12** | **24** |
> | DPM-Solver++ (k=1) | 1 | 10 | 20 |
> | DPM-Solver++ (k=2) | 1 | 10 | **20** |
> | DPM-Solver++ (k=3) | 1 | 10 | **20** |
>
> **DPM-Solver++ 的评估次数和 DDIM 一样，但精度接近 DPM-Solver-3！**

### 3.2 为什么在带 Guidance 时优势更大？

> [!tip] CFG 效率优势
> - 每次评估 = 2 次模型前向（有条件 + 无条件）
> - DPM-Solver-2 每步 2 次评估 = 4 次前向
> - DPM-Solver++ 每步 1 次评估 = 2 次前向
>
> **效率直接翻倍！**

---

## 第四部分：统一框架

> [!note] 统一框架
> DPM-Solver++ 论文还做了一件漂亮的事：把 DPM-Solver 和 PNDM 统一到一个框架里。所有方法都可以写成同一个公式：
>
> $$x_{n+1} = \frac{\alpha_{n+1}}{\alpha_n} x_n + \alpha_{n+1} \sum_{j=0}^{k-1} \omega_j \cdot \tilde{D}_{n+1-j}$$
>
> 唯一的区别是权重 $\omega_j$ 怎么算：

| 方法 | $\omega_j$ 来源 |
|------|----------------|
| DDIM / PNDM | 从 DDIM 的特殊结构推导（不太系统） |
| DPM-Solver++ | Newton-Cotes 求积公式（经典数学工具） |

DPM-Solver++ 的系数更"正规"，来自数学里的标准工具。

---

## 第五部分：效果

### 5.1 无引导采样（CIFAR-10）

> [!example] 实验结果
> | 方法 | 10 步 FID | 15 步 FID | 20 步 FID |
> |------|----------|----------|----------|
> | DDIM | 13.36 | 6.22 | 4.62 |
> | DPM-Solver-2 | 4.70 | 3.93 | 3.68 |
> | DPM-Solver-3 | 3.76 | 3.57 | 3.49 |
> | **DPM-Solver++ (k=3)** | **3.72** | **3.55** | **3.48** |
>
> DPM-Solver++ (k=3) 和 DPM-Solver-3 质量相当，但**评估次数更少**。

### 5.2 带 Classifier-Free Guidance（ImageNet 256×256）

> [!example] 带 CFG 的实验结果
> | 方法 | 10 步 FID | 每步评估次数 |
> |------|----------|------------|
> | DPM-Solver-2 | 6.89 | 2 |
> | DPM-Solver-3 | 4.54 | **3** |
> | **DPM-Solver++ (k=3)** | **3.94** | **1** |
>
> DPM-Solver++ 用更少的评估次数达到了更好的效果！

### 5.3 实际应用

> [!tip] 实际应用
> DPM-Solver++ 是 **Stable Diffusion 的默认采样器**。你用 Stable Diffusion 生成图片时，大概率就在用它。

---

## 第六部分：总结

> [!abstract] DPM-Solver++ 的两个核心改进
> | 改进 | DPM-Solver | DPM-Solver++ | 好处 |
> |------|-----------|--------------|------|
> | 预测类型 | 噪声预测 $\epsilon_\theta$ | 数据预测 $x_\theta$ | 公式更简单，不需要中间步插值 |
> | 求解方法 | 单步法 | 多步法 | 无额外模型评估，效率更高 |
>
> **一句话总结**：DPM-Solver++ 通过"预测干净图片"和"复用历史评估"两个技巧，用和 DDIM 一样的评估次数，达到了接近 DPM-Solver-3 的精度。在带 Classifier-Free Guidance 的现代扩散模型中，效率优势更加明显。
