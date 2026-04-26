# Diffusion Disillation

## What it distillation

- Generative models learn p(x), but it is implicity
- We can distill p(x) into
  - a smaller model
  - a fewer steps model
  - different modality
  - editing

## Few-step sampling

- Rectfied flows
- Consistency models
- Meanflow

## Rectified flows

**Rectified Flows（矫正流）** 是 Flow Matching 框架下的一种特殊生成模型，核心思想是**学习一个近似直线的概率路径，让数据与噪声之间的“运输”变得尽可能短而直**。

如果把生成模型比作把一团噪声“搬运”成数据，那么 Rectified Flow 背后的直观就是：

- 普通扩散模型（如 DDPM）的粒子路径是弯弯曲曲的，需要很多步才能走到终点。
- Rectified Flow **强制训练出几乎笔直的路径**，这样只需很少的步数（甚至一步）就能完成生成。

它的实现分为两个关键部分：

---

### 1. 训练：直接学直线路径

从数据 $x_0$ 和高斯噪声 $x_1$ 中采样，构造一条**直线插值**  
$$x_t = (1-t)\,x_0 + t\,x_1, \quad t \in [0,1]$$  
并让网络 $v_\theta(t,x_t)$ 去预测**恒定的方向向量**  
$$u = x_1 - x_0$$  
损失就是最简单的均方误差 $\|v_\theta(x_t,t) - (x_1-x_0)\|^2$。

和 DDPM 相比，这里没有复杂的噪声调度系数，路径完全是一条直线，因此也叫 **linear schedule**。

---

### 2. 矫正（Reflow）：让路径变得更直

虽然上面的训练让期望路径是直线，但实际学到的向量场从整体来看可能还是有点“弯”——也就是说，从任意噪声点出发积分到数据点，轨迹并不是严格直线。为了进一步拉直，Rectified Flow 引入一个**反复矫正**的步骤：

1. 用训练好的模型进行 ODE 采样，生成很多数据-噪声对 $(\hat{x}_0, \hat{x}_1)$。
2. **把这些新对儿当作新的训练数据**，重新训练一个一模一样的直线 Flow Matching 模型。
3. 可以重复多次（2‑ReFlow、3‑ReFlow…）。

每一次矫正，都相当于在“蒸馏”前一个模型：新模型的训练目标仍然是 $x_1 - x_0$，但这里的 $x_0$ 和 $x_1$ 不再是随意组合的数据和噪声，而是**同一个模型走过的实际起点和终点**。这会让整个积分轨迹越来越接近直线，最后用极少的欧拉步就能生成高质量样本，极限情况下甚至能做到**一步生成**（即从噪声直接跳到数据，无需中间步）。

---

### 本质与优势

Rectified Flow 可以看成是 **概率流 ODE 的直线化 + 自蒸馏** 的结合：

- 训练上，和 Flow Matching 一样简单高效，无需模拟扩散前向过程。
- 采样上，因为路径被逐渐拉直，用 2～5 步就能达到或超过 DDPM 1000 步的效果。
- 理论上，它提供了一种“运输映射”（transport map）的构造方法，可以和最优传输（Optimal Transport）联系起来。

## Progressive distillation(渐进式蒸馏)

1. Train a teacher model
2. Sample clean image, add noise
3. Take 2 DDIM stapes using teacher model
4. Train a student model to predict 2 steps

## Consistency models
    

## Score distillation

## Conclusion
