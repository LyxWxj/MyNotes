---
type: Note
related_to:
  - "[[DDPM]]"
  - "[[DDIM]]"
  - "[[FlowMatching]]"
  - "[[RectifiedFlow]]"
  - "[[Diffusion]]"
status: Active
---

# DDPM、DDIM、Flow Matching 与 Rectified Flow 的关系

> [!cite] 关联笔记
> - [[DDPM]] / [[DDIM]] / [[FlowMatching]] / [[RectifiedFlow]]
> - 背景：[[Diffusion-SDE-ODE]]（概率流 ODE）
> - 参考：Hammour Yue. *Diffusion学习笔记（二十）——深入理解Rectified Flow，完善统一扩散框架*（知乎）

---

## 1. 总览

> [!warning] 记号差异（重要）
> | | DDPM / DDIM | Flow Matching / Rectified Flow |
> |---|---|---|
> | 数据 | $x_0$ | $x_1$ |
> | 噪声 | $x_T$ / $\varepsilon$ | $x_0$ |
> | 时间方向 | $t: 0 \to T$（数据→噪声） | $t: 0 \to 1$（噪声→数据） |
>
> 下表统一按"生成方向"叙述，避免因记号不同产生误解。

| 方法 | 建模对象 | 插值 / 前向过程 | 训练目标 | 采样方式 | 典型步数 |
|---|---|---|---|---|---|
| DDPM | 离散马尔可夫扩散 SDE | $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon$ | 预测噪声 $\varepsilon$（等价 score） | 随机反向 SDE 去噪 | ~1000 |
| DDIM | 同一模型的非马尔可夫前向 | 边际 $q(x_t|x_0)$ 与 DDPM 相同 | **复用 DDPM 网络**（$\varepsilon$ 预测） | 确定性 ODE（可调 $\sigma_t$） | 10-50 |
| Flow Matching | 连续 ODE 速度场 | $x_t = (1-t)x_0 + t x_1$（直线） | 预测速度 $v = x_1 - x_0$ | Euler / ODE solver | 5-20 |
| Rectified Flow | 连续 ODE 速度场 + reflow | 同上（因果化线性插值） | 同上（CFM 损失） | Euler；reflow 后 1-2 步 | 1-50 |

---

## 2. 逐对关系

### 2.1 DDPM → DDIM：同一网络的采样加速

- DDIM 构造一族**非马尔可夫**前向过程，保持边际 $q(x_t|x_0)$ 与 DDPM 完全一致，因此**无需重新训练**。
- $\sigma_t = 0$ 时反向过程退化为**确定性 ODE**（即概率流 ODE 的离散形式），20-50 步即可；$\sigma_t$ 取 DDPM 的值则恢复随机 DDPM。
- 本质：DDIM 是**采样算法**，不改变模型与训练目标，把"随机去噪"换成"ODE 积分"。

### 2.2 DDPM / Score-based → Flow Matching：框架的简化

- DDPM 的路线：先设计前向 SDE，再用 ELBO 推导训练目标；Flow Matching 直接定义概率路径 + 回归速度场，**跳过 SDE 机制与调度设计**。
- 数学上 DDPM 对应的概率流 ODE 等价于 Flow Matching 框架中选择"弯曲插值"（$\sqrt{\bar\alpha_t}$ 加权）的特例（见 [[FlowMatching]] §5）。
- 连接两者的常见参数化：**v-prediction**（$v = \sqrt{\bar\alpha_t}\,\varepsilon - \sqrt{1-\bar\alpha_t}\,x$），把噪声预测与速度预测统一。

### 2.3 Flow Matching ↔ Rectified Flow：同训练、不同侧重

> [!important] 最容易被混淆的一对
> 在线性插值下，两者的**训练损失完全相同**（CFM 损失，最优解都是 $v(z,t) = \mathbb{E}[X_1-X_0 \mid X_t=z]$），且独立发表于同一时期。
>
> - **Flow Matching**（Lipman et al.）：强调"条件路径 → 边际向量场"的构造，路径选择自由（高斯/线性/OT）。
> - **Rectified Flow**（Liu et al.）：强调因果化、边际保持、传输成本不增，以及独有的 **reflow 迭代拉直**。
>
> 实际差别在采样端：标准 FM 通常 5-20 步；RF + reflow 后可 1-2 步。

### 2.4 DDIM ↔ Rectified Flow：弯曲 vs 直线

- 在 RF 的统一框架 $X_t = \alpha_t X_1 + \beta_t \xi$ 中，DDIM（VP-ODE）对应非线性 $\alpha_t = \sqrt{\bar\alpha_t}, \beta_t = \sqrt{1-\bar\alpha_t}$ → **弯曲轨迹、速度时变**；
- RF 对应 $\alpha_t = t, \beta_t = 1-t$ → **直线轨迹、速度恒定**。
- 直线路径使 Euler 积分无离散误差，这是 RF 能"一步生成"而 DDIM 需要 10-50 步的根本原因（参见 [[RectifiedFlow]] §6.2）。

> [!note] 但 reflow 并不只对 RF 有效
> 用配对数据二次训练（2-rectified）同样能加速 DDIM 等非线性 ODE——本质是让 ODE 贴近**各自的插值方程**：DDIM 贴近弯曲的插值曲线（离散误差变小、采样加速），RF 贴近直线插值（退化为一步）。区别只在插值形状，机制是同一个，见 [[RectifiedFlow]] §6.5。

---

## 3. 数学联系

### 3.1 插值与预测目标的对应

| 方法 | 中间点 $x_t$ | 网络预测 | 与速度场的关系 |
|---|---|---|---|
| DDPM | $\sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon$ | $\varepsilon$ | score $\approx -\varepsilon / \sqrt{1-\bar\alpha_t}$ |
| DDIM | 同上（边际一致） | $\varepsilon$（复用） | 确定性 ODE 的离散化 |
| FM / RF | $(1-t)x_0 + tx_1$ | $v = x_1 - x_0$ | $v$ 本身就是速度 |
| v-prediction（SD3 等） | $\bar\alpha_t$ 插值 | $v = \sqrt{\bar\alpha_t}\,\varepsilon - \sqrt{1-\bar\alpha_t}\,x$ | 与 score / $\varepsilon$ 线性等价 |

### 3.2 最优解都是"条件期望"

- DDPM：$\varepsilon_\theta(x_t,t) \approx \mathbb{E}[\varepsilon \mid x_t]$；
- FM / RF：$v_\theta(x_t,t) \approx \mathbb{E}[X_1-X_0 \mid X_t]$；
- DDIM / 概率流 ODE：其速度场同样由条件期望导出。

> [!abstract] 统一母题
> 四者都在拟合"**给定当前点，所有可能轨迹方向的平均**"。区别只在：插值路径的形状（弯曲 vs 直线）、训练目标的形式（噪声 vs 速度）、以及采样时是否引入随机性（SDE vs ODE）。

### 3.3 连续化视角

```
离散 DDPM ──(连续极限)──→ VP-SDE ──(概率流, Song 2021)──→ PF-ODE ≡ DDIM
                                                               │
FM / RF 直接定义连续 ODE ──(RF 视角)──→ DDIM/PF-ODE 是弯曲插值特例
RF + reflow ──→ 直线度 O(1/K) ──→ 单步 Euler 精确 ──→ 1-step 生成
```

---

### 3.4 一步生成的统一机制（ε-prediction 视角）

> [!important] 任何插值 + 配对数据二次训练 → 一步生成
> 设插值方程 $X_t = I_t(X_0, \varepsilon)$（DDPM 取 $I_t = \sqrt{\alpha_t}X_0 + \sqrt{1-\alpha_t}\,\varepsilon$，RF 取 $I_t = tX_1 + (1-t)X_0$）。reflow 让网络学会插值本身，采样时**直接反解插值方程** $\hat{X}_0 = I_t^{-1}(X_t, \varepsilon^{**}_\theta)$：
> - DDPM 类：$\hat{X}_0 = \left(X_t - \sqrt{1-\alpha_t}\,\varepsilon^{**}_\theta\right)/\sqrt{\alpha_t}$，免校正、一步生成；
> - RF：反解恰好等于欧拉一步。
>
> 所以"一步生成"不是直线路径的专属，而是 reflow 的通用性质；**直线只是 RF 反解插值方程时的副产品**。

### 3.5 统一框架的两条路线

> [!note] Flow Matching vs 随机插值（Stochastic Interpolants）
> | | Flow Matching | 随机插值 / Rectified Flow |
> |---|---|---|
> | 出发点 | $X_1 = \varphi_t(X_0, X_t)$ | $X_t = I_t(X_0, X_1)$ |
> | 关注 | 条件路径 $p(X_t \mid X_0)$（前向过程） | 速度 $v(X_t,t) = \mathbb{E}[\dot{X}_t \mid X_t]$ |
> | 利于 | ε-prediction | v-prediction |
>
> 两者完全等价。DDPM、DDIM、FM、RF 都是这条统一框架下的特例，差异在于插值形状（弯曲 vs 直线）、预测目标（ε vs v）、以及是否做 reflow。

---

## 4. 采样效率对比

| 方法 | 典型步数 | 原因 |
|---|---|---|
| DDPM | ~1000 | 随机马尔可夫去噪，每步只能"退"一小步噪声 |
| DDIM | 10-50 | 确定性 ODE，但轨迹弯曲 → 一阶 Euler 离散误差 |
| Flow Matching | 5-20 | 直线插值，但独立耦合下边际速度场仍有曲率 |
| Rectified Flow | 1-2（reflow 后） | reflow 迭代拉直，单步 Euler 几乎精确 |

> [!note] 步数不是唯一指标
> 现代模型（SD3/Flux 等）通常不跑 reflow，靠大规模训练 + 低步数调度器（如 EDM / DPM-Solver）在 1-4 步内达到实用质量，见 [[DPM-Solver]]。

---

## 5. 实践选择建议

- **已有 DDPM 模型想加速** → [[DDIM]] / [[DPM-Solver]]：无需训练，即插即用。
- **从头训练、希望少步采样** → Flow Matching / Rectified Flow：现代 T2I 的默认选择。
- **目标是一步生成** → RF + reflow（+ 蒸馏），如 InstaFlow；或直接蒸馏路线（SDXL-Turbo、LCM 等，见 [[Distillation]]）。

---

## 6. 一句话总结

> [!abstract] 演进主线
> **DDPM** 定义了问题（SDE + 噪声预测）；**DDIM** 把同一模型变成 ODE，大幅加速采样；**Flow Matching** 把整个框架简化成"直线插值 + 回归速度"；**Rectified Flow** 在此基础上用 reflow 让 ODE 贴近插值方程，实现单步生成。四者是同一"概率路径 + 条件期望回归"母题下的不同选择：**路径形状、预测目标、采样随机性**三个维度各有取舍。更深的统一视角（真实 ODE 并非直线、ε-prediction 下 reflow 对任意扩散模型有效）见 [[RectifiedFlow]] §5 与 §6.5。
