# DreamerV3: Mastering Diverse Domains through World Models (2023)

> **论文**: DreamerV3: Mastering Diverse Domains through World Models
> **作者**: Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap
> **来源**: arXiv 2301.04104
> **分类**: cs.LG, cs.AI

---

## 一、核心贡献

1. **通用 RL 算法**：单一配置在 150+ 任务上超越专用方法
2. **有利的缩放特性**：更大模型 → 更高数据效率和最终性能
3. **Minecraft 里程碑**：首个从零收集钻石的算法（无人类数据/课程）

---

## 二、三个神经网络

DreamerV3 同时训练三个网络，**不共享梯度**：

| 网络 | 功能 |
|------|------|
| **世界模型** (World Model) | 学习环境模型，预测未来 |
| **评论家** (Critic) | 评估状态价值 |
| **行为者** (Actor) | 选择动作 |

---

## 三、Symlog 预测

### 问题

- 不同领域的目标值尺度差异巨大
- 平方损失在大目标上可能发散
- 绝对/Huber 损失学习停滞
- 运行统计归一化引入非平稳性

### 解决方案

**Symlog 变换**：压缩大正值和负值的幅度

$$\text{symlog}(x) = \text{sign}(x) \ln(|x| + 1)$$
$$\text{symexp}(x) = \text{sign}(x)(\exp(|x|) - 1)$$

**损失**：

$$\mathcal{L}(\theta) = \frac{1}{2}(f(x,\theta) - \text{symlog}(y))^2$$

**特性**：
- 对称压缩正负大值
- 原点附近近似恒等
- 不影响小目标的学习

---

## 四、世界模型 (RSSM)

### 组件

1. **序列模型**：$h_t = f_\phi(h_{t-1}, z_{t-1}, a_{t-1})$
2. **编码器**：$z_t \sim q_\phi(z_t | h_t, x_t)$
3. **动力学预测器**：$\hat{z}_t \sim p_\phi(\hat{z}_t | h_t)$
4. **奖励预测器**：$\hat{r}_t \sim p_\phi(\hat{r}_t | h_t, z_t)$
5. **继续预测器**：$\hat{c}_t \sim p_\phi(\hat{c}_t | h_t, z_t)$
6. **解码器**：$\hat{x}_t \sim p_\phi(\hat{x}_t | h_t, z_t)$

### 总损失

$$\mathcal{L}(\phi) = \sum_{t=1}^{T}(\beta_{\text{pred}}\mathcal{L}_{\text{pred}} + \beta_{\text{dyn}}\mathcal{L}_{\text{dyn}} + \beta_{\text{rep}}\mathcal{L}_{\text{rep}})$$

权重：$\beta_{\text{pred}}=1$, $\beta_{\text{dyn}}=0.5$, $\beta_{\text{rep}}=0.1$

### 个体损失

**预测损失**：

$$\mathcal{L}_{\text{pred}} = -\ln p_\phi(x_t|z_t,h_t) - \ln p_\phi(r_t|z_t,h_t) - \ln p_\phi(c_t|z_t,h_t)$$

**动力学损失**：

$$\mathcal{L}_{\text{dyn}} = \max(1, \text{KL}[\text{sg}(q_\phi(z_t|h_t,x_t)) \| p_\phi(z_t|h_t)])$$

**表示损失**：

$$\mathcal{L}_{\text{rep}} = \max(1, \text{KL}[q_\phi(z_t|h_t,x_t) \| \text{sg}(p_\phi(z_t|h_t))])$$

### Free Bits 机制

两个 KL 损失在 1 nat (≈1.44 bits) 以下被截断：
- 当已充分最小化时禁用，专注于预测损失
- 结合 KL 平衡解决不同视觉复杂度的正则化强度需求

### Unimix 技巧

类别分布参数化为 1% 均匀 + 99% 网络输出的混合：
- 防止近确定性分布
- 确保良好缩放的 KL 损失

---

## 五、Actor-Critic 学习

### 在想象中学习

- Actor 和 Critic 完全从世界模型预测的抽象序列学习
- 折扣因子 $\gamma = 0.997$
- 预测水平 $T = 16$

### Bootstrapped λ-Returns

$$R_t^\lambda = r_t + \gamma c_t((1-\lambda)v_\psi(s_{t+1}) + \lambda R_{t+1}^\lambda)$$

$\lambda = 0.95$

### Critic：离散回归 + Twohot 编码

**关键创新**：使用离散回归而非连续回归

- 返回值 symlog 变换后离散化为 K=255 个桶
- 桶范围 $B = [-20, ..., +20]$
- **Twohot 编码**：连续值的 onehot 推广

$$\text{twohot}(x)_i = \begin{cases} |b_{k+1}-x|/|b_{k+1}-b_k| & i=k \\ |b_k-x|/|b_{k+1}-b_k| & i=k+1 \\ 0 & \text{else} \end{cases}$$

**优势**：加速稀疏奖励环境中的学习（奖励/返回值的双峰分布）

### Actor：返回值归一化 + 熵正则化

$$\mathcal{L}(\theta) = \sum_{t=1}^T \text{sg}(R_t^\lambda)/\max(1, S) - \eta \text{H}[\pi_\theta(a_t|s_t)]$$

- 返回值缩放 $S = \text{Per}(R_t^\lambda, 95) - \text{Per}(R_t^\lambda, 5)$（百分位数范围）
- 熵缩放 $\eta = 3 \times 10^{-4}$
- 防止放大稀疏返回值

---

## 六、实验结果

### 跨域 SOTA（固定超参数）

| 领域 | 任务数 | 步数 | SOTA 对比 |
|------|--------|------|----------|
| 本体感觉控制 | 18 | 500K | 超越 D4PG, DMPO, MPO |
| 视觉控制 | 20 | 1M | 超越 DrQ-v2, CURL |
| Atari 100K | 26 | 400K | 超越 IRIS, SPR, SimPLe |
| Atari 200M | 55 | 200M | 超越 Rainbow, IQN |
| BSuite | 23 | - | 超越 Bootstrap DQN, Muesli |
| Crafter | - | - | 超越 PPO, OC-SA, DreamerV2 |
| DMLab | 8 | 50M | 匹配 IMPALA (10B 步) |

### Minecraft 钻石里程碑

- **首个从零收集钻石的算法**
- 40 个种子，100M 步训练
- 24/40 种子至少收集一个钻石
- 首个钻石在 29M 步后
- 训练时间：17 GPU 天（vs VPT 的 720 V100 GPU × 9 天）

---

## 七、缩放特性

| 尺寸 | GRU | CNN 乘数 | 参数 |
|------|-----|---------|------|
| XS | 256 | 24 | 8M |
| S | 512 | 32 | 18M |
| M | 1024 | 48 | 37M |
| L | 2048 | 64 | 77M |
| XL | 4096 | 96 | 200M |

**关键发现**：增加模型大小直接转化为更高的数据效率和最终性能

---

## 八、与 DreamerV2 的关键区别

| 特性 | DreamerV2 | DreamerV3 |
|------|-----------|-----------|
| 预测 | 标准重建 | **Symlog 预测** |
| KL 正则化 | 可调 KL 缩放 | **Free bits + KL 平衡** |
| 返回值归一化 | 固定熵调优 | **百分位数归一化 + 分母最大值** |
| 类别分布 | 标准 | **Unimix (1% 均匀 + 99% 网络)** |
| 激活函数 | - | **LayerNorm + SiLU** |
| 卷积 | - | **Same-padded stride-2, kernel 3** |
| Critic | 慢目标网络 | **快 Critic + EMA 正则化** |

---

## 九、固定超参数表

| 名称 | 值 |
|------|-----|
| 回放容量 | 10⁶ |
| Batch 大小 | 16 |
| Batch 长度 | 64 |
| 潜在数量 | 32 |
| 每个潜在类别 | 32 |
| 学习率 (WM) | 10⁻⁴ |
| 想象水平 | 15 |
| 折扣水平 | 333 |
| λ | 0.95 |
| Critic EMA 衰减 | 0.98 |
| 熵缩放 | 3·10⁻⁴ |
| 学习率 (AC) | 3·10⁻⁵ |

---

## 十、关键洞察

1. **Symlog 是关键**：统一不同尺度的目标值
2. **Free bits + KL 平衡**：解决正则化强度的领域依赖性
3. **离散回归 Critic**：加速稀疏奖励学习
4. **百分位数归一化**：鲁棒的返回值缩放
5. **单一配置通用**：无需领域特定调优

---

## 十一、与你研究方向的关联

1. **DiT 在世界模型中的应用**：DreamerV3 的 RSSM 可与 DiT 结合
2. **Symlog 技术**：可应用于扩散模型的损失设计
3. **离散回归**：可能改善扩散模型的价值估计
4. **缩放特性**：世界模型也遵循缩放定律
