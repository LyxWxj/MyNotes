# VAPO（带价值的 RL 策略优化，Value Augmented PPO）

> 论文：Yu et al. (ByteDance Seed), *VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks*, arXiv:2504.05118（2025）
> 相关笔记：[[01-PPO（近端策略优化）]]、[[07-DAPO]]、[[02-GRPO（组相对策略优化）]]

## 一、定位与动机

无价值模型算法（GRPO/DAPO）省掉了 critic，但也付出了**样本效率**的代价：outcome 级优势在长推理任务上粒度太粗，且无法提供过程性信号。VAPO 论证：**价值模型路线（PPO）的理论上限更高**——关键问题不是"要不要 critic"，而是"如何把 critic 训好"。

论文把 PPO 在高级推理任务上失败的原因归结为**三大挑战**：

1. **价值模型偏差（value model bias）**：奖励极度稀疏（绝大多数 step 奖励为 0，只有末尾 ±1），critic 从头随机初始化学不动，产生系统性高估/低估；
2. **异构序列长度（heterogeneous sequence lengths）**：同一 batch 内回复长度从几百到几万 token，固定 λ 的 GAE 对不同长度序列的偏差-方差特性完全不同；
3. **奖励信号稀疏（sparse rewards）**：几乎只有末尾有信号，中间 token 的梯度基本靠价值模型"搬运"。

VAPO 用**七大组件**逐一解决，最终在 Qwen2.5-32B 上把 AIME 2024 做到 **60.4 分**（对比：PPO 约 5 分、GRPO 47、DAPO 50）。

## 二、七大组件

### 1. Value Pretraining（价值模型预训练）

用**固定的参考策略**采样一批轨迹，用 λ=1 的蒙特卡洛回报（真实最终奖励）直接回归训练 critic，约 50 步：

$$\mathcal{L}_{\mathrm{VP}}=\mathbb{E}\left[\big(V_\phi(s_t)-\sum_{t'=t}^{T}\gamma^{t'-t}r_{t'}\big)^2\right]$$

目的：给 critic 一个**正确的起点**，消除随机初始化的系统性偏差（挑战 1 的主解法）。

### 2. Decoupled-GAE（解耦 λ）

critic 与 policy 使用**不同的 GAE λ**：

$$\lambda_{\mathrm{critic}}=1.0\quad(\text{低偏差，学习真实回报}),\qquad \lambda_{\mathrm{policy}}<1\ (\text{如 }0.95)\quad(\text{低方差，稳定更新})$$

critic 需要"看得准"（低偏差），policy 需要"稳"（低方差）——两者解耦，互不拖累。

### 3. Length-Adaptive GAE（长度自适应 λ）

固定 λ 对长序列不公平：序列越长，GAE 展开的乘积项越多，方差越大。VAPO 让 policy 的 λ 随序列长度衰减：

$$\lambda_{\mathrm{policy}}(l)=1-\frac{1}{\alpha\cdot l},\qquad \alpha=0.5$$

即：**回复越长，λ 越小**（更看重近期优势），把长序列的方差控制在稳定范围（挑战 2 的解法）。

### 4. Token-Level Policy Loss（token 级损失）

同 DAPO：损失按**总 token 数**归一化，消除"短回复 token 被放大、长回复 token 被稀释"的长度偏差：

$$\mathcal{L}_{\mathrm{policy}}=\mathbb{E}\left[\frac{1}{\sum_i|o_i|}\sum_{i,t}\min\Big(r_{i,t}\widehat{A}_{i,t},\ \mathrm{clip}(r_{i,t})\widehat{A}_{i,t}\Big)\right]$$

### 5. Clip-Higher（非对称裁剪）

同 DAPO：$\varepsilon_{\mathrm{low}}=0.2,\ \varepsilon_{\mathrm{high}}=0.28$，给正优势方向更大更新空间，缓解熵崩塌。

### 6. Positive Example LM Loss（正样本模仿学习）

对**高奖励回复**额外施加一个语言建模（NLL）损失，强迫模型"记住"正确解题路径：

$$\mathcal{L}=\mathcal{L}_{\mathrm{PPO}}+\mu\cdot\mathcal{L}_{\mathrm{NLL}},\qquad \mathcal{L}_{\mathrm{NLL}}=-\mathbb{E}\left[\frac{1}{|y^+|}\sum_{t}\log\pi_\theta(y^+_t|x,y^+_{<t})\right]$$

其中 $y^+$ 为得分最高的回复（阈值以上）。这相当于在 RL 中混入一小部分模仿学习，显著加速早期收敛。

### 7. Group-Sampling（分组采样）

每个 prompt 采样 $G=16$ 个回复（大组），降低组间方差、给 value 学习提供更稳的目标（与 GRPO 的大 G 同理）。

## 三、算法过程

```
0. 预热：用参考策略采样 + MC 回报预训练 critic（约 50 步）
1. 循环（每轮）：
   a. 每个 prompt 采样 G=16 个回复（Group-Sampling）
   b. 规则验证器打分（正确 +1 / 错误 -1）
   c. Decoupled-GAE：critic λ=1.0；policy λ 按长度自适应 λ=1-1/(α·l)
   d. Token-Level loss + Clip-Higher + Positive Example LM Loss（μ 权重）+ 熵正则
   e. 更新策略与 critic（critic 可用 value loss clip）
```

## 四、结果与消融

（Qwen2.5-32B，AIME 2024）

| 方法 | AIME 2024 |
|---|---|
| 朴素 PPO | ~5 |
| GRPO | 47 |
| DAPO | 50 |
| **VAPO** | **60.4** |

消融要点：

- Value Pretraining 贡献最大（无它则训练崩或不收敛）；
- Decoupled-GAE + Length-Adaptive GAE 共同把训练曲线拉稳（去掉任何一个，长序列方差回升）；
- Token-Level Loss 与 Clip-Higher 与 DAPO 中一致有效；
- Positive Example LM Loss 加速收敛（尤其训练早期）；
- 大 G（16）对 value 学习稳定至关重要。

## 五、与 DAPO / GRPO / PPO 的关系

- **价值路线回归 SOTA**：VAPO 证明"带 critic"不必然是劣势，**critic 训好之后样本效率显著高于无价值算法**（同等算力下 AIME 更高）；
- **复用 DAPO**：Clip-Higher、Token-Level Loss 两项直接来自 DAPO——两条路线在工程技巧上是共享的；
- **对 GRPO**：VAPO 的价值模型 + 长度自适应 λ 提供了 GRPO 系没有的"过程性优势"，代价是双模型显存与更复杂的训练管线。

## 六、优缺点与适用场景

**优点**：SOTA 性能（2025 上半年推理 RL 公开最高）；样本效率高；可扩展至过程奖励、agentic 多步任务（token 级优势的价值体现）。

**缺点**：工程复杂度高（critic 预训练、双 λ、长度自适应、正样本 loss 五个组件都要调）；显存/算力成本高于无价值算法；组件间耦合，消融和排障成本高。

**适用场景**：预算充足、追求性能上限的推理 RL；超长 CoT（Length-Adaptive GAE 的价值所在）；需要细粒度（token 级）优势的 agentic / 过程奖励任务。

## 七、参考

- VAPO：Yu et al., arXiv:2504.05118
- DAPO：arXiv:2503.14476（组件复用来源）
- PPO/GAE：arXiv:1707.06347、arXiv:1506.02438
- 实现参考：verl（`vapo` 相关 trainer 配置）、官方 ByteDance Seed 开源
