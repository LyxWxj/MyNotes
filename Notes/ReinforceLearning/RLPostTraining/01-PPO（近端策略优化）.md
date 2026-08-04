# PPO（近端策略优化，Proximal Policy Optimization）

> 论文：Schulman et al., *Proximal Policy Optimization Algorithms*, arXiv:1707.06347（2017）
> 配套：TRPO（arXiv:1502.05477）、GAE（arXiv:1506.02438）
> 相关笔记：[[02-GRPO（组相对策略优化）]]、[[03-DPO（直接偏好优化）]]、[[08-VAPO]]

## 一、定位与动机

PPO 是 **on-policy 策略梯度（policy gradient）家族**中最广泛使用的算法，也是 LLM 后训练（RLHF / RL）中最先被大规模采用的算法（InstructGPT 即基于 PPO）。

经典策略梯度（REINFORCE）面临两个问题：

1. **步长（学习率）敏感**：沿着 `∇ log π_θ(a|s) · R` 方向更新，步长过大一步就把策略推坏，之后很难恢复；
2. **样本效率低**：每个样本只能用一次（on-policy），方差大，需要大量采样。

TRPO 用"约束每次更新前后策略的 KL 距离不超过 δ"来保证安全更新，但要解二阶近似（共轭梯度 + Fisher 矩阵），计算复杂、与神经网络 + 大规模分布式训练难配合。PPO 的目标是**用一阶方法近似 TRPO 的安全性**：实现简单、稳定、计算开销小。

## 二、核心思想

PPO 的核心是 **clip（裁剪）替代约束**：允许用旧策略采样的数据做多次梯度更新（重要性采样），但把每次更新的幅度限制在 `[1-ε, 1+ε]` 之内。一旦新旧策略偏离太远，梯度就被"截断"，防止一步把策略推崩。

- **重要性采样**：旧策略 π_θold 采样的轨迹，用比率 `π_θ / π_θold` 修正后可以近似当前策略的期望——这就是"off-policy 复用、on-policy 意图"；
- **clip 目标（surrogate objective）**：对使策略偏离过大的方向直接封顶；
- 配合 **GAE** 估计 advantage，进一步降低方差；
- 可选 **自适应 KL 惩罚**（PPO-penalty 变体）。

## 三、数学原理

### 3.1 策略梯度

目标是最大化期望回报：

$$J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_{t}\gamma^t r_t\right],\qquad
\nabla_\theta J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_{t}\nabla_\theta\log\pi_\theta(a_t|s_t)\,\widehat{A}_t\right]$$

其中 $\widehat{A}_t$ 是优势函数估计：该动作比"平均水平"好多少。

### 3.2 重要性采样比率

用旧策略 π_θold 采样，对第 t 步：

$$r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t|s_t)}$$

$r_t(\theta)>1$ 表示新策略更可能做出该动作。

### 3.3 PPO-clip 目标

$$L^{\mathrm{CLIP}}(\theta)=\mathbb{E}_t\left[\min\Big(r_t(\theta)\widehat{A}_t,\ \mathrm{clip}\big(r_t(\theta),1-\varepsilon,1+\varepsilon\big)\widehat{A}_t\Big)\right]$$

- 当 $\widehat{A}_t>0$（好动作）：鼓励增大该动作概率，但 $r_t$ 超过 $1+\varepsilon$ 的部分被 clip 掉，不会"过度兴奋"；
- 当 $\widehat{A}_t<0$（坏动作）：鼓励降低概率，但 $r_t$ 低于 $1-\varepsilon$ 的部分被 clip 掉；
- `min` 取两者较小者，保证最终目标永远**不优于**被 clip 的保守目标（pessimistic bound），这是 PPO 稳定性的关键。

### 3.4 GAE 优势估计

GAE（Generalized Advantage Estimation）用指数加权平均多条 n-step 收益，在**偏差-方差**之间插值（λ 越大越偏向低偏差、高方差）：

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

$$\widehat{A}_t^{\mathrm{GAE}(\gamma,\lambda)} = \sum_{l=0}^{T-t-1}(\gamma\lambda)^l\,\delta_{t+l}$$

- λ=0 等价于 TD(0)（高偏差低方差）；λ=1 等价于蒙特卡洛收益（低偏差高方差）；
- 需要价值模型 $V$（critic）提供基线。

### 3.5 完整目标

$$L(\theta)=L^{\mathrm{CLIP}}(\theta) - c_1\,L^{V}(\theta) + c_2\,H[\pi_\theta]$$

- 价值损失：$L^V=\mathbb{E}_t\big[(V_\theta(s_t)-\widehat{R}_t)^2\big]$，$\widehat{R}_t$ 为回报目标（可用 GAE target：$\widehat{A}_t+V(s_t)$），可加 clip 防止价值模型更新过猛；
- 熵奖励 $H[\pi_\theta]$：防止策略过早坍缩到确定性分布（LLM 场景对应"熵崩塌"问题）。

### 3.6 PPO-penalty（自适应 KL）变体

$$L^{\mathrm{KL}}(\theta)=\mathbb{E}_t\left[\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t|s_t)}\widehat{A}_t\right]-\beta\,\mathrm{KL}[\pi_\theta\|\pi_{\theta_{\mathrm{old}}}]$$

每轮根据实际 KL 与目标 KL 的比值调整 β（偏大则减半、偏小则加倍），等价于"软版 TRPO"。

## 四、算法过程

```
1. 初始化策略 π_θ、价值模型 V_φ
2. 循环（每轮）：
   a. 用 π_θold 与环境交互，收集一批轨迹 {(s_t, a_t, r_t)}，保存 log π_θold(a_t|s_t)
   b. 用 V_φ 与 GAE 计算每个动作的 advantage Â_t
   c. 更新价值模型：最小化 (V_φ(s_t) - target_t)^2
   d. 将这批数据切分 minibatch，多 epoch（如 3-4）更新策略：
      L = E[min(r_t·Â, clip(r_t, 1±ε)·Â)] - c1·L_V + c2·H
   e. 清空 buffer；π_θold ← π_θ，进入下一轮
```

## 五、LLM 场景中的 PPO（token 级）

把"生成一个回复"建模为 MDP：状态 = prompt + 已生成前缀，动作 = 下一个 token。奖励通常只在序列末尾给出（outcome reward），于是：

- **损失按 token 求和并除以序列长度**：

$$L^{\mathrm{CLIP}}=\mathbb{E}_{x\sim\mathcal{D},y\sim\pi_{\theta_{\mathrm{old}}}}\left[\frac{1}{|y|}\sum_{t=1}^{|y|}\min\Big(r_t(\theta)\widehat{A}_t,\ \mathrm{clip}(r_t(\theta),1-\varepsilon,1+\varepsilon)\widehat{A}_t\Big)\right]$$

- 每个 token 的 $r_t(\theta)=\pi_\theta(o_t|x,o_{<t})/\pi_{\theta_{\mathrm{old}}}(o_t|x,o_{<t})$，attention mask 之外的 padding 不参与 loss；
- **KL 正则**通常加在 reward 上（对参考策略 π_ref）：`r_total = r - β·KL(π_θ || π_ref)`（per-token 累计）；
- value model 与 policy 同规模，训练成本约为两倍模型；
- 工程实现（如 verl 的 `compute_gae_advantage_return` / `compute_policy_loss`）：双 clip（对负 advantage 单独设更紧的下界）、token 级 mask、对 value loss 做 clip 都是常见增强。

## 六、主要变体

| 变体 | 改动 |
|---|---|
| PPO-clip | 用 clip 目标（默认） |
| PPO-penalty | 用自适应 KL 惩罚替代 clip |
| Dual-clip | 对负 advantage 的 clip 下界更紧，防止熵崩塌（LLM 场景常用） |
| Clip-range 调度 | ε 随训练衰减 |
| Vanilla PPO（LLM） | 去掉 per-token 折扣、只用序列级 reward + GAE |

## 七、优缺点与适用场景

**优点**
- 通用、稳定、样本效率高（价值基线大幅降方差）；
- 理论成熟、生态最大（verl/RLHF 全家桶都优先支持）；
- 可处理稠密奖励、过程奖励、多步交互（agentic）场景。

**缺点**
- 需要训一个同规模 value model：显存、算力翻倍，且**价值估计偏差**在长 CoT、稀疏奖励下会放大（见 [[08-VAPO]]）；
- 对超参（ε、λ、学习率）敏感，工程调参成本高。

**适用场景**：通用 RLHF；有稠密/过程奖励的任务；对稳定性和样本效率要求高的场景；需要 critic 提供细粒度优势的长推理任务（配合 VAPO 类改进）。

## 八、参考

- PPO：Schulman et al., *Proximal Policy Optimization Algorithms*, arXiv:1707.06347
- TRPO：Schulman et al., *Trust Region Policy Optimization*, arXiv:1502.05477
- GAE：Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation*, arXiv:1506.02438
- InstructGPT（PPO 用于 LLM）：arXiv:2203.02155
- 本地参考实现：`verl/verl/trainer/ppo/core_algos.py`（`compute_gae_advantage_return`、`compute_policy_loss`）、`Relax/relax/utils/training/ppo_utils.py`（`vanilla_gae`、`chunked_gae`）
