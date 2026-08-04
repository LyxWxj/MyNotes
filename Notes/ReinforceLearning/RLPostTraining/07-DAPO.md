# DAPO（解耦裁剪与动态采样策略优化，Decoupled Clip and Dynamic sAmpling Policy Optimization）

> 论文：Yu et al. (ByteDance Seed), *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*, arXiv:2503.14476（2025）
> 相关笔记：[[02-GRPO（组相对策略优化）]]、[[08-VAPO]]、[[04-GSPO（组序列策略优化）]]

## 一、定位与动机

DeepSeek-R1 证明了 GRPO 系算法在**千亿参数、超长 CoT** 上可行，但社区复现时发现：**中小规模（7B~30B）模型用 GRPO 训练要么不收敛、要么性能远低于预期**。DAPO 论文系统分析了 GRPO 在小规模 + 大规模训练中的四大失败模式，并给出四项针对性修复，使 **Qwen2.5-32B 在 AIME 2024 上达到 50 分**（同规模 GRPO 无法稳定收敛；DAPO 同时是第一个开源全量 RL 训练 32B 推理模型的系统）。

四大问题：

1. **熵崩塌（entropy collapse）**：训练后期策略分布坍缩到几乎确定，探索停止，loss 表面看还在下降但性能停滞；
2. **奖励噪声 / 低信噪比（low SNR）**：非正确组的优势值趋近零，梯度信号微弱、不稳定；
3. **长度偏差（length bias / reward hacking）**：模型学会"多写废话"来提高正确率，回复越拉越长；
4. **训练不稳定**：clip 不对称、归一化方式不当导致梯度爆炸或消失。

## 二、四大技术

### 2.1 Clip-Higher（非对称裁剪）

标准 clip 是**对称**的 `[1-ε, 1+ε]`。DAPO 发现：对称 clip 对"负优势侧"过度抑制，且正优势侧的更新幅度被 ε=0.2 限制住，导致训练后期推进力不足、熵崩塌。

解法：**上下界解耦**：

$$\mathrm{clip}\big(r_{i,t}(\theta),\,1-\varepsilon_{\mathrm{low}},\,1+\varepsilon_{\mathrm{high}}\big),\qquad \varepsilon_{\mathrm{low}}=0.2,\ \varepsilon_{\mathrm{high}}=0.28$$

正优势方向允许更大的更新步（1.28 vs 0.8），给"好样本"更大的推进空间，同时保持对负方向的保护。

### 2.2 Dynamic Sampling（动态采样）

组内归一化有一个退化情形：如果一组回复**全部正确或全部错误**，则 $\widehat{A}_i=0$（或数值不稳定），这组数据对训练毫无贡献，还引入噪声。

解法：**过滤掉全对/全错的组**，只保留"组内存在分歧"的样本：

$$0<|\{o_i:\mathrm{is\_equivalent}(a,o_i)\}|<G$$

其中 $a$ 为标准答案，$\mathrm{is\_equivalent}$ 为答案等价判断。等价地，丢弃 $\max_i\widehat{A}_i=\min_i\widehat{A}_i$（全同奖励）的组。

### 2.3 Token-Level Policy Gradient Loss（token 级损失）

GRPO 原版损失对每条回复的 token loss 先求平均、再对组求平均：

$$\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\min\big(r_{i,t}\widehat{A}_{i,t},\ \mathrm{clip}(r_{i,t})\widehat{A}_{i,t}\big)$$

这导致**每条回复（无论长短）对梯度的贡献相同**：长回复中每个 token 的梯度被稀释，短回复的 token 反而被放大——等价于给短回复更高的权重，诱导模型走捷径。

解法：**统一按总 token 数归一化**：

$$\mathcal{J}_{\mathrm{DAPO}}(\theta)=\mathbb{E}\left[\frac{1}{\sum_i |o_i|}\sum_{i=1}^{G}\sum_{t=1}^{|o_i|}\min\Big(r_{i,t}(\theta)\widehat{A}_{i,t},\ \mathrm{clip}\big(r_{i,t}(\theta),1-\varepsilon_{\mathrm{low}},1+\varepsilon_{\mathrm{high}}\big)\widehat{A}_{i,t}\Big)\right]$$

每个 token 的梯度贡献与回复长度无关，长度偏差从目标函数层面被消除。

### 2.4 Overlong Reward Shaping（超长回复奖励塑形）

对超出长度上限的回复，不能直接截断（会丢分），也不该只给 0（模型会学到"超长无害"）。DAPO 用**分段线性惩罚**（论文 Eq. 13）：

$$R_{\mathrm{length}}(y)=
\begin{cases}
0, & |y|\le L_{\max}-L_{\mathrm{cache}}\\
\dfrac{(L_{\max}-L_{\mathrm{cache}})-|y|}{L_{\mathrm{cache}}}, & L_{\max}-L_{\mathrm{cache}}<|y|\le L_{\max}\\
-1, & |y|>L_{\max}
\end{cases}$$

- 前 $L_{\max}-L_{\mathrm{cache}}$ 个 token 完全不受惩罚（保证思考空间）；
- 中间缓冲区间内线性衰减到 -1（梯度平滑，模型学会自己收敛长度）；
- 超过 $L_{\max}$ 直接 -1（硬惩罚）。

实验设置：$L_{\max}=16384$、$L_{\mathrm{cache}}=4096$、最大生成长度 20480 token。

## 三、完整目标与算法过程

完整目标（含规则奖励与 KL）：

$$\mathcal{J}_{\mathrm{DAPO}}=\mathbb{E}\left[\frac{1}{\sum_i|o_i|}\sum_{i,t}\min\big(r_{i,t}\widehat{A}_{i,t},\ \mathrm{clip}(r_{i,t},1-\varepsilon_{\mathrm{low}},1+\varepsilon_{\mathrm{high}})\widehat{A}_{i,t}\big)\right]-\beta\,\mathbb{D}_{\mathrm{KL}}[\pi_\theta\|\pi_{\mathrm{ref}}]$$

其中 $r_{i,t}(\theta)=\dfrac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{\mathrm{old}}}(o_{i,t}|q,o_{i,<t})}$，$\widehat{A}_{i,t}=\dfrac{R_i-\mathrm{mean}(\{R_i\})}{\mathrm{std}(\{R_i\})}$，$R_i$ 为规则奖励（正确 +1 / 错误 -1）加长度塑形。

```
1. 采样组（G 个回复/prompt），旧 logprobs
2. 规则验证器打分 + Overlong Reward Shaping
3. Dynamic Sampling：丢弃全对/全错组
4. 组内归一化 advantage
5. Token-Level loss（总 token 归一化）+ Clip-Higher + KL
6. 多 epoch minibatch 更新
```

## 四、实验结果与消融

（Qwen2.5-32B，AIME 2024）

| 方法 | 关键结果 |
|---|---|
| 基线 GRPO | 不收敛 / 性能远低于预期（熵崩塌） |
| + Dynamic Sampling | 训练更稳，但后期熵仍下降 |
| + Clip-Higher | 熵崩塌显著缓解，探索持续 |
| + Token-Level Loss | 长度偏差下降（平均长度回落），训练曲线更平滑 |
| + Overlong Shaping | 超长回复比例大幅下降，奖励更干净 |
| **完整 DAPO** | **AIME 2024 50 分**；过程监督 vs 结果监督：过程监督更强但更贵 |

论文还强调"熵"是训练健康度的核心指标：GRPO 系算法训练时需持续监控 per-token 熵，熵崩则性能上限被锁死。

## 五、与 GRPO / VAPO 的关系

- **对 GRPO**：DAPO 的四项技术全部可以"插拔"到 GRPO 上，因此它是 GRPO 的**工程增强版**而非新框架；
- **对 VAPO**：VAPO（见 [[08-VAPO]]）吸收了两项 DAPO 技术（Clip-Higher、Token-Level Loss），再补上价值模型侧的四项（Value Pretraining、Decoupled-GAE、Length-Adaptive GAE、Positive Example LM Loss），把 PPO 系带到 AIME 60.4——**DAPO 与 VAPO 是"无价值 vs 有价值"两条路线各自的 SOTA**；
- **对 GSPO**：DAPO 的 token 级 IS 在超长序列上仍有方差问题，GSPO 的序列级 IS 可与 DAPO 技巧组合（社区已广泛混合使用）。

## 六、优缺点与适用场景

**优点**：四项技术均有清晰的问题-解法对应，可独立消融、按需启用；开源系统（含训练代码）；中小规模即可复现 SOTA 推理能力。

**缺点**：Dynamic Sampling 会丢弃部分数据（浪费采样）；Overlong Shaping 的 $L_{\max}/L_{\mathrm{cache}}$ 需要按任务调；仍是 outcome 级信号，样本效率不如价值类算法。

**适用场景**：中小规模（≤30B）推理模型 RL；数学/代码等可验证奖励任务；任何"GRPO 不收敛"的排障起点（先加 Dynamic Sampling 和 Clip-Higher）。

## 七、参考

- DAPO：Yu et al., arXiv:2503.14476（官方开源：bytedance/dapo）
- 本地实现参考：verl 的 DAPO 配置（`dapo` 相关的 clip-higher、动态采样、overlong shaping 开关）
