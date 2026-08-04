# KTO（Kahneman-Tversky Optimization，前景理论优化）

> 论文：Ethayarajh et al., *KTO: Model Alignment as Prospect Theoretic Optimization*, arXiv:2402.01306（ICML 2024）
> 相关笔记：[[03-DPO（直接偏好优化）]]

## 一、定位与动机

DPO 需要**成对偏好数据** `(x, y_w, y_l)`，但真实世界的用户反馈绝大多数是**二值信号**：点赞/点踩、接受/拒绝、通过/不通过。KTO 的目标：**只用"这个回复好不好"的二值标签**完成对齐，不需要配对。

理论动机更深刻：论文提出 **HALO（Human-Aware Loss Functions）**——LLM 对齐损失之所以有效，部分原因在于它们隐式编码了人类的认知偏差。KTO 直接把 Kahneman-Tversky 前景理论（prospect theory）中的人类价值函数搬进损失函数：

- **参考点依赖**：人不是评价绝对好坏，而是相对于某个"预期水平"评价得失；
- **损失厌恶（loss aversion）**：同样大小的损失比收益更刺痛（经典参数 λ≈2.25）；
- **边际敏感递减**：离参考点越远，同样增量带来的感受变化越小。

## 二、核心思想

对每个样本 `(x, y)` 只标 `y` 是 **desirable（可取）** 还是 **undesirable（不可取）**。KTO 定义：

1. **隐式奖励**（同 DPO）：$r_{\mathrm{KTO}}(x,y)=\beta\log\dfrac{\pi_\theta(y|x)}{\pi_{\mathrm{ref}}(y|x)}$；
2. **参考点**：不是"同一 prompt 的另一个回复"，而是"所有见过的人机交互的平均水平"：

$$z_{\mathrm{ref}}=\mathbb{E}_{x'\sim\mathcal{D}}\left[\beta\,\mathrm{KL}\big(\pi_\theta(y'|x')\,\|\,\pi_{\mathrm{ref}}(y'|x')\big)\right]$$

3. **价值函数 v_KTO**：把隐式奖励与参考点的差套上 sigmoid——desirable 样本要求"高于参考点越多越好"，undesirable 样本要求"低于参考点越多越好"（注意方向）。

## 三、数学原理

### 3.1 损失函数（论文 Eq. 7）

$$\mathcal{L}_{\mathrm{KTO}}(\pi_\theta,\pi_{\mathrm{ref}})=\mathbb{E}_{x,y\sim\mathcal{D}}\Big[w(y)\big(1-v_{\mathrm{KTO}}(x,y;\beta)\big)\Big]$$

其中：

$$r_{\mathrm{KTO}}(x,y)=\beta\log\frac{\pi_\theta(y|x)}{\pi_{\mathrm{ref}}(y|x)}$$

$$z_{\mathrm{ref}}=\mathbb{E}_{x'\sim\mathcal{D}}\Big[\beta\,\mathrm{KL}\big(\pi_\theta(y'|x')\,\|\,\pi_{\mathrm{ref}}(y'|x')\big)\Big]$$

$$v_{\mathrm{KTO}}(x,y;\beta)=
\begin{cases}
\sigma\big(r_{\mathrm{KTO}}(x,y)-z_{\mathrm{ref}}\big), & y\sim y_{\mathrm{desirable}}|x\\
\sigma\big(z_{\mathrm{ref}}-r_{\mathrm{KTO}}(x,y)\big), & y\sim y_{\mathrm{undesirable}}|x
\end{cases}$$

$$w(y)=
\begin{cases}
\lambda_{D}, & y\sim y_{\mathrm{desirable}}|x\\
\lambda_{U}, & y\sim y_{\mathrm{undesirable}}|x
\end{cases}$$

### 3.2 直观理解

- **desirable 样本**：想让 $v=\sigma(r-z_{\mathrm{ref}})\to 1$，即把该回复的隐式奖励抬到参考点之上（同时 KL 项会水涨船高，逼着模型只提升"真正可取"的方向——论文称为"必须学到什么让输出可取"）；
- **undesirable 样本**：想让 $v=\sigma(z_{\mathrm{ref}}-r)\to 1$，即把该回复压到参考点之下；
- **饱和保护（Theorem 1）**：当 desirable 样本的奖励已经远高于参考点（或 undesirable 样本已远低于参考点）时，$v\to 1$、损失趋零，**模型不再从"已经足够好/足够坏"的样本学习**——这是与普通二分类交叉熵的关键区别，也是 KTO 抗噪声的机制。

### 3.3 KL 项的工程实现

论文不直接算 $z_{\mathrm{ref}}$ 的期望，而是**在 batch 内用不配对样本估计**（每个 $x'$ 配一个无关的 $y'_U$）：

$$z_{\mathrm{ref}}\approx \max\left(0,\ \frac{1}{m}\sum_{i=1}^{m}\log\frac{\pi_\theta(y'_{U,i}|x'_i)}{\pi_{\mathrm{ref}}(y'_{U,i}|x'_i)}\right)$$

**KL 项不反传梯度**（stop-gradient），只作为"饱和程度控制器"；这使训练显著更稳。

### 3.4 超参数

- $\beta=0.1$：接近最优（同 DPO 的 β 语义：越小越允许偏离 π_ref）；
- 权重平衡：$\dfrac{\lambda_{D}\,n_{D}}{\lambda_{U}\,n_{U}}\in[1,\frac{4}{3}]$，且至少一个为 1。
  - 1:1 数据 → $\lambda_U=1,\ \lambda_D\in[1,1.33]$；
  - 丢掉 90% desirable 后（10:1 不平衡）→ $\lambda_U=1,\ \lambda_D\in[10,13.33]$。

## 四、算法过程

```
1. 准备二值标注数据 {(x, y, desirable/undesirable)}（点赞/点踩、通过/失败均可）
2. 冻结 π_ref，初始化 π_θ
3. 对每个 batch：
   a. 计算 π_θ 与 π_ref 的 logprob，得到隐式奖励 r_KTO(x, y)
   b. 用 batch 内不配对样本估计 KL 参考点 z_ref（stop-grad）
   c. 按 desirable/undesirable 分别算 v_KTO 与损失，乘以 w(y) 权重
   d. 反向传播更新 π_θ
4. 全程 offline，无采样、无 RM
```

## 五、与 DPO 的关系与对比

- **数据转换**：一组 DPO 配对数据 `(x, y_w, y_l)` 可以拆成 2n 个 KTO 样本（`y_w` 标 desirable、`y_l` 标 undesirable）。论文发现：**用拆出来的二值数据训练 KTO 甚至能优于原配对的 DPO**；
- **极端不平衡**：论文实验显示 KTO 在 desirable 样本减少 90% 时仍能匹配 DPO 性能——二值信号 + 权重平衡天然抗数据倾斜；
- **无需配对**：真实产品反馈（点赞率）直接可用，数据管线大幅简化。

| 维度 | DPO | KTO |
|---|---|---|
| 数据 | 配对偏好 | 二值信号（可从不配对数据构建） |
| 参考点 | 同一 prompt 的 $y_l$ | 全局 KL 期望 $z_{\mathrm{ref}}$ |
| 损失形态 | log-sigmoid（pairwise） | 加权 1-σ（pointwise + 参考点） |
| 抗噪声 | 对错误配对敏感 | 有饱和保护，更抗噪 |
| 训练目标 | 最大化偏好似然 | 最大化"前景理论意义上的效用" |

## 六、优缺点与适用场景

**优点**：数据要求最低（二值标签，工业界唾手可得）；抗噪声、抗数据不平衡；离线训练稳定；DPO 数据可直接复用。

**缺点**：pointwise 信号信息量低于 pairwise，理论上限略低；参考点估计（batch KL）对 batch 组成敏感；同样存在 offline 对齐的分布外问题。

**适用场景**：对话/写作等**没有规则验证器**的对齐任务；用户反馈只有赞/踩的工业场景；数据量少、噪声大的冷启动对齐。不适合需要强推理探索的任务（应选 GRPO/GSPO 系）。

## 七、参考

- KTO：Ethayarajh et al., arXiv:2402.01306
- 前景理论：Kahneman & Tversky, *Prospect Theory: An Analysis of Decision under Risk*, Econometrica 1979；Tversky & Kahneman, *Advances in Prospect Theory*, 1992
- 实现参考：verl `dpo_trainer.py`（KTO 模式）、TRL `KTOTrainer`
