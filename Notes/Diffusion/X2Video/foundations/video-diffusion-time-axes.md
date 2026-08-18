---
type: Note
related_to: "[[X2Video]]"
status: Active
---

# 视频扩散中的两个时间轴：帧序列、去噪与 rollout

视频扩散文献中的 `t` 容易与“第 `t` 帧”混淆。实际至少存在两个独立时间轴：视频时间轴用帧索引 `i` 表示，扩散时间轴用 `t` 表示噪声强度或去噪进度。

## 两个时间轴

```text
视频时间轴：    帧 1 -> 帧 2 -> 帧 3 -> 帧 4
                 i

扩散时间轴：    噪声 x_1 -> ... -> 干净 x_0
                 t=1          t=0
```

`x_t^i` 表示“视频第 `i` 帧在扩散时间 `t` 下的状态”。因此，`i` 决定当前生成哪一帧，`t` 决定这一帧处于多大噪声。

## PF-ODE 轨迹

PF-ODE（Probability Flow ODE）是扩散模型在连续时间上的确定性去噪路径。对当前帧或 chunk，它描述：

```text
x_1^i -> x_0.9^i -> x_0.8^i -> ... -> x_0^i
```

这条轨迹沿的是扩散时间 `t`，不是“从第 1 帧走到第 2 帧”的视频序列时间。视频模型的速度场仍可条件化于历史帧、未来帧、文本或动作：

- AR 教师/学生通常使用 `(x_t^i, x^{<i}, t)`；
- 双向教师可能使用完整视频上下文 `(x_t^i, x_t^{<i}, x_t^{>i}, t)`。

因此，“教师轨迹”通常指一帧或一个 chunk 在扩散时间轴上的状态序列；它不是单纯的帧序列生成日志。

## 完整 AR 视频生成是两层循环

```text
外层：帧序列生成
for i = 1 ... N:
    内层：当前帧的扩散去噪
    for diffusion step:
        denoise frame i
```

少步蒸馏主要减少内层扩散调用；AR 架构负责外层帧序列的因果展开。

## 学生学习教师的什么

在因果蒸馏中，学生通常拟合当前帧的条件流映射：

$$
G_\theta(x_t^i, x^{<i}, t) \rightarrow x_0^i。
$$

含义是：给定当前帧在扩散时间 `t` 的噪声状态，以及已经生成的历史帧，预测该帧的干净结果。多步学生需要多次沿 `t` 调用；少步学生则试图用 1、2 或 4 次调用完成同一帧。

同时，AR 模型还要学习外层序列条件关系：

$$
p(x^{1:N}) = \prod_i p(x^i \mid x^{<i})。
$$

这两个学习目标分别对应“当前帧如何去噪”和“下一帧如何依赖历史”，不能混为同一个时间过程。

## Teacher forcing 与 student self-rollout

### Teacher forcing

训练第 `i` 帧时使用真实视频前缀：

$$
p_\theta(x^i \mid x_{\mathrm{gt}}^{<i})。
$$

例如第 3 帧依赖真实的第 1、2 帧。它的优点是训练稳定，缺点是推理时模型会看到自己生成的、可能带误差的历史。

### Student self-rollout

推理或 Self Forcing 训练时，后续帧使用学生自己已经生成的前缀：

```text
y^1：学生生成
y^2：条件为 y^1
y^3：条件为 y^1、y^2
```

公式为：

$$
\hat{x}^i \sim p_\theta(\hat{x}^i \mid \hat{x}^{<i})。
$$

它沿视频时间轴 `i` 展开，和沿扩散时间 `t` 的 PF-ODE 积分是两回事。Self Forcing 在 DMD 阶段使用 self-rollout，是为了让训练看到与推理相同的历史分布，减轻 exposure bias。

## Causal Forcing 的轨迹蒸馏

Causal Forcing 先训练 AR 扩散教师，然后对第 `i` 帧在真实前缀条件下生成 PF-ODE 轨迹：

```text
x_1^i -> x_t1^i -> x_t2^i -> ... -> x_0^i
```

学生学习这些扩散时间点上的配对关系，例如 `(x_t^i, x_0^i)`。Causal Forcing++ 进一步只让 AR 教师在线执行一次相邻时间步的 ODE 更新，把 `t` 与 `t - Δt` 的预测做一致性匹配，从而避免离线保存整条轨迹。

## Causal Forcing 系列的三阶段管线

可以把 Causal Forcing 和 Causal Forcing++ 看作一条分工明确的流水线：先得到可流式生成的多步 AR 教师，再训练低延迟少步 AR 学生，最后用质量更强的双向模型细化学生分布。

```text
双向多步视频扩散模型
  -> Stage 1：teacher forcing AR diffusion training
  -> 多步 AR 教师
  -> Stage 2：causal ODE 或 causal CD
  -> 少步 AR 学生初始化
  -> Stage 3：asymmetric DMD + student self-rollout
  -> 高质量、低延迟的少步 AR 视频生成器
```

### Stage 1：AR diffusion training

起点是多步的双向视频扩散模型。它生成第 `i` 帧时可以利用前后帧，质量高，却不能真正流式交互。Stage 1 用 teacher forcing 将其微调为因果 AR 扩散教师：将干净视频前缀与当前帧的噪声副本组织为输入，并施加 causal attention mask，使模型只能看历史。

训练当前帧时，条件是干净真实前缀：

$$
p_\phi(x^i \mid x_{\mathrm{gt}}^{<i},c)。
$$

其中 `c` 是文本等条件。得到的 AR 教师已经能够按帧序列生成，但每帧仍要多步去噪；而且训练时使用真实历史、推理时使用自身历史，仍有 exposure bias。

```text
训练第 i 帧：真实前缀 x_gt^<i + 当前噪声帧 x_t^i
推理第 i 帧：已生成前缀 x_hat^<i + 当前噪声帧 x_t^i
```

### Stage 2a：causal ODE initialization

用 Stage 1 的 AR 教师为少步 AR 学生建立初始化。固定真实历史 `x_{\mathrm{gt}}^{<i}` 后，AR 教师会为当前帧生成 PF-ODE 去噪轨迹：

```text
x_1^i -> x_t1^i -> x_t2^i -> ... -> x_0^i
```

随机采样轨迹中的时间点 `t`，让学生直接回归该轨迹的终点：

$$
\min_\theta\;\mathbb{E}\left[
\left\|G_\theta(x_t^i,x_{\mathrm{gt}}^{<i},t)-x_0^i\right\|^2
\right]。
$$

学生由此学到“给定历史和当前噪声状态，该帧应流向哪个干净结果”。它仍按帧自回归，但当前帧内部可用少数扩散调用完成。因果 ODE 初始化的不足是：必须离线求解、保存大量完整 PF-ODE 轨迹。

### Stage 2b：causal CD initialization

Causal Forcing++ 用 causal consistency distillation（causal CD）替代离线轨迹蒸馏。训练时无需保存整条轨迹，而是：

1. 从真实第 `i` 帧构造 `x_t^i`；
2. AR 教师在相同真实前缀条件下仅执行一次 ODE 更新，得到 $\hat{x}_{t-\Delta t}^i$；
3. 让学生在 `t` 时的预测与 EMA 学生 $\theta^-$ 在相邻时刻 `t-Δt` 的预测一致。

$$
\min_\theta\;\mathbb{E}\left[
w(t)d\left(
G_\theta(x_t^i,x_{\mathrm{gt}}^{<i},t),
G_{\theta^-}(\hat{x}_{t-\Delta t}^i,x_{\mathrm{gt}}^{<i},t-\Delta t)
\right)
\right]。
$$

直觉上，同一条教师 PF-ODE 轨迹上相邻的两个状态，最终必须对应相同的干净帧：

```text
x_t^i ----------------------> x_0^i
  | 教师执行一次 ODE 步
  v
x_(t-Δt)^i ------------------> x_0^i
```

理想条件下，causal CD 与 causal ODE 蒸馏学习同一个 AR 条件流映射；差异在监督方式。前者使用在线局部配对，避免离线轨迹的生成和存储，也使单次优化跨越的时间间隔更小。有限模型容量、有限训练和数值误差下，两者的参数与实际结果不必严格相同。

### Stage 3：asymmetric DMD

Stage 2 的学生已经能少步、逐帧生成，但其质量上限仍受 AR 教师限制。Stage 3 用高质量双向扩散模型进行 distribution matching，让学生向更优的视频分布靠拢。

它之所以称为 **asymmetric**，是因为角色不对称：

| 角色 | 架构 | 职责 |
|---|---|---|
| 学生 $G_\theta$ | 因果 AR | 低延迟、按帧生成视频 |
| $s_{\mathrm{real}}$ | 冻结的双向扩散模型 | 估计真实高质量视频分布的 score |
| $s_{\mathrm{fake}}$ | 在线训练的 score 模型 | 估计学生生成分布的 score |

学生首先执行 self-rollout，得到完整的自身生成视频 $\tilde{x}$：

```text
y^1：学生生成
y^2：以 y^1 为条件生成
y^3：以 y^1、y^2 为条件生成
...
```

再将 $\tilde{x}$ 加噪为 $\tilde{x}_t$，DMD 用两个 score 的差更新学生：

$$
\nabla_\theta\mathbb{E}_t[D_{\mathrm{KL}}(p_{\theta,t}(\tilde{x}_t)\|p_{\mathrm{data},t}(\tilde{x}_t))]
=-\mathbb{E}\left[
\left(s_{\mathrm{real}}(\tilde{x}_t,t)-s_{\mathrm{fake}}(\tilde{x}_t,t)\right)
\frac{\partial\tilde{x}}{\partial\theta}
\right]。
$$

score 差给出“学生视频应往真实高质量视频分布移动的方向”。self-rollout 使 Stage 3 训练时见到的前缀也是学生自身生成结果，因此直接针对推理时的 exposure bias。

> [!important] 两个阶段使用历史的区别
> - **Stage 2**：通常条件化于真实前缀 $x_{\mathrm{gt}}^{<i}$，稳定学习正确的 AR 条件流映射。
> - **Stage 3**：条件化于学生前缀 $\tilde{x}^{<i}$，处理真实 rollout 时不断累积的历史误差。

### 相机可控的版本

若相机位姿或动作记为 `a`，原有条件分布从

$$
p(x^i\mid x^{<i},c)
$$

扩展为：

$$
p(x^i\mid x^{<i},c,a)。
$$

相机条件必须贯穿全部阶段：

| 阶段 | 相机条件的作用 |
|---|---|
| Stage 1 | 用相机可控数据将双向模型微调为相机可控 AR 教师 |
| Stage 2 | AR 教师在相同相机条件下求 PF-ODE 或单步 ODE；学生学习同一条件下的少步生成 |
| Stage 3 | 学生按文本和相机条件 self-rollout，$s_{\mathrm{real}}$ 与 $s_{\mathrm{fake}}$ 也接收相同条件 |

否则学生与 score 模型会针对不同的相机运动评估同一视频，DMD 梯度不再对应所需的条件分布。

## 双向教师为何造成错配

双向教师在生成第 `i` 帧时可能使用未来帧，而 AR 学生只能看到历史。如果固定当前噪声状态，却改变未来上下文，双向教师可能产生不同的干净目标；学生没有足够输入区分这些目标，只能回归到条件均值，导致模糊或不对齐。这就是帧级注入性问题。

> `i` 决定生成视频中的哪一帧，`t` 决定当前帧去噪到哪一步；PF-ODE 走 `t`，AR rollout 走 `i`，student self-rollout 是在 `i` 轴上使用学生自己的历史。
