# GRPO 稳定训练排障清单

> 适用对象：以 GRPO 为基线的推理 RL（数学/代码等可验证奖励任务）。GSPO、DAPO、VAPO 的改进项在文中以"修复手段"给出，原理见对应笔记。
> 相关笔记：[[02-GRPO（组相对策略优化）]]、[[04-GSPO（组序列策略优化）]]、[[07-DAPO]]、[[08-VAPO]]、[[09-SFT到RL全流程示例配置]]

## 一、使用说明

按"症状"查表 → 看根因 → 选修复手段。**每次只改一个变量**，改完至少观察几十步（batch 小则上百步）再下结论。核心原则：RL 训练曲线不是"越降越好"，要同时盯住 **熵、长度、KL、reward** 四张曲线。

## 二、症状 → 根因 → 修复

### 1. 训练崩溃 / 发散（NaN、梯度爆炸）

| 根因 | 修复手段 |
|---|---|
| 学习率过大 | `actor.optim.lr` 降到 1e-6~3e-6 量级；PPO 的 critic 用 1e-5 量级 |
| 梯度爆炸 | 开 `grad_clip`（如 1.0）；检查 loss 是否未按 token 数归一化 |
| padding 参与计算 | `model.use_remove_padding=True`；确认 mask 覆盖所有 padding token |
| 数据污染 | 检查超长 prompt/空 completion/奖励函数异常值；`data.truncation='error'` 先暴露问题 |
| token 级 IS 方差累积 | 切 GSPO 序列级 IS（`loss_mode=gspo`，clip 3e-4/4e-4）——长序列崩溃的头号修复 |
| 双 clip 未开 | `clip_ratio_c`（如 10.0）：负优势侧更紧的下界，防止"压低概率"时梯度爆炸 |

### 2. 熵崩塌（entropy → 0，探索停止）

典型曲线：per-token 熵前期骤降后贴 0，reward 平台期。

| 根因 | 修复手段 |
|---|---|
| 对称 clip 限制正优势更新 | **Clip-Higher**：`clip_ratio_low=0.2`、`clip_ratio_high=0.28`（DAPO/VAPO 同款） |
| KL 惩罚过强 | 降低 `kl_loss_coef`（0.001 → 0.0003）或 `use_kl_in_reward` 的 β；GSPO 实践中可 β=0 |
| 采样温度/top-p 过保守 | rollout `temperature=1.0`、`top_p=1.0`（训练采样不要用 greedy） |
| 数据过易/过难 | prompt 难度失衡：全对组无梯度（见症状 3），全错组也无有效信号 |
| 熵奖励未开 | `entropy_coeff` 试探性给 1e-4~1e-3（大多数 GRPO 配置默认 0，靠其他手段） |

**判断标准**：训练中熵应**缓慢下降**；若 10 步内腰斩，立即排查。

### 3. Reward 不涨 / 停滞

| 根因 | 修复手段 |
|---|---|
| 组内无分歧（全对/全错） | **Dynamic Sampling**：`algorithm.filter_groups.enable=True`（DAPO 技术，过滤全同奖励组） |
| 组太小，优势估计方差大 | `rollout.n` 从 5 → 8~16（组内统计更稳） |
| 奖励信号问题 | 检查 verifier：先离线批量验证 verifier 准确率；正确性 +1/-1 与 +1/0 的分布差异 |
| 优势未归一化 | 确认 `norm_adv_by_std_in_grpo=true`（组内除 std）；或换全局 batch 归一化（GBN） |
| 基线策略太弱 | 先 SFT 再 RL；或加 Positive Example LM Loss（VAPO 的 μ·L_NLL） |
| prompt 与模型能力不匹配 | 用 pass@k 预筛：SFT 后采样看正确率，选 10%~60% 正确率的任务集 |

### 4. 长度失控 / reward hacking（回复无限膨胀）

| 根因 | 修复手段 |
|---|---|
| 长回复获得更高奖励（相关性） | 检查 reward 与长度的相关系数；对 verifier 本身做长度偏差审计 |
| loss 按"每条样本平均" | **Token-Level Loss**：`loss_agg_mode=token-mean`（总 token 归一化，DAPO 技术） |
| 无长度惩罚 | **Overlong Shaping**：`reward.reward_manager.name=dapo` + `overlong_buffer_cfg.len=4096`、`penalty_factor=1.0`（$L_{\max}$ 前不罚、缓冲区内线性衰减、超限 -1） |
| max 长度设太小被截断 | `max_response_length` 要覆盖模型自然生成长度；截断会产生"被迫正确"的假信号 |

### 5. 训练震荡 / 不稳定（loss 或 reward 锯齿状）

| 根因 | 修复手段 |
|---|---|
| off-policy 程度高（多 epoch 更新） | 减少 ppo epoch（3→1~2）；增大 `train_batch_size` 让数据更"新鲜" |
| 序列长度异构剧烈 | GSPO 序列级 IS；或对过长/过短样本分桶（bucket）处理 |
| 学习率无 warmup | `lr_warmup_steps` 给 10~100 步 |
| batch 统计不稳定 | 换全局归一化（GBN，DeepSeek-V3.1 风格）或增大组大小 G |
| 权重同步延迟（分离集群） | 检查 rollout 端策略是否滞后过多（见 [[RL框架设计核心问题与项目对比]] 的 staleness 讨论） |

### 6. MoE / 超大模型 / 超长 CoT 崩

| 根因 | 修复手段 |
|---|---|
| token 级 IS 方差在 MoE 中被路由噪声放大 | **直接换 GSPO**（Qwen3 的结论：MoE RL 天然稳定，无需额外策略） |
| 局部 token 概率突变被 clip 放大 | GSPO 序列级 clip（整条回复一个权重，`clip_ratio_c=10.0`） |

### 7. KL 异常（爆炸或归零）

| 现象 | 修复手段 |
|---|---|
| KL 爆炸（策略偏离 ref 太远） | 提高 `kl_loss_coef`；换 `kl_loss_type=low_var_kl`；确认 ref 模型路径正确（别把 base 当 ref） |
| KL 归零（策略与 ref 几乎相同，学不动） | 降低 kl 系数；检查是否把 `use_kl_loss` 与 `use_kl_in_reward` 同时开了导致双重惩罚 |
| KL 曲线锯齿 | KL 目标调度（`kl_ctrl.type=adaptive`）或用固定小系数 |

### 8. 价值模型问题（PPO/VAPO 路线）

| 现象 | 修复手段 |
|---|---|
| value loss 不降 / 训练不收敛 | **Value Pretraining**：固定策略采样 + λ=1 MC 回报预训练 critic 约 50 步（VAPO） |
| 价值估计有偏（稀疏奖励） | **Decoupled-GAE**：critic λ=1.0、policy λ=0.95（VAPO） |
| 长序列方差大 | **Length-Adaptive GAE**：$\lambda_{\mathrm{policy}}=1-\frac{1}{\alpha l}$，α=0.5（VAPO） |
| critic 与 actor 学习率失衡 | critic LR 通常比 actor 大 5~10 倍（1e-5 vs 1e-6） |
| 显存不足 | critic 开 `param_offload`；或干脆换 GRPO/GSPO 无价值路线 |

### 9. 显存 / 吞吐问题

| 现象 | 修复手段 |
|---|---|
| OOM | `ppo_max_token_len_per_gpu` 调小（动态 batch）；`enable_gradient_checkpointing`；ref/critic 参数 offload |
| 吞吐低 | rollout 开 `enable_chunked_prefill`、调 `gpu_memory_utilization`、`tensor_model_parallel_size`；train/rollout 用 `free_cache_engine` 释放引擎 |
| 长尾样本拖慢整轮 | `use_dynamic_bsz` 按 token 预算切 batch；数据侧过滤超长 prompt |

## 三、监控指标健康区间速查

| 指标 | 健康区间 / 信号 | 危险信号 |
|---|---|---|
| `actor/entropy` | 缓慢下降，训练结束仍 >0.2~0.5（依词表） | 10 步内腰斩 / 贴 0 |
| `reward/mean` | 单调上升，后期平台 | 不涨 + 熵贴 0 |
| `actor/kl` | 稳定在 0.01~0.1 量级 | 数量级漂移、锯齿 |
| 平均回复长度 | 缓慢增长后收敛 | 线性膨胀不停 |
| `clip ratio` | 个位数百分比 | >20%（off-policy 严重） |
| `filter_groups/evicted` | 存在但不过半 | >50% 组被过滤（任务太难/太易） |
| `grad_norm` | 稳定，无尖峰 | 周期性爆尖 |
| 验证集 pass@k | 与训练 reward 同趋势 | 训练涨、验证不涨（过拟合/verifier 泄漏） |

## 四、分步排障流程（决策树）

```
训练出问题
├─ 先看四张曲线：熵 / 长度 / KL / reward
│   ├─ 熵贴 0        → 症状 2（Clip-Higher 优先）
│   ├─ 长度膨胀      → 症状 4（Token-Level Loss + Overlong Shaping）
│   ├─ KL 异常       → 症状 7
│   └─ reward 停滞   → 症状 3（Dynamic Sampling 优先）
├─ 崩溃 / NaN        → 症状 1（先查数据与 mask，再切 GSPO）
├─ 震荡              → 症状 5
├─ 只发生在超大模型  → 症状 6（换 GSPO）
└─ 都没问题但性能差  → 回到数据与奖励：
    1. 离线验证 verifier 准确率
    2. 检查 prompt 难度分布（pass@k 10%~60%）
    3. 检查 SFT 起点质量
```

**稳定训练的标准配方（推荐顺序）**：

1. GRPO 基线 + `use_kl_loss` + `norm_adv_by_std_in_grpo`（跑通冒烟）；
2. 加 Dynamic Sampling（`filter_groups.enable=True`）；
3. 加 Clip-Higher（0.2 / 0.28）+ `clip_ratio_c=10.0`；
4. 加 Token-Level Loss（`loss_agg_mode=token-mean`）；
5. 加 Overlong Shaping（`reward.reward_manager.name=dapo`）；
6. 仍不稳 → 切 GSPO（`loss_mode=gspo`，clip 3e-4/4e-4，β 可设 0）；
7. 追求上限且预算充足 → VAPO 七件套（[[08-VAPO]]）。

## 五、最后的手段

- **回滚 checkpoint**：模型崩溃往往不可逆（GSPO 论文原话：resume 也救不回来），保留每 N 步 checkpoint（`save_freq`），崩溃后回滚 + 改配置重来；
- **缩小规模复现**：用 0.5B~1B 模型 + 小数据集复现问题，迭代成本低一个数量级；
- **先离线后在线**：一切怀疑都先用离线生成 + 静态 reward 脚本验证，再动训练配置。

## 六、参考

- 算法原理：[[02-GRPO（组相对策略优化）]]、[[04-GSPO（组序列策略优化）]]、[[07-DAPO]]、[[08-VAPO]]
- 配置落地：[[09-SFT到RL全流程示例配置]]
- 论文：DAPO arXiv:2503.14476、GSPO arXiv:2507.18071、VAPO arXiv:2504.05118、Dr.GRPO arXiv:2504.08919
