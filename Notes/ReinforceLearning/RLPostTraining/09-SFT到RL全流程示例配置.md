# SFT → RL 全流程示例配置（以 verl 为例）

> 说明：本笔记给出"从数据到可运行的推理 RL"的完整链路与可参考配置。配置项以本地仓库 `verl/examples/` 的真实脚本为准（2026-08 快照），key 名称可能随版本演进；算法原理见本目录各算法笔记。
> 相关笔记：[[02-GRPO（组相对策略优化）]]、[[04-GSPO（组序列策略优化）]]、[[07-DAPO]]、[[08-VAPO]]、[[RL框架设计核心问题与项目对比]]

## 一、总体流程

```
原始数据 ──预处理──▶ parquet 数据集 ──▶ （可选）SFT ──▶ RL 训练
                                                    │
                    ┌───────────────────────────────┤
                    │ 1. 采样（rollout）  2. 奖励计算  3. 优势估计
                    │ 4. 策略更新         5. 权重同步  6. 评估/监控
                    └───────────────────────────────┘
```

以 verl 为例，一条链路对应三段配置：`sft_trainer`（SFT）→ `main_ppo`（RL，统一入口，内部按 `algorithm.adv_estimator` 切换算法）→ 奖励 manager / 评估回调。

## 二、阶段 0：数据准备（parquet 格式）

RL 数据集的**最小必需字段**（verl 约定）：

| 字段 | 说明 | 示例 |
|---|---|---|
| `prompt` | chat 格式消息（list of {role, content}） | `[{"role":"user","content":"..."}]` |
| `data_source` | 数据来源标记（便于统计） | `openai/gsm8k` |
| `reward_model` | 奖励元信息 | `{"style":"rule","ground_truth":"42"}` |
| `ability` | 能力标记（可选） | `math` |
| `extra_info` | 任意附加信息（可选） | `{"split":"train","index":0}` |

预处理脚本参考（本地仓库）：`verl/examples/data_preprocess/gsm8k.py`、`math_dataset.py`、`aime2024_multiturn_w_tool.py`。核心动作：

1. 提取标准答案（GSM8K 取 `#### ` 后的数字；MATH 取最后一个 `\boxed{...}`）；
2. 组装 prompt（可附加指令，如 `Let's think step by step and output the final answer after "####".`）；
3. 导出 parquet（训练/验证分开），`train.parquet` + `test.parquet` 放同一目录。

**提示**：prompt 质量决定 RL 上限。过滤空 prompt、超长 prompt、重复样本；验证集选与训练分布不同（如训练 GSM8K/MATH、验证 AIME），避免过拟合评估集。

## 三、阶段 1：SFT（可选，但强烈推荐）

从 base 模型直接 RL 通常不稳定；先用指令数据 SFT 一个"会说人话"的起点（`Qwen2.5-0.5B` 级别的配置示例，取自 `verl/examples/sft/gsm8k/run_qwen2_5_0_5b_fsdp.sh`）：

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    -m verl.trainer.sft_trainer \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.messages_key=messages \
    data.micro_batch_size_per_gpu=4 \
    optim.lr=1e-4 \
    engine=fsdp \
    model.path="Qwen/Qwen2.5-0.5B-Instruct" \
    model.use_remove_padding=true \
    trainer.default_local_dir=/tmp/sft-ckpt \
    trainer.project_name=gsm8k-sft \
    trainer.total_epochs=1
```

要点：
- **SFT 数据也走同一 parquet 格式**（`messages_key` 指定消息字段）；
- 可开 LoRA 快速试跑（`model.lora_rank` 等），正式训练建议全参；
- 产出 checkpoint 作为后续 RL 的 `actor_rollout_ref.model.path` 与 `actor_rollout_ref.ref.model.path`（参考模型用 SFT 或冻结的 base）。

## 四、阶段 2：RL 训练配置（核心）

### 4.1 统一入口与算法开关

```bash
python3 -m verl.trainer.main_ppo \
    data.train_files="['$HOME/data/gsm8k/train.parquet', '$HOME/data/math/train.parquet']" \
    data.val_files="['$HOME/data/gsm8k/test.parquet', '$HOME/data/aime-2024.parquet']" \
    algorithm.adv_estimator=grpo \   # gae | grpo | rloo | reinforce_plus_plus | remax
    ...
```

算法差异集中在几组 key：

| 算法 | 关键配置 | 参考笔记 |
|---|---|---|
| PPO | `adv_estimator=gae` + `critic.model.path` | [[01-PPO（近端策略优化）]] |
| GRPO | `adv_estimator=grpo` + `use_kl_loss=True` | [[02-GRPO（组相对策略优化）]] |
| GSPO | GRPO 基础上 `policy_loss.loss_mode=gspo` | [[04-GSPO（组序列策略优化）]] |
| DAPO | GRPO 基础上 clip-higher + token-mean + filter_groups + overlong shaping | [[07-DAPO]] |
| RLOO / REINFORCE++ | `adv_estimator=rloo / reinforce_plus_plus` | [[06-RLOO与REINFORCE++]] |

### 4.2 GRPO 最小配置（Qwen3-4B 风格，取自 `grpo_trainer/run_qwen3_4b_fsdp.sh`）

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=3000 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=verl_grpo_example \
    trainer.total_epochs=15 \
    trainer.save_freq=20 \
    trainer.test_freq=5
```

**关键参数语义**：
- `train_batch_size`（每轮 rollout 的 prompt 数）× `rollout.n`（每组采样数）= 本轮总样本数；
- `ppo_mini_batch_size`：一次梯度更新的样本数，`grad_accum = train_batch_size / mini_batch_size`；
- `ppo_max_token_len_per_gpu`：动态 batch（`use_dynamic_bsz`）按**总 token 数**卡每卡显存上限，长序列场景务必设置；
- `kl_loss_type=low_var_kl`：低方差 KL 估计（对应 GRPO 的无偏 KL 项）；
- `data.truncation='error'`：超长 prompt 直接报错（暴露数据问题），而不是悄悄截断。

### 4.3 GSPO：一行切换

在 GRPO 基础上（取自 `gspo_trainer/run_qwen3_8b_fsdp.sh`）：

```bash
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.clip_ratio_low=3e-4 \
    actor_rollout_ref.actor.clip_ratio_high=4e-4 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.use_kl_loss=False
```

注意：**序列级 IS 的 clip 范围比 token 级小 2~3 个数量级**（3e-4 vs 0.2），不要直接沿用 GRPO 的 0.2/0.2。`clip_ratio_c` 是双 clip 系数（负优势侧更紧）。

### 4.4 DAPO：GRPO 的稳定化全套（取自 `dapo_7b_math_fsdp2_4_12.sh`）

```bash
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.filter_groups.enable=True \          # Dynamic Sampling（过滤全对/全错组）
    algorithm.filter_groups.metric=max_reward_minus_min_reward \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \   # Clip-Higher
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \  # Token-Level Loss（总 token 归一化）
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    reward.reward_manager.name=dapo \              # Overlong Reward Shaping
    +reward.reward_kwargs.overlong_buffer_cfg.enable=True \
    +reward.reward_kwargs.overlong_buffer_cfg.len=4096 \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward.reward_kwargs.max_resp_len=8192 \
    data.max_prompt_length=2048 \
    data.max_response_length=8192
```

对应 [[07-DAPO]] 的四大技术：`clip_ratio_low/high`（Clip-Higher）、`filter_groups`（Dynamic Sampling）、`loss_agg_mode=token-mean`（Token-Level Loss）、`reward_kwargs.overlong_buffer_cfg`（Overlong Shaping，`overlong_buffer_len` 即论文的 $L_{\mathrm{cache}}$）。

### 4.5 PPO / VAPO：价值模型路径（取自 `ppo_trainer/run_qwen3_8b_fsdp.sh`）

```bash
    algorithm.adv_estimator=gae \
    algorithm.gamma=1.0 \
    algorithm.lam=1.0 \
    critic.model.path=$CRITIC_MODEL_PATH \
    critic.optim.lr=1e-5 \
    critic.use_dynamic_bsz=True \
    trainer.critic_warmup=0
```

VAPO 的三项价值侧改进（Value Pretraining、Decoupled-GAE、Length-Adaptive GAE）在 verl 中由 `critic_warmup`（预热轮数）、GAE 的 λ 参数及对应实验分支支持，实践时先确认所用版本是否实现了 `lambda_policy != lambda_critic`。

## 五、阶段 3：奖励实现

### 5.1 规则验证器（rule-based verifier）是最常见的起点

```python
# 伪代码：GSM8K 风格（参考 verl/examples/data_preprocess/gsm8k.py + utils/reward_score/）
def compute_reward(completion, ground_truth):
    # 1. 提取模型答案：最后一个 #### 后的数字 / 最后一个 \boxed{...}
    answer = extract_answer(completion)
    # 2. 字符串规整：去逗号、去空格、统一小数格式
    answer = normalize(answer)
    # 3. 等价性判断（数学题建议用符号化比较，如 math_verify 库）
    return 1.0 if answer == normalize(ground_truth) else -1.0
```

- 正确 +1 / 错误 -1（DAPO/GRPO 常用），或 +1/0（注意 0 会让"不回答"与"答错"同分）；
- **答案等价性判断是最大坑**：`2` vs `2.0`、`1/2` vs `0.5`、`\frac{1}{2}` vs `\frac 12`。MATH 类题目建议用符号化等价检查（verl 的 `math_verify` / `math_dapo`）；
- 奖励函数要**独立于模型、可复现**，并在接入 RL 前用历史生成批量验证正确率（防止 verifier bug 悄悄污染整个训练）。

### 5.2 多奖励组合

```python
reward = w1 * correctness + w2 * format + w3 * length_penalty
```

- DAPO 的 Overlong Shaping 就是"正确性 + 长度惩罚"的组合（见 `verl/workers/reward_manager/dapo.py`）；
- 模型奖励（GenRM / LLM-as-judge）需要异步/服务化接入，避免阻塞 rollout（见 [[RL框架设计核心问题与项目对比]] 中 Relax 的 GenRM、verl-omni 的 reward loop）；
- 奖励需要**归一化或组内相对化**：GRPO 的组内归一化天然吸收奖励尺度差异。

## 六、阶段 4：评估与监控

```bash
    trainer.val_before_train=True \   # 训练前先跑一次验证，确认基线
    trainer.test_freq=10 \            # 每 N 轮验证一次
    trainer.log_val_generations=10 \  # 保存验证集生成样例（人工看质量）
    trainer.logger='["console","wandb"]'
```

**必须监控的指标**（详见 [[10-GRPO稳定训练排障清单]]）：

| 指标 | 健康信号 |
|---|---|
| `actor/entropy`（per-token 熵） | 缓慢下降、不崩到 0 |
| `reward/mean`、`pass@k` | 单调上升 |
| `critic/...`（PPO） | 价值损失下降、advantage 分布稳定 |
| `actor/kl` | 在目标范围内（如 0.1 量级） |
| `response length` | 不无限膨胀 |
| `clip ratio` | 高则 off-policy 严重 |
| 吞吐（tokens/s/GPU） | 关注 rollout/train 各自占比 |

## 七、常见坑速查

| 现象 | 检查点 |
|---|---|
| 训练直接 NaN | `grad_clip`、LR 过大、`use_remove_padding` 未开、混入坏数据 |
| 不收敛 | `filter_groups` 没开、组内 reward 无方差、prompt 太难/太易 |
| 回复越来越长 | Overlong Shaping 未开、`loss_agg_mode` 非 token-mean、reward 与长度正相关 |
| OOM | `ppo_max_token_len_per_gpu` 调小、ref/critic 开 `param_offload`、开 `enable_gradient_checkpointing` |
| 权重不同步 | 检查 rollout 与 train 的权重同步配置（colocate 内存拷贝 / 分离集群 NCCL 广播） |

## 八、端到端起步清单

1. 预处理：`examples/data_preprocess/gsm8k.py` → `~/data/gsm8k/{train,test}.parquet`；
2. SFT（可选）：`examples/sft/gsm8k/run_qwen2_5_0_5b_fsdp.sh` 产出起点模型；
3. 奖励冒烟测试：离线对一批生成算 reward，确认正确率与分布合理；
4. 小规模 RL 冒烟：`examples/grpo_trainer/run_qwen3_4b_fsdp.sh`，`total_epochs=2`、小 batch，盯住熵与 KL；
5. 稳定后加 DAPO 四件套（4.4 节）或切 GSPO（4.3 节）；
6. 预算充足再上 VAPO 价值路线（4.5 节）；
7. 每轮评估 + 人工看 `log_val_generations`，不要只看 reward 数字。

## 九、参考脚本路径（本地仓库）

- SFT：`verl/examples/sft/gsm8k/run_qwen2_5_0_5b_fsdp.sh`
- GRPO：`verl/examples/grpo_trainer/run_qwen3_4b_fsdp.sh`、`run_qwen2_5_32b_fsdp.sh`
- GSPO：`verl/examples/gspo_trainer/run_qwen3_8b_fsdp.sh`
- DAPO：`verl/verl/experimental/one_step_off_policy/shell/dapo_7b_math_fsdp2_4_12.sh`
- PPO：`verl/examples/ppo_trainer/run_qwen3_8b_fsdp.sh`
- 数据预处理：`verl/examples/data_preprocess/gsm8k.py`、`math_dataset.py`
- 奖励：`verl/verl/utils/reward_score/math_verify.py`、`math_dapo.py`；`verl/verl/workers/reward_manager/dapo.py`
