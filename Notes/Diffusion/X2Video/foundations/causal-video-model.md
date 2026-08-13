---
type: Note
related_to:
  - "[[X2Video]]"
  - "[[diffusion-forcing]]"
  - "[[self-forcing]]"
  - "[[causal-forcing]]"
  - "[[omniforcing]]"
  - "[[rolling-forcing]]"
status: Active
---

# 因果视频模型：基本流程与蒸馏方法

> [!note] 专题导航
> 本目录入口：[[X2Video]]。如果刚开始接触视频扩散中的两个时间概念，请先阅读 [[video-diffusion-time-axes]]。

> [!info] 概念定位
> 因果视频模型是一种**自回归序列生成**框架，通过因果分解和因果注意力机制，实现可变长度、逐步生成的视频生成。从 Full Sequence Diffusion 模型蒸馏为因果模型，需要解决架构差距和帧级注入性问题。

## 🎯 核心定义

> [!important] 因果视频模型的关键特征
> - **自回归分解**：将联合分布分解为条件分布的乘积
> - **因果注意力**：每个 token 只能关注过去的 token，不能关注未来
> - **可变长度**：可以生成任意长度的序列
> - **逐步生成**：逐帧或逐 chunk 生成，支持实时交互
> - **KV 缓存**：利用键值缓存提高生成效率

## 📐 数学基础：自回归分解

### 联合分布分解

**Full Sequence Diffusion**：

```
p(x₁:N) = 直接建模整个序列的联合分布
```

**因果视频模型**：

```
p(x₁:N) = ∏ᵢ₌₁ᴺ p(xᵢ|x<ᵢ)
         = p(x₁) · p(x₂|x₁) · p(x₃|x₁:₂) · ... · p(xN|x₁:N-1)
```

> [!note] 分解的意义
> - 将复杂的联合分布分解为简单的条件分布
> - 每个条件分布可以独立建模
> - 支持逐步生成，每步只预测下一个 token

### 扩散过程的因果化

**标准扩散（Full Sequence）**：

```python
# 所有帧使用相同噪声级别
t = uniform(0, T)
x_t = add_noise(x_0, t)
noise_pred = model(x_t, t)
```

**因果扩散（Causal Diffusion）**：

```python
# 每帧独立采样噪声级别
for i in range(num_frames):
    t_i = uniform(0, T)  # 每帧独立采样
    x_i_t = add_noise(x_i_0, t_i)
    
# 使用因果掩码
noise_pred = model(x_t, t, causal_mask=True)
```

## 🔄 因果视频模型的基本流程

### 1. 整体架构

> [!abstract] 因果视频模型的组件
> 1. **因果 VAE**：将视频编码为潜在表示，保持时间因果性
> 2. **因果 Transformer**：使用因果注意力机制的 Transformer
> 3. **KV 缓存**：存储已生成帧的键值对，避免重复计算
> 4. **自回归采样器**：逐帧生成视频

### 2. 训练阶段

**Teacher Forcing 训练**：

```python
# 基于真实上下文训练
for i in range(num_frames):
    # 输入：真实的历史帧 x_{<i}
    # 目标：预测当前帧 x_i
    loss = model(x_real_<i, x_i)
```

**Diffusion Forcing 训练**：

```python
# 基于噪声上下文训练
for i in range(num_frames):
    # 每帧独立采样噪声级别
    t_i = sample_noise_level()
    x_i_t = add_noise(x_i, t_i)
    x_<i_t = add_noise(x_<i, t_<i)  # 历史帧也有噪声
    
    # 输入：噪声历史帧和当前噪声帧
    # 目标：预测当前帧的噪声
    loss = model(x_<i_t, x_i_t, t_i)
```

### 3. 推理阶段

**自回归生成**：

```python
# 逐帧生成
generated_frames = []
kv_cache = None

for i in range(num_frames):
    # 输入：已生成的历史帧
    if i == 0:
        context = None
    else:
        context = generated_frames
    
    # 生成当前帧
    x_i = model.generate(context, kv_cache)
    
    # 更新KV缓存
    kv_cache = model.update_kv_cache(kv_cache, x_i)
    
    generated_frames.append(x_i)
```

> [!important] KV 缓存机制
> - **目的**：避免重复计算历史帧的键值
> - **结构**：存储每一层的键 (K) 和值 (V) 张量
> - **更新**：每生成一帧，追加新的 KV 到缓存
> - **淘汰**：当缓存过大时，移除最早的 KV（滚动缓存）

### 4. 注意力机制

**Full Sequence 注意力**：

```
Attention(Q, K, V) = softmax(QKᵀ/√d)V
# 每个token可以关注所有token
```

**因果注意力**：

```
CausalAttention(Q, K, V) = softmax(mask(QKᵀ/√d))V
# mask: 下三角矩阵，防止关注未来token
```

> [!note] 因果掩码示例
>
> ```
> [[1, 0, 0, 0],
>  [1, 1, 0, 0],
>  [1, 1, 1, 0],
>  [1, 1, 1, 1]]
> ```
> - 第 i 行只能关注前 i 个 token
> - 确保生成时不会 " 偷看 " 未来信息

## 🔧 从 Full Sequence Diffusion 蒸馏为因果模型

### 1. 核心挑战：架构差距

> [!warning] 架构差距问题
> - **Full Sequence**：全注意力，可以看到所有帧
> - **因果模型**：因果注意力，只能看到过去帧
> - **差距**：直接转换会导致性能显著下降

**实验证据（来自 Causal Forcing 论文）**：

```
标准DMD（双向学生）: 性能高
Self Forcing（因果学生）: 性能低19.3% Dynamic Degree
```

### 2. 关键原则：帧级注入性

> [!important] 帧级注入性定义
> 对于映射 ϕ_AR: (xᵢᵗ, t) → xᵢ⁰，帧级注入性成立的条件是：
>
> ∀t ∈ (0,1], ∀{xⱼᵗ}ᴺⱼ₌₁, {yⱼᵗ}ᴺⱼ₌₁:
> 如果 xᵢᵗ = yᵢᵗ，则 ϕ_AR(xᵢᵗ, t) = ϕ_AR(yᵢᵗ, t)

**直觉解释**：

- 在因果模型中，每个噪声帧必须映射到唯一的干净帧
- 如果同一个噪声帧对应多个可能的干净帧，模型无法学习正确的映射
- 这会导致模型预测条件期望，生成模糊结果

**违反帧级注入性的后果**：

```python
# 最优解变成条件期望
G*_θ(xᵢᵗ, x<ᵢᵗ, t) = E[xᵢ⁰ | xᵢᵗ, x<ᵢᵗ, t]
# 而不是真实的干净帧
```

> [!tip] 为什么坍缩到条件期望
> 完整推导见 [[causal-forcing]] 的"为什么违反注入性会坍缩到条件期望"：MSE 回归最优解即条件期望；注入性被违反 = 同一学生输入对应多个干净目标，最优解只能是多目标平均 → 高频抵消 → 模糊残影。

### 3. 标准蒸馏流程（两阶段）

> [!abstract] Self Forcing 的蒸馏流程
> **阶段 1：ODE 蒸馏**
> 1. 使用双向教师模型采样 PF-ODE 轨迹
> 2. 训练因果学生模型学习流映射
> 3. 目标：最小化 MSE(G_θ(xᵢᵗ), xᵢ⁰)
>
> **阶段 2：DMD 蒸馏**
> 1. 使用 ODE 蒸馏初始化的学生模型
> 2. 应用分布匹配蒸馏（DMD）
> 3. 进一步提升生成质量

**问题**：

- 阶段 1 中，双向教师的 PF-ODE 在帧级是非注入的
- 同一个噪声帧 xᵢᵗ可能对应多个不同的干净帧 xᵢ⁰
- 这违反了帧级注入性，导致性能下降

### 4. Causal Forcing 的解决方案

> [!tip] Causal Forcing 的三阶段方法
> **阶段 1：教师强制训练 AR 扩散模型**
> ```python
> # 使用Teacher Forcing训练因果扩散模型
> for i in range(num_frames):
>     # 输入：干净的历史帧 x_{<i}
>     # 目标：预测当前帧 x_i
>     ar_teacher = train_with_teacher_forcing(x_real)
> ```
>
> **阶段 2：因果 ODE 蒸馏**
> ```python
> # 使用AR教师进行ODE蒸馏
> for trajectory in ar_teacher.sample_trajectories():
>     # AR教师的PF-ODE满足帧级注入性
>     # 因为它是因果的，每个噪声帧只对应一个干净帧
>     student_loss = MSE(student(xᵢᵗ), xᵢ⁰)
> ```
>
> **阶段 3：非对称 DMD**
> ```python
> # 应用DMD进一步提升质量
> student = apply_dmd(student, bidirectional_teacher)
> ```

> [!success] 为什么 Causal Forcing 有效？
> - **AR 教师**：因果架构，PF-ODE 满足帧级注入性
> - **帧级注入性**：每个噪声帧唯一对应一个干净帧
> - **正确流映射**：学生可以准确学习教师的流映射
> - **性能提升**：超越 Self Forcing 19.3% Dynamic Degree

## 📊 蒸馏流程详细对比

### 方法 1：标准 DMD（双向学生）

```
双向教师 → 标准DMD → 双向学生（少步）
```

- ✅ 满足视频级注入性
- ✅ 性能高
- ❌ 仍然是双向架构，无法实时生成

### 方法 2：Self Forcing（因果学生）

```
双向教师 → ODE蒸馏（违反帧级注入性）→ 因果学生 → DMD → 少步因果学生
```

- ❌ 违反帧级注入性
- ❌ 性能下降 19.3%
- ✅ 因果架构，支持实时生成

### 方法 3：Causal Forcing（因果学生）

```
AR教师（Teacher Forcing训练）→ 因果ODE蒸馏（满足帧级注入性）→ 因果学生 → DMD → 少步因果学生
```

- ✅ 满足帧级注入性
- ✅ 性能超越 Self Forcing
- ✅ 因果架构，支持实时生成

## 🧭 Forcing 家族全景对比

> [!note] 一句话脉络
> **Teacher Forcing → Diffusion Forcing → Self Forcing** 解决的是 " 训练时条件历史用什么 "（暴露偏差问题）；**Rolling Forcing / Causal Forcing / Omni Forcing** 解决的是因果化之后的工程与理论问题（长程漂移、架构差距、跨模态不稳定）。六个方法共享同一条主线：把双向扩散模型变成**少步、因果、可实时流式**的生成器。

### 核心矛盾：暴露偏差

> [!warning] 贯穿始终的问题
> AR 模型训练时条件于 ground-truth 历史，推理时条件于自身不完美的输出 → 条件分布不匹配 → 误差逐帧累积。由于去噪损失需要 " 模型预测 + 对应 ground-truth 条件 " 的配对，暴露偏差难以直接优化消除。

### 1. Teacher Forcing (TF) — 一切的基础

> [!abstract] 一句话定位
> 最经典的 AR 训练范式：训练时永远用**真实干净历史**作条件，学习 p(xᵢ | x₀^{<i})。

- **训练范式**：当前帧 xᵢ 在噪声级别 tⱼ 的条件分布为 p(x_{tⱼ}ᵢ | x₀^{<i})，历史帧全部是数据集的干净帧，逐帧去噪损失
- **解决的问题**：让 AR 扩散模型学会 " 给定干净历史、预测下一帧 " 这一基础条件分布
- **优点**：实现简单、监督信号强、训练稳定收敛快
- **缺点**：**暴露偏差**——训练条件（真实历史）与推理条件（自生成历史）不同分布，长序列误差累积
- **补充发现**：Causal Forcing 证明，对 AR 扩散模型来说 TF 反而比 DF 更适合训练教师（干净前缀才能定义良好的流映射）

### 2. Diffusion Forcing (DF) — 噪声即掩码

> [!abstract] 一句话定位
> 把噪声级别当作 " 部分掩码 "，每个 token 有独立噪声级别，统一 next-token 预测与全序列扩散。

- **训练范式**：历史帧也以**独立噪声级别**加噪，条件分布为 p(x_{tⱼ}ᵢ | x_{t≥0}^{<i})，模型一次性去噪整个序列；TF 是 DF 的特例（历史全干净、当前帧全噪声）
- **解决的问题**：(1) 可变长度生成（1 到数千 token）；(2) 引导采样与规划（MCG 蒙特卡洛引导）；(3) 部分观测/混合噪声序列建模
- **优点**：统一了时间轴掩码（TF）与噪声轴掩码（全序列扩散），灵活性与长程规划能力最强
- **缺点**：训练仍基于 ground-truth 加噪历史，推理基于自生成历史 → **暴露偏差依然存在**（SkyReels-V2 用 DF 做长视频，ΔDrift 达 5.59）；严格逐帧因果在音视频双流上会因 25:3 频率不对称崩溃（Omni Forcing 挑战 1）

### 3. Self Forcing (SF) — 训练=推理

> [!abstract] 一句话定位
> 第一个 " 训练与推理同分布 " 的 AR 视频扩散范式：训练时用**自生成历史**自回归展开，配合视频级 DMD 分布匹配损失。

- **训练范式**：训练中自回归展开（带 KV 缓存），每帧条件于先前**自生成输出**；随机采样去噪步 + 梯度截断（只回传到最终去噪步）；DMD（或 SiD/GAN）匹配加噪后的分布
- **解决的问题**：**暴露偏差**——训练输出与推理输出同分布，模型在训练中学会从自身错误中纠正
- **优点**：消除暴露偏差；单 H100 上 17 FPS、亚秒延迟；质量媲美显著更慢的双向模型；滚动 KV 缓存支持任意长度外推
- **缺点**：
  - **违反帧级注入性**（Causal Forcing）：ODE 蒸馏使用双向教师，其 PF-ODE 在帧级非注入，学生坍缩到条件期望 E[xᵢ⁰ | xᵢᵗ] → 模糊，性能比标准 DMD 双向学生低 19.3% Dynamic Degree
  - **长程误差累积**（Rolling Forcing）：严格逐帧因果让每帧继承并放大前帧误差，超出训练时间窗口后漂移显著（ΔDrift 1.66）
  - 训练开销大（自回归展开 + DMD）

### 4. Rolling Forcing (RF) — 滚动窗口抗漂移

> [!abstract] 一句话定位
> 面向多分钟长视频的流式生成：用**滚动联合去噪窗口**打破严格逐帧因果，抑制误差累积。

- **训练/推理范式**：
  - 窗口长度 = 去噪步数 T=5；窗口内帧噪声级别**渐进升高**、双向注意力互相精修，每前滚一步产出一个干净帧
  - **Attention Sink**：保留首帧 KV 作为全局上下文锚点，动态 RoPE 冻结其相对位置
  - 训练：非重叠窗口（i ≡ j mod T）只对子集窗口算梯度，并与 SF 目标 50/50 混合训练（避免混合噪声级别导致的不自然相机运动）
- **解决的问题**：**长时程误差累积/质量漂移**；长时间全局一致性；实时性（单 GPU 16 FPS、亚秒延迟、多分钟生成）
- **优点**：ΔDrift 仅 0.01（对比 SF 1.66、CausVid 2.18）；帧定稿前可互精修纠正局部错误；整体质量最高
- **缺点**：中间帧离开窗口后无记忆（只保留首帧锚点）；大窗口 + DMD 训练内存高；交互场景延迟增加（未来帧被预生成）；需要混合训练目标

### 5. Causal Forcing (CF) — 理论正确的蒸馏

> [!abstract] 一句话定位
> 从理论上解决 " 双向教师 → 因果学生 " 的**架构差距**：用 AR 教师做 ODE 蒸馏，满足帧级注入性。

- **训练范式**：三阶段 (1) AR 教师 TF 训练（干净视频 + 噪声副本拼接，因果掩码）；(2) 因果 ODE 蒸馏（AR 教师的 PF-ODE 天然满足帧级注入性）；(3) 非对称 DMD（以双向基础模型为教师）
- **解决的问题**：(1) 双向教师 ODE 蒸馏违反帧级注入性 → 学生只能学条件期望 → 模糊；(2) 证明 SF 的 ODE 蒸馏存在分布不匹配 G*(xᵢᵗ) = E[xᵢ⁰ | xᵢᵗ] ≁ p_data；(3) 理论 + 经验证明 TF 优于 DF 训练 AR 教师
- **优点**：理论完备；全面超越 SF（Dynamic Degree +19.3%、VisionReward +8.7%、Instruction Following +16.7%）；首次让因果学生超越双向学生；推理延迟不变
- **缺点**：三阶段流程复杂、训练成本高；仍限于单模态视频

### 6. Omni Forcing (OF) — 跨模态双流

> [!abstract] 一句话定位
> 首个把双向**音视频联合**扩散模型（LTX-2，14B 视频 + 5B 音频）蒸馏为实时流式生成器的框架。

- **训练/推理范式**：
  - 1s 宏块对齐：3 视频 latent + 25 音频 latent；利用 VAE stride（首帧 stride=1）推导 N = 1 + K·f → Global Prefix B₀ 零截断设计
  - 四路非对称因果掩码（V→V / A→A / V→A / A→V，块内双向、块间严格因果）
  - Audio Sink Token S=16 + Identity RoPE：注意力分母从 i 扩到 i+S，恢复熵，抑制 Softmax 坍塌与梯度爆炸
  - 三阶段：双向 DMD → 因果 ODE 回归（速度场回归）→ 联合 Self-Forcing DMD（冻结教师）
  - 模态独立滚动 KV-Cache：层内解耦，每步复杂度 O(L)，支持非对称张量并行
- **解决的问题**：(1) 双流因果化的训练不稳定（25:3 频率不对称 → 音频 token 稀疏 → Softmax 坍塌/NaN）；(2) 跨模态曝光偏差 → 去同步；(3) 联合音视频的 TTFC 延迟
- **优点**：单 GPU ~25 FPS、TTFC ~0.7s、~35× 加速；质量与教师相当（CLIP 0.322 反超教师）；VBench 每帧指标反超教师；首个音视频联合流式
- **缺点**：仅 LTX-2 验证；设计依赖 25:3 VAE stride；1s 宏块固定无消融；三阶段训练重（32 GPU）；与教师的小幅差距来自因果感受野

### 📋 总结对比表

| 方法 | 训练条件来源 | 教师 | 核心解决的问题 | 主要优点 | 主要缺点 | 实时性 |
|------|-------------|------|---------------|---------|---------|--------|
| Teacher Forcing | 真实干净历史 | 无 | 基础条件建模 | 简单、稳定 | 暴露偏差 | 否（多步） |
| Diffusion Forcing | 独立加噪真实历史 | 无 | 可变长度/引导/部分观测 | 统一框架、灵活 | 暴露偏差仍在 | 否（多步） |
| Self Forcing | 自生成历史 | 双向 | 暴露偏差 | 训练=推理、17 FPS | 违反注入性、长程漂移 | 是 |
| Rolling Forcing | 自生成历史 + 滚动窗口 | 双向 | 长程误差累积 | ΔDrift≈0、多分钟 | 中间帧无记忆、训练重 | 是（16 FPS） |
| Causal Forcing | 干净历史（AR 教师） | AR + 双向 | 架构差距/注入性 | 理论完备、性能最高 | 三阶段复杂 | 是 |
| Omni Forcing | 自生成历史（双流） | 双向（LTX-2） | 双流不稳定/去同步 | 音视频流式 25 FPS | 依赖 VAE stride | 是（25 FPS） |

### 🧬 演进脉络

> [!tip] 两条演进轴
> - **训练条件轴（解决暴露偏差）**：TF（真实干净历史）→ DF（加噪真实历史）→ SF（自生成历史）
> - **蒸馏/架构轴（因果化之后）**：CausVid（首个 DMD 因果蒸馏）→ SF（暴露偏差）→ CF（帧级注入性/架构差距）→ RF（长程漂移）→ OF（跨模态双流）
> - **共同技术栈**：少步化（DMD/ODE 蒸馏）+ 因果化（注意力掩码）+ 分布对齐（视频级 DMD 损失）+ 长程稳定（KV 缓存 / attention sink 锚点 / 滚动窗口）

## 🔬 技术细节：ODE 蒸馏

### 标准 ODE 蒸馏

```python
# 双向教师 → 双向学生
def standard_ode_distillation(teacher, student):
    for x0 in dataset:
        # 采样PF-ODE轨迹
        trajectory = teacher.sample_trajectory(x0)
        
        for (xt, t, x0) in trajectory:
            # 训练学生学习流映射
            student_loss = MSE(student(xt, t), x0)
```

**满足视频级注入性**：

- 对于任意噪声视频 xₜ，存在唯一的干净视频 x₀
- 学生可以准确学习流映射

### 因果 ODE 蒸馏

```python
# AR教师 → 因果学生
def causal_ode_distillation(ar_teacher, student):
    for x0 in dataset:
        # AR教师采样轨迹（因果）
        trajectory = ar_teacher.sample_trajectory(x0)
        
        for (xᵢᵗ, t, xᵢ⁰) in trajectory:
            # 满足帧级注入性
            # 每个噪声帧xᵢᵗ唯一对应干净帧xᵢ⁰
            student_loss = MSE(student(xᵢᵗ, x<ᵢᵗ, t), xᵢ⁰)
```

**满足帧级注入性**：

- AR 教师是因果的，每帧独立去噪
- 对于任意噪声帧 xᵢᵗ，存在唯一的干净帧 xᵢ⁰
- 学生可以准确学习帧级流映射

## 🎯 关键洞察

> [!important] 蒸馏成功的关键
> 1. **架构匹配**：教师和学生应具有相似的架构（都是因果或都是双向）
> 2. **注入性保证**：配对数据必须满足注入性条件
> 3. **流映射学习**：学生必须能准确学习教师的流映射
> 4. **分布匹配**：通过 DMD 进一步对齐生成分布

> [!warning] 常见陷阱
> 1. **直接蒸馏**：从双向教师直接蒸馏因果学生，违反帧级注入性
> 2. **忽略架构差距**：认为 DMD 阶段可以弥补架构差距
> 3. **噪声级别不当**：使用统一噪声级别而非独立噪声级别
> 4. **缓存管理不当**：KV 缓存更新策略不合适

## 📈 性能对比

**基于 Causal Forcing 论文的实验结果**：

| 方法 | Dynamic Degree | VisionReward | Instruction Following |
|------|---------------|--------------|----------------------|
| 标准 DMD（双向学生） | 基准 | 基准 | 基准 |
| Self Forcing（因果学生） | -19.3% | -8.7% | -16.7% |
| Causal Forcing（因果学生） | 超越基准 | 超越基准 | 超越基准 |

> [!success] Causal Forcing 的突破
> - 首次实现因果学生超越双向学生
> - 证明了帧级注入性的重要性
> - 为实时交互式视频生成开辟新道路

## 🚀 实际应用

### 实时交互式视频生成

```python
# 使用因果模型实现实时生成
def realtime_video_generation(model, user_input):
    video_stream = []
    kv_cache = None
    
    while True:
        # 获取用户输入（如相机控制）
        control_signal = get_user_input()
        
        # 生成下一帧
        next_frame = model.generate(
            context=video_stream,
            control=control_signal,
            kv_cache=kv_cache
        )
        
        # 更新缓存和视频流
        kv_cache = model.update_cache(kv_cache, next_frame)
        video_stream.append(next_frame)
        
        # 实时显示
        display_frame(next_frame)
```

### 应用场景

> [!tip] 因果视频模型的应用
> 1. **游戏模拟**：实时生成游戏画面
> 2. **世界建模**：交互式世界模拟
> 3. **机器人学习**：实时生成机器人视角
> 4. **直播内容**：实时视频内容生成
> 5. **虚拟现实**：沉浸式 VR 体验

## 📚 总结

> [!quote] 核心要点
> 1. **因果视频模型**通过自回归分解和因果注意力实现逐步生成
> 2. **帧级注入性**是从 Full Sequence 蒸馏到因果模型的关键原则
> 3. **Causal Forcing**通过使用 AR 教师解决帧级注入性问题
> 4. **KV 缓存**是因果模型高效推理的关键技术
> 5. **实时交互**是因果模型相比 Full Sequence 模型的核心优势
> 6. **Forcing 家族**沿 " 训练条件（暴露偏差）" 与 " 因果化蒸馏（架构差距/漂移）" 两条轴演进，详见 "🧭 Forcing 家族全景对比 "

> [!quote] 引用
> - **Teacher Forcing**: 经典序列模型训练范式
> - **Diffusion Forcing**: Next-token Prediction Meets Full-Sequence Diffusion（arXiv 2407.01392）
> - **Self Forcing**: Bridging the Train-Test Gap in Autoregressive Video Diffusion（arXiv 2506.08009）
> - **Rolling Forcing**: Autoregressive Long Video Diffusion in Real Time（arXiv 2509.25161）
> - **Causal Forcing**: Autoregressive Diffusion Distillation Done Right（arXiv 2602.02214）
> - **Omni Forcing**: Unleashing Real-time Joint Audio-Visual Generation（arXiv 2603.11647）
