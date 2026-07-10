---
type: Note
related_to: "[[X2Video]]"
status: Active
url: http://arxiv.org/abs/2407.01392
---

# Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion

> [!info] 论文信息
> - **作者**: Boyuan Chen, Diego Marti Monso, Yilun Du, Max Simchowitz, Russ Tedrake, Vincent Sitzmann
> - **日期**: 2024-12-10
> - **arXiv**: [2407.01392](http://arxiv.org/abs/2407.01392)
> - **项目主页**: https://boyuan.space/diffusion-forcing

## 摘要

Diffusion Forcing 是一种新的训练范式，其中扩散模型被训练来对一组具有独立每 token 噪声级别的 token 进行去噪。该方法将 Diffusion Forcing 应用于序列生成建模，通过训练一个因果 next-token 预测模型来生成一个或多个未来 token，而无需完全扩散过去的 token。该方法结合了 next-token 预测模型（如可变长度生成）和全序列扩散模型（如引导采样到理想轨迹的能力）的优势。

> [!tip] 核心思想
> 将扩散过程中的噪声级别视为一种"部分掩码"，每个 token 可以有不同的噪声级别，从而统一了 next-token 预测和全序列扩散的优势。

## 主要贡献

1. **提出 Diffusion Forcing**：一种新的概率序列模型，具有 next-token 预测模型的灵活性，同时能够像全序列扩散模型一样进行长程引导。
2. **决策制定框架**：引入一种新的决策制定框架，允许将 Diffusion Forcing 同时用作策略（policy）和规划器（planner）。
3. **理论证明**：证明在适当条件下，优化所提出的训练目标可以最大化所有子序列的似然下界。
4. **广泛实验评估**：在视频生成、基于模型的规划、视觉模仿学习和时间序列预测等多个领域评估 Causal Diffusion Forcing（CDF），展示了其独特能力。

> [!important] 关键创新
> - **可变长度生成**：可以生成从下一个 token 到数千个 token 的序列，即使对于连续 token 也能保持稳定。
> - **引导采样**：像全序列扩散一样，接受引导以获得高回报的生成。
> - **蒙特卡洛引导（MCG）**：利用因果性、灵活范围和可变噪声调度，显著提高高回报生成的采样效果。

## 方法概述

### 统一视角：噪声作为部分掩码

> [!important] 核心洞察
> 噪声级别可以被视为一种"部分掩码"，统一了时间轴和噪声轴两个维度。

- **时间轴掩码**：Teacher forcing 将每个 token xt 在时间 t 掩码，并从过去 x1:t−1 进行预测。
- **噪声轴掩码**：全序列前向扩散逐渐向数据添加噪声，可以视为沿噪声轴的部分掩码。
- **Diffusion Forcing**：结合两个轴，每个 token 可以有不同的噪声级别 kt，形成 (x_kt_t)1≤t≤T 的序列。

**数学表示**：
```python
# 噪声级别作为掩码程度
x_kt_t = √(ᾱ_kt) * x0_t + √(1-ᾱ_kt) * ϵ

# 当 kt = 0: x_0_t = x0_t (完全可见，无掩码)
# 当 kt = K: x_K_t ≈ N(0, I) (完全掩码，纯噪声)
# 中间值: 部分掩码，保留部分信息
```

### 因果扩散强制（Causal Diffusion Forcing, CDF）

> [!note] CDF的核心特点
> - 使用因果架构（如 RNN 或掩码 Transformer）实现
> - 训练模型一次性去噪整个序列，每个 token 有独立的噪声级别
> - 采样时，CDF 逐渐将高斯噪声帧序列去噪为干净样本
> - 不同帧在每个去噪步骤可能有不同的噪声级别

## 🔬 Diffusion Forcing Training 详解

### Algorithm 1: Diffusion Forcing Training

> [!abstract] 算法伪代码
> ```
> 1: loop
> 2:     Sample trajectory of observations (x1, ..., xT)
> 3:     for t = 1, ..., T do
> 4:         Sample independent noise level kt ∈ {0, 1, ..., K}
> 5:         x_kt_t = ForwardDiffuse(xt, kt)
> 6:         Define ϵt = (x_kt_t - √(ᾱ_kt) * xt) / √(1-ᾱ_kt)
> 7:         Update zt ∼ pθ(zt|zt−1, x_kt_t, kt)
> 8:         Set ˆϵt = ϵθ(zt−1, x_kt_t, kt)
> 9:     end for
> 10:    L = MSELoss([ˆϵ1, ..., ˆϵn], [ϵ1, ..., ϵn])
> 11:    Backprop with L and update θ
> 12: end loop
> ```

### Token级去噪机制详解

> [!important] 核心问题
> Diffusion Forcing如何对每个token进行去噪？
> 每个token有不同的噪声级别，模型如何处理这种异构输入？

#### 1. 噪声级别的含义

```python
# 噪声级别kt表示token被"掩码"的程度
# kt = 0: 完全可见，无噪声（干净token）
# kt = K: 完全掩码，纯噪声
# 0 < kt < K: 部分掩码，保留部分信息

# 数学表示
x_kt_t = √(ᾱ_kt) * x0_t + √(1-ᾱ_kt) * ϵ
# 其中 ᾱ_kt = ∏_{s=1}^{kt} (1-β_s) 是累积噪声调度
```

> [!note] 直觉理解
> - **kt = 0**: token完全清晰，模型可以看到完整信息
> - **kt = K**: token完全模糊，模型只能看到噪声
> - **中间值**: token部分模糊，模型可以看到部分信息
> - **关键**: 每个token的kt可以不同，实现"部分掩码"

#### 2. 因果架构如何处理不同噪声级别

```python
# 因果架构的核心：隐藏状态zt
# zt = fθ(zt-1, x_kt_t, kt)

# 对于每个token t:
# 1. 输入: 噪声token x_kt_t 和 噪声级别 kt
# 2. 状态: 前一隐藏状态 zt-1（编码历史信息）
# 3. 输出: 更新后的隐藏状态 zt
```

> [!important] 关键机制
> 因果架构通过隐藏状态zt实现以下功能：
> 
> 1. **历史信息编码**: zt-1包含所有过去token的信息
> 2. **噪声感知**: 输入kt让模型知道当前token的噪声程度
> 3. **部分恢复**: 模型学习从任意噪声级别恢复干净token
> 4. **因果约束**: 只能访问过去信息，不能访问未来

#### 3. 去噪预测的实现

```python
# 模型预测噪声的函数
def predict_noise(zt-1, x_kt_t, kt):
    """
    预测当前token的噪声
    
    Args:
        zt-1: 前一隐藏状态 [batch, hidden_dim]
        x_kt_t: 当前噪声token [batch, channels, ...]
        kt: 噪声级别 (标量)
    
    Returns:
        ˆϵt: 预测的噪声 [batch, channels, ...]
    """
    # 1. 编码噪声级别
    kt_embedding = noise_level_embedding(kt)  # [batch, embed_dim]
    
    # 2. 编码噪声token
    x_embedding = token_encoder(x_kt_t)  # [batch, embed_dim]
    
    # 3. 融合信息
    # 融合历史状态、当前token和噪声级别
    combined = torch.cat([zt-1, x_embedding, kt_embedding], dim=-1)
    
    # 4. 预测噪声
    ˆϵt = noise_predictor(combined)
    
    return ˆϵt
```

> [!note] 噪声预测的关键
> - **输入**: 历史状态zt-1、当前噪声token x_kt_t、噪声级别kt
> - **输出**: 预测的噪声ˆϵt
> - **目标**: 学习从任意噪声级别恢复干净token

#### 4. 训练目标的数学形式

```python
# 训练目标：最小化预测噪声与真实噪声的差异
L(θ) = E_{kt,xt,ϵt}[∑_{t=1}^T ||ϵt - ϵθ(zt-1, x_kt_t, kt)||²]

# 其中:
# - kt ∼ Uniform([K]^T): 每帧独立均匀采样噪声级别
# - xt ∼ pdata: 从数据分布采样
# - ϵt ∼ N(0, I): 标准高斯噪声
# - zt-1: 由模型递归计算的隐藏状态
```

> [!abstract] 损失函数的直觉
> 1. **采样**: 随机选择噪声级别kt，对每个token添加不同程度的噪声
> 2. **预测**: 模型预测每个token的噪声
> 3. **比较**: 计算预测噪声与真实噪声的MSE
> 4. **优化**: 最小化损失，提高去噪能力

#### 5. 部分去噪的实现

```python
# 关键概念：模型学习"部分去噪"
# 对于噪声级别kt，模型预测的是从kt到0的噪声

# 例如：
# - kt = 100: 模型预测从100到0的噪声（完全去噪）
# - kt = 50: 模型预测从50到0的噪声（部分去噪）
# - kt = 0: 无需预测（已经是干净token）

# 数学表示
# 模型学习: ϵθ(zt-1, x_kt_t, kt) ≈ ϵt
# 其中 ϵt 是从x0_t到x_kt_t的噪声
```

> [!important] 部分去噪的意义
> - **灵活性**: 模型可以从任意噪声级别开始去噪
> - **效率**: 可以跳过不必要的去噪步骤
> - **质量**: 可以针对不同噪声级别优化去噪策略

#### 6. 因果约束的实现

```python
# 因果约束：每个token只能看到过去信息
# 实现方式：因果注意力掩码

# 对于token t:
# - 可以看到: token 1, 2, ..., t-1（通过隐藏状态zt-1）
# - 不能看到: token t+1, t+2, ..., T（未来信息）

# 因果掩码示例（Transformer实现）
causal_mask = torch.tril(torch.ones(T, T))  # 下三角矩阵
# 1 0 0 0
# 1 1 0 0
# 1 1 1 0
# 1 1 1 1
```

> [!note] 因果约束的重要性
> - **自回归生成**: 支持逐步生成，每步只预测下一个token
> - **实时交互**: 可以在生成过程中接受用户输入
> - **一致性**: 确保生成时不会"偷看"未来信息
> - **稳定性**: 避免误差累积导致的崩溃

### 详细步骤解析

**Step 1: 采样训练轨迹**
```python
# 从数据集中采样一个视频序列
(x1, x2, ..., xT) = sample_from_dataset()  # 例如：81帧视频
```

**Step 2-3: 为每帧独立采样噪声级别**
```python
# 关键创新：每帧独立采样噪声级别
for t in range(1, T+1):
    kt = random.randint(0, K)  # 例如：K=1000
    # kt 独立于其他帧的噪声级别
```

> [!important] 独立噪声级别的意义
> - **Teacher Forcing**：kt = 0 对所有t（干净上下文）
> - **Full Sequence Diffusion**：kt = k 对所有t（统一噪声）
> - **Diffusion Forcing**：kt 独立采样（每帧不同噪声）

**Step 4-5: 前向扩散过程**
```python
# 对每帧应用前向扩散
x_kt_t = √(ᾱ_kt) * xt + √(1-ᾱ_kt) * ϵt
# 其中 ᾱ_kt = ∏_{s=1}^{kt} (1-β_s)
# ϵt ∼ N(0, I) 是标准高斯噪声
```

**Step 6: 定义目标噪声**
```python
# 从噪声帧反推目标噪声
ϵt = (x_kt_t - √(ᾱ_kt) * xt) / √(1-ᾱ_kt)
# 这是前向扩散过程中添加的噪声
```

**Step 7: 更新隐藏状态**
```python
# 使用RNN或因果Transformer更新隐藏状态
zt = fθ(zt-1, x_kt_t, kt)
# zt 捕获了历史信息和当前噪声帧的信息
```

> [!note] 隐藏状态的作用
> - **zt**：编码了到时间t为止的所有信息
> - **输入**：前一隐藏状态 zt-1、当前噪声帧 x_kt_t、噪声级别 kt
> - **输出**：更新后的隐藏状态 zt
> - **因果性**：只能访问过去信息，不能访问未来

**Step 8: 预测噪声**
```python
# 使用模型预测噪声
ˆϵt = ϵθ(zt-1, x_kt_t, kt)
# 输入：前一隐藏状态、当前噪声帧、噪声级别
# 输出：预测的噪声
```

**Step 9-10: 计算损失**
```python
# 对整个序列计算MSE损失
L = MSE(ˆϵt, ϵt) for t in 1 to T
# 损失衡量预测噪声与真实噪声的差异
```

> [!abstract] 损失函数的数学形式
> L(θ) = E_{kt,xt,ϵt}[∑_{t=1}^T ||ϵt - ϵθ(zt-1, x_kt_t, kt)||²]
> 
> 其中：
> - kt ∼ Uniform([K]^T)：独立均匀采样噪声级别
> - xt ∼ pdata：从数据分布采样
> - ϵt ∼ N(0, I)：标准高斯噪声

### 训练过程的直觉理解

> [!tip] 训练的核心思想
> 1. **多样性**：每帧独立采样噪声级别，覆盖所有可能的噪声配置
> 2. **因果性**：模型只能看到过去帧，模拟推理时的自回归场景
> 3. **去噪能力**：学习从任意噪声级别恢复干净帧
> 4. **泛化性**：训练覆盖所有噪声配置，推理时可以灵活使用

## 🎯 DF Sampling with Guidance 详解

### Algorithm 2: DF Sampling with Guidance

> [!abstract] 算法伪代码
> ```
> 1:  Input: Model θ, scheduling matrix K, initial latent z0, guidance cost c(·)
> 2:  Initialize x1, . . . , xT ∼ N(0, σ²_K I)
> 3:  for row m = M −1, ..., 0 do
> 4:      for t = 1, . . . , T do
> 5:          z_new_t ∼ pθ(zt | zt−1, xt, K_{m+1,t})
> 6:          k ← K_{m,t}, w ∼ N(0, I)
> 7:          x_new_t ← (1/√α_k) * (xt - (1-α_k)/√(1-ᾱ_k) * ϵθ(z_new_t, xt, k)) + σ_k * w
> 8:          Update zt ← z_new_t
> 9:      end for
> 10:     x1:H ← AddGuidance(x_new_1:H, ∇x log c(x_new_1:H))
> 11: end for
> 12: Return x1:T
> ```

### Token级采样去噪详解

> [!important] 核心问题
> 在采样阶段，如何对每个token进行去噪？
> 调度矩阵K如何控制每个token的去噪过程？

#### 1. 调度矩阵K的设计

```python
# 调度矩阵K: [M, T]
# M: 去噪步骤数
# T: 序列长度（token数量）

# K[m, t]: 第m步第t个token的噪声级别

# 设计策略示例：
def create_scheduling_matrix(M, T, K_max=1000):
    """
    创建调度矩阵
    
    Args:
        M: 去噪步骤数
        T: 序列长度
        K_max: 最大噪声级别
    
    Returns:
        K_matrix: [M, T] 调度矩阵
    """
    K_matrix = torch.zeros(M, T)
    
    for m in range(M):
        for t in range(T):
            # 策略1: 统一调度（所有token相同）
            K_matrix[m, t] = int(K_max * (1 - m / M))
            
            # 策略2: 渐进调度（早期token先去噪）
            # K_matrix[m, t] = max(0, int(K_max * (1 - m/M)) - t * 10)
            
            # 策略3: 自定义调度
            # K_matrix[m, t] = custom_schedule(m, t)
    
    return K_matrix.long()
```

> [!note] 调度矩阵的意义
> - **统一调度**: 所有token在同一步骤有相同噪声级别
> - **渐进调度**: 早期token先去噪，后期token后去噪
> - **灵活调度**: 可以自定义每个token的去噪进度
> - **关键**: K[m,t]决定了第m步第t个token的噪声级别

#### 2. 采样去噪的详细步骤

```python
def sample_with_denoising(model, T, M, K_max=1000):
    """
    采样并去噪整个序列
    
    Args:
        model: 训练好的Diffusion Forcing模型
        T: 序列长度
        M: 去噪步骤数
        K_max: 最大噪声级别
    
    Returns:
        x: 生成的序列 [batch, channels, T, ...]
    """
    # Step 1: 初始化为纯噪声
    x = torch.randn(batch_size, channels, T, H, W)  # 所有token都是纯噪声
    
    # Step 2: 创建调度矩阵
    K_matrix = create_scheduling_matrix(M, T, K_max)
    
    # Step 3: 初始化隐藏状态
    z = torch.zeros(batch_size, hidden_dim)
    
    # Step 4: 去噪循环
    for m in range(M - 1, -1, -1):  # 从高噪声到低噪声
        x_new = torch.zeros_like(x)
        
        for t in range(T):  # 遍历每个token
            # 4.1: 获取当前噪声级别
            k = K_matrix[m, t]
            
            # 4.2: 更新隐藏状态（因果约束）
            z = model.update_state(z, x[:, :, t], k)
            
            # 4.3: 预测噪声
            noise_pred = model.predict_noise(z, x[:, :, t], k)
            
            # 4.4: 去噪步骤
            x_new[:, :, t] = denoise_step(x[:, :, t], noise_pred, k)
        
        # Step 5: 更新序列
        x = x_new
    
    return x
```

#### 3. 单token去噪步骤详解

```python
def denoise_step(x_k, noise_pred, k, K=1000):
    """
    单个token的去噪步骤
    
    Args:
        x_k: 噪声token [batch, channels, ...]
        noise_pred: 预测的噪声 [batch, channels, ...]
        k: 当前噪声级别
        K: 最大噪声级别
    
    Returns:
        x_prev: 去噪后的token（噪声级别k-1）
    """
    # 计算噪声调度参数
    beta_start, beta_end = 0.0001, 0.02
    betas = torch.linspace(beta_start, beta_end, K)
    alphas = 1 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    
    # 当前噪声级别的参数
    alpha_k = alphas[k]
    alpha_bar_k = alpha_bar[k]
    
    # 计算去噪均值
    # 公式: μ_θ(x_k, k) = (1/√α_k) * (x_k - (1-α_k)/√(1-ᾱ_k) * ϵ_θ(x_k, k))
    mean = (1 / torch.sqrt(alpha_k)) * (
        x_k - (1 - alpha_k) / torch.sqrt(1 - alpha_bar_k) * noise_pred
    )
    
    # 添加随机噪声（除非是最后一步）
    if k > 0:
        # 计算方差
        # 公式: σ_k² = (1-ᾱ_{k-1})/(1-ᾱ_k) * β_k
        alpha_bar_prev = alpha_bar[k-1]
        variance = (1 - alpha_bar_prev) / (1 - alpha_bar_k) * betas[k]
        sigma = torch.sqrt(variance)
        
        # 采样随机噪声
        noise = torch.randn_like(x_k)
        
        # 去噪后的token
        x_prev = mean + sigma * noise
    else:
        # 最后一步，不添加噪声
        x_prev = mean
    
    return x_prev
```

> [!important] 去噪公式的数学推导
> 基于DDPM的反向过程：
> 
> **前向过程**:
> ```
> q(x_k | x_0) = N(x_k; √(ᾱ_k) * x_0, (1-ᾱ_k)I)
> ```
> 
> **反向过程**:
> ```
> p_θ(x_{k-1} | x_k) = N(x_{k-1}; μ_θ(x_k, k), σ_k²I)
> ```
> 
> **均值计算**:
> ```
> μ_θ(x_k, k) = (1/√α_k) * (x_k - (1-α_k)/√(1-ᾱ_k) * ϵ_θ(x_k, k))
> ```
> 
> **方差计算**:
> ```
> σ_k² = (1-ᾱ_{k-1})/(1-ᾱ_k) * β_k
> ```
> 
> **采样**:
> ```
> x_{k-1} = μ_θ(x_k, k) + σ_k * ϵ, ϵ ∼ N(0, I)
> ```

#### 4. 因果约束在采样中的体现

```python
# 因果约束：每个token只能看到过去信息
# 在采样时，这体现为：

for t in range(T):
    # 更新隐藏状态时，只能使用过去信息
    z = model.update_state(z, x[:, :, t], k)
    
    # 预测噪声时，只能使用过去信息
    noise_pred = model.predict_noise(z, x[:, :, t], k)
    
    # 关键: z包含了所有过去token的信息，但不包含未来信息

# 对比Full Sequence Diffusion:
# 在Full Sequence中，每个token可以关注所有其他token
# 在Diffusion Forcing中，每个token只能关注过去的token
```

> [!note] 因果约束的优势
> 1. **自回归生成**: 支持逐步生成，每步只预测下一个token
> 2. **实时交互**: 可以在生成过程中接受用户输入
> 3. **长序列生成**: 可以生成比训练序列更长的视频
> 4. **稳定性**: 避免误差累积导致的崩溃

#### 5. 部分去噪的实现

```python
# 关键概念: 每个token可以处于不同的去噪阶段
# 例如: token 1可能已经去噪到k=10，而token 5还在k=100

# 调度矩阵K的设计决定了每个token的去噪进度
K_matrix = [
    [100, 100, 100, 100, 100],  # 步骤1: 所有token都是高噪声
    [80, 80, 80, 80, 80],       # 步骤2: 所有token噪声降低
    [60, 60, 60, 60, 60],       # 步骤3: 继续降低
    [40, 40, 40, 40, 40],       # 步骤4: 继续降低
    [20, 20, 20, 20, 20],       # 步骤5: 继续降低
    [0, 0, 0, 0, 0],            # 步骤6: 所有token干净
]

# 或者渐进调度:
K_matrix = [
    [100, 90, 80, 70, 60],      # 步骤1: 早期token噪声更低
    [80, 70, 60, 50, 40],       # 步骤2: 继续渐进
    [60, 50, 40, 30, 20],       # 步骤3: 继续渐进
    [40, 30, 20, 10, 0],        # 步骤4: 早期token已干净
    [20, 10, 0, 0, 0],          # 步骤5: 更多token干净
    [0, 0, 0, 0, 0],            # 步骤6: 所有token干净
]
```

> [!important] 部分去噪的意义
> - **灵活性**: 可以控制每个token的去噪进度
> - **效率**: 可以跳过不必要的去噪步骤
> - **质量**: 可以针对不同token优化去噪策略
> - **组合性**: 可以组合不同去噪阶段的token

### 详细步骤解析

**Step 1-2: 初始化**
```python
# 输入
model = trained_diffusion_forcing_model
K = scheduling_matrix  # 调度矩阵，定义每帧每步的噪声级别
z0 = initial_latent  # 初始潜变量
c = guidance_cost_function  # 引导成本函数

# 初始化所有帧为纯噪声
x1, x2, ..., xT = N(0, σ²_K * I)  # 例如：σ_K = 1
```

> [!note] 调度矩阵K
> - **形状**：M × T（M个去噪步骤，T帧）
> - **K[m,t]**：第m步第t帧的噪声级别
> - **设计**：可以灵活设计，例如让某些帧先去噪

**Step 3: 外层循环（去噪步骤）**
```python
# 从高噪声到低噪声
for m in range(M-1, -1, -1):
    # 每一步降低所有帧的噪声级别
```

**Step 4-8: 内层循环（帧处理）**
```python
for t in range(1, T+1):
    # Step 5: 更新隐藏状态
    z_new_t = model.update_state(zt-1, xt, K[m+1, t])
    
    # Step 6: 获取当前噪声级别和随机噪声
    k = K[m, t]  # 当前噪声级别
    w = N(0, I)  # 随机噪声
    
    # Step 7: 去噪步骤（核心）
    noise_pred = model.predict_noise(z_new_t, xt, k)
    x_new_t = (1/√α_k) * (xt - (1-α_k)/√(1-ᾱ_k) * noise_pred) + σ_k * w
    
    # Step 8: 更新状态
    zt = z_new_t
```

> [!important] 去噪公式的数学推导
> 基于DDPM的反向过程：
> ```
> pθ(x_{k-1}|x_k) = N(x_{k-1}; μ_θ(x_k, k), σ_k²I)
> ```
> 
> 其中均值为：
> ```
> μ_θ(x_k, k) = (1/√α_k) * (x_k - (1-α_k)/√(1-ᾱ_k) * ϵ_θ(x_k, k))
> ```
> 
> 采样过程：
> ```
> x_{k-1} = μ_θ(x_k, k) + σ_k * ϵ, ϵ ∼ N(0, I)
> ```

**Step 9-10: 应用引导**
```python
# 计算引导梯度
gradient = ∇x log c(x_new_1:H)

# 应用引导到序列
x1:H = x_new_1:H + guidance_scale * gradient
```

> [!tip] 引导机制详解
> - **分类器引导**：使用预训练分类器的梯度
> - **成本函数引导**：直接优化成本函数 c(x)
> - **蒙特卡洛引导（MCG）**：DF特有的引导方法
> - **灵活范围**：可以只引导前H帧，其余帧自由生成

### 引导的关键特性

> [!important] DF引导的独特优势
> 1. **灵活范围**：可以引导任意长度的子序列
> 2. **变量长度**：生成长度可以超过训练长度
> 3. **组合性**：可以组合不同子序列的引导
> 4. **因果性**：引导时保持因果约束

### 调度矩阵设计

> [!note] 调度矩阵K的设计策略
> **策略1：统一调度**
> ```
> K[m,t] = k_m  # 所有帧相同
> ```
> 
> **策略2：渐进调度**
> ```
> K[m,t] = max(0, k_m - t*δ)  # 早期帧先去噪
> ```
> 
> **策略3：灵活调度**
> ```
> K[m,t] = custom_schedule(m, t)  # 自定义调度
> ```

### "Zig-zag"采样策略

> [!important] 核心思想
> **噪声级别 = 不确定性程度**
> - 高噪声（k接近K）：高不确定性
> - 低噪声（k接近0）：低不确定性
> - 无噪声（k=0）：完全确定

**"Zig-zag"采样过程**：
```
初始: [x1K, x2K, x3K] = [噪声, 噪声, 噪声]      # 所有未来都不确定
步骤1: [x01, xK/2 2, x3K] = [清晰, 部分, 噪声]    # 近未来更确定
步骤2: [x01, x02, xK/2 3] = [清晰, 清晰, 部分]    # 中等未来更确定
步骤3: [x01, x02, x03] = [清晰, 清晰, 清晰]        # 所有未来都确定
```

> [!tip] "Zig-zag"采样的优势
> 1. **时间结构编码**：近未来比远未来更确定，符合现实直觉
> 2. **灵活引导**：近未来可应用强引导，远未来保持灵活性
> 3. **有效采样**：逐步增加确定性，避免一次性去噪
> 4. **序列引导**：使得引导更有效，近未来引导效果更明显

**与Full Sequence Diffusion的对比**：
- **Full Sequence**：所有token同时去噪，没有时间结构
- **Diffusion Forcing**："Zig-zag"去噪，编码"近未来更确定"的直觉

## 🔄 训练与推理的对比

| 方面 | 训练阶段 | 推理阶段 |
|------|----------|----------|
| **输入** | 真实数据序列 | 随机噪声序列 |
| **噪声级别** | 独立采样 | 调度矩阵定义 |
| **目标** | 学习去噪 | 生成新序列 |
| **引导** | 无 | 可选 |
| **长度** | 固定（训练长度） | 灵活（可超过训练长度） |

## 💡 核心创新点

> [!success] Diffusion Forcing的突破
> 1. **独立噪声级别**：每帧可以有不同的噪声级别
> 2. **因果架构**：保持自回归生成能力
> 3. **灵活引导**：支持任意范围的引导
> 4. **变量长度**：生成长度可超过训练长度
> 5. **组合性**：可以组合不同子序列

## 理论保证

### 定理 3.1：ELBO优化

> [!abstract] 定理 3.1（非正式）
> Diffusion Forcing 训练过程（算法 1）优化了对所有子序列 token 的期望对数似然 ln pθ((x_kt_t)1≤t≤T) 的重新加权证据下界（ELBO），其中期望是对噪声级别 k1:T ∼ [K]^T 和根据前向过程添加噪声的 x_kt_t 取平均。此外，在适当条件下，优化 (3.1) 同时最大化了所有噪声级别序列的似然下界。

### 理论意义

> [!important] 理论贡献
> 1. **统一框架**：证明DF训练目标统一了Teacher Forcing和Full Sequence Diffusion
> 2. **ELBO优化**：DF训练最大化了数据似然的下界
> 3. **普遍性**：对所有噪声级别序列同时优化
> 4. **有效性**：为DF的成功提供理论基础

### 数学证明概要

**目标函数**：
```
L(θ) = E_{k1:T, x1:T, ϵ1:T}[∑_{t=1}^T ||ϵt - ϵθ(zt-1, x_kt_t, kt)||²]
```

**ELBO分解**：
```
log pθ(x1:T) ≥ E_q[log pθ(x1:T|z1:T)] - KL(q(z1:T|x1:T) || pθ(z1:T))
```

**关键洞察**：
- DF训练目标等价于重新加权的ELBO
- 对所有噪声级别序列同时优化
- 理论上保证了模型的有效性

## 实验结果

### 视频生成

> [!success] 视频生成结果
> - **长序列稳定性**：CDF能够生成比训练序列更长的视频，而基线方法会发散
> - **时间一致性**：生成的视频保持时间连贯性
> - **质量**：生成质量与全序列扩散模型相当

### 基于模型的规划

> [!success] 规划任务结果
> - **蒙特卡洛引导（MCG）**：比标准引导方法获得更高的回报
> - **树搜索**：支持高效的树搜索，用于决策制定
> - **长程规划**：能够规划长程轨迹

### 视觉模仿学习

> [!success] 模仿学习结果
> - **双重角色**：同时作为策略（policy）和规划器（planner）
> - **泛化性**：能够泛化到未见过的状态
> - **稳定性**：生成稳定的动作序列

### 时间序列预测

> [!success] 时间序列结果
> - **连续数据稳定性**：在连续数据上保持稳定
> - **长程预测**：能够预测长程时间序列
> - **准确性**：预测准确性优于基线方法

## 应用与扩展

- **树搜索**：支持高效的树搜索，用于决策制定。
- **组合性**：可以组合训练数据中观察到的子序列，具有用户确定的记忆范围。
- **因果不确定性**：不同 token 可以有不同的不确定性级别，反映其在序列中的位置。

## 局限性与未来工作

- 当前实现主要基于 RNN 架构，Transformer 实现可能进一步提升性能。
- 在非常长的序列上，计算成本可能成为挑战。
- 未来工作可以探索更复杂的引导策略和更大规模的应用。

## 与其他方法的详细对比

### 去噪机制对比表格

| 特性 | Full Sequence Diffusion | Diffusion Forcing | Teacher Forcing |
|------|------------------------|-------------------|-----------------|
| **噪声级别** | 统一（所有token相同） | 独立（每token不同） | 无噪声（kt=0） |
| **去噪单位** | 整个序列 | 单个token | 单个token |
| **注意力** | 全注意力（非因果） | 因果注意力 | 因果注意力 |
| **隐藏状态** | 无 | 有（编码历史） | 有（编码历史） |
| **去噪步骤** | 整体去噪 | 逐token去噪 | 无去噪（直接预测） |
| **调度矩阵** | 无（统一调度） | 有（灵活调度） | 无 |

### 去噪过程对比

**Full Sequence Diffusion**:
```python
# 所有token使用相同噪声级别
k = random.randint(0, K)  # 统一噪声级别
x_k = forward_diffuse(x_0, k)  # 所有token相同噪声

# 整体去噪
for m in range(M):
    # 所有token一起去噪
    noise_pred = model(x_k, k)  # 全注意力
    x_k = denoise_step(x_k, noise_pred, k)
```

**Diffusion Forcing**:
```python
# 每个token独立采样噪声级别
for t in range(T):
    k_t = random.randint(0, K)  # 独立噪声级别
    x_kt_t = forward_diffuse(x_0_t, k_t)

# 逐token去噪
for m in range(M):
    for t in range(T):
        # 每个token独立去噪
        z = model.update_state(z, x_kt_t, k_t)  # 因果约束
        noise_pred = model.predict_noise(z, x_kt_t, k_t)
        x_kt_t = denoise_step(x_kt_t, noise_pred, k_t)
```

**Teacher Forcing**:
```python
# 所有token无噪声（kt=0）
# 直接预测下一个token，无需去噪
for t in range(T):
    context = x_0[:t]  # 真实历史
    x_pred = model.predict_next(context)  # 直接预测
```

### 关键区别图示

> [!important] 去噪机制的可视化对比
> 
> **Full Sequence Diffusion**:
> ```
> 步骤1: [噪声, 噪声, 噪声, 噪声] → 整体去噪
> 步骤2: [部分, 部分, 部分, 部分] → 整体去噪
> 步骤3: [清晰, 清晰, 清晰, 清晰] → 完成
> ```
> 
> **Diffusion Forcing**:
> ```
> 步骤1: [噪声100, 噪声80, 噪声60, 噪声40] → 逐token去噪
> 步骤2: [噪声80, 噪声60, 噪声40, 噪声20] → 逐token去噪
> 步骤3: [噪声60, 噪声40, 噪声20, 清晰0] → 逐token去噪
> 步骤4: [噪声40, 噪声20, 清晰0, 清晰0] → 逐token去噪
> 步骤5: [噪声20, 清晰0, 清晰0, 清晰0] → 逐token去噪
> 步骤6: [清晰0, 清晰0, 清晰0, 清晰0] → 完成
> ```
> 
> **Teacher Forcing**:
> ```
> 步骤1: 预测token1
> 步骤2: 使用token1预测token2
> 步骤3: 使用token1-2预测token3
> 步骤4: 使用token1-3预测token4
> ```

### 性能对比

| 指标 | Full Sequence | Diffusion Forcing | Teacher Forcing |
|------|---------------|-------------------|-----------------|
| **生成质量** | 高 | 高 | 中等 |
| **长程稳定性** | 好 | 好 | 差 |
| **实时生成** | 不支持 | 支持 | 支持 |
| **引导能力** | 强 | 强 | 无 |
| **变量长度** | 不支持 | 支持 | 支持 |
| **计算效率** | 中等 | 高 | 高 |
| **训练稳定性** | 好 | 好 | 好 |

### Teacher Forcing

> [!warning] Teacher Forcing的局限
> - **训练**：基于真实上下文训练，kt = 0 对所有t
> - **推理**：基于自生成上下文推理，存在暴露偏差
> - **引导**：无法进行引导采样
> - **稳定性**：在连续数据上不稳定，误差累积

**训练代码示例**：
```python
# Teacher Forcing训练
for t in range(1, T+1):
    # 输入：真实的历史帧
    context = x_real[1:t]
    # 目标：预测当前帧
    target = x_real[t]
    # 所有帧都是干净的（kt = 0）
```

### Full Sequence Diffusion

> [!warning] Full Sequence的局限
> - **训练**：所有帧使用相同噪声级别，kt = k
> - **架构**：全注意力，非因果
> - **长度**：只能生成固定长度序列
> - **实时**：无法用于实时生成

**训练代码示例**：
```python
# Full Sequence训练
k = random.randint(0, K)  # 统一噪声级别
for t in range(1, T+1):
    # 所有帧使用相同的噪声级别k
    x_k_t = forward_diffuse(x_real[t], k)
```

### Diffusion Forcing

> [!success] Diffusion Forcing的优势
> - **训练**：每帧独立采样噪声级别
> - **架构**：因果架构，支持自回归生成
> - **长度**：可变长度生成
> - **引导**：支持灵活的引导采样
> - **稳定性**：在连续数据上保持稳定

**训练代码示例**：
```python
# Diffusion Forcing训练
for t in range(1, T+1):
    # 每帧独立采样噪声级别
    kt = random.randint(0, K)
    x_kt_t = forward_diffuse(x_real[t], kt)
    # 因果架构，只能看到过去帧
```

### AR-Diffusion

> [!note] AR-Diffusion的特点
> - 使用因果架构进行全序列文本扩散
> - 噪声级别沿时间轴线性相关（不是独立）
> - 与Diffusion Forcing的关键区别：噪声级别不是独立的

## 🔬 蒙特卡洛引导（MCG）详解

### MCG的原理

> [!important] MCG的核心思想
> 利用Diffusion Forcing的因果性、灵活范围和可变噪声调度，显著提高高回报生成的采样效果。

**MCG算法**：
```python
def monte_carlo_guidance(model, cost_function, num_samples=100):
    best_sequence = None
    best_cost = -inf
    
    for _ in range(num_samples):
        # 采样一个序列
        sequence = model.sample()
        
        # 计算成本
        cost = cost_function(sequence)
        
        # 更新最佳序列
        if cost > best_cost:
            best_cost = cost
            best_sequence = sequence
    
    return best_sequence
```

### MCG的优势

> [!tip] MCG相比标准引导的优势
> 1. **全局优化**：考虑整个序列的成本，而非单步
> 2. **灵活范围**：可以引导任意长度的子序列
> 3. **组合性**：可以组合不同子序列的引导
> 4. **因果性**：引导时保持因果约束

## 📊 完整代码示例

### Diffusion Forcing训练实现

```python
import torch
import torch.nn as nn

class DiffusionForcingTrainer:
    def __init__(self, model, K=1000):
        self.model = model  # 因果架构的模型（RNN或因果Transformer）
        self.K = K  # 最大噪声级别
        
    def forward_diffuse(self, x0, k):
        """前向扩散过程"""
        # 计算噪声调度
        alpha_bar = self.compute_alpha_bar(k)
        
        # 采样噪声
        epsilon = torch.randn_like(x0)
        
        # 添加噪声
        x_k = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * epsilon
        
        return x_k, epsilon
    
    def compute_alpha_bar(self, k):
        """计算ᾱ_k"""
        # 线性噪声调度
        betas = torch.linspace(0.0001, 0.02, self.K)
        alphas = 1 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        
        return alpha_bar[k]
    
    def train_step(self, x_sequence):
        """训练步骤"""
        batch_size, channels, T, H, W = x_sequence.shape
        
        # 存储每帧的预测噪声和真实噪声
        predicted_noises = []
        target_noises = []
        
        # 初始化隐藏状态
        z_prev = torch.zeros(batch_size, self.model.hidden_dim)
        
        for t in range(T):
            # 采样独立噪声级别
            k_t = torch.randint(0, self.K, (1,)).item()
            
            # 前向扩散
            x_kt_t, epsilon_t = self.forward_diffuse(x_sequence[:, :, t], k_t)
            
            # 更新隐藏状态
            z_t = self.model.update_state(z_prev, x_kt_t, k_t)
            
            # 预测噪声
            epsilon_pred = self.model.predict_noise(z_prev, x_kt_t, k_t)
            
            # 存储
            predicted_noises.append(epsilon_pred)
            target_noises.append(epsilon_t)
            
            # 更新前一状态
            z_prev = z_t
        
        # 计算损失
        predicted_noises = torch.stack(predicted_noises, dim=2)
        target_noises = torch.stack(target_noises, dim=2)
        loss = nn.MSELoss()(predicted_noises, target_noises)
        
        return loss

# 使用示例
model = CausalDiffusionModel()
trainer = DiffusionForcingTrainer(model)

for batch in dataloader:
    loss = trainer.train_step(batch)
    loss.backward()
    optimizer.step()
```

### DF Sampling with Guidance实现

```python
import torch

class DiffusionForcingSampler:
    def __init__(self, model, M=50, K=1000):
        self.model = model
        self.M = M  # 去噪步骤数
        self.K = K  # 最大噪声级别
        
    def create_scheduling_matrix(self, T):
        """创建调度矩阵"""
        # 简单策略：所有帧从高噪声开始，逐步降低
        K_matrix = torch.zeros(self.M, T)
        for m in range(self.M):
            for t in range(T):
                # 线性降低噪声级别
                K_matrix[m, t] = int(self.K * (1 - m / self.M))
        
        return K_matrix.long()
    
    def denoise_step(self, x_k, noise_pred, k):
        """单步去噪"""
        # 计算alpha
        alpha = self.compute_alpha(k)
        alpha_bar = self.compute_alpha_bar(k)
        
        # 计算均值
        mean = (1 / torch.sqrt(alpha)) * (
            x_k - (1 - alpha) / torch.sqrt(1 - alpha_bar) * noise_pred
        )
        
        # 添加随机噪声
        sigma = torch.sqrt(1 - alpha_bar)
        noise = torch.randn_like(x_k)
        
        x_prev = mean + sigma * noise
        
        return x_prev
    
    def sample(self, T, guidance_function=None, guidance_scale=1.0):
        """采样序列"""
        # 初始化为纯噪声
        x = torch.randn(1, 3, T, 64, 64)
        
        # 创建调度矩阵
        K_matrix = self.create_scheduling_matrix(T)
        
        # 初始化隐藏状态
        z = torch.zeros(1, self.model.hidden_dim)
        
        # 去噪循环
        for m in range(self.M - 1, -1, -1):
            x_new = torch.zeros_like(x)
            
            for t in range(T):
                # 更新隐藏状态
                z = self.model.update_state(z, x[:, :, t], K_matrix[m+1, t])
                
                # 预测噪声
                noise_pred = self.model.predict_noise(z, x[:, :, t], K_matrix[m, t])
                
                # 去噪
                x_new[:, :, t] = self.denoise_step(
                    x[:, :, t], noise_pred, K_matrix[m, t]
                )
            
            # 应用引导
            if guidance_function is not None:
                gradient = self.compute_guidance_gradient(
                    x_new, guidance_function
                )
                x_new = x_new + guidance_scale * gradient
            
            x = x_new
        
        return x
    
    def compute_guidance_gradient(self, x, guidance_function):
        """计算引导梯度"""
        x.requires_grad_(True)
        cost = guidance_function(x)
        gradient = torch.autograd.grad(cost, x)[0]
        return gradient

# 使用示例
sampler = DiffusionForcingSampler(model)

# 定义引导函数（例如：最大化某些属性）
def guidance_cost(x):
    # 示例：最大化视频的"动态程度"
    return compute_motion_score(x)

# 采样
generated_video = sampler.sample(
    T=100,
    guidance_function=guidance_cost,
    guidance_scale=0.1
)
```

### 蒙特卡洛引导（MCG）实现

```python
def monte_carlo_guidance(model, cost_function, T=100, num_samples=100):
    """蒙特卡洛引导采样"""
    best_sequence = None
    best_cost = -float('inf')
    
    sampler = DiffusionForcingSampler(model)
    
    for _ in range(num_samples):
        # 采样一个序列
        sequence = sampler.sample(T)
        
        # 计算成本
        cost = cost_function(sequence)
        
        # 更新最佳序列
        if cost > best_cost:
            best_cost = cost
            best_sequence = sequence.detach().clone()
    
    return best_sequence

# 使用示例
best_video = monte_carlo_guidance(
    model=model,
    cost_function=guidance_cost,
    T=100,
    num_samples=50
)
```

## 🔧 关键函数解析

### 前向扩散函数

```python
def forward_diffuse(x0, k, K=1000):
    """
    前向扩散过程
    
    Args:
        x0: 原始数据 [batch, channels, ...]
        k: 噪声级别 (0到K)
        K: 最大噪声级别
    
    Returns:
        x_k: 添加噪声后的数据
        epsilon: 添加的噪声
    """
    # 线性噪声调度
    beta_start, beta_end = 0.0001, 0.02
    betas = torch.linspace(beta_start, beta_end, K)
    alphas = 1 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    
    # 采样噪声
    epsilon = torch.randn_like(x0)
    
    # 添加噪声
    alpha_bar_k = alpha_bar[k]
    x_k = torch.sqrt(alpha_bar_k) * x0 + torch.sqrt(1 - alpha_bar_k) * epsilon
    
    return x_k, epsilon
```

### 反向去噪函数

```python
def denoise_step(x_k, noise_pred, k, K=1000):
    """
    反向去噪步骤
    
    Args:
        x_k: 噪声数据
        noise_pred: 预测的噪声
        k: 当前噪声级别
        K: 最大噪声级别
    
    Returns:
        x_prev: 去噪后的数据
    """
    # 计算alpha
    beta_start, beta_end = 0.0001, 0.02
    betas = torch.linspace(beta_start, beta_end, K)
    alphas = 1 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    
    alpha_k = alphas[k]
    alpha_bar_k = alpha_bar[k]
    
    # 计算均值
    mean = (1 / torch.sqrt(alpha_k)) * (
        x_k - (1 - alpha_k) / torch.sqrt(1 - alpha_bar_k) * noise_pred
    )
    
    # 添加随机噪声
    if k > 0:
        sigma = torch.sqrt((1 - alpha_bar_k-1) / (1 - alpha_bar_k) * betas[k])
        noise = torch.randn_like(x_k)
        x_prev = mean + sigma * noise
    else:
        x_prev = mean
    
    return x_prev
```

## 🎯 关键要点总结

> [!quote] Diffusion Forcing的核心贡献
> 1. **独立噪声级别**：每帧可以有不同的噪声级别，统一了TF和Full Sequence
> 2. **因果架构**：保持自回归生成能力，支持实时交互
> 3. **灵活引导**：支持任意范围的引导，包括MCG
> 4. **变量长度**：生成长度可超过训练长度
> 5. **理论保证**：优化ELBO，最大化数据似然下界
> 6. **广泛应用**：视频生成、规划、模仿学习、时间序列

## 🎓 学习建议

> [!note] 理解Diffusion Forcing的路径
> 1. **基础概念**：先理解标准扩散模型（DDPM）的基本原理
> 2. **序列建模**：学习Teacher Forcing和自回归模型的概念
> 3. **噪声级别**：理解噪声级别作为"部分掩码"的统一视角
> 4. **因果架构**：理解因果注意力掩码的作用
> 5. **引导机制**：学习分类器引导和MCG的原理
> 6. **代码实现**：通过代码实现加深理解

## 🔮 未来发展方向

> [!tip] Diffusion Forcing的潜在发展方向
> 1. **Transformer实现**：将RNN架构替换为因果Transformer
> 2. **更大规模**：扩展到更大规模的模型和数据
> 3. **多模态融合**：结合文本、图像、音频等多模态
> 4. **实时优化**：进一步优化推理效率
> 5. **新应用领域**：探索更多应用场景

## 📚 扩展阅读

> [!quote] 相关论文
> - **Self Forcing**: Bridging the Train-Test Gap in Autoregressive Video Diffusion
> - **Causal Forcing**: Autoregressive Diffusion Distillation Done Right
> - **Generative View Stitching**: Camera-guided video generation
> - **Full Sequence Diffusion Models**: Wan, CogVideoX, Step-Video-T2V

## 总结

> [!quote] 核心要点
> Diffusion Forcing 是一种创新的训练范式，成功结合了 next-token 预测和全序列扩散的优点。通过允许每个 token 有独立的噪声级别，它实现了：
> 
> 1. **可变长度生成**：从单个token到数千个token的灵活生成
> 2. **引导采样**：支持任意范围的引导，包括MCG
> 3. **长程稳定性**：在连续数据上保持稳定
> 4. **因果架构**：支持实时交互式生成
> 5. **理论保证**：优化ELBO，最大化数据似然下界
> 
> 该方法在视频生成、规划和决策制定等多个领域展示了强大的能力，为序列生成建模提供了新的方向。

> [!quote] 引用
> Chen, B., Monso, D. M., Du, Y., Simchowitz, M., Tedrake, R., & Sitzmann, V. (2024). Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion. *Advances in Neural Information Processing Systems*, 37.
