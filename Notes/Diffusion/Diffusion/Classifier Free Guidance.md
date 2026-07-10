# Classifier Free Guidance

> 从 Classifier Guidance 到 Classifier-Free Guidance：条件生成模型如何引导采样方向。

---

## 1. 问题背景：为什么要"引导"？

扩散模型从纯噪声逐步去噪生成样本。无条件生成时，模型从 $p(x)$ 中采样，输出是随机的。实际需求往往是**条件生成**——给定文本/类别，生成符合描述的图像。

核心问题：**如何在去噪过程中注入条件信号，使生成结果既高质量又忠实于条件？**

---

## 2. Classifier Guidance

### 2.1 核心思想

Dhariwal & Nichol, 2021（"Diffusion Models Beat GANs on Image Synthesis"）提出：利用一个**预训练的分类器** $p_\phi(y \mid x_t)$ 来引导去噪方向。

在去噪的每一步，用分类器对含噪图像 $x_t$ 做分类，计算梯度，将无条件生成的采样方向向"更可能属于目标类别 $y$"的方向偏移。

### 2.2 数学推导

目标是从条件分布 $p(x \mid y)$ 采样。由贝叶斯定理：

$$
\nabla_{x_t} \log p(x_t \mid y) = \nabla_{x_t} \log p(x_t) + \nabla_{x_t} \log p(y \mid x_t)
$$

- $\nabla_{x_t} \log p(x_t)$：无条件扩散模型的 score（即 $\epsilon_\theta(x_t, t)$ 给出的方向）
- $\nabla_{x_t} \log p(y \mid x_t)$：分类器对含噪样本的梯度

实际操作中，对预测的均值做修正：

$$
\hat{\mu}_\theta(x_t, y) = \mu_\theta(x_t) + s \cdot \Sigma \cdot \nabla_{x_t} \log p_\phi(y \mid x_t)
$$

其中 $s$ 是引导强度（guidance scale），$\Sigma$ 是当前步的方差。

### 2.3 优缺点

| 优点 | 缺点 |
|------|------|
| 理论清晰，直接利用贝叶斯公式 | 需要额外训练一个**在含噪样本上工作**的分类器 |
| 引导强度 $s$ 可调，灵活控制条件服从度 | 分类器需要在每个噪声等级 $t$ 都能正确分类，训练成本高 |
| 显著提升图像质量和多样性 | 仅适用于有标签的分类任务，无法处理自由文本条件 |

---

## 3. Classifier-Free Guidance (CFG)

### 3.1 核心思想

Ho & Salimans, 2022 提出：**不需要额外的分类器**，而是用扩散模型自身同时学习条件生成和无条件生成，通过两者的线性组合来模拟分类器梯度的效果。

关键洞察：Classifier Guidance 中的梯度项 $\nabla_{x_t} \log p(y \mid x_t)$ 可以近似为：

$$
\nabla_{x_t} \log p(y \mid x_t) \approx \nabla_{x_t} \log p(x_t \mid y) - \nabla_{x_t} \log p(x_t)
$$

即条件 score 与无条件 score 之差。CFG 直接在**预测噪声层面**做这个差值操作。

### 3.2 训练方式

训练时以概率 $p_{\text{uncond}}$（通常 10%~20%）**随机丢弃条件**：

```python
# 训练时的条件丢弃
if random() < p_uncond:
    c = ∅          # 用空标记（如空字符串 embedding）替代真实条件
else:
    c = real_cond   # 保留真实条件
```

这样同一个模型 $\epsilon_\theta(x_t, t, c)$ 既学会了条件去噪，也学会了无条件去噪。

### 3.3 采样时的组合

采样时，对每一步同时做两次前向传播：

$$
\hat{\epsilon} = \epsilon_\theta(x_t, t, \varnothing) + w \cdot \big[\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \varnothing)\big]
$$

其中：
- $\epsilon_\theta(x_t, t, \varnothing)$：无条件预测（$c = \varnothing$）
- $\epsilon_\theta(x_t, t, c)$：条件预测
- $w$：引导强度（guidance scale），通常取 7~15

整理后：

$$
\hat{\epsilon} = (1 - w) \cdot \epsilon_\theta(x_t, t, \varnothing) + w \cdot \epsilon_\theta(x_t, t, c)
$$

当 $w = 1$ 时退化为纯条件生成；$w > 1$ 时增强条件服从度，代价是略微降低多样性。

### 3.4 计算优化：一步完成

朴素实现需要两次前向传播。实际工程中，将条件和无条件的输入拼接为一个 batch：

```python
# 将 [无条件, 条件] 拼成 batch_size=2*B 的输入，一次前向传播完成
x_double = torch.cat([x_t, x_t], dim=0)
t_double = torch.cat([t, t], dim=0)
c_double = torch.cat([empty_cond, real_cond], dim=0)

eps_all = model(x_double, t_double, c_double)  # 一次前向
eps_uncond, eps_cond = eps_all.chunk(2, dim=0)

eps_guided = eps_uncond + w * (eps_cond - eps_uncond)
```

### 3.5 CFG 的本质理解

CFG 在 score 层面做了如下操作：

$$
s_{\text{cfg}} = s_{\text{uncond}} + w \cdot (s_{\text{cond}} - s_{\text{uncond}})
$$

几何直觉：
- $s_{\text{cond}} - s_{\text{uncond}}$ 是从"任意图像"指向"符合条件的图像"的**方向向量**
- $w > 1$ 时沿此方向"走更远"，结果更符合条件，但可能偏离数据流形
- 这等价于隐式地训练了一个分类器 $\nabla_{x_t} \log p(y \mid x_t)$，无需显式训练

### 3.6 与 Classifier Guidance 的对比

| | Classifier Guidance | Classifier-Free Guidance |
|---|---|---|
| 是否需要额外模型 | 需要含噪分类器 | 不需要，单模型搞定 |
| 条件类型 | 仅分类标签 | 任意条件（文本、图像、类别等） |
| 训练复杂度 | 高（需训练含噪分类器） | 低（随机丢弃条件即可） |
| 推理开销 | 1 次扩散前向 + 1 次分类器前向 | 2 次扩散前向（可 batch 合并为 1 次） |
| 生成质量 | 好 | 更好（已成为事实标准） |
| 适用范围 | 学术论文为主 | 几乎所有现代 T2I 模型 |

---

## 4. 现代 T2I 模型中的条件嵌入

### 4.1 整体架构

以 Stable Diffusion / SDXL / FLUX 为代表的 Latent Diffusion Model (LDM) 架构：

```
文本提示 ──→ Text Encoder ──→ 条件 embedding ──┐
                                                ├─→ Cross-Attention ──→ UNet/DiT ──→ 去噪预测
噪声 latent ──→ 加噪 latent ─────────────────┘
```

条件信号通过 **Cross-Attention** 注入到去噪网络中。

### 4.2 文本编码器的演进

| 模型 | 文本编码器 | 输出维度 | 特点 |
|------|-----------|---------|------|
| SD 1.x | CLIP ViT-L/14 | 768 | 英文为主，语义理解有限 |
| SD 2.x | OpenCLIP ViT-H/14 | 1024 | 更好的语义，但对负面提示不敏感 |
| SDXL | CLIP ViT-L + OpenCLIP ViT-bigG（双编码器拼接） | 1280 + 1280 = 2560 | 双编码器互补，长文本能力提升 |
| DALL·E 3 | T5-XXL | — | 更强的语言理解 |
| FLUX | CLIP + T5-XXL（双编码器） | — | T5 提供长文本语义，CLIP 提供视觉对齐 |

### 4.3 Cross-Attention 条件注入

在 UNet 的每个 Cross-Attention 层中：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right) V
$$

- $Q$：来自图像 latent 的 query（空间特征）
- $K, V$：来自文本 embedding 的 key 和 value

文本 token 的语义信息通过注意力权重"路由"到图像的不同空间位置。

### 4.4 CFG 在 T2I 中的实践

现代 T2I 模型的 CFG 实现：

```python
# Stable Diffusion / FLUX 的 CFG 采样
prompt_embeds = text_encoder(prompt)          # 条件 embedding
negative_embeds = text_encoder(negative_prompt) # 负面提示（默认空字符串）

# 两种 embedding 拼接为 batch
encoder_hidden_states = torch.cat([negative_embeds, prompt_embeds])

# UNet/DiT 一次前向
noise_pred = unet(latent, t, encoder_hidden_states)
noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)

# CFG 组合
noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
```

### 4.5 CFG 与 Negative Prompt 的关系

Negative Prompt 不是独立于 CFG 的功能，而是 CFG 框架的自然延伸。

**核心思想**：将 CFG 公式中的 $\varnothing$（空条件）替换为负面提示的 embedding。

原始 CFG：

$$
\hat{\epsilon} = \epsilon_\theta(x_t, t, \varnothing) + w \cdot \big[\epsilon_\theta(x_t, t, c_{\text{pos}}) - \epsilon_\theta(x_t, t, \varnothing)\big]
$$

加入 Negative Prompt 后：

$$
\hat{\epsilon} = \epsilon_\theta(x_t, t, c_{\text{neg}}) + w \cdot \big[\epsilon_\theta(x_t, t, c_{\text{pos}}) - \epsilon_\theta(x_t, t, c_{\text{neg}})\big]
$$

**直觉理解**：
- 无 Negative Prompt：从"任意图像"方向向"符合条件的图像"方向偏移
- 有 Negative Prompt：从"符合负面描述的图像"方向向"符合条件的图像"方向偏移

等于告诉模型：**往这个方向走，但离那个方向越远越好。**

**实际例子**：
```
正面提示: "a photo of a cat"
负面提示: "blurry, low quality"
```
- 基线 $\epsilon(x_t, t, c_{\text{neg}})$：预测"如果要生成模糊低质量的猫"的去噪方向
- 条件 $\epsilon(x_t, t, c_{\text{pos}})$：预测"如果要生成猫"的去噪方向
- CFG 组合后：沿着"生成猫但远离模糊低质量"的方向去噪

**代码实现**：
```python
if negative_prompt:
    c_neg = text_encoder(negative_prompt)   # 用户指定的负面条件
else:
    c_neg = text_encoder("")                # 空字符串 = 无条件

eps_guided = eps_neg + w * (eps_pos - eps_neg)
```

**关键结论**：Negative Prompt 之所以能工作，完全是因为 CFG 的框架允许把"基线条件"从空字符串换成任意内容。没有 CFG，就没有 Negative Prompt。

### 4.6 引导强度的权衡

| guidance_scale $w$ | 效果 |
|---|---|
| $w = 1$ | 纯条件生成，多样性高，但条件服从度低 |
| $w = 7 \sim 8$ | SD 1.5 默认值，质量与多样性平衡 |
| $w = 5 \sim 6$ | SDXL 默认值（因模型更强，不需要太高的引导） |
| $w > 20$ | 过度引导，图像出现过饱和、伪影、重复纹理 |

### 4.6 高级引导技术

| 技术 | 思路 |
|------|------|
| **Dynamic CFG** | 不同去噪步使用不同的 $w$，前期高引导建立结构，后期低引导补充细节 |
| **CFG Rescale**（Ghost Sampling） | 对 CFG 结果做 rescale，减轻过饱和问题 |
| **Smooth CFG** | 在 $w$ 上加噪声平滑，减少采样过程中的震荡 |
| **Attend-and-Excite** | 确保所有文本 token 都被图像 attend 到，提升组合生成的语义一致性 |
| **Structured CFG** | 对不同 token 维度使用不同的引导强度 |

---

## 5. 总结

```
Classifier Guidance          Classifier-Free Guidance
（需额外含噪分类器）          （单模型，随机丢弃条件）
        │                            │
        ▼                            ▼
  p(x|y) ∝ p(x)·p(y|x)     ε̂ = ε(x,∅) + w·[ε(x,c) - ε(x,∅)]
  用分类器梯度引导方向         用模型自身的条件/无条件预测之差引导
        │                            │
        ▼                            ▼
  仅适用于分类条件             适用于任意条件（文本、图像...）
                             已成为现代 T2I 的事实标准
                                      │
                                      ▼
                             Negative Prompt：将 ∅ 替换为 c_neg
                             ε̂ = ε(x,c_neg) + w·[ε(x,c_pos) - ε(x,c_neg)]
```

**一句话总结**：CFG 的精髓在于——让模型同时学会"有条件"和"无条件"两种行为，采样时用两者的差值来放大条件的影响，从而在不引入额外模型的前提下实现高质量的条件引导生成。Negative Prompt 是 CFG 的自然延伸——将"无条件"基线替换为"负面条件"，让模型远离不想要的方向。
