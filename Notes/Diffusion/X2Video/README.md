---
type: Note
related_to: "[[X2Video]]"
status: Active
---

# X2Video：视频扩散与自回归生成

这是视频扩散、因果视频模型和实时 AR 蒸馏的专题入口。建议先阅读基础概念，再按方法演进阅读 Forcing 家族，最后查阅具体基座模型与长视频控制方法。

## 阅读路径

### 1. 基础概念

- [[video-diffusion-time-axes]]：帧时间、扩散时间、PF-ODE 轨迹与 student self-rollout 的统一解释。
- [[full-sequence-diffusion]]：双向、整段视频扩散的标准处理流程。
- [[causal-video-model]]：从联合视频分布到因果 AR 视频模型的总览。

### 2. Forcing 与扩散蒸馏

- [[diffusion-forcing]]：将 next-token prediction 与 full-sequence diffusion 统一起来。
- [[self-forcing]]：用学生 self-rollout 对齐训练和推理。
- [[causal-forcing]]：用 AR 教师修复双向教师到 AR 学生的帧级注入性问题。
- [[causal-forcing-plus-plus]]：用 causal consistency distillation 替代昂贵的因果 ODE 轨迹蒸馏。
- [[rolling-forcing]]：滚动去噪窗口与长视频生成。
- [[omniforcing]]：音视频联合的非对称因果蒸馏。

### 3. 基座模型

- [[wan]]：Wan 系列视频基础模型与推理优化。
- [[cogvideox]]：CogVideoX 的 3D 因果 VAE 与 Expert Transformer。
- [[step-video-t2v]]：Step-Video-T2V 的模型、数据和训练实践。

### 4. 长视频与可控生成

- [[generative-view-stitching]]：无需训练的生成式视角拼接。

## 核心概念关系

```text
Full-sequence diffusion
    -> 因果注意力 / AR 分解
    -> Teacher Forcing / Diffusion Forcing
    -> Self Forcing（训练 = 学生 self-rollout）
    -> Causal Forcing（AR 教师的因果 ODE）
    -> Causal Forcing++（在线 causal CD）
```

## 术语速查

| 术语 | 含义 |
|---|---|
| 帧索引 `i` | 视频序列中的第 `i` 帧，属于视频时间轴 |
| 扩散时间 `t` | 当前帧的噪声强度/去噪进度，属于扩散时间轴 |
| PF-ODE 轨迹 | 沿扩散时间从噪声状态积分到干净状态的路径 |
| AR rollout | 沿帧索引从前到后生成视频 |
| student self-rollout | 后续帧使用学生自己生成的历史前缀 |
| 少步蒸馏 | 减少每个帧或 chunk 内的扩散去噪调用次数 |
