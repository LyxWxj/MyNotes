# 娄雨轩 Daily Report — 2026-05-14

## 1. 多实例服务启动验证

多实例服务已可正常启动运行，3 个实例均健康：

```bash
bash scripts/start_multi_instance.sh start -m ../model -p 9000 -n 4
```

启动后系统报告两轮 metrics，均显示 3/3 实例健康：

```
============================================================
INFO:__main__:
============================================================
SYSTEM METRICS REPORT
============================================================
Uptime: 60s
Total Requests: 0
Successful: 0
Failed: 0
Rejected (over capacity): 0
Availability: 100.00%
Healthy Instances: 3/3
------------------------------------------------------------
INSTANCE DETAILS:
  Instance rank=1: processed=0, avg_time=0.00s, errors=0, error_rate=0.00%, healthy=True
  Instance rank=2: processed=0, avg_time=0.00s, errors=0, error_rate=0.00%, healthy=True
  Instance rank=3: processed=0, avg_time=0.00s, errors=0, error_rate=0.00%, healthy=True
============================================================
INFO:__main__:
============================================================
SYSTEM METRICS REPORT
============================================================
Uptime: 90s
Total Requests: 0
Successful: 0
Failed: 0
Rejected (over capacity): 0
Availability: 100.00%
Healthy Instances: 3/3
------------------------------------------------------------
INSTANCE DETAILS:
Instance rank=1: processed=0, avg_time=0.00s, errors=0, error_rate=0.00%, healthy=True
Instance rank=2: processed=0, avg_time=0.00s, errors=0, error_rate=0.00%, healthy=True
Instance rank=3: processed=0, avg_time=0.00s, errors=0, error_rate=0.00%, healthy=True
============================================================
```

---

## 2. Bug 修复：RoPE 设备不匹配

### 2.1 问题

启动后 rank1 实例报错：

```
RuntimeError: Expected all tensors to be on the same device. Expected NPU tensor,
please check whether the input tensor device is correct.
```

报错位置：`qwen_image_transformer.py:327` 的 `rope_params` 方法。

### 2.2 原因

`QwenEmbedLayer3DRope.rope_params` 中调用 
```python
torch.outer(index, 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)))`。
```
`index` 来自 `torch.arange(4096, device=device)` 位于 NPU，而右侧临时计算的 `1.0 / torch.pow(theta, ...)` 在 CPU 上生成，`torch.outer` 要求两个输入在同一设备。

### 2.3 修复

移除 `torch.arange` 的 `device` 参数，改为在 CPU 上生成后由下游自动搬运：

```python
# 修改前
pos_index = torch.arange(4096, device=device)

# 修改后
pos_index = torch.arange(4096)
```

---

## 3. 图像生成质量问题排查

### 3.1 问题一：输出全黑 / 极暗

**现象**：调用 `/v1/images/generations` 接口生成的图片几乎全黑。

```bash
curl -X POST http://localhost:9000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "a beautiful sunset over mountains, photorealistic",
    "negative_prompt": "blurry, low quality",
    "size": "1024x1024",
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
    "seed": 42
  }' -o response.json

# 提取 base64 并解码为 png
python3 -c "
import json, base64
with open('response.json') as f:
    data = json.load(f)
img_b64 = data['data'][0]['b64_json']
with open('output.png', 'wb') as f:
    f.write(base64.b64decode(img_b64))
print('Saved to output.png')
"
```

![output.png](output.png)

**疑似原因**：`multi_instance_launcher.py:278` 的 `_result_to_b64_png` 方法缺少反归一化。VAE decode 输出范围为 `[-1, 1]`，原代码直接 `value * 255`，导致：
- 负值 clip 到 0（黑）
- 零值为 0（黑）
- 仅正值有微弱亮度

**修复**：

```python
# 修改前
image_np = (image_np * 255).clip(0, 255).astype(np.uint8)

# 修改后
image_np = ((image_np + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
```

### 3.2 问题二：guidance_scale 未传递

修复反归一化后图像依旧异常（![output(2).png](output(2).png)）。排查发现 `guidance_scale` 参数未从 API 层传递到推理管线，导致 CFG 始终使用默认值。

**修复**：在以下三个位置新增 `guidance_scale` 参数传递：

**1) `multi_instance_launcher.py`** — API 层接收并下发：

```python
guidance_scale = request_data.get("guidance_scale")
guidance_scale = float(guidance_scale) if guidance_scale is not None else None

result = self._process_generation(prompt, w, h, steps, seed, guidance_scale)
```

**2) `pipeline_qwen_image_multi_instance.py`** — 管线层接收并传入采样参数：

```python
def generate_image(self, ..., guidance_scale: Optional[float] = None):
    sampling_params = OmniDiffusionSamplingParams(
        ...,
        guidance_scale=guidance_scale,
    )
```

**3) `_execute_pipeline`** — 写入 metadata：

```python
metadata = {
    ...,
    "guidance_scale": guidance_scale if guidance_scale is not None else 1.0,
}
```

### 3.3 当前状态

传递 `guidance_scale=7.5` 后图像质量仍不理想，输出依旧模糊：

![output(3).png](output(3).png)

---

## 4. 待解决

- [ ] 图像模糊问题：需进一步排查 scheduler 配置、采样步数、或模型权重加载是否正确
- [ ] 加载模型的时候出现Unexpected Keyword，并且有些模型参数疑似在safeTensor中找不到（Attention-qk_projection）。