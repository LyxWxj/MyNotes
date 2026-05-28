# 娄雨轩 Daily Report — 2026-05-21

## QwenImage Multi-Instance Pipeline 调试准备

### 背景

vllm-omni 的 `multi_instance_scheduler.py` + `pipeline_qwen_image_multi_instance.py` 无法生成正确的图片。19 日已验证权重加载完全正确（diffusers 参考与 vllm-omni 两种加载方式下 1933 个参数全部匹配），说明问题不在权重，而在推理过程的激活值或组件间交互。

已有正确参考实现：`gen_image.py`（diffusers 标准 pipeline，单卡可以正常出图）。目标是通过对比两者推理过程的中间激活，定位多卡 pipeline 的具体故障点。

---

### 工作内容

#### 1. 搭建多卡最小复现脚本 `pipeline_multi_device.py`

为便于控制变量调试，在不改动 vllm-omni 源码的前提下，复现其核心架构：

- **模型分布：** text_encoder + VAE 在 `npu:2`，transformer 在 `npu:3`
- **Transformer 加载：** 复用 vllm-omni 原生的 meta → to_empty → safetensors + key 转换 + RoPE 重建流程，与 `multi_instance_scheduler.py` 保持一致
- **Denoising loop：** 手动编写，逐 step 调用 vllm-omni transformer forward + scheduler.step
- **对比能力：** 脚本可与 `gen_image.py` 对齐 prompt、seed、step 数，直接对比每步输出

#### 2. 编写中间激活对比工具

| 脚本 | 用途 |
|------|------|
| `gen_image.py` | 正确基线。注册 transformer 逐 block hook + text_encoder + vae + 每步 latents dump，出图正常，结果可信 |
| `compare_intermediate_outputs.py` | 精简版，只 dump transformer 整体输出、text_encoder、vae、每步 latents，适合快速比对 |
| `pipeline_multi_device.py` | 多卡版，同样 dump transformer 整体输出、text_encoder、vae、每步 latents，与上述脚本对齐 |

三个脚本共用同一套 `ActivationDumper` 工具类，dump 格式一致，可直接逐层比对 shape、mean、std、abs_max 等统计量。

#### 3. 验证 transformer 权重加载一致性 `compare_weights.py`

在调试之前需要确认：`pipeline_multi_device.py` 使用的 vllm-omni 加载方式（meta → to_empty → safetensors + QKV 融合 + RoPE 重建）是否与 diffusers 标准 `from_pretrained` 等价。编写 `compare_weights.py` 分别用两种方式加载 transformer，将 diffusers state_dict key 转换为 vllm-omni 格式（to_q/k/v → to_qkv 合并，to_out.0 → to_out 重命名），逐参数比对 shape 和数值。结果：1933 个参数全部匹配。证明权重加载链路没有问题，问题出在推理过程的激活值。

#### 4. 排除 NPU FA mask 形状兼容性问题

`pipeline_multi_device.py` 使用 vllm-omni 原生 transformer forward，其内置了 `encoder_hidden_states_mask.all() → mask = None` 逻辑，天然绕过 NPU FlashAttentionScore 对 mask 形状的严格限制。

---

### 当前状态

- [x] 权重一致性验证（19 日）— 1933 参数全部匹配
- [x] 正确参考脚本 `gen_image.py` — 单卡正常出图，含逐 block dump
- [x] 多卡最小复现 `pipeline_multi_device.py` — 架构对齐 vllm-omni，可 dump 对比
- [x] 中间激活对比工具 `compare_intermediate_outputs.py` — 快速比对入口
- [x] 最小复现样例搭建完成

### 下一步

1. 在 NPU 环境运行 `gen_image.py` 和 `pipeline_multi_device.py`，分别产出参考基线和多卡版本的中间激活 dump
2. 从 text_encoder 输出开始逐层比对（text_encoder → transformer 每步输出 → latents → VAE），锁定第一个出现偏差的环节
3. 定位到具体环节后，对比对应模块的输入/输出/参数，找出 vllm-omni 实现与 diffusers 参考的具体差异

---

### 附录：`pipeline_multi_device.py`

```python
"""
最小 Pipeline: text_encoder + VAE 在 npu:2, transformer 在 npu:3
transformer 使用 vllm-omni 的加载方式 (meta -> to_empty -> safetensors + key 转换).
"""

# !!! 必须在 import diffusers / vllm-omni 之前设 !!!
import os
os.environ["DIFFUSERS_ATTN_BACKEND"] = "_native_npu"

import gc
import glob
import json
import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor

# ====================================================================== #
#  配置                                                                   #
# ====================================================================== #
MODEL = "/home/ma-user/work/model"
TE_VAE_DEVICE = "npu:2"
TF_DEVICE = "npu:3"
DUMP_DIR = "dump_multi_device"
PROMPT = "a cute cat sitting on a table"
HEIGHT, WIDTH = 1024, 1024
NUM_STEPS = 30
SEED = 42


# ====================================================================== #
#  Diffusers -> vllm-omni key 转换 (从 compare_weights.py 复制)            #
# ====================================================================== #
import re

def convert_diffusers_to_vllm_keys(sd: dict) -> dict:
    qkv_groups: dict[str, dict[str, tuple[str, torch.Tensor]]] = {}
    _qkv_re = re.compile(r"^(.+\.(?:to_[qkv]|add_[qkv]_proj))\.(weight|bias)$")
    keys_to_remove = []
    for key in list(sd.keys()):
        m = _qkv_re.match(key)
        if m:
            full_prefix = m.group(1)
            suffix = m.group(2)
            parent = full_prefix.rsplit(".", 1)[0]
            if ".to_" in full_prefix:
                base = parent + ".to_qkv"
            else:
                base = parent + ".add_kv_proj"
            group_key = f"{base}@@{suffix}"
            qkv_groups.setdefault(group_key, {})[full_prefix] = (key, sd[key])
            keys_to_remove.append(key)
    for key in keys_to_remove:
        del sd[key]
    def _qkv_sort_key(prefix: str) -> int:
        if prefix.endswith(".to_q") or prefix.endswith(".add_q_proj"): return 0
        elif prefix.endswith(".to_k") or prefix.endswith(".add_k_proj"): return 1
        elif prefix.endswith(".to_v") or prefix.endswith(".add_v_proj"): return 2
        return 3
    for group_key, parts in qkv_groups.items():
        base, suffix = group_key.split("@@")
        sorted_prefixes = sorted(parts.keys(), key=_qkv_sort_key)
        combined = torch.cat([parts[p][1] for p in sorted_prefixes], dim=0)
        sd[f"{base}.{suffix}"] = combined
    for key in list(sd.keys()):
        new_key = key.replace(".to_out.0.", ".to_out.")
        if new_key != key:
            sd[new_key] = sd.pop(key)
    return sd


# ====================================================================== #
#  加载 Transformer (vllm-omni 方式: meta -> to_empty -> safetensors)      #
# ====================================================================== #
def load_transformer_vllm(model_path, device="npu:3"):
    # 初始化分布式 (单进程, gloo 即可满足 vllm-omni 的 group 需求)
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29599")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", init_method="env://",
                                world_size=1, rank=0)

    from vllm_omni.diffusion.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
        model_parallel_is_initialized,
    )
    if not model_parallel_is_initialized():
        init_distributed_environment(world_size=1, rank=0, local_rank=0)
        initialize_model_parallel(
            data_parallel_size=1, cfg_parallel_size=1,
            sequence_parallel_size=1, ulysses_degree=1, ring_degree=1,
            tensor_parallel_size=1, pipeline_parallel_size=1,
        )

    from vllm_omni.diffusion.data import (
        DiffusionParallelConfig, OmniDiffusionConfig, TransformerConfig,
    )
    from vllm.config import DeviceConfig, VllmConfig
    from vllm.config.vllm import set_current_vllm_config
    from vllm_omni.diffusion.forward_context import set_forward_context
    from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import (
        QwenImageTransformer2DModel,
    )
    from vllm_omni.diffusion.utils.tf_utils import get_transformer_config_kwargs

    print(f"[vllm-omni] 构建 OmniDiffusionConfig...")
    config_path = os.path.join(model_path, "transformer", "config.json")
    with open(config_path, "r") as f:
        tf_cfg = json.load(f)

    od_config = OmniDiffusionConfig(
        model=model_path,
        model_class_name="QwenImagePipeline",
        parallel_config=DiffusionParallelConfig(
            pipeline_parallel_size=1, tensor_parallel_size=1, data_parallel_size=1,
        ),
    )
    od_config.tf_model_config = TransformerConfig.from_dict(tf_cfg)

    vllm_config = VllmConfig(device_config=DeviceConfig(device="npu"))
    transformer_kwargs = get_transformer_config_kwargs(
        od_config.tf_model_config, QwenImageTransformer2DModel
    )

    print(f"[vllm-omni] meta 上构建 transformer 骨架 (bfloat16)...")
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("meta"):
            with set_current_vllm_config(vllm_config), set_forward_context(
                vllm_config=vllm_config, omni_diffusion_config=od_config,
            ):
                transformer = QwenImageTransformer2DModel(
                    od_config=od_config, **transformer_kwargs
                )

        for p in transformer.parameters(recurse=True):
            if p.dtype == torch.float32:
                p.data = p.data.to(dtype=torch.bfloat16)
        for b in transformer.buffers(recurse=True):
            if b.dtype == torch.float32:
                b.data = b.data.to(dtype=torch.bfloat16)

        print(f"[vllm-omni] materialize 到 {device}...")
        transformer = transformer.to_empty(device=device)
    finally:
        torch.set_default_dtype(original_dtype)

    # 加载 safetensors + key 转换
    transformer_dir = os.path.join(model_path, "transformer")
    weight_files = sorted(glob.glob(os.path.join(transformer_dir, "*.safetensors")))
    if not weight_files:
        weight_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))

    merged_sd = {}
    for wf in weight_files:
        sd = load_file(wf)
        sd = {k: v.to(dtype=torch.bfloat16) for k, v in sd.items()}
        merged_sd.update(sd)
        del sd

    print(f"[vllm-omni] raw keys={len(merged_sd)}, 转换 key 格式...")
    converted_sd = convert_diffusers_to_vllm_keys(merged_sd)
    del merged_sd
    gc.collect()

    missing, unexpected = transformer.load_state_dict(converted_sd, strict=False)
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected keys")
    if missing:
        non_buf = [m for m in missing
                   if not (m.endswith("pos_freqs") or m.endswith("neg_freqs"))]
        if non_buf:
            print(f"  [warn] {len(non_buf)} non-buffer missing keys")
    del converted_sd
    gc.collect()

    # 重建 RoPE buffer
    for module in transformer.modules():
        if hasattr(module, "pos_freqs") and module.pos_freqs.device.type == "meta":
            pos_index = torch.arange(4096, device=device)
            axes_dim = module.axes_dim
            theta = module.theta
            module.pos_freqs = torch.cat([
                module.rope_params(pos_index, axes_dim[0], theta),
                module.rope_params(pos_index, axes_dim[1], theta),
                module.rope_params(pos_index, axes_dim[2], theta),
            ], dim=1)
        if hasattr(module, "neg_freqs") and module.neg_freqs.device.type == "meta":
            neg_index = torch.arange(4096, device=device).flip(0) * -1 - 1
            axes_dim = module.axes_dim
            theta = module.theta
            module.neg_freqs = torch.cat([
                module.rope_params(neg_index, axes_dim[0], theta),
                module.rope_params(neg_index, axes_dim[1], theta),
                module.rope_params(neg_index, axes_dim[2], theta),
            ], dim=1)

    transformer.eval()
    gc.collect()
    mem_alloc = torch.npu.memory_allocated(device) / 1024**3
    mem_reserved = torch.npu.memory_reserved(device) / 1024**3
    print(f"[vllm-omni] transformer 就绪 (alloc={mem_alloc:.2f}GB, reserved={mem_reserved:.2f}GB)")

    return transformer, vllm_config, od_config


# ====================================================================== #
#  Dump 工具                                                              #
# ====================================================================== #
class ActivationDumper:
    def __init__(self, dump_dir="dump_activations"):
        self.dump_dir = dump_dir
        self.hooks = []
        self.step = 0
        os.makedirs(dump_dir, exist_ok=True)

    def _save(self, name, tensor):
        d = os.path.join(self.dump_dir, f"step_{self.step:04d}")
        os.makedirs(d, exist_ok=True)
        t = tensor.detach().float().cpu()
        torch.save(t, os.path.join(d, f"{name}.pt"))
        meta = {
            "step": self.step, "name": name,
            "shape": list(t.shape), "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "mean": t.mean().item(), "std": t.std().item(),
            "abs_max": t.abs().max().item(),
            "has_nan": bool(torch.isnan(t).any()),
            "has_inf": bool(torch.isinf(t).any()),
        }
        with open(os.path.join(d, "meta.jsonl"), "a") as f:
            f.write(json.dumps(meta) + "\n")

    def hook_module(self, module, name):
        def fn(mod, inp, out):
            if isinstance(out, torch.Tensor):
                self._save(name, out)
            elif isinstance(out, tuple) and len(out) > 0 and isinstance(out[0], torch.Tensor):
                self._save(name, out[0])
        h = module.register_forward_hook(fn)
        self.hooks.append(h)

    def dump_tensor(self, name, tensor):
        self._save(name, tensor)

    def next_step(self):
        self.step += 1

    def close(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ====================================================================== #
#  辅助函数                                                               #
# ====================================================================== #
def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096,
                    base_shift=0.5, max_shift=1.15):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def pack_latents(latents, batch_size, num_ch, h, w):
    latents = latents.view(batch_size, num_ch, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(batch_size, (h // 2) * (w // 2), num_ch * 4)


def unpack_latents(latents, height, width, vae_scale_factor):
    B, _, C = latents.shape
    lh, lw = height // vae_scale_factor, width // vae_scale_factor
    latents = latents.view(B, lh // 2, lw // 2, C // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    return latents.reshape(B, C // 4, 1, lh, lw)


PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    "Describe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:"
    "<|im_end|>\n"
    "<|im_start|>user\n"
    "{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


# ====================================================================== #
#  1. 加载模型                                                             #
# ====================================================================== #
dumper = ActivationDumper(DUMP_DIR)

print(f"[1/5] 加载 Text Encoder -> {TE_VAE_DEVICE}")
text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL, subfolder="text_encoder", torch_dtype=torch.bfloat16,
).to(TE_VAE_DEVICE)
text_encoder.eval()

print(f"[2/5] 加载 Tokenizer")
tokenizer = Qwen2Tokenizer.from_pretrained(MODEL, subfolder="tokenizer")

print(f"[3/5] 加载 Transformer (vllm-omni 方式) -> {TF_DEVICE}")
transformer, vllm_config, od_config = load_transformer_vllm(MODEL, device=TF_DEVICE)

print(f"[4/5] 加载 VAE -> {TE_VAE_DEVICE}")
from diffusers.models import AutoencoderKLQwenImage
vae = AutoencoderKLQwenImage.from_pretrained(
    MODEL, subfolder="vae", torch_dtype=torch.bfloat16,
).to(TE_VAE_DEVICE)
vae.eval()
if hasattr(vae, "enable_tiling"):
    vae.enable_tiling()
if hasattr(vae, "enable_slicing"):
    vae.enable_slicing()

print(f"[5/5] 加载 Scheduler")
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL, subfolder="scheduler")


# ====================================================================== #
#  2. 注册 hook — 只 dump transformer 整体输出 + text_encoder + vae        #
# ====================================================================== #
dumper.hook_module(transformer, "transformer_output")
dumper.hook_module(text_encoder.model, "text_encoder_model")
dumper.hook_module(vae, "vae")


# ====================================================================== #
#  3. 执行 Pipeline                                                       #
# ====================================================================== #
print(f"\n开始生成 (seed={SEED}, steps={NUM_STEPS}, {HEIGHT}x{WIDTH})")

device_te = torch.device(TE_VAE_DEVICE)
device_tf = torch.device(TF_DEVICE)
vae_scale_factor = 2 ** len(vae.temperal_downsample) if hasattr(vae, "temperal_downsample") else 8
num_channels_latents = getattr(transformer, "in_channels", 64) // 4

# ---- A) Text Encode (npu:2) ----
prompt_text = PROMPT_TEMPLATE.format(PROMPT)
txt_tokens = tokenizer(
    prompt_text, max_length=1024, padding=True, truncation=True, return_tensors="pt",
)

with torch.inference_mode():
    txt_tokens = txt_tokens.to(device_te)
    encoder_out = text_encoder(
        input_ids=txt_tokens.input_ids,
        attention_mask=txt_tokens.attention_mask,
        output_hidden_states=True,
    )
    hidden_states = encoder_out.hidden_states[-1]
    drop_idx = 34
    prompt_embeds = hidden_states[:, drop_idx:].contiguous()
    prompt_embeds_mask = txt_tokens.attention_mask[:, drop_idx:].contiguous()
    del hidden_states, encoder_out

dumper.dump_tensor("prompt_embeds_raw", prompt_embeds)
dumper.dump_tensor("prompt_embeds_mask_raw", prompt_embeds_mask)
print(f"  Text Encode: embeds={prompt_embeds.shape}, mask={prompt_embeds_mask.shape}")

# ---- B) Prepare latents (npu:3) ----
generator = torch.Generator(device="cpu").manual_seed(SEED)
latents = randn_tensor(
    (1, num_channels_latents, HEIGHT // vae_scale_factor, WIDTH // vae_scale_factor),
    generator=generator, device=device_tf, dtype=torch.bfloat16,
)
latents = pack_latents(latents, 1, num_channels_latents,
                       HEIGHT // vae_scale_factor, WIDTH // vae_scale_factor)
dumper.dump_tensor("latents_init", latents)
print(f"  Latents: {latents.shape}")

# ---- C) Prepare timesteps ----
sigmas = np.linspace(1.0, 1.0 / NUM_STEPS, NUM_STEPS)
image_seq_len = latents.shape[1]
mu = calculate_shift(image_seq_len)
scheduler.set_timesteps(NUM_STEPS, sigmas=sigmas, mu=mu, device=device_tf)
timesteps = scheduler.timesteps
img_shapes = [[(1, HEIGHT // vae_scale_factor // 2, WIDTH // vae_scale_factor // 2)]]

# ---- D) Denoising loop (npu:3, vllm-omni transformer) ----
prompt_embeds = prompt_embeds.to(device=device_tf, dtype=torch.bfloat16)
prompt_embeds_mask = prompt_embeds_mask.to(device=device_tf)
txt_seq_lens = [int(x) for x in prompt_embeds_mask.long().sum(dim=1).tolist()]
mask_for_model = prompt_embeds_mask.to(dtype=prompt_embeds.dtype)

from vllm.config.vllm import set_current_vllm_config
from vllm_omni.diffusion.forward_context import set_forward_context

with torch.inference_mode():
    with set_current_vllm_config(vllm_config), set_forward_context(
        vllm_config=vllm_config, omni_diffusion_config=od_config,
    ):
        for step_idx, t in enumerate(timesteps):
            noise_pred = transformer(
                hidden_states=latents,
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_mask=mask_for_model,
                timestep=t.unsqueeze(0).to(dtype=latents.dtype, device=device_tf),
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
                return_dict=False,
            )[0]

            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            dumper.dump_tensor(f"latent_step_{step_idx:03d}", latents)
            del noise_pred
            dumper.next_step()
            print(f"  step {step_idx + 1}/{NUM_STEPS}, t={t.item():.4f}")

# ---- E) VAE decode (npu:2) ----
latents = latents.to(device_te, dtype=torch.bfloat16)
latents = unpack_latents(latents, HEIGHT, WIDTH, vae_scale_factor)
dumper.dump_tensor("latents_before_vae", latents)

latents_mean = torch.tensor(vae.config.latents_mean, device=device_te, dtype=vae.dtype)
latents_std = torch.tensor(vae.config.latents_std, device=device_te, dtype=vae.dtype)
latents = latents.to(vae.dtype)
latents = latents * latents_std.view(1, -1, 1, 1, 1) + latents_mean.view(1, -1, 1, 1, 1)

with torch.inference_mode():
    image = vae.decode(latents, return_dict=False)[0][:, :, 0]

dumper.dump_tensor("vae_output", image)
print(f"  VAE decode: {image.shape}")

# ---- F) Save result ----
from diffusers.image_processor import VaeImageProcessor
image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2)
pil_image = image_processor.postprocess(image, output_type="pil")[0]
pil_image.save("output_multi_device.png")
dumper.close()

print(f"\nDone!  output_multi_device.png, dump: {DUMP_DIR}/")
```

### 附录 B：`compare_weights.py`

```python
"""
比对两种加载方式的 transformer 权重：
  1. diffusers QwenImageTransformer2DModel.from_pretrained（参考）
  2. vllm-omni QwenImageTransformer2DModel（meta → to_empty → safetensors + key 转换）

由于 vllm-omni 版本将 to_q/to_k/to_v 合并为 to_qkv，参数名不同，
需要把 diffusers 的 state_dict 转为 vllm-omni 格式后再比对。
"""

import gc
import glob
import json
import os
import re
import sys

import torch
from safetensors.torch import load_file

MODEL = "/home/ma-user/work/model"
DEVICE = "cpu"  # 比对用 CPU，避免 NPU 显存问题


# ====================================================================== #
#  diffusers → vllm-omni key 转换（复现 _convert_diffusers_to_vllm_state_dict）#
# ====================================================================== #
def convert_diffusers_to_vllm_keys(sd: dict) -> dict:
    """把 diffusers 格式的 state_dict key 转为 vllm-omni 格式。

    转换规则:
      to_q + to_k + to_v          → to_qkv   (cat dim=0, q/k/v 顺序)
      add_q_proj + add_k_proj + add_v_proj → add_kv_proj
      to_out.0.weight             → to_out.weight
    """
    qkv_groups: dict[str, dict[str, tuple[str, torch.Tensor]]] = {}
    _qkv_re = re.compile(
        r"^(.+\.(?:to_[qkv]|add_[qkv]_proj))\.(weight|bias)$"
    )

    keys_to_remove = []
    for key in list(sd.keys()):
        m = _qkv_re.match(key)
        if m:
            full_prefix = m.group(1)
            suffix = m.group(2)
            parent = full_prefix.rsplit(".", 1)[0]
            if ".to_" in full_prefix:
                base = parent + ".to_qkv"
            else:
                base = parent + ".add_kv_proj"
            group_key = f"{base}@@{suffix}"
            qkv_groups.setdefault(group_key, {})[full_prefix] = (key, sd[key])
            keys_to_remove.append(key)

    for key in keys_to_remove:
        del sd[key]

    def _qkv_sort_key(prefix: str) -> int:
        if prefix.endswith(".to_q") or prefix.endswith(".add_q_proj"):
            return 0
        elif prefix.endswith(".to_k") or prefix.endswith(".add_k_proj"):
            return 1
        elif prefix.endswith(".to_v") or prefix.endswith(".add_v_proj"):
            return 2
        return 3

    for group_key, parts in qkv_groups.items():
        base, suffix = group_key.split("@@")
        sorted_prefixes = sorted(parts.keys(), key=_qkv_sort_key)
        combined = torch.cat([parts[p][1] for p in sorted_prefixes], dim=0)
        sd[f"{base}.{suffix}"] = combined

    # to_out.0. → to_out.
    for key in list(sd.keys()):
        new_key = key.replace(".to_out.0.", ".to_out.")
        if new_key != key:
            sd[new_key] = sd.pop(key)

    return sd


# ====================================================================== #
#  方式 1: diffusers from_pretrained                                      #
# ====================================================================== #
def load_via_diffusers(model_path):
    from diffusers.models import QwenImageTransformer2DModel

    print("[方式1] diffusers from_pretrained 加载中...")
    transformer = QwenImageTransformer2DModel.from_pretrained(
        model_path, subfolder="transformer", torch_dtype=torch.bfloat16,
    )
    return transformer


# ====================================================================== #
#  方式 2: vllm-omni meta → to_empty → safetensors                        #
# ====================================================================== #
def load_via_vllm_omni(model_path, device="cpu"):
    # 初始化 vllm-omni 分布式环境（单进程）
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29599")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", init_method="env://",
                                world_size=1, rank=0)

    from vllm_omni.diffusion.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
        model_parallel_is_initialized,
    )
    if not model_parallel_is_initialized():
        init_distributed_environment(world_size=1, rank=0, local_rank=0)
        initialize_model_parallel(
            data_parallel_size=1, cfg_parallel_size=1,
            sequence_parallel_size=1, ulysses_degree=1, ring_degree=1,
            tensor_parallel_size=1, pipeline_parallel_size=1,
        )

    from vllm_omni.diffusion.data import (
        DiffusionParallelConfig, OmniDiffusionConfig, TransformerConfig,
    )
    from vllm.config import DeviceConfig, VllmConfig
    from vllm.config.vllm import set_current_vllm_config
    from vllm_omni.diffusion.forward_context import set_forward_context
    from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import (
        QwenImageTransformer2DModel,
    )
    from vllm_omni.diffusion.utils.tf_utils import get_transformer_config_kwargs

    print("[方式2] vllm-omni 手动加载中...")

    # 1) 构建 OmniDiffusionConfig
    config_path = os.path.join(model_path, "transformer", "config.json")
    with open(config_path, "r") as f:
        tf_cfg = json.load(f)

    od_config = OmniDiffusionConfig(
        model=model_path,
        model_class_name="QwenImagePipeline",
        parallel_config=DiffusionParallelConfig(
            pipeline_parallel_size=1, tensor_parallel_size=1, data_parallel_size=1,
        ),
    )
    od_config.tf_model_config = TransformerConfig.from_dict(tf_cfg)

    # 2) 构建 transformer 空壳
    vllm_config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    transformer_kwargs = get_transformer_config_kwargs(
        od_config.tf_model_config, QwenImageTransformer2DModel
    )

    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("meta"):
            with set_current_vllm_config(vllm_config), set_forward_context(
                vllm_config=vllm_config, omni_diffusion_config=od_config,
            ):
                transformer = QwenImageTransformer2DModel(
                    od_config=od_config, **transformer_kwargs
                )

        for p in transformer.parameters(recurse=True):
            if p.dtype == torch.float32:
                p.data = p.data.to(dtype=torch.bfloat16)
        for b in transformer.buffers(recurse=True):
            if b.dtype == torch.float32:
                b.data = b.data.to(dtype=torch.bfloat16)

        transformer = transformer.to_empty(device=device)
    finally:
        torch.set_default_dtype(original_dtype)

    # 3) 加载 safetensors + key 转换
    transformer_dir = os.path.join(model_path, "transformer")
    weight_files = sorted(glob.glob(os.path.join(transformer_dir, "*.safetensors")))
    if not weight_files:
        weight_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))

    merged_sd = {}
    for wf in weight_files:
        sd = load_file(wf)
        sd = {k: v.to(dtype=torch.bfloat16) for k, v in sd.items()}
        merged_sd.update(sd)
        del sd

    print(f"  原始 key 数: {len(merged_sd)}")
    converted_sd = convert_diffusers_to_vllm_keys(merged_sd)
    del merged_sd
    gc.collect()
    print(f"  转换后 key 数: {len(converted_sd)}")

    missing, unexpected = transformer.load_state_dict(converted_sd, strict=False)
    if unexpected:
        print(f"  [警告] {len(unexpected)} unexpected keys")
    if missing:
        non_buf = [m for m in missing if not (m.endswith("pos_freqs") or m.endswith("neg_freqs"))]
        if non_buf:
            print(f"  [警告] {len(non_buf)} non-buffer missing keys")

    # 4) 重建 RoPE buffer
    for module in transformer.modules():
        if hasattr(module, "pos_freqs") and module.pos_freqs.device.type == "meta":
            pos_index = torch.arange(4096, device=device)
            axes_dim = module.axes_dim
            theta = module.theta
            module.pos_freqs = torch.cat([
                module.rope_params(pos_index, axes_dim[0], theta),
                module.rope_params(pos_index, axes_dim[1], theta),
                module.rope_params(pos_index, axes_dim[2], theta),
            ], dim=1)
        if hasattr(module, "neg_freqs") and module.neg_freqs.device.type == "meta":
            neg_index = torch.arange(4096, device=device).flip(0) * -1 - 1
            axes_dim = module.axes_dim
            theta = module.theta
            module.neg_freqs = torch.cat([
                module.rope_params(neg_index, axes_dim[0], theta),
                module.rope_params(neg_index, axes_dim[1], theta),
                module.rope_params(neg_index, axes_dim[2], theta),
            ], dim=1)

    del converted_sd
    gc.collect()

    return transformer


# ====================================================================== #
#  验证参数共享                                                           #
# ====================================================================== #
def check_parameter_sharing(model):
    """检查模型中 image_rope_prepare.img_in 和 img_in 是否共享参数。"""
    img_in = model.img_in
    img_in_rp = model.image_rope_prepare.img_in
    is_same_module = img_in is img_in_rp
    is_same_weight = img_in.weight is img_in_rp.weight
    is_same_bias = img_in.bias is img_in_rp.bias

    print(f"\n{'='*70}")
    print(f"参数共享检查:")
    print(f"  model.img_in is model.image_rope_prepare.img_in: {is_same_module}")
    print(f"  weight 是同一对象: {is_same_weight}")
    print(f"  bias 是同一对象:   {is_same_bias}")

    # 检查 state_dict 中 img_in 相关的 key
    sd = model.state_dict()
    img_in_keys = [k for k in sd.keys() if 'img_in' in k]
    print(f"  state_dict 中 img_in 相关 key: {img_in_keys}")

    # 检查 pos_embed 共享
    pos_embed = model.pos_embed
    pos_embed_rp = model.image_rope_prepare.pos_embed
    print(f"  model.pos_embed is model.image_rope_prepare.pos_embed: {pos_embed is pos_embed_rp}")
    print(f"{'='*70}\n")

    return is_same_module, is_same_weight


# ====================================================================== #
#  比对                                                                   #
# ====================================================================== #
def compare_models(model_diffusers, model_vllm, atol=1e-5):
    """比对两个模型。

    关键: vllm-omni 版本将 to_q/to_k/to_v 合并为 to_qkv，
    参数名不同。需要把 diffusers 的 state_dict 转为 vllm-omni 格式后再比对。
    """
    sd_diff = model_diffusers.state_dict()
    sd_vllm = model_vllm.state_dict()

    # 将 diffusers 的 key 转为 vllm-omni 格式
    sd_diff_converted = convert_diffusers_to_vllm_keys(dict(sd_diff))

    keys_diff = set(sd_diff_converted.keys())
    keys_vllm = set(sd_vllm.keys())

    only_diff = keys_diff - keys_vllm
    only_vllm = keys_vllm - keys_diff
    common = keys_diff & keys_vllm

    print(f"\n{'='*70}")
    print(f"diffusers 转换后 key 数: {len(keys_diff)}")
    print(f"vllm-omni key 数:        {len(keys_vllm)}")
    print(f"共有 key 数:             {len(common)}")
    print(f"仅 diffusers 有:         {len(only_diff)}")
    print(f"仅 vllm-omni 有:         {len(only_vllm)}")

    if only_diff:
        print(f"\n  仅 diffusers 有的 key (前 10):")
        for k in sorted(only_diff)[:10]:
            print(f"    {k}  shape={sd_diff_converted[k].shape}")

    if only_vllm:
        print(f"\n  仅 vllm-omni 有的 key (前 10):")
        for k in sorted(only_vllm)[:10]:
            print(f"    {k}  shape={sd_vllm[k].shape}")

    # 比对共有参数
    mismatch_shape = []
    mismatch_value = []
    match_count = 0

    for key in sorted(common):
        ta = sd_diff_converted[key]
        tb = sd_vllm[key]

        if ta.shape != tb.shape:
            mismatch_shape.append((key, ta.shape, tb.shape))
            continue

        if not torch.allclose(ta.float(), tb.float(), atol=atol):
            max_diff = (ta.float() - tb.float()).abs().max().item()
            mean_diff = (ta.float() - tb.float()).abs().mean().item()
            mismatch_value.append((key, ta.shape, max_diff, mean_diff))
        else:
            match_count += 1

    print(f"\n{'='*70}")
    print(f"完全匹配: {match_count}/{len(common)}")

    if mismatch_shape:
        print(f"\nshape 不匹配: {len(mismatch_shape)} 个")
        for key, sa, sb in mismatch_shape:
            print(f"  {key}: diffusers={sa} vs vllm={sb}")

    if mismatch_value:
        print(f"\n数值不匹配 (atol={atol}): {len(mismatch_value)} 个")
        for key, shape, max_d, mean_d in mismatch_value:
            print(f"  {key}  shape={shape}  max_diff={max_d:.8f}  mean_diff={mean_d:.8f}")
    else:
        print(f"\n所有共有参数数值完全一致 (atol={atol})")

    total_issues = len(only_diff) + len(only_vllm) + len(mismatch_shape) + len(mismatch_value)
    print(f"\n{'='*70}")
    if total_issues == 0:
        print("结论: 两个模型完全一致 ✓")
    else:
        print(f"结论: 发现 {total_issues} 处差异 ✗")
    print(f"{'='*70}\n")

    return {
        "only_diff": only_diff,
        "only_vllm": only_vllm,
        "mismatch_shape": mismatch_shape,
        "mismatch_value": mismatch_value,
        "match_count": match_count,
    }


# ====================================================================== #
#  主函数                                                                 #
# ====================================================================== #
if __name__ == "__main__":
    model_path = MODEL

    model_diffusers = load_via_diffusers(model_path)
    model_vllm = load_via_vllm_omni(model_path, device=DEVICE)

    # 验证参数共享
    check_parameter_sharing(model_vllm)

    # 比对权重
    result = compare_models(model_diffusers, model_vllm, atol=1e-5)
```

### 附录 C：`compare_intermediate_outputs.py`

```python
import os
import json
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, DiffusionPipeline
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer
from diffusers.models import QwenImageTransformer2DModel, AutoencoderKLQwenImage

MODEL = "/home/ma-user/work/model"
DEVICE = "npu:0"
DUMP_DIR = "dump_activations"


# ====================================================================== #
#  Dump 工具                                                              #
# ====================================================================== #
class ActivationDumper:
    def __init__(self, dump_dir="dump_activations"):
        self.dump_dir = dump_dir
        self.hooks = []
        self.step = 0
        os.makedirs(dump_dir, exist_ok=True)

    def _save(self, name, tensor):
        d = os.path.join(self.dump_dir, f"step_{self.step:04d}")
        os.makedirs(d, exist_ok=True)
        t = tensor.detach().float().cpu()
        torch.save(t, os.path.join(d, f"{name}.pt"))
        meta = {
            "step": self.step,
            "name": name,
            "shape": list(t.shape),
            "dtype": str(tensor.dtype),
            "mean": t.mean().item(),
            "std": t.std().item(),
            "abs_max": t.abs().max().item(),
            "has_nan": bool(torch.isnan(t).any()),
            "has_inf": bool(torch.isinf(t).any()),
        }
        with open(os.path.join(d, "meta.jsonl"), "a") as f:
            f.write(json.dumps(meta) + "\n")

    def hook_module(self, module, name):
        def fn(mod, inp, out):
            if isinstance(out, torch.Tensor):
                self._save(name, out)
            elif isinstance(out, tuple) and len(out) > 0 and isinstance(out[0], torch.Tensor):
                self._save(name, out[0])
        h = module.register_forward_hook(fn)
        self.hooks.append(h)

    def dump_tensor(self, name, tensor):
        self._save(name, tensor)

    def next_step(self):
        self.step += 1

    def close(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ====================================================================== #
#  加载模型                                                               #
# ====================================================================== #
dumper = ActivationDumper(DUMP_DIR)

print("加载 Transformer...")
transformer = QwenImageTransformer2DModel.from_pretrained(
    MODEL, subfolder="transformer", torch_dtype=torch.bfloat16,
)

print("加载 Text Encoder...")
text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL, subfolder="text_encoder", torch_dtype=torch.bfloat16,
).to(DEVICE)

print("加载 VAE...")
vae = AutoencoderKLQwenImage.from_pretrained(
    MODEL, subfolder="vae", torch_dtype=torch.bfloat16,
).to(DEVICE)

print("加载 Tokenizer...")
tokenizer = Qwen2Tokenizer.from_pretrained(MODEL, subfolder="tokenizer")

print("加载 Scheduler...")
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL, subfolder="scheduler")


# ====================================================================== #
#  注册 hook（dump transformer 最终输出 + text_encoder + vae）             #
# ====================================================================== #
dumper.hook_module(transformer, "transformer_final")
dumper.hook_module(text_encoder.model, "text_encoder_model")
dumper.hook_module(vae, "vae")


# ====================================================================== #
#  组装 Pipeline 并用 callback 在每步 dump latents                         #
# ====================================================================== #
print("组装 Pipeline...")
pipe = DiffusionPipeline.from_pretrained(
    MODEL,
    transformer=transformer,
    text_encoder=text_encoder,
    vae=vae,
    tokenizer=tokenizer,
    scheduler=scheduler,
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()


# 用 callback_on_step_end 在每个 denoising step 后 dump latents
def dump_callback(pipeline, step_index, timestep, callback_kwargs):
    latents = callback_kwargs.get("latents", None)
    if latents is not None:
        dumper.dump_tensor(f"latent_step_{step_index:03d}", latents)
    dumper.next_step()
    print(f"  step {step_index + 1}/30, t={timestep.item():.4f}")
    return callback_kwargs


# ====================================================================== #
#  生成                                                                   #
# ====================================================================== #
prompt = "a cute cat sitting on a table"
print(f"生成中: {prompt}")
generator = torch.Generator(device="cpu").manual_seed(42)
image = pipe(
    prompt=prompt,
    num_inference_steps=30,
    generator=generator,
    callback_on_step_end=dump_callback,
).images[0]

dumper.close()
image.save("output.png")
print("图片已保存到: output.png")
print(f"激活张量已保存到: {DUMP_DIR}/")
```
