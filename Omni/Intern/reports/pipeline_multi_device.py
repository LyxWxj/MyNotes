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
# in_channels=64 是 packed 特征维度，实际 latent 通道数 = in_channels // 4 = 16
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
latents_4d = randn_tensor(
    (1, num_channels_latents, HEIGHT // vae_scale_factor, WIDTH // vae_scale_factor),
    generator=generator, device=device_tf, dtype=torch.bfloat16,
)
print(f"  Latents (4D): {latents_4d.shape}")

# ---- C) Prepare timesteps (用 packed seq_len，与参考 pipeline 对齐) ----
sigmas = np.linspace(1.0, 1.0 / NUM_STEPS, NUM_STEPS)
lh = HEIGHT // vae_scale_factor  # 128
lw = WIDTH  // vae_scale_factor  # 128
image_seq_len = (lh // 2) * (lw // 2)  # = 4096 (packed)
mu = calculate_shift(image_seq_len)
scheduler.set_timesteps(NUM_STEPS, sigmas=sigmas, mu=mu, device=device_tf)
timesteps = scheduler.timesteps
img_shapes = [[(1, lh // 2, lw // 2)]]  # packed 空间维度 = (64, 64)

# pack latents for scheduler: [1, 64, 128, 128] -> [1, 4096, 256]
latents = pack_latents(latents_4d, 1, num_channels_latents, lh, lw)
dumper.dump_tensor("latents_init", latents)
print(f"  Latents (packed for scheduler): {latents.shape}")

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
            # transformer 直接接收 packed latents [1, 4096, 64]
            # 参考 pipeline: timestep / 1000 (行 237)
            noise_pred = transformer(
                hidden_states=latents,
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_mask=mask_for_model,
                timestep=(t / 1000).unsqueeze(0).to(dtype=latents.dtype, device=device_tf),
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
print(f"  Latents before VAE: {latents.shape}")

latents = latents.to(vae.dtype)
# 与参考 pipeline 对齐：用 z_dim=16 做 mean/std 归一化
latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(device_te, latents.dtype)
latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(device_te, latents.dtype)
latents = latents / latents_std + latents_mean

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
