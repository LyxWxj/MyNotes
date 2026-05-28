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
