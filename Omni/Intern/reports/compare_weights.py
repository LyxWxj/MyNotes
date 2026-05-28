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
