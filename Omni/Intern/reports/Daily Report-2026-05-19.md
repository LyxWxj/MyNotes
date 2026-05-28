# 娄雨轩 Daily Report — 2026-05-19

## QwenImage Multi-Instance Pipeline 问题排查

### 背景

启动 `script/start_multi_instance.sh` 时出现大量 WARNING，输出图像为无意义噪点。排查 `multi_instance_scheduler.py` 的权重加载逻辑。

---

### 问题 1: unexpected keys / missing keys WARNING

**现象：**
```
WARNING Rank 3: 98 unexpected keys in ./model/transformer/diffusion_pytorch_model-00003-of-00009.safetensors
WARNING Rank 2: 362 weight keys missing, e.g. ['image_rope_prepare.img_in.bias', 'transformer_blocks.0.attn.add_kv_proj.bias', ...]
```

**根因：** safetensors 的 key 命名（diffusers 风格）与 vllm-omni 模型的 key 命名不一致：

| 差异 | safetensors (diffusers) | vllm-omni 模型 |
|------|------------------------|---------------|
| img_in 位置 | `img_in.weight` (顶层) | `image_rope_prepare.img_in.weight` (包在 ImageRopePrepare 里) |
| QKV 投影 | `to_q`, `to_k`, `to_v` (分开) | `to_qkv` (QKVParallelLinear 融合) |
| add KV 投影 | `add_q_proj`, `add_k_proj`, `add_v_proj` | `add_kv_proj` (融合) |
| 输出投影 | `to_out.0.weight` (Sequential) | `to_out.weight` (RowParallelLinear) |

**修复：** 在 `load_state_dict` 之前做 key 重命名和权重堆叠（见下方代码）。

**注意：** `load_weights()` 方法不能直接用，因为它依赖 vLLM 自定义 Parameter 的 `weight_loader` 属性，而 scheduler 用 `torch.device("meta")` + `to_empty()` 创建的模型参数是普通 `torch.nn.Parameter`，没有该属性。

---

### 问题 2: QKV 堆叠顺序错误（导致输出噪点）

**现象：** 修复 key 映射后仍输出噪点。

**根因：** safetensors 的 key 按字母序存储：`to_k`, `to_q`, `to_v`。原始代码用 `torch.cat` 按遍历顺序拼接，实际拼出 `[k, q, v]` 而非预期的 `[q, k, v]`。

vLLM 的 `load_weights` 用 `weight_loader(param, weight, shard_id)` 按 shard_id 指定偏移位置，不受 key 顺序影响。但手动 `torch.cat` 是顺序敏感的。

**修复：** 用 staging dict 收集 q/k/v，最后按固定 `["q", "k", "v"]` 顺序拼接。

---

### 问题 3: `image_rope_prepare.img_in` missing key（可忽略）

**现象：** 修复后仍有 2 个 missing key：`image_rope_prepare.img_in.bias/weight`。

**原因：** `self.img_in = nn.Linear(...)` 先作为顶层 submodule 注册，然后传给 `ImageRopePrepare`。PyTorch state_dict 只记录第一次注册的路径，所以实际 key 是 `img_in.*`，`image_rope_prepare.img_in.*` 不会出现。两者指向同一个对象，权重已通过 `img_in.*` 加载。可忽略。

---

### 问题 4: `_init_shared_services` 未指定 bfloat16（导致文本编码结果错误）

**现象：** transformer 权重加载修复后，输出仍与 diffusers 参考 pipeline 不一致。

**根因：** `_init_shared_services` 中加载 text_encoder 和 vae 时未显式指定 `torch_dtype=torch.bfloat16`，导致模型以默认 float32 加载。文本编码结果与参考 pipeline（显式 bfloat16）存在数值差异，进而影响去噪过程。

**修复：** 在 `from_pretrained` 调用中显式指定 `torch_dtype=torch.bfloat16`，与参考 pipeline 对齐：

```python
text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    self.model_path,
    subfolder="text_encoder",
    local_files_only=local_files_only,
    torch_dtype=torch.bfloat16,  # 显式指定
).to(device)

vae = AutoencoderKLQwenImage.from_pretrained(
    self.model_path,
    subfolder="vae",
    local_files_only=local_files_only,
    torch_dtype=torch.bfloat16,  # 显式指定
).to(device)
```

修复后输出已对齐参考 pipeline。

---

### 当前状态：输出已对齐参考 pipeline

---

### 最终修复代码

`multi_instance_scheduler.py` 第 4 节（加载权重）替换为：

```python
# ---- 4) 加载权重 ----
from safetensors.torch import load_file

transformer_dir = os.path.join(self.model_path, "transformer")
weight_files = sorted(glob.glob(os.path.join(transformer_dir, "*.safetensors")))
if not weight_files:
    weight_files = sorted(glob.glob(os.path.join(self.model_path, "*.safetensors")))

if weight_files:
    logger.info(
        f"Rank {self.local_rank}: loading {len(weight_files)} safetensors files"
    )
    all_sd: dict[str, torch.Tensor] = {}
    for wf in weight_files:
        sd = load_file(wf, device="cpu")
        sd = {k: v.to(dtype=torch.bfloat16) for k, v in sd.items()}
        all_sd.update(sd)
        del sd
        gc.collect()

    stacked_params_mapping = [
        (".to_qkv", ".to_q", "q"),
        (".to_qkv", ".to_k", "k"),
        (".to_qkv", ".to_v", "v"),
        (".add_kv_proj", ".add_q_proj", "q"),
        (".add_kv_proj", ".add_k_proj", "k"),
        (".add_kv_proj", ".add_v_proj", "v"),
    ]
    new_sd: dict[str, torch.Tensor] = {}
    qkv_staging: dict[str, dict[str, torch.Tensor]] = {}

    for key, val in all_sd.items():
        # to_out.0.xxx -> to_out.xxx (diffusers Sequential -> plain Linear)
        if ".to_out.0." in key:
            new_sd[key.replace(".to_out.0.", ".to_out.")] = val
            continue

        # q/k/v -> 收集到 staging
        matched = False
        for fused_suffix, src_suffix, shard_id in stacked_params_mapping:
            if src_suffix + "." in key:
                new_key = key.replace(src_suffix + ".", fused_suffix + ".")
                if new_key not in qkv_staging:
                    qkv_staging[new_key] = {}
                qkv_staging[new_key][shard_id] = val
                matched = True
                break
        if matched:
            continue

        # 其余 key 原样保留
        new_sd[key] = val

    # 按 q, k, v 固定顺序拼接
    for new_key, shards in qkv_staging.items():
        parts = []
        for sid in ["q", "k", "v"]:
            if sid in shards:
                parts.append(shards[sid])
        new_sd[new_key] = torch.cat(parts, dim=0)

    del all_sd, qkv_staging
    gc.collect()
    missing, unexpected = transformer.load_state_dict(new_sd, strict=False)
    if unexpected:
        logger.warning(f"Rank {self.local_rank}: {len(unexpected)} unexpected keys")
    non_buffer_missing = [
        m for m in missing if not (m.endswith("pos_freqs") or m.endswith("neg_freqs"))
    ]
    if non_buffer_missing:
        logger.warning(
            f"Rank {self.local_rank}: {len(non_buffer_missing)} weight keys missing, "
            f"e.g. {sorted(non_buffer_missing)}"
        )
    del new_sd
    gc.collect()
    torch.npu.empty_cache()
else:
    logger.warning(
        f"Rank {self.local_rank}: no safetensors weights found; transformer is random-init"
    )
```

---

### 问题 5: 两种加载方式的权重一致性验证

**目的：** 验证 `multi_instance_scheduler.py` 的手动加载方式（meta → to_empty → safetensors stream load + QKV 堆叠 + RoPE 重建）与 diffusers 参考流水线的 `from_pretrained` 加载方式产出的模型状态是否一致。

**方法：** 编写对比脚本 `compare_weights.py`，分别用两种方式加载 transformer，逐参数比对 state_dict（key 集合、shape、数值）。

**结果：** 两种加载方式的模型状态完全一致（1933 个参数，全部匹配，无数值差异）。

**结论：** 权重加载逻辑无误，问题不在权重本身，而在推理过程中的激活值。

## 下一步
需要对 diffusion 每个 step 的中间激活张量进行 dump 调试，但该过程耗时较长且难以定位问题根源。
