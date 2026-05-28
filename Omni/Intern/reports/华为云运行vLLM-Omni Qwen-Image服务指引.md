## 镜像选择

vLLM,vLLM-ascend的安装需要gcc>=9，因此在选择镜像的过程中尽量选gcc版本高的镜像，然而镜像版本上并没有标注gcc/g++版本，不过我们可以选择尽可能高的pytorch版本（高pytorch版本也依赖高gcc版本）\
![[Pasted image 20260515164654.png]]\
最后我们选择西南-贵阳一区域$\rightarrow$ 环境配置$\rightarrow$Verl$\rightarrow$ `0.8.0-pytorch_2.9.0-cann_8.5.1-py_3.11-hce_2.0.2512-aarch64-snt9b`\
硬件实例规格选择`4* ascend-snt9b3 | 96 vCPUs | 768 GiB (modelarts.bm.arm.d910b.kat2ne.48xlarge.4`，对应4块昇腾910B2 64GB

## 环境配置

```bash
# 检查gcc版本
gcc --version

# 检查torch, torch_npu版本
pip list | grep torch*
# 2.9.0

# 检查vllm, vllm-ascend版本
pip list | grep vllm*
0.16.0
```

镜像已经预先安装好pytorch和vllm但是vllm并非是我们需要的版本(0.14.0)\
然而不能通过`pip install vllm==0.14.0`来覆盖原来的`vllm 0.16.0`\
在此之前先克隆项目并下载模型

```bash
# 存放模型的文件夹
mkdir model
# 下载模型
pip install modelcope
modelscope download --model Qwen/Qwen-Image --local_dir ./model

# 下载项目
git clone https://git.bookug.cc/lingchaofan/vllm-omni.git
```

该项目有三个分支:

```bash
cd vllm-omni && git branch -a
```

> stage_separation\
> state_separation\
> remotes/origin/stage_separation\
> remotes/origin/state_separation\
> remotes/origin/v0.14.0

切换到`stage_separation`分支

```bash
git switch stage_separation
```

需要开一个python虚拟环境

```bash
pip install uv
```

使用uv管理虚拟环境，在当前目录下初始化uv环境

```bash
uv venv .venv
source .venv/bin/activate
```

在安装当前项目之前要将`pyproject.toml`中的`fa3-fwd==0.01`改为`fa3-fwd>=0.01`(华为云上找不到`fa3-fwd==0.01`版本)

```bash
# 设置镜像源
export UV_INDEX_URL=https://mirrors.huaweicloud.com/repository/pypi/simple
# 安装项目
uv pip install -e .
```

但是此时(.venv)虚拟环境下还没有安装vllm，需要手动安装vllm和vllm-ascend

```bash
uv pip install vllm==0.14.0
uv pip install vllm-ascend==0.14.0rc1
```

> 注意不可以写成一步: `uv pip install vllm==0.14.0 vllm-ascend==0.14.0rc1`,会出现依赖解析错误

## 启动服务

在vllm-omni目录下启动服务：

```bash
bash scripts/start_multi_instance.sh start -m ./model -p 9000 -n 4
```

可能遇到的问题：

### 找不到`multi_instance_laucher.py`

**解决方法**：修改`start_multi_instance.sh`脚本 line118`local source_base="/home/wa-user/work/vllm-omni"`line254`local launcher_script="/home/ma-user/work/vllm-omni/vllm_omni/diffusion/scheduler/multi_instance_launcher.py"`

### TypeError:

```python
^TypeErrornon-default argument 'stage_connector_config' follows default argument
: 
non-default argument 'stage_connector_config' follows default argument  File "/home/ma-user/anaconda3/envs/PyTorch-2.9.0/lib/python3.11/dataclasses.py", line 1027, in _process_class
```

**原因**：\
`vllm_omni/config/model.py` 中 `OmniModelConfig` 使用了双层装饰器：

```python
@config
@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class OmniModelConfig(ModelConfig):
```

而 vllm-omni 0.18.0 版本中仅使用单层：

```python
@config(config=ConfigDict(arbitrary_types_allowed=True))
class OmniModelConfig(ModelConfig):
```

双层装饰器导致 pydantic 在处理继承自 `ModelConfig`（本身也是 pydantic dataclass）的子类时，字段排序出错。

**解决方案**（任选其一）：

- 去掉 `@config`，保留 `@dataclass(config=ConfigDict(...))`
- 去掉 `@dataclass(config=ConfigDict(...))`，将 `@config` 改为 `@config(config=ConfigDict(arbitrary_types_allowed=True))`

### 模型加载时发生设备不匹配：

发生在`rope_param`中的设备不匹配

```bash
RuntimeError: Expected all tensors to be on the same device. Expected NPU tensor,
please check whether the input tensor device is correct.
```

**解决方法**\
`multi_instance_schedular.py` line 426,438\
修改`torch.arange`

```python
# 修改前
pos_index = torch.arange(4096, device=device)

# 修改后
pos_index = torch.arange(4096)
```

修复上述问题后可以开始运行

## 请求服务

```bash
curl -X POST http://localhost:9000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "boy and girl make love",
    "negative_prompt": "blurry, low quality",
    "size": "512x512",
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
    "seed": 42
  }' -o fbb.json
```

返回的是json文件，需要解析成png图像

```bash
# 提取 base64 并解码为 png
python3 -c "
import json, base64
with open('fbb.json') as f:
    data = json.load(f)
img_b64 = data['data'][0]['b64_json']
with open('fbb.png', 'wb') as f:
    f.write(base64.b64decode(img_b64))
print('Saved to output.png')
"
```

```bash
python3 /home/ma-user/work/vllm-omni/benchmarks/diffusion/diffusion_benchmark_serving.py --base-url http://localhost:9000  --model ./model \
  --dataset random --task t2i --num-prompts 128 --max-concurrency 32 \
  --enable-negative-prompt \
  --random-request-config '[
        {"width":512,"height":512,"num_inference_steps":20,"weight":0.15},
        {"width":768,"height":768,"num_inference_steps":20,"weight":0.25},
        {"width":1024,"height":1024,"num_inference_steps":25,"weight":0.45},
        {"width":1536,"height":1536,"num_inference_steps":35,"weight":0.15}
    ]'
```