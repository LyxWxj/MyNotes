---
type: Note
related_to: "[[vllm-omni]]"
status: Active
url: https://github.com/vllm-project/vllm-omni/blob/main/docs/features/comfyui.md
---

# ComfyUI Integration

vLLM-Omni提供基于其在线服务API的ComfyUI集成，可以向本地或远程运行的vLLM-Omni服务发送模型推理请求。

## 要求

- Python 3.12或更高版本
- [ComfyUI已安装](https://docs.comfy.org/installation/system_requirements)
- [vLLM-Omni已安装](https://docs.vllm.ai/projects/vllm-omni/en/latest/getting_started/installation/)（同一设备或可通过互联网发现的另一设备）
- 无需安装除ComfyUI已要求的包之外的额外包

> **提示**：如果在同一设备上运行ComfyUI和vLLM-Omni，可以创建单独的虚拟环境并使用不同的Python版本。

## 安装

1. 将`apps/ComfyUI-vLLM-Omni`文件夹复制到ComfyUI安装的`custom_nodes`子文件夹
2. 目录结构应为：`ComfyUI/custom_nodes/ComfyUI-vLLM-Omni`
3. 如果复制时ComfyUI正在运行，需要重启ComfyUI以加载此扩展

### 启动服务

**ComfyUI端**：
```bash
cd ComfyUI

# 常规方式
python main.py

# 如果主要使用此节点，更快启动
python main.py --cpu
```

**vLLM-Omni端**：
```bash
vllm serve The_Model_ID_to_Serve --omni --port 8000
```

检查**ComfyUI侧边栏 -> Node Library**，应有一个名为**vLLM-Omni**的新文件夹。

## 快速开始

此扩展提供以下节点：

| 节点 | 用途 |
|------|------|
| **Generate Image** | 文本到图像和图像到图像任务 |
| **Generate Video** | 文本到视频和图像到视频任务 |
| **Multimodality Understanding** | 多模态到文本和多模态到音频任务 |
| **TTS** 和 **TTS Voice Clone** | TTS任务 |

此扩展还提供示例工作流（在**ComfyUI侧边栏 -> Templates -> vLLM-Omni**）

> **注意**：节点UI和功能设计旨在匹配vLLM-Omni在线服务接口，无法提供超出接口支持的功能。

## 构建简单工作流

1. 将生成节点拖到画布上
2. 根据需要获取内置多媒体文件加载器节点：
   - **image->Load Image**
   - **image->video->Load Video**
   - **audio->Load Audio**
3. 根据需要获取内置多媒体文件预览节点：
   - **image->Preview Image**
   - **image->video->Save Video**
   - **audio->Preview Audio**
   - **utils->Preview as Text**
4. 如果要调整采样参数，从**vLLM-Omni-> Sampling Params**获取相应节点：
   - 对于多阶段模型，可以连接多个**AR Sampling Params**和**Diffusion Sampling Params**节点到**Multi-Stage Sampling Params List**节点
   - 对于某些多阶段模型（如BAGEL），[仅一个阶段的采样参数通过vLLM-Omni的在线服务API暴露和可调](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/examples/online_serving/bagel/)
   - 对于所有阶段都是自回归或扩散的多阶段模型，也可以只连接单个Sampling Params节点，表示此采样参数集将用于所有阶段

## 相关链接

- [ComfyUI集成README](https://github.com/vllm-project/vllm-omni/tree/main/apps/ComfyUI-vLLM-Omni)
