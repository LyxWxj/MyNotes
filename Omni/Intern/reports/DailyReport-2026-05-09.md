# 娄雨轩 Daily Report — 2026-05-09

## 华为云 ModelArts 远程开发环境配置（前两日总结）

### 问题回顾

前两日在华为云 ModelArts 平台配置远程开发环境，遇到多个问题：

1. **节点选择**：主流节点（华北-北京四、华北-乌兰察布一、华南-广州等）均无合适镜像，仅 **西南-贵阳一** 节点有满足需求的镜像。
2. **SSH 连接**：最初 SSH 直连被拒绝；切换到 VS Code ModelArts 插件后发现插件不支持贵阳节点；改用华南-广州节点后又遇 `PublicKey Error`，需修正本地和远程 SSH 文件权限为 600/700。
3. **VS Code Server 先决条件不满足**：旧镜像基于 Ubuntu 18.04，glibc 版本过低，无法运行新版 VS Code Server。
4. **Docker 镜像不兼容**：升级到 Ubuntu 22.04 镜像后，华为云平台 Docker 版本与镜像内核不匹配，容器无法启动。
5. **patchelf 方案不可行**：尝试用 `patchelf` 手动指定 glibc 动态库路径，但 `glibc-all-in-one` 仓库中 **不包含 aarch64 架构版本的 glibc**，而 ModelArts 服务器为 ARM 架构，该方案无法实施。

### 最终解决方案

选择 **西南-贵阳一** 节点的 `Pytorch2.7.1-cann8.3.rc1-py3.11-hce2.0.2509-aarch64-snt9b` 镜像。该镜像的系统 glibc 版本已满足 VS Code Server 的要求，可直接通过 VS Code Remote SSH 连接。

## vLLM + vLLM-Ascend 源码编译安装

### 背景

需要在昇腾 NPU 环境上部署 vLLM 推理框架，但该环境无法通过 `pip install vllm` 直接安装，需从源码编译。

### 安装步骤

#### Step 1：从源码安装 vLLM

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm && VLLM_TARGET_DEVICE=empty pip install .
```

`VLLM_TARGET_DEVICE=empty` 跳过特定硬件后端的编译，仅安装 vLLM 核心框架。

#### Step 2：升级 PyTorch 版本

镜像预装的 `torch 2.7.1` 和 `torch-npu 2.7.1` 版本不满足 vllm-ascend 的依赖要求，需先升级到 2.9.0：

```bash
pip install torch==2.9.0 torch-npu==2.9.0
```

#### Step 3：安装 vLLM-Ascend

vllm-ascend 不能以 vLLM 相同的方式（`VLLM_TARGET_DEVICE=empty pip install .`）从源码安装，需直接通过 pip 安装：

```bash
pip install vllm-ascend
```

#### Step 4：环境配置与 ModelScope 安装

```bash
export VLLM_USE_MODELSCOPE=true && pip install modelscope==1.22.0
```

`VLLM_USE_MODELSCOPE=true` 让 vLLM 从 ModelScope 而非 HuggingFace 下载模型，适配国内网络环境。



