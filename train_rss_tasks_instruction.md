# RSS 三个 Wan + pi0.5 任务训练说明

本文档说明如何在全新机器/集群上准备 RLinf 环境，下载 RSS 三个任务所需模型，并用 8 机 64 卡启动训练。以下命令默认在 RLinf 仓库根目录执行：

```bash
cd /path/to/RLinf
```

三个训练配置：

- `examples/embodiment/config/wan_tower_pi05_grpo_subtask.yaml`
- `examples/embodiment/config/wan_battery_pi05_grpo_subtask.yaml`
- `examples/embodiment/config/wan_bottle_pi05_grpo.yaml`

三个任务的配置路径已统一改为相对路径，模型默认放在 `RLinf/models/` 下。

## 1. 机器与系统要求

建议 8 台同构 GPU 节点，每台 8 卡。每台节点需要满足：

- Ubuntu 22.04 或兼容 Linux 环境
- NVIDIA driver、CUDA 和 NCCL 可用
- 节点之间网络互通，head 节点的 Ray 端口能被 worker 访问
- 共享文件系统，或确保每台节点上的 RLinf 代码和 `models/` 目录路径一致
- Python/Ray/RLinf 依赖版本一致

常用系统工具：

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs curl wget tmux build-essential

git lfs install
```

如果集群不能使用 `sudo`，请让管理员预装上述工具；Python 依赖可继续用用户目录安装。

## 2. 获取代码

```bash
git clone https://github.com/RLinf/RLinf.git
cd RLinf
```

如果是在已有工作区使用本文档，直接进入现有仓库即可。

## 3. 安装 Python 环境

### 方案 A：使用官方/预构建容器

具身任务推荐使用容器，因为 Wan/OpenPI 依赖较重。进入容器后切换到 OpenPI 环境：

```bash
source switch_env openpi
cd /path/to/RLinf
```

如果使用 Apptainer，需要把代码和模型所在的共享目录 bind 进容器，并保证 8 个节点看到的路径一致。

### 方案 B：本地 UV 环境

在每台节点上执行，或在共享文件系统上安装一次并确保所有节点可访问：

```bash
cd /path/to/RLinf
bash requirements/install.sh embodied --model openpi --env wan --venv .venv
source .venv/bin/activate
```

说明：`--env wan` 会下载 `https://github.com/Shirk6/diffsynth-studio-rlinf.git` 的 `downsample` 分支，并以 editable 方式安装到当前 venv。

国内网络可加 `--use-mirror`：

```bash
bash requirements/install.sh embodied --model openpi --env wan --venv .venv --use-mirror
```

安装完成后确认 Ray 和 CUDA 可用：

```bash
python -c "import torch, ray; print(torch.__version__, torch.cuda.device_count(), ray.__version__)"
```

## 4. 下载模型到相对路径

所有模型下载到 `RLinf/models/`。如果各节点共享同一文件系统，只需在共享目录下载一次；否则每台节点都要下载到同样的相对路径。

可选：国内网络设置 Hugging Face 镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

安装下载工具：

```bash
python -m pip install -U huggingface_hub
```

下载 3 个 Wan 世界模型和 3 个 pi0.5 初始策略：

```bash
cd /path/to/RLinf
mkdir -p models

hf download Shirk6/wan-tower \
  --repo-type model \
  --local-dir models/wan-tower

hf download Shirk6/wan-battery \
  --repo-type model \
  --local-dir models/wan-battery

hf download Shirk6/wan-bottle \
  --repo-type model \
  --local-dir models/wan-bottle

hf download Shirk6/pi05_tower-of-hanoi-game_weighted_bc_pytorch \
  --repo-type model \
  --local-dir models/pi05_tower-of-hanoi-game_weighted_bc_pytorch

hf download Shirk6/pi05_insert-mouse-battery_weighted_bc_pytorch \
  --repo-type model \
  --local-dir models/pi05_insert-mouse-battery_weighted_bc_pytorch

hf download Shirk6/pi05_seal-water-bottle-cap_weighted_bc_pytorch \
  --repo-type model \
  --local-dir models/pi05_seal-water-bottle-cap_weighted_bc_pytorch
```

如果 `hf download` 失败，也可以用 Git LFS：

```bash
cd /path/to/RLinf
mkdir -p models
cd models

git clone https://huggingface.co/Shirk6/wan-tower
git clone https://huggingface.co/Shirk6/wan-battery
git clone https://huggingface.co/Shirk6/wan-bottle
git clone https://huggingface.co/Shirk6/pi05_tower-of-hanoi-game_weighted_bc_pytorch
git clone https://huggingface.co/Shirk6/pi05_insert-mouse-battery_weighted_bc_pytorch
git clone https://huggingface.co/Shirk6/pi05_seal-water-bottle-cap_weighted_bc_pytorch
```

下载后检查关键文件/目录：

```bash
ls models/wan-tower
ls models/wan-battery
ls models/wan-bottle
ls models/pi05_tower-of-hanoi-game_weighted_bc_pytorch
ls models/pi05_insert-mouse-battery_weighted_bc_pytorch
ls models/pi05_seal-water-bottle-cap_weighted_bc_pytorch
```

## 5. 配置路径约定

三个 YAML 使用从 `RLinf/` 仓库根目录出发的相对路径：

```yaml
env:
  train:
    wan_wm_hf_ckpt_path: models/wan-xxx

rollout:
  model:
    model_path: models/pi05_xxx

actor:
  model:
    model_path: models/pi05_xxx
```

因此启动训练前必须先 `cd /path/to/RLinf`。不要从上级目录或其他工作目录直接运行训练脚本，否则相对路径会解析错误。

## 6. 启动 8 机 64 卡 Ray 集群

Ray 会继承执行 `ray start` 时的 Python 环境和环境变量，所以每个节点都必须先进入同一个容器或激活同一个 venv，再启动 Ray。

### 方式 A：手动启动

在所有节点先进入环境：

```bash
cd /path/to/RLinf
source .venv/bin/activate  # 如果使用容器内 openpi 环境，则改为 source switch_env openpi
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

head 节点：

```bash
export RLINF_NODE_RANK=0
ray start --head --port=6379 --node-ip-address=<head_ip>
```

worker 节点，rank 从 1 到 7：

```bash
export RLINF_NODE_RANK=<1..7>
ray start --address=<head_ip>:6379
```

检查总 GPU 数：

```bash
ray status
bash ray_utils/check_ray.sh 64
```

### 方式 B：使用仓库脚本

适用于共享文件系统，脚本会通过 `ray_utils/ray_head_ip.txt` 传递 head IP。

```bash
cd /path/to/RLinf
rm -f ray_utils/ray_head_ip.txt
```

8 个节点分别执行：

```bash
# head
export RLINF_NODE_RANK=0
RANK=0 bash ray_utils/start_ray.sh

# worker 1
export RLINF_NODE_RANK=1
RANK=1 bash ray_utils/start_ray.sh

# ...

# worker 7
export RLINF_NODE_RANK=7
RANK=7 bash ray_utils/start_ray.sh
```

在任意已加入 Ray 的节点检查：

```bash
bash ray_utils/check_ray.sh 64
```

## 7. 启动训练

下面命令只在 head 节点执行一次。`cluster.num_nodes=8` 必须保留，否则配置默认只按单节点调度。

### Tower

```bash
cd /path/to/RLinf
bash examples/embodiment/run_embodiment.sh \
  wan_tower_pi05_grpo_subtask \
  cluster.num_nodes=8
```

### Battery

```bash
cd /path/to/RLinf
bash examples/embodiment/run_embodiment.sh \
  wan_battery_pi05_grpo_subtask \
  cluster.num_nodes=8
```

### Bottle

```bash
cd /path/to/RLinf
bash examples/embodiment/run_embodiment.sh \
  wan_bottle_pi05_grpo \
  cluster.num_nodes=8
```

这三个配置的 `cluster.component_placement` 都是：

```yaml
actor,env,rollout: all
```

也就是每个任务都会使用整个 8 机 64 卡 Ray 集群。三项任务应顺序运行；如果要同时跑三项任务，需要分别申请三套 8 机 64 卡资源，或者改 `component_placement` 和 batch/env 配置做资源切分。

## 8. 停止与重启

训练结束或需要重建环境时，在每个节点执行：

```bash
ray stop
```

如果使用 `ray_utils/start_ray.sh`，重启前清理 head IP 文件：

```bash
rm -f ray_utils/ray_head_ip.txt
```

修改 Python 环境、模型路径、`RLINF_NODE_RANK` 或 Ray 端口后，都应先 `ray stop` 再重新启动 Ray。

## 9. 常见问题

- `ray status` 只有部分节点：检查 head IP、端口、防火墙，以及 worker 是否连到了同一个 head。
- worker import 到旧代码：Ray 在 `ray start` 时冻结环境；更新代码或依赖后需要所有节点 `ray stop` 并重启。
- 找不到模型路径：确认训练命令是在 `RLinf/` 仓库根目录执行，且 `models/` 是当前目录下的子目录。
- 只有 8 卡被使用：检查启动训练时是否加了 `cluster.num_nodes=8`，以及 `bash ray_utils/check_ray.sh 64` 是否通过。
- 三个任务同时互相抢资源：当前配置每个任务都用 `all`，同一套 64 卡上应一次只跑一个任务。
