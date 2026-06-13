# RSS 三个 Wan + pi0.5 任务 Gear 训练说明

本文档说明如何在当前共享文件系统环境中，用 `/usr/local/bin/gear` 启动三个相互独立的 8 机 64 H100 GPU 高优先级训练。

默认仓库路径：

```bash
export RLINF_ROOT=/mnt/amlfs-01/home/fangqiz/workspace/RLinf
cd "${RLINF_ROOT}"
```

三个训练配置：

- `wan_tower_pi05_grpo_subtask`
- `wan_battery_pi05_grpo_subtask`
- `wan_bottle_pi05_grpo`

对应 YAML：

- `examples/embodiment/config/wan_tower_pi05_grpo_subtask.yaml`
- `examples/embodiment/config/wan_battery_pi05_grpo_subtask.yaml`
- `examples/embodiment/config/wan_bottle_pi05_grpo.yaml`

## 1. 当前环境约定

本次申请的 8 个节点使用共享文件系统，并且和当前环境拥有完全相同的文件系统。因此：

- 不需要在 Gear 任务里重新下载 RLinf 代码。
- 不需要在 Gear 任务里下载模型或数据集。
- 不使用 `--local-workdir` 或 `--workdir-path` 上传工作区压缩包。
- 直接用 `--workdir "${RLINF_ROOT}"` 指向共享路径。
- `scripts/gear_run_rss_task.sh` 会调用 `scripts/gear_prepare_rss_task.py --skip-model-download`，只校验共享模型目录是否存在。

启动前确认共享模型路径存在：

```bash
cd "${RLINF_ROOT}"
ls models/wan-tower
ls models/wan-battery
ls models/wan-bottle
ls models/pi05_tower-of-hanoi-game_weighted_bc_pytorch
ls models/pi05_insert-mouse-battery_weighted_bc_pytorch
ls models/pi05_seal-water-bottle-cap_weighted_bc_pytorch
```

## 2. W&B 登录

不要把 W&B API key 写入 README、代码、命令行参数或 Git 提交。训练脚本不会读取 `WANDB_API_KEY` 环境变量，也不会硬编码 key。

在共享文件系统的同一个用户环境里登录一次：

```bash
cd "${RLINF_ROOT}"
wandb login --relogin
python -m wandb login --verify
```

`wandb login --relogin` 会交互式提示粘贴 API key，并把登录凭据写入本地用户配置。Gear 训练启动时只运行 `python -m wandb login --verify` 检查登录状态。

如果某个 API key 已经贴到聊天、README、代码或日志里，应在 W&B 后台撤销并重新生成。

## 3. Hugging Face checkpoint 上传

当前 `scripts/gear_run_rss_task.sh` 会为三项任务默认设置 checkpoint 上传仓库：

- Tower: `Shirk6/rlinf-wan-tower-pi05-grpo-subtask-ckpts`
- Battery: `Shirk6/rlinf-wan-battery-pi05-grpo-subtask-ckpts`
- Bottle: `Shirk6/rlinf-wan-bottle-pi05-grpo-ckpts`

因此启动前需要在提交 Gear 的 shell 中提供 Hugging Face token，并通过 Ray runtime env 传给训练任务：

```bash
export HF_TOKEN=<your_huggingface_token>

export RUNTIME_ENV_JSON="$(
python - <<'PY'
import json
import os

token = os.environ["HF_TOKEN"]
print(json.dumps({
    "env_vars": {
        "HF_TOKEN": token,
        "HUGGING_FACE_HUB_TOKEN": token,
        "HUGGINGFACE_HUB_TOKEN": token,
    }
}))
PY
)"
```

不要把 Hugging Face token 写入仓库。

## 4. Gear 参数

先设置 Gear 公共参数。`GEAR_POOL` 必须替换成可用的 H100 资源池名称：

```bash
export GEAR=/usr/local/bin/gear
export GEAR_POOL=<h100_pool_name>
export GEAR_IMAGE=nvcr.io/nvidian/gear-trinity-train:latest
export GEAR_TEMPLATE="${RLINF_ROOT}/gear_workflows/ray_cluster_rlinf.yaml"
export GEAR_LOG_DIR="${RLINF_ROOT}/logs/gear_submit"

mkdir -p "${GEAR_LOG_DIR}"
```

可用资源池可用以下命令查询：

```bash
"${GEAR}" info list
"${GEAR}" info available --pool "${GEAR_POOL}" --count 16
```

## 5. 启动三个独立训练

以下三条命令分别申请独立的 8 节点 Ray 集群。每个节点按 H100 池默认规格提供 8 张 GPU，因此每个任务是 8 机 64 H100 GPU。`--priority high` 保留高优先级。

三个命令用后台进程提交，互不等待：

```bash
cd "${RLINF_ROOT}"

(
  "${GEAR}" ray fast \
    "cd ${RLINF_ROOT} && bash scripts/gear_run_rss_task.sh tower wan_tower_pi05_grpo_subtask" \
    --pool "${GEAR_POOL}" \
    --num-nodes 8 \
    --workflow-name "rlinf-rss-tower-$(date +%Y%m%d-%H%M%S)" \
    --image "${GEAR_IMAGE}" \
    --template "${GEAR_TEMPLATE}" \
    --workdir "${RLINF_ROOT}" \
    --runtime-env-json "${RUNTIME_ENV_JSON}" \
    --priority high
) >"${GEAR_LOG_DIR}/tower.submit.log" 2>&1 &

(
  "${GEAR}" ray fast \
    "cd ${RLINF_ROOT} && bash scripts/gear_run_rss_task.sh battery wan_battery_pi05_grpo_subtask" \
    --pool "${GEAR_POOL}" \
    --num-nodes 8 \
    --workflow-name "rlinf-rss-battery-$(date +%Y%m%d-%H%M%S)" \
    --image "${GEAR_IMAGE}" \
    --template "${GEAR_TEMPLATE}" \
    --workdir "${RLINF_ROOT}" \
    --runtime-env-json "${RUNTIME_ENV_JSON}" \
    --priority high
) >"${GEAR_LOG_DIR}/battery.submit.log" 2>&1 &

(
  "${GEAR}" ray fast \
    "cd ${RLINF_ROOT} && bash scripts/gear_run_rss_task.sh bottle wan_bottle_pi05_grpo" \
    --pool "${GEAR_POOL}" \
    --num-nodes 8 \
    --workflow-name "rlinf-rss-bottle-$(date +%Y%m%d-%H%M%S)" \
    --image "${GEAR_IMAGE}" \
    --template "${GEAR_TEMPLATE}" \
    --workdir "${RLINF_ROOT}" \
    --runtime-env-json "${RUNTIME_ENV_JSON}" \
    --priority high
) >"${GEAR_LOG_DIR}/bottle.submit.log" 2>&1 &

jobs -l
```

这会同时申请三套 8 机 64 H100 GPU 资源，总计 24 节点、192 张 H100。不要把三个任务提交到同一个 Ray 集群里；这些配置的 `cluster.component_placement` 都使用 `actor,env,rollout: all`，单个任务会占满整套 64 卡。

## 6. 查看状态和日志

本地提交日志：

```bash
tail -f "${GEAR_LOG_DIR}/tower.submit.log"
tail -f "${GEAR_LOG_DIR}/battery.submit.log"
tail -f "${GEAR_LOG_DIR}/bottle.submit.log"
```

Gear 工作流状态：

```bash
"${GEAR}" info list --pool "${GEAR_POOL}"
"${GEAR}" ray ls-clusters --pool "${GEAR_POOL}"
```

如果要查看某个 Ray workflow 的日志，使用 Gear 输出里的 workflow 名称：

```bash
"${GEAR}" ray logs <workflow_name>
"${GEAR}" ray tail <workflow_name>
```

## 7. 脚本行为

`scripts/gear_run_rss_task.sh` 每个任务内部会执行：

1. 切到 OpenPI 环境。
2. 设置 `REPO_PATH`、`EMBODIED_PATH`、`MUJOCO_GL=egl`、`PYOPENGL_PLATFORM=egl`、`ROBOT_PLATFORM=LIBERO`。
3. 验证 W&B 已登录，但不读取 W&B API key 环境变量。
4. 等待 8 个 Ray 节点加入。
5. 在所有节点校验共享模型目录。
6. 启动 Hugging Face checkpoint watcher。
7. 等待 Ray 暴露 64 张 GPU。
8. 启动：

```bash
bash examples/embodiment/run_embodiment.sh <config_name> LIBERO cluster.num_nodes=8
```

## 8. 常见问题

- `W&B is not logged in`：在共享文件系统同一用户下运行 `wandb login --relogin`，再运行 `python -m wandb login --verify`。
- `Missing required model paths`：共享 `models/` 目录不完整，先在当前环境补齐路径。
- `bash ray_utils/check_ray.sh 64` 超时：检查 Gear 是否实际分配了 8 个 8-GPU 节点，以及 Ray worker 是否全部加入 head。
- 只有 8 卡被使用：确认训练命令包含 `cluster.num_nodes=8`。`scripts/gear_run_rss_task.sh` 已默认加上。
- 三个任务互相抢资源：说明它们被提交到同一个 Ray 集群或资源池没有足够节点。本文档的三条 `gear ray fast` 命令会申请三套独立集群。
