#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <tower|battery|bottle> <config_name>" >&2
    exit 2
fi

TASK="$1"
CONFIG_NAME="$2"

case "${TASK}" in
    tower)
        HF_REPO_ID="${HF_REPO_ID:-Shirk6/rlinf-wan-tower-pi05-grpo-subtask-ckpts}"
        ;;
    battery)
        HF_REPO_ID="${HF_REPO_ID:-Shirk6/rlinf-wan-battery-pi05-grpo-subtask-ckpts}"
        ;;
    bottle)
        HF_REPO_ID="${HF_REPO_ID:-Shirk6/rlinf-wan-bottle-pi05-grpo-ckpts}"
        ;;
    *)
        echo "Unknown task: ${TASK}" >&2
        exit 2
        ;;
esac

if [ -f /opt/venv/openpi/bin/activate ]; then
    source /opt/venv/openpi/bin/activate
elif command -v switch_env >/dev/null 2>&1; then
    source switch_env openpi
fi

export REPO_PATH="$(pwd)"
export EMBODIED_PATH="${REPO_PATH}/examples/embodiment"
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export ROBOT_PLATFORM="${ROBOT_PLATFORM:-LIBERO}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN}}"
export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN}}"

if [ -z "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN, HUGGING_FACE_HUB_TOKEN, or HUGGINGFACE_HUB_TOKEN is required for checkpoint upload." >&2
    exit 2
fi

if ! python -m wandb login --verify >/dev/null 2>&1; then
    echo "W&B is not logged in. Run 'wandb login --relogin' once on the shared filesystem before launching." >&2
    exit 2
fi

python scripts/gear_prepare_rss_task.py \
    --task "${TASK}" \
    --expected-nodes 8 \
    --node-wait-timeout-seconds "${RSS_PREPARE_NODE_WAIT_TIMEOUT_SECONDS:-1800}" \
    --node-wait-poll-seconds "${RSS_PREPARE_NODE_WAIT_POLL_SECONDS:-10}" \
    --hf-repo-id "${HF_REPO_ID}" \
    --experiment-name "${CONFIG_NAME}" \
    --node-delay-seconds "${RSS_PREPARE_NODE_DELAY_SECONDS:-120}" \
    --max-concurrent-nodes "${RSS_PREPARE_MAX_CONCURRENT_NODES:-1}" \
    --skip-model-download
timeout "${RAY_GPU_WAIT_TIMEOUT:-1800}" bash ray_utils/check_ray.sh 64
bash examples/embodiment/run_embodiment.sh \
    "${CONFIG_NAME}" \
    "${ROBOT_PLATFORM}" \
    cluster.num_nodes=8
