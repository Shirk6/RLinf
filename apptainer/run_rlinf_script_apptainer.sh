#!/usr/bin/env bash
set -euo pipefail

ACCOUNT="${ACCOUNT:-peilab}"
PARTITION="${PARTITION:-normal}"
QOS="${QOS:-normal_debug_qos}"
GPUS="${GPUS:-2}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEM="${MEM:-128G}"
TIME="${TIME:-01:00:00}"
JOB_NAME="${JOB_NAME:-rlinf-wan-openpi-debug}"

SRK_ROOT="${SRK_ROOT:-/project/peilab/srk}"
RLINF_ROOT="${RLINF_ROOT:-/project/peilab/srk/rss_2026_ws/RLinf}"
SIF_IMAGE="${SIF_IMAGE:-${RLINF_APPTAINER_IMAGE:-${SRK_ROOT}/.cache/enroot/rlinf-embodied-wan-openpi.sif}}"
VENV_NAME="${RLINF_VENV:-openpi}"
RLINF_SCRIPT="${RLINF_SCRIPT:-${RLINF_ROOT}/examples/embodiment/run_embodiment.sh}"
RLINF_ARGS="${RLINF_ARGS:-wan_tower_pi05_grpo}"
LOG_SUBMIT_DIR="${LOG_SUBMIT_DIR:-${RLINF_ROOT}/apptainer/slurm_logs}"
RLINF_JOB_TMP="${RLINF_JOB_TMP:-/tmp/rli-${SLURM_JOB_ID:-manual}}"
TMPDIR="${TMPDIR:-${RLINF_JOB_TMP}/tmp}"
RAY_TMPDIR="${RAY_TMPDIR:-${RLINF_JOB_TMP}/ray}"
RAY_TMP_DIR="${RAY_TMP_DIR:-${RAY_TMPDIR}}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${RLINF_JOB_TMP}/triton-cache}"
WANDB_DIR="${WANDB_DIR:-${RLINF_JOB_TMP}/wandb}"

if [[ ! -f "${SIF_IMAGE}" ]]; then
    echo "ERROR: SIF image not found: ${SIF_IMAGE}" >&2
    echo "Build it with: cd ${RLINF_ROOT} && bash apptainer/build_embodied_wan_openpi.sh" >&2
    exit 2
fi

mkdir -p "${LOG_SUBMIT_DIR}"

echo "Starting RLinf script through Apptainer"
echo "  account=${ACCOUNT} partition=${PARTITION} qos=${QOS} gpus=${GPUS}"
echo "  sif=${SIF_IMAGE}"
echo "  repo=${RLINF_ROOT}"
echo "  venv=${VENV_NAME}"
echo "  script=${RLINF_SCRIPT}"
echo "  args=${RLINF_ARGS}"
echo "  job_tmp=${RLINF_JOB_TMP}"

exec srun \
    --account="${ACCOUNT}" \
    --partition="${PARTITION}" \
    --qos="${QOS}" \
    --job-name="${JOB_NAME}" \
    --nodes=1 \
    --gpus="${GPUS}" \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --mem="${MEM}" \
    --time="${TIME}" \
    bash -lc "module load apptainer && apptainer exec --nv --bind '${SRK_ROOT}:${SRK_ROOT}' --bind '${RLINF_ROOT}:/workspace/RLinf' '${SIF_IMAGE}' bash -lc '
        set -euo pipefail
        set +u
        if command -v switch_env >/dev/null 2>&1; then
            source switch_env \"${VENV_NAME}\"
        elif [[ -f /usr/local/bin/switch_env ]]; then
            source /usr/local/bin/switch_env \"${VENV_NAME}\"
        elif [[ -f /opt/venv/${VENV_NAME}/bin/activate ]]; then
            source /opt/venv/${VENV_NAME}/bin/activate
        fi
        set -u

        export SRK_ROOT=\"${SRK_ROOT}\"
        export RLINF_ROOT=\"/workspace/RLinf\"
        export REPO_PATH=\"/workspace/RLinf\"
        export EMBODIED_PATH=\"/workspace/RLinf/examples/embodiment\"
        export VENV_DIR=\"/opt/venv/${VENV_NAME}\"
        export PYTHONPATH=\"/workspace/RLinf:\${PYTHONPATH:-}\"
        export HF_HOME=\"${HF_HOME:-${SRK_ROOT}/models/.hf-cache}\"
        export TRANSFORMERS_CACHE=\"${TRANSFORMERS_CACHE:-${SRK_ROOT}/models/.hf-cache/hub}\"
        export TRITON_CACHE_DIR=\"${TRITON_CACHE_DIR}\"
        export WANDB_DIR=\"${WANDB_DIR}\"
        export RLINF_JOB_TMP=\"${RLINF_JOB_TMP}\"
        export TMPDIR=\"${TMPDIR}\"
        export RAY_TMPDIR=\"${RAY_TMPDIR}\"
        export RAY_TMP_DIR=\"${RAY_TMP_DIR}\"
        export MUJOCO_GL=\"${MUJOCO_GL:-osmesa}\"
        export PYOPENGL_PLATFORM=\"${PYOPENGL_PLATFORM:-osmesa}\"
        export RLINF_DISABLE_CUDA_IPC=\"${RLINF_DISABLE_CUDA_IPC:-1}\"
        export ROBOT_PLATFORM=\"${ROBOT_PLATFORM:-LIBERO}\"
        export PRETRAINED_DIR=\"${PRETRAINED_DIR:-${SRK_ROOT}/models}\"
        export VLA_MODEL_PATH=\"${VLA_MODEL_PATH:-${SRK_ROOT}/models/Openvla-oft-SFT-libero-spatial-traj1}\"
        export BC_MODEL_PATH=\"${BC_MODEL_PATH:-${SRK_ROOT}/models/Openvla-oft-SFT-libero-spatial-traj1}\"
        export WAN_MODEL_PATH=\"${WAN_MODEL_PATH:-${SRK_ROOT}/models/RLinf-Wan-LIBERO-Spatial}\"
        export WAN_MODEL_DIT_PATH=\"${WAN_MODEL_DIT_PATH:-${SRK_ROOT}/models/RLinf-Wan-LIBERO-Spatial/model-00001.safetensors}\"

        mkdir -p \"\${HF_HOME}\" \"\${TRANSFORMERS_CACHE}\" \"\${TRITON_CACHE_DIR}\" \"\${WANDB_DIR}\" \"\${TMPDIR}\" \"\${RAY_TMPDIR}\"
        cd /workspace/RLinf
        exec bash \"${RLINF_SCRIPT}\" ${RLINF_ARGS}
    '"
