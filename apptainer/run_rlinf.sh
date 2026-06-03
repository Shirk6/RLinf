#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRK_ROOT="${SRK_ROOT:-/project/peilab/srk}"

IMAGE_PATH="${RLINF_APPTAINER_IMAGE:-${SRK_ROOT}/.cache/enroot/rlinf-embodied-wan-openpi.sif}"
VENV_NAME="${RLINF_VENV:-openpi}"
CACHE_DIR="${RLINF_APPTAINER_CACHE:-${REPO_ROOT}/.apptainer/cache}"
TMP_DIR="${RLINF_APPTAINER_TMP:-${REPO_ROOT}/.apptainer/tmp}"

if ! command -v apptainer >/dev/null 2>&1; then
    echo "apptainer is not available on PATH." >&2
    exit 1
fi

if [ ! -f "${IMAGE_PATH}" ]; then
    echo "Apptainer image not found: ${IMAGE_PATH}" >&2
    echo "Build it with: ${SCRIPT_DIR}/build_embodied_wan_openpi.sh" >&2
    exit 1
fi

mkdir -p "${CACHE_DIR}" "${TMP_DIR}" "${REPO_ROOT}/logs"

APPTAINER_ARGS=(
    --nv
    --pwd /workspace/RLinf
    --bind "${REPO_ROOT}:/workspace/RLinf"
    --bind "${CACHE_DIR}:/opt/.cache"
    --bind "${TMP_DIR}:/tmp"
)

if [ -n "${RLINF_APPTAINER_BIND:-}" ]; then
    APPTAINER_ARGS+=(--bind "${RLINF_APPTAINER_BIND}")
fi

if [ "$#" -eq 0 ]; then
    exec apptainer shell "${APPTAINER_ARGS[@]}" "${IMAGE_PATH}"
fi

exec apptainer exec "${APPTAINER_ARGS[@]}" "${IMAGE_PATH}" \
    bash -lc "source /opt/venv/${VENV_NAME}/bin/activate && cd /workspace/RLinf && exec \"\$@\"" bash "$@"
