#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRK_ROOT="${SRK_ROOT:-/project/peilab/srk}"

IMAGE_PATH="${RLINF_APPTAINER_IMAGE:-${SRK_ROOT}/.cache/enroot/rlinf-embodied-wan-openpi.sif}"
DEF_FILE="${RLINF_APPTAINER_DEF:-${SCRIPT_DIR}/rlinf-embodied-wan-openpi.def}"
BUILD_FLAGS="${RLINF_APPTAINER_BUILD_FLAGS:-}"

cd "${REPO_ROOT}"

if ! command -v apptainer >/dev/null 2>&1; then
    echo "apptainer is not available on PATH." >&2
    exit 1
fi

mkdir -p "$(dirname "${IMAGE_PATH}")"

# Use RLINF_APPTAINER_BUILD_FLAGS for site-specific options such as:
#   export RLINF_APPTAINER_BUILD_FLAGS="--fakeroot"
#   export RLINF_APPTAINER_BUILD_FLAGS="--force"
apptainer build ${BUILD_FLAGS} "${IMAGE_PATH}" "${DEF_FILE}"

echo "Built ${IMAGE_PATH}"
