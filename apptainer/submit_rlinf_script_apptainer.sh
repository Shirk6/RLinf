#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ACCOUNT="${ACCOUNT:-peilab}"
PARTITION="${PARTITION:-normal}"
QOS="${QOS:-normal_qos}"
GPUS="${GPUS:-2}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEM="${MEM:-128G}"
TIME="${TIME:-12:00:00}"
JOB_NAME="${JOB_NAME:-rlinf-wan-openpi}"

SRK_ROOT="${SRK_ROOT:-/project/peilab/srk}"
RLINF_ROOT="${RLINF_ROOT:-/project/peilab/srk/rss_2026_ws/RLinf}"
LOG_SUBMIT_DIR="${LOG_SUBMIT_DIR:-${RLINF_ROOT}/apptainer/slurm_logs}"
mkdir -p "${LOG_SUBMIT_DIR}"

sbatch \
    --account="${ACCOUNT}" \
    --partition="${PARTITION}" \
    --qos="${QOS}" \
    --job-name="${JOB_NAME}" \
    --nodes=1 \
    --gpus="${GPUS}" \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --mem="${MEM}" \
    --time="${TIME}" \
    --output="${LOG_SUBMIT_DIR}/%x-%j.out" \
    --error="${LOG_SUBMIT_DIR}/%x-%j.err" \
    --export=ALL \
    "${SCRIPT_DIR}/run_rlinf_script_apptainer.sh"
