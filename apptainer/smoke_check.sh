#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/tmp/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}"

source /opt/venv/openpi/bin/activate
python -c 'import torch, diffsynth, openpi; print("openpi ok", torch.__version__, "cuda", torch.cuda.is_available(), "devices", torch.cuda.device_count())'

source /opt/venv/openvla-oft/bin/activate
python -c 'import torch, diffsynth; print("openvla-oft ok", torch.__version__, "cuda", torch.cuda.is_available(), "devices", torch.cuda.device_count())'

ldconfig -p | grep -E 'libOSMesa|libEGL|libGLX|libavcodec|libibverbs|libvulkan' | head -20
