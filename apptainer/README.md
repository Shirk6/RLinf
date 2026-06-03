# RLinf WAN on HKUST Superpod

This directory contains Apptainer wrappers for running the local RLinf WAN
environment without sudo. The launch path mirrors `/project/peilab/srk/superpod_wan`:
`srun`/`sbatch` loads the Apptainer module, binds the project paths, activates a
venv inside the SIF, then runs the RLinf script.

The image contains both:

- `openvla-oft + wan`
- `openpi + wan`

The default runtime environment is `openpi`.

## Files

- `rlinf-embodied-wan-openpi.def`: builds the SIF with both Wan venvs.
- `build_embodied_wan_openpi.sh`: builds the SIF image.
- `run_rlinf_script_apptainer.sh`: starts an `srun` job and runs an RLinf script inside Apptainer.
- `submit_rlinf_script_apptainer.sh`: submits the same script path through `sbatch`.
- `run_rlinf.sh`: direct Apptainer shell/exec helper for non-Slurm debugging.

## Defaults

- SIF image: `/project/peilab/srk/.cache/enroot/rlinf-embodied-wan-openpi.sif`
- Repo: `/project/peilab/srk/rss_2026_ws/RLinf`
- Models: `/project/peilab/srk/models`
- VLA: `/project/peilab/srk/models/Openvla-oft-SFT-libero-spatial-traj1`
- WAN: `/project/peilab/srk/models/RLinf-Wan-LIBERO-Spatial`
- Slurm logs: `/project/peilab/srk/rss_2026_ws/RLinf/apptainer/slurm_logs`

The launcher sets `RLINF_DISABLE_CUDA_IPC=1` by default because unprivileged
Apptainer can block PyTorch CUDA IPC with `pidfd_getfd: Operation not permitted`.

## 1. Build the SIF

```bash
cd /project/peilab/srk/rss_2026_ws/RLinf
module load apptainer
bash apptainer/build_embodied_wan_openpi.sh
```

If your cluster requires fakeroot:

```bash
RLINF_APPTAINER_BUILD_FLAGS="--fakeroot" bash apptainer/build_embodied_wan_openpi.sh
```

## 2. Run a short interactive Slurm job

Default command:

```bash
cd /project/peilab/srk/rss_2026_ws/RLinf
TIME=00:45:00 \
GPUS=2 \
RLINF_VENV=openpi \
RLINF_SCRIPT=/project/peilab/srk/rss_2026_ws/RLinf/examples/embodiment/run_embodiment.sh \
RLINF_ARGS='wan_tower_pi05_grpo' \
bash apptainer/run_rlinf_script_apptainer.sh
```

Run the OpenVLA-OFT Wan config instead:

```bash
RLINF_VENV=openvla-oft \
RLINF_ARGS='wan_libero_spatial_grpo_openvlaoft' \
bash apptainer/run_rlinf_script_apptainer.sh
```

Useful overrides:

```bash
SIF_IMAGE=/path/to/rlinf-embodied-wan-openpi.sif
VLA_MODEL_PATH=/path/to/vla
WAN_MODEL_PATH=/path/to/wan
WAN_MODEL_DIT_PATH=/path/to/wan/model-00001.safetensors
HF_HOME=/path/to/hf-cache
```

## 3. Submit a batch job

```bash
cd /project/peilab/srk/rss_2026_ws/RLinf
RLINF_VENV=openpi \
RLINF_ARGS='wan_tower_pi05_grpo' \
TIME=12:00:00 \
bash apptainer/submit_rlinf_script_apptainer.sh
```

Watch logs:

```bash
tail -f /project/peilab/srk/rss_2026_ws/RLinf/apptainer/slurm_logs/rlinf-wan-openpi-<jobid>.out
```

## 4. Direct Apptainer Debugging

For a shell without Slurm:

```bash
bash apptainer/run_rlinf.sh
```

For a direct command:

```bash
bash apptainer/run_rlinf.sh bash examples/embodiment/run_embodiment.sh wan_tower_pi05_grpo
```
