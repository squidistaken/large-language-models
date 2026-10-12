#!/bin/bash

set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: run_job.sh PROFILE UV_ENVIRONMENT HYDRA_OVERRIDE [HYDRA_OVERRIDE ...]" >&2
  exit 2
fi

PROFILE="$1"
UV_ENVIRONMENT="$2"
shift 2

PROJECT_ROOT="$(pwd -P)"

source "$PROJECT_ROOT/slurm/env.sh"

export UV_PROJECT_ENVIRONMENT="$PROJECT_ROOT/$UV_ENVIRONMENT"

if [ ! -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]; then
  echo "ERROR: uv environment is missing: $UV_PROJECT_ENVIRONMENT" >&2
  exit 1
fi

if [ -f "$PROJECT_ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/.env"
  set +a
fi

if [ -z "${HF_HOME:-}" ] && [ -d "/scratch/$USER" ]; then
  export HF_HOME="/scratch/$USER/hf-cache"
fi

JOB_TMP_ROOT="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
export TMPDIR="$JOB_TMP_ROOT/xed-${SLURM_JOB_ID:-manual}"
mkdir -p "$TMPDIR"

echo "Profile: $PROFILE"
echo "uv environment: $UV_ENVIRONMENT"
echo "Overrides: $*"
echo "HF_HOME: ${HF_HOME:-default}"
echo "WANDB_MODE: ${WANDB_MODE:-disabled}"
echo "Job: ${SLURM_JOB_ID:-not-running-under-slurm}"
echo "Node: $(hostname)"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
echo "Start: $(date --iso-8601=seconds)"

# CPU-only profiles (gpu_type: none) skip the GPU check.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
uv run --frozen --no-sync python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("The Slurm profile requested a GPU, but PyTorch cannot use it")

result = (torch.ones((32, 32), device="cuda") @ torch.ones((32, 32), device="cuda")).sum()
torch.cuda.synchronize()

print(f"PyTorch: {torch.__version__}, CUDA {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)} {torch.cuda.get_device_capability(0)}")
print(f"CUDA calculation result: {result.item():.0f}")
PY
fi

srun uv run --frozen --no-sync python main.py "$@"

echo "End: $(date --iso-8601=seconds)"
