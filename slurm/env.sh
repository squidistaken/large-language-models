#!/bin/bash

PYTHON_MODULE="${PYTHON_MODULE:-Python/3.13.5-GCCcore-14.3.0}"
UV_MODULE="${UV_MODULE:-uv/0.10.7}"

if command -v module >/dev/null 2>&1; then
  module purge
  module load "$PYTHON_MODULE"
  module load "$UV_MODULE"
fi

use_login_environment() {
  local root="$1"

  if [ -x "$root/.venv/bin/python" ]; then
    export UV_PROJECT_ENVIRONMENT="$root/.venv"
  elif [ -x "$root/.venv-rtx/bin/python" ]; then
    export UV_PROJECT_ENVIRONMENT="$root/.venv-rtx"
  else
    echo "ERROR: no uv environment is available; run: ./slurm/setup_uv.sh" >&2
    return 1
  fi
}
