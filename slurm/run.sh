#!/bin/bash

# Submit one main.py action as a Slurm job, e.g.
#   ./slurm/run.sh --slurm-config a100 action=finetune finetune.condition=en

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ "$#" -eq 0 ]; then
  echo "Usage: run.sh --slurm-config NAME [--dependency DEP] [--dry-run] action=... [HYDRA_OVERRIDE ...]" >&2
  exit 2
fi

# shellcheck source=slurm/env.sh
source "$SCRIPT_DIR/env.sh"

use_login_environment "$PROJECT_ROOT"
cd "$PROJECT_ROOT"

exec uv run --frozen --no-sync python "$SCRIPT_DIR/submit.py" "$@"
