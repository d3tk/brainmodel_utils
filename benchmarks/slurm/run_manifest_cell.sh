#!/bin/bash
#SBATCH --job-name=bm-grid
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=192G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail

: "${CONFIGURATION_MANIFEST:?set CONFIGURATION_MANIFEST}"
: "${OUTPUT_DIR:?set OUTPUT_DIR outside the repository}"
: "${FLY_DATA_ROOT:?set FLY_DATA_ROOT}"
: "${SOURCE_FLY:?set SOURCE_FLY}"
: "${TARGET_FLY:?set TARGET_FLY}"

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PYTHON=${PYTHON:-$REPO/.venv/bin/python}
CONFIG_ID=${CONFIG_ID:-}
if [[ -z "$CONFIG_ID" ]]; then
  : "${SLURM_ARRAY_TASK_ID:?set CONFIG_ID or submit this script as an array}"
  CONFIG_ID=$(
    "$PYTHON" - "$CONFIGURATION_MANIFEST" "$SLURM_ARRAY_TASK_ID" <<'PY'
import json
import sys

with open(sys.argv[1]) as stream:
    manifest = json.load(stream)
print(manifest["configurations"][int(sys.argv[2])]["id"])
PY
  )
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
SHARDS=${SHARDS:-1}
MAX_CONCURRENT=${MAX_CONCURRENT:-$SHARDS}
mkdir -p "$OUTPUT_DIR"
cd "$REPO"

nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv
cmd=(
  "$PYTHON" -m benchmarks.campaign
  --configuration-manifest "$CONFIGURATION_MANIFEST"
  --config-id "$CONFIG_ID"
  --output-dir "$OUTPUT_DIR"
  --stem "${STEM:-grid}"
  --workers 1
  --shards "$SHARDS"
  --max-concurrent "$MAX_CONCURRENT"
  --python "$PYTHON"
  --fly-data-root "$FLY_DATA_ROOT"
  --events-dir "$EVENTS_DIR"
  --source-fly "$SOURCE_FLY"
  --target-fly "$TARGET_FLY"
  --direction "${DIRECTION:-right}"
  --region "${REGION:-IPS}"
  --event-source "${EVENT_SOURCE:-historical}"
  --ridge-max-batch-size "${RIDGE_MAX_BATCH_SIZE:-all}"
  --ridge-workspace-bytes "${RIDGE_WORKSPACE_BYTES:-4294967296}"
  --save-precision
  --warmup
)
if [[ -n "${EVENTS_DIR:-}" ]]; then
  cmd+=(--events-dir "$EVENTS_DIR")
fi
if [[ "$SHARDS" -gt 1 ]]; then
  cmd+=(--devices 0)
fi

printf '%q ' "${cmd[@]}"
echo
"${cmd[@]}"
