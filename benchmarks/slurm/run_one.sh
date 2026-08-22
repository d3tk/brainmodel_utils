#!/bin/bash
#SBATCH --job-name=bm-run
#SBATCH --partition=general
#SBATCH --qos=normal
# No debug/array/CPU partition: request a general GPU node and run CPU or CUDA there.
#SBATCH --account=anayebi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=196G
#SBATCH --time=08:00:00
#SBATCH --output=/home/dkutsche/brainmodel_utils/benchmarks/results/slurm/%x-%j.out
#SBATCH --error=/home/dkutsche/brainmodel_utils/benchmarks/results/slurm/%x-%j.err
set -euo pipefail
REPO=/home/dkutsche/brainmodel_utils
export FLY_DATA_ROOT=${FLY_DATA_ROOT:-/data/group_data/neuroagents_lab/neural_datasets/brezovec_drosophila_data/000727}
export EVENTS_DIR=${EVENTS_DIR:-/data/group_data/neuroagents_lab/autonomous-drosophila/outputs/interanimal/turn_m15_p0p4_v1/events}
PYTHON=${PYTHON:-/home/dkutsche/autonomous-drosophila/.venv/bin/python}
BACKEND=${BACKEND:-cpu}
WORKERS=${WORKERS:-8}
BOOTSTRAP_ITERS=${BOOTSTRAP_ITERS:-20}
TARGET_LIMIT=${TARGET_LIMIT:-}
SEED=${SEED:-1000}
STEM=${STEM:-screen}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$REPO/benchmarks/results/slurm" "$REPO/benchmarks/results/campaign"
cd "$REPO"
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
  export TMPDIR="$SLURM_TMPDIR"
  export JOBLIB_TEMP_FOLDER="$SLURM_TMPDIR"
fi
cmd=(
  "$PYTHON" -m benchmarks.campaign
  --output-dir "$REPO/benchmarks/results/campaign"
  --stem "$STEM"
  --backend "$BACKEND"
  --workers "$WORKERS"
  --bootstrap-iters "$BOOTSTRAP_ITERS"
  --seed "$SEED"
  --jobs "${JOBS:-1}"
  --replica "${REPLICA:-0}"
  --max-concurrent "${MAX_CONCURRENT:-1}"
  --python "$PYTHON"
  --source-fly "${SOURCE_FLY:-fly_100}"
  --target-fly "${TARGET_FLY:-fly_099}"
  --direction "${DIRECTION:-right}"
  --events-dir "$EVENTS_DIR"
)
if [[ -n "$TARGET_LIMIT" ]]; then
  cmd+=(--target-limit "$TARGET_LIMIT")
fi
if [[ -n "${CORRELATION_TARGET_BATCH_SIZE:-}" ]]; then
  cmd+=(--correlation-target-batch-size "$CORRELATION_TARGET_BATCH_SIZE")
fi
if [[ -n "${RIDGE_WORKSPACE_BYTES:-}" ]]; then
  cmd+=(--ridge-workspace-bytes "$RIDGE_WORKSPACE_BYTES")
fi
if [[ -n "${DTYPE:-}" ]]; then
  cmd+=(--dtype "$DTYPE")
fi
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} backend=$BACKEND workers=$WORKERS"
date -Is
printf '%q ' "${cmd[@]}"
echo
"${cmd[@]}"
date -Is
