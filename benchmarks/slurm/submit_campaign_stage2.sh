#!/bin/bash
# Stage 2: CUDA 20-boot A/B (default vs Ridge-all 4 GiB) and CPU float64 precision rerun.
# Does not cancel cpu64_w1. Does not submit 200-boot CUDA, repeats, or throughput.
set -euo pipefail
REPO=/home/dkutsche/brainmodel_utils
RUN="$REPO/benchmarks/slurm/run_one.sh"
LOG="$REPO/benchmarks/results/slurm/CAMPAIGN_STAGE2_JOBS.txt"
CPU_PY=/home/dkutsche/autonomous-drosophila/.venv/bin/python
CUDA_PY=/home/dkutsche/brainmodel_utils/.venv/bin/python
mkdir -p "$REPO/benchmarks/results/slurm" "$REPO/benchmarks/results/campaign"

COMMON=(
  --partition=general
  --qos=normal
  --account=anayebi
  --gres=gpu:1
)

submit() {
  local job
  job=$(sbatch --parsable "$@")
  echo "$job $*" | tee -a "$LOG"
}

echo "Submitting campaign stage 2" | tee -a "$LOG"

# Matched 20-bootstrap CUDA A/B. Keep correlation chunking at the default 512.
submit --job-name=bm-cuda20-def \
  --cpus-per-task=8 --mem=96G --time=02:00:00 \
  "${COMMON[@]}" \
  --export=ALL,BACKEND=cuda,WORKERS=1,BOOTSTRAP_ITERS=20,SEED=1000,STEM=cuda20_default,PYTHON="$CUDA_PY",SOURCE_FLY=fly_100,TARGET_FLY=fly_099,DIRECTION=right,WARMUP=1 \
  "$RUN"

submit --job-name=bm-cuda20-opt \
  --cpus-per-task=8 --mem=96G --time=02:00:00 \
  "${COMMON[@]}" \
  --export=ALL,BACKEND=cuda,WORKERS=1,BOOTSTRAP_ITERS=20,SEED=1000,STEM=cuda20_ridge4g_all,PYTHON="$CUDA_PY",SOURCE_FLY=fly_100,TARGET_FLY=fly_099,DIRECTION=right,WARMUP=1,RIDGE_MAX_BATCH_SIZE=all,RIDGE_WORKSPACE_BYTES=4294967296 \
  "$RUN"

# CPU float64 precision: 32G OOM'd; request 96G and keep per-target metric dumps.
submit --job-name=bm-prec-cpu_f64 \
  --cpus-per-task=8 --mem=96G --time=01:00:00 \
  "${COMMON[@]}" \
  --export=ALL,BACKEND=cpu,WORKERS=8,BOOTSTRAP_ITERS=20,TARGET_LIMIT=1024,SEED=1000,STEM=prec20t1024_cpu_f64,PYTHON="$CPU_PY",DTYPE=float64,COMPUTE_DTYPE=float64,SAVE_PRECISION=1,SOURCE_FLY=fly_100,TARGET_FLY=fly_099,DIRECTION=right \
  "$RUN"

echo "--- queue ---" | tee -a "$LOG"
squeue --me -o '%.18i %.9P %.16j %.2t %.10M %.6D %R' | tee -a "$LOG"
