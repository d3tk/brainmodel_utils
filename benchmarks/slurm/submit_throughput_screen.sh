#!/bin/bash
# Throughput screen on identical 32-CPU / 1-GPU allocations.
# 64 bootstraps per comparison (divides 32/16/8 workers; long enough for GPU contention).
# CPU: 1x32, 2x16, 4x8. CUDA: 1/2/4/8 processes, workers=1, Ridge-all 4 GiB.
set -euo pipefail
REPO=/home/dkutsche/brainmodel_utils
RUN="$REPO/benchmarks/slurm/run_one.sh"
LOG="$REPO/benchmarks/results/slurm/THROUGHPUT_SCREEN_JOBS.txt"
CPU_PY=/home/dkutsche/autonomous-drosophila/.venv/bin/python
CUDA_PY=/home/dkutsche/brainmodel_utils/.venv/bin/python
mkdir -p "$REPO/benchmarks/results/slurm" "$REPO/benchmarks/results/campaign"

COMMON=(
  --partition=general
  --qos=normal
  --account=anayebi
  --nodes=1
  --ntasks=1
  --cpus-per-task=32
  --gres=gpu:1
  --mem=448G
  --time=03:00:00
)

submit() {
  local job
  job=$(sbatch --parsable "$@")
  echo "$job $*" | tee -a "$LOG"
}

echo "Submitting throughput 64-bootstrap screen" | tee -a "$LOG"

submit --job-name=bm-t64-c1x32 \
  "${COMMON[@]}" \
  --export=ALL,BACKEND=cpu,WORKERS=32,JOBS=1,MAX_CONCURRENT=1,BOOTSTRAP_ITERS=64,SEED=1000,STEM=thru64_cpu_1x32,PYTHON="$CPU_PY",SOURCE_FLY=fly_100,TARGET_FLY=fly_099,DIRECTION=right \
  "$RUN"

submit --job-name=bm-t64-c2x16 \
  "${COMMON[@]}" \
  --export=ALL,BACKEND=cpu,WORKERS=16,JOBS=2,MAX_CONCURRENT=2,BOOTSTRAP_ITERS=64,SEED=1000,STEM=thru64_cpu_2x16,PYTHON="$CPU_PY",SOURCE_FLY=fly_100,TARGET_FLY=fly_099,DIRECTION=right \
  "$RUN"

submit --job-name=bm-t64-c4x8 \
  "${COMMON[@]}" \
  --export=ALL,BACKEND=cpu,WORKERS=8,JOBS=4,MAX_CONCURRENT=4,BOOTSTRAP_ITERS=64,SEED=1000,STEM=thru64_cpu_4x8,PYTHON="$CPU_PY",SOURCE_FLY=fly_100,TARGET_FLY=fly_099,DIRECTION=right \
  "$RUN"

submit --job-name=bm-t64-g1 \
  "${COMMON[@]}" \
  --export=ALL,BACKEND=cuda,WORKERS=1,JOBS=1,MAX_CONCURRENT=1,BOOTSTRAP_ITERS=64,SEED=1000,STEM=thru64_cuda_x1,PYTHON="$CUDA_PY",SOURCE_FLY=fly_100,TARGET_FLY=fly_099,DIRECTION=right,WARMUP=1,RIDGE_MAX_BATCH_SIZE=all,RIDGE_WORKSPACE_BYTES=4294967296 \
  "$RUN"

submit --job-name=bm-t64-g2 \
  "${COMMON[@]}" \
  --export=ALL,BACKEND=cuda,WORKERS=1,JOBS=2,MAX_CONCURRENT=2,BOOTSTRAP_ITERS=64,SEED=1000,STEM=thru64_cuda_x2,PYTHON="$CUDA_PY",SOURCE_FLY=fly_100,TARGET_FLY=fly_099,DIRECTION=right,WARMUP=1,RIDGE_MAX_BATCH_SIZE=all,RIDGE_WORKSPACE_BYTES=4294967296 \
  "$RUN"

submit --job-name=bm-t64-g4 \
  "${COMMON[@]}" \
  --export=ALL,BACKEND=cuda,WORKERS=1,JOBS=4,MAX_CONCURRENT=4,BOOTSTRAP_ITERS=64,SEED=1000,STEM=thru64_cuda_x4,PYTHON="$CUDA_PY",SOURCE_FLY=fly_100,TARGET_FLY=fly_099,DIRECTION=right,WARMUP=1,RIDGE_MAX_BATCH_SIZE=all,RIDGE_WORKSPACE_BYTES=4294967296 \
  "$RUN"

submit --job-name=bm-t64-g8 \
  "${COMMON[@]}" \
  --export=ALL,BACKEND=cuda,WORKERS=1,JOBS=8,MAX_CONCURRENT=8,BOOTSTRAP_ITERS=64,SEED=1000,STEM=thru64_cuda_x8,PYTHON="$CUDA_PY",SOURCE_FLY=fly_100,TARGET_FLY=fly_099,DIRECTION=right,WARMUP=1,RIDGE_MAX_BATCH_SIZE=all,RIDGE_WORKSPACE_BYTES=4294967296 \
  "$RUN"

echo "--- queue ---" | tee -a "$LOG"
squeue --me -o '%.18i %.9P %.16j %.2t %.10M %.6D %R' | tee -a "$LOG"
