#!/bin/bash
# Float64 packing. CPU default in sklearn/brainmodel_utils is float64;
# the benchmark harness had been forcing float32 on load.
# CUDA 1x 64-boot and 1x 200-boot f64 are already running (10197917, 10197918).
set -euo pipefail
REPO=/home/dkutsche/brainmodel_utils
RUN="$REPO/benchmarks/slurm/run_one.sh"
LOG="$REPO/benchmarks/results/slurm/F64_PACKING_JOBS.txt"
CPU_PY=/home/dkutsche/autonomous-drosophila/.venv/bin/python
CUDA_PY=/home/dkutsche/brainmodel_utils/.venv/bin/python

COMMON=(
  --partition=general
  --qos=normal
  --account=anayebi
  --nodes=1
  --ntasks=1
  --cpus-per-task=32
)
# CPU f64 RSS is ~2x f32 (~400 GiB at 32 workers f32).
CPU_ALLOC=( "${COMMON[@]}" --gres=gpu:1 --mem=960G --time=03:00:00 )
CUDA_ALLOC=( "${COMMON[@]}" --gres=gpu:1 --mem=128G --time=03:00:00 )
# 8x f64 will not fit in 48 GiB (8x f32 was 28 GiB).
CUDA8_ALLOC=( "${COMMON[@]}" --gres=gpu:A100_80GB:1 --mem=128G )

submit() {
  local job
  job=$(sbatch --parsable "$@")
  echo "$job $*" | tee -a "$LOG"
}

echo "Submitting float64 packing" | tee -a "$LOG"

CPU64=ALL,BACKEND=cpu,DTYPE=float64,COMPUTE_DTYPE=float64,SEED=1000,PYTHON="$CPU_PY",SOURCE_FLY=fly_100,TARGET_FLY=fly_099,DIRECTION=right,BOOTSTRAP_ITERS=64

submit --job-name=bm-t64-c1x32-f64 \
  "${CPU_ALLOC[@]}" \
  --export=${CPU64},WORKERS=32,JOBS=1,MAX_CONCURRENT=1,STEM=thru64_cpu_f64_1x32 \
  "$RUN"

submit --job-name=bm-t64-c2x16-f64 \
  "${CPU_ALLOC[@]}" \
  --export=${CPU64},WORKERS=16,JOBS=2,MAX_CONCURRENT=2,STEM=thru64_cpu_f64_2x16 \
  "$RUN"

submit --job-name=bm-t64-c4x8-f64 \
  "${CPU_ALLOC[@]}" \
  --export=${CPU64},WORKERS=8,JOBS=4,MAX_CONCURRENT=4,STEM=thru64_cpu_f64_4x8 \
  "$RUN"

# Latency reference: CPU 1x32 at 200 bootstraps, float64.
submit --job-name=bm-cpu200-f64 \
  --partition=general --qos=normal --account=anayebi --nodes=1 --ntasks=1 \
  --cpus-per-task=32 --gres=gpu:1 --mem=960G --time=03:00:00 \
  --export=ALL,BACKEND=cpu,WORKERS=32,JOBS=1,MAX_CONCURRENT=1,BOOTSTRAP_ITERS=200,SEED=1000,STEM=cpu200_f64_x32,PYTHON="$CPU_PY",DTYPE=float64,COMPUTE_DTYPE=float64,SOURCE_FLY=fly_100,TARGET_FLY=fly_099,DIRECTION=right \
  "$RUN"

CUDA64=ALL,BACKEND=cuda,WORKERS=1,DTYPE=float64,COMPUTE_DTYPE=float64,CORRELATION_DTYPE=float64,RIDGE_MAX_BATCH_SIZE=all,RIDGE_WORKSPACE_BYTES=8589934592,WARMUP=1,SEED=1000,PYTHON="$CUDA_PY",SOURCE_FLY=fly_100,TARGET_FLY=fly_099,DIRECTION=right,BOOTSTRAP_ITERS=64

# 2x/4x: 8 GiB workspace each; ~16/32 GiB VRAM expected, fits 48 GB.
submit --job-name=bm-t64-g2-f64 \
  "${CUDA_ALLOC[@]}" \
  --export=${CUDA64},JOBS=2,MAX_CONCURRENT=2,STEM=thru64_cuda_f64_x2 \
  "$RUN"

submit --job-name=bm-t64-g4-f64 \
  "${CUDA_ALLOC[@]}" \
  --export=${CUDA64},JOBS=4,MAX_CONCURRENT=4,STEM=thru64_cuda_f64_x4 \
  "$RUN"

# 8x: 4 GiB workspace so 8 processes can fit on 80 GB.
CUDA8=ALL,BACKEND=cuda,WORKERS=1,DTYPE=float64,COMPUTE_DTYPE=float64,CORRELATION_DTYPE=float64,RIDGE_MAX_BATCH_SIZE=all,RIDGE_WORKSPACE_BYTES=4294967296,WARMUP=1,SEED=1000,PYTHON="$CUDA_PY",SOURCE_FLY=fly_100,TARGET_FLY=fly_099,DIRECTION=right

submit --job-name=bm-t64-g8-f64 \
  "${CUDA8_ALLOC[@]}" --time=03:00:00 \
  --export=${CUDA8},JOBS=8,MAX_CONCURRENT=8,BOOTSTRAP_ITERS=64,STEM=thru64_cuda_f64_x8 \
  "$RUN"

submit --job-name=bm-t200-g8-f64 \
  "${CUDA8_ALLOC[@]}" --time=04:00:00 \
  --export=${CUDA8},JOBS=8,MAX_CONCURRENT=8,BOOTSTRAP_ITERS=200,STEM=thru200_cuda_f64_x8 \
  "$RUN"

echo "--- queue ---" | tee -a "$LOG"
squeue --me -o '%.18i %.20j %.2t %.10M %.10l %R' | tee -a "$LOG"
