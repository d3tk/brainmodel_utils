#!/bin/bash
# Stage 1 of the Babel campaign: GPU 2-boot batching, CPU 64-boot scaling,
# and a 20-bootstrap 1024-target precision screen.
set -euo pipefail
REPO=/home/dkutsche/brainmodel_utils
RUN="$REPO/benchmarks/slurm/run_one.sh"
LOG="$REPO/benchmarks/results/slurm/CAMPAIGN_STAGE1_JOBS.txt"
CPU_PY=/home/dkutsche/autonomous-drosophila/.venv/bin/python
CUDA_PY=/home/dkutsche/brainmodel_utils/.venv/bin/python
mkdir -p "$REPO/benchmarks/results/slurm" "$REPO/benchmarks/results/campaign"
: >"$LOG"

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

echo "Submitting campaign stage 1" | tee -a "$LOG"

submit --job-name=bm-gpu-batch \
  --cpus-per-task=8 --mem=96G --time=03:00:00 \
  "${COMMON[@]}" \
  "$REPO/benchmarks/slurm/run_gpu_batch_sweep.sh"

submit_cpu64() {
  local workers=$1 cpus=$2 mem=$3 time=$4
  submit --job-name="bm-cpu64-w${workers}" \
    --cpus-per-task="$cpus" --mem="$mem" --time="$time" \
    "${COMMON[@]}" \
    --export=ALL,BACKEND=cpu,WORKERS="$workers",BOOTSTRAP_ITERS=64,SEED=1000,STEM="cpu64_w${workers}",PYTHON="$CPU_PY",SOURCE_FLY=fly_100,TARGET_FLY=fly_099,DIRECTION=right \
    "$RUN"
}

# 8- and 32-worker 200-bootstrap results already exist.
submit_cpu64 1 8 32G 10:00:00
submit_cpu64 2 8 48G 08:00:00
submit_cpu64 4 8 80G 04:00:00
submit_cpu64 12 12 196G 03:00:00
submit_cpu64 16 16 256G 02:00:00
submit_cpu64 25 32 384G 02:00:00

submit_prec() {
  local name=$1 backend=$2 workers=$3 python=$4 extra=$5
  submit --job-name="bm-prec-${name}" \
    --cpus-per-task=8 --mem=32G --time=01:00:00 \
    "${COMMON[@]}" \
    --export=ALL,BACKEND="$backend",WORKERS="$workers",BOOTSTRAP_ITERS=20,TARGET_LIMIT=1024,SEED=1000,STEM="prec20t1024_${name}",PYTHON="$python",DTYPE=float64,SAVE_PRECISION=1,SOURCE_FLY=fly_100,TARGET_FLY=fly_099,DIRECTION=right,${extra} \
    "$RUN"
}

submit_prec cpu_f64 cpu 8 "$CPU_PY" COMPUTE_DTYPE=float64
submit_prec cpu_f32 cpu 8 "$CPU_PY" COMPUTE_DTYPE=float32
submit_prec cuda_hyb cuda 1 "$CUDA_PY" "COMPUTE_DTYPE=float32,CORRELATION_DTYPE=float32,WARMUP=1"
submit_prec cuda_f64 cuda 1 "$CUDA_PY" "COMPUTE_DTYPE=float64,CORRELATION_DTYPE=float64,WARMUP=1"

echo "--- queue ---" | tee -a "$LOG"
squeue --me -o '%.18i %.9P %.16j %.2t %.10M %.6D %R' | tee -a "$LOG"
