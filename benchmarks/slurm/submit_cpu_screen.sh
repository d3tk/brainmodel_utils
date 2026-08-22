#!/bin/bash
# Submit CPU worker-count screens on general GPU nodes.
set -euo pipefail
REPO=/home/dkutsche/brainmodel_utils
SCRIPT="$REPO/benchmarks/slurm/run_one.sh"
LOG="$REPO/benchmarks/results/slurm/CPU_SCREEN_JOBS.txt"
mkdir -p "$REPO/benchmarks/results/slurm" "$REPO/benchmarks/results/campaign"
: >"$LOG"
submit_one() {
  local workers=$1 cpus=$2 mem=$3
  local job
  job=$(
    sbatch --parsable \
      --job-name="bm-cpu-w${workers}" \
      --partition=general \
      --qos=normal \
      --account=anayebi \
      --cpus-per-task="$cpus" \
      --mem="$mem" \
      --gres=gpu:1 \
      --time=02:00:00 \
      --export=ALL,BACKEND=cpu,WORKERS="$workers",BOOTSTRAP_ITERS=5,TARGET_LIMIT=128,STEM=cpu_screen_b5t128,PYTHON=/home/dkutsche/autonomous-drosophila/.venv/bin/python \
      "$SCRIPT"
  )
  echo "$job workers=$workers cpus=$cpus mem=$mem" | tee -a "$LOG"
}
submit_one 1 8 32G
submit_one 2 8 48G
submit_one 4 8 64G
submit_one 8 8 80G
submit_one 16 16 128G
cat "$LOG"
squeue --me -n bm-cpu-w1,bm-cpu-w2,bm-cpu-w4,bm-cpu-w8,bm-cpu-w16
