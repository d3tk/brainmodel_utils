#!/bin/bash
# Full IPS-CV comparison: fly_100 → fly_099 right, all IPS voxels, 200 bootstraps.
# Matches autonomous-drosophila cpu-scale percentile99 (not the 128-target screen).
set -euo pipefail
REPO=/home/dkutsche/brainmodel_utils
SCRIPT="$REPO/benchmarks/slurm/run_one.sh"
EVENTS_DIR=/data/group_data/neuroagents_lab/autonomous-drosophila/outputs/interanimal/turn_m15_p0p4_v1/events
COMMON=(
  --partition=general
  --qos=normal
  --account=anayebi
  --gres=gpu:1
)
export EVENTS_DIR
export SOURCE_FLY=fly_100
export TARGET_FLY=fly_099
export DIRECTION=right
export BOOTSTRAP_ITERS=200
export SEED=1000
unset TARGET_LIMIT
unset PROTOCOL_RELATIVE_TIMES

echo "Submitting IPS-CV-matched full comparisons on general"

# Same worker count as the 1.77 h IPS CV percentile-99 CPU-scale point.
sbatch --parsable --job-name=bm-ips-cpu32 --cpus-per-task=32 --mem=384G --time=12:00:00 \
  "${COMMON[@]}" \
  --export=ALL,BACKEND=cpu,WORKERS=32,BOOTSTRAP_ITERS=200,SEED=1000,STEM=ips_cv_f100_t099_right_b200,PYTHON=/home/dkutsche/autonomous-drosophila/.venv/bin/python,SOURCE_FLY,TARGET_FLY,DIRECTION,EVENTS_DIR \
  "$SCRIPT"

# 8 workers: same packing as the local 3.5 h fly_087 protocol.
sbatch --parsable --job-name=bm-ips-cpu8 --cpus-per-task=8 --mem=196G --time=16:00:00 \
  "${COMMON[@]}" \
  --export=ALL,BACKEND=cpu,WORKERS=8,BOOTSTRAP_ITERS=200,SEED=1000,STEM=ips_cv_f100_t099_right_b200,PYTHON=/home/dkutsche/autonomous-drosophila/.venv/bin/python,SOURCE_FLY,TARGET_FLY,DIRECTION,EVENTS_DIR \
  "$SCRIPT"

# CUDA, one process, full comparison.
sbatch --parsable --job-name=bm-ips-cuda1 --cpus-per-task=8 --mem=96G --time=08:00:00 \
  "${COMMON[@]}" \
  --export=ALL,BACKEND=cuda,WORKERS=1,BOOTSTRAP_ITERS=200,SEED=1000,STEM=ips_cv_f100_t099_right_b200,PYTHON=/home/dkutsche/brainmodel_utils/.venv/bin/python,SOURCE_FLY,TARGET_FLY,DIRECTION,EVENTS_DIR \
  "$SCRIPT"

squeue --me -o '%.18i %.9P %.16j %.2t %.10M %R' | grep -E 'JOBID|bm-ips' || true
