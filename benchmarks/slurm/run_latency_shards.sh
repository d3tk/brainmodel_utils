#!/bin/bash
# One comparison, one A100 80GB, multiple workers=1 CUDA processes.
# 64-bootstrap screen: 1x64, 2x32, 4x16, 8x8. Wall time includes load, warmup, merge.
#SBATCH --job-name=bm-lat64-shards
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --account=anayebi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH --mem=384G
#SBATCH --time=04:00:00
#SBATCH --output=/home/dkutsche/brainmodel_utils/benchmarks/results/slurm/%x-%j.out
#SBATCH --error=/home/dkutsche/brainmodel_utils/benchmarks/results/slurm/%x-%j.err
set -euo pipefail
REPO=/home/dkutsche/brainmodel_utils
PYTHON=$REPO/.venv/bin/python
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export FLY_DATA_ROOT=${FLY_DATA_ROOT:-/data/group_data/neuroagents_lab/neural_datasets/brezovec_drosophila_data/000727}
export EVENTS_DIR=${EVENTS_DIR:-/data/group_data/neuroagents_lab/autonomous-drosophila/outputs/interanimal/turn_m15_p0p4_v1/events}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
cd "$REPO"
nvidia-smi -L
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
date -Is
for shards in 1 2 4 8; do
  echo "=== shards=$shards ==="
  date -Is
  extra=(--save-precision --warmup --workers 1)
  if [[ "$shards" -gt 1 ]]; then
    extra+=(--shards "$shards" --devices 0)
  fi
  "$PYTHON" -m benchmarks.campaign \
    --output-dir "$REPO/benchmarks/results/campaign" \
    --stem "lat64_f64_s${shards}" \
    --backend cuda \
    --bootstrap-iters 64 \
    --seed 1000 \
    --python "$PYTHON" \
    --source-fly fly_100 \
    --target-fly fly_099 \
    --direction right \
    --events-dir "$EVENTS_DIR" \
    --dtype float64 \
    --compute-dtype float64 \
    --correlation-dtype float64 \
    --ridge-max-batch-size all \
    --ridge-workspace-bytes 8589934592 \
    "${extra[@]}"
  date -Is
done
date -Is
