#!/bin/bash
#SBATCH --job-name=bm-flycheck
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --account=anayebi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/home/dkutsche/brainmodel_utils/benchmarks/results/slurm/%x-%j.out
#SBATCH --error=/home/dkutsche/brainmodel_utils/benchmarks/results/slurm/%x-%j.err
# No debug/array/CPU partition: general GPU node, CPU backend.
set -euo pipefail
REPO=/home/dkutsche/brainmodel_utils
export FLY_DATA_ROOT=${FLY_DATA_ROOT:-/data/group_data/neuroagents_lab/neural_datasets/brezovec_drosophila_data/000727}
PYTHON=${PYTHON:-/home/dkutsche/autonomous-drosophila/.venv/bin/python}
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
mkdir -p "$REPO/benchmarks/results/slurm"
cd "$REPO"
echo "host=$(hostname) job=${SLURM_JOB_ID:-none}"
date -Is
echo "=== tree (depth 2) ==="
find "$FLY_DATA_ROOT" -maxdepth 2 -printf '%y %p\n' | sort | head -80
echo "=== NWB ==="
ls -lh "$FLY_DATA_ROOT"/sub-fly-087/*.nwb 2>/dev/null | head || true
ls -lh "$FLY_DATA_ROOT"/sub-fly-089/*.nwb 2>/dev/null | head || true
echo "=== events ==="
ls -lh "$FLY_DATA_ROOT"/inter_animal_consistency/two_fly_pilot/events | head || true
ls -lh "$FLY_DATA_ROOT"/inter_animal_consistency/two_fly_pilot/event_tensors/short | head || true
"$PYTHON" -m benchmarks.benchmark_neural_backends \
  --suite production \
  --fly-data-root "$FLY_DATA_ROOT" \
  --source-fly fly_087 \
  --target-fly fly_089 \
  --direction right \
  --region IPS \
  --event-source historical \
  --split-strategy blocked \
  --splits 5 \
  --consistency-map percentile \
  --percentile 99 \
  --alpha 1 \
  --backend cpu \
  --workers 1 \
  --bootstrap-iters 1 \
  --target-limit 8 \
  --seed 1000 \
  --workloads consistency \
  --output "$REPO/benchmarks/results/slurm/flycheck_${SLURM_JOB_ID:-local}.json"
date -Is
