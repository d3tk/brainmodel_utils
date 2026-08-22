#!/bin/bash
#SBATCH --job-name=bm-probe
#SBATCH --partition=general
#SBATCH --qos=normal
# No debug/array/CPU partition: request a general GPU node and run CPU or CUDA there.
#SBATCH --account=anayebi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/home/dkutsche/brainmodel_utils/benchmarks/results/slurm/probe-%j.out
#SBATCH --error=/home/dkutsche/brainmodel_utils/benchmarks/results/slurm/probe-%j.err
set -euo pipefail
REPO=/home/dkutsche/brainmodel_utils
FLY_DATA_ROOT=${FLY_DATA_ROOT:-/data/group_data/neuroagents_lab/neural_datasets/brezovec_drosophila_data/000727}
mkdir -p "$REPO/benchmarks/results/slurm"
cd "$REPO"
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}"
date -Is
echo "=== fly data ==="
ls -ld "$FLY_DATA_ROOT" || true
ls "$FLY_DATA_ROOT" | head || true
echo "=== python ==="
PYTHON=/home/dkutsche/autonomous-drosophila/.venv/bin/python
if [[ ! -x "$PYTHON" ]]; then
  PYTHON=python3
fi
"$PYTHON" -c 'import sys; print(sys.executable, sys.version)'
"$PYTHON" -c 'import torch, sys; print("torch", torch.__version__, "cuda", torch.cuda.is_available())' || true
"$PYTHON" -c 'import cupy; print("cupy", cupy.__version__, cupy.cuda.runtime.getDeviceCount())' || true
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" -m benchmarks.benchmark_neural_backends \
  --inventory-only \
  --output "$REPO/benchmarks/results/slurm/probe_inventory_${SLURM_JOB_ID:-local}.json"
"$PYTHON" -m benchmarks.benchmark_neural_backends \
  --suite smoke \
  --backend cpu \
  --workloads consistency \
  --monitor-interval 0.05 \
  --output "$REPO/benchmarks/results/slurm/probe_cpu_smoke_${SLURM_JOB_ID:-local}.json"
if "$PYTHON" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
  "$PYTHON" -m benchmarks.benchmark_neural_backends \
    --suite smoke \
    --backend cuda \
    --workers 1 \
    --workloads consistency \
    --monitor-interval 0.05 \
    --output "$REPO/benchmarks/results/slurm/probe_cuda_smoke_${SLURM_JOB_ID:-local}.json" \
    || echo "CUDA smoke failed; cupy/cuML may be missing in this env"
fi
date -Is
