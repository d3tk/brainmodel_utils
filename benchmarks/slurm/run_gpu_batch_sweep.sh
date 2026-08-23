#!/bin/bash
# Sequential 2-bootstrap CUDA batching sweep on one GPU.
#SBATCH --job-name=bm-gpu-batch
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --account=anayebi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/home/dkutsche/brainmodel_utils/benchmarks/results/slurm/%x-%j.out
#SBATCH --error=/home/dkutsche/brainmodel_utils/benchmarks/results/slurm/%x-%j.err
set -euo pipefail
REPO=/home/dkutsche/brainmodel_utils
RUN="$REPO/benchmarks/slurm/run_one.sh"
export PYTHON=${PYTHON:-/home/dkutsche/brainmodel_utils/.venv/bin/python}
export FLY_DATA_ROOT=${FLY_DATA_ROOT:-/data/group_data/neuroagents_lab/neural_datasets/brezovec_drosophila_data/000727}
export EVENTS_DIR=${EVENTS_DIR:-/data/group_data/neuroagents_lab/autonomous-drosophila/outputs/interanimal/turn_m15_p0p4_v1/events}
export SOURCE_FLY=fly_100
export TARGET_FLY=fly_099
export DIRECTION=right
export BACKEND=cuda
export WORKERS=1
export SEED=1000
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
mkdir -p "$REPO/benchmarks/results/slurm" "$REPO/benchmarks/results/campaign"
cd "$REPO"
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
  export TMPDIR="$SLURM_TMPDIR"
  export JOBLIB_TEMP_FOLDER="$SLURM_TMPDIR"
fi

echo "host=$(hostname) job=${SLURM_JOB_ID:-none}"
date -Is
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset} node=${SLURM_JOB_NODELIST:-unset}"
lscpu | sed -n '1,40p'
free -h
nvidia-smi -L || true
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,power.draw --format=csv || true
"$PYTHON" -c 'import cupy, torch; print("cupy", cupy.__version__, "torch", torch.__version__, "cuda", torch.cuda.is_available())'
"$PYTHON" -m benchmarks.benchmark_neural_backends \
  --inventory-only \
  --backend cuda \
  --workers 1 \
  --output "$REPO/benchmarks/results/campaign/gpu_batch_inventory_${SLURM_JOB_ID:-local}.json"

mib() { echo $(($1 * 1024 * 1024)); }
gib() { echo $(($1 * 1024 * 1024 * 1024)); }

status=0
run_cfg() {
  echo "=== $* ==="
  date -Is
  if ! bash "$RUN"; then
    echo "FAILED: STEM=${STEM:-unset}"
    status=1
  fi
}

export BOOTSTRAP_ITERS=1
export WARMUP=0
export STEM=gpu2b_cold_default
unset CORRELATION_TARGET_BATCH_SIZE RIDGE_WORKSPACE_BYTES RIDGE_MAX_BATCH_SIZE
run_cfg cold default

export BOOTSTRAP_ITERS=2
export WARMUP=1
unset RIDGE_WORKSPACE_BYTES RIDGE_MAX_BATCH_SIZE
for corr in 128 512 2048 8192 all; do
  export CORRELATION_TARGET_BATCH_SIZE=$corr
  export STEM="gpu2b_c${corr}_r512m_cap1024"
  run_cfg "correlation $corr"
done

export CORRELATION_TARGET_BATCH_SIZE=512
export RIDGE_MAX_BATCH_SIZE=all
for spec in 256M:$(mib 256) 512M:$(mib 512) 1G:$(gib 1) 2G:$(gib 2) 4G:$(gib 4) 8G:$(gib 8); do
  label=${spec%%:*}
  bytes=${spec#*:}
  export RIDGE_WORKSPACE_BYTES=$bytes
  export STEM="gpu2b_c512_r${label}_all"
  run_cfg "ridge $label uncapped"
done

date -Is
echo "sweep_status=$status"
exit "$status"
