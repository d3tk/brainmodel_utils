#!/bin/bash
#SBATCH --job-name=bm-cudaenv
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --account=anayebi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/dkutsche/brainmodel_utils/benchmarks/results/slurm/%x-%j.out
#SBATCH --error=/home/dkutsche/brainmodel_utils/benchmarks/results/slurm/%x-%j.err
set -euo pipefail
REPO=/home/dkutsche/brainmodel_utils
VENV="$REPO/.venv"
cd "$REPO"
mkdir -p "$REPO/benchmarks/results/slurm"
echo "host=$(hostname) job=${SLURM_JOB_ID:-none}"
date -Is
export PATH="$HOME/.local/bin:$PATH"
if [[ ! -x "$VENV/bin/python" ]]; then
  uv venv --python 3.12 "$VENV"
fi
uv pip install --python "$VENV/bin/python" \
  --index https://pypi.nvidia.com \
  --index-strategy unsafe-best-match \
  -e "$REPO" \
  "scikit-learn<1.8" \
  "cuml-cu12==25.10.*" \
  "cupy-cuda12x[ctk]" \
  torch
"$VENV/bin/python" - <<'PY'
import torch
import cupy
import cuml
print("python ok")
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "count", torch.cuda.device_count())
print("cupy", cupy.__version__, "devices", cupy.cuda.runtime.getDeviceCount())
print("cuml", cuml.__version__)
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY
date -Is
