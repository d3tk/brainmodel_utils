#!/bin/bash
#SBATCH --job-name=bm-events
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --account=anayebi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/home/dkutsche/brainmodel_utils/benchmarks/results/slurm/%x-%j.out
#SBATCH --error=/home/dkutsche/brainmodel_utils/benchmarks/results/slurm/%x-%j.err
set -euo pipefail
REPO=/home/dkutsche/brainmodel_utils
DANDI=/data/group_data/neuroagents_lab/neural_datasets/brezovec_drosophila_data/000727
AD_EVENTS=/data/group_data/neuroagents_lab/autonomous-drosophila/outputs/interanimal/turn_m15_p0p4_v1/events
STAGE="$REPO/benchmarks/data/events"
PYTHON=${PYTHON:-/home/dkutsche/autonomous-drosophila/.venv/bin/python}
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$REPO/benchmarks/results/slurm" "$STAGE"
echo "host=$(hostname) job=${SLURM_JOB_ID:-none}"
date -Is
echo "=== candidate event dirs ==="
ls -ld "$DANDI/inter_animal_consistency/two_fly_pilot" 2>/dev/null || echo "no two_fly_pilot"
ls -ld "$AD_EVENTS" 2>/dev/null || echo "no AD events"
echo "=== AD events ==="
ls -lh "$AD_EVENTS" 2>/dev/null | head -40 || true
echo "=== find npz/csv ==="
find /data/group_data/neuroagents_lab -name 'fly_087_right.npz' -o -name 'fly_087_turn_bouts.csv' 2>/dev/null | head -20 || true
"$PYTHON" - <<PY
from pathlib import Path
import numpy as np
ad = Path("$AD_EVENTS")
stage = Path("$STAGE")
stage.mkdir(parents=True, exist_ok=True)
if not ad.is_dir():
    raise SystemExit("AD events dir missing")
for path in sorted(ad.glob("fly_087_*.npz")) + sorted(ad.glob("fly_089_*.npz")):
    with np.load(path, allow_pickle=False) as data:
        keys = list(data.files)
        n = data["event_times"].size if "event_times" in data.files else None
        rel = data["relative_times"].shape if "relative_times" in data.files else None
        print(f"{path.name} keys={keys} n_events={n} relative={rel}")
    dest = stage / path.name
    if not dest.exists():
        dest.write_bytes(path.read_bytes())
        print(f"staged {dest}")
print("stage", stage, "files", sorted(p.name for p in stage.glob("*.npz")))
PY
export EVENTS_DIR="$STAGE"
"$PYTHON" -m benchmarks.benchmark_neural_backends \
  --suite production \
  --fly-data-root "$DANDI" \
  --events-dir "$STAGE" \
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
  --output "$REPO/benchmarks/results/slurm/events_smoke_${SLURM_JOB_ID:-local}.json"
date -Is
