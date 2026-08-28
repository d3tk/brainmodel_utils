#!/bin/bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 CONFIGURATION_MANIFEST OUTPUT_DIR" >&2
  exit 2
fi

MANIFEST=$(realpath "$1")
OUTPUT_DIR=$(realpath -m "$2")
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PARTITION=${PARTITION:-general}
if [[ "$PARTITION" == "preempt" ]]; then
  QOS=${QOS:-preempt_qos}
  MAX_IN_FLIGHT=${MAX_IN_FLIGHT:-24}
else
  QOS=${QOS:-normal}
  MAX_IN_FLIGHT=${MAX_IN_FLIGHT:-8}
fi

options=(
  --partition "$PARTITION"
  --qos "$QOS"
  --gres gpu:1
  --constraint "${GPU_CONSTRAINT:-VRAM_48GB}"
  --array "0-15%${MAX_IN_FLIGHT}"
  --export "ALL,CONFIGURATION_MANIFEST=$MANIFEST,OUTPUT_DIR=$OUTPUT_DIR"
)
if [[ -n "${SBATCH_ACCOUNT:-}" ]]; then
  options+=(--account "$SBATCH_ACCOUNT")
fi

sbatch "${options[@]}" "$SCRIPT_DIR/run_manifest_cell.sh"
