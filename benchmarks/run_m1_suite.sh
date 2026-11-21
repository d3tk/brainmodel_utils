#!/usr/bin/env bash
# Reproducible local (Apple Silicon) benchmark suite for Phase 5.
# Runs serially; each measurement is a fresh subprocess with 1 BLAS/OpenMP thread.
# Usage: benchmarks/run_m1_suite.sh [python]   (default: .venv/bin/python)
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${1:-.venv/bin/python}"
OUT=benchmarks/results
mkdir -p "$OUT"
B="$PY benchmarks/bench_consistency.py"

# Small tier, production max_iter=10000: every optimization stage.
$B --tier small --boots 3 --repeats 3 --variants reference vectorized cache_only cached batched --batch-sizes 64 | tee "$OUT/small_b3.txt"
$B --tier small --boots 10 --repeats 1 --variants reference cached batched --batch-sizes 64 | tee "$OUT/small_b10.txt"
# Medium tier (max_iter=50: fast-converging regime, correlation cost visible).
$B --tier medium --boots 2 --repeats 3 | tee "$OUT/medium_b2.txt"
# Wide-source tier: production source dimensionality.
$B --tier wide --boots 2 --repeats 3 --variants reference cached | tee "$OUT/wide_b2.txt"
# Memory tier: large coefficient matrices (27118 x 4096).
$B --tier memory --boots 1 --repeats 1 --variants reference cached batched --batch-sizes 64 256 1024 | tee "$OUT/memory_b1.txt"
# Two workers (process-tree memory, serialization).
$B --tier medium --boots 4 --repeats 1 --jobs 2 --variants reference cached batched --batch-sizes 256 | tee "$OUT/medium_b4_j2.txt"
