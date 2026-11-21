# Deploying the optimized consistency fork (IPS sweep)

Branch `perf/ips-consistency`. Commits, oldest first:

| Commit | Content | Revert effect |
|---|---|---|
| `5f1ea27` | frozen `consistency_reference.py`, synthetic data, fixture tooling, baseline test fix | none (no behaviour change) |
| `175aeb8` | `consistency_fast.py` + equivalence tests | none on its own (not wired in) |
| `6083825` | public `get_linregress_consistency` → `consistency_fast` | restores the original public behaviour |
| later | benchmarks, results, this document | none |

`autonomous_drosophila` needs no source changes. It imports
`brainmodel_utils.metrics.consistency.get_linregress_consistency` from the
checkout path it is given (`load_consistency_function`), and that import path,
signature, defaults and return format are unchanged.

## 1. Install on the cluster

```bash
cd /path/to/brainmodel_utils          # the checkout autonomous_drosophila points at
git fetch && git checkout perf/ips-consistency
git rev-parse HEAD                     # record this; see "Provenance" below
```

The cluster environment must provide `xarray`, which the original fork also
imports. The locked versions used locally were numpy 2.0.2, scipy 1.17.1,
scikit-learn 1.8.0, joblib 1.5.3, threadpoolctl 3.6.0.

## 2. Validate on the cluster before any production use

Run each step on a compute node, not a login node.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -c "import brainmodel_utils.metrics.consistency as c, scipy; \
print(c.__file__, c.active_implementation(), scipy.__version__, c._fast.VECTORIZED_PEARSON_AVAILABLE)"
# expect: <checkout>/brainmodel_utils/metrics/consistency.py fast 1.17.x True

python -m pytest -q test                               # bitwise equivalence on synthetic data
BRAINMODEL_SLOW_TESTS=1 python -m pytest -q test -k saved_reference   # 3/10/50-bootstrap fixtures
```

* The bitwise tests compare the fast and reference implementations live, so
  they are valid on the cluster's BLAS. Linux/OpenBLAS can round differently
  from macOS/Accelerate, which is why the suite checks this on the cluster too.
* The saved-fixture tests were recorded on an M1. On a different platform
  they fall back to `rtol=1e-5, atol=1e-6` automatically. A failure there
  means a platform difference in the *reference* itself, not in the fast
  path; rerun `test/make_reference_fixtures.py` on the cluster if needed.
* If `VECTORIZED_PEARSON_AVAILABLE` is `False` (SciPy is not 1.17), the fast
  path still runs, with the bitwise per-target Pearson loop. Only the
  full-data cache and memory savings apply in that case.

Then repeat one completed production comparison: source `fly_098`, target
`fly_099`, right, α=0.01, l1_ratio=0.25, 50 bootstraps, the original start
seed and 5 blocked folds. Run it twice, with
`BRAINMODEL_UTILS_CONSISTENCY_IMPL=reference` and without, saving the full
arrays (all 7 metrics, train and test). Compare NaN masks, shapes and dtypes
exactly, and require `np.array_equal(..., equal_nan=True)`. Any difference is
a blocker, because the unbatched fast path is designed to be bitwise identical
on the same machine.

## 3. Choose the memory mode

| Mode | How | Numerics | Coefficient memory per worker |
|---|---|---|---|
| default | nothing to set | bitwise identical | 1 full matrix (ref: 3) |
| batched | `export BRAINMODEL_UTILS_TARGET_BATCH_SIZE=256` | coefficients identical; predictions within BLAS rounding (~1e-7 rel. float32) | `batch × n_source` |

At production size (27,118 × 27,118), one coefficient matrix is 2.74 GiB in
float32 or 5.48 GiB in float64. The reference keeps three alive per split, so
about 8.2 / 16.4 GiB per worker. The default fast path keeps one (2.74 / 5.48
GiB). Batched mode keeps one batch (256 × 27,118 × 4 B ≈ 26 MiB). Use batched
mode only after checking on the cluster that its differences from the
reference are acceptable for the analysis (step 2 run with the variable set,
compared with a tolerance).

Then measure throughput (Phase 7.3/7.4): wall time, **node-level** peak RAM
(`sacct -o MaxRSS,AveRSS` is per step; use node-level monitoring for packed
jobs), CPU utilisation and ConvergenceWarnings. Pack 1, 2, then 4 comparisons
per node while memory allows. Optimise completed comparisons per node-hour.

## 4. Resume the sweep

* Do not recompute finished comparisons. Keep task IDs, fly pairs,
  directions, hyperparameters, start seeds and output structure.
* Launch only missing or incomplete tasks.
* Record the fork commit per comparison (e.g. add `git -C <checkout> rev-parse
  HEAD` to each job's log or output JSON sidecar). Comparisons done before the
  switch were produced by `3e0d06d`.

## 5. Rollback

Fastest, no reinstall, takes effect at the next call:

```bash
export BRAINMODEL_UTILS_CONSISTENCY_IMPL=reference
```

Permanent:

```bash
git checkout 3e0d06d        # original fork
# or keep the tooling but restore the original public behaviour:
git revert 6083825
```

Also unset `BRAINMODEL_UTILS_TARGET_BATCH_SIZE` if it was set. The reference
implementation ignores it.
