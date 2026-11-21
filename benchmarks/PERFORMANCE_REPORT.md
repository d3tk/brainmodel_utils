# Optimizing `get_linregress_consistency`: changes and results

Branch `perf/ips-consistency` of the `brainmodel_utils` fork, based on `3e0d06d`.
All measurements were taken on an Apple M1 (4 performance + 4 efficiency cores,
8 GB RAM) with Python 3.12.11, numpy 2.0.2, scipy 1.17.1 and scikit-learn 1.8.0,
the versions locked by `autonomous_drosophila`. BLAS and OpenMP were pinned to
1 thread. Cluster validation (Phase 7) is still to do; see
[DEPLOYMENT.md](DEPLOYMENT.md).

## Summary

- **Same results.** The default optimized path is **bitwise identical** to the
  original: values, NaN masks, shapes, dtypes and key order, on every synthetic
  test case and on the saved 3/10/50-bootstrap reference outputs (97/97 tests
  pass).
- **Faster.** Wall-clock time drops by **1.39×–1.62×** at production
  regression settings and **1.55×** when fits converge quickly. The gain grows
  with the bootstrap count.
- **Less RAM.** Peak live memory for regression coefficients drops by **47%**
  (exact mode) or **85%** (optional target batching, 256 targets per batch).
  At production size that is about **11.9 → 6.4 GiB** per worker (exact) or
  **→ ~0.1 GiB** (batched), float32.
- **No change to CPU efficiency.** Each run stays single-threaded (≈1.0 core
  busy). The speedup comes from doing less work: CPU-seconds fall in proportion
  to wall time. The lower memory per worker is what allows more concurrent
  comparisons per node.
- **No changes to the caller.** `autonomous_drosophila` needs no edits, and the
  unmodified `scripts/consistency.py::fit_tensors` gives identical scores
  through the new path.

## What changed

| File | Purpose |
|---|---|
| `brainmodel_utils/metrics/consistency_reference.py` | verbatim copy of the original implementation, kept callable as the baseline |
| `brainmodel_utils/metrics/consistency_fast.py` | optimized implementation |
| `brainmodel_utils/metrics/consistency.py` | public entry point, now routed to the fast path, with an env-var rollback switch |
| `test/synthetic_consistency_data.py` | deterministic synthetic fly pairs and edge cases |
| `test/test_consistency_fast.py` | 90+ equivalence and API tests |
| `test/fixtures/reference_primary_b{3,10,50}.npz` | saved full reference outputs |
| `benchmarks/bench_consistency.py`, `run_m1_suite.sh` | runtime / RAM / CPU benchmark |

Statistical methodology is untouched: same trial averages, split halves,
bootstrap seeds, blocked CV folds, `ElasticNet` objective and Spearman-Brown
correction. All three optimizations below sit in `consistency_fast.py` and can
be switched on and off independently
(`compute_linregress_consistency(vectorize=..., cache_full=..., target_batch_size=...)`),
which is how each was validated on its own.

### 1. Vectorized Pearson correlations

The original calls `scipy.stats.pearsonr` once per target voxel, three times
(r_xy, r_xx, r_yy), for both train and test, in every split of every bootstrap:

```python
# original (consistency_reference.py)
for n in range(Y_train.shape[1]):
    curr_train_res = get_consistency_per_neuron(
        X=Y_pred_train[:, n], Y=Y_train[:, n],
        X1=Y1_pred_train[:, n], X2=Y2_pred_train[:, n],
        Y1=Y1_train[:, n], Y2=Y2_train[:, n], metric=metric,
    )
```

The replacement computes all targets at once with a row-wise replica of
SciPy 1.17's own `pearsonr` algorithm. Matching the original to the last bit
took three details, each found by testing rather than assumed:

```python
def pearsonr_rows(x, y):
    ...
    # (a) C-contiguous rows, so NumPy reduces each row with the same pairwise
    #     summation it uses on the reference's 1-D inputs (a transposed view is
    #     summed in a different order and differs in the last bit for n = 11).
    x = np.ascontiguousarray(x, dtype=dtype)
    y = np.ascontiguousarray(y, dtype=dtype)
    ...  # constant-input NaNs, mean-centering, max-abs rescaling: as in SciPy
    # (b) Accelerate's BLAS dot rounds differently for inputs that are not
    #     16-byte aligned; SciPy's 1-D call always gets a fresh (aligned)
    #     array, so every row is copied onto a 64-byte boundary first.
    r = np.vecdot(_aligned_rows(xm / normxm), _aligned_rows(ym / normym), axis=-1)
    r = np.clip(r, -1.0, 1.0)
    r[const_xy] = np.nan
    if n == 2:
        r = np.round(r)  # SciPy returns exactly +/-1 for two points
    return r
```

```python
def consistency_columns(X, Y, X1, X2, Y1, Y2, r_xy=None):
    ...
    r_product = r_xx_sb * r_yy_sb
    undefined = (r_product < 0) | np.isclose(r_product, 0)
    # (c) np.power, not ``**``: array ``** 0.5`` takes a sqrt fast path that
    #     rounds differently from the scalar power the original evaluates.
    denom_sb = np.power(r_product, 0.5)
    r_xy_n_sb = r_xy / denom_sb
    if undefined.any():
        # The original stores Python float NaN for these units, which makes
        # np.array() of its list float64; reproduce that dtype.
        out_dtype = np.promote_types(denom_sb.dtype, np.float64)
        denom_sb = np.where(undefined, np.nan, denom_sb.astype(out_dtype))
        r_xy_n_sb = np.where(undefined, np.nan, r_xy_n_sb.astype(out_dtype))
```

Corrected correlations are **not** clipped. Values above 1, NaNs from negative
reliabilities and NaNs from constant predictions all come out exactly as
before. On SciPy versions other than 1.17 the code falls back to the original
per-target loop, which is exact by construction.

### 2. Fit the full-data mapping once per fold, not once per bootstrap

Every bootstrap in the original fits three mappings per fold. One of them, the
fit on the full trial averages, is identical in every bootstrap:

```python
# original: per bootstrap, per fold
neural_map_full.fit(X_train, Y_train)   # same inputs in every bootstrap
neural_map_1.fit(X1_train, Y1_train)
neural_map_2.fit(X2_train, Y2_train)    # all three maps alive at the end
```

The optimized version fits it once per fold during setup and keeps only its
predictions and the seed-independent `r_xy`. It also drops every fitted model
before the next fit starts:

```python
def _prepare(...):
    ...
    full_cache = [
        _full_data_entry(m, X, Y, s, vectorize, target_batch_size, counter)
        for m, s in zip(map_kwargs_list, splits)
    ]

def fit_predict(map_kwargs, X_train, Y_train, X_test, target_batch_size=0, counter=None):
    neural_map = PipelineNeuralMap(**map_kwargs)
    neural_map.fit(X_train, Y_train)
    pred_train = neural_map.predict(X_train)
    pred_test = neural_map.predict(X_test)
    del neural_map   # release the coefficient matrix before the next fit
```

For 50 bootstraps × 5 folds that is **505 fits instead of 750**; the fit
counter in the tests checks this. Caching is only used when refitting is
deterministic. Regressors with an unseeded or stateful `random_state` (e.g.
`selection="random"` without an int seed), non-Pearson metrics, RSA and unknown
map types go to the original implementation unchanged.

### 3. Optional target batching (off by default)

sklearn's `ElasticNet.fit` solves each target column independently, so the
targets can be fit in batches with only one batch of coefficients in memory:

```python
for start in range(0, n_targets, target_batch_size):
    stop = min(start + target_batch_size, n_targets)
    neural_map = PipelineNeuralMap(**map_kwargs)
    neural_map.fit(X_train, Y_train[:, start:stop])
    pred_train[:, start:stop] = neural_map.predict(X_train)
    pred_test[:, start:stop] = neural_map.predict(X_test)
    del neural_map
```

Enable it with `export BRAINMODEL_UTILS_TARGET_BATCH_SIZE=256` or
`target_batch_size=256`. It only applies to `ElasticNet`/`Lasso` without a
precomputed Gram matrix.

**It is not bitwise exact, which is why it is opt-in.** Coefficients and
iteration counts are bitwise identical to the unbatched fit (tested). The BLAS
matrix product in `predict` (`X @ coef_.T`) and the intercept, however, are
summed in a different order when the matrix is narrower. Predictions therefore
differ by about one float32 ULP, and the tests check them against the rigorous
reduction-order error bound. Most downstream metrics differ by about 1e-7
relative. The exception is ill-conditioned values: when a split-half
correlation is within 1e-6 of −1, the Spearman-Brown factor `2r/(1+r)` turns a
one-ULP change into a large absolute change in `r_xx_sb` (a difference of
6.8e5 on a value of −3.05e6, on a 2–3-point test fold). NaN masks never changed. Whether that is
acceptable is an analysis decision.

### Public API and rollback

`brainmodel_utils.metrics.consistency.get_linregress_consistency` keeps its
signature, defaults and return format. Every name the module used to define or
import is still re-exported. The implementation is chosen on each call:

```python
def active_implementation():
    impl = os.environ.get(IMPL_ENV, "").strip().lower() or "fast"
    ...
if active_implementation() == "reference":
    return _reference.get_linregress_consistency(**call_kwargs)
return _fast.get_linregress_consistency(**call_kwargs)
```

`export BRAINMODEL_UTILS_CONSISTENCY_IMPL=reference` restores the original
without reinstalling.

## Correctness results

| Check | Result |
|---|---|
| 13 synthetic variants (independent, strong/weak coupling, unequal trials/voxels, constant & near-constant voxels, negative reliability, NaN trials, skipped bootstraps, NaN outputs, corrected r > 1, float64) | bitwise identical |
| Saved reference outputs, primary dataset (40/40 trials, 14 timepoints, 128/128 voxels), 3 / 10 / 50 bootstraps | bitwise identical |
| Each stage alone (vectorize on/off × cache on/off) | bitwise identical |
| Ridge, percentile, PLS maps; xarray inputs; 2-D model features; default random splits; per-split `map_kwargs` with skipped splits; 2 joblib workers; seeded `selection="random"` | bitwise identical |
| Unmodified `autonomous_drosophila` `fit_tensors` (ElasticNet α=0.01, l1=0.25) | identical raw and corrected scores |
| Every benchmark run below (exact mode) | bitwise identical to the reference run |
| Full suite | **97 passed** (`BRAINMODEL_SLOW_TESTS=1 pytest test`) |

## Performance results

All tiers use 14 timepoints and the 5 production blocked folds. Wall time is
the median of 3 repeats where repeats > 1. *Cores* = CPU-seconds / wall time.
*Δpeak* = peak RSS minus RSS just before the call, so input data and library
imports are excluded.

### Clock time and CPU

**Production regression settings** (`max_iter=10000`), 128 source × 128 target
voxels, 40 trials. Here the regression dominates, so the gain mostly comes from
fitting fewer models:

| Variant | 3 bootstraps | speedup | CPU s | cores | fits |
|---|---:|---:|---:|---:|---:|
| reference | 65.1 s | 1.00× | 69.6 | 1.07 | 45 |
| vectorized Pearson only | 63.4 s | 1.03× | 67.3 | 1.06 | 45 |
| cache only | 48.5 s | 1.34× | 51.3 | 1.06 | 35 |
| **vectorized + cache (default)** | **46.7 s** | **1.39×** | 49.8 | 1.07 | 35 |
| + batching (64) | 46.4 s | 1.40× | 49.6 | 1.07 | 70 batch fits |

| Variant | 10 bootstraps | speedup | CPU s | cores | fits |
|---|---:|---:|---:|---:|---:|
| reference | 215.5 s | 1.00× | 229.3 | 1.06 | 150 |
| **default** | **133.3 s** | **1.62×** | 142.0 | 1.07 | 105 |
| + batching (64) | 133.0 s | 1.62× | 141.9 | 1.07 | 210 batch fits |

For reference, the original took 1,151 s for 50 bootstraps on this dataset.

**Fast-converging regression** (1,024 × 1,024 voxels, `max_iter=50`). When fits
are cheap, the per-voxel Pearson loop is a large share of the cost, and
vectorizing it alone gives 1.28×:

| Variant | 2 bootstraps | speedup | CPU s | cores |
|---|---:|---:|---:|---:|
| reference | 34.8 s | 1.00× | 35.0 | 1.00 |
| vectorized only | 27.1 s | 1.28× | 27.2 | 1.01 |
| cache only | 29.4 s | 1.18× | 29.5 | 1.00 |
| **default** | **22.4 s** | **1.55×** | 22.5 | 1.00 |
| batching 64 / 256 / 1024 | 22.2 / 22.2 / 22.3 s | 1.56–1.57× | ≈22.3 | 1.00 |

**Production source width** (27,118 source × 64 target voxels, 20 trials,
`max_iter=50`, 2 bootstraps): reference 27.3 s, default 23.7 s (**1.15×**).
With only 2 bootstraps the cache saves 5 of 30 fits; the saving grows with
bootstraps.

**Two joblib workers** (1,024 × 1,024, `max_iter=50`, 4 bootstraps):

| Variant | wall | speedup | CPU s | cores busy |
|---|---:|---:|---:|---:|
| reference | 33.3 s | 1.00× | 69.3 | 2.08 |
| default | 22.6 s | 1.47× | 41.6 | 1.84 |
| batched (256) | 22.2 s | 1.50× | 41.2 | 1.86 |

CPU utilization is slightly lower with the cache (1.84 vs. 2.08 cores) because
the 5 full-data fits run serially in the parent before the workers start. That
does not matter for total CPU, which is 40% lower, but it is a serial section
that grows in relative cost with many workers (see *Next steps*).

**Expected speedup at 50 bootstraps.** If the regression dominates, the floor
is the fit ratio, 750 / 505 = 1.49×. If fits converge quickly on real data,
expect more, as the Pearson savings add on top (1.55× at only 2 bootstraps in
the medium tier). The 10-bootstrap result (1.62×) already exceeds the fit ratio
for that bootstrap count (1.43×).

### RAM

**Memory tier**: 27,118 source × 4,096 target voxels, 20 trials, 1 bootstrap,
`max_iter=5`. One float32 coefficient matrix here is 444 MB.

| Variant | wall | Δpeak RSS | peak RSS (process) |
|---|---:|---:|---:|
| reference | 151.6 s | 1,705 MB | 2,028 MB |
| **default** | 136.5 s | **1,031 MB (−40%)** | 1,354 MB |
| batched 64 | 121.4 s | **78 MB (−95%)** | 401 MB |
| batched 256 | 125.9 s | **65 MB (−96%)** | 388 MB |
| batched 1024 | 133.2 s | 289 MB (−83%) | 612 MB |

Live NumPy allocations (`tracemalloc`, 27,118 × 1,024, one fold), in units of
one coefficient matrix:

| Variant | peak live allocations |
|---|---:|
| reference | 4.35 matrices |
| default | 2.32 matrices (−47%) |
| batched 256 | 0.64 matrices (−85%) |

Why these numbers: sklearn's `ElasticNet.fit` itself transiently holds about
2.26 matrices (`_set_intercept` divides `coef_` into a new array, plus a
boolean finiteness check). The original keeps two finished maps alive while
fitting the third, giving 2 + 2.26 ≈ 4.3. The optimized path only ever has the
fit in progress (≈ 2.3). Batching scales that down to one batch.

**At production size** (27,118 × 27,118; one float32 matrix = 2.74 GiB), per
worker:

| Mode | float32 | float64 |
|---|---:|---:|
| reference (≈4.35 matrices) | ≈ 11.9 GiB | ≈ 23.8 GiB |
| default (≈2.3 matrices) | ≈ 6.4 GiB | ≈ 12.7 GiB |
| batched 256 (≈2.3 × 256 / 27,118 matrices) | ≈ 0.06 GiB + data | ≈ 0.12 GiB + data |

These are projections from the measured ratios. Node-level memory has to be
measured on the cluster (Phase 7).

On the small and medium tiers coefficient matrices are tiny (≤ 4 MB), so peak
RSS is the same for every variant (≈ 150–175 MB, dominated by imports and
data). With two workers the process tree peaks at about 550–575 MB for all
variants, again dominated by each worker's own imports.

## Caveats

- **Synthetic data.** All results come from synthetic data. Convergence speed,
  and therefore the share of time spent in regression vs. correlations, depends
  on the real data.
- **Platform.** Bitwise equality was established on macOS/Accelerate. Linux
  OpenBLAS may round differently, so run the test suite on the cluster first; it
  compares both implementations live.
- **Laptop tier settings.** The medium, wide and memory tiers use a reduced
  `max_iter` so they finish on a laptop. Memory and fit counts do not depend on
  it; time per fit does.
- **Swap.** The M1 was under memory pressure from other applications during the
  runs (≈ 8 GB swap in use). Per-run swap deltas were recorded and were small or
  negative for the timed runs.
- **Warning counts.** ConvergenceWarnings from the full-data fit are emitted
  once per fold instead of once per bootstrap. Python shows each warning
  location once by default either way.

## Next steps

1. Cluster validation per [DEPLOYMENT.md](DEPLOYMENT.md): run the test suite,
   re-run `fly_098 → fly_099` (right) with both implementations and compare
   every array, then measure node-level RAM and comparisons per node-hour at 1,
   2 and 4 concurrent comparisons.
2. Optional: fit the 5 full-data maps in parallel when `num_parallel_jobs > 1`.
   They currently run serially before the bootstrap workers start. With 16
   workers and 50 bootstraps that serial section reduces the ideal 1.49× to
   about 1.30×; parallelizing it would recover most of that, with identical
   results.

## Reproducing

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e . numpy==2.0.2 scipy==1.17.1 \
    scikit-learn==1.8.0 joblib==1.5.3 threadpoolctl==3.6.0 xarray pytest psutil
.venv/bin/python -m pytest -q test                         # ~6 min
BRAINMODEL_SLOW_TESTS=1 .venv/bin/python -m pytest -q test # ~20 min
benchmarks/run_m1_suite.sh                                 # ~60 min; writes benchmarks/results/
```
