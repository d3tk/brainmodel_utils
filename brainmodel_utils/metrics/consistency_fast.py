"""Optimized ``get_linregress_consistency`` that reproduces the reference results.

Statistical methodology is unchanged: same trial averages, split halves,
bootstrap seeds, cross-validation splits, regression objective and
Spearman-Brown noise correction as ``consistency_reference``. Three changes
make it cheaper:

1. Vectorized Pearson correlations. Per-target ``scipy.stats.pearsonr`` calls
   are replaced by a row-wise replica of SciPy's own algorithm. Each step uses
   the same NumPy reductions, and the final dot product runs on 64-byte aligned
   rows. That matters because BLAS ``dot`` kernels round misaligned inputs
   differently, and the reference always passes freshly allocated (aligned) 1-D
   arrays. Output is bitwise identical to the per-target loop on the verified
   SciPy versions; other versions fall back to that loop.

2. Full-data mapping cache. The full trial-average mapping does not depend on
   the bootstrap seed, so it is fit once per split rather than once per
   bootstrap. Only its predictions (and the seed-independent ``r_xy``) are kept.
   Every fitted map is released before the next one is fit, so at most one
   coefficient matrix is alive at a time (the reference holds three).

3. Optional target batching (off by default). ``ElasticNet``/``Lasso`` fit each
   target column independently, so fitting column batches reproduces the
   coefficients bitwise and only one batch of coefficients is ever in memory.
   Predictions can still differ from the unbatched fit by about one ULP,
   because the BLAS matrix product rounds differently when the matrix width
   changes. Enable with ``target_batch_size=`` or the
   ``BRAINMODEL_UTILS_TARGET_BATCH_SIZE`` environment variable.

Any configuration outside the verified fast path (non-Pearson metrics, RSA,
stochastic regressors, unknown map types, inputs the reference rejects)
delegates to ``consistency_reference`` unchanged.
"""

import os
import warnings

import numpy as np
import scipy
import xarray as xr
from joblib import Parallel, delayed
from scipy import stats

from brainmodel_utils.core.utils import make_list
from brainmodel_utils.metrics import consistency_reference as _reference
from brainmodel_utils.metrics.consistency_reference import sb_correction
from brainmodel_utils.metrics.utils import (
    concat_dict_sp,
    generic_trial_avg,
    get_splithalves,
    sphalf_input_checker,
)
from brainmodel_utils.neural_mappers import PipelineNeuralMap
from brainmodel_utils.neural_mappers.utils import generate_train_test_splits

try:  # the norm SciPy's pearsonr uses internally
    from scipy._lib._array_api import xp_vector_norm as _scipy_vector_norm
except ImportError:  # pragma: no cover - other SciPy layouts use the loop
    _scipy_vector_norm = None

TARGET_BATCH_SIZE_ENV = "BRAINMODEL_UTILS_TARGET_BATCH_SIZE"

# Key order of the reference result dicts.
METRIC_NAMES = ("r_xy_n_sb", "r_xx", "r_xx_sb", "r_yy", "r_yy_sb", "r_xy", "denom_sb")

# SciPy releases whose pearsonr statistic the row-wise replica was verified
# against bitwise (test/test_consistency_fast.py).
_VERIFIED_SCIPY = ((1, 17),)
_SCIPY_VERSION = tuple(int(p) for p in scipy.__version__.split(".")[:2])
VECTORIZED_PEARSON_AVAILABLE = (
    _SCIPY_VERSION in _VERIFIED_SCIPY
    and hasattr(np, "vecdot")
    and _scipy_vector_norm is not None
)

# sklearn estimators that are deterministic given fixed inputs (modulo the
# random_state checks in _is_deterministic).
_DETERMINISTIC_REGRESSIONS = frozenset(
    {
        "Ridge",
        "RidgeCV",
        "ElasticNet",
        "Lasso",
        "LinearRegression",
        "MultiTaskElasticNet",
        "MultiTaskLasso",
        "Lars",
        "LassoLars",
        "BayesianRidge",
    }
)
_KNOWN_MAP_TYPES = frozenset({"sklinear", "percentile", "pls", "identity"})
# Fit every target column independently (see sklearn ElasticNet.fit).
_BATCHABLE_REGRESSIONS = frozenset({"ElasticNet", "Lasso"})
_ALIGN_BYTES = 64


# ---------------------------------------------------------------------------
# Vectorized Pearson correlation
# ---------------------------------------------------------------------------
def _aligned_rows(a):
    """Copy 2-D ``a`` into a buffer whose rows each start on a 64-byte boundary."""
    n_rows, n_cols = a.shape
    isz = a.itemsize
    step = -(-n_cols * isz // _ALIGN_BYTES) * _ALIGN_BYTES // isz
    raw = np.empty(n_rows * step + _ALIGN_BYTES // isz, dtype=a.dtype)
    offset = (-raw.ctypes.data % _ALIGN_BYTES) // isz
    buf = raw[offset : offset + n_rows * step].reshape(n_rows, step)
    buf[:, :n_cols] = a
    return buf[:, :n_cols]


def pearsonr_rows(x, y):
    """``scipy.stats.pearsonr(x[i], y[i]).statistic`` for every row ``i``.

    Mirrors SciPy 1.17's implementation step by step (constant-input NaNs,
    max-abs rescaling, clipping, exact +/-1 for two observations) without the
    p-value computation.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    dtype = np.result_type(x.dtype, y.dtype)
    if not np.issubdtype(dtype, np.inexact):
        dtype = np.float64
    if x.shape[-1] != y.shape[-1]:
        raise ValueError("`x` and `y` must have the same length along `axis`.")
    n = x.shape[-1]
    if n < 2:
        raise ValueError("`x` and `y` must have length at least 2.")
    # C-contiguous rows: NumPy then reduces each row with the same pairwise
    # summation it applies to the reference's 1-D per-target inputs. A strided
    # (e.g. transposed) layout is reduced in a different order.
    x = np.ascontiguousarray(x, dtype=dtype)
    y = np.ascontiguousarray(y, dtype=dtype)
    threshold = np.finfo(dtype).eps ** 0.75

    const_x = np.all(x == x[..., 0:1], axis=-1)
    const_y = np.all(y == y[..., 0:1], axis=-1)
    const_xy = const_x | const_y
    if np.any(const_xy):
        warnings.warn(
            stats.ConstantInputWarning(
                "An input array is constant; the correlation coefficient is not defined."
            ),
            stacklevel=3,
        )
        x = np.where(const_x[..., np.newaxis], np.nan, x)
        y = np.where(const_y[..., np.newaxis], np.nan, y)

    xmean = np.mean(x, axis=-1, keepdims=True)
    ymean = np.mean(y, axis=-1, keepdims=True)
    xm = x - xmean
    ym = y - ymean
    xmax = np.max(np.abs(xm), axis=-1, keepdims=True)
    ymax = np.max(np.abs(ym), axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        normxm = xmax * _scipy_vector_norm(xm / xmax, axis=-1, keepdims=True, xp=np)
        normym = ymax * _scipy_vector_norm(ym / ymax, axis=-1, keepdims=True, xp=np)

    nconst_x = np.any(normxm < threshold * np.abs(xmean), axis=-1)
    nconst_y = np.any(normym < threshold * np.abs(ymean), axis=-1)
    if np.any((nconst_x | nconst_y) & ~const_xy):
        warnings.warn(
            stats.NearConstantInputWarning(
                "An input array is nearly constant; the computed "
                "correlation coefficient may be inaccurate."
            ),
            stacklevel=3,
        )

    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.vecdot(_aligned_rows(xm / normxm), _aligned_rows(ym / normym), axis=-1)
    r = np.clip(r, -1.0, 1.0)
    r[const_xy] = np.nan
    if n == 2:
        r = np.round(r)
    return r


def _pearsonr_columns(a, b):
    """Pearson r between matching columns of two (observations, units) arrays."""
    return pearsonr_rows(np.asarray(a).T, np.asarray(b).T)


def consistency_columns(X, Y, X1, X2, Y1, Y2, r_xy=None):
    """Vectorized ``get_consistency_per_neuron`` over columns (metric="pearsonr").

    Inputs are (observations, units). ``r_xy`` may be supplied when it is
    already known (it does not depend on the split-half seed). Values and
    dtypes match ``np.array`` of the reference per-unit results.
    """
    # Same checks as sphalf_input_checker(dim_val=1) applied to every column.
    assert X.shape[0] == Y.shape[0]
    assert X.shape == X1.shape == X2.shape
    assert Y.shape == Y1.shape == Y2.shape
    assert np.isfinite(X1).all() and np.isfinite(X2).all()
    assert np.isfinite(Y1).all() and np.isfinite(Y2).all()

    if r_xy is None:
        r_xy = _pearsonr_columns(X, Y)
    r_xx = _pearsonr_columns(X1, X2)
    r_xx_sb = sb_correction(r_xx)
    r_yy = _pearsonr_columns(Y1, Y2)
    r_yy_sb = sb_correction(r_yy)

    r_product = r_xx_sb * r_yy_sb
    undefined = (r_product < 0) | np.isclose(r_product, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        # np.power, not ``**``: array ``** 0.5`` takes a sqrt fast path that can
        # round differently from the scalar power the reference evaluates.
        denom_sb = np.power(r_product, 0.5)
        r_xy_n_sb = r_xy / denom_sb
    if undefined.any():
        # The reference stores the Python float ``np.nan`` for these units, so
        # np.array() of its per-unit list promotes to float64.
        out_dtype = np.promote_types(denom_sb.dtype, np.float64)
        denom_sb = np.where(undefined, np.nan, denom_sb.astype(out_dtype))
        r_xy_n_sb = np.where(undefined, np.nan, r_xy_n_sb.astype(out_dtype))

    return {
        "r_xy_n_sb": r_xy_n_sb,
        "r_xx": r_xx,
        "r_xx_sb": r_xx_sb,
        "r_yy": r_yy,
        "r_yy_sb": r_yy_sb,
        "r_xy": r_xy,
        "denom_sb": denom_sb,
    }


def _consistency_columns_loop(X, Y, X1, X2, Y1, Y2):
    """Reference per-unit loop (used when the vectorized replica is unverified)."""
    out = {k: [] for k in METRIC_NAMES}
    for n in range(Y.shape[1]):
        res = _reference.get_consistency_per_neuron(
            X=X[:, n], Y=Y[:, n], X1=X1[:, n], X2=X2[:, n], Y1=Y1[:, n], Y2=Y2[:, n],
            metric="pearsonr",
        )
        for k in METRIC_NAMES:
            out[k].append(res[k])
    return {k: np.array(v) for k, v in out.items()}


# ---------------------------------------------------------------------------
# Mapping fits
# ---------------------------------------------------------------------------
def _is_stateful_rng(value):
    return isinstance(value, (np.random.RandomState, np.random.Generator))


def _regression_is_deterministic(inner):
    regression_type = inner.get("regression_type", "Ridge")
    if regression_type not in _DETERMINISTIC_REGRESSIONS:
        return False
    kw = inner.get("regression_kwargs") or {}
    random_state = kw.get("random_state")
    if _is_stateful_rng(random_state):
        return False
    unseeded = random_state is None
    if kw.get("selection") == "random" and unseeded:
        return False
    if kw.get("solver") in ("sag", "saga") and unseeded:
        return False
    return True


def _is_deterministic(map_kwargs):
    """True when refitting a map on the same data reproduces the same fit.

    Only then may the full-data fit be shared across bootstraps.
    """
    if map_kwargs is None:
        return True
    if not isinstance(map_kwargs, dict):
        return False
    map_type = str(map_kwargs.get("map_type", "")).lower()
    if map_type not in _KNOWN_MAP_TYPES:
        return False
    if map_type in ("sklinear", "percentile"):
        return _regression_is_deterministic(map_kwargs.get("map_kwargs") or {})
    return True


def _is_batchable(map_kwargs):
    if str(map_kwargs.get("map_type", "")).lower() != "sklinear":
        return False
    if map_kwargs.get("factor_kwargs") is not None:
        return False
    inner = map_kwargs.get("map_kwargs") or {}
    if inner.get("weights_save_nm") is not None:
        return False
    if inner.get("regression_type", "Ridge") not in _BATCHABLE_REGRESSIONS:
        return False
    kw = inner.get("regression_kwargs") or {}
    # A precomputed X^T y would be computed per batch by BLAS; keep unbatched.
    return kw.get("precompute", False) is False


def resolve_target_batch_size(target_batch_size=None):
    if target_batch_size is None:
        target_batch_size = os.environ.get(TARGET_BATCH_SIZE_ENV, "").strip() or 0
    target_batch_size = int(target_batch_size)
    if target_batch_size < 0:
        raise ValueError("target_batch_size must be >= 0 (0 disables batching)")
    return target_batch_size


def fit_predict(map_kwargs, X_train, Y_train, X_test, target_batch_size=0, counter=None):
    """Fit ``PipelineNeuralMap(**map_kwargs)`` and return (train, test) predictions.

    The fitted map is dropped before returning. With ``target_batch_size`` > 0
    and a batchable regression, targets are fit in column batches and only one
    batch of coefficients exists at a time.
    """
    n_targets = Y_train.shape[1]
    if target_batch_size and n_targets > target_batch_size and _is_batchable(map_kwargs):
        Y_train = np.asarray(Y_train)
        pred_train = pred_test = None
        for start in range(0, n_targets, target_batch_size):
            stop = min(start + target_batch_size, n_targets)
            neural_map = PipelineNeuralMap(**map_kwargs)
            neural_map.fit(X_train, Y_train[:, start:stop])
            p_train = neural_map.predict(X_train)
            p_test = neural_map.predict(X_test)
            del neural_map
            if counter is not None:
                counter["fits"] += 1
            if pred_train is None:
                pred_train = np.empty((p_train.shape[0], n_targets), dtype=p_train.dtype)
                pred_test = np.empty((p_test.shape[0], n_targets), dtype=p_test.dtype)
            pred_train[:, start:stop] = p_train
            pred_test[:, start:stop] = p_test
    else:
        neural_map = PipelineNeuralMap(**map_kwargs)
        neural_map.fit(X_train, Y_train)
        pred_train = neural_map.predict(X_train)
        pred_test = neural_map.predict(X_test)
        del neural_map
        if counter is not None:
            counter["fits"] += 1
    assert pred_train.shape == Y_train.shape
    assert pred_test.shape == (X_test.shape[0], n_targets)
    return pred_train, pred_test


# ---------------------------------------------------------------------------
# Per-split / per-bootstrap computation
# ---------------------------------------------------------------------------
def _full_data_entry(map_kwargs, X, Y, split, vectorize, target_batch_size, counter):
    """Seed-independent quantities for one split: full-data predictions and r_xy."""
    if map_kwargs is None:
        return None
    assert isinstance(map_kwargs, dict)
    train_idx, test_idx = split["train"], split["test"]
    Y_train, Y_test = np.asarray(Y[train_idx]), np.asarray(Y[test_idx])
    pred_train, pred_test = fit_predict(
        map_kwargs, X[train_idx], Y[train_idx], X[test_idx], target_batch_size, counter
    )
    entry = {
        "Y_train": Y_train,
        "Y_test": Y_test,
        "pred_train": pred_train,
        "pred_test": pred_test,
    }
    if vectorize:
        entry["r_xy_train"] = _pearsonr_columns(pred_train, Y_train)
        entry["r_xy_test"] = _pearsonr_columns(pred_test, Y_test)
    return entry


def _persplit(X, Y, X1, X2, Y1, Y2, map_kwargs, split, full, vectorize,
              target_batch_size, counter):
    if map_kwargs is None:
        return None
    sphalf_input_checker(X=X, Y=Y, X1=X1, X2=X2, Y1=Y1, Y2=Y2)
    assert isinstance(map_kwargs, dict)
    train_idx, test_idx = split["train"], split["test"]
    if full is None:  # uncached mode: refit the full-data map for this bootstrap
        full = _full_data_entry(map_kwargs, X, Y, split, vectorize, target_batch_size, counter)

    p1_train, p1_test = fit_predict(
        map_kwargs, X1[train_idx], Y1[train_idx], X1[test_idx], target_batch_size, counter
    )
    p2_train, p2_test = fit_predict(
        map_kwargs, X2[train_idx], Y2[train_idx], X2[test_idx], target_batch_size, counter
    )
    Y1_train, Y1_test = np.asarray(Y1[train_idx]), np.asarray(Y1[test_idx])
    Y2_train, Y2_test = np.asarray(Y2[train_idx]), np.asarray(Y2[test_idx])

    out = {}
    for part, Yp, p1, p2, Y1p, Y2p in (
        ("train", full["Y_train"], p1_train, p2_train, Y1_train, Y2_train),
        ("test", full["Y_test"], p1_test, p2_test, Y1_test, Y2_test),
    ):
        pred = full[f"pred_{part}"]
        if vectorize:
            out[part] = consistency_columns(
                pred, Yp, p1, p2, Y1p, Y2p, r_xy=full[f"r_xy_{part}"]
            )
        else:
            out[part] = _consistency_columns_loop(pred, Yp, p1, p2, Y1p, Y2p)
    return out


def _persphalftrial(source, target, X, Y, splits, map_kwargs_list, sphseed,
                    full_cache, vectorize, target_batch_size, counter=None):
    X1, X2 = get_splithalves(source, seed=sphseed)
    if (not np.isfinite(X1).all()) or (not np.isfinite(X2).all()):
        return None
    Y1, Y2 = get_splithalves(target, seed=sphseed)
    if (not np.isfinite(Y1).all()) or (not np.isfinite(Y2).all()):
        return None
    results_arr = [
        _persplit(
            X, Y, X1, X2, Y1, Y2,
            map_kwargs=map_kwargs_list[s_idx],
            split=split,
            full=None if full_cache is None else full_cache[s_idx],
            vectorize=vectorize,
            target_batch_size=target_batch_size,
            counter=counter,
        )
        for s_idx, split in enumerate(splits)
    ]
    return concat_dict_sp(results_arr)


def _fast_path_supported(source, metric, map_kwargs, kwargs):
    if metric != "pearsonr":
        return False
    if set(kwargs) - {"num_train_test_splits", "train_frac"}:
        return False  # the reference raises TypeError for these
    if getattr(source, "ndim", None) not in (2, 3):
        return False
    configs = map_kwargs if isinstance(map_kwargs, list) else [map_kwargs]
    return all(_is_deterministic(m) for m in configs)


def _prepare(source, target, map_kwargs, splits, num_train_test_splits, train_frac,
             vectorize, cache_full, target_batch_size, counter):
    """Seed-independent setup: trial averages, splits and the full-data cache."""
    if source.ndim == 3:
        X = generic_trial_avg(source)
    else:
        assert source.ndim == 2
        X = source
    assert np.isfinite(X).all()
    assert target.ndim == 3
    Y = generic_trial_avg(target)
    assert np.isfinite(Y).all()

    if splits is None:
        splits = generate_train_test_splits(
            num_stim=X.shape[0], num_splits=num_train_test_splits, train_frac=train_frac
        )
    assert isinstance(splits, list)
    if not isinstance(map_kwargs, list):
        assert isinstance(map_kwargs, dict)
        map_kwargs = make_list(map_kwargs, num_times=len(splits))
    map_kwargs_list = [map_kwargs[s_idx] for s_idx in range(len(splits))]

    full_cache = None
    if cache_full:
        full_cache = [
            _full_data_entry(m, X, Y, s, vectorize, target_batch_size, counter)
            for m, s in zip(map_kwargs_list, splits)
        ]
    return dict(
        source=source, target=target, X=X, Y=Y, splits=splits,
        map_kwargs_list=map_kwargs_list, full_cache=full_cache,
        vectorize=vectorize, target_batch_size=target_batch_size,
    )


def _run(task_kwargs, seeds, num_parallel_jobs, counter):
    if num_parallel_jobs == 1:
        results_arr = [
            _persphalftrial(sphseed=seed, counter=counter, **task_kwargs) for seed in seeds
        ]
    else:
        results_arr = Parallel(n_jobs=num_parallel_jobs)(
            delayed(_persphalftrial)(sphseed=seed, **task_kwargs) for seed in seeds
        )
    target = task_kwargs["target"]
    if isinstance(target, xr.DataArray):
        return concat_dict_sp(
            results_arr,
            xarray_target=target,
            xarray_dims=["trial_bootstrap_iters", "train_test_splits", "units"],
        )
    return concat_dict_sp(results_arr)


def compute_linregress_consistency(
    source,
    target,
    map_kwargs,
    num_bootstrap_iters=1000,
    num_parallel_jobs=1,
    start_seed=1234,
    metric="pearsonr",
    splits=None,
    num_train_test_splits=5,
    train_frac=0.8,
    vectorize=None,
    cache_full=True,
    target_batch_size=0,
    counter=None,
):
    """Optimized core without the reference fallback.

    ``vectorize``/``cache_full``/``target_batch_size`` toggle the individual
    optimizations (staged benchmarks and tests). ``counter`` is an optional dict
    whose "fits" entry counts regression fits (single process only).
    """
    assert metric == "pearsonr"
    if vectorize is None:
        vectorize = VECTORIZED_PEARSON_AVAILABLE
    task_kwargs = _prepare(
        source, target, map_kwargs, splits, num_train_test_splits, train_frac,
        vectorize, cache_full, target_batch_size, counter,
    )
    seeds = range(start_seed, start_seed + num_bootstrap_iters)
    return _run(task_kwargs, seeds, num_parallel_jobs, counter)


def get_linregress_consistency(
    source,
    target,
    map_kwargs,
    num_bootstrap_iters=1000,
    num_parallel_jobs=1,
    start_seed=1234,
    metric="pearsonr",
    splits=None,
    target_batch_size=None,
    **kwargs
):
    """Drop-in replacement for the reference ``get_linregress_consistency``.

    Same arguments and return format (see ``consistency_reference``), plus
    ``target_batch_size`` (default: ``$BRAINMODEL_UTILS_TARGET_BATCH_SIZE``,
    else 0 = no batching).
    """
    reference_kwargs = dict(
        source=source,
        target=target,
        map_kwargs=map_kwargs,
        num_bootstrap_iters=num_bootstrap_iters,
        num_parallel_jobs=num_parallel_jobs,
        start_seed=start_seed,
        metric=metric,
        splits=splits,
        **kwargs
    )
    if num_bootstrap_iters <= 0 or not _fast_path_supported(source, metric, map_kwargs, kwargs):
        return _reference.get_linregress_consistency(**reference_kwargs)
    batch = resolve_target_batch_size(target_batch_size)
    try:
        task_kwargs = _prepare(
            source, target, map_kwargs, splits,
            kwargs.get("num_train_test_splits", 5), kwargs.get("train_frac", 0.8),
            VECTORIZED_PEARSON_AVAILABLE, True, batch, None,
        )
    except Exception:
        # Setup the reference would handle differently (it validates inside each
        # bootstrap and may never reach a failing full-data fit): defer to it so
        # behaviour, including any exception raised, is unchanged.
        return _reference.get_linregress_consistency(**reference_kwargs)
    seeds = range(start_seed, start_seed + num_bootstrap_iters)
    return _run(task_kwargs, seeds, num_parallel_jobs, None)
