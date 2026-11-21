"""Numerical-equivalence tests: optimized vs. reference consistency.

The unbatched optimized path is required to be *bitwise* identical to the
reference (values, NaN masks, shapes, dtypes, key order). Target batching is
opt-in and checked against documented tolerances, because BLAS rounds the
batched prediction products differently (see consistency_fast docstring).

Slow tests (10 and 50 bootstraps on the primary dataset) run when
BRAINMODEL_SLOW_TESTS=1.
"""

import os
import warnings

import numpy as np
import pytest
import xarray as xr
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNet

import make_reference_fixtures as fixtures
import synthetic_consistency_data as sd
from brainmodel_utils.metrics import consistency_fast as F
from brainmodel_utils.metrics import consistency_reference as R
from brainmodel_utils.neural_mappers import PipelineNeuralMap

SPLITS = sd.make_blocked_splits()
SLOW = os.environ.get("BRAINMODEL_SLOW_TESTS") == "1"
needs_vectorized = pytest.mark.skipif(
    not F.VECTORIZED_PEARSON_AVAILABLE,
    reason="vectorized Pearson replica not verified for this SciPy",
)

pytestmark = pytest.mark.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def assert_results_identical(ref, opt):
    assert list(ref) == list(opt) == ["train", "test"]
    for part in ref:
        assert list(ref[part]) == list(opt[part]), part
        for key in ref[part]:
            a, b = ref[part][key], opt[part][key]
            assert type(a) is type(b), (part, key)
            assert a.shape == b.shape, (part, key)
            assert a.dtype == b.dtype, (part, key, a.dtype, b.dtype)
            np.testing.assert_array_equal(np.isnan(a), np.isnan(b), err_msg=f"{part}/{key} NaN mask")
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=f"{part}/{key}")
            if isinstance(a, xr.DataArray):
                assert a.dims == b.dims
                assert a.coords.to_dataset().equals(b.coords.to_dataset())


def assert_results_close(ref, opt, rtol, atol, nan_mask_tolerance=0.0):
    """Tolerance comparison; returns the max |difference| over finite values."""
    worst = 0.0
    for part in ref:
        assert list(ref[part]) == list(opt[part])
        for key in ref[part]:
            a, b = np.asarray(ref[part][key]), np.asarray(opt[part][key])
            assert a.shape == b.shape and a.dtype == b.dtype, (part, key)
            mask_mismatch = np.mean(np.isnan(a) != np.isnan(b))
            assert mask_mismatch <= nan_mask_tolerance, (part, key, mask_mismatch)
            both = np.isfinite(a) & np.isfinite(b)
            np.testing.assert_allclose(b[both], a[both], rtol=rtol, atol=atol, err_msg=f"{part}/{key}")
            np.testing.assert_array_equal(np.isinf(a), np.isinf(b))
            if both.any():
                worst = max(worst, float(np.max(np.abs(a[both].astype(np.float64) - b[both]))))
    return worst


def run_both(source, target, boots=2, map_kwargs=None, **kwargs):
    kw = dict(
        source=source,
        target=target,
        map_kwargs=sd.enet_map_kwargs() if map_kwargs is None else map_kwargs,
        num_bootstrap_iters=boots,
        start_seed=42,
        num_parallel_jobs=1,
        metric="pearsonr",
    )
    kw.setdefault("splits", SPLITS)
    kw.update(kwargs)
    return R.get_linregress_consistency(**kw), F.get_linregress_consistency(**kw)


class FitCounter:
    """Counts PipelineNeuralMap.fit calls (the reference has no counter hook)."""

    def __init__(self, monkeypatch):
        self.n = 0
        original = PipelineNeuralMap.fit

        def counting_fit(map_self, *args, **kwargs):
            self.n += 1
            return original(map_self, *args, **kwargs)

        monkeypatch.setattr(PipelineNeuralMap, "fit", counting_fit)


# ---------------------------------------------------------------------------
# Phase 1: synthetic data and splits
# ---------------------------------------------------------------------------
def test_blocked_splits_match_production():
    sizes = [len(s["test"]) for s in SPLITS]
    assert sizes == [3, 3, 3, 3, 2]
    for s in SPLITS:
        assert np.all(np.diff(s["test"]) == 1)
        assert sorted(np.concatenate([s["train"], s["test"]]).tolist()) == list(range(14))


def test_primary_dataset_spec():
    source, target = sd.primary()
    assert source.shape == (40, 14, 128) and target.shape == (40, 14, 128)
    assert source.dtype == target.dtype == np.float32
    assert len(sd.RELATIVE_TIMES) == 14
    assert sd.RELATIVE_TIMES[0] == -15.0 and sd.RELATIVE_TIMES[-1] == -8.0
    s2, t2 = sd.primary()
    np.testing.assert_array_equal(source, s2)
    np.testing.assert_array_equal(target, t2)


@pytest.mark.parametrize("name", list(sd.VARIANTS))
def test_variants_deterministic(name):
    a = sd.VARIANTS[name]()
    b = sd.VARIANTS[name]()
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_reference_reproduces_saved_fixture():
    path = fixtures.fixture_path(3)
    if not path.exists():
        pytest.skip("run test/make_reference_fixtures.py --boots 3")
    saved, meta = fixtures.load(path)
    again = R.get_linregress_consistency(**fixtures.run_kwargs(3))
    if meta["env"] == fixtures.environment():
        assert_results_identical(saved, again)
    else:
        assert_results_close(saved, again, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Phase 2: vectorized Pearson
# ---------------------------------------------------------------------------
def _pearson_cases():
    rng = np.random.default_rng(0)
    for dtype in (np.float32, np.float64):
        for n_obs in (2, 3, 4, 7, 8, 9, 11, 12, 14, 17, 40):
            n_units = 97
            scale = rng.uniform(1e-3, 1e3, size=n_units)
            offset = rng.normal(size=n_units) * 1e3
            a = rng.normal(size=(n_obs, n_units)) * scale + offset
            b = 0.4 * a + rng.normal(size=(n_obs, n_units)) * scale
            a[:, 0] = 2.5  # constant
            b[:, 1] = -1.0
            a[:, 2] = 1e4 + 1e-9 * rng.normal(size=n_obs)  # nearly constant
            b[:, 3] = -a[:, 3]  # r = -1
            b[:, 4] = a[:, 4]  # r = 1
            yield dtype, a.astype(dtype), b.astype(dtype)


@needs_vectorized
def test_pearsonr_rows_bitwise_matches_scipy():
    for dtype, a, b in _pearson_cases():
        ref = np.array([pearsonr(a[:, n], b[:, n])[0] for n in range(a.shape[1])])
        got = F._pearsonr_columns(a, b)
        assert got.dtype == ref.dtype == dtype
        np.testing.assert_array_equal(got, ref, err_msg=f"{dtype.__name__} n_obs={a.shape[0]}")


@needs_vectorized
def test_pearsonr_rows_layout_independent():
    """Strided, transposed, sliced and offset inputs all give the same bits."""
    rng = np.random.default_rng(1)
    a = rng.normal(size=(11, 64)).astype(np.float32)
    b = rng.normal(size=(11, 64)).astype(np.float32)
    ref = np.array([pearsonr(a[:, n], b[:, n])[0] for n in range(64)])
    big = np.zeros((11, 200), np.float32)
    big[:, 3:67] = a
    np.testing.assert_array_equal(F._pearsonr_columns(big[:, 3:67], b), ref)
    np.testing.assert_array_equal(F._pearsonr_columns(np.asfortranarray(a), b), ref)
    np.testing.assert_array_equal(F.pearsonr_rows(np.ascontiguousarray(a.T), b.T), ref)


def test_pearsonr_rows_rejects_single_observation():
    with pytest.raises(ValueError):
        F.pearsonr_rows(np.ones((3, 1)), np.ones((3, 1)))


def _per_unit_reference(X, Y, X1, X2, Y1, Y2):
    out = {k: [] for k in F.METRIC_NAMES}
    for n in range(Y.shape[1]):
        res = R.get_consistency_per_neuron(
            X=X[:, n], Y=Y[:, n], X1=X1[:, n], X2=X2[:, n], Y1=Y1[:, n], Y2=Y2[:, n]
        )
        assert list(res) == list(F.METRIC_NAMES)
        for k in out:
            out[k].append(res[k])
    return {k: np.array(v) for k, v in out.items()}


@needs_vectorized
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n_obs", [2, 3, 11])
def test_consistency_columns_bitwise(dtype, n_obs):
    rng = np.random.default_rng(n_obs)
    n = 300
    base = rng.normal(size=(n_obs, n))
    arrs = [base * rng.uniform(0, 1.5) + rng.normal(size=(n_obs, n)) for _ in range(6)]
    X, Y, X1, X2, Y1, Y2 = [x.astype(dtype) for x in arrs]
    X2[:, :20] = -X1[:, :20]  # r_xx = -1 -> SB correction -inf
    Y2[:, 20:40] = Y1[:, 20:40]  # r_yy = 1
    X1[:, 40:50] = 0.5  # constant -> NaN
    Y2[:, 50:60] = -Y1[:, 50:60] + 1e-3 * rng.normal(size=(n_obs, 10)).astype(dtype)
    ref = _per_unit_reference(X, Y, X1, X2, Y1, Y2)
    got = F.consistency_columns(X, Y, X1, X2, Y1, Y2)
    assert list(got) == list(ref)
    for k in ref:
        assert got[k].dtype == ref[k].dtype, k
        np.testing.assert_array_equal(got[k], ref[k], err_msg=k)


def test_consistency_columns_all_defined_keeps_input_dtype():
    """No undefined units -> the reference keeps float32 for denom/r_xy_n_sb."""
    rng = np.random.default_rng(3)
    base = rng.normal(size=(11, 20))
    arrs = [(base + 0.05 * rng.normal(size=base.shape)).astype(np.float32) for _ in range(6)]
    ref = _per_unit_reference(*arrs)
    got = F.consistency_columns(*arrs)
    assert ref["denom_sb"].dtype == got["denom_sb"].dtype == np.float32
    for k in ref:
        np.testing.assert_array_equal(got[k], ref[k])


# ---------------------------------------------------------------------------
# End-to-end equivalence on every synthetic variant
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", [v for v in sd.VARIANTS if v != "primary"])
def test_variant_bitwise_equivalence(name):
    source, target = sd.VARIANTS[name]()
    boots = 6 if name == "nan_trials_skipping" else 2
    ref, opt = run_both(source, target, boots=boots)
    assert_results_identical(ref, opt)


def test_skipped_bootstraps_reproduced():
    source, target = sd.nan_trials_skipping()
    ref, opt = run_both(source, target, boots=6)
    assert ref["test"]["r_xy"].shape[0] < 6  # the variant really skips some seeds
    assert_results_identical(ref, opt)


def test_edge_cases_are_exercised():
    """Guard against fixtures silently losing the behaviour they exist to test."""
    stats = {}
    for name in ("negative_reliability", "above_one", "constant_voxels", "nan_outputs"):
        ref, _ = run_both(*sd.VARIANTS[name](), boots=1)
        stats[name] = ref["test"]
    assert (stats["negative_reliability"]["r_yy"] < 0).any()
    assert (np.abs(stats["above_one"]["r_xy_n_sb"]) > 1).any()
    assert np.isnan(stats["constant_voxels"]["r_yy"]).any()
    assert np.isnan(stats["nan_outputs"]["r_xy_n_sb"]).any()


def test_constant_predictions_from_strong_regularization():
    source, target = sd.make_fly_pair(n_source_voxels=20, n_target_voxels=16, seed=7)
    ref, opt = run_both(source, target, boots=2, map_kwargs=sd.enet_map_kwargs(alpha=100.0))
    assert np.isnan(ref["test"]["r_xy"]).all()  # all-zero coefficients
    assert_results_identical(ref, opt)


@pytest.mark.parametrize("seed", [0, 1])
def test_primary_two_bootstraps_bitwise(seed):
    if seed and not SLOW:
        pytest.skip("BRAINMODEL_SLOW_TESTS=1")
    source, target = sd.primary()
    ref, opt = run_both(source, target, boots=2, start_seed=42 + 100 * seed)
    assert_results_identical(ref, opt)


@pytest.mark.parametrize("boots", [3, 10, 50])
def test_primary_against_saved_reference(boots):
    if boots > 3 and not SLOW:
        pytest.skip("BRAINMODEL_SLOW_TESTS=1")
    path = fixtures.fixture_path(boots)
    if not path.exists():
        pytest.skip(f"run test/make_reference_fixtures.py --boots {boots}")
    saved, meta = fixtures.load(path)
    opt = F.get_linregress_consistency(**fixtures.run_kwargs(boots))
    if meta["env"] == fixtures.environment():
        assert_results_identical(saved, opt)
    else:  # different BLAS/platform than the fixture: tolerance only
        assert_results_close(saved, opt, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Phase 3: full-data caching
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("vectorize", [False, True])
@pytest.mark.parametrize("cache_full", [False, True])
def test_each_optimization_stage_is_bitwise(vectorize, cache_full):
    if vectorize and not F.VECTORIZED_PEARSON_AVAILABLE:
        pytest.skip("vectorized replica unavailable")
    source, target = sd.constant_voxels()
    kw = dict(source=source, target=target, map_kwargs=sd.enet_map_kwargs(),
              num_bootstrap_iters=2, start_seed=42, splits=SPLITS)
    ref = R.get_linregress_consistency(**kw)
    opt = F.compute_linregress_consistency(**kw, vectorize=vectorize, cache_full=cache_full)
    assert_results_identical(ref, opt)


def test_full_data_predictions_unchanged():
    source, target = sd.independent()
    X = np.nanmean(source, 0)
    Y = np.nanmean(target, 0)
    for split in SPLITS:
        nm = PipelineNeuralMap(**sd.enet_map_kwargs())
        nm.fit(X[split["train"]], Y[split["train"]])
        entry = F._full_data_entry(sd.enet_map_kwargs(), X, Y, split, True, 0, None)
        np.testing.assert_array_equal(entry["pred_train"], nm.predict(X[split["train"]]))
        np.testing.assert_array_equal(entry["pred_test"], nm.predict(X[split["test"]]))


@pytest.mark.parametrize("boots", [1, 3])
def test_fit_counts(monkeypatch, boots):
    source, target = sd.strongly_correlated()
    counter = FitCounter(monkeypatch)
    R.get_linregress_consistency(source=source, target=target, map_kwargs=sd.enet_map_kwargs(),
                                 num_bootstrap_iters=boots, start_seed=42, splits=SPLITS)
    assert counter.n == 3 * 5 * boots
    counter.n = 0
    stats = {"fits": 0}
    F.compute_linregress_consistency(source=source, target=target, map_kwargs=sd.enet_map_kwargs(),
                                     num_bootstrap_iters=boots, start_seed=42, splits=SPLITS,
                                     counter=stats)
    assert counter.n == stats["fits"] == 5 + 2 * 5 * boots


def test_fit_count_production_scale_formula():
    # 50 bootstraps x 5 folds: 750 fits in the reference, 505 when cached.
    assert 3 * 5 * 50 == 750 and 5 + 2 * 5 * 50 == 505


def test_bootstrap_order_and_seeds_unchanged():
    source, target = sd.weakly_correlated()
    ref, opt = run_both(source, target, boots=3, start_seed=7)
    assert_results_identical(ref, opt)
    # Each bootstrap row equals a single-bootstrap run at that seed.
    for i, seed in enumerate(range(7, 10)):
        single = F.get_linregress_consistency(source=source, target=target, map_kwargs=sd.enet_map_kwargs(),
                                              num_bootstrap_iters=1, start_seed=seed, splits=SPLITS)
        np.testing.assert_array_equal(single["test"]["r_xx"][0], opt["test"]["r_xx"][i])


def _random_selection(random_state):
    kw = sd.enet_map_kwargs()
    kw["map_kwargs"]["regression_kwargs"].update(selection="random", random_state=random_state)
    return kw


def test_seeded_random_selection_uses_cache_and_matches():
    source, target = sd.strongly_correlated()
    mk = _random_selection(0)
    assert F._fast_path_supported(source, "pearsonr", mk, {})
    ref, opt = run_both(source, target, boots=2, map_kwargs=mk)
    assert_results_identical(ref, opt)


@pytest.mark.parametrize("random_state", [None, np.random.RandomState(0)])
def test_unseeded_or_stateful_randomness_falls_back_to_reference(random_state):
    source, _ = sd.strongly_correlated()
    assert not F._fast_path_supported(source, "pearsonr", _random_selection(random_state), {})


# ---------------------------------------------------------------------------
# Phase 4: target batching (opt-in, tolerance-based)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_batched_elasticnet_coefficients_bitwise(dtype):
    """Per-target independence: batching reproduces coef_ and n_iter_ exactly."""
    source, target = sd.make_fly_pair(n_source_voxels=60, n_target_voxels=130, dtype=dtype, seed=3)
    X, Y = np.nanmean(source, 0)[:11], np.nanmean(target, 0)[:11]
    params = dict(alpha=0.01, l1_ratio=0.25, max_iter=10000)
    full = ElasticNet(**params).fit(X, Y)
    for batch in (64, 128, 7):
        coefs, iters = [], []
        for start in range(0, 130, batch):
            m = ElasticNet(**params).fit(X, Y[:, start:start + batch])
            coefs.append(np.atleast_2d(m.coef_))
            iters.extend(np.atleast_1d(m.n_iter_))
        np.testing.assert_array_equal(np.concatenate(coefs), full.coef_)
        assert iters == list(full.n_iter_)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("batch", [64, 128, 256, 7, 1])
def test_batched_predictions_within_rounding_bound(dtype, batch):
    """Batched and unbatched predictions differ only by summation order.

    Coefficients are bitwise identical (test above); BLAS evaluates
    ``X @ coef.T`` (and the intercept ``X_offset @ coef.T``) with a different
    reduction order when the matrix width changes. Two orderings of a length-n
    dot product each have error <= n*u*sum|x_i*w_i| (u = unit roundoff), so the
    two results differ by at most twice that. Prediction = X@w - X_offset@w +
    y_offset, so the bound uses (|X| + |X_offset|) @ |w|.
    """
    source, target = sd.make_fly_pair(n_source_voxels=60, n_target_voxels=130, dtype=dtype, seed=4)
    X, Y = np.nanmean(source, 0), np.nanmean(target, 0)
    split = SPLITS[1]
    X_train, Y_train, X_test = X[split["train"]], Y[split["train"]], X[split["test"]]
    args = (sd.enet_map_kwargs(), X_train, Y_train, X_test)
    p_train, p_test = F.fit_predict(*args, target_batch_size=0)
    b_train, b_test = F.fit_predict(*args, target_batch_size=batch)
    assert b_train.dtype == p_train.dtype and b_train.shape == p_train.shape

    reg = sd.enet_map_kwargs()["map_kwargs"]["regression_kwargs"]
    model = ElasticNet(**reg).fit(X_train, Y_train)
    W = np.abs(model.coef_).astype(np.float64)
    X_offset = np.abs(X_train.astype(np.float64).mean(0))
    u = np.finfo(dtype).eps / 2
    n = X.shape[1] + 2
    for Xp, p, b in ((X_train, p_train, b_train), (X_test, p_test, b_test)):
        scale = (np.abs(Xp.astype(np.float64)) + X_offset) @ W.T + np.abs(p.astype(np.float64))
        bound = 2 * n * u * scale
        diff = np.abs(b.astype(np.float64) - p)
        assert (diff <= bound).all(), float(np.max(diff / bound))


@pytest.mark.parametrize("batch", [64, 128])
def test_batched_consistency_close(batch):
    """130 targets / batch 64 leaves a remainder batch of 2."""
    source, target = sd.make_fly_pair(n_source_voxels=48, n_target_voxels=130, seed=5)
    kw = dict(source=source, target=target, map_kwargs=sd.enet_map_kwargs(),
              num_bootstrap_iters=2, start_seed=42, splits=SPLITS)
    ref = R.get_linregress_consistency(**kw)
    opt = F.get_linregress_consistency(**kw, target_batch_size=batch)
    worst = assert_results_close(ref, opt, rtol=1e-4, atol=1e-5)
    assert worst < 1e-4


def test_batch_size_env_var(monkeypatch):
    monkeypatch.setenv(F.TARGET_BATCH_SIZE_ENV, "256")
    assert F.resolve_target_batch_size() == 256
    assert F.resolve_target_batch_size(0) == 0
    monkeypatch.delenv(F.TARGET_BATCH_SIZE_ENV)
    assert F.resolve_target_batch_size() == 0
    with pytest.raises(ValueError):
        F.resolve_target_batch_size(-1)


def test_batching_only_for_independent_target_regressions():
    assert F._is_batchable(sd.enet_map_kwargs())
    ridge = {"map_type": "sklinear", "map_kwargs": {"regression_type": "Ridge"}}
    assert not F._is_batchable(ridge)
    pre = sd.enet_map_kwargs()
    pre["map_kwargs"]["regression_kwargs"]["precompute"] = True
    assert not F._is_batchable(pre)
    assert not F._is_batchable({"map_type": "percentile"})


# ---------------------------------------------------------------------------
# API compatibility and delegation
# ---------------------------------------------------------------------------
def test_parallel_two_workers_matches_serial():
    source, target = sd.different_trial_counts()
    kw = dict(source=source, target=target, map_kwargs=sd.enet_map_kwargs(),
              num_bootstrap_iters=3, start_seed=42, splits=SPLITS)
    serial = R.get_linregress_consistency(**kw)
    parallel = F.get_linregress_consistency(**kw, num_parallel_jobs=2)
    assert_results_identical(serial, parallel)


def test_xarray_inputs():
    source, target = sd.make_fly_pair(n_source_voxels=12, n_target_voxels=10, seed=5)
    tx = xr.DataArray(target, dims=("trials", "stimuli", "units"), coords={"units": np.arange(100, 110)})
    sx = xr.DataArray(source, dims=("trials", "stimuli", "units"))
    ref, opt = run_both(sx, tx, boots=2)
    assert isinstance(opt["test"]["r_xy_n_sb"], xr.DataArray)
    assert_results_identical(ref, opt)


def test_model_features_source():
    source, target = sd.make_fly_pair(n_source_voxels=16, n_target_voxels=12, seed=6)
    ref, opt = run_both(source.mean(0), target, boots=2)
    assert_results_identical(ref, opt)


def test_default_random_splits_and_kwargs():
    source, target = sd.make_fly_pair(n_source_voxels=16, n_target_voxels=12, seed=8)
    ref, opt = run_both(source, target, boots=2, splits=None, num_train_test_splits=3, train_frac=0.7)
    assert ref["test"]["r_xy"].shape[1] == 3
    assert_results_identical(ref, opt)


def test_per_split_map_kwargs_with_skipped_split():
    source, target = sd.make_fly_pair(n_source_voxels=16, n_target_voxels=12, seed=9)
    mk = [sd.enet_map_kwargs(), None, sd.enet_map_kwargs(alpha=0.1), sd.enet_map_kwargs(), None]
    ref, opt = run_both(source, target, boots=2, map_kwargs=mk)
    assert ref["test"]["r_xy"].shape[1] == 3
    assert_results_identical(ref, opt)


@pytest.mark.parametrize(
    "map_kwargs",
    [
        {"map_type": "sklinear", "map_kwargs": {"regression_type": "Ridge", "regression_kwargs": {"alpha": 1.0}}},
        {"map_type": "percentile"},
        {"map_type": "percentile", "map_kwargs": {"percentile": 75, "identity": False,
                                                   "regression_type": "Ridge", "regression_kwargs": {"alpha": 1.0}}},
        {"map_type": "pls", "map_kwargs": {"n_components": 3}},
    ],
)
def test_other_map_types(map_kwargs):
    source, target = sd.make_fly_pair(n_source_voxels=16, n_target_voxels=12, seed=10)
    ref, opt = run_both(source, target, boots=2, map_kwargs=map_kwargs)
    assert_results_identical(ref, opt)


@pytest.mark.parametrize("metric", ["spearmanr", "rsa_pearsonr"])
def test_other_metrics_delegate_to_reference(metric):
    source, target = sd.make_fly_pair(n_source_voxels=10, n_target_voxels=8, seed=11)
    mk = {"map_type": "identity"} if metric.startswith("rsa") else sd.enet_map_kwargs()
    assert not F._fast_path_supported(source, metric, mk, {})
    ref, opt = run_both(source[:, :, :8], target, boots=2, map_kwargs=mk, metric=metric)
    assert_results_identical(ref, opt)


def test_zero_bootstraps():
    source, target = sd.make_fly_pair(n_source_voxels=8, n_target_voxels=8, seed=12)
    ref, opt = run_both(source, target, boots=0)
    assert dict(ref["test"]) == dict(opt["test"]) == {}


def test_invalid_inputs_raise_like_reference():
    source, target = sd.make_fly_pair(n_source_voxels=8, n_target_voxels=8, seed=13)
    source[:, 3, 2] = np.nan  # trial average is NaN -> reference asserts
    with pytest.raises(AssertionError):
        R.get_linregress_consistency(source=source, target=target, map_kwargs=sd.enet_map_kwargs(),
                                     num_bootstrap_iters=1, splits=SPLITS)
    with pytest.raises(AssertionError):
        F.get_linregress_consistency(source=source, target=target, map_kwargs=sd.enet_map_kwargs(),
                                     num_bootstrap_iters=1, splits=SPLITS)
    with pytest.raises(TypeError):
        F.get_linregress_consistency(source=source, target=target, map_kwargs=sd.enet_map_kwargs(),
                                     num_bootstrap_iters=1, splits=SPLITS, bogus=1)


def test_warnings_still_emitted():
    source, target = sd.constant_voxels()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        F.get_linregress_consistency(source=source, target=target, map_kwargs=sd.enet_map_kwargs(),
                                     num_bootstrap_iters=1, splits=SPLITS)
    names = {type(w.message).__name__ for w in caught}
    assert "ConstantInputWarning" in names


# ---------------------------------------------------------------------------
# Phase 6: public import path
# ---------------------------------------------------------------------------
def test_public_signature_unchanged():
    import inspect

    from brainmodel_utils.metrics import consistency

    params = inspect.signature(consistency.get_linregress_consistency).parameters
    assert list(params) == list(inspect.signature(R.get_linregress_consistency).parameters)
    for name, p in inspect.signature(R.get_linregress_consistency).parameters.items():
        assert params[name].default == p.default and params[name].kind == p.kind
    for name in ("sb_correction", "get_consistency_per_neuron", "get_linregress_consistency_persplit",
                 "get_linregress_consistency_persphalftrial", "concat_dict_sp", "generic_trial_avg",
                 "get_splithalves", "PipelineNeuralMap", "generate_train_test_splits"):
        assert hasattr(consistency, name), name


@pytest.mark.parametrize("impl", ["fast", "reference", ""])
def test_public_entry_point_matches_reference(monkeypatch, impl):
    from brainmodel_utils.metrics import consistency

    monkeypatch.setenv(consistency.IMPL_ENV, impl)
    called = {}
    for mod, label in ((F, "fast"), (R, "reference")):
        original = mod.get_linregress_consistency

        def spy(*a, _orig=original, _label=label, **k):
            called.setdefault(_label, 0)
            called[_label] += 1
            return _orig(*a, **k)

        monkeypatch.setattr(mod, "get_linregress_consistency", spy)
    source, target = sd.make_fly_pair(n_source_voxels=12, n_target_voxels=10, seed=14)
    kw = dict(source=source, target=target, map_kwargs=sd.enet_map_kwargs(), num_bootstrap_iters=2,
              num_parallel_jobs=1, start_seed=42, splits=SPLITS, metric="pearsonr")
    out = consistency.get_linregress_consistency(**kw)
    assert ("fast" in called) == (impl in ("fast", ""))
    monkeypatch.undo()
    assert_results_identical(R.get_linregress_consistency(**kw), out)


def test_public_entry_point_rejects_unknown_impl(monkeypatch):
    from brainmodel_utils.metrics import consistency

    monkeypatch.setenv(consistency.IMPL_ENV, "gpu")
    with pytest.raises(ValueError):
        consistency.get_linregress_consistency(source=np.zeros((2, 3, 2)), target=np.zeros((2, 3, 2)),
                                               map_kwargs={}, num_bootstrap_iters=1)


def test_reference_rollback_ignores_batch_kwarg(monkeypatch):
    from brainmodel_utils.metrics import consistency

    monkeypatch.setenv(consistency.IMPL_ENV, "reference")
    source, target = sd.make_fly_pair(n_source_voxels=8, n_target_voxels=8, seed=15)
    consistency.get_linregress_consistency(source=source, target=target, map_kwargs=sd.enet_map_kwargs(),
                                           num_bootstrap_iters=1, splits=SPLITS, target_batch_size=4)


def test_production_caller_import_path(tmp_path):
    """Mirror autonomous_drosophila.load_consistency_function: path insert + import."""
    import importlib
    import sys

    repo = str(fixtures.HERE.parent)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    module = importlib.import_module("brainmodel_utils.metrics.consistency")
    assert module.__file__.startswith(repo)
    assert module.get_linregress_consistency.__module__ == "brainmodel_utils.metrics.consistency"
