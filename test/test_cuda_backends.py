import warnings

import numpy as np
import pytest
import xarray as xr
from sklearn.linear_model import Ridge

from brainmodel_utils.metrics.consistency import get_linregress_consistency
from brainmodel_utils.neural_mappers import backends
from brainmodel_utils.neural_mappers.percentile_neural_map import PercentileNeuralMap
from brainmodel_utils.neural_mappers.sklinear_neural_map import SKLinearNeuralMap
from brainmodel_utils.neural_mappers.utils import generate_train_test_splits

pytestmark = pytest.mark.cuda

RIDGE_RTOL = 2e-3
RIDGE_ATOL = 2e-3
SCORE_RTOL = 2e-4
SCORE_ATOL = 2e-4
CORRELATION_RTOL = 2e-5
CORRELATION_ATOL = 2e-5
CONSISTENCY_RTOL = 5e-3
CONSISTENCY_ATOL = 5e-3
CONSISTENCY_METRICS = {
    "r_xy_n_sb",
    "r_xx",
    "r_xx_sb",
    "r_yy",
    "r_yy_sb",
    "r_xy",
    "denom_sb",
}
SCIENTIFIC_PERCENTILES = (0, 50, 75, 100)
SCIENTIFIC_ALPHAS = (0.01, 1.0, 100.0, 10000.0)


def _require_cupy_cuda():
    if not backends.cuda_available():
        pytest.skip("CuPy and a visible CUDA device are required")


def _require_torch_cuda():
    if not backends.torch_cuda_available():
        pytest.skip("CUDA-enabled PyTorch and a visible CUDA device are required")


def _regression_data(n_samples, n_features, n_targets, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    weights = rng.normal(size=(n_features, n_targets))
    intercept = rng.normal(size=n_targets)
    Y = X @ weights + intercept + 0.02 * rng.normal(size=(n_samples, n_targets))
    return X, Y


@pytest.mark.parametrize(
    ("shape", "alpha", "fit_intercept"),
    [
        ((64, 9, 1), 0.01, True),
        ((41, 13, 1), 1.0, False),
        ((57, 8, 4), 100.0, True),
        ((32, 47, 3), 10000.0, False),
    ],
)
def test_cuda_ridge_matches_cpu(shape, alpha, fit_intercept):
    _require_cupy_cuda()
    n_samples, n_features, n_targets = shape
    X, Y = _regression_data(*shape, seed=n_samples + n_features + n_targets)
    target = Y[:, 0] if n_targets == 1 else Y
    target_2d = Y[:, :1] if n_targets == 1 else Y
    kwargs = {"alpha": alpha, "fit_intercept": fit_intercept}
    cpu = SKLinearNeuralMap(regression_kwargs=kwargs, backend="cpu")
    cuda = SKLinearNeuralMap(regression_kwargs=kwargs, backend="cuda")

    cpu.fit(X, target)
    cuda.fit(X, target)
    cpu_predictions = cpu.predict(X)
    cuda_predictions = cuda.predict(X)

    assert n_samples != n_features
    assert isinstance(cuda._mapper.coef_, np.ndarray)
    assert isinstance(cuda_predictions, np.ndarray)
    assert cuda._mapper.coef_.dtype == cpu._mapper.coef_.dtype
    assert cuda_predictions.dtype == cpu_predictions.dtype
    assert cuda._mapper.coef_.shape == cpu._mapper.coef_.shape
    assert (
        np.asarray(cuda._mapper.intercept_).shape
        == np.asarray(cpu._mapper.intercept_).shape
    )
    assert cuda_predictions.shape == (n_samples, n_targets)
    np.testing.assert_allclose(
        cuda._mapper.coef_,
        cpu._mapper.coef_,
        rtol=RIDGE_RTOL,
        atol=RIDGE_ATOL,
    )
    np.testing.assert_allclose(
        cuda._mapper.intercept_,
        cpu._mapper.intercept_,
        rtol=RIDGE_RTOL,
        atol=RIDGE_ATOL,
    )
    np.testing.assert_allclose(
        cuda_predictions,
        cpu_predictions,
        rtol=RIDGE_RTOL,
        atol=RIDGE_ATOL,
    )
    np.testing.assert_allclose(
        cuda._mapper.score(X, target_2d),
        cpu._mapper.score(X, target_2d),
        rtol=SCORE_RTOL,
        atol=SCORE_ATOL,
    )
    np.testing.assert_allclose(
        cuda.score(target_2d, cuda_predictions),
        cpu.score(target_2d, cpu_predictions),
        rtol=SCORE_RTOL,
        atol=SCORE_ATOL,
        equal_nan=True,
    )


def test_cuda_batched_ridge_handles_float32_duplicate_columns():
    _require_cupy_cuda()
    rng = np.random.default_rng(191)
    basis = rng.normal(size=(64, 3)).astype(np.float32)
    X = np.column_stack((basis[:, 0], basis[:, 0], basis[:, 1], basis[:, 2]))
    Y = np.column_stack(
        (2 * basis[:, 0] - basis[:, 1], -0.5 * basis[:, 0] + basis[:, 2])
    )
    kwargs = {"alpha": 0.01, "solver": "auto"}
    cpu = SKLinearNeuralMap(regression_kwargs=kwargs, backend="cpu")
    cuda = SKLinearNeuralMap(regression_kwargs=kwargs, backend="cuda")

    cpu.fit(X, Y)
    cuda.fit(X, Y)

    assert cuda._mapper.coef_.dtype == np.float32
    assert cuda._mapper.compute_dtype == np.dtype(np.float32)
    assert not cuda._mapper.promoted_to_float64
    assert cuda.predict(X).dtype == np.float32
    assert cuda._mapper.prediction_compute_dtype == np.dtype(np.float32)
    assert cuda._mapper.prediction_output_dtype == np.dtype(np.float32)
    np.testing.assert_allclose(
        cuda.predict(X), cpu.predict(X), rtol=RIDGE_RTOL, atol=RIDGE_ATOL
    )


def test_cuda_grouped_ridge_handles_float32_duplicate_columns():
    _require_cupy_cuda()
    rng = np.random.default_rng(192)
    basis = rng.normal(size=(64, 3)).astype(np.float32)
    X = np.column_stack((basis[:, 0], basis[:, 0], basis[:, 1], basis[:, 2]))
    Y = np.column_stack(
        (
            2 * basis[:, 0] - basis[:, 1],
            -0.5 * basis[:, 0] + basis[:, 2],
            basis[:, 1] - basis[:, 2],
        )
    )
    selections = [np.array([0, 1, 2]), np.array([0, 1, 3]), np.array([2, 3])]
    kwargs = {"alpha": 0.01, "solver": "auto"}
    expected = np.column_stack(
        [
            Ridge(**kwargs).fit(X[:, selected], Y[:, target]).predict(X[:, selected])
            for target, selected in enumerate(selections)
        ]
    )

    models, grouped = backends.fit_grouped_ridge_cuda(X, Y, selections, kwargs)
    actual = grouped.predict(X)

    assert all(model.coef_.dtype == np.float32 for model in models)
    assert grouped.compute_dtype == np.dtype(np.float32)
    assert not grouped.promoted_to_float64
    assert grouped.prediction_output_dtype == np.dtype(np.float64)
    np.testing.assert_allclose(actual, expected, rtol=RIDGE_RTOL, atol=RIDGE_ATOL)


def _percentile_data():
    rng = np.random.default_rng(20260820)
    n_samples = 72
    latent = rng.normal(size=(n_samples, 4))
    X = np.column_stack(
        (
            latent[:, 0] + 0.03 * rng.normal(size=n_samples),
            -0.8 * latent[:, 0] + 0.12 * rng.normal(size=n_samples),
            latent[:, 1] + 0.04 * rng.normal(size=n_samples),
            0.65 * latent[:, 1] + 0.25 * rng.normal(size=n_samples),
            latent[:, 2] + 0.06 * rng.normal(size=n_samples),
            -0.75 * latent[:, 2] + 0.15 * rng.normal(size=n_samples),
            latent[:, 3] + 0.08 * rng.normal(size=n_samples),
            rng.normal(size=n_samples),
            rng.normal(size=n_samples),
            np.ones(n_samples),
        )
    )
    Y = np.column_stack(
        (
            0.9 * latent[:, 0] + 0.1 * rng.normal(size=n_samples),
            0.8 * latent[:, 1]
            - 0.25 * latent[:, 3]
            + 0.12 * rng.normal(size=n_samples),
            -0.85 * latent[:, 2] + 0.1 * rng.normal(size=n_samples),
            np.full(n_samples, 2.0),
        )
    )
    return X, Y


def _scientific_grid_data(dtype):
    rng = np.random.default_rng(20260824)
    X_train = rng.normal(size=(80, 8))
    X_test = rng.normal(size=(24, 8))
    X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
    X_test = (X_test - X_test.mean(axis=0)) / X_test.std(axis=0)
    coefficients = np.asarray(
        [
            [9.0, -2.0, 1.0],
            [7.0, 8.0, -3.0],
            [5.0, -6.0, 7.0],
            [3.0, 4.0, -8.0],
            [1.5, -3.0, 5.0],
            [-1.0, 2.0, -4.0],
            [-3.0, 1.0, 2.0],
            [-5.0, -1.0, -1.0],
        ]
    )
    intercept = np.asarray([0.5, -1.0, 2.0])
    Y_train = X_train @ coefficients + intercept
    Y_train += 0.02 * rng.normal(size=Y_train.shape)
    Y_test = X_test @ coefficients + intercept
    Y_test += 0.02 * rng.normal(size=Y_test.shape)
    return tuple(
        np.asarray(array, dtype=dtype) for array in (X_train, Y_train, X_test, Y_test)
    )


def _grid_tolerances(dtype):
    if np.dtype(dtype) == np.float32:
        return {"rtol": 2e-3, "atol": 2e-4}
    return {"rtol": 1e-10, "atol": 1e-12}


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("alpha", SCIENTIFIC_ALPHAS)
@pytest.mark.parametrize("percentile", SCIENTIFIC_PERCENTILES)
def test_cuda_percentile_scientific_grid_matches_cpu(dtype, alpha, percentile):
    _require_cupy_cuda()
    _require_torch_cuda()
    X_train, Y_train, X_test, Y_test = _scientific_grid_data(dtype)
    regression_kwargs = {
        "alpha": alpha,
        "fit_intercept": True,
        "solver": "auto",
    }
    cpu = PercentileNeuralMap(
        percentile=percentile,
        identity=False,
        regression_kwargs=regression_kwargs,
        backend="cpu",
    )
    cuda = PercentileNeuralMap(
        percentile=percentile,
        identity=False,
        regression_kwargs=regression_kwargs,
        backend="cuda",
    )

    cpu.fit(X_train, Y_train)
    cuda.fit(X_train, Y_train)
    tolerances = _grid_tolerances(dtype)

    assert not cpu._identity and not cuda._identity
    assert cpu._corrs.dtype == cuda._corrs.dtype == np.dtype(dtype)
    np.testing.assert_allclose(cuda._corrs, cpu._corrs, equal_nan=True, **tolerances)
    np.testing.assert_allclose(
        cuda._thresholds, cpu._thresholds, equal_nan=True, **tolerances
    )
    for cuda_sources, cpu_sources in zip(cuda._source_indices, cpu._source_indices):
        np.testing.assert_array_equal(cuda_sources, cpu_sources)
    for cuda_mapper, cpu_mapper in zip(cuda._mappers, cpu._mappers):
        assert cuda_mapper.coef_.shape == cpu_mapper.coef_.shape
        assert cuda_mapper.coef_.dtype == cpu_mapper.coef_.dtype == np.dtype(dtype)
        np.testing.assert_allclose(cuda_mapper.coef_, cpu_mapper.coef_, **tolerances)
        np.testing.assert_allclose(
            cuda_mapper.intercept_, cpu_mapper.intercept_, **tolerances
        )

    cpu_train = cpu.predict(X_train)
    cuda_train = cuda.predict(X_train)
    cpu_test = cpu.predict(X_test)
    cuda_test = cuda.predict(X_test)
    assert cpu_test.dtype == cuda_test.dtype == np.float64
    np.testing.assert_allclose(cuda_train, cpu_train, **tolerances)
    np.testing.assert_allclose(cuda_test, cpu_test, **tolerances)
    np.testing.assert_allclose(
        cuda.score(Y_test, cuda_test), cpu.score(Y_test, cpu_test), **tolerances
    )

    report = cuda._cuda_batch_report
    dtype_name = np.dtype(dtype).name
    assert report["correlation_dtype_requested"] is None
    assert report["correlation_dtype_effective"] == dtype_name
    assert report["ridge_compute_dtype"] == dtype_name
    assert report["coefficient_dtype"] == dtype_name
    assert report["prediction_compute_dtype"] == dtype_name
    assert report["prediction_output_dtype"] == "float64"
    assert not report["ridge_promoted_to_float64"]

    if percentile == 0:
        expected_sources = np.arange(X_train.shape[1])
        direct = Ridge(**regression_kwargs).fit(X_train, Y_train)
        direct_train = direct.predict(X_train)
        direct_test = direct.predict(X_test)
        assert report["ridge_shared_design"]
        for target, sources in enumerate(cuda._source_indices):
            np.testing.assert_array_equal(sources, expected_sources)
            np.testing.assert_allclose(
                cuda._mappers[target].coef_,
                direct.coef_[target],
                **tolerances,
            )
            np.testing.assert_allclose(
                cuda._mappers[target].intercept_,
                np.asarray(direct.intercept_[target]).reshape(1),
                **tolerances,
            )
        np.testing.assert_allclose(cuda_train, direct_train, **tolerances)
        np.testing.assert_allclose(cuda_test, direct_test, **tolerances)
        np.testing.assert_allclose(
            cuda.score(Y_test, cuda_test),
            cuda.score(Y_test, direct_test),
            **tolerances,
        )
    elif percentile == 100:
        assert all(indices.size == 1 for indices in cuda._source_indices)


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_percentile_100_fits_ridge_and_alpha_changes_the_mapping(backend):
    if backend == "cuda":
        _require_cupy_cuda()
        _require_torch_cuda()
    X_train, Y_train, X_test, _ = _scientific_grid_data(np.float64)
    fitted = []
    for alpha in (0.01, 10000.0):
        mapper = PercentileNeuralMap(
            percentile=100,
            identity=False,
            regression_kwargs={"alpha": alpha, "solver": "auto"},
            backend=backend,
        )
        mapper.fit(X_train, Y_train)
        fitted.append(mapper)

    low_alpha, high_alpha = fitted
    assert not low_alpha._identity and not high_alpha._identity
    for low_sources, high_sources in zip(
        low_alpha._source_indices, high_alpha._source_indices
    ):
        assert low_sources.size == 1
        np.testing.assert_array_equal(low_sources, high_sources)
    low_coefficients = np.concatenate(
        [mapper.coef_.reshape(-1) for mapper in low_alpha._mappers]
    )
    high_coefficients = np.concatenate(
        [mapper.coef_.reshape(-1) for mapper in high_alpha._mappers]
    )
    assert np.max(np.abs(low_coefficients - high_coefficients)) > 1e-2
    assert not np.allclose(low_alpha.predict(X_test), high_alpha.predict(X_test))


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_percentile_100_keeps_exact_maximum_ties_before_ridge(backend):
    if backend == "cuda":
        _require_cupy_cuda()
        _require_torch_cuda()
    rng = np.random.default_rng(84)
    best = rng.normal(size=48)
    X = np.column_stack((best, best, rng.normal(size=best.size)))
    Y = best[:, np.newaxis]
    mapper = PercentileNeuralMap(
        percentile=100,
        identity=False,
        regression_kwargs={"alpha": 1.0, "solver": "auto"},
        backend=backend,
    )

    mapper.fit(X, Y)

    np.testing.assert_array_equal(mapper._source_indices[0], [0, 1])
    assert mapper._mappers[0].coef_.shape == (2,)
    assert mapper.predict(X).shape == Y.shape


@pytest.mark.parametrize(
    ("source_dtype", "target_dtype", "expected_dtype"),
    [
        (np.float32, np.float32, np.float32),
        (np.float32, np.float64, np.float64),
        (np.float64, np.float32, np.float64),
        (np.float64, np.float64, np.float64),
    ],
)
def test_cuda_correlation_dtype_is_inferred_from_inputs(
    source_dtype, target_dtype, expected_dtype
):
    _require_torch_cuda()
    X, Y, _, _ = _scientific_grid_data(np.float64)

    correlations = backends.correlations_cuda(
        X.astype(source_dtype), Y.astype(target_dtype)
    )

    assert correlations.dtype == np.dtype(expected_dtype)


def test_cuda_correlation_dtype_can_be_overridden_for_diagnostics():
    _require_torch_cuda()
    X, Y, _, _ = _scientific_grid_data(np.float64)

    correlations = backends.correlations_cuda(X, Y, dtype=np.float32)

    assert correlations.dtype == np.float32


def _fit_percentile(X, Y, backend):
    mapper = PercentileNeuralMap(
        percentile=70,
        identity=False,
        regression_kwargs={"alpha": 0.5},
        backend=backend,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mapper.fit(X, Y)
    return mapper


def _percentile_thresholds(mapper):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.asarray(
            [
                np.nanpercentile(mapper._corrs[:, i], mapper._percentile)
                for i in range(mapper._n_targets)
            ]
        )


def _selected_sources(mapper):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return [
            np.flatnonzero(mapper._get_sources(i)) for i in range(mapper._n_targets)
        ]


def _percentile_edge_case_data():
    rng = np.random.default_rng(73)
    basis = rng.normal(size=(37, 9))
    basis -= basis.mean(axis=0)
    basis = np.linalg.qr(basis)[0]
    X = np.column_stack(
        (
            basis[:, 0],
            basis[:, 0],
            basis[:, 0],
            basis[:, 1],
            basis[:, 1],
            basis[:, 1],
            basis[:, 2],
            basis[:, 2],
            basis[:, 2],
            basis[:, 3:],
            np.ones(basis.shape[0]),
        )
    )
    coefficients = np.asarray(
        (
            (5, 0, 0, -3, -2, -1, 1, 2, 3),
            (0, 0, -5, -3, -2, -1, 1, 2, 3),
            (-3, 5, 0, -2, -1, 0, 1, 2, 3),
        )
    ).T
    Y = np.column_stack((basis @ coefficients, np.full(basis.shape[0], 2.0)))
    return X, Y


@pytest.mark.parametrize("percentile", [0, 1, 50, 70, 99, 99.5, 100])
def test_cuda_percentile_topk_matches_numpy_edge_cases(percentile):
    _require_torch_cuda()
    X, Y = _percentile_edge_case_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        correlations, thresholds, selections = backends.correlations_cuda(
            X, Y, percentile
        )
        expected_thresholds = np.nanpercentile(correlations, percentile, axis=0)
        source_rank = np.arange(correlations.shape[0])
        fallback = np.flatnonzero(
            source_rank >= np.nanpercentile(source_rank, percentile)
        )

        expected_selections = []
        for target, threshold in enumerate(expected_thresholds):
            tolerance = backends.correlation_cutoff_tolerance(threshold)
            selected = np.flatnonzero(correlations[:, target] >= threshold - tolerance)
            expected_selections.append(selected if selected.size else fallback)

    assert np.isnan(correlations[-1, :-1]).all()
    assert np.isnan(correlations[:, -1]).all()
    assert any(
        np.unique(correlations[np.isfinite(correlations[:, target]), target]).size
        < np.isfinite(correlations[:, target]).sum()
        for target in range(correlations.shape[1] - 1)
    )
    np.testing.assert_array_equal(thresholds, expected_thresholds)
    for actual, expected in zip(selections, expected_selections):
        np.testing.assert_array_equal(actual, expected)


def test_cuda_percentile_matches_cpu_float32_and_selection():
    _require_cupy_cuda()
    _require_torch_cuda()
    X, Y = _percentile_data()
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    cpu = _fit_percentile(X, Y, "cpu")
    cuda = _fit_percentile(X, Y, "cuda")
    expected_sources = ([0, 5, 8], [1, 2, 3], [5, 7, 8], [7, 8, 9])

    assert cuda._corrs.dtype == np.float32
    np.testing.assert_allclose(
        cuda._corrs,
        cpu._corrs,
        rtol=CORRELATION_RTOL,
        atol=CORRELATION_ATOL,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        _percentile_thresholds(cuda),
        _percentile_thresholds(cpu),
        rtol=CORRELATION_RTOL,
        atol=CORRELATION_ATOL,
        equal_nan=True,
    )

    for mapper in (cpu, cuda):
        for actual, expected in zip(_selected_sources(mapper), expected_sources):
            np.testing.assert_array_equal(actual, expected)
    for cuda_mapper, cpu_mapper in zip(cuda._mappers, cpu._mappers):
        assert cuda_mapper.coef_.shape == cpu_mapper.coef_.shape
        assert (
            np.asarray(cuda_mapper.intercept_).shape
            == np.asarray(cpu_mapper.intercept_).shape
        )

    for target_idx, threshold in enumerate(_percentile_thresholds(cpu)):
        finite = cpu._corrs[np.isfinite(cpu._corrs[:, target_idx]), target_idx]
        if finite.size:
            assert np.min(np.abs(finite - threshold)) > 5e-4

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cuda_predictions = cuda.predict(X)
        cpu_predictions = cpu.predict(X)
    np.testing.assert_allclose(
        cuda_predictions, cpu_predictions, rtol=RIDGE_RTOL, atol=RIDGE_ATOL
    )


def test_cuda_percentile_can_skip_host_correlation_matrix(tmp_path):
    _require_cupy_cuda()
    _require_torch_cuda()
    X, Y = _percentile_data()
    X = X.astype(np.float32)[:, ::-1]
    Y = Y.astype(np.float32)[:, ::-1]
    assert X.strides[1] < 0
    assert Y.strides[1] < 0
    cpu = _fit_percentile(X, Y, "cpu")
    weights_path = tmp_path / "cuda_percentile_weights.npz"
    cuda = PercentileNeuralMap(
        percentile=70,
        identity=False,
        regression_kwargs={"alpha": 0.5},
        weights_save_nm=weights_path,
        store_correlations=False,
        backend="cuda",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cuda.fit(X, Y)

    assert cuda._corrs is None
    for actual, expected in zip(_selected_sources(cuda), _selected_sources(cpu)):
        np.testing.assert_array_equal(actual, expected)
    cuda_prediction = cuda.predict(X)
    cpu_prediction = cpu.predict(X)
    assert cuda_prediction.dtype == cpu_prediction.dtype == np.float64
    np.testing.assert_allclose(
        cuda_prediction, cpu_prediction, rtol=RIDGE_RTOL, atol=RIDGE_ATOL
    )
    with np.load(weights_path) as saved:
        np.testing.assert_allclose(saved["arr_0"], np.stack(cuda._weights))


def test_cuda_percentile_accepts_xarray_and_preserves_identity_selection():
    _require_torch_cuda()
    X, Y = _percentile_data()
    X = xr.DataArray(X[:, :9], dims=("stimuli", "units"))
    Y = xr.DataArray(Y[:, :3], dims=("stimuli", "units"))
    cpu = PercentileNeuralMap(backend="cpu")
    cuda = PercentileNeuralMap(backend="cuda")

    cpu.fit(X, Y)
    cuda.fit(X, Y)

    assert cuda._corrs.dtype == np.float64
    np.testing.assert_allclose(
        cuda._corrs,
        cpu._corrs,
        rtol=CORRELATION_RTOL,
        atol=CORRELATION_ATOL,
    )
    np.testing.assert_allclose(
        _percentile_thresholds(cuda),
        _percentile_thresholds(cpu),
        rtol=CORRELATION_RTOL,
        atol=CORRELATION_ATOL,
    )
    for actual, expected in zip(_selected_sources(cuda), _selected_sources(cpu)):
        np.testing.assert_array_equal(actual, expected)
    assert [int(source) for source in cuda._mappers] == [0, 2, 5]
    np.testing.assert_array_equal(cuda.predict(X), cpu.predict(X))


def _consistency_data():
    rng = np.random.default_rng(410)
    source = rng.normal(size=(24, 7))
    weights = rng.normal(size=(7, 3))
    target_mean = source @ weights + rng.normal(size=3)
    target = target_mean + 0.08 * rng.normal(size=(16, 24, 3))
    return source, target


@pytest.mark.parametrize("map_type", ["sklinear", "percentile"])
def test_cuda_full_consistency_matches_cpu_for_every_metric(map_type):
    _require_cupy_cuda()
    if map_type == "percentile":
        _require_torch_cuda()
    source, target = _consistency_data()
    splits = generate_train_test_splits(
        num_stim=source.shape[0],
        num_splits=2,
        train_frac=2 / 3,
        start_seed=17,
    )
    common = {
        "source": source,
        "target": target,
        "num_bootstrap_iters": 2,
        "num_parallel_jobs": 1,
        "start_seed": 71,
        "splits": splits,
    }
    mapper_options = {
        "regression_type": "Ridge",
        "regression_kwargs": {"alpha": 0.5, "fit_intercept": True},
    }
    if map_type == "percentile":
        mapper_options.update({"percentile": 70, "identity": False})
    cpu = get_linregress_consistency(
        **common,
        map_kwargs={
            "map_type": map_type,
            "map_kwargs": dict(mapper_options, backend="cpu"),
        },
    )
    cuda = get_linregress_consistency(
        **common,
        map_kwargs={
            "map_type": map_type,
            "map_kwargs": dict(mapper_options, backend="cuda"),
        },
    )

    assert set(cpu) == set(cuda) == {"train", "test"}
    for partition in ("train", "test"):
        assert set(cpu[partition]) == set(cuda[partition]) == CONSISTENCY_METRICS
        for metric in CONSISTENCY_METRICS:
            assert cpu[partition][metric].shape == (2, 2, target.shape[-1])
            np.testing.assert_allclose(
                cuda[partition][metric],
                cpu[partition][metric],
                rtol=CONSISTENCY_RTOL,
                atol=CONSISTENCY_ATOL,
                equal_nan=True,
            )
