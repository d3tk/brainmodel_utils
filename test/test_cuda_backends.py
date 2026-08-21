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


def _require_cuml_cuda():
    if not backends.cuda_available():
        pytest.skip("cuML and a visible CUDA device are required")


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
        ((64, 9, 1), 0.05, True),
        ((41, 13, 1), 1.0, False),
        ((57, 8, 4), 0.5, True),
        ((32, 47, 3), 4.0, False),
    ],
)
def test_cuda_ridge_matches_cpu(shape, alpha, fit_intercept):
    _require_cuml_cuda()
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
    _require_cuml_cuda()
    rng = np.random.default_rng(191)
    basis = rng.normal(size=(64, 3)).astype(np.float32)
    X = np.column_stack((basis[:, 0], basis[:, 0], basis[:, 1], basis[:, 2]))
    Y = np.column_stack(
        (2 * basis[:, 0] - basis[:, 1], -0.5 * basis[:, 0] + basis[:, 2])
    )
    kwargs = {"alpha": 1e-6}
    cpu = SKLinearNeuralMap(regression_kwargs=kwargs, backend="cpu")
    cuda = SKLinearNeuralMap(regression_kwargs=kwargs, backend="cuda")

    cpu.fit(X, Y)
    cuda.fit(X, Y)

    assert cuda._mapper.coef_.dtype == np.float32
    np.testing.assert_allclose(
        cuda.predict(X), cpu.predict(X), rtol=RIDGE_RTOL, atol=RIDGE_ATOL
    )


def test_cuda_grouped_ridge_handles_float32_duplicate_columns():
    _require_cuml_cuda()
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
    kwargs = {"alpha": 1e-6}
    expected = np.column_stack(
        [
            Ridge(**kwargs)
            .fit(X[:, selected].astype(np.float64), Y[:, target])
            .predict(X[:, selected])
            for target, selected in enumerate(selections)
        ]
    )

    models, grouped = backends.fit_grouped_ridge_cuda(X, Y, selections, kwargs)
    actual = grouped.predict(X)

    assert all(model.coef_.dtype == np.float32 for model in models)
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
        selected = np.flatnonzero(correlations[:, target] >= threshold)
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
    _require_cuml_cuda()
    _require_torch_cuda()
    X, Y = _percentile_data()
    cpu = _fit_percentile(X, Y, "cpu")
    cpu_float32 = _fit_percentile(X.astype(np.float32), Y.astype(np.float32), "cpu")
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
        cuda._corrs,
        cpu_float32._corrs,
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

    for mapper in (cpu, cpu_float32, cuda):
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
    _require_cuml_cuda()
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

    assert cuda._corrs.dtype == np.float32
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
    _require_cuml_cuda()
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
