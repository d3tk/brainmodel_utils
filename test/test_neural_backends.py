import inspect
import subprocess
import sys
import warnings

import numpy as np
import pytest
import xarray as xr
from sklearn.linear_model import Lasso, Ridge

from brainmodel_utils.metrics.consistency import (
    _get_consistency_per_neuron_pearson,
    _mapping_uses_cuda,
    get_consistency_per_neuron,
    get_linregress_consistency,
)
from brainmodel_utils.neural_mappers import backends
from brainmodel_utils.neural_mappers.percentile_neural_map import PercentileNeuralMap
from brainmodel_utils.neural_mappers.sklinear_neural_map import SKLinearNeuralMap


def _regression_data():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(30, 5))
    weights = rng.normal(size=(5, 3))
    return X, X @ weights + 0.2


def test_vectorized_pearson_consistency_matches_per_target_path():
    rng = np.random.default_rng(33)
    arrays = [rng.normal(size=(19, 7)) for _ in range(6)]

    actual = _get_consistency_per_neuron_pearson(*arrays)
    expected = [
        get_consistency_per_neuron(*(array[:, target] for array in arrays))
        for target in range(arrays[0].shape[1])
    ]

    for metric in actual:
        np.testing.assert_allclose(
            actual[metric],
            [target[metric] for target in expected],
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_vectorized_pearson_matches_scipy_edge_cases(dtype):
    offset = dtype(1e12)
    step = dtype(np.spacing(offset))
    varying = offset + step * np.arange(8, dtype=dtype)
    reverse = varying[::-1]
    constant = np.full(8, offset, dtype=dtype)
    X = np.column_stack((varying, constant, varying))
    Y = np.column_stack((reverse, varying, constant))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        actual = _get_consistency_per_neuron_pearson(X, Y, X, Y, X, Y)
        expected = [
            get_consistency_per_neuron(
                X[:, target],
                Y[:, target],
                X[:, target],
                Y[:, target],
                X[:, target],
                Y[:, target],
            )
            for target in range(X.shape[1])
        ]

    for metric in actual:
        np.testing.assert_allclose(
            actual[metric],
            [target[metric] for target in expected],
            rtol=2e-6 if dtype == np.float32 else 1e-12,
            atol=2e-6 if dtype == np.float32 else 1e-12,
            equal_nan=True,
        )


def test_backend_is_last_optional_constructor_argument():
    sklinear = inspect.signature(SKLinearNeuralMap).parameters
    percentile = inspect.signature(PercentileNeuralMap).parameters

    assert list(sklinear)[-1] == "backend"
    assert list(percentile)[-1] == "backend"
    assert sklinear["backend"].default == "cpu"
    assert percentile["backend"].default == "cpu"
    assert percentile["store_correlations"].default is True

    linear_mapper = SKLinearNeuralMap("Ridge", {"alpha": 2.0}, "linear.npz")
    percentile_mapper = PercentileNeuralMap(
        80, False, "Ridge", {"alpha": 2.0}, "percentile.npz"
    )
    assert linear_mapper._weights_save_nm == "linear.npz"
    assert percentile_mapper._weights_save_nm == "percentile.npz"


@pytest.mark.parametrize("mapper", [SKLinearNeuralMap, PercentileNeuralMap])
def test_invalid_backend_is_rejected(mapper):
    with pytest.raises(ValueError, match="backend"):
        mapper(backend="tpu")


def test_cpu_import_does_not_load_cuda_dependencies():
    script = """
import importlib.abc
import sys

class BlockCudaImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'cuml', 'cupy', 'torch'}:
            raise AssertionError(f'unexpected CUDA import: {fullname}')
        return None

sys.meta_path.insert(0, BlockCudaImports())
import numpy as np
from brainmodel_utils.neural_mappers import PercentileNeuralMap, SKLinearNeuralMap
X = np.arange(24, dtype=float).reshape(8, 3)
Y = X[:, :2]
linear = SKLinearNeuralMap()
linear.fit(X, Y)
linear.predict(X)
percentile = PercentileNeuralMap()
percentile.fit(X, Y)
percentile.predict(X)
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_auto_falls_back_when_cuda_is_unavailable(monkeypatch):
    monkeypatch.setattr(backends, "cuda_available", lambda: False)
    X, Y = _regression_data()
    mapper = SKLinearNeuralMap(backend="auto")

    mapper.fit(X, Y)

    assert isinstance(mapper._mapper, Ridge)


def test_percentile_auto_falls_back_when_cuda_is_unavailable(monkeypatch):
    monkeypatch.setattr(backends, "cuda_available", lambda: False)
    X, Y = _regression_data()
    mapper = PercentileNeuralMap(percentile=50, identity=False, backend="auto")

    mapper.fit(X, Y)

    assert all(isinstance(regressor, Ridge) for regressor in mapper._mappers)


@pytest.mark.parametrize(
    ("regression_type", "regression_kwargs", "expected_type"),
    [
        ("Lasso", {"alpha": 0.01}, Lasso),
        ("Ridge", {"positive": True}, Ridge),
    ],
)
def test_auto_uses_cpu_for_unsupported_configuration(
    monkeypatch, regression_type, regression_kwargs, expected_type
):
    monkeypatch.setattr(backends, "cuda_available", lambda: True)
    X, Y = _regression_data()
    mapper = SKLinearNeuralMap(
        regression_type=regression_type,
        regression_kwargs=regression_kwargs,
        backend="auto",
    )

    mapper.fit(X, Y[:, 0])

    assert isinstance(mapper._mapper, expected_type)


def test_per_target_alpha_falls_back_or_errors(monkeypatch):
    monkeypatch.setattr(backends, "cuda_available", lambda: True)
    X, Y = _regression_data()
    alpha = np.array([0.5, 1.0, 2.0])

    auto_mapper = SKLinearNeuralMap(regression_kwargs={"alpha": alpha}, backend="auto")
    auto_mapper.fit(X, Y)

    assert isinstance(auto_mapper._mapper, Ridge)
    with pytest.raises(ValueError, match="scalar alpha"):
        SKLinearNeuralMap(regression_kwargs={"alpha": alpha}, backend="cuda").fit(X, Y)


@pytest.mark.parametrize("solver", [None, "eig"])
def test_auto_rejects_solver_without_cpu_fallback(solver):
    X, Y = _regression_data()
    mapper = SKLinearNeuralMap(
        regression_kwargs={"solver": solver},
        backend="auto",
    )

    with pytest.raises(ValueError, match="solver"):
        mapper.fit(X, Y)


def test_auto_handles_known_cuml_import_incompatibility(monkeypatch):
    def incompatible_cuml():
        raise AttributeError("BaseEstimator._get_default_requests")

    monkeypatch.setattr(backends, "_load_cuml_ridge", incompatible_cuml)
    X, Y = _regression_data()
    mapper = SKLinearNeuralMap(backend="auto")

    mapper.fit(X, Y)

    assert mapper._resolved_backend == "cpu"


def test_cuda_consistency_rejects_multiple_jobs():
    X, Y = _regression_data()
    target = np.stack((Y, Y + 0.1, Y - 0.1, Y + 0.2))
    map_kwargs = {
        "map_type": "sklinear",
        "map_kwargs": {"backend": "cuda"},
    }

    with pytest.raises(ValueError, match="num_parallel_jobs=1"):
        get_linregress_consistency(
            source=X,
            target=target,
            map_kwargs=map_kwargs,
            num_bootstrap_iters=2,
            num_parallel_jobs=2,
        )


def test_auto_cpu_consistency_keeps_parallelism(monkeypatch):
    monkeypatch.setattr(backends, "cuda_available", lambda: False)
    monkeypatch.setattr(backends, "torch_cuda_available", lambda: False)
    map_kwargs = {
        "map_type": "percentile",
        "map_kwargs": {"percentile": 50, "identity": False, "backend": "auto"},
    }

    assert not _mapping_uses_cuda(map_kwargs)


@pytest.mark.parametrize("mapper", [SKLinearNeuralMap, PercentileNeuralMap])
def test_explicit_cuda_reports_unavailable_runtime(mapper, monkeypatch):
    monkeypatch.setattr(backends, "cuda_available", lambda: False)
    monkeypatch.setattr(backends, "torch_cuda_available", lambda: False)
    X, Y = _regression_data()

    with pytest.raises(RuntimeError, match="CUDA|cuML"):
        mapper(backend="cuda").fit(X, Y)


@pytest.mark.parametrize("one_dimensional", [False, True])
def test_sklinear_cpu_matches_sklearn(one_dimensional, monkeypatch):
    monkeypatch.setattr(
        backends,
        "cuda_available",
        lambda: pytest.fail("the CPU backend queried CUDA availability"),
    )
    X, Y = _regression_data()
    target = Y[:, 0] if one_dimensional else Y
    kwargs = {"alpha": 0.75, "fit_intercept": True}
    expected = Ridge(**kwargs).fit(X, target).predict(X)
    mapper = SKLinearNeuralMap(regression_kwargs=kwargs)

    mapper.fit(X, target)
    actual = mapper.predict(X)

    expected = expected[:, None] if one_dimensional else expected
    np.testing.assert_allclose(actual, expected)
    assert actual.shape == (X.shape[0], 1 if one_dimensional else Y.shape[1])
    assert actual.dtype == expected.dtype
    assert mapper._n_source == X.shape[1]
    assert mapper._n_targets == expected.shape[1]
    assert mapper._fitted
    assert isinstance(mapper._mapper, Ridge)


@pytest.mark.parametrize("perturbation", [0.0, 1e-5])
def test_batched_ridge_solve_handles_float32_collinear_columns(perturbation):
    rng = np.random.default_rng(190)
    basis = rng.normal(size=(64, 3)).astype(np.float32)
    repeated = basis[:, 0] + perturbation * rng.normal(size=basis.shape[0])
    X = np.column_stack((basis[:, 0], repeated, basis[:, 1], basis[:, 2])).astype(
        np.float32
    )
    Y = np.column_stack(
        (2 * basis[:, 0] - basis[:, 1], -0.5 * basis[:, 0] + basis[:, 2])
    )
    alpha = 1e-6
    expected = Ridge(alpha=alpha).fit(X.astype(np.float64), Y).predict(X)

    weights, intercept, _promoted = backends._solve_ridge(np, X, Y, alpha, True)
    actual = X @ weights + intercept

    assert weights.dtype == intercept.dtype == np.float32
    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)


def test_sklinear_saves_fitted_weights(tmp_path):
    X, Y = _regression_data()
    weights_path = tmp_path / "weights.npz"
    mapper = SKLinearNeuralMap(weights_save_nm=weights_path)

    mapper.fit(X, Y)

    np.testing.assert_allclose(mapper._weights, mapper._mapper.coef_)
    with np.load(weights_path) as saved:
        np.testing.assert_allclose(saved["arr_0"], mapper._mapper.coef_)


def test_percentile_uses_signed_correlations():
    target = np.arange(8, dtype=float)
    X = np.column_stack((-target, target, np.roll(target, 1)))
    mapper = PercentileNeuralMap()

    mapper.fit(X, target)

    np.testing.assert_allclose(mapper._corrs[:2, 0], [-1.0, 1.0])
    np.testing.assert_array_equal(mapper.predict(X)[:, 0], X[:, 1])


def test_percentile_identity_uses_first_source_when_correlations_tie():
    target = np.arange(8, dtype=float)
    X = np.column_stack((target, target, -target))
    mapper = PercentileNeuralMap()

    mapper.fit(X, target)

    assert mapper._mappers == [np.int64(0)]
    np.testing.assert_array_equal(mapper.predict(X)[:, 0], X[:, 0])


def test_percentile_constant_target_keeps_cpu_fallback():
    X = np.arange(24, dtype=float).reshape(8, 3)
    target = np.ones(8)
    mapper = PercentileNeuralMap()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mapper.fit(X, target)

    assert mapper._mappers == [np.int64(X.shape[1] - 1)]
    np.testing.assert_array_equal(mapper.predict(X)[:, 0], X[:, -1])


def test_percentile_accepts_xarray_like_numpy():
    X, Y = _regression_data()
    numpy_mapper = PercentileNeuralMap()
    xarray_mapper = PercentileNeuralMap()

    numpy_mapper.fit(X, Y)
    xarray_mapper.fit(xr.DataArray(X), xr.DataArray(Y))

    np.testing.assert_allclose(xarray_mapper._corrs, numpy_mapper._corrs)
    np.testing.assert_allclose(
        xarray_mapper.predict(xr.DataArray(X)), numpy_mapper.predict(X)
    )


def test_percentile_regression_fits_one_mapper_per_target():
    X, Y = _regression_data()
    mapper = PercentileNeuralMap(
        percentile=50,
        identity=False,
        regression_kwargs={"alpha": 0.5},
    )

    mapper.fit(X, Y)
    predictions = mapper.predict(X)

    assert len(mapper._mappers) == Y.shape[1]
    assert all(isinstance(regressor, Ridge) for regressor in mapper._mappers)
    assert predictions.shape == Y.shape
    assert np.isfinite(predictions).all()


def test_percentile_saves_variable_length_weights(tmp_path):
    rng = np.random.default_rng(91)
    first = rng.normal(size=40)
    second = rng.normal(size=40)
    X = np.column_stack((first, first, second, rng.normal(size=40)))
    Y = np.column_stack((first, second))
    weights_path = tmp_path / "percentile_weights.npz"
    mapper = PercentileNeuralMap(
        percentile=75,
        identity=False,
        weights_save_nm=weights_path,
    )

    mapper.fit(X, Y)

    assert [weights.shape for weights in mapper._weights] == [(2,), (1,)]
    with np.load(weights_path, allow_pickle=True) as saved:
        weights = saved["arr_0"]
    assert weights.dtype == object
    for actual, expected in zip(weights, mapper._weights):
        np.testing.assert_allclose(actual, expected)


def test_percentile_refit_recomputes_thresholds():
    X, Y = _regression_data()
    mapper = PercentileNeuralMap(percentile=50, identity=False)

    mapper.fit(X, Y)
    first_thresholds = mapper._thresholds.copy()
    mapper.fit(X, Y[:, ::-1])

    np.testing.assert_allclose(mapper._thresholds, first_thresholds[::-1])
