import inspect

import numpy as np

__all__ = [
    "can_batch_cuda_ridge",
    "correlations_cuda",
    "cuda_available",
    "fit_grouped_ridge_cuda",
    "make_regressor",
    "make_resolved_regressor",
    "resolve_backend",
    "resolve_torch_backend",
    "torch_cuda_available",
    "validate_backend",
]

_BACKENDS = {"auto", "cpu", "cuda"}
_CUDA_RIDGE_KWARGS = {"alpha", "fit_intercept", "solver"}
_CUDA_RIDGE_SOLVERS = {"auto", "svd"}
_CUDA_CORRELATION_TARGET_BATCH_SIZE = 512


def validate_backend(backend):
    if backend not in _BACKENDS:
        choices = ", ".join(sorted(_BACKENDS))
        raise ValueError(f"backend must be one of {choices}; got {backend!r}.")
    return backend


def _load_cuml_ridge():
    from cuml.linear_model import Ridge

    return Ridge


def _load_cupy():
    import cupy

    return cupy


def _load_torch():
    import torch

    return torch


def cuda_available():
    try:
        _load_cuml_ridge()
        cupy = _load_cupy()
        return cupy.cuda.runtime.getDeviceCount() > 0
    except (AttributeError, ImportError, OSError, RuntimeError):
        return False


def torch_cuda_available():
    try:
        return _load_torch().cuda.is_available()
    except (ImportError, OSError, RuntimeError):
        return False


def _ridge_config_error(regression_type, regression_kwargs):
    if regression_type != "Ridge":
        return f"CUDA backend supports only regression_type='Ridge'; got {regression_type!r}."

    unsupported = set(regression_kwargs) - _CUDA_RIDGE_KWARGS
    if unsupported:
        names = ", ".join(sorted(unsupported))
        return f"CUDA Ridge does not support regression kwargs: {names}."

    solver = regression_kwargs.get("solver", "auto")
    if solver not in _CUDA_RIDGE_SOLVERS:
        return f"CUDA Ridge does not support solver={solver!r}."

    alpha = regression_kwargs.get("alpha", 1.0)
    if np.ndim(alpha) != 0:
        return "CUDA Ridge requires a scalar alpha."
    try:
        alpha = float(alpha)
    except (TypeError, ValueError):
        return "CUDA Ridge requires a numeric alpha."
    if not np.isfinite(alpha) or alpha < 0:
        return "CUDA Ridge requires a finite, non-negative alpha."

    return None


def _installed_ridge_config_error(regression_kwargs):
    try:
        parameters = inspect.signature(_load_cuml_ridge()).parameters
    except (TypeError, ValueError):
        return None

    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return None

    unsupported = set(regression_kwargs) - set(parameters)
    if unsupported:
        names = ", ".join(sorted(unsupported))
        return f"Installed cuML Ridge does not support regression kwargs: {names}."
    return None


def resolve_backend(backend, regression_type, regression_kwargs):
    validate_backend(backend)
    if backend == "cpu":
        return "cpu"

    config_error = _ridge_config_error(regression_type, regression_kwargs)
    if config_error is not None:
        if backend == "cuda":
            raise ValueError(config_error)
        if regression_kwargs.get("solver", "auto") in (None, "eig"):
            raise ValueError(config_error)
        return "cpu"

    if not cuda_available():
        if backend == "cuda":
            raise RuntimeError(
                "CUDA backend is unavailable; install a CUDA-enabled cuML stack "
                "and ensure an NVIDIA GPU is visible."
            )
        return "cpu"

    config_error = _installed_ridge_config_error(regression_kwargs)
    if config_error is not None:
        if backend == "cuda":
            raise ValueError(config_error)
        return "cpu"

    return "cuda"


def resolve_torch_backend(backend):
    validate_backend(backend)
    if backend == "cpu":
        return "cpu"
    if torch_cuda_available():
        return "cuda"
    if backend == "cuda":
        raise RuntimeError(
            "CUDA percentile mapping is unavailable; install CUDA-enabled PyTorch "
            "and ensure an NVIDIA GPU is visible."
        )
    return "cpu"


class _CuMLRidge:
    def __init__(self, regression_kwargs):
        self._ridge = _load_cuml_ridge()
        self._regression_kwargs = regression_kwargs
        self._estimators = None

    def fit(self, X, Y):
        cupy = _load_cupy()
        X_array = np.ascontiguousarray(np.asarray(X))
        Y_array = np.ascontiguousarray(np.asarray(Y))
        dtype = _cuda_dtype(X_array, Y_array)
        X_device = cupy.ascontiguousarray(cupy.asarray(X_array, dtype=dtype))
        Y_device = cupy.ascontiguousarray(cupy.asarray(Y_array, dtype=dtype))
        self._n_features = X_array.shape[1]
        self._target_ndim = Y_array.ndim
        self._n_targets = 1 if Y_array.ndim == 1 else Y_array.shape[1]

        if self._n_targets > 1 and can_batch_cuda_ridge(self._regression_kwargs):
            weights, intercept = _solve_ridge(
                cupy,
                X_device,
                Y_device,
                float(self._regression_kwargs.get("alpha", 1.0)),
                self._regression_kwargs.get("fit_intercept", True),
            )
            self._coef_device = weights.T
            self._intercept_device = intercept.reshape(-1)
            self.coef_ = cupy.asnumpy(self._coef_device)
            self.intercept_ = self._normalize_intercept(
                cupy.asnumpy(self._intercept_device)
            )
            self._estimators = None
            self._estimator = None
            return self

        targets = (
            [Y_device]
            if Y_array.ndim == 1
            else [
                cupy.ascontiguousarray(Y_device[:, target_idx])
                for target_idx in range(self._n_targets)
            ]
        )
        self._estimators = [
            self._ridge(output_type="numpy", **self._regression_kwargs)
            for _ in range(self._n_targets)
        ]
        for estimator, target in zip(self._estimators, targets):
            estimator.fit(X_device, target)

        self._estimator = self._estimators[0]
        coefficients = np.stack(
            [np.asarray(estimator.coef_).reshape(-1) for estimator in self._estimators]
        )
        intercepts = np.asarray(
            [
                np.asarray(estimator.intercept_).reshape(-1)[0]
                for estimator in self._estimators
            ]
        )
        if self._n_targets == 1:
            coefficients = coefficients[0]
        if self._target_ndim == 1:
            intercepts = intercepts[0]
        self.coef_ = self._normalize_coefficients(coefficients)
        self.intercept_ = self._normalize_intercept(intercepts)
        return self

    def predict(self, X):
        cupy = _load_cupy()
        X_array = np.ascontiguousarray(np.asarray(X))
        dtype = _cuda_dtype(X_array, self.coef_)
        X_device = cupy.ascontiguousarray(cupy.asarray(X_array, dtype=dtype))
        if self._estimators is None:
            predictions = X_device @ self._coef_device.T
            predictions += self._intercept_device
            return cupy.asnumpy(predictions)
        predictions = np.column_stack(
            [
                np.asarray(estimator.predict(X_device)).reshape(-1)
                for estimator in self._estimators
            ]
        )
        if self._n_targets == 1:
            return predictions[:, 0]
        return predictions

    def score(self, X, Y, sample_weight=None):
        from sklearn.metrics import r2_score

        return r2_score(
            np.asarray(Y),
            self.predict(X),
            sample_weight=sample_weight,
            multioutput="uniform_average",
        )

    def _normalize_coefficients(self, coefficients):
        coefficients = np.asarray(coefficients)
        if self._n_targets == 1:
            return coefficients.reshape(self._n_features)
        if coefficients.shape == (self._n_targets, self._n_features):
            return coefficients
        if coefficients.shape == (self._n_features, self._n_targets):
            return coefficients.T
        raise RuntimeError(
            f"Unexpected cuML Ridge coefficient shape {coefficients.shape}."
        )

    def _normalize_intercept(self, intercept):
        if not self._regression_kwargs.get("fit_intercept", True):
            return 0.0
        intercept = np.asarray(intercept)
        if self._target_ndim == 1:
            return intercept.reshape(-1)[0]
        return intercept.reshape(self._n_targets)


class _FittedRidge:
    def __init__(self, coefficients, intercept):
        self.coef_ = np.asarray(coefficients)
        self.intercept_ = np.asarray(intercept).reshape(1)

    def predict(self, X):
        return np.asarray(X) @ self.coef_.reshape(-1) + self.intercept_

    def score(self, X, Y, sample_weight=None):
        from sklearn.metrics import r2_score

        return r2_score(
            np.asarray(Y).reshape(-1),
            self.predict(X),
            sample_weight=sample_weight,
        )


class _GroupedRidge:
    def __init__(self, cupy, selections, models, dtype, batch_size):
        self._cupy = cupy
        self._batches = []
        for start in range(0, len(models), batch_size):
            stop = min(start + batch_size, len(models))
            batch_selections = selections[start:stop]
            batch_features = max(indices.size for indices in batch_selections)
            indices = np.zeros((stop - start, batch_features), dtype=np.int32)
            coefficients = np.zeros((stop - start, batch_features), dtype=dtype)
            intercepts = np.empty(stop - start, dtype=dtype)
            for target, (selected, model) in enumerate(
                zip(batch_selections, models[start:stop])
            ):
                indices[target, : selected.size] = selected
                coefficients[target, : selected.size] = model.coef_
                intercepts[target] = model.intercept_[0]
            self._batches.append(
                (
                    start,
                    stop,
                    cupy.asarray(indices),
                    cupy.asarray(coefficients),
                    cupy.asarray(intercepts),
                )
            )

    def predict(self, X):
        cupy = self._cupy
        X = cupy.asarray(np.ascontiguousarray(np.asarray(X)))
        predictions = cupy.empty((X.shape[0], self._batches[-1][1]), dtype=cupy.float64)
        for start, stop, indices, coefficients, intercepts in self._batches:
            selected = X[:, indices]
            predictions[:, start:stop] = (selected * coefficients[np.newaxis]).sum(
                axis=-1
            ) + intercepts
        return cupy.asnumpy(predictions)


def _cuda_dtype(*arrays):
    dtype = np.result_type(*(np.asarray(array).dtype for array in arrays))
    return np.float64 if dtype == np.float64 else np.float32


def can_batch_cuda_ridge(regression_kwargs):
    solver = regression_kwargs.get("solver")
    alpha = regression_kwargs.get("alpha", 1.0)
    try:
        alpha = float(alpha)
    except (TypeError, ValueError):
        return False
    return solver in (None, "auto") and np.isfinite(alpha) and alpha > 0


def _solve_ridge(cupy, X, Y, alpha, fit_intercept):
    if fit_intercept:
        X_offset = X.mean(axis=-2, keepdims=True)
        Y_offset = Y.mean(axis=-2, keepdims=True)
        X_centered = X - X_offset
        Y_centered = Y - Y_offset
    else:
        X_offset = cupy.zeros(X.shape[:-2] + (1, X.shape[-1]), dtype=X.dtype)
        Y_offset = cupy.zeros(Y.shape[:-2] + (1, Y.shape[-1]), dtype=Y.dtype)
        X_centered = X
        Y_centered = Y

    if X.shape[-1] > X.shape[-2]:
        gram = X_centered @ cupy.swapaxes(X_centered, -1, -2)
        diagonal = cupy.arange(gram.shape[-1])
        gram[..., diagonal, diagonal] += alpha
        dual = cupy.linalg.solve(gram, Y_centered)
        weights = cupy.swapaxes(X_centered, -1, -2) @ dual
    else:
        gram = cupy.swapaxes(X_centered, -1, -2) @ X_centered
        diagonal = cupy.arange(gram.shape[-1])
        gram[..., diagonal, diagonal] += alpha
        weights = cupy.linalg.solve(
            gram, cupy.swapaxes(X_centered, -1, -2) @ Y_centered
        )
    intercept = Y_offset - X_offset @ weights
    return weights, intercept


def _grouped_batch_size(n_samples, n_features, n_targets, itemsize):
    matrix_width = min(n_samples, n_features)
    bytes_per_target = itemsize * (
        3 * n_samples * n_features + 3 * matrix_width * matrix_width
    )
    return max(1, min(n_targets, 1024, (512 * 1024**2) // bytes_per_target))


def fit_grouped_ridge_cuda(X, Y, source_indices, regression_kwargs):
    if not can_batch_cuda_ridge(regression_kwargs):
        raise ValueError("batched CUDA Ridge requires solver='auto' and alpha > 0.")

    cupy = _load_cupy()
    X = np.ascontiguousarray(np.asarray(X))
    Y = np.ascontiguousarray(np.asarray(Y))
    dtype = _cuda_dtype(X, Y)
    selections = [
        np.flatnonzero(indices)
        if np.asarray(indices).dtype == np.bool_
        else np.asarray(indices, dtype=np.int64)
        for indices in source_indices
    ]
    max_features = max(indices.size for indices in selections)
    batch_size = _grouped_batch_size(
        X.shape[0], max_features, Y.shape[1], np.dtype(dtype).itemsize
    )
    alpha = float(regression_kwargs.get("alpha", 1.0))
    fit_intercept = regression_kwargs.get("fit_intercept", True)
    models = []

    for start in range(0, Y.shape[1], batch_size):
        stop = min(start + batch_size, Y.shape[1])
        batch_selections = selections[start:stop]
        batch_features = max(indices.size for indices in batch_selections)
        X_batch = np.zeros((stop - start, X.shape[0], batch_features), dtype=dtype)
        for target, indices in enumerate(batch_selections):
            X_batch[target, :, : indices.size] = X[:, indices]
        Y_batch = np.ascontiguousarray(
            Y[:, start:stop].T[:, :, np.newaxis], dtype=dtype
        )
        weights, intercepts = _solve_ridge(
            cupy,
            cupy.asarray(X_batch),
            cupy.asarray(Y_batch),
            alpha,
            fit_intercept,
        )
        weights = cupy.asnumpy(weights[:, :, 0])
        intercepts = cupy.asnumpy(intercepts[:, 0, 0])
        models.extend(
            _FittedRidge(weights[target, : indices.size], intercepts[target])
            for target, indices in enumerate(batch_selections)
        )
    return models, _GroupedRidge(cupy, selections, models, dtype, batch_size)


def make_resolved_regressor(regression_type, regression_kwargs, backend):
    if backend == "cuda":
        return _CuMLRidge(regression_kwargs)

    from sklearn import linear_model

    return getattr(linear_model, regression_type)(**regression_kwargs)


def make_regressor(regression_type, regression_kwargs, backend="cpu"):
    resolved_backend = resolve_backend(backend, regression_type, regression_kwargs)
    return make_resolved_regressor(regression_type, regression_kwargs, resolved_backend)


def correlations_cuda(X, Y, percentile=None, return_correlations=True):
    torch = _load_torch()
    allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.no_grad():
            X_array = np.ascontiguousarray(np.asarray(X))
            X_device = torch.as_tensor(X_array, dtype=torch.float32, device="cuda")
            X_device = X_device - X_device.mean(dim=0)
            X_device = X_device / torch.linalg.vector_norm(X_device, dim=0)
            if percentile is None:
                Y_device = torch.as_tensor(
                    np.ascontiguousarray(np.asarray(Y)),
                    dtype=torch.float32,
                    device="cuda",
                )
                Y_device = Y_device - Y_device.mean(dim=0)
                Y_device = Y_device / torch.linalg.vector_norm(Y_device, dim=0)
                correlations = X_device.T @ Y_device
                return correlations.cpu().numpy()

            Y = np.ascontiguousarray(np.asarray(Y))
            correlations = None
            if return_correlations:
                correlations = np.empty(
                    (X_device.shape[1], Y.shape[1]), dtype=np.float32
                )
            thresholds = np.empty(Y.shape[1], dtype=np.float32)
            selections = []
            source_rank = np.arange(X_device.shape[1])
            fallback = np.flatnonzero(
                source_rank >= np.nanpercentile(source_rank, percentile)
            )
            for start in range(0, Y.shape[1], _CUDA_CORRELATION_TARGET_BATCH_SIZE):
                stop = min(start + _CUDA_CORRELATION_TARGET_BATCH_SIZE, Y.shape[1])
                Y_device = torch.as_tensor(
                    Y[:, start:stop], dtype=torch.float32, device="cuda"
                )
                Y_device = Y_device - Y_device.mean(dim=0)
                Y_device = Y_device / torch.linalg.vector_norm(Y_device, dim=0)
                correlation_batch = X_device.T @ Y_device
                finite_counts = torch.isfinite(correlation_batch).sum(dim=0)
                safe_counts = finite_counts.clamp_min(1)
                rank = (safe_counts.to(correlation_batch.dtype) - 1) * (
                    percentile / 100.0
                )
                rank_low = torch.floor(rank).to(torch.int64)
                rank_high = torch.ceil(rank).to(torch.int64)
                descending_low = safe_counts - 1 - rank_low
                descending_high = safe_counts - 1 - rank_high
                top_values = torch.topk(
                    torch.nan_to_num(
                        correlation_batch,
                        nan=-torch.inf,
                        posinf=torch.inf,
                        neginf=-torch.inf,
                    ),
                    int(descending_low.max().item()) + 1,
                    dim=0,
                ).values
                columns = torch.arange(stop - start, device=X_device.device)
                low_values = top_values[descending_low, columns]
                high_values = top_values[descending_high, columns]
                fraction = rank - rank_low
                difference = high_values - low_values
                threshold_batch = low_values + difference * fraction
                threshold_batch = torch.where(
                    fraction >= 0.5,
                    high_values - difference * (1 - fraction),
                    threshold_batch,
                )
                threshold_batch[finite_counts == 0] = torch.nan
                selected_batch = (correlation_batch >= threshold_batch).T
                if return_correlations:
                    correlations[:, start:stop] = correlation_batch.cpu().numpy()
                thresholds[start:stop] = threshold_batch.cpu().numpy()
                selected_rows, selected_sources = selected_batch.nonzero(as_tuple=True)
                counts = (
                    torch.bincount(selected_rows, minlength=stop - start).cpu().numpy()
                )
                selected_sources = selected_sources.cpu().numpy()
                offsets = np.concatenate(([0], np.cumsum(counts)))
                selections.extend(
                    selected_sources[offsets[target] : offsets[target + 1]].copy()
                    if counts[target]
                    else fallback.copy()
                    for target in range(stop - start)
                )
            return correlations, thresholds, selections
    finally:
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
