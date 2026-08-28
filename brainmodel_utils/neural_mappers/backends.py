import numpy as np

from .profiling import phase

__all__ = [
    "CORRELATION_CUTOFF_EPS_MULTIPLIER",
    "DEFAULT_CUDA_CORRELATION_TARGET_BATCH_SIZE",
    "DEFAULT_CUDA_RIDGE_MAX_BATCH_SIZE",
    "DEFAULT_CUDA_RIDGE_WORKSPACE_BYTES",
    "can_batch_cuda_ridge",
    "correlation_cutoff_tolerance",
    "correlations_cuda",
    "cuda_available",
    "fit_grouped_ridge_cuda",
    "last_cuda_batch_report",
    "make_regressor",
    "make_resolved_regressor",
    "resolve_backend",
    "resolve_correlation_target_batch_size",
    "resolve_ridge_max_batch_size",
    "resolve_torch_backend",
    "set_last_cuda_batch_report",
    "torch_cuda_available",
    "validate_backend",
]

_BACKENDS = {"cpu", "cuda"}
_CUDA_RIDGE_KWARGS = {"alpha", "fit_intercept", "solver"}
_CUDA_RIDGE_SOLVERS = {"auto"}
DEFAULT_CUDA_CORRELATION_TARGET_BATCH_SIZE = 512
DEFAULT_CUDA_RIDGE_WORKSPACE_BYTES = 512 * 1024**2
DEFAULT_CUDA_RIDGE_MAX_BATCH_SIZE = 1024
CORRELATION_CUTOFF_EPS_MULTIPLIER = 8
_CUDA_CORRELATION_TARGET_BATCH_SIZE = DEFAULT_CUDA_CORRELATION_TARGET_BATCH_SIZE
LAST_CUDA_BATCH_REPORT = None


def resolve_ridge_max_batch_size(n_targets, max_batch_size=None):
    n_targets = int(n_targets)
    if n_targets < 1:
        raise ValueError("n_targets must be positive")
    if max_batch_size is None:
        return min(n_targets, DEFAULT_CUDA_RIDGE_MAX_BATCH_SIZE)
    if isinstance(max_batch_size, str):
        if max_batch_size.lower() != "all":
            raise ValueError("ridge max batch size must be a positive integer or 'all'")
        return n_targets
    max_batch_size = int(max_batch_size)
    if max_batch_size < 1:
        raise ValueError("ridge max batch size must be positive")
    return min(n_targets, max_batch_size)


def set_last_cuda_batch_report(report):
    global LAST_CUDA_BATCH_REPORT
    LAST_CUDA_BATCH_REPORT = None if report is None else dict(report)
    return LAST_CUDA_BATCH_REPORT


def last_cuda_batch_report():
    if LAST_CUDA_BATCH_REPORT is None:
        return None
    return dict(LAST_CUDA_BATCH_REPORT)


def resolve_correlation_target_batch_size(n_targets, target_batch_size=None):
    if n_targets < 1:
        raise ValueError("n_targets must be positive")
    if target_batch_size is None:
        target_batch_size = DEFAULT_CUDA_CORRELATION_TARGET_BATCH_SIZE
    if isinstance(target_batch_size, str):
        if target_batch_size.lower() != "all":
            raise ValueError(
                "correlation target batch size must be a positive integer or 'all'"
            )
        return int(n_targets)
    target_batch_size = int(target_batch_size)
    if target_batch_size < 1:
        raise ValueError("correlation target batch size must be positive")
    return min(int(n_targets), target_batch_size)


def validate_backend(backend):
    if backend not in _BACKENDS:
        choices = ", ".join(sorted(_BACKENDS))
        raise ValueError(f"backend must be one of {choices}; got {backend!r}.")
    return backend


def _load_cupy():
    import cupy

    return cupy


def _load_torch():
    import torch

    return torch


def cuda_available():
    try:
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
    if not np.isfinite(alpha) or alpha <= 0:
        return "CUDA Ridge requires a finite, positive alpha."

    fit_intercept = regression_kwargs.get("fit_intercept", True)
    if not isinstance(fit_intercept, (bool, np.bool_)):
        return "CUDA Ridge requires fit_intercept to be boolean."

    return None


def resolve_backend(backend, regression_type, regression_kwargs):
    validate_backend(backend)
    if backend == "cpu":
        return "cpu"

    config_error = _ridge_config_error(regression_type, regression_kwargs)
    if config_error is not None:
        raise ValueError(config_error)

    if not cuda_available():
        raise RuntimeError(
            "CUDA backend is unavailable; install CUDA-enabled CuPy "
            "and ensure an NVIDIA GPU is visible."
        )

    return "cuda"


def resolve_torch_backend(backend):
    validate_backend(backend)
    if backend == "cpu":
        return "cpu"
    if torch_cuda_available():
        return "cuda"
    raise RuntimeError(
        "CUDA percentile mapping is unavailable; install CUDA-enabled PyTorch "
        "and ensure an NVIDIA GPU is visible."
    )


class _CudaRidge:
    def __init__(self, regression_kwargs):
        self._regression_kwargs = dict(regression_kwargs)

    def fit(self, X, Y):
        cupy = _load_cupy()
        X_array = np.ascontiguousarray(np.asarray(X))
        Y_array = np.ascontiguousarray(np.asarray(Y))
        if Y_array.ndim not in (1, 2):
            raise ValueError("CUDA Ridge targets must be one- or two-dimensional.")
        dtype = _cuda_dtype(X_array, Y_array)
        X_device = cupy.ascontiguousarray(cupy.asarray(X_array, dtype=dtype))
        Y_matrix = Y_array[:, np.newaxis] if Y_array.ndim == 1 else Y_array
        Y_device = cupy.ascontiguousarray(cupy.asarray(Y_matrix, dtype=dtype))
        self._n_features = X_array.shape[1]
        self._target_ndim = Y_array.ndim
        self._n_targets = Y_matrix.shape[1]
        weights, intercept, promoted = _solve_ridge(
            cupy,
            X_device,
            Y_device,
            float(self._regression_kwargs.get("alpha", 1.0)),
            self._regression_kwargs.get("fit_intercept", True),
        )
        self.promoted_to_float64 = promoted
        self.compute_dtype = np.dtype(dtype)
        self._coef_device = weights.T
        self._intercept_device = intercept.reshape(-1)
        coefficients = cupy.asnumpy(self._coef_device)
        intercepts = cupy.asnumpy(self._intercept_device)
        if self._n_targets == 1:
            coefficients = coefficients[0]
        if self._target_ndim == 1:
            intercepts = intercepts[0]
        self.coef_ = coefficients
        self.intercept_ = (
            intercepts if self._regression_kwargs.get("fit_intercept", True) else 0.0
        )
        return self

    def predict(self, X):
        cupy = _load_cupy()
        X_array = np.ascontiguousarray(np.asarray(X))
        dtype = _cuda_dtype(X_array, self.coef_)
        X_device = cupy.ascontiguousarray(cupy.asarray(X_array, dtype=dtype))
        predictions = X_device @ self._coef_device.T
        predictions += self._intercept_device
        self.prediction_compute_dtype = np.dtype(predictions.dtype)
        predictions = cupy.asnumpy(predictions)
        self.prediction_output_dtype = predictions.dtype
        return predictions[:, 0] if self._n_targets == 1 else predictions

    def score(self, X, Y, sample_weight=None):
        from sklearn.metrics import r2_score

        return r2_score(
            np.asarray(Y),
            self.predict(X),
            sample_weight=sample_weight,
            multioutput="uniform_average",
        )


class _FittedRidge:
    def __init__(self, coefficients, intercept):
        self.coef_ = np.asarray(coefficients).reshape(-1)
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
        self.compute_dtype = np.dtype(dtype)
        self.prediction_output_dtype = np.dtype(np.float64)
        self.shared_design = False
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
                coefficients[target, : selected.size] = model.coef_.reshape(-1)
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


class _SharedDesignRidge:
    def __init__(self, cupy, selection, models, dtype, batch_size):
        self._cupy = cupy
        self._selection = cupy.asarray(selection, dtype=cupy.int64)
        self.compute_dtype = np.dtype(dtype)
        self.prediction_output_dtype = np.dtype(np.float64)
        self.shared_design = True
        self._batches = []
        for start in range(0, len(models), batch_size):
            stop = min(start + batch_size, len(models))
            coefficients = np.stack(
                [model.coef_.reshape(-1) for model in models[start:stop]], axis=0
            )
            intercepts = np.asarray(
                [model.intercept_[0] for model in models[start:stop]], dtype=dtype
            )
            self._batches.append(
                (
                    start,
                    stop,
                    cupy.asarray(coefficients, dtype=dtype),
                    cupy.asarray(intercepts, dtype=dtype),
                )
            )

    def predict(self, X):
        cupy = self._cupy
        X = cupy.asarray(np.ascontiguousarray(np.asarray(X)))
        selected = X[:, self._selection]
        predictions = cupy.empty((X.shape[0], self._batches[-1][1]), dtype=cupy.float64)
        for start, stop, coefficients, intercepts in self._batches:
            predictions[:, start:stop] = selected @ coefficients.T + intercepts
        return cupy.asnumpy(predictions)


def _cuda_dtype(*arrays):
    dtype = np.result_type(*(np.asarray(array).dtype for array in arrays))
    return np.float64 if dtype == np.float64 else np.float32


def can_batch_cuda_ridge(regression_kwargs):
    solver = regression_kwargs.get("solver", "auto")
    alpha = regression_kwargs.get("alpha", 1.0)
    try:
        alpha = float(alpha)
    except (TypeError, ValueError):
        return False
    return solver == "auto" and np.isfinite(alpha) and alpha > 0


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

    use_dual = X.shape[-1] > X.shape[-2]
    X_transposed = cupy.swapaxes(X_centered, -1, -2)
    gram = X_centered @ X_transposed if use_dual else X_transposed @ X_centered

    if use_dual:
        diagonal = cupy.arange(gram.shape[-1])
        gram[..., diagonal, diagonal] += alpha
        dual = cupy.linalg.solve(gram, Y_centered)
        weights = X_transposed @ dual
    else:
        diagonal = cupy.arange(gram.shape[-1])
        gram[..., diagonal, diagonal] += alpha
        weights = cupy.linalg.solve(gram, X_transposed @ Y_centered)
    intercept = Y_offset - X_offset @ weights
    return weights, intercept, False


def _grouped_batch_size(
    n_samples,
    n_features,
    n_targets,
    itemsize,
    workspace_bytes=None,
    max_batch_size=None,
):
    if workspace_bytes is None:
        workspace_bytes = DEFAULT_CUDA_RIDGE_WORKSPACE_BYTES
    workspace_bytes = int(workspace_bytes)
    if workspace_bytes < 1:
        raise ValueError("ridge workspace bytes must be positive")
    matrix_width = min(n_samples, n_features)
    bytes_per_target = itemsize * (
        3 * n_samples * n_features + 3 * matrix_width * matrix_width
    )
    fitted = max(1, workspace_bytes // max(1, bytes_per_target))
    cap = resolve_ridge_max_batch_size(n_targets, max_batch_size)
    return max(1, min(n_targets, cap, fitted)), bytes_per_target


def _shared_design_batch_size(
    n_samples,
    n_features,
    n_targets,
    itemsize,
    workspace_bytes=None,
    max_batch_size=None,
):
    if workspace_bytes is None:
        workspace_bytes = DEFAULT_CUDA_RIDGE_WORKSPACE_BYTES
    workspace_bytes = int(workspace_bytes)
    if workspace_bytes < 1:
        raise ValueError("ridge workspace bytes must be positive")
    matrix_width = min(n_samples, n_features)
    bytes_per_target = itemsize * (3 * n_samples + 2 * n_features + matrix_width)
    fitted = max(1, workspace_bytes // max(1, bytes_per_target))
    cap = resolve_ridge_max_batch_size(n_targets, max_batch_size)
    return max(1, min(n_targets, cap, fitted)), bytes_per_target


def fit_grouped_ridge_cuda(
    X,
    Y,
    source_indices,
    regression_kwargs,
    workspace_bytes=None,
    max_batch_size=None,
):
    if not can_batch_cuda_ridge(regression_kwargs):
        raise ValueError("batched CUDA Ridge requires solver='auto' and alpha > 0.")

    cupy = _load_cupy()
    X = np.ascontiguousarray(np.asarray(X))
    Y = np.ascontiguousarray(np.asarray(Y))
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("batched CUDA Ridge requires two-dimensional X and Y")
    dtype = _cuda_dtype(X, Y)
    selections = [
        np.flatnonzero(indices)
        if np.asarray(indices).dtype == np.bool_
        else np.asarray(indices, dtype=np.int64)
        for indices in source_indices
    ]
    if len(selections) != Y.shape[1]:
        raise ValueError("one source selection is required for each target")
    if any(indices.size == 0 for indices in selections):
        raise ValueError("each CUDA Ridge target requires at least one source")
    max_features = max(indices.size for indices in selections)
    shared_design = all(
        np.array_equal(selections[0], indices) for indices in selections[1:]
    )
    batch_sizer = _shared_design_batch_size if shared_design else _grouped_batch_size
    batch_size, bytes_per_target = batch_sizer(
        X.shape[0],
        max_features,
        Y.shape[1],
        np.dtype(dtype).itemsize,
        workspace_bytes=workspace_bytes,
        max_batch_size=max_batch_size,
    )
    alpha = float(regression_kwargs.get("alpha", 1.0))
    fit_intercept = regression_kwargs.get("fit_intercept", True)
    models = []

    with phase("ridge"):
        if shared_design:
            selection = selections[0]
            X_device = cupy.asarray(np.ascontiguousarray(X[:, selection], dtype=dtype))
            for start in range(0, Y.shape[1], batch_size):
                stop = min(start + batch_size, Y.shape[1])
                Y_device = cupy.asarray(
                    np.ascontiguousarray(Y[:, start:stop], dtype=dtype)
                )
                weights, intercepts, _ = _solve_ridge(
                    cupy, X_device, Y_device, alpha, fit_intercept
                )
                weights = cupy.asnumpy(weights)
                intercepts = cupy.asnumpy(intercepts).reshape(-1)
                models.extend(
                    _FittedRidge(weights[:, target], intercepts[target])
                    for target in range(stop - start)
                )
        else:
            for start in range(0, Y.shape[1], batch_size):
                stop = min(start + batch_size, Y.shape[1])
                batch_selections = selections[start:stop]
                batch_features = max(indices.size for indices in batch_selections)
                X_batch = np.zeros(
                    (stop - start, X.shape[0], batch_features), dtype=dtype
                )
                for target, indices in enumerate(batch_selections):
                    X_batch[target, :, : indices.size] = X[:, indices]
                Y_batch = np.ascontiguousarray(
                    Y[:, start:stop].T[:, :, np.newaxis], dtype=dtype
                )
                weights, intercepts, _ = _solve_ridge(
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
    if shared_design:
        grouped = _SharedDesignRidge(cupy, selections[0], models, dtype, batch_size)
    else:
        grouped = _GroupedRidge(cupy, selections, models, dtype, batch_size)
    grouped.batch_size = batch_size
    grouped.workspace_bytes = (
        DEFAULT_CUDA_RIDGE_WORKSPACE_BYTES
        if workspace_bytes is None
        else int(workspace_bytes)
    )
    grouped.bytes_per_target = bytes_per_target
    grouped.promoted_to_float64 = False
    grouped.max_batch_size = resolve_ridge_max_batch_size(Y.shape[1], max_batch_size)
    set_last_cuda_batch_report(
        {
            "n_samples": int(X.shape[0]),
            "n_targets": int(Y.shape[1]),
            "max_features": int(max_features),
            "ridge_workspace_bytes": grouped.workspace_bytes,
            "ridge_bytes_per_target": int(bytes_per_target),
            "ridge_batch_size_effective": int(batch_size),
            "ridge_max_batch_size_requested": max_batch_size,
            "ridge_max_batch_size_effective": grouped.max_batch_size,
            "ridge_promoted_to_float64": False,
            "ridge_shared_design": shared_design,
            "ridge_compute_dtype": str(grouped.compute_dtype),
            "coefficient_dtype": str(grouped.compute_dtype),
            "prediction_compute_dtype": str(grouped.compute_dtype),
            "prediction_output_dtype": str(grouped.prediction_output_dtype),
        }
    )
    return models, grouped


def make_resolved_regressor(regression_type, regression_kwargs, backend):
    if backend == "cuda":
        return _CudaRidge(regression_kwargs)

    from sklearn import linear_model

    return getattr(linear_model, regression_type)(**regression_kwargs)


def make_regressor(regression_type, regression_kwargs, backend="cpu"):
    resolved_backend = resolve_backend(backend, regression_type, regression_kwargs)
    return make_resolved_regressor(regression_type, regression_kwargs, resolved_backend)


def _torch_correlation_dtype(dtype, *arrays):
    torch = _load_torch()
    if dtype is None:
        dtype = _cuda_dtype(*arrays)
    dtype = np.dtype(dtype)
    if dtype == np.float64:
        return torch.float64, np.float64
    if dtype == np.float32:
        return torch.float32, np.float32
    raise ValueError("CUDA correlations support only float32 or float64")


def correlation_cutoff_tolerance(thresholds):
    thresholds = np.asarray(thresholds)
    dtype = thresholds.dtype
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        dtype = np.dtype(np.float64)
        thresholds = thresholds.astype(dtype)
    scale = np.maximum(1, np.abs(thresholds))
    return CORRELATION_CUTOFF_EPS_MULTIPLIER * np.finfo(dtype).eps * scale


def correlations_cuda(
    X,
    Y,
    percentile=None,
    return_correlations=True,
    target_batch_size=None,
    dtype=None,
):
    torch = _load_torch()
    torch_dtype, numpy_dtype = _torch_correlation_dtype(dtype, X, Y)
    allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with phase("correlation"), torch.no_grad():
            X_array = np.ascontiguousarray(np.asarray(X))
            X_device = torch.as_tensor(X_array, dtype=torch_dtype, device="cuda")
            X_device = X_device - X_device.mean(dim=0)
            X_device = X_device / torch.linalg.vector_norm(X_device, dim=0)
            if percentile is None:
                Y_device = torch.as_tensor(
                    np.ascontiguousarray(np.asarray(Y)),
                    dtype=torch_dtype,
                    device="cuda",
                )
                Y_device = Y_device - Y_device.mean(dim=0)
                Y_device = Y_device / torch.linalg.vector_norm(Y_device, dim=0)
                correlations = X_device.T @ Y_device
                return correlations.cpu().numpy()

            Y = np.ascontiguousarray(np.asarray(Y))
            batch_size = resolve_correlation_target_batch_size(
                Y.shape[1], target_batch_size
            )
            correlations = None
            if return_correlations:
                correlations = np.empty(
                    (X_device.shape[1], Y.shape[1]), dtype=numpy_dtype
                )
            thresholds = np.empty(Y.shape[1], dtype=numpy_dtype)
            selections = []
            source_rank = np.arange(X_device.shape[1])
            fallback = np.flatnonzero(
                source_rank >= np.nanpercentile(source_rank, percentile)
            )
            for start in range(0, Y.shape[1], batch_size):
                stop = min(start + batch_size, Y.shape[1])
                Y_device = torch.as_tensor(
                    Y[:, start:stop], dtype=torch_dtype, device="cuda"
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
                cutoff_tolerance = (
                    CORRELATION_CUTOFF_EPS_MULTIPLIER
                    * torch.finfo(correlation_batch.dtype).eps
                    * torch.maximum(
                        torch.ones_like(threshold_batch), threshold_batch.abs()
                    )
                )
                selected_batch = (
                    correlation_batch >= threshold_batch - cutoff_tolerance
                ).T
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
