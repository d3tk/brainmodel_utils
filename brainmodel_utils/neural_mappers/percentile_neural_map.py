import numpy as np

from . import neural_map_base as nm
from .backends import (
    CORRELATION_CUTOFF_EPS_MULTIPLIER,
    can_batch_cuda_ridge,
    correlation_cutoff_tolerance,
    correlations_cuda,
    fit_grouped_ridge_cuda,
    make_resolved_regressor,
    resolve_backend,
    resolve_correlation_target_batch_size,
    resolve_ridge_max_batch_size,
    resolve_torch_backend,
    set_last_cuda_batch_report,
    validate_backend,
)

__all__ = ["PercentileNeuralMap"]


class PercentileNeuralMap(nm.NeuralMapBase):
    """
    This is the implementation for performing a percentile based fitting procedure
    between source and target. Here we select the top percentile neuron(s) that best match with
    each individual target. Default is 100th percentile (e.g. one-to-one) and correlation as the metric.
    """

    def __init__(
        self,
        percentile=100,
        identity=True,
        regression_type="Ridge",
        regression_kwargs=None,
        weights_save_nm=None,
        store_correlations=True,
        correlation_target_batch_size=None,
        ridge_workspace_bytes=None,
        ridge_max_batch_size=None,
        correlation_dtype=None,
        backend="cpu",
    ):
        super().__init__()
        # Size of subset of source features that correlate well with targets.
        # Default is 100th percentile, meaning only select the best source
        # feature.
        self._regression_type = regression_type
        self._regression_kwargs = {} if regression_kwargs is None else regression_kwargs
        self._percentile = percentile
        self._identity = identity
        if self._identity:
            assert self._percentile == 100, (
                "Cannot use identity mapping if percentile != 100."
            )
        self._corrs = None
        self._thresholds = None
        self._mappers = None
        self._source_indices = None
        self._grouped_mapper = None
        self._cuda_source_indices = None
        self._weights_save_nm = weights_save_nm
        self._weights = None
        self._backend = validate_backend(backend)
        self._resolved_backend = None
        self._store_correlations = store_correlations
        self._correlation_target_batch_size = correlation_target_batch_size
        self._ridge_workspace_bytes = ridge_workspace_bytes
        self._ridge_max_batch_size = ridge_max_batch_size
        self._correlation_dtype = correlation_dtype
        self._cuda_batch_report = None

    def _gather_correlations(self, X, Y):
        """
        Subroutine to gather Pearson correlations between source and targets.

        Updates:
            self._corrs : (numpy.ndarray) of dimensions (source, targets). Entry ij
                          is the correlation between the column i in the source
                          features and column j in the targets.
        """
        assert X.ndim == Y.ndim == 2
        assert X.shape[0] == Y.shape[0]
        assert self._n_targets is not None

        if self._resolved_backend == "cuda":
            (
                self._corrs,
                self._thresholds,
                self._cuda_source_indices,
            ) = correlations_cuda(
                X,
                Y,
                self._percentile,
                return_correlations=self._store_correlations,
                target_batch_size=self._correlation_target_batch_size,
                dtype=self._correlation_dtype,
            )
            return

        # First, remove column means
        X_ = X - X.mean(axis=0)
        Y_ = Y - Y.mean(axis=0)

        # Second, normalize each column vector to norm 1
        X_ = X_ / np.linalg.norm(X_, axis=0)
        Y_ = Y_ / np.linalg.norm(Y_, axis=0)

        # Now compute dot product = Pearson correlation
        self._corrs = np.dot(X_.T, Y_)  # (source, target)

    def _gather_thresholds(self):
        self._thresholds = np.nanpercentile(self._corrs, self._percentile, axis=0)

    def fit(self, X, Y):
        """
        Performs the percentile based fitting procedure.

        Inputs:
            X : (numpy.ndarray) of dimensions (num_samples, num_features)
            Y : (numpy.ndarray) of dimensions (num_samples, num_targets)
            M : (numpy.ndarray) of dimensions (num_samples, num_samples) of pairwise distances for negative EMD metric.
        """
        assert X.shape[0] == Y.shape[0], (
            f"Source and target sample dimension do not match: {X.shape} and {Y.shape}"
        )

        # If target is a one-dimensional vector, make it into a column vector
        if Y.ndim == 1:
            Y = Y[:, np.newaxis]

        assert X.ndim == 2
        assert Y.ndim == 2
        self._n_source = X.shape[1]
        self._n_targets = Y.shape[1]
        self._thresholds = None
        self._cuda_source_indices = None
        self._grouped_mapper = None
        self._cuda_batch_report = None

        if self._identity:
            self._resolved_backend = resolve_torch_backend(self._backend)
        else:
            correlation_backend = resolve_torch_backend(self._backend)
            regression_backend = resolve_backend(
                self._backend, self._regression_type, self._regression_kwargs
            )
            self._resolved_backend = (
                "cuda" if correlation_backend == regression_backend == "cuda" else "cpu"
            )

        self._gather_correlations(X, Y)
        if self._corrs is not None:
            assert self._corrs.shape == (self._n_source, self._n_targets)
        if self._thresholds is None:
            self._gather_thresholds()

        # For each target neuron, find the source feature that maximally
        # correlates with it.
        self._weights = []
        self._mappers = []
        self._num_source_units = []
        if (
            not self._identity
            and self._resolved_backend == "cuda"
            and can_batch_cuda_ridge(self._regression_kwargs)
        ):
            self._source_indices = self._cuda_source_indices
            self._num_source_units = [
                int(indices.size) for indices in self._source_indices
            ]
            self._mappers, self._grouped_mapper = fit_grouped_ridge_cuda(
                X,
                Y,
                self._source_indices,
                self._regression_kwargs,
                workspace_bytes=self._ridge_workspace_bytes,
                max_batch_size=self._ridge_max_batch_size,
            )
            grouped = self._grouped_mapper
            correlation_dtype_requested = (
                None
                if self._correlation_dtype is None
                else str(np.dtype(self._correlation_dtype))
            )
            correlation_dtype_effective = str(np.asarray(self._thresholds).dtype)
            coefficient_dtype = str(
                np.result_type(*(mapper.coef_.dtype for mapper in self._mappers))
            )
            self._cuda_batch_report = {
                "n_samples": int(X.shape[0]),
                "n_targets": int(self._n_targets),
                "source_input_dtype": str(np.asarray(X).dtype),
                "target_input_dtype": str(np.asarray(Y).dtype),
                "correlation_target_batch_size_requested": (
                    self._correlation_target_batch_size
                ),
                "correlation_target_batch_size_effective": (
                    resolve_correlation_target_batch_size(
                        self._n_targets, self._correlation_target_batch_size
                    )
                ),
                "ridge_workspace_bytes_requested": getattr(
                    grouped, "workspace_bytes", self._ridge_workspace_bytes
                ),
                "ridge_batch_size_effective": getattr(grouped, "batch_size", None),
                "ridge_bytes_per_target": getattr(grouped, "bytes_per_target", None),
                "ridge_max_batch_size_requested": self._ridge_max_batch_size,
                "ridge_max_batch_size_effective": resolve_ridge_max_batch_size(
                    self._n_targets, self._ridge_max_batch_size
                ),
                "ridge_promoted_to_float64": getattr(
                    grouped, "promoted_to_float64", False
                ),
                "ridge_shared_design": getattr(grouped, "shared_design", False),
                "correlation_dtype": correlation_dtype_requested,
                "correlation_dtype_requested": correlation_dtype_requested,
                "correlation_dtype_effective": correlation_dtype_effective,
                "selection_cutoff_epsilon_multiplier": (
                    CORRELATION_CUTOFF_EPS_MULTIPLIER
                ),
                "ridge_compute_dtype": str(grouped.compute_dtype),
                "coefficient_dtype": coefficient_dtype,
                "prediction_compute_dtype": str(grouped.compute_dtype),
                "prediction_output_dtype": str(grouped.prediction_output_dtype),
            }
            set_last_cuda_batch_report(self._cuda_batch_report)
            if self._weights_save_nm is not None:
                self._weights = [mapper.coef_ for mapper in self._mappers]
        else:
            self._source_indices = []
            for i in range(self._n_targets):
                source_idxs = self._get_sources(i)
                assert source_idxs.shape[0] == self._n_source
                self._source_indices.append(np.flatnonzero(source_idxs))
                source_features = X[:, source_idxs]
                self._num_source_units.append(source_features.shape[-1])

                # If identity mapping, then we only want one source neuron
                if self._identity:
                    assert source_idxs.sum() >= 1 and source_idxs.ndim == 1
                    best_source_idx = np.argwhere(source_idxs).flatten()
                    assert best_source_idx.size >= 1
                    mapper = best_source_idx[
                        0
                    ]  # We append the best source idx, taking the first one if there are ties
                else:
                    if source_features.ndim == 1:
                        source_features = source_features[:, np.newaxis]

                    mapper = make_resolved_regressor(
                        self._regression_type,
                        self._regression_kwargs,
                        self._resolved_backend,
                    )
                    mapper.fit(source_features, Y[:, i][:, np.newaxis])

                self._mappers.append(mapper)
                if self._weights_save_nm is not None:
                    self._weights.append(mapper.coef_)

        if self._weights_save_nm is not None:
            # save after fitting
            print(f"Saving weights to {self._weights_save_nm}")
            weight_shapes = {weight.shape for weight in self._weights}
            if len(weight_shapes) == 1:
                saved_weights = np.stack(self._weights)
            else:
                saved_weights = np.empty(len(self._weights), dtype=object)
                saved_weights[:] = self._weights
            np.savez(self._weights_save_nm, saved_weights)
        self._fitted = True

    def _get_sources(self, target_idx):
        # Returns array of booleans of length number of sources
        if self._cuda_source_indices is not None:
            source_idxs = np.zeros(self._n_source, dtype=bool)
            source_idxs[self._cuda_source_indices[target_idx]] = True
            return source_idxs
        assert self._corrs.shape == (self._n_source, self._n_targets)
        corrs = self._corrs[:, target_idx]
        percentile = self._thresholds[target_idx]
        source_idxs = corrs >= percentile - correlation_cutoff_tolerance(percentile)
        if (~source_idxs).all():
            # In this case, the source corrs were all nan, so nanpercentile returned all False.
            # This was because on some trials the target neuron can be constant (0) firing rate,
            # so no source unit is better than any other in that case to subselect from.
            # Thus, we select the last 1-percentile in terms of rank order of source units.
            corrs = np.arange(len(corrs))
            percentile = np.nanpercentile(corrs, self._percentile)
            source_idxs = corrs >= percentile
        return source_idxs

    def predict(self, X):
        assert self._n_source is not None, "Fit procedure was not run."
        assert self._n_targets is not None, "Fit procedure was not run."
        assert self._mappers is not None, "Fit procedure was not run."
        assert self._fitted, "Fitting failed."
        assert len(self._mappers) == self._n_targets
        assert X.shape[1] == self._n_source, (
            f"Number of train features ({self._n_source}) does not align with number of test features ({X.shape[1]})"
        )

        n_samples = X.shape[0]
        if self._grouped_mapper is not None:
            return self._grouped_mapper.predict(X)
        preds = np.full((n_samples, self._n_targets), np.nan)
        for i, mapper in enumerate(self._mappers):
            if self._identity:
                assert isinstance(mapper, np.int64)  # mapper contains the source index
                curr_pred = X[:, mapper]
            else:
                source_features = X[:, self._source_indices[i]]
                assert mapper.coef_.shape[-1] == source_features.shape[1], (
                    f"Linear map takes in {mapper.coef_.shape[-1]} features, not {source_features.shape[1]}."
                )
                curr_pred = mapper.predict(source_features).flatten()

                assert curr_pred.ndim == 1 and curr_pred.shape[0] == n_samples
            preds[:, i] = curr_pred

        return preds
