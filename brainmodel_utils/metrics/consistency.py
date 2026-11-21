"""Public consistency API.

``get_linregress_consistency`` runs the optimized implementation
(``consistency_fast``), which reproduces the original results bitwise. Set
``BRAINMODEL_UTILS_CONSISTENCY_IMPL=reference`` to run the unmodified original
(``consistency_reference``) instead; the variable is read on every call, so no
reinstall is needed to roll back.

All other names that used to live in this module are re-exported unchanged.
"""

import os

# Names below are re-exported so existing imports from this module keep working.
import numpy as np  # noqa: F401
import xarray as xr  # noqa: F401
from joblib import Parallel, delayed  # noqa: F401
import copy  # noqa: F401
from brainmodel_utils.core.utils import dict_app, dict_np, make_list  # noqa: F401
from brainmodel_utils.metrics.utils import (  # noqa: F401
    sphalf_input_checker,
    str_to_metric_func,
    generic_trial_avg,
    get_splithalves,
    concat_dict_sp,
)
from brainmodel_utils.neural_mappers import PipelineNeuralMap  # noqa: F401
from brainmodel_utils.neural_mappers.utils import generate_train_test_splits  # noqa: F401
from brainmodel_utils.metrics import consistency_fast as _fast
from brainmodel_utils.metrics import consistency_reference as _reference
from brainmodel_utils.metrics.consistency_reference import (  # noqa: F401
    sb_correction,
    get_consistency_per_neuron,
    get_linregress_consistency_persplit,
    get_linregress_consistency_persphalftrial,
)

IMPL_ENV = "BRAINMODEL_UTILS_CONSISTENCY_IMPL"
_IMPLEMENTATIONS = ("fast", "reference")


def active_implementation():
    impl = os.environ.get(IMPL_ENV, "").strip().lower() or "fast"
    if impl not in _IMPLEMENTATIONS:
        raise ValueError(f"{IMPL_ENV} must be one of {_IMPLEMENTATIONS}, got {impl!r}")
    return impl


def get_linregress_consistency(
    source,
    target,
    map_kwargs,
    num_bootstrap_iters=1000,
    num_parallel_jobs=1,
    start_seed=1234,
    metric="pearsonr",
    splits=None,
    **kwargs
):
    """
    The main function for computing the linear regression consistency (noise corrected)
    between source and target.

    Arguments:
    -------------------
    source: Either model features (stimuli x units), or neural features (trials x stimuli x units)
    target: Neural features (trials x stimuli x units), usually from a different animal if the source features are neural too.
    map_kwargs: Either a dict or a list of dicts (one per train/test split) specifying the linear regression parameters.
    num_bootstrap_iters: How many split-halves to compute.
    num_parallel_jobs: Number of parallel jobs to parallelize the outermost for loop over split-half trials.
    start_seed: Starting seed for generating split halves (for reproducibility).
    metric (optional): Correlation metric (across stimuli) used per neuron.
            Supported metrics: "pearsonr", "spearmanr", "rsa_pearsonr", "rsa_spearmanr"
    splits (optional): Your own list of {"train", "test"} split indices.
            If "splits" is None, you can additionally specify train_frac and num_train_test_splits to generate your own.
    target_batch_size (optional): Fit ElasticNet/Lasso targets in batches of this
            size to bound coefficient memory. Default: $BRAINMODEL_UTILS_TARGET_BATCH_SIZE,
            else 0 (off). Coefficients are unchanged; predictions may differ from
            the unbatched fit at the floating-point rounding level.

    Returns (see README for full details):
    --------------------------------------
    dict
        A dictionary with two main keys, "train" and "test", each mapping to a dict containing:

        - "r_xy_n_sb": Noise-corrected (Spearman-Brown) per-neuron correlation
                       indicating predictivity of the source for the target.
        - "r_xx", "r_xx_sb": Split-half correlation and Spearman-Brown–corrected
                            correlation for the source’s predicted responses.
        - "r_yy", "r_yy_sb": Split-half correlation and Spearman-Brown–corrected
                            correlation for the actual target responses.
        - "r_xy": Raw correlation between predicted and actual responses (uncorrected).
        - "denom_sb": Denominator for the noise-corrected correlation
                      (i.e., sqrt(r_xx_sb * r_yy_sb)).

        Each of these values is shaped
        (num_bootstrap_iters, number_of_splits, number_of_units),
        or returned as an xarray.DataArray with labeled dimensions if the target is an xarray.
        The key "r_xy_n_sb" is typically the primary measure of interest.
    """
    call_kwargs = dict(
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
    if active_implementation() == "reference":
        call_kwargs.pop("target_batch_size", None)
        return _reference.get_linregress_consistency(**call_kwargs)
    return _fast.get_linregress_consistency(**call_kwargs)
