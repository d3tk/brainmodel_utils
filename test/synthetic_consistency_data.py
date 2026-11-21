"""Deterministic synthetic "fly pair" tensors for consistency equivalence tests.

Every generator returns ``(source, target)`` with layout
``(trials, timepoints, voxels)``, matching the production IPS interface. Activity
is built from low-dimensional latent time courses that are partially shared
between the two animals, mixed into voxels by separate source/target spatial
matrices, plus independent per-trial noise.

Nothing here needs real Drosophila data.
"""

from __future__ import annotations

import numpy as np

# Production window: the 14 event-relative samples between about -15 s and -8 s.
NUM_TIMEPOINTS = 14
RELATIVE_TIMES = np.linspace(-15.0, -8.0, NUM_TIMEPOINTS)
NUM_FOLDS = 5
PRIMARY_SEED = 42

# Production ElasticNet configuration (autonomous_drosophila ips_enet_cv).
ENET_ALPHA = 0.01
ENET_L1_RATIO = 0.25
ENET_MAX_ITER = 10000


def enet_map_kwargs(alpha=ENET_ALPHA, l1_ratio=ENET_L1_RATIO, max_iter=ENET_MAX_ITER):
    return {
        "map_type": "sklinear",
        "map_kwargs": {
            "regression_type": "ElasticNet",
            "regression_kwargs": {
                "alpha": float(alpha),
                "l1_ratio": float(l1_ratio),
                "max_iter": int(max_iter),
            },
        },
    }


def make_blocked_splits(num_timepoints=NUM_TIMEPOINTS, num_folds=NUM_FOLDS):
    """Contiguous folds, identical to autonomous_drosophila ``make_full_data_splits``.

    For 14 timepoints the test folds have sizes 3, 3, 3, 3, 2.
    """
    indices = np.arange(num_timepoints, dtype=np.int64)
    folds = [np.asarray(b, dtype=np.int64) for b in np.array_split(indices, num_folds)]
    return [
        {
            "train": np.concatenate([f for i, f in enumerate(folds) if i != k]),
            "test": folds[k],
        }
        for k in range(num_folds)
    ]


def make_fly_pair(
    n_source_trials=40,
    n_target_trials=40,
    n_timepoints=NUM_TIMEPOINTS,
    n_source_voxels=128,
    n_target_voxels=128,
    n_latents=4,
    shared=0.8,
    signal_scale=1.0,
    noise_scale=1.0,
    seed=PRIMARY_SEED,
    dtype=np.float32,
):
    """Correlated but non-identical synthetic flies.

    ``shared`` in [0, 1] sets how much of the target's latent drive is the
    source's latent drive (0 = independent animals).
    """
    rng = np.random.default_rng(seed)
    # Smooth latent time courses (random walk, then standardized).
    latents = np.cumsum(rng.standard_normal((n_timepoints, n_latents)), axis=0)
    latents = (latents - latents.mean(0)) / latents.std(0)
    private = np.cumsum(rng.standard_normal((n_timepoints, n_latents)), axis=0)
    private = (private - private.mean(0)) / private.std(0)
    target_latents = shared * latents + np.sqrt(1.0 - shared**2) * private

    w_source = rng.standard_normal((n_latents, n_source_voxels)) / np.sqrt(n_latents)
    w_target = rng.standard_normal((n_latents, n_target_voxels)) / np.sqrt(n_latents)
    source_offset = rng.normal(0.0, 0.5, size=n_source_voxels)
    target_offset = rng.normal(0.0, 0.5, size=n_target_voxels)

    source_mean = signal_scale * latents @ w_source + source_offset
    target_mean = signal_scale * target_latents @ w_target + target_offset
    source = source_mean[None] + noise_scale * rng.standard_normal(
        (n_source_trials, n_timepoints, n_source_voxels)
    )
    target = target_mean[None] + noise_scale * rng.standard_normal(
        (n_target_trials, n_timepoints, n_target_voxels)
    )
    return source.astype(dtype), target.astype(dtype)


def primary():
    """Phase 1.1 dataset: 40/40 trials, 14 timepoints, 128/128 voxels, float32, seed 42."""
    return make_fly_pair()


# Edge-case variants use fewer voxels than the primary dataset so the live
# reference-vs-optimized comparisons stay fast; the code paths are identical.
EDGE_VOXELS = dict(n_source_voxels=48, n_target_voxels=40)


def _edge_pair(**kwargs):
    return make_fly_pair(**{**EDGE_VOXELS, **kwargs})


def independent():
    return _edge_pair(shared=0.0, seed=101)


def strongly_correlated():
    return _edge_pair(shared=0.98, noise_scale=0.2, seed=102)


def weakly_correlated():
    return _edge_pair(shared=0.2, noise_scale=3.0, seed=103)


def different_trial_counts():
    # Odd source count gives unequal split halves (15 / 16).
    return _edge_pair(n_source_trials=31, n_target_trials=40, seed=104)


def different_voxel_counts():
    return make_fly_pair(n_source_voxels=36, n_target_voxels=50, seed=105)


def constant_voxels():
    """Exactly constant and nearly constant voxels on both sides."""
    source, target = _edge_pair(seed=106)
    target[:, :, 0] = 3.25  # exactly constant -> pearsonr NaN
    target[:, :, 1] = 0.0
    source[:, :, 2] = -1.5
    # Constant over time within every trial, but differing across trials.
    target[:, :, 3] = np.arange(target.shape[0], dtype=target.dtype)[:, None]
    # Nearly constant: tiny fluctuation on a large offset.
    rng = np.random.default_rng(1060)
    target[:, :, 4] = 1000.0 + 1e-4 * rng.standard_normal(target.shape[:2])
    return source, target


def negative_reliability():
    """Pure-noise target voxels with few trials, so split-half r_yy is often negative."""
    source, target = _edge_pair(n_target_trials=6, seed=107)
    rng = np.random.default_rng(1070)
    target[:, :, 20:] = rng.standard_normal(target[:, :, 20:].shape).astype(target.dtype)
    return source, target


def nan_trials():
    """NaNs in individual trials that never wipe out a whole split half."""
    source, target = _edge_pair(seed=108)
    source[3, 5, 10] = np.nan
    source[7, :, 20] = np.nan
    target[0, 2, 7] = np.nan
    target[11, :, 30:36] = np.nan
    return source, target


def nan_trials_skipping():
    """Few trials with mostly-NaN voxels: some bootstrap split halves are all-NaN.

    The original implementation skips those bootstraps (fewer output rows),
    which the optimized version must reproduce.
    """
    source, target = _edge_pair(n_source_trials=8, n_target_trials=8, seed=109)
    target[:5, 4, 30] = np.nan  # 3 valid trials -> an all-NaN half for some seeds
    source[2:7, 9, 11] = np.nan
    return source, target


def nan_outputs():
    """Original returns NaN: constant predictions and negative reliability products."""
    source, target = _edge_pair(n_target_trials=6, seed=110)
    rng = np.random.default_rng(1100)
    target[:, :, 30:] = rng.standard_normal(target[:, :, 30:].shape).astype(target.dtype)
    return source, target


def above_one():
    """Low reliability with real shared signal: corrected r often exceeds 1."""
    return _edge_pair(
        n_source_trials=6, n_target_trials=6, shared=0.95, noise_scale=2.5, seed=111
    )


def float64_pair():
    return _edge_pair(seed=112, dtype=np.float64)


VARIANTS = {
    "primary": primary,
    "independent": independent,
    "strongly_correlated": strongly_correlated,
    "weakly_correlated": weakly_correlated,
    "different_trial_counts": different_trial_counts,
    "different_voxel_counts": different_voxel_counts,
    "constant_voxels": constant_voxels,
    "negative_reliability": negative_reliability,
    "nan_trials": nan_trials,
    "nan_trials_skipping": nan_trials_skipping,
    "nan_outputs": nan_outputs,
    "above_one": above_one,
    "float64": float64_pair,
}
