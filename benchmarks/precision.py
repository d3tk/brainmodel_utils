"""Precision artifacts and pre-registered numerical gates."""

import json
from pathlib import Path

import numpy as np

CONSISTENCY_METRICS = (
    "r_xy_n_sb",
    "r_xx",
    "r_xx_sb",
    "r_yy",
    "r_yy_sb",
    "r_xy",
    "denom_sb",
)

PRODUCTION_GATES = {
    "max_abs_canonical_metric_difference": 0.005,
    "max_abs_global_finite_mean_test_r_xy_n_sb_difference": 0.001,
    "max_finite_mask_mismatch_fraction": 1e-5,
}


def _as_float(array):
    return np.asarray(array, dtype=np.float64)


def selected_index_sets(selections):
    return [np.asarray(indices, dtype=np.int64) for indices in selections]


def jaccard(left, right):
    left = np.unique(np.asarray(left, dtype=np.int64))
    right = np.unique(np.asarray(right, dtype=np.int64))
    if left.size == 0 and right.size == 0:
        return 1.0
    union = np.union1d(left, right)
    if union.size == 0:
        return 1.0
    return float(np.intersect1d(left, right).size / union.size)


def finite_mask(array):
    return np.isfinite(_as_float(array))


def save_precision_fit(
    path,
    X_train,
    Y_train,
    X_test,
    mapper,
    metrics=None,
    extra=None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    selections = getattr(mapper, "_source_indices", None) or []
    thresholds = getattr(mapper, "_thresholds", None)
    correlations = getattr(mapper, "_corrs", None)
    train_prediction = mapper.predict(X_train)
    test_prediction = mapper.predict(X_test)
    counts = np.asarray(
        [np.asarray(indices).size for indices in selections], dtype=np.int64
    )
    payload = {
        "train_prediction": np.asarray(train_prediction),
        "test_prediction": np.asarray(test_prediction),
        "train_prediction_finite": finite_mask(train_prediction),
        "test_prediction_finite": finite_mask(test_prediction),
        "selected_source_counts": counts,
        "n_source": np.int64(getattr(mapper, "_n_source", X_train.shape[1])),
        "n_targets": np.int64(getattr(mapper, "_n_targets", Y_train.shape[1])),
    }
    if thresholds is not None:
        payload["thresholds"] = np.asarray(thresholds)
    if correlations is not None:
        payload["correlations"] = np.asarray(correlations)
    if selections:
        max_len = max(int(indices.size) for indices in selections)
        packed = np.full((len(selections), max_len), -1, dtype=np.int64)
        for target, indices in enumerate(selections):
            packed[target, : indices.size] = np.asarray(indices, dtype=np.int64)
        payload["selected_source_indices"] = packed
    if metrics is not None:
        for split in ("train", "test"):
            for name, values in metrics[split].items():
                payload["{}_{}".format(split, name)] = np.asarray(values)
    np.savez_compressed(path, **payload)
    report = {
        "file": str(path.resolve()),
        "targets": int(payload["n_targets"]),
        "selected_source_count_mean": float(counts.mean()) if counts.size else None,
    }
    if extra:
        report.update(extra)
    sidecar = path.with_suffix(".json")
    with open(sidecar, "w") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return report


def load_precision_fit(path):
    with np.load(path, allow_pickle=True) as payload:
        return {name: payload[name] for name in payload.files}


def _metric_absdiff(current, reference, split, name):
    if name not in current or name not in reference:
        return None
    current_values = _as_float(current[name])
    reference_values = _as_float(reference[name])
    if current_values.shape != reference_values.shape:
        return None
    finite = finite_mask(current_values) & finite_mask(reference_values)
    if not finite.any():
        return {
            "max_abs": None,
            "finite_mask_mismatch_fraction": float(
                np.mean(finite_mask(current_values) != finite_mask(reference_values))
            ),
        }
    difference = current_values - reference_values
    return {
        "max_abs": float(np.nanmax(np.abs(difference))),
        "p50_abs": float(np.nanpercentile(np.abs(difference[finite]), 50)),
        "p95_abs": float(np.nanpercentile(np.abs(difference[finite]), 95)),
        "sign_change_fraction": float(
            np.mean(np.sign(current_values[finite]) != np.sign(reference_values[finite]))
        ),
        "finite_mask_mismatch_fraction": float(
            np.mean(finite_mask(current_values) != finite_mask(reference_values))
        ),
    }


def compare_precision(current, reference, gates=None):
    gates = dict(PRODUCTION_GATES if gates is None else gates)
    current_metrics = current
    reference_metrics = reference
    if not isinstance(current, dict):
        current_metrics = load_precision_fit(current)
        reference_metrics = load_precision_fit(reference)
    metric_comparisons = {}
    max_abs = []
    mask_mismatches = []
    for split in ("train", "test"):
        metric_comparisons[split] = {}
        for name in CONSISTENCY_METRICS:
            current_key = "{}_{}".format(split, name)
            reference_key = "{}_{}".format(split, name)
            if current_key in current_metrics and reference_key in reference_metrics:
                comparison = _metric_absdiff(
                    current_metrics, reference_metrics, split, current_key
                )
            elif (
                isinstance(current_metrics.get(split), dict)
                and name in current_metrics[split]
            ):
                comparison = _metric_absdiff(
                    current_metrics[split], reference_metrics[split], split, name
                )
            else:
                continue
            metric_comparisons[split][name] = comparison
            if comparison.get("max_abs") is not None:
                max_abs.append(comparison["max_abs"])
            if comparison.get("finite_mask_mismatch_fraction") is not None:
                mask_mismatches.append(comparison["finite_mask_mismatch_fraction"])

    mean_keys = (
        ("test_r_xy_n_sb", "test_r_xy_n_sb"),
        ("test", "r_xy_n_sb"),
    )
    current_mean = None
    reference_mean = None
    if "test_r_xy_n_sb" in current_metrics:
        current_values = _as_float(current_metrics["test_r_xy_n_sb"])
        reference_values = _as_float(reference_metrics["test_r_xy_n_sb"])
        current_mean = float(np.nanmean(current_values[np.isfinite(current_values)]))
        reference_mean = float(
            np.nanmean(reference_values[np.isfinite(reference_values)])
        )
    elif isinstance(current_metrics.get("test"), dict):
        current_values = _as_float(current_metrics["test"]["r_xy_n_sb"])
        reference_values = _as_float(reference_metrics["test"]["r_xy_n_sb"])
        current_mean = float(np.nanmean(current_values[np.isfinite(current_values)]))
        reference_mean = float(
            np.nanmean(reference_values[np.isfinite(reference_values)])
        )
    mean_difference = (
        None if current_mean is None else abs(current_mean - reference_mean)
    )

    selection = None
    if (
        "selected_source_indices" in current_metrics
        and "selected_source_indices" in reference_metrics
    ):
        current_sel = current_metrics["selected_source_indices"]
        reference_sel = reference_metrics["selected_source_indices"]
        jaccards = []
        disagreements = 0
        for target in range(min(current_sel.shape[0], reference_sel.shape[0])):
            left = current_sel[target][current_sel[target] >= 0]
            right = reference_sel[target][reference_sel[target] >= 0]
            jaccards.append(jaccard(left, right))
            if set(left.tolist()) != set(right.tolist()):
                disagreements += 1
        thresholds = None
        if (
            "thresholds" in current_metrics
            and "thresholds" in reference_metrics
            and "correlations" in reference_metrics
        ):
            thresholds = _as_float(reference_metrics["thresholds"])
        selection = {
            "targets": int(min(current_sel.shape[0], reference_sel.shape[0])),
            "disagreements": disagreements,
            "mean_jaccard": float(np.mean(jaccards)) if jaccards else None,
            "min_jaccard": float(np.min(jaccards)) if jaccards else None,
        }

    max_abs_metric = max(max_abs) if max_abs else None
    mask_mismatch = max(mask_mismatches) if mask_mismatches else 0.0
    gates_passed = True
    failures = []
    if (
        max_abs_metric is not None
        and max_abs_metric > gates["max_abs_canonical_metric_difference"]
    ):
        gates_passed = False
        failures.append("canonical_metric")
    if (
        mean_difference is not None
        and mean_difference
        > gates["max_abs_global_finite_mean_test_r_xy_n_sb_difference"]
    ):
        gates_passed = False
        failures.append("global_finite_mean_test_r_xy_n_sb")
    if mask_mismatch > gates["max_finite_mask_mismatch_fraction"]:
        gates_passed = False
        failures.append("finite_mask")
    return {
        "gates": gates,
        "gates_passed": gates_passed,
        "failures": failures,
        "max_abs_canonical_metric_difference": max_abs_metric,
        "abs_global_finite_mean_test_r_xy_n_sb_difference": mean_difference,
        "finite_mask_mismatch_fraction": mask_mismatch,
        "metrics": metric_comparisons,
        "selection": selection,
    }
