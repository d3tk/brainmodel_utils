"""Learned-fit precision artifacts and numerical acceptance gates."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmarks.artifacts import (
    artifact_record,
    atomic_savez,
    atomic_write_json,
    validate_artifact_record,
)

PRECISION_SCHEMA_VERSION = 2
CONSISTENCY_METRICS = (
    "r_xy_n_sb",
    "r_xx",
    "r_xx_sb",
    "r_yy",
    "r_yy_sb",
    "r_xy",
    "denom_sb",
)
MAPPER_ROLES = ("full", "half_1", "half_2")
MAPPER_FIELDS = (
    "correlations",
    "thresholds",
    "cutoff_below",
    "cutoff_above",
    "selected_source_counts",
    "selected_source_indices",
    "coefficients",
    "intercepts",
    "train_prediction",
    "test_prediction",
    "train_prediction_finite",
    "test_prediction_finite",
)
FLOAT64_GATES = {
    "mapping_rtol": 1e-10,
    "mapping_atol": 1e-12,
    "max_metric_p95_abs_difference": 1e-12,
    "max_abs_global_finite_mean_test_r_xy_n_sb_difference": 1e-10,
    "normalized_metric_atol": 1e-8,
    "normalized_metric_rtol": 1e-10,
    "max_finite_mask_mismatch_fraction": 0.0,
    "require_exact_selection": True,
}
PRODUCTION_GATES = FLOAT64_GATES
_MAPPING_NUMERIC_FIELDS = (
    "correlations",
    "thresholds",
    "cutoff_below",
    "cutoff_above",
    "coefficients",
    "intercepts",
    "train_prediction",
    "test_prediction",
)
_MAPPING_DIAGNOSTIC_FIELDS = frozenset(("cutoff_below", "cutoff_above"))
_PROVENANCE_FIELDS = (
    "configuration",
    "dataset",
    "seed",
    "bootstrap_index",
    "fold_index",
    "dtypes",
    "input_checksums",
    "repository",
)
_REQUIRED_METADATA_FIELDS = set(_PROVENANCE_FIELDS).union(
    {"schema_version", "roles", "backend_requested", "backend_effective"}
)


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


def array_sha256(array):
    array = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _unwrap_mapper(mapper):
    inner = getattr(mapper, "map", None)
    return mapper if inner is None else inner


def _pack_learned_mapping(mapper):
    mapper = _unwrap_mapper(mapper)
    selections = getattr(mapper, "_source_indices", None)
    mappers = getattr(mapper, "_mappers", None)
    correlations = getattr(mapper, "_corrs", None)
    thresholds = getattr(mapper, "_thresholds", None)
    if selections is None or mappers is None:
        raise ValueError("precision diagnostics require a fitted percentile mapper")
    if correlations is None or thresholds is None:
        raise ValueError(
            "precision diagnostics require store_correlations=True during fitting"
        )

    selections = selected_index_sets(selections)
    counts = np.asarray([indices.size for indices in selections], dtype=np.int64)
    if counts.size == 0 or np.any(counts < 1):
        raise ValueError("every diagnostic target must select at least one source")
    width = int(counts.max())
    packed_indices = np.full((counts.size, width), -1, dtype=np.int64)
    coefficient_dtype = np.result_type(
        *[np.asarray(model.coef_).dtype for model in mappers]
    )
    coefficients = np.full((counts.size, width), np.nan, dtype=coefficient_dtype)
    intercepts = np.empty(counts.size, dtype=coefficient_dtype)
    for target, (indices, model) in enumerate(zip(selections, mappers)):
        coefficient = np.asarray(model.coef_).reshape(-1)
        if coefficient.size != indices.size:
            raise ValueError("coefficient and selected-source counts disagree")
        packed_indices[target, : indices.size] = indices
        coefficients[target, : indices.size] = coefficient
        intercepts[target] = np.asarray(model.intercept_).reshape(-1)[0]

    correlations = np.asarray(correlations)
    thresholds = np.asarray(thresholds)
    if correlations.shape != (mapper._n_source, counts.size):
        raise ValueError("correlation matrix shape does not match the fitted mapper")
    below = np.full(counts.size, np.nan, dtype=correlations.dtype)
    above = np.full(counts.size, np.nan, dtype=correlations.dtype)
    for target in range(counts.size):
        values = correlations[:, target]
        values = values[np.isfinite(values)]
        if values.size == 0 or not np.isfinite(thresholds[target]):
            continue
        lower = values[values < thresholds[target]]
        upper = values[values >= thresholds[target]]
        if lower.size:
            below[target] = lower.max()
        if upper.size:
            above[target] = upper.min()
    return {
        "correlations": correlations,
        "thresholds": thresholds,
        "cutoff_below": below,
        "cutoff_above": above,
        "selected_source_counts": counts,
        "selected_source_indices": packed_indices,
        "coefficients": coefficients,
        "intercepts": intercepts,
    }


def _metric_payload(metrics):
    payload = {}
    for split in ("train", "test"):
        block = metrics.get(split)
        if not isinstance(block, dict):
            raise TypeError("precision metrics must contain train and test blocks")
        if set(block) != set(CONSISTENCY_METRICS):
            raise ValueError("precision metrics must contain the canonical metric set")
        for name in CONSISTENCY_METRICS:
            values = np.asarray(block[name])
            key = f"{split}_{name}"
            payload[key] = values
            payload[key + "_finite"] = finite_mask(values)
    return payload


def _fit_payload(role, fit):
    required = {"mapper", "X_train", "X_test"}
    missing = required.difference(fit)
    if missing:
        raise ValueError(
            "{} precision fit is missing {}".format(role, ", ".join(sorted(missing)))
        )
    mapper = fit["mapper"]
    X_train = np.asarray(fit["X_train"])
    X_test = np.asarray(fit["X_test"])
    learned = _pack_learned_mapping(mapper)
    train_prediction = np.asarray(mapper.predict(X_train))
    test_prediction = np.asarray(mapper.predict(X_test))
    learned.update(
        {
            "train_prediction": train_prediction,
            "test_prediction": test_prediction,
            "train_prediction_finite": finite_mask(train_prediction),
            "test_prediction_finite": finite_mask(test_prediction),
        }
    )
    return {f"{role}_{name}": value for name, value in learned.items()}


def save_precision_diagnostic(
    path,
    fits,
    metrics,
    target_indices,
    train_indices,
    test_indices,
    metadata,
    owner_path=None,
):
    """Save a numeric learned-fit artifact and a JSON provenance sidecar."""

    roles = tuple(fits)
    if roles != MAPPER_ROLES:
        raise ValueError("precision diagnostics require full, half_1, and half_2 fits")
    payload = {
        "target_indices": np.asarray(target_indices, dtype=np.int64),
        "train_indices": np.asarray(train_indices, dtype=np.int64),
        "test_indices": np.asarray(test_indices, dtype=np.int64),
    }
    payload.update(_metric_payload(metrics))
    input_checksums = {}
    for role, fit in fits.items():
        payload.update(_fit_payload(role, fit))
        input_checksums[role] = {
            name: array_sha256(fit[name])
            for name in ("X_train", "X_test", "Y_train", "Y_test")
            if name in fit
        }

    document = dict(metadata)
    document.update(
        {
            "schema_version": PRECISION_SCHEMA_VERSION,
            "roles": list(roles),
            "input_checksums": input_checksums,
        }
    )
    errors = validate_precision_artifact(payload, document)
    if errors:
        raise ValueError("invalid precision artifact: {}".format("; ".join(errors)))

    path = Path(path)
    sidecar = path.with_suffix(".json")
    atomic_savez(path, payload)
    document["arrays"] = artifact_record(sidecar, path)
    atomic_write_json(sidecar, document)
    if owner_path is None:
        return {
            "arrays": str(path.resolve()),
            "metadata": str(sidecar.resolve()),
        }
    return {
        "arrays": artifact_record(owner_path, path),
        "metadata": artifact_record(owner_path, sidecar),
    }


def save_precision_fit(
    path,
    X_train,
    Y_train,
    X_test,
    mapper,
    metrics=None,
    extra=None,
):
    """Save one fitted map for focused diagnostics outside consistency runs."""

    if metrics is None:
        raise ValueError("precision fits require the canonical train/test metrics")
    target_count = np.asarray(Y_train).shape[-1]
    fit = {
        "mapper": mapper,
        "X_train": X_train,
        "X_test": X_test,
        "Y_train": Y_train,
    }
    payload = {
        "target_indices": np.arange(target_count, dtype=np.int64),
        "train_indices": np.arange(np.asarray(X_train).shape[0], dtype=np.int64),
        "test_indices": np.arange(np.asarray(X_test).shape[0], dtype=np.int64),
    }
    payload.update(_metric_payload(metrics))
    payload.update(_fit_payload("full", fit))
    document = dict(extra or {})
    document.update(
        {
            "schema_version": PRECISION_SCHEMA_VERSION,
            "roles": ["full"],
            "input_checksums": {
                "full": {
                    "X_train": array_sha256(X_train),
                    "X_test": array_sha256(X_test),
                    "Y_train": array_sha256(Y_train),
                }
            },
        }
    )
    errors = validate_precision_artifact(payload, document)
    if errors:
        raise ValueError("invalid precision artifact: {}".format("; ".join(errors)))
    path = Path(path)
    sidecar = path.with_suffix(".json")
    atomic_savez(path, payload)
    document["arrays"] = artifact_record(sidecar, path)
    atomic_write_json(sidecar, document)
    return {"file": str(path.resolve()), "metadata_file": str(sidecar.resolve())}


def load_precision_fit(path):
    with np.load(path, allow_pickle=False) as payload:
        return {name: payload[name] for name in payload.files}


def load_precision_diagnostic(path):
    path = Path(path)
    arrays = load_precision_fit(path)
    sidecar = path.with_suffix(".json")
    with open(sidecar) as stream:
        metadata = json.load(stream)
    record = metadata.get("arrays")
    if record is None:
        raise ValueError("precision metadata is missing its arrays record")
    recorded_path = validate_artifact_record(sidecar, record)
    if recorded_path.resolve() != path.resolve():
        raise ValueError("precision metadata points to a different arrays file")
    return arrays, metadata


def _roles_from_arrays(arrays, metadata):
    if isinstance(metadata.get("roles"), list):
        return tuple(metadata["roles"])
    roles = []
    for role in MAPPER_ROLES:
        if f"{role}_correlations" in arrays:
            roles.append(role)
    return tuple(roles)


def _expected_keys(roles):
    keys = {"target_indices", "train_indices", "test_indices"}
    for split in ("train", "test"):
        for name in CONSISTENCY_METRICS:
            key = f"{split}_{name}"
            keys.update((key, key + "_finite"))
    for role in roles:
        keys.update(f"{role}_{field}" for field in MAPPER_FIELDS)
    return keys


def validate_precision_artifact(arrays, metadata):
    errors = []
    if not isinstance(arrays, dict) or not arrays:
        return ["precision artifact contains no arrays"]
    if not isinstance(metadata, dict):
        return ["precision metadata must be an object"]
    if metadata.get("schema_version") != PRECISION_SCHEMA_VERSION:
        errors.append("unsupported precision schema")
    missing_metadata = sorted(_REQUIRED_METADATA_FIELDS.difference(metadata))
    if missing_metadata:
        errors.append("missing metadata: {}".format(", ".join(missing_metadata)))
    roles = _roles_from_arrays(arrays, metadata)
    if not roles or any(role not in MAPPER_ROLES for role in roles):
        errors.append("precision artifact has invalid mapper roles")
    expected = _expected_keys(roles)
    missing = sorted(expected.difference(arrays))
    unexpected = sorted(set(arrays).difference(expected))
    if missing:
        errors.append("missing arrays: {}".format(", ".join(missing)))
    if unexpected:
        errors.append("unexpected arrays: {}".format(", ".join(unexpected)))
    if missing:
        return errors

    for name, value in arrays.items():
        array = np.asarray(value)
        if array.dtype.hasobject or array.dtype.kind in "SUO":
            errors.append(f"{name} is not a numeric non-pickled array")
    target_indices = np.asarray(arrays["target_indices"])
    train_indices = np.asarray(arrays["train_indices"])
    test_indices = np.asarray(arrays["test_indices"])
    if target_indices.ndim != 1 or target_indices.size < 1:
        errors.append("target_indices must be a non-empty vector")
        return errors
    if len(np.unique(target_indices)) != target_indices.size:
        errors.append("target_indices must be unique")
    dataset = metadata.get("dataset")
    target_selection = (
        dataset.get("diagnostic_target_selection")
        if isinstance(dataset, dict)
        else None
    )
    if target_selection is not None:
        if not isinstance(target_selection, dict):
            errors.append("diagnostic target selection must be an object")
        else:
            required = {
                "strategy",
                "full_target_count",
                "selected_target_count",
                "positions",
                "original_target_ids",
            }
            missing_selection = sorted(required.difference(target_selection))
            if missing_selection:
                errors.append(
                    "diagnostic target selection is missing: {}".format(
                        ", ".join(missing_selection)
                    )
                )
            else:
                positions = np.asarray(target_selection["positions"])
                original_ids = np.asarray(target_selection["original_target_ids"])
                full_count = target_selection["full_target_count"]
                selected_count = target_selection["selected_target_count"]
                valid_counts = (
                    isinstance(full_count, int)
                    and full_count >= 1
                    and isinstance(selected_count, int)
                    and selected_count >= 1
                )
                if not valid_counts:
                    errors.append("diagnostic target counts are invalid")
                if target_selection["strategy"] not in (
                    "all",
                    "evenly_spaced",
                    "explicit",
                ):
                    errors.append("diagnostic target strategy is invalid")
                if not valid_counts or (
                    positions.dtype.kind not in "iu"
                    or positions.shape != target_indices.shape
                    or len(np.unique(positions)) != positions.size
                    or np.any(positions < 0)
                    or np.any(positions >= full_count)
                ):
                    errors.append("diagnostic target positions are invalid")
                if selected_count != target_indices.size:
                    errors.append("diagnostic selected target count is invalid")
                if not np.array_equal(original_ids, target_indices):
                    errors.append("diagnostic original target IDs do not match arrays")
    for name, indices in (
        ("train_indices", train_indices),
        ("test_indices", test_indices),
    ):
        if indices.ndim != 1 or indices.size < 1:
            errors.append(f"{name} must be a non-empty vector")
    target_count = target_indices.size

    for split in ("train", "test"):
        for metric in CONSISTENCY_METRICS:
            key = f"{split}_{metric}"
            values = np.asarray(arrays[key])
            mask = np.asarray(arrays[key + "_finite"])
            if values.ndim < 1 or values.shape[-1] != target_count:
                errors.append(f"{key} target shape mismatch")
            if mask.dtype != np.bool_ or mask.shape != values.shape:
                errors.append(f"{key} finite mask mismatch")
            elif not np.array_equal(mask, finite_mask(values)):
                errors.append(f"{key} finite mask is stale")

    for role in roles:
        prefix = role + "_"
        correlations = np.asarray(arrays[prefix + "correlations"])
        thresholds = np.asarray(arrays[prefix + "thresholds"])
        counts = np.asarray(arrays[prefix + "selected_source_counts"])
        indices = np.asarray(arrays[prefix + "selected_source_indices"])
        coefficients = np.asarray(arrays[prefix + "coefficients"])
        intercepts = np.asarray(arrays[prefix + "intercepts"])
        train_prediction = np.asarray(arrays[prefix + "train_prediction"])
        test_prediction = np.asarray(arrays[prefix + "test_prediction"])
        if correlations.ndim != 2 or correlations.shape[1] != target_count:
            errors.append(f"{prefix}correlations shape mismatch")
        if thresholds.shape != (target_count,):
            errors.append(f"{prefix}thresholds shape mismatch")
        for field in ("cutoff_below", "cutoff_above"):
            if np.asarray(arrays[prefix + field]).shape != (target_count,):
                errors.append(f"{prefix}{field} shape mismatch")
        if counts.shape != (target_count,) or np.any(counts < 1):
            errors.append(f"{prefix}selected source counts are invalid")
        if indices.ndim != 2 or indices.shape[0] != target_count:
            errors.append(f"{prefix}selected indices shape mismatch")
        elif counts.size and np.any(counts > indices.shape[1]):
            errors.append(f"{prefix}selected counts exceed packed width")
        if coefficients.shape != indices.shape:
            errors.append(f"{prefix}coefficient shape mismatch")
        if intercepts.shape != (target_count,):
            errors.append(f"{prefix}intercept shape mismatch")
        if train_prediction.shape != (train_indices.size, target_count):
            errors.append(f"{prefix}train prediction shape mismatch")
        if test_prediction.shape != (test_indices.size, target_count):
            errors.append(f"{prefix}test prediction shape mismatch")
        for split, prediction in (
            ("train", train_prediction),
            ("test", test_prediction),
        ):
            mask = np.asarray(arrays[prefix + split + "_prediction_finite"])
            if mask.dtype != np.bool_ or mask.shape != prediction.shape:
                errors.append(f"{prefix}{split} prediction mask mismatch")
            elif not np.array_equal(mask, finite_mask(prediction)):
                errors.append(f"{prefix}{split} prediction mask is stale")
    return errors


def _coerce_artifact(value):
    if isinstance(value, (str, Path)):
        return load_precision_diagnostic(value)
    if isinstance(value, dict) and set(value) == {"arrays", "metadata"}:
        return value["arrays"], value["metadata"]
    if isinstance(value, dict):
        return value, {}
    raise TypeError("precision artifact must be a path or mapping")


def _array_comparison(current, reference, allow_all_nonfinite=False):
    current = _as_float(current)
    reference = _as_float(reference)
    if current.shape != reference.shape:
        return {"shape_matches": False, "comparable": 0}
    current_finite = np.isfinite(current)
    reference_finite = np.isfinite(reference)
    comparable = current_finite & reference_finite
    mask_mismatch = float(np.mean(current_finite != reference_finite))
    if not comparable.any():
        return {
            "shape_matches": True,
            "comparable": 0,
            "all_nonfinite_allowed": bool(allow_all_nonfinite and mask_mismatch == 0.0),
            "finite_mask_mismatch_fraction": mask_mismatch,
        }
    difference = np.abs(current[comparable] - reference[comparable])
    return {
        "shape_matches": True,
        "comparable": int(comparable.sum()),
        "max_abs": float(difference.max()),
        "p50_abs": float(np.percentile(difference, 50)),
        "p95_abs": float(np.percentile(difference, 95)),
        "sign_change_fraction": float(
            np.mean(np.sign(current[comparable]) != np.sign(reference[comparable]))
        ),
        "finite_mask_mismatch_fraction": mask_mismatch,
    }


def _selection_comparison(current, reference, role):
    current_counts = np.asarray(current[role + "_selected_source_counts"])
    reference_counts = np.asarray(reference[role + "_selected_source_counts"])
    current_indices = np.asarray(current[role + "_selected_source_indices"])
    reference_indices = np.asarray(reference[role + "_selected_source_indices"])
    if current_counts.shape != reference_counts.shape:
        return {"shape_matches": False, "disagreements": None}
    jaccards = []
    disagreements = 0
    for target in range(current_counts.size):
        left = current_indices[target, : current_counts[target]]
        right = reference_indices[target, : reference_counts[target]]
        jaccards.append(jaccard(left, right))
        if not np.array_equal(left, right):
            disagreements += 1
    return {
        "shape_matches": current_indices.shape == reference_indices.shape,
        "targets": int(current_counts.size),
        "disagreements": int(disagreements),
        "mean_jaccard": float(np.mean(jaccards)),
        "min_jaccard": float(np.min(jaccards)),
    }


def _worst_metric(current, reference):
    worst = None
    for split in ("train", "test"):
        for name in CONSISTENCY_METRICS:
            key = f"{split}_{name}"
            left = _as_float(current[key])
            right = _as_float(reference[key])
            if left.shape != right.shape:
                continue
            finite = np.isfinite(left) & np.isfinite(right)
            if not finite.any():
                continue
            difference = np.where(finite, np.abs(left - right), -np.inf)
            index = np.unravel_index(int(np.argmax(difference)), difference.shape)
            candidate = {
                "split": split,
                "metric": name,
                "index": [int(value) for value in index],
                "current": float(left[index]),
                "reference": float(right[index]),
                "abs_difference": float(difference[index]),
            }
            denominator_key = f"{split}_denom_sb"
            denominator = _as_float(reference[denominator_key])
            if denominator.shape == right.shape:
                value = denominator[index]
                candidate["reference_denom_sb"] = (
                    float(value) if np.isfinite(value) else None
                )
            if worst is None or candidate["abs_difference"] > worst["abs_difference"]:
                worst = candidate
    return worst


def compare_precision(
    current,
    reference,
    gates=None,
    allow_provenance_mismatches=(),
):
    gates = dict(FLOAT64_GATES if gates is None else gates)
    allow_provenance_mismatches = tuple(allow_provenance_mismatches)
    unknown_allowances = sorted(
        set(allow_provenance_mismatches).difference(_PROVENANCE_FIELDS)
    )
    if unknown_allowances:
        raise ValueError(
            "unknown provenance mismatch fields: {}".format(
                ", ".join(unknown_allowances)
            )
        )
    current_arrays, current_metadata = _coerce_artifact(current)
    reference_arrays, reference_metadata = _coerce_artifact(reference)
    failures = []
    validity_failures = []
    structural = {
        "current": validate_precision_artifact(current_arrays, current_metadata),
        "reference": validate_precision_artifact(reference_arrays, reference_metadata),
    }
    if structural["current"]:
        failures.append("current_structure")
        validity_failures.append("current_structure")
    if structural["reference"]:
        failures.append("reference_structure")
        validity_failures.append("reference_structure")
    if set(current_arrays) != set(reference_arrays):
        failures.append("array_keys")
        validity_failures.append("array_keys")
    if failures:
        return {
            "gates": gates,
            "gates_passed": False,
            "comparison_valid": False,
            "failures": failures,
            "validity_failures": validity_failures,
            "structural_errors": structural,
            "provenance_mismatch_allowlist": list(allow_provenance_mismatches),
            "metrics": {},
            "mappings": {},
            "selection": {},
            "worst_metric": None,
        }

    for name in ("target_indices", "train_indices", "test_indices"):
        if not np.array_equal(current_arrays[name], reference_arrays[name]):
            failures.append(name)
            validity_failures.append(name)
    metadata_mismatches = []
    for name in _PROVENANCE_FIELDS:
        if current_metadata.get(name) != reference_metadata.get(name):
            metadata_mismatches.append(name)
    disallowed_metadata_mismatches = [
        name for name in metadata_mismatches if name not in allow_provenance_mismatches
    ]
    if disallowed_metadata_mismatches:
        failures.append("provenance")
        validity_failures.append("provenance")

    roles = _roles_from_arrays(current_arrays, current_metadata)
    selection = {}
    mappings = {}
    mask_mismatches = []
    mapping_rtol = gates["mapping_rtol"]
    mapping_atol = gates["mapping_atol"]
    for role in roles:
        selection[role] = _selection_comparison(current_arrays, reference_arrays, role)
        if not selection[role].get("shape_matches", False):
            failures.append(f"{role}_selection_shape")
            validity_failures.append(f"{role}_selection_shape")
        if gates.get("require_exact_selection", True) and selection[role].get(
            "disagreements"
        ):
            failures.append(f"{role}_selection")
        mappings[role] = {}
        for field in _MAPPING_NUMERIC_FIELDS:
            key = f"{role}_{field}"
            comparison = _array_comparison(
                current_arrays[key],
                reference_arrays[key],
                allow_all_nonfinite=field in ("cutoff_below", "cutoff_above"),
            )
            mappings[role][field] = comparison
            mask_mismatch = comparison.get("finite_mask_mismatch_fraction")
            if mask_mismatch is not None and field not in _MAPPING_DIAGNOSTIC_FIELDS:
                mask_mismatches.append(mask_mismatch)
            if comparison.get("comparable", 0) == 0:
                if not comparison.get("all_nonfinite_allowed", False):
                    failures.append(f"{role}_{field}_no_evidence")
                    validity_failures.append(f"{role}_{field}_no_evidence")
                continue
            left = _as_float(current_arrays[key])
            right = _as_float(reference_arrays[key])
            finite = np.isfinite(left) & np.isfinite(right)
            if field not in _MAPPING_DIAGNOSTIC_FIELDS and not np.allclose(
                left[finite],
                right[finite],
                rtol=mapping_rtol,
                atol=mapping_atol,
            ):
                failures.append(f"{role}_{field}")

    metrics = {"train": {}, "test": {}}
    metric_p95 = []
    for split in ("train", "test"):
        for name in CONSISTENCY_METRICS:
            key = f"{split}_{name}"
            comparison = _array_comparison(current_arrays[key], reference_arrays[key])
            metrics[split][name] = comparison
            if comparison.get("comparable", 0) == 0:
                failures.append(f"{split}_{name}_no_evidence")
                validity_failures.append(f"{split}_{name}_no_evidence")
                continue
            if name != "r_xy_n_sb":
                metric_p95.append(comparison["p95_abs"])
            mask_mismatches.append(comparison["finite_mask_mismatch_fraction"])

    max_metric_p95 = max(metric_p95) if metric_p95 else None
    if (
        max_metric_p95 is None
        or max_metric_p95 > gates["max_metric_p95_abs_difference"]
    ):
        failures.append("metric_p95")
    current_test = _as_float(current_arrays["test_r_xy_n_sb"])
    reference_test = _as_float(reference_arrays["test_r_xy_n_sb"])
    current_finite = current_test[np.isfinite(current_test)]
    reference_finite = reference_test[np.isfinite(reference_test)]
    if current_finite.size == 0 or reference_finite.size == 0:
        mean_difference = None
        failures.append("global_finite_mean_test_r_xy_n_sb_no_evidence")
        validity_failures.append("global_finite_mean_test_r_xy_n_sb_no_evidence")
    else:
        mean_difference = abs(float(current_finite.mean() - reference_finite.mean()))
        if (
            mean_difference
            > gates["max_abs_global_finite_mean_test_r_xy_n_sb_difference"]
        ):
            failures.append("global_finite_mean_test_r_xy_n_sb")
    normalized_differences = []
    normalized_relative_differences = []
    normalized_within_gate = True
    for split in ("train", "test"):
        left = _as_float(current_arrays[f"{split}_r_xy_n_sb"])
        right = _as_float(reference_arrays[f"{split}_r_xy_n_sb"])
        finite = np.isfinite(left) & np.isfinite(right)
        if not finite.any():
            normalized_within_gate = False
            continue
        difference = np.abs(left[finite] - right[finite])
        tolerance = gates["normalized_metric_atol"] + gates[
            "normalized_metric_rtol"
        ] * np.abs(right[finite])
        normalized_differences.append(difference)
        nonzero = right[finite] != 0
        if nonzero.any():
            normalized_relative_differences.append(
                difference[nonzero] / np.abs(right[finite][nonzero])
            )
        normalized_within_gate &= bool(np.all(difference <= tolerance))
    normalized_max = max(
        (float(values.max()) for values in normalized_differences if values.size),
        default=None,
    )
    normalized_relative_max = max(
        (
            float(values.max())
            for values in normalized_relative_differences
            if values.size
        ),
        default=None,
    )
    if normalized_max is None or not normalized_within_gate:
        failures.append("normalized_metric")
    mask_mismatch = max(mask_mismatches) if mask_mismatches else 1.0
    if mask_mismatch > gates["max_finite_mask_mismatch_fraction"]:
        failures.append("finite_mask")

    failures = list(dict.fromkeys(failures))
    validity_failures = list(dict.fromkeys(validity_failures))
    return {
        "gates": gates,
        "gates_passed": not failures,
        "comparison_valid": not validity_failures,
        "failures": failures,
        "validity_failures": validity_failures,
        "structural_errors": structural,
        "metadata_mismatches": metadata_mismatches,
        "provenance_mismatch_allowlist": list(allow_provenance_mismatches),
        "allowed_provenance_mismatches": [
            name for name in metadata_mismatches if name in allow_provenance_mismatches
        ],
        "disallowed_provenance_mismatches": disallowed_metadata_mismatches,
        "max_metric_p95_abs_difference": max_metric_p95,
        "metric_p95_scope": "canonical metrics except separately gated r_xy_n_sb",
        "abs_global_finite_mean_test_r_xy_n_sb_difference": mean_difference,
        "max_normalized_metric_difference": normalized_max,
        "max_normalized_metric_relative_difference": normalized_relative_max,
        "normalized_metric_within_gate": normalized_within_gate,
        "finite_mask_mismatch_fraction": mask_mismatch,
        "metrics": metrics,
        "mappings": mappings,
        "selection": selection,
        "worst_metric": _worst_metric(current_arrays, reference_arrays),
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("current")
    parser.add_argument("reference")
    parser.add_argument("--output", default="-")
    parser.add_argument(
        "--allow-provenance-mismatch",
        action="append",
        choices=_PROVENANCE_FIELDS,
        default=[],
        help="provenance field allowed to differ; repeat for multiple fields",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="return success for a structurally valid comparison even when numerical gates fail",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    comparison = compare_precision(
        args.current,
        args.reference,
        allow_provenance_mismatches=args.allow_provenance_mismatch,
    )
    if args.output == "-":
        print(json.dumps(comparison, indent=2, sort_keys=True, allow_nan=False))
    else:
        atomic_write_json(args.output, comparison)
    passed = (
        comparison["comparison_valid"]
        if args.report_only
        else comparison["gates_passed"]
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
