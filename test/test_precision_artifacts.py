import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.precision import (
    CONSISTENCY_METRICS,
    MAPPER_ROLES,
    PRECISION_SCHEMA_VERSION,
    build_parser,
    compare_precision,
    load_precision_diagnostic,
    save_precision_diagnostic,
    validate_precision_artifact,
)


def _artifact():
    target_count = 3
    arrays = {
        "target_indices": np.asarray([2, 5, 9], dtype=np.int64),
        "train_indices": np.asarray([0, 1, 2, 3], dtype=np.int64),
        "test_indices": np.asarray([4, 5], dtype=np.int64),
    }
    metric = np.asarray([[[0.2, 0.4, 0.6]]], dtype=np.float64)
    for split_offset, split in enumerate(("train", "test")):
        for metric_offset, name in enumerate(CONSISTENCY_METRICS):
            key = f"{split}_{name}"
            values = metric + split_offset * 0.01 + metric_offset * 0.001
            arrays[key] = values.copy()
            arrays[key + "_finite"] = np.isfinite(values)

    correlations = np.asarray(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [0.2, 0.3, 0.4],
        ],
        dtype=np.float64,
    )
    counts = np.asarray([2, 1, 3], dtype=np.int64)
    indices = np.asarray([[0, 2, -1], [1, -1, -1], [0, 2, 3]], dtype=np.int64)
    coefficients = np.asarray(
        [[0.5, 0.25, np.nan], [0.75, np.nan, np.nan], [0.2, 0.3, 0.4]],
        dtype=np.float64,
    )
    for role_offset, role in enumerate(MAPPER_ROLES):
        arrays[f"{role}_correlations"] = correlations + role_offset * 0.001
        arrays[f"{role}_thresholds"] = np.asarray([0.4, 0.5, 0.6])
        arrays[f"{role}_cutoff_below"] = np.asarray([0.2, 0.3, 0.4])
        arrays[f"{role}_cutoff_above"] = np.asarray([0.4, 0.5, 0.6])
        arrays[f"{role}_selected_source_counts"] = counts.copy()
        arrays[f"{role}_selected_source_indices"] = indices.copy()
        arrays[f"{role}_coefficients"] = coefficients + role_offset * 0.001
        arrays[f"{role}_intercepts"] = np.asarray([0.1, 0.2, 0.3])
        train_prediction = np.arange(12, dtype=np.float64).reshape(4, target_count)
        test_prediction = np.arange(6, dtype=np.float64).reshape(2, target_count)
        arrays[f"{role}_train_prediction"] = train_prediction + role_offset
        arrays[f"{role}_test_prediction"] = test_prediction + role_offset
        arrays[f"{role}_train_prediction_finite"] = np.isfinite(train_prediction)
        arrays[f"{role}_test_prediction_finite"] = np.isfinite(test_prediction)
    metadata = {
        "schema_version": PRECISION_SCHEMA_VERSION,
        "roles": list(MAPPER_ROLES),
        "configuration": {
            "percentile": 75,
            "alpha": 1.0,
            "identity": False,
            "solver": "auto",
        },
        "dataset": {"source_fly": "fly_087", "target_fly": "fly_089"},
        "seed": 1000,
        "bootstrap_index": 0,
        "fold_index": 0,
        "dtypes": {"input": "float64", "correlation": "float64"},
        "input_checksums": {"full": {"X_train": "abc"}},
        "repository": {"commit": "abc123", "dirty": False},
        "backend_requested": "cpu",
        "backend_effective": "cpu",
    }
    return {"arrays": arrays, "metadata": metadata}


def test_identical_precision_artifacts_pass_every_gate():
    reference = _artifact()
    current = copy.deepcopy(reference)

    comparison = compare_precision(current, reference)

    assert comparison["gates_passed"]
    assert comparison["failures"] == []
    assert comparison["selection"]["full"]["min_jaccard"] == 1.0


@pytest.mark.parametrize(
    "mutation",
    (
        lambda artifact: artifact["arrays"].pop("full_coefficients"),
        lambda artifact: artifact["arrays"].update({"unexpected": np.asarray([1.0])}),
        lambda artifact: artifact["metadata"].pop("dataset"),
        lambda artifact: artifact["arrays"].update(
            {"full_coefficients": np.asarray([object()], dtype=object)}
        ),
    ),
)
def test_precision_structure_rejects_missing_extra_or_non_numeric_evidence(mutation):
    current = _artifact()
    mutation(current)

    comparison = compare_precision(current, _artifact())

    assert not comparison["gates_passed"]
    assert "current_structure" in comparison["failures"]


def test_precision_comparison_rejects_target_order_and_provenance_mismatch():
    current = _artifact()
    current["arrays"]["target_indices"] = current["arrays"]["target_indices"][::-1]
    current["metadata"]["dataset"]["direction"] = "left"
    reference = _artifact()
    reference["metadata"]["dataset"]["direction"] = "right"

    comparison = compare_precision(current, reference)

    assert not comparison["gates_passed"]
    assert "target_indices" in comparison["failures"]
    assert "provenance" in comparison["failures"]


def test_precision_comparison_allows_and_reports_named_provenance_mismatches():
    current = _artifact()
    reference = _artifact()
    current["metadata"]["dtypes"] = {"input": "float32"}
    current["metadata"]["dataset"]["array_dtype"] = "float32"
    reference["metadata"]["dataset"]["array_dtype"] = "float64"
    current["metadata"]["input_checksums"] = {"full": {"X_train": "float32"}}

    strict = compare_precision(current, reference)
    comparison = compare_precision(
        current,
        reference,
        allow_provenance_mismatches=("dtypes", "dataset", "input_checksums"),
    )

    assert not strict["gates_passed"]
    assert strict["disallowed_provenance_mismatches"] == [
        "dataset",
        "dtypes",
        "input_checksums",
    ]
    assert comparison["gates_passed"]
    assert comparison["metadata_mismatches"] == [
        "dataset",
        "dtypes",
        "input_checksums",
    ]
    assert comparison["allowed_provenance_mismatches"] == [
        "dataset",
        "dtypes",
        "input_checksums",
    ]
    assert comparison["disallowed_provenance_mismatches"] == []


def test_precision_cli_accepts_repeated_provenance_allowances():
    args = build_parser().parse_args(
        [
            "current.npz",
            "reference.npz",
            "--allow-provenance-mismatch",
            "dtypes",
            "--allow-provenance-mismatch",
            "input_checksums",
        ]
    )

    assert args.allow_provenance_mismatch == ["dtypes", "input_checksums"]


def test_precision_comparison_rejects_unknown_provenance_allowance():
    with pytest.raises(ValueError, match="unknown provenance mismatch fields"):
        compare_precision(
            _artifact(),
            _artifact(),
            allow_provenance_mismatches=("unknown",),
        )


def test_precision_comparison_rejects_selection_and_mapping_changes():
    current = _artifact()
    current["arrays"]["full_selected_source_indices"][0, 1] = 3
    current["arrays"]["half_1_coefficients"][0, 0] += 1e-4
    current["arrays"]["half_2_test_prediction"][0, 0] += 1e-4

    comparison = compare_precision(current, _artifact())

    assert not comparison["gates_passed"]
    assert "full_selection" in comparison["failures"]
    assert "half_1_coefficients" in comparison["failures"]
    assert "half_2_test_prediction" in comparison["failures"]
    assert comparison["comparison_valid"]


def test_precision_cutoff_neighbors_are_diagnostic_not_acceptance_gates():
    current = _artifact()
    current["arrays"]["half_1_cutoff_below"][0] -= 0.5

    comparison = compare_precision(current, _artifact())

    assert comparison["gates_passed"]
    assert comparison["mappings"]["half_1"]["cutoff_below"]["max_abs"] == 0.5


def test_precision_report_mode_is_distinct_from_float64_acceptance():
    current = _artifact()
    current["arrays"]["test_r_xy_n_sb"][0] += 0.5

    comparison = compare_precision(current, _artifact())
    args = build_parser().parse_args(["current.npz", "reference.npz", "--report-only"])

    assert comparison["comparison_valid"]
    assert not comparison["gates_passed"]
    assert args.report_only


def test_normalized_metric_uses_its_dedicated_mean_and_max_gates():
    current = _artifact()
    current["arrays"]["test_r_xy_n_sb"] += 5e-12

    comparison = compare_precision(current, _artifact())

    assert comparison["gates_passed"]
    assert comparison["max_normalized_metric_difference"] == pytest.approx(5e-12)
    assert comparison["metric_p95_scope"].endswith("r_xy_n_sb")


def test_normalized_metric_gate_scales_for_large_reference_values():
    reference = _artifact()
    current = copy.deepcopy(reference)
    reference["arrays"]["train_r_xy_n_sb"][:] = 10_000.0
    current["arrays"]["train_r_xy_n_sb"][:] = 10_000.0 + 5e-7

    comparison = compare_precision(current, reference)

    assert comparison["gates_passed"]
    assert comparison["max_normalized_metric_difference"] > 1e-8
    assert comparison["max_normalized_metric_relative_difference"] < 1e-10


def test_precision_comparison_rejects_arrays_without_finite_evidence():
    current = _artifact()
    reference = _artifact()
    for artifact in (current, reference):
        artifact["arrays"]["test_r_xx"][:] = np.nan
        artifact["arrays"]["test_r_xx_finite"][:] = False

    comparison = compare_precision(current, reference)

    assert not comparison["gates_passed"]
    assert not comparison["comparison_valid"]
    assert "test_r_xx_no_evidence" in comparison["failures"]


def test_precision_report_identifies_large_canonical_metric_cell():
    current = _artifact()
    current["arrays"]["test_r_xy_n_sb"][0, 0, 1] += 0.54

    comparison = compare_precision(current, _artifact())

    assert not comparison["gates_passed"]
    assert comparison["worst_metric"]["metric"] == "r_xy_n_sb"
    assert comparison["worst_metric"]["index"] == [0, 0, 1]
    assert comparison["worst_metric"]["abs_difference"] == pytest.approx(0.54)


def test_precision_comparison_rejects_finite_mask_mismatch():
    current = _artifact()
    current["arrays"]["test_r_xy"][0, 0, 0] = np.nan
    current["arrays"]["test_r_xy_finite"][0, 0, 0] = False

    comparison = compare_precision(current, _artifact())

    assert not comparison["gates_passed"]
    assert "finite_mask" in comparison["failures"]


class _FakeMapper:
    def __init__(self):
        self._n_source = 4
        self._source_indices = [
            np.asarray([0, 2]),
            np.asarray([1]),
            np.asarray([0, 2, 3]),
        ]
        self._mappers = [
            SimpleNamespace(
                coef_=np.asarray([0.5, 0.25]), intercept_=np.asarray([0.1])
            ),
            SimpleNamespace(coef_=np.asarray([0.75]), intercept_=np.asarray([0.2])),
            SimpleNamespace(
                coef_=np.asarray([0.2, 0.3, 0.4]), intercept_=np.asarray([0.3])
            ),
        ]
        self._corrs = _artifact()["arrays"]["full_correlations"]
        self._thresholds = np.asarray([0.4, 0.5, 0.6])

    def predict(self, X):
        X = np.asarray(X)
        return X[:, :3]


def test_save_precision_diagnostic_writes_numeric_npz_and_provenance(tmp_path):
    X = np.arange(24, dtype=np.float64).reshape(6, 4)
    Y = np.arange(18, dtype=np.float64).reshape(6, 3)
    fits = {}
    for role in MAPPER_ROLES:
        fits[role] = {
            "mapper": _FakeMapper(),
            "X_train": X[:4],
            "X_test": X[4:],
            "Y_train": Y[:4],
            "Y_test": Y[4:],
        }
    metrics = {
        split: {
            name: np.asarray([0.2, 0.4, 0.6], dtype=np.float64)
            for name in CONSISTENCY_METRICS
        }
        for split in ("train", "test")
    }
    metadata = _artifact()["metadata"]
    metadata.pop("schema_version")
    metadata.pop("roles")
    metadata.pop("input_checksums")
    path = tmp_path / "learned.npz"
    owner = tmp_path / "result.json"

    records = save_precision_diagnostic(
        path,
        fits,
        metrics,
        target_indices=[2, 5, 9],
        train_indices=[0, 1, 2, 3],
        test_indices=[4, 5],
        metadata=metadata,
        owner_path=owner,
    )
    arrays, sidecar = load_precision_diagnostic(path)

    assert validate_precision_artifact(arrays, sidecar) == []
    assert records["arrays"]["path"] == "learned.npz"
    assert records["metadata"]["path"] == "learned.json"
    assert all(not value.dtype.hasobject for value in arrays.values())
    with open(path.with_suffix(".json")) as stream:
        saved_metadata = json.load(stream)
    assert set(saved_metadata["input_checksums"]) == set(MAPPER_ROLES)


def test_precision_structure_validates_diagnostic_target_selection():
    artifact = _artifact()
    artifact["metadata"]["dataset"]["diagnostic_target_selection"] = {
        "strategy": "evenly_spaced",
        "full_target_count": 12,
        "selected_target_count": 3,
        "positions": [0, 4, 11],
        "original_target_ids": [2, 5, 9],
    }

    assert validate_precision_artifact(artifact["arrays"], artifact["metadata"]) == []

    artifact["metadata"]["dataset"]["diagnostic_target_selection"][
        "original_target_ids"
    ] = [2, 5, 10]
    assert "diagnostic original target IDs do not match arrays" in (
        validate_precision_artifact(artifact["arrays"], artifact["metadata"])
    )
