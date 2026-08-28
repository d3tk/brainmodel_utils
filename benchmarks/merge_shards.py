"""Concatenate sharded consistency arrays along the bootstrap axis."""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np

from benchmarks.artifacts import (
    artifact_record,
    atomic_savez,
    atomic_write_json,
    precision_metrics_path,
    relative_artifact_path,
)
from benchmarks.benchmark_neural_backends import (
    WORKLOAD_KEYS,
    _consistency_metric_summaries,
)
from benchmarks.precision import CONSISTENCY_METRICS

CONSISTENCY_KEY = WORKLOAD_KEYS["consistency"]
METRIC_NAMES = CONSISTENCY_METRICS
EXPECTED_METRIC_KEYS = frozenset(METRIC_NAMES)
EXPECTED_NPZ_KEYS = frozenset(
    f"{split}_{metric}" for split in ("train", "test") for metric in METRIC_NAMES
)


def _load_document(path):
    with open(path) as stream:
        document = json.load(stream)
    if not isinstance(document, dict):
        raise TypeError(f"shard JSON must contain an object: {path}")
    return document


def _validate_arrays(arrays, source):
    if set(arrays) != {"train", "test"}:
        raise ValueError(f"{source} must contain exactly train and test arrays")
    reference_shape = None
    validated = {"train": {}, "test": {}}
    for split in ("train", "test"):
        names = set(arrays[split])
        if names != EXPECTED_METRIC_KEYS:
            missing = sorted(EXPECTED_METRIC_KEYS.difference(names))
            extra = sorted(names.difference(EXPECTED_METRIC_KEYS))
            raise ValueError(
                f"{source} {split} metric keys differ: missing={missing}, extra={extra}"
            )
        for name in METRIC_NAMES:
            values = np.asarray(arrays[split][name])
            if values.ndim != 3:
                raise ValueError(
                    f"{source} {split}_{name} must be [bootstrap, fold, target], got {values.shape}"
                )
            if values.dtype.kind not in "fc":
                raise ValueError(
                    f"{source} {split}_{name} must be floating point, got {values.dtype}"
                )
            if reference_shape is None:
                reference_shape = values.shape
            elif values.shape != reference_shape:
                raise ValueError(
                    f"metric shapes differ within {source}: {values.shape} != {reference_shape}"
                )
            validated[split][name] = values
    return validated


def arrays_from_npz(path):
    with np.load(path, allow_pickle=False) as loaded:
        keys = frozenset(loaded.files)
        if keys != EXPECTED_NPZ_KEYS:
            missing = sorted(EXPECTED_NPZ_KEYS.difference(keys))
            extra = sorted(keys.difference(EXPECTED_NPZ_KEYS))
            raise ValueError(
                f"metric NPZ keys differ for {path}: missing={missing}, extra={extra}"
            )
        arrays = {"train": {}, "test": {}}
        for key in loaded.files:
            split, name = key.split("_", 1)
            arrays[split][name] = np.asarray(loaded[key])
    return _validate_arrays(arrays, str(path))


def load_shard(result_path):
    result_path = Path(result_path)
    document = _load_document(result_path)
    if document.get("completion", {}).get("status") != "complete":
        raise ValueError(f"shard is not marked complete: {result_path}")
    consistency = document.get("benchmarks", {}).get(CONSISTENCY_KEY)
    if not isinstance(consistency, dict):
        raise TypeError(f"shard is missing {CONSISTENCY_KEY}: {result_path}")
    metrics_file = precision_metrics_path(result_path, consistency)
    arrays = arrays_from_npz(metrics_file)
    bootstrap_iters = int(consistency.get("bootstrap_iterations", -1))
    array_bootstraps = arrays["test"]["r_xy_n_sb"].shape[0]
    if bootstrap_iters != array_bootstraps:
        raise ValueError(
            f"bootstrap metadata does not match arrays for {result_path}: {bootstrap_iters} != {array_bootstraps}"
        )
    return {
        "path": result_path,
        "document": document,
        "arrays": arrays,
        "seed": int(document["seed"]),
        "bootstrap_iters": bootstrap_iters,
        "seconds": float(consistency["total_seconds"]),
    }


def _without_keys(value, keys):
    copied = copy.deepcopy(value)
    if isinstance(copied, dict):
        for key in keys:
            copied.pop(key, None)
    return copied


def _identity(document):
    dataset = document.get("dataset", {})
    metadata = document.get("metadata", {})
    consistency = document["benchmarks"][CONSISTENCY_KEY]
    return {
        "schema_version": document.get("schema_version"),
        "suite": document.get("suite"),
        "workloads": document.get("workloads"),
        "backend_requested": document.get("backend_requested"),
        "backend_effective": document.get("backend_effective"),
        "backend_effective_by_workload": document.get("backend_effective_by_workload"),
        "split_seed": document.get("split_seed"),
        "dtypes": document.get("dtypes"),
        "config": _without_keys(document.get("config"), ("bootstrap_iters",)),
        "dataset": {
            "origin": dataset.get("origin"),
            "fixture": dataset.get("fixture"),
            "provenance": _without_keys(
                dataset.get("provenance"), ("preparation_seconds",)
            ),
            "source": dataset.get("source"),
            "target": dataset.get("target"),
            "split_strategy": dataset.get("split_strategy"),
            "train_test_indices": dataset.get("train_test_indices"),
        },
        "consistency": {
            "mapping": consistency.get("mapping"),
            "metric": consistency.get("metric"),
            "train_test_splits": consistency.get("train_test_splits"),
            "workers": consistency.get("workers"),
            "ridge_parameters": consistency.get("ridge_parameters"),
            "warmup": consistency.get("warmup"),
        },
        "repository": metadata.get("repository"),
        "python": metadata.get("python"),
        "libraries": metadata.get("libraries"),
        "campaign": _without_keys(document.get("campaign"), ("assignment_checksum",)),
    }


def _require_compatible(shards):
    reference = _identity(shards[0]["document"])
    for shard in shards[1:]:
        if _identity(shard["document"]) != reference:
            raise ValueError(
                "shard configuration/provenance differs from {}: {}".format(
                    shards[0]["path"], shard["path"]
                )
            )


def _ordered_shards(shards, expected_seed=None, expected_bootstrap_iters=None):
    ordered = sorted(shards, key=lambda item: (item["seed"], str(item["path"])))
    intervals = [
        (shard["seed"], shard["seed"] + shard["bootstrap_iters"]) for shard in ordered
    ]
    for index in range(1, len(intervals)):
        previous = intervals[index - 1]
        current = intervals[index]
        if current[0] < previous[1]:
            raise ValueError(f"bootstrap seed ranges overlap: {previous} and {current}")
        if current[0] > previous[1]:
            raise ValueError(
                f"bootstrap seed ranges are not contiguous: {previous} and {current}"
            )
    if expected_seed is not None and intervals[0][0] != expected_seed:
        raise ValueError(
            f"merged range starts at {intervals[0][0]}, expected {expected_seed}"
        )
    total = sum(end - start for start, end in intervals)
    if expected_bootstrap_iters is not None and total != expected_bootstrap_iters:
        raise ValueError(
            f"merged bootstrap total is {total}, expected {expected_bootstrap_iters}"
        )
    if (
        expected_seed is not None
        and expected_bootstrap_iters is not None
        and intervals[-1][1] != expected_seed + expected_bootstrap_iters
    ):
        raise ValueError("merged range does not end at the expected seed")
    return ordered


def merge_raw_arrays(shards, expected_seed=None, expected_bootstrap_iters=None):
    if not shards:
        raise ValueError("at least one shard is required")
    _require_compatible(shards)
    ordered = _ordered_shards(
        shards,
        expected_seed=expected_seed,
        expected_bootstrap_iters=expected_bootstrap_iters,
    )
    reference = _validate_arrays(ordered[0]["arrays"], str(ordered[0]["path"]))
    for shard in ordered[1:]:
        current = _validate_arrays(shard["arrays"], str(shard["path"]))
        for split in ("train", "test"):
            for name in METRIC_NAMES:
                if current[split][name].shape[1:] != reference[split][name].shape[1:]:
                    raise ValueError(
                        f"metric shape differs across shards for {split}_{name}"
                    )
                if current[split][name].dtype != reference[split][name].dtype:
                    raise ValueError(
                        f"metric dtype differs across shards for {split}_{name}"
                    )
    merged = {"train": {}, "test": {}}
    for split in ("train", "test"):
        for name in METRIC_NAMES:
            merged[split][name] = np.concatenate(
                [shard["arrays"][split][name] for shard in ordered], axis=0
            )
    return ordered, merged


def compare_merged(current, reference, atol=1e-12):
    current = _validate_arrays(current, "candidate")
    reference = _validate_arrays(reference, "reference")
    report = {
        "atol": float(atol),
        "keys": {},
        "identical_finite_masks": True,
        "has_finite_evidence": True,
        "max_abs": 0.0,
    }
    for split in ("train", "test"):
        for name in METRIC_NAMES:
            key = f"{split}_{name}"
            left = current[split][name]
            right = reference[split][name]
            if left.shape != right.shape:
                raise ValueError(
                    f"comparison shape differs for {key}: {left.shape} != {right.shape}"
                )
            left_mask = np.isfinite(left)
            right_mask = np.isfinite(right)
            masks_match = bool(np.array_equal(left_mask, right_mask))
            report["identical_finite_masks"] &= masks_match
            both = left_mask & right_mask
            has_evidence = bool(both.any())
            report["has_finite_evidence"] &= has_evidence
            max_abs = (
                float(np.max(np.abs(left[both] - right[both])))
                if has_evidence
                else None
            )
            if max_abs is not None:
                report["max_abs"] = max(report["max_abs"], max_abs)
            report["keys"][key] = {
                "shape": list(left.shape),
                "finite_count": int(np.count_nonzero(both)),
                "identical_finite_masks": masks_match,
                "max_abs": max_abs,
                "within_atol": bool(masks_match and has_evidence and max_abs <= atol),
            }
    report["passed"] = bool(
        report["identical_finite_masks"]
        and report["has_finite_evidence"]
        and report["max_abs"] <= atol
    )
    return report


def _flatten_arrays(arrays):
    return {
        f"{split}_{name}": arrays[split][name]
        for split in ("train", "test")
        for name in METRIC_NAMES
    }


def merged_document(
    ordered_shards,
    arrays,
    wall_seconds,
    merge_seconds,
    output_path,
    metrics_path,
):
    document = copy.deepcopy(ordered_shards[0]["document"])
    consistency = document["benchmarks"][CONSISTENCY_KEY]
    summaries = _consistency_metric_summaries(arrays)
    primary = summaries["test"]["r_xy_n_sb"]
    metric = np.asarray(arrays["test"]["r_xy_n_sb"])
    shard_seconds = [shard["seconds"] for shard in ordered_shards]
    consistency.update(
        {
            "total_seconds": float(wall_seconds) + float(merge_seconds),
            "bootstrap_iterations": int(metric.shape[0]),
            "result_shape": list(metric.shape),
            "metric_summaries": summaries,
            "finite_result_fraction": primary["global"]["finite_fraction"],
            "mean_test_r_xy_n_sb": primary["global"]["mean"],
            "warmup": any(
                shard["document"]["benchmarks"][CONSISTENCY_KEY].get("warmup")
                for shard in ordered_shards
            ),
            "precision_metrics": artifact_record(output_path, metrics_path),
        }
    )
    consistency.pop("precision_metrics_file", None)
    document["config"]["bootstrap_iters"] = int(metric.shape[0])
    document["seed"] = ordered_shards[0]["seed"]
    document["shard_merge"] = {
        "n_shards": len(ordered_shards),
        "shard_files": [
            relative_artifact_path(output_path, shard["path"])
            for shard in ordered_shards
        ],
        "seed_ranges": [
            {"start": shard["seed"], "count": shard["bootstrap_iters"]}
            for shard in ordered_shards
        ],
        "total_bootstrap_iterations": int(metric.shape[0]),
        "shard_seconds": shard_seconds,
        "load_imbalance": (
            float(max(shard_seconds) / min(shard_seconds))
            if min(shard_seconds) > 0
            else None
        ),
        "merge_seconds": float(merge_seconds),
        "wall_seconds_end_to_end": float(wall_seconds) + float(merge_seconds),
        "gpu_hours": float(sum(shard_seconds) / 3600.0),
    }
    document["completion"] = {
        "status": "complete",
        "artifact_schema_version": 1,
    }
    return document


def merge_shard_files(
    result_paths,
    output_path,
    wall_seconds=None,
    metrics_path=None,
    reference_metrics=None,
    atol=1e-12,
    expected_seed=None,
    expected_bootstrap_iters=None,
):
    started = time.perf_counter()
    shards = [load_shard(path) for path in result_paths]
    ordered, arrays = merge_raw_arrays(
        shards,
        expected_seed=expected_seed,
        expected_bootstrap_iters=expected_bootstrap_iters,
    )
    output_path = Path(output_path)
    metrics_path = (
        Path(metrics_path)
        if metrics_path is not None
        else output_path.with_suffix(".metrics.npz")
    )
    atomic_savez(metrics_path, _flatten_arrays(arrays))
    if wall_seconds is None:
        wall_seconds = max(shard["seconds"] for shard in ordered)
    merge_seconds = time.perf_counter() - started
    document = merged_document(
        ordered,
        arrays,
        wall_seconds,
        merge_seconds,
        output_path,
        metrics_path,
    )
    if reference_metrics is not None:
        document["merge_validation"] = compare_merged(
            arrays, arrays_from_npz(reference_metrics), atol=atol
        )
        if not document["merge_validation"]["passed"]:
            raise ValueError("merged shard metrics did not match the reference")
    final_merge_seconds = time.perf_counter() - started
    document["shard_merge"]["merge_seconds"] = final_merge_seconds
    document["shard_merge"]["wall_seconds_end_to_end"] = (
        float(wall_seconds) + final_merge_seconds
    )
    document["shard_merge"]["timing_scope"] = (
        "load, validate, merge, metric artifact write, and completion-record preparation"
    )
    document["benchmarks"][CONSISTENCY_KEY]["total_seconds"] = (
        float(wall_seconds) + final_merge_seconds
    )
    atomic_write_json(output_path, document)
    return document, arrays


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics-output")
    parser.add_argument("--reference-metrics")
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--expected-seed", type=int)
    parser.add_argument("--expected-bootstrap-iters", type=int)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    document, _arrays = merge_shard_files(
        args.inputs,
        args.output,
        metrics_path=args.metrics_output,
        reference_metrics=args.reference_metrics,
        atol=args.atol,
        expected_seed=args.expected_seed,
        expected_bootstrap_iters=args.expected_bootstrap_iters,
    )
    if args.reference_metrics:
        sidecar = Path(args.output).with_suffix(".compare.json")
        atomic_write_json(sidecar, document["merge_validation"])
    print(json.dumps(document["shard_merge"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
