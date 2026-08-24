"""Concatenate sharded consistency arrays along the bootstrap axis.

Do not average JSON means or medians. Reload the raw metric tensors, join them
in seed order, then recompute summaries.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from benchmarks.benchmark_neural_backends import (
    WORKLOAD_KEYS,
    _consistency_metric_summaries,
)

IDENTITY_PATHS = (
    ("dataset", "train_test_indices"),
    ("dataset", "split_strategy"),
    ("dataset", "source", "dtype"),
    ("dataset", "target", "dtype"),
    ("dataset", "source", "shape"),
    ("dataset", "target", "shape"),
    ("config", "percentile"),
    ("config", "splits"),
    ("backend_requested",),
    ("benchmarks", "linregress_consistency", "mapping"),
)


def _lookup(document, path):
    value = document
    for key in path:
        value = value[key]
    return value


def _metrics_path(result_path, document=None):
    if document is not None:
        recorded = (
            document.get("benchmarks", {})
            .get(WORKLOAD_KEYS["consistency"], {})
            .get("precision_metrics_file")
        )
        if recorded:
            return Path(recorded)
    return Path(result_path).with_suffix(".metrics.npz")


def arrays_from_npz(path):
    loaded = np.load(path)
    arrays = {"train": {}, "test": {}}
    for key in loaded.files:
        split, name = key.split("_", 1)
        arrays[split][name] = np.asarray(loaded[key])
    return arrays


def load_shard(result_path):
    result_path = Path(result_path)
    with open(result_path) as stream:
        document = json.load(stream)
    metrics_file = _metrics_path(result_path, document)
    if not metrics_file.is_file():
        raise FileNotFoundError(
            "shard {} is missing raw metric arrays at {}".format(
                result_path, metrics_file
            )
        )
    consistency = document["benchmarks"][WORKLOAD_KEYS["consistency"]]
    return {
        "path": result_path,
        "document": document,
        "arrays": arrays_from_npz(metrics_file),
        "seed": int(document["seed"]),
        "bootstrap_iters": int(consistency["bootstrap_iterations"]),
        "seconds": float(consistency["total_seconds"]),
    }


def _require_compatible(shards):
    reference = shards[0]["document"]
    for shard in shards[1:]:
        for path in IDENTITY_PATHS:
            left = _lookup(reference, path)
            right = _lookup(shard["document"], path)
            if left != right:
                raise ValueError(
                    "shard identity mismatch at {}: {} vs {}".format(
                        ".".join(path), left, right
                    )
                )


def merge_raw_arrays(shards):
    ordered = sorted(shards, key=lambda item: (item["seed"], str(item["path"])))
    _require_compatible(ordered)
    merged = {"train": {}, "test": {}}
    for split in ("train", "test"):
        names = sorted(ordered[0]["arrays"][split])
        for name in names:
            merged[split][name] = np.concatenate(
                [shard["arrays"][split][name] for shard in ordered], axis=0
            )
    return ordered, merged


def compare_merged(current, reference, atol=1e-12):
    report = {
        "atol": atol,
        "keys": {},
        "identical_finite_masks": True,
        "max_abs": 0.0,
    }
    for split in ("train", "test"):
        for name in sorted(reference[split]):
            key = "{}_{}".format(split, name)
            left = np.asarray(current[split][name])
            right = np.asarray(reference[split][name])
            left_mask = np.isfinite(left)
            right_mask = np.isfinite(right)
            masks_match = bool(np.array_equal(left_mask, right_mask))
            report["identical_finite_masks"] = (
                report["identical_finite_masks"] and masks_match
            )
            both = left_mask & right_mask
            max_abs = (
                float(np.max(np.abs(left[both] - right[both]))) if both.any() else 0.0
            )
            report["max_abs"] = max(report["max_abs"], max_abs)
            report["keys"][key] = {
                "identical_finite_masks": masks_match,
                "max_abs": max_abs,
                "within_atol": bool(max_abs <= atol),
            }
    report["passed"] = report["identical_finite_masks"] and report["max_abs"] <= atol
    return report


def write_arrays(path, arrays):
    payload = {
        "{}_{}".format(split, name): values
        for split, block in arrays.items()
        for name, values in block.items()
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return str(path)


def merged_document(ordered_shards, arrays, wall_seconds, merge_seconds):
    document = json.loads(json.dumps(ordered_shards[0]["document"]))
    consistency = document["benchmarks"][WORKLOAD_KEYS["consistency"]]
    summaries = _consistency_metric_summaries(arrays)
    primary = summaries["test"]["r_xy_n_sb"]
    metric = np.asarray(arrays["test"]["r_xy_n_sb"])
    shard_seconds = [shard["seconds"] for shard in ordered_shards]
    consistency.update(
        {
            "total_seconds": float(wall_seconds),
            "bootstrap_iterations": int(metric.shape[0]),
            "result_shape": list(metric.shape),
            "metric_summaries": summaries,
            "finite_result_fraction": primary["global"]["finite_fraction"],
            "mean_test_r_xy_n_sb": primary["global"]["mean"],
            "warmup": any(
                shard["document"]["benchmarks"][WORKLOAD_KEYS["consistency"]].get(
                    "warmup"
                )
                for shard in ordered_shards
            ),
        }
    )
    document["seed"] = ordered_shards[0]["seed"]
    document["shard_merge"] = {
        "n_shards": len(ordered_shards),
        "shard_files": [str(shard["path"]) for shard in ordered_shards],
        "shard_seeds": [shard["seed"] for shard in ordered_shards],
        "shard_bootstrap_iters": [shard["bootstrap_iters"] for shard in ordered_shards],
        "shard_seconds": shard_seconds,
        "load_imbalance": (
            float(max(shard_seconds) / min(shard_seconds))
            if min(shard_seconds) > 0
            else None
        ),
        "merge_seconds": float(merge_seconds),
        "wall_seconds_end_to_end": float(wall_seconds),
        "gpu_hours": float(sum(shard_seconds) / 3600.0),
    }
    return document


def merge_shard_files(result_paths, output_path, wall_seconds=None):
    started = time.perf_counter()
    shards = [load_shard(path) for path in result_paths]
    ordered, arrays = merge_raw_arrays(shards)
    merge_seconds = time.perf_counter() - started
    if wall_seconds is None:
        wall_seconds = max(shard["seconds"] for shard in ordered) + merge_seconds
    document = merged_document(ordered, arrays, wall_seconds, merge_seconds)
    output_path = Path(output_path)
    metrics_path = output_path.with_suffix(".metrics.npz")
    document["benchmarks"][WORKLOAD_KEYS["consistency"]][
        "precision_metrics_file"
    ] = write_arrays(metrics_path, arrays)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as stream:
        json.dump(document, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return document, arrays


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference-metrics")
    parser.add_argument("--atol", type=float, default=1e-12)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    document, arrays = merge_shard_files(args.inputs, args.output)
    if args.reference_metrics:
        report = compare_merged(
            arrays, arrays_from_npz(args.reference_metrics), atol=args.atol
        )
        sidecar = Path(args.output).with_suffix(".compare.json")
        with open(sidecar, "w") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
        if not report["passed"]:
            return 1
    print(json.dumps(document["shard_merge"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
