"""Launch restart-safe benchmark comparisons with unique seed ranges."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from benchmarks.artifacts import (
    artifact_record,
    atomic_promote,
    atomic_write_json,
    relative_artifact_path,
    sha256_file,
    validate_artifact_record,
)
from benchmarks.config_manifest import (
    configuration_by_id,
    load_configuration_manifest,
    repository_identity,
)
from benchmarks.merge_shards import CONSISTENCY_KEY, arrays_from_npz, merge_shard_files

CAMPAIGN_SCHEMA_VERSION = 1


def _canonical_sha256(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _parse_target_indices(value):
    try:
        indices = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "target indices must be comma-separated integers"
        ) from error
    if not indices or any(index < 0 for index in indices):
        raise argparse.ArgumentTypeError("target indices must be non-negative")
    if indices != sorted(set(indices)):
        raise argparse.ArgumentTypeError("target indices must be sorted and unique")
    return indices


def protocol_flags(args):
    flags = [
        "--suite",
        "production",
        "--fly-data-root",
        args.fly_data_root,
        "--source-fly",
        args.source_fly,
        "--target-fly",
        args.target_fly,
        "--direction",
        args.direction,
        "--region",
        args.region,
        "--event-source",
        args.event_source,
        "--split-strategy",
        "blocked",
        "--splits",
        "5",
        "--consistency-map",
        "percentile",
        "--percentile",
        str(args.percentile),
        "--alpha",
        str(args.alpha),
        "--solver",
        args.solver,
        "--workloads",
        "consistency",
    ]
    if args.events_dir:
        flags.extend(["--events-dir", args.events_dir])
    return flags


def assign_seed_ranges(base_seed, bootstrap_iters, n_jobs, replica=0):
    start = int(base_seed) + int(replica) * int(bootstrap_iters) * int(n_jobs)
    return [
        {
            "job_index": index,
            "seed": start + index * int(bootstrap_iters),
            "bootstrap_iters": int(bootstrap_iters),
        }
        for index in range(int(n_jobs))
    ]


def output_path(directory, stem, job_index, replica, backend, workers, config_id=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if config_id:
        stem = f"{stem}_{config_id}"
    return directory / f"{stem}_r{replica}_j{job_index}_{backend}_w{workers}.json"


def _staging_path(path):
    path = Path(path)
    return path.with_name(f".{path.stem}.partial{path.suffix}")


def _batch_size(value, default=None):
    value = default if value is None else value
    if value is None or str(value).lower() == "all":
        return value
    return int(value)


def _expected_mapping(args, correlation_dtype):
    options = {
        "regression_type": "Ridge",
        "regression_kwargs": {
            "alpha": float(args.alpha),
            "fit_intercept": True,
            "solver": args.solver,
        },
        "correlation_target_batch_size": _batch_size(
            args.correlation_target_batch_size, 512
        ),
        "correlation_dtype": correlation_dtype,
        "backend": args.backend,
        "percentile": float(args.percentile),
        "identity": False,
    }
    if args.ridge_workspace_bytes is not None:
        options["ridge_workspace_bytes"] = args.ridge_workspace_bytes
    if args.ridge_max_batch_size is not None:
        options["ridge_max_batch_size"] = _batch_size(args.ridge_max_batch_size)
    if args.backend == "cuda":
        options["store_correlations"] = False
    return {"map_type": "percentile", "map_kwargs": options}


def _expected_result(assignment, args):
    input_dtype = args.dtype or "float32"
    mapper_dtype = args.compute_dtype or input_dtype
    correlation_dtype = args.correlation_dtype or mapper_dtype
    execution = getattr(args, "manifest_execution", {})
    expected = {
        "backend_requested": args.backend,
        "backend_expected": execution.get("backend_expected", args.backend),
        "workers": args.workers,
        "seed": assignment["seed"],
        "split_seed": 0,
        "bootstrap_iters": assignment["bootstrap_iters"],
        "percentile": float(args.percentile),
        "alpha": float(args.alpha),
        "solver": args.solver,
        "dtypes": {
            "preparation": input_dtype,
            "mapper_input": mapper_dtype,
            "correlation": correlation_dtype,
            "compute_override_explicit": args.compute_dtype is not None,
            "correlation_override_explicit": args.correlation_dtype is not None,
            "mixed_precision_allowed": False,
        },
        "mapping": _expected_mapping(args, correlation_dtype),
        "fly_data_root": str(Path(args.fly_data_root).resolve()),
        "source_fly": args.source_fly,
        "target_fly": args.target_fly,
        "direction": args.direction,
        "region": args.region,
        "event_source": args.event_source,
        "target_limit": args.target_limit,
        "target_indices": args.target_indices,
        "repository": execution.get("repository") or repository_identity(),
        "environment": execution.get("environment")
        or {"python": platform.python_version()},
        "folds": execution.get("folds"),
        "configuration_manifest_checksum": getattr(
            args, "configuration_manifest_checksum", None
        ),
        "configuration_grid_checksum": getattr(
            args, "configuration_grid_checksum", None
        ),
        "configuration_id": getattr(args, "config_id", None),
        "configuration_execution_checksum": getattr(
            args, "configuration_execution_checksum", None
        ),
    }
    expected["assignment_checksum"] = _canonical_sha256(expected)
    return expected


def build_command(python, benchmark_module, assignment, args):
    output = output_path(
        args.output_dir,
        args.stem,
        assignment["job_index"],
        args.replica,
        args.backend,
        args.workers,
        getattr(args, "config_id", None),
    )
    staging_output = _staging_path(output)
    metrics_output = output.with_suffix(".metrics.npz")
    staging_metrics = _staging_path(metrics_output)
    command = [
        python,
        "-m",
        benchmark_module,
        *protocol_flags(args),
        "--backend",
        args.backend,
        "--workers",
        str(args.workers),
        "--bootstrap-iters",
        str(assignment["bootstrap_iters"]),
        "--seed",
        str(assignment["seed"]),
        "--output",
        str(staging_output),
    ]
    if args.target_indices is not None:
        command.extend(
            ["--target-indices", ",".join(str(index) for index in args.target_indices)]
        )
    elif args.target_limit is not None:
        command.extend(["--target-limit", str(args.target_limit)])
    if args.correlation_target_batch_size is not None:
        command.extend(
            [
                "--correlation-target-batch-size",
                str(args.correlation_target_batch_size),
            ]
        )
    if args.ridge_workspace_bytes is not None:
        command.extend(["--ridge-workspace-bytes", str(args.ridge_workspace_bytes)])
    if args.ridge_max_batch_size is not None:
        command.extend(["--ridge-max-batch-size", str(args.ridge_max_batch_size)])
    if args.dtype is not None:
        command.extend(["--dtype", args.dtype])
    if args.compute_dtype is not None:
        command.extend(["--compute-dtype", args.compute_dtype])
    if args.correlation_dtype is not None:
        command.extend(["--correlation-dtype", args.correlation_dtype])
    if args.warmup:
        command.append("--warmup")
    if args.save_precision:
        command.extend(["--save-precision-artifacts", str(staging_metrics)])
    built = dict(assignment)
    built.update(
        {
            "output": str(output),
            "staging_output": str(staging_output),
            "metrics_output": str(metrics_output) if args.save_precision else None,
            "staging_metrics": str(staging_metrics) if args.save_precision else None,
            "command": command,
            "status": "pending",
        }
    )
    built["expected"] = _expected_result(built, args)
    if getattr(args, "device_ids", None):
        device = args.device_ids[built["job_index"] % len(args.device_ids)]
        built["env"] = {"CUDA_VISIBLE_DEVICES": str(device)}
    return built


def _validate_recorded_inputs(value):
    if isinstance(value, dict):
        if {"path", "size_bytes", "mtime_ns"}.issubset(value):
            path = Path(value["path"])
            if not path.is_file():
                raise FileNotFoundError(f"recorded input does not exist: {path}")
            stat = path.stat()
            if int(value["size_bytes"]) != int(stat.st_size):
                raise ValueError(f"recorded input size changed: {path}")
            if int(value["mtime_ns"]) != int(stat.st_mtime_ns):
                raise ValueError(f"recorded input modification time changed: {path}")
            checksum = value.get("sha256")
            if checksum is not None and checksum != sha256_file(path):
                raise ValueError(f"recorded input checksum changed: {path}")
        for child in value.values():
            _validate_recorded_inputs(child)
    elif isinstance(value, list):
        for child in value:
            _validate_recorded_inputs(child)


def _validate_environment(metadata, expected):
    libraries = metadata.get("libraries", {})
    for name, value in expected.items():
        if name == "python":
            actual = metadata.get("python")
        elif name == "platform":
            actual = metadata.get("platform")
        else:
            actual = libraries.get(name)
        if actual != value:
            raise ValueError(f"environment does not match at {name}")


def _validate_document(document, expected):
    if document.get("suite") != "production":
        raise ValueError("result suite is not production")
    if document.get("workloads") != ["consistency"]:
        raise ValueError("result workloads do not match the campaign")
    if document.get("backend_requested") != expected["backend_requested"]:
        raise ValueError("requested backend does not match the campaign")
    if document.get("backend_effective") != expected["backend_expected"]:
        raise ValueError("effective backend does not match the campaign")
    effective = document.get("backend_effective_by_workload", {}).get("consistency")
    if effective != expected["backend_expected"]:
        raise ValueError("consistency workload used the wrong backend")
    if int(document.get("seed", -1)) != expected["seed"]:
        raise ValueError("result seed does not match the campaign")
    if int(document.get("split_seed", -1)) != expected["split_seed"]:
        raise ValueError("result split seed does not match the campaign")

    config = document.get("config", {})
    expected_config = {
        "bootstrap_iters": expected["bootstrap_iters"],
        "consistency_map": "percentile",
        "percentile": expected["percentile"],
        "split_strategy": "blocked",
        "splits": 5,
    }
    for name, value in expected_config.items():
        if config.get(name) != value:
            raise ValueError(f"result config does not match at {name}")

    requested_dtypes = document.get("dtypes", {}).get("requested")
    if requested_dtypes != expected["dtypes"]:
        raise ValueError("requested dtypes do not match the campaign")
    effective_dtypes = document.get("dtypes", {}).get("effective", {})
    if effective_dtypes.get("preparation") != {
        "source": expected["dtypes"]["preparation"],
        "target": expected["dtypes"]["preparation"],
    }:
        raise ValueError("effective preparation dtype does not match")
    if effective_dtypes.get("mapper_input") != {
        "source": expected["dtypes"]["mapper_input"],
        "target": expected["dtypes"]["mapper_input"],
    }:
        raise ValueError("effective mapper dtype does not match")

    dataset = document.get("dataset", {})
    provenance = dataset.get("provenance")
    if dataset.get("origin") != "brezovec_fly" or not isinstance(provenance, dict):
        raise ValueError("campaign result is not fly data")
    expected_provenance = {
        "source_fly": expected["source_fly"],
        "target_fly": expected["target_fly"],
        "direction": expected["direction"],
        "region": expected["region"],
    }
    for name, value in expected_provenance.items():
        if provenance.get(name) != value:
            raise ValueError(f"result provenance does not match at {name}")
    if str(Path(provenance.get("root", "")).resolve()) != expected["fly_data_root"]:
        raise ValueError("fly data root does not match the campaign")
    for role in ("source_events", "target_events"):
        if provenance.get(role, {}).get("event_source") != expected["event_source"]:
            raise ValueError(f"{role} does not match the campaign")
    selection = provenance.get("target_selection", {})
    if expected["target_indices"] is not None:
        if selection.get("strategy") != "explicit":
            raise ValueError("result did not use explicit target indices")
        if selection.get("indices") != expected["target_indices"]:
            raise ValueError("target indices do not match the campaign")
    elif expected["target_limit"] is not None:
        if selection.get("strategy") != "evenly_spaced":
            raise ValueError("result did not use evenly spaced targets")
        if selection.get("selected_count") != expected["target_limit"]:
            raise ValueError("target limit does not match the campaign")
    elif selection.get("strategy") != "all" or selection.get("indices") is not None:
        raise ValueError("result did not use all targets")
    if dataset.get("source", {}).get("dtype") != expected["dtypes"]["preparation"]:
        raise ValueError("source dtype does not match the campaign")
    if dataset.get("target", {}).get("dtype") != expected["dtypes"]["preparation"]:
        raise ValueError("target dtype does not match the campaign")
    if dataset.get("selected_fold_indices") != [0, 1, 2, 3, 4]:
        raise ValueError("result does not contain all five folds")
    if (
        expected["folds"] is not None
        and dataset.get("train_test_indices") != expected["folds"]
    ):
        raise ValueError("blocked folds do not match the configuration manifest")
    _validate_recorded_inputs(provenance)

    consistency = document.get("benchmarks", {}).get(CONSISTENCY_KEY)
    if not isinstance(consistency, dict):
        raise TypeError("result is missing the consistency workload")
    expected_consistency = {
        "backend_requested": expected["backend_requested"],
        "backend_effective": expected["backend_expected"],
        "bootstrap_iterations": expected["bootstrap_iters"],
        "workers": expected["workers"],
        "metric": "pearsonr",
        "train_test_splits": 5,
        "mapping": expected["mapping"],
        "ridge_parameters": {
            "alpha": expected["alpha"],
            "fit_intercept": True,
            "solver": expected["solver"],
        },
    }
    for name, value in expected_consistency.items():
        if consistency.get(name) != value:
            raise ValueError(f"consistency metadata does not match at {name}")
    workload_dtypes = consistency.get("dtypes", {})
    for name in ("input_source", "input_target"):
        if workload_dtypes.get(name) != expected["dtypes"]["mapper_input"]:
            raise ValueError(f"consistency dtype does not match at {name}")
    if workload_dtypes.get("correlation") != expected["dtypes"]["correlation"]:
        raise ValueError("consistency correlation dtype does not match")
    shape = consistency.get("result_shape")
    if not isinstance(shape, list) or shape[:2] != [expected["bootstrap_iters"], 5]:
        raise ValueError("consistency result shape does not match the campaign")
    if selection.get("selected_count") != shape[2]:
        raise ValueError("target selection count does not match result shape")

    metadata = document.get("metadata", {})
    repository = metadata.get("repository", {})
    for name, value in expected["repository"].items():
        if repository.get(name) != value:
            raise ValueError(f"repository identity does not match at {name}")
    _validate_environment(metadata, expected["environment"])
    return consistency


def _campaign_metadata(expected):
    names = (
        "assignment_checksum",
        "configuration_manifest_checksum",
        "configuration_grid_checksum",
        "configuration_id",
        "configuration_execution_checksum",
    )
    return {name: expected[name] for name in names if expected.get(name) is not None}


def _load_json(path):
    with open(path) as stream:
        document = json.load(stream)
    if not isinstance(document, dict):
        raise TypeError("result JSON must contain an object")
    return document


def validate_completed_assignment(assignment):
    output = Path(assignment["output"])
    document = _load_json(output)
    completion = document.get("completion", {})
    if completion.get("status") != "complete" or not completion.get("resumable"):
        raise ValueError("result is not a resumable completed assignment")
    consistency = _validate_document(document, assignment["expected"])
    if document.get("campaign") != _campaign_metadata(assignment["expected"]):
        raise ValueError("campaign identity does not match the assignment")
    metrics_output = assignment.get("metrics_output")
    if not metrics_output:
        raise ValueError("resumable assignments require a precision artifact")
    record = consistency.get("precision_metrics")
    if not isinstance(record, dict):
        raise TypeError("result is missing its precision artifact record")
    artifact_path = validate_artifact_record(output, record)
    if artifact_path.resolve() != Path(metrics_output).resolve():
        raise ValueError("precision artifact path does not match the campaign")
    arrays = arrays_from_npz(artifact_path)
    if arrays["test"]["r_xy_n_sb"].shape[0] != assignment["bootstrap_iters"]:
        raise ValueError("precision artifact has the wrong bootstrap count")
    return document


def _finalize_assignment(assignment):
    staging_output = Path(assignment["staging_output"])
    document = _load_json(staging_output)
    consistency = _validate_document(document, assignment["expected"])
    metrics_output = assignment.get("metrics_output")
    if metrics_output:
        staging_metrics = Path(assignment["staging_metrics"])
        record = consistency.get("precision_metrics")
        artifact_path = validate_artifact_record(staging_output, record)
        if artifact_path.resolve() != staging_metrics.resolve():
            raise ValueError("staged precision path does not match the assignment")
        arrays = arrays_from_npz(staging_metrics)
        if arrays["test"]["r_xy_n_sb"].shape[0] != assignment["bootstrap_iters"]:
            raise ValueError("staged precision artifact has the wrong bootstrap count")
        atomic_promote(staging_metrics, metrics_output)
        consistency["precision_metrics"] = artifact_record(
            assignment["output"], metrics_output
        )
        consistency.pop("precision_metrics_file", None)
    document["campaign"] = _campaign_metadata(assignment["expected"])
    document["completion"] = {
        "status": "complete",
        "artifact_schema_version": 1,
        "resumable": bool(metrics_output),
    }
    atomic_write_json(assignment["output"], document)
    staging_output.unlink()
    if metrics_output:
        validate_completed_assignment(assignment)


def _clear_staging(assignment):
    for name in ("staging_output", "staging_metrics"):
        value = assignment.get(name)
        if value:
            path = Path(value)
            if path.is_file():
                path.unlink()


def run_local(assignments, max_concurrent):
    pending = []
    running = []
    failures = []
    for assignment in assignments:
        try:
            validate_completed_assignment(assignment)
        except (OSError, KeyError, TypeError, ValueError) as error:
            assignment["resume_validation"] = str(error)
            assignment["status"] = "pending"
            pending.append(assignment)
        else:
            assignment["status"] = "skipped"
            assignment["returncode"] = 0

    while pending or running:
        while pending and len(running) < max_concurrent:
            assignment = pending.pop(0)
            _clear_staging(assignment)
            env = os.environ.copy()
            env.update(assignment.get("env", {}))
            process = subprocess.Popen(assignment["command"], env=env)
            assignment["status"] = "running"
            running.append((assignment, process))
        completed = []
        for assignment, process in running:
            code = process.poll()
            if code is None:
                continue
            assignment["returncode"] = code
            if code == 0:
                try:
                    _finalize_assignment(assignment)
                except (OSError, KeyError, TypeError, ValueError) as error:
                    assignment["finalization_error"] = str(error)
                    assignment["status"] = "failed"
                    failures.append(assignment)
                else:
                    assignment["status"] = "completed"
            else:
                assignment["status"] = "failed"
                failures.append(assignment)
            completed.append((assignment, process))
        if completed:
            running = [item for item in running if item not in completed]
        elif running:
            time.sleep(0.05)
    return failures


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stem", default="anchor")
    parser.add_argument("--backend", choices=("cpu", "cuda"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--bootstrap-iters", type=int)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--shards",
        type=int,
        default=1,
        help="split one comparison into workers=1 processes and merge raw bootstraps",
    )
    parser.add_argument("--devices", help="comma-separated CUDA device ids")
    parser.add_argument("--replica", type=int, default=0)
    parser.add_argument("--max-concurrent", type=int, default=1)
    parser.add_argument(
        "--fly-data-root",
        default=os.environ.get("FLY_DATA_ROOT"),
    )
    parser.add_argument("--events-dir", default=os.environ.get("EVENTS_DIR"))
    parser.add_argument("--source-fly", default=os.environ.get("SOURCE_FLY", "fly_100"))
    parser.add_argument("--target-fly", default=os.environ.get("TARGET_FLY", "fly_099"))
    parser.add_argument("--direction", choices=("left", "right"), default="right")
    parser.add_argument("--region", choices=("IPS", "IPS_L", "IPS_R"), default="IPS")
    parser.add_argument(
        "--event-source", choices=("validated", "historical"), default="historical"
    )
    parser.add_argument("--percentile", type=float, default=99.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--solver", choices=("auto",), default="auto")
    parser.add_argument("--configuration-manifest", type=Path)
    parser.add_argument("--config-id")
    parser.add_argument("--target-limit", type=int)
    parser.add_argument("--target-indices", type=_parse_target_indices)
    parser.add_argument("--correlation-target-batch-size")
    parser.add_argument("--ridge-workspace-bytes", type=int)
    parser.add_argument("--ridge-max-batch-size")
    parser.add_argument("--dtype", choices=("float32", "float64"))
    parser.add_argument("--compute-dtype", choices=("float32", "float64"))
    parser.add_argument("--correlation-dtype", choices=("float32", "float64"))
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--save-precision", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--benchmark-module", default="benchmarks.benchmark_neural_backends"
    )
    return parser


def _apply_configuration_manifest(parser, args):
    if bool(args.configuration_manifest) != bool(args.config_id):
        parser.error("--configuration-manifest and --config-id must be used together")
    if args.configuration_manifest is None:
        args.backend = args.backend or "cpu"
        args.seed = 1000 if args.seed is None else args.seed
        args.bootstrap_iters = (
            200 if args.bootstrap_iters is None else args.bootstrap_iters
        )
        args.manifest_execution = {}
        return
    manifest = load_configuration_manifest(args.configuration_manifest)
    configuration = configuration_by_id(manifest, args.config_id)
    execution = manifest["execution"]
    manifest_backend = execution["backend_requested"]
    if args.backend is not None and args.backend != manifest_backend:
        parser.error("--backend conflicts with the configuration manifest")
    args.backend = manifest_backend
    if execution["backend_expected"] != manifest_backend:
        parser.error("configuration manifest backend expectation is inconsistent")
    if execution["repository"] != repository_identity():
        parser.error("configuration manifest repository identity is stale")
    seed_ranges = execution["seed_ranges"]
    if len(seed_ranges) != 1:
        parser.error("campaigns require one seed range in the configuration manifest")
    seed_range = seed_ranges[0]
    if args.seed is not None and args.seed != seed_range["start"]:
        parser.error("seed range conflicts with the configuration manifest")
    if args.bootstrap_iters is not None and args.bootstrap_iters != seed_range["count"]:
        parser.error("seed range conflicts with the configuration manifest")
    args.seed = seed_range["start"]
    args.bootstrap_iters = seed_range["count"]
    if len(execution["folds"]) != 5:
        parser.error("campaigns require five folds in the configuration manifest")
    target_indices = execution["target_indices"]
    if target_indices == "all":
        if args.target_indices is not None or args.target_limit is not None:
            parser.error("target selection conflicts with the configuration manifest")
        args.target_indices = None
    else:
        if args.target_limit is not None:
            parser.error("--target-limit conflicts with the configuration manifest")
        if args.target_indices is not None and args.target_indices != target_indices:
            parser.error("--target-indices conflicts with the configuration manifest")
        args.target_indices = list(target_indices)
    args.percentile = configuration["percentile"]
    args.alpha = configuration["alpha"]
    args.solver = configuration["solver"]
    for name, manifest_key in (
        ("dtype", "input_dtype"),
        ("compute_dtype", "compute_dtype"),
        ("correlation_dtype", "correlation_dtype"),
    ):
        current = getattr(args, name)
        expected = execution[manifest_key]
        if current is not None and current != expected:
            parser.error(
                "--{} conflicts with the configuration manifest".format(
                    name.replace("_", "-")
                )
            )
        setattr(args, name, expected)
    args.configuration_manifest_checksum = manifest["checksum"]
    args.configuration_grid_checksum = manifest["grid_checksum"]
    args.configuration_execution_checksum = configuration["execution_checksum"]
    args.manifest_execution = execution


def _validate_args(parser, args):
    if not args.fly_data_root:
        parser.error("--fly-data-root or FLY_DATA_ROOT is required")
    for name in ("workers", "bootstrap_iters", "jobs", "shards", "max_concurrent"):
        if getattr(args, name) < 1:
            parser.error("--{} must be positive".format(name.replace("_", "-")))
    if args.backend == "cuda" and args.workers != 1:
        parser.error("CUDA comparisons require --workers 1")
    if args.shards > 1 and args.jobs != 1:
        parser.error("--shards describes one comparison and requires --jobs 1")
    if args.target_limit is not None and args.target_limit < 1:
        parser.error("--target-limit must be positive")
    if args.target_limit is not None and args.target_indices is not None:
        parser.error("--target-limit and --target-indices are mutually exclusive")
    input_dtype = args.dtype or "float32"
    if args.compute_dtype is not None and args.compute_dtype != input_dtype:
        parser.error("campaigns do not allow mixed compute precision")
    mapper_dtype = args.compute_dtype or input_dtype
    if args.correlation_dtype is not None and args.correlation_dtype != mapper_dtype:
        parser.error("campaigns do not allow mixed correlation precision")


def _manifest_projection(manifest_path, args, assignments):
    projected = []
    omitted = {"output", "staging_output", "metrics_output", "staging_metrics"}
    for assignment in assignments:
        item = {key: value for key, value in assignment.items() if key not in omitted}
        for key in omitted:
            value = assignment.get(key)
            item[key] = relative_artifact_path(manifest_path, value) if value else None
        projected.append(item)
    return {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "backend": args.backend,
        "workers": args.workers,
        "replica": args.replica,
        "configuration_manifest": (
            relative_artifact_path(manifest_path, args.configuration_manifest)
            if args.configuration_manifest
            else None
        ),
        "configuration_manifest_checksum": getattr(
            args, "configuration_manifest_checksum", None
        ),
        "configuration_grid_checksum": getattr(
            args, "configuration_grid_checksum", None
        ),
        "configuration_id": args.config_id,
        "assignments": projected,
    }


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    _apply_configuration_manifest(parser, args)
    _validate_args(parser, args)
    args.device_ids = None
    if args.devices:
        args.device_ids = [
            item.strip() for item in args.devices.split(",") if item.strip()
        ]
        if not args.device_ids:
            parser.error("--devices must list at least one device")
    total_bootstraps = args.bootstrap_iters
    n_jobs = args.jobs
    if args.shards > 1:
        if total_bootstraps % args.shards != 0:
            parser.error("--bootstrap-iters must divide evenly by --shards")
        n_jobs = args.shards
        args.bootstrap_iters = total_bootstraps // args.shards
        args.save_precision = True
        if args.max_concurrent == 1:
            args.max_concurrent = args.shards
    assignments = [
        build_command(args.python, args.benchmark_module, assignment, args)
        for assignment in assign_seed_ranges(
            args.seed, args.bootstrap_iters, n_jobs, args.replica
        )
    ]
    campaign_stem = f"{args.stem}_{args.config_id}" if args.config_id else args.stem
    manifest = Path(args.output_dir) / f"{campaign_stem}_r{args.replica}_manifest.json"
    atomic_write_json(manifest, _manifest_projection(manifest, args, assignments))
    if args.dry_run:
        for assignment in assignments:
            print(" ".join(assignment["command"]))
        return 0
    started = time.perf_counter()
    failures = run_local(assignments, args.max_concurrent)
    atomic_write_json(manifest, _manifest_projection(manifest, args, assignments))
    if failures:
        return 1
    if args.shards > 1:
        merged_path = (
            Path(args.output_dir) / f"{campaign_stem}_r{args.replica}_merged.json"
        )
        wall_seconds = (
            None
            if any(assignment["status"] == "skipped" for assignment in assignments)
            else time.perf_counter() - started
        )
        merge_shard_files(
            [assignment["output"] for assignment in assignments],
            merged_path,
            wall_seconds=wall_seconds,
            expected_seed=assignments[0]["seed"],
            expected_bootstrap_iters=total_bootstraps,
        )
        print(merged_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
