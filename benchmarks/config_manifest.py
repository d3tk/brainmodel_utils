"""Generate the frozen percentile-by-alpha benchmark grid."""

import argparse
import copy
import hashlib
import json
import platform
import subprocess
from pathlib import Path

from benchmarks.artifacts import atomic_write_json

MANIFEST_SCHEMA_VERSION = 1
PERCENTILES = (0, 50, 75, 100)
ALPHAS = (0.01, 1.0, 100.0, 10000.0)


def _canonical_json(document):
    return json.dumps(
        document, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def document_checksum(document):
    payload = copy.deepcopy(document)
    payload.pop("checksum", None)
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _alpha_token(alpha):
    value = f"{alpha:g}"
    return value.replace(".", "p").replace("-", "m")


def configuration_id(percentile, alpha):
    return f"p{int(percentile):03d}_a{_alpha_token(alpha)}"


def scientific_grid():
    return [
        {
            "id": configuration_id(percentile, alpha),
            "percentile": int(percentile),
            "alpha": float(alpha),
            "identity": False,
            "solver": "auto",
        }
        for percentile in PERCENTILES
        for alpha in ALPHAS
    ]


def scientific_grid_checksum():
    return hashlib.sha256(_canonical_json(scientific_grid())).hexdigest()


def blocked_folds(num_stimuli, num_folds):
    if num_folds < 2 or num_stimuli < num_folds:
        raise ValueError("blocked folds require at least two non-empty folds")
    quotient, remainder = divmod(int(num_stimuli), int(num_folds))
    folds = []
    start = 0
    indices = list(range(int(num_stimuli)))
    for fold_index in range(int(num_folds)):
        stop = start + quotient + (1 if fold_index < remainder else 0)
        test = indices[start:stop]
        train = indices[:start] + indices[stop:]
        folds.append({"fold_index": fold_index, "train": train, "test": test})
        start = stop
    return folds


def repository_identity(root=None):
    root = Path(root or Path(__file__).resolve().parents[1])

    def git_output(*arguments):
        try:
            completed = subprocess.run(  # noqa: UP022 -- Python 3.6 support
                ["git", "-C", str(root)] + list(arguments),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,  # noqa: UP021 -- Python 3.6 support
                check=False,
            )
        except OSError:
            return None
        return completed.stdout.strip() if completed.returncode == 0 else None

    commit = git_output("rev-parse", "HEAD")
    status = git_output("status", "--porcelain")
    listed = git_output(
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "--",
        "brainmodel_utils",
        "benchmarks",
        "setup.py",
        "pytest.ini",
    )
    paths = [] if listed is None else listed.splitlines()
    source_paths = [
        path
        for path in paths
        if path.endswith(".py") or path in {"setup.py", "pytest.ini"}
    ]
    source_digest = hashlib.sha256()
    for relative in sorted(source_paths):
        path = root / relative
        if not path.is_file():
            continue
        source_digest.update(relative.encode("utf-8"))
        source_digest.update(b"\0")
        source_digest.update(path.read_bytes())
        source_digest.update(b"\0")
    return {
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
        "source_sha256": source_digest.hexdigest(),
    }


def _validate_seed_ranges(seed_ranges):
    if not isinstance(seed_ranges, (list, tuple)) or not seed_ranges:
        raise ValueError("configuration manifests require at least one seed range")
    normalized = []
    for seed_range in seed_ranges:
        if not isinstance(seed_range, dict):
            raise TypeError("seed ranges must be objects")
        start = seed_range.get("start", seed_range.get("seed"))
        count = seed_range.get("count", seed_range.get("bootstrap_iters"))
        if not isinstance(start, int) or not isinstance(count, int) or count < 1:
            raise ValueError("seed ranges require integer start and positive count")
        normalized.append({"start": start, "count": count})
    normalized.sort(key=lambda item: item["start"])
    for previous, current in zip(  # noqa: RUF007 -- Python 3.6 support
        normalized, normalized[1:]
    ):
        if current["start"] < previous["start"] + previous["count"]:
            raise ValueError("configuration manifest seed ranges overlap")
    return normalized


def _validate_folds(folds):
    if not isinstance(folds, (list, tuple)) or not folds:
        raise ValueError("configuration manifests require at least one fold")
    normalized = []
    universe = None
    for fold_index, fold in enumerate(folds):
        if not isinstance(fold, dict):
            raise TypeError("folds must be objects")
        if set(fold).difference({"fold_index", "train", "test"}):
            raise ValueError("folds contain unexpected fields")
        recorded_index = fold.get("fold_index", fold_index)
        if recorded_index != fold_index:
            raise ValueError("fold indices must be contiguous and ordered")
        train = fold.get("train")
        test = fold.get("test")
        if (
            not isinstance(train, list)
            or not isinstance(test, list)
            or not train
            or not test
        ):
            raise ValueError("each fold requires non-empty train and test indices")
        if any(not isinstance(index, int) or index < 0 for index in train + test):
            raise ValueError("fold indices must be non-negative integers")
        if train != sorted(set(train)) or test != sorted(set(test)):
            raise ValueError("fold train and test indices must be sorted and unique")
        if set(train).intersection(test):
            raise ValueError("fold train and test indices overlap")
        current_universe = sorted(train + test)
        if universe is None:
            universe = current_universe
        elif current_universe != universe:
            raise ValueError("all folds must cover the same stimulus indices")
        normalized.append(
            {"fold_index": fold_index, "train": list(train), "test": list(test)}
        )
    return normalized


def _validate_target_indices(target_indices):
    if target_indices == "all":
        return "all"
    if not isinstance(target_indices, list) or not target_indices:
        raise ValueError("target_indices must be 'all' or a non-empty list")
    if any(not isinstance(index, int) or index < 0 for index in target_indices):
        raise ValueError("target indices must be non-negative integers")
    if target_indices != sorted(set(target_indices)):
        raise ValueError("target indices must be sorted and unique")
    return list(target_indices)


def _validate_execution(execution):
    expected_keys = {
        "seed_ranges",
        "folds",
        "target_indices",
        "backend_requested",
        "backend_expected",
        "input_dtype",
        "compute_dtype",
        "correlation_dtype",
        "repository",
        "environment",
    }
    if set(execution) != expected_keys:
        missing = sorted(expected_keys.difference(execution))
        extra = sorted(set(execution).difference(expected_keys))
        raise ValueError(
            f"configuration execution fields differ: missing={missing}, extra={extra}"
        )
    if execution["seed_ranges"] != _validate_seed_ranges(execution["seed_ranges"]):
        raise ValueError("seed ranges are not canonical")
    if execution["folds"] != _validate_folds(execution["folds"]):
        raise ValueError("folds are not canonical")
    if execution["target_indices"] != _validate_target_indices(
        execution["target_indices"]
    ):
        raise ValueError("target indices are not canonical")
    requested = execution["backend_requested"]
    expected = execution["backend_expected"]
    if requested not in {"cpu", "cuda"} or expected not in {"cpu", "cuda"}:
        raise ValueError("configuration manifests cannot request backend=auto")
    if requested != expected:
        raise ValueError("requested and expected backends must match")
    for key in ("input_dtype", "compute_dtype", "correlation_dtype"):
        if execution[key] not in {"float32", "float64"}:
            raise ValueError(f"invalid configuration dtype at {key}")
    repository = execution["repository"]
    if (
        not isinstance(repository, dict)
        or set(repository) != {"commit", "dirty", "source_sha256"}
        or not isinstance(repository["commit"], str)
        or not repository["commit"]
        or not isinstance(repository["dirty"], bool)
        or not isinstance(repository["source_sha256"], str)
        or len(repository["source_sha256"]) != 64
    ):
        raise ValueError(
            "repository identity requires commit, dirty state, and source checksum"
        )
    environment = execution["environment"]
    if not isinstance(environment, dict) or not environment:
        raise ValueError("environment identity must be a non-empty object")
    _canonical_json(environment)


def build_configuration_manifest(
    seed_ranges,
    folds,
    target_indices,
    backend_requested="cuda",
    backend_expected="cuda",
    input_dtype="float64",
    compute_dtype="float64",
    correlation_dtype=None,
    repository=None,
    environment=None,
):
    if backend_requested not in {"cpu", "cuda"}:
        raise ValueError("configuration manifests cannot request backend=auto")
    if backend_expected not in {"cpu", "cuda"}:
        raise ValueError("configuration manifests cannot expect backend=auto")
    correlation_dtype = correlation_dtype or compute_dtype
    execution = {
        "seed_ranges": _validate_seed_ranges(seed_ranges),
        "folds": _validate_folds(folds),
        "target_indices": _validate_target_indices(target_indices),
        "backend_requested": backend_requested,
        "backend_expected": backend_expected,
        "input_dtype": input_dtype,
        "compute_dtype": compute_dtype,
        "correlation_dtype": correlation_dtype,
        "repository": repository or repository_identity(),
        "environment": environment
        or {
            "python": platform.python_version(),
        },
    }
    _validate_execution(execution)
    configurations = scientific_grid()
    for configuration in configurations:
        configuration["execution_checksum"] = hashlib.sha256(
            _canonical_json({"configuration": configuration, "execution": execution})
        ).hexdigest()
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "grid": "percentile_alpha_v1",
        "grid_checksum": scientific_grid_checksum(),
        "execution": execution,
        "configurations": configurations,
    }
    manifest["checksum"] = document_checksum(manifest)
    return manifest


def validate_configuration_manifest(manifest):
    if not isinstance(manifest, dict):
        raise TypeError("configuration manifest must be an object")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported configuration manifest schema")
    expected_manifest_keys = {
        "schema_version",
        "grid",
        "grid_checksum",
        "execution",
        "configurations",
        "checksum",
    }
    if set(manifest) != expected_manifest_keys:
        raise ValueError("configuration manifest contains unexpected fields")
    if manifest.get("grid") != "percentile_alpha_v1":
        raise ValueError("unknown configuration grid")
    if manifest.get("grid_checksum") != scientific_grid_checksum():
        raise ValueError("configuration scientific-grid checksum mismatch")
    if manifest.get("checksum") != document_checksum(manifest):
        raise ValueError("configuration manifest checksum mismatch")
    expected = scientific_grid()
    actual = manifest.get("configurations")
    if not isinstance(actual, list) or len(actual) != len(expected):
        raise ValueError("configuration manifest must contain the exact 16-cell grid")
    execution = manifest.get("execution")
    if not isinstance(execution, dict):
        raise TypeError("configuration manifest is missing execution settings")
    _validate_execution(execution)
    for actual_cell, expected_cell in zip(actual, expected):
        if set(actual_cell) != set(expected_cell).union({"execution_checksum"}):
            raise ValueError("configuration grid contains unexpected fields")
        comparable = {key: actual_cell.get(key) for key in expected_cell}
        if comparable != expected_cell:
            raise ValueError("configuration grid is not canonical")
        expected_checksum = hashlib.sha256(
            _canonical_json({"configuration": expected_cell, "execution": execution})
        ).hexdigest()
        if actual_cell.get("execution_checksum") != expected_checksum:
            raise ValueError(
                "configuration execution checksum mismatch: {}".format(
                    expected_cell["id"]
                )
            )
    return manifest


def load_configuration_manifest(path):
    with open(path) as stream:
        manifest = json.load(stream)
    return validate_configuration_manifest(manifest)


def configuration_by_id(manifest, config_id):
    validate_configuration_manifest(manifest)
    matches = [
        configuration
        for configuration in manifest["configurations"]
        if configuration["id"] == config_id
    ]
    if len(matches) != 1:
        raise ValueError(f"unknown configuration id: {config_id}")
    return matches[0]


def _parse_target_indices(value):
    if value == "all":
        return "all"
    return [int(item) for item in value.split(",") if item]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--bootstrap-iters", type=int, default=200)
    parser.add_argument("--stimuli", type=int, default=39)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--target-indices", default="all")
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    manifest = build_configuration_manifest(
        seed_ranges=[{"seed": args.seed, "bootstrap_iters": args.bootstrap_iters}],
        folds=blocked_folds(args.stimuli, args.folds),
        target_indices=_parse_target_indices(args.target_indices),
        backend_requested=args.backend,
        backend_expected=args.backend,
        input_dtype=args.dtype,
        compute_dtype=args.dtype,
    )
    atomic_write_json(args.output, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
