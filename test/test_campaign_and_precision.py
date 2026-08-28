import copy
import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.campaign import assign_seed_ranges
from benchmarks.telemetry import parse_cuda_visible_devices, resolve_gpu_identity
from brainmodel_utils.neural_mappers.backends import (
    DEFAULT_CUDA_RIDGE_MAX_BATCH_SIZE,
    DEFAULT_CUDA_RIDGE_WORKSPACE_BYTES,
    _grouped_batch_size,
    resolve_correlation_target_batch_size,
    resolve_ridge_max_batch_size,
)


def test_event_files_finds_flat_npz(tmp_path):
    from benchmarks.benchmark_neural_backends import _event_files

    window = tmp_path / "fly_087_right.npz"
    window.write_bytes(b"placeholder")
    found, _bouts = _event_files("/missing", "fly_087", "right", events_dir=tmp_path)
    assert found == window
    assert found.is_file()


def test_protocol_flags_default_to_ips_cv_pair():
    from benchmarks.campaign import build_parser, protocol_flags

    args = build_parser().parse_args(
        ["--output-dir", "/tmp", "--fly-data-root", "/data/fly"]
    )
    flags = protocol_flags(args)
    assert "--source-fly" in flags and "fly_100" in flags
    assert "--target-fly" in flags and "fly_099" in flags
    assert "--events-dir" not in flags


def test_assign_seed_ranges_are_unique_and_non_overlapping():
    first = assign_seed_ranges(1000, 200, 3, replica=0)
    second = assign_seed_ranges(1000, 200, 3, replica=1)
    seeds = [item["seed"] for item in first + second]
    assert seeds == [1000, 1200, 1400, 1600, 1800, 2000]
    assert len(set(seeds)) == 6


def test_resolve_gpu_identity_uses_uuid_under_visible_devices():
    inventory = [
        {"nvml_index": 0, "name": "A", "uuid": "GPU-aaa", "driver_version": "1"},
        {"nvml_index": 1, "name": "B", "uuid": "GPU-bbb", "driver_version": "1"},
        {"nvml_index": 2, "name": "C", "uuid": "GPU-ccc", "driver_version": "1"},
    ]
    identity = resolve_gpu_identity(0, inventory=inventory, cuda_visible_devices="2,1")
    assert parse_cuda_visible_devices("2,1") == [2, 1]
    assert identity["uuid"] == "GPU-ccc"
    assert identity["nvml_index"] == 2
    assert identity["compute_index"] == 0


def test_correlation_and_ridge_batch_helpers_record_effective_sizes():
    assert resolve_correlation_target_batch_size(16636, None) == 512
    assert resolve_correlation_target_batch_size(16636, "all") == 16636
    assert resolve_correlation_target_batch_size(100, 2048) == 100
    default = _grouped_batch_size(
        31, 200, 400, 4, workspace_bytes=DEFAULT_CUDA_RIDGE_WORKSPACE_BYTES
    )
    smaller = _grouped_batch_size(31, 200, 400, 4, workspace_bytes=8 * 1024**2)
    larger = _grouped_batch_size(31, 200, 400, 4, workspace_bytes=2 * 1024**3)
    assert default[0] >= smaller[0]
    assert larger[0] >= default[0]
    assert default[1] > 0
    capped = _grouped_batch_size(31, 200, 4000, 4, workspace_bytes=8 * 1024**3)
    uncapped = _grouped_batch_size(
        31, 200, 4000, 4, workspace_bytes=8 * 1024**3, max_batch_size="all"
    )
    assert capped[0] == DEFAULT_CUDA_RIDGE_MAX_BATCH_SIZE
    assert uncapped[0] == 4000
    assert resolve_ridge_max_batch_size(27118, None) == 1024
    assert resolve_ridge_max_batch_size(27118, "all") == 27118


def test_campaign_shards_split_contiguous_seeds():
    from benchmarks.campaign import assign_seed_ranges, build_parser

    args = build_parser().parse_args(
        [
            "--output-dir",
            "/tmp",
            "--shards",
            "4",
            "--bootstrap-iters",
            "64",
            "--seed",
            "1000",
            "--dry-run",
        ]
    )
    total = args.bootstrap_iters
    assert total % 4 == 0
    per_shard = 16
    ranges = assign_seed_ranges(1000, per_shard, 4)
    assert [item["seed"] for item in ranges] == [1000, 1016, 1032, 1048]
    assert [item["bootstrap_iters"] for item in ranges] == [16, 16, 16, 16]
    assert ranges[-1]["seed"] + per_shard == 1000 + 64


def _canonical_metric_arrays(values):
    from benchmarks.merge_shards import METRIC_NAMES

    values = np.asarray(values, dtype=np.float64)
    return {
        split: {
            name: values + split_index + metric_index / 100.0
            for metric_index, name in enumerate(METRIC_NAMES)
        }
        for split_index, split in enumerate(("train", "test"))
    }


def _shard_record(tmp_path, seed, values):
    values = np.asarray(values, dtype=np.float64)
    bootstrap_iters = values.shape[0]
    document = {
        "schema_version": 3,
        "suite": "production",
        "backend_requested": "cpu",
        "backend_effective": "cpu",
        "backend_effective_by_workload": {"consistency": "cpu"},
        "seed": seed,
        "split_seed": 0,
        "workloads": ["consistency"],
        "config": {
            "bootstrap_iters": bootstrap_iters,
            "consistency_map": "percentile",
            "percentile": 0.0,
            "splits": 1,
            "split_strategy": "blocked",
        },
        "dataset": {
            "origin": "fixture",
            "fixture": "/fixture.npz",
            "provenance": {"fixture_sha256": "abc", "preparation_seconds": 0.1},
            "split_strategy": "blocked",
            "source": {"dtype": "float64", "shape": [4, 6]},
            "target": {"dtype": "float64", "shape": [4, 2]},
            "train_test_indices": [{"train": [0, 1], "test": [2, 3]}],
        },
        "metadata": {
            "repository": {"commit": "abc", "dirty": False},
            "python": "3.11",
            "libraries": {"numpy": np.__version__},
        },
        "benchmarks": {
            "linregress_consistency": {
                "mapping": {
                    "map_type": "percentile",
                    "map_kwargs": {
                        "identity": False,
                        "percentile": 0.0,
                        "regression_kwargs": {"alpha": 1.0, "solver": "auto"},
                    },
                },
                "metric": "pearsonr",
                "train_test_splits": 1,
                "workers": 1,
                "ridge_parameters": {"alpha": 1.0, "solver": "auto"},
                "warmup": False,
                "bootstrap_iterations": bootstrap_iters,
                "total_seconds": 1.0,
                "result_shape": list(values.shape),
            }
        },
        "completion": {"status": "complete", "artifact_schema_version": 1},
    }
    return {
        "path": tmp_path / f"s{seed}.json",
        "document": document,
        "arrays": _canonical_metric_arrays(values),
        "seed": seed,
        "bootstrap_iters": bootstrap_iters,
        "seconds": 1.0,
    }


def _write_shard(tmp_path, seed, values):
    from benchmarks.artifacts import artifact_record, atomic_savez, atomic_write_json

    shard = _shard_record(tmp_path, seed, values)
    metrics_path = shard["path"].with_suffix(".metrics.npz")
    payload = {
        f"{split}_{name}": array
        for split, block in shard["arrays"].items()
        for name, array in block.items()
    }
    atomic_savez(metrics_path, payload)
    consistency = shard["document"]["benchmarks"]["linregress_consistency"]
    consistency["precision_metrics"] = artifact_record(shard["path"], metrics_path)
    atomic_write_json(shard["path"], shard["document"])
    return shard["path"]


def test_merge_shards_concatenates_bootstrap_axis_and_masks(tmp_path):
    from benchmarks.merge_shards import compare_merged, merge_raw_arrays

    first = np.array([[[1.0, np.nan]], [[2.0, 3.0]]])
    second = np.array([[[4.0, 5.0]], [[np.nan, 6.0]]])
    _, merged = merge_raw_arrays(
        [_shard_record(tmp_path, 12, second), _shard_record(tmp_path, 10, first)],
        expected_seed=10,
        expected_bootstrap_iters=4,
    )
    assert merged["test"]["r_xy_n_sb"].shape[0] == 4
    np.testing.assert_allclose(
        merged["train"]["r_xy_n_sb"][:, 0, 0], [1.0, 2.0, 4.0, np.nan], equal_nan=True
    )
    reference = {
        split: {name: values.copy() for name, values in block.items()}
        for split, block in merged.items()
    }
    report = compare_merged(merged, reference, atol=1e-12)
    assert report["passed"]
    assert report["identical_finite_masks"]


def test_merge_comparison_rejects_arrays_without_finite_evidence():
    from benchmarks.merge_shards import compare_merged

    candidate = _canonical_metric_arrays(np.ones((2, 1, 2)))
    reference = copy.deepcopy(candidate)
    candidate["test"]["r_xy_n_sb"][:] = np.nan
    reference["test"]["r_xy_n_sb"][:] = np.nan

    report = compare_merged(candidate, reference)

    assert not report["has_finite_evidence"]
    assert not report["passed"]
    assert report["keys"]["test_r_xy_n_sb"]["max_abs"] is None


@pytest.mark.parametrize(
    ("second_seed", "message"),
    [(11, "overlap"), (13, "not contiguous")],
)
def test_merge_shards_rejects_overlapping_and_gapped_seeds(
    tmp_path, second_seed, message
):
    from benchmarks.merge_shards import merge_raw_arrays

    values = np.ones((2, 1, 2), dtype=np.float64)
    shards = [
        _shard_record(tmp_path, 10, values),
        _shard_record(tmp_path, second_seed, values),
    ]
    with pytest.raises(ValueError, match=message):
        merge_raw_arrays(shards)


def test_merge_shards_rejects_provenance_metric_and_shape_mismatches(tmp_path):
    from benchmarks.merge_shards import merge_raw_arrays

    values = np.ones((2, 1, 2), dtype=np.float64)
    first = _shard_record(tmp_path, 10, values)
    second = _shard_record(tmp_path, 12, values)
    second["document"]["dataset"]["provenance"]["fixture_sha256"] = "different"
    with pytest.raises(ValueError, match="configuration/provenance"):
        merge_raw_arrays([first, second])

    second = _shard_record(tmp_path, 12, values)
    del second["arrays"]["test"]["r_xy"]
    with pytest.raises(ValueError, match="metric keys differ"):
        merge_raw_arrays([first, second])

    second = _shard_record(tmp_path, 12, values)
    second["arrays"]["test"]["r_xy"] = np.ones((2, 1, 3))
    with pytest.raises(ValueError, match="metric shapes differ"):
        merge_raw_arrays([first, second])


def test_merge_shard_files_writes_verified_relative_artifacts_and_totals(tmp_path):
    from benchmarks.artifacts import validate_artifact_record
    from benchmarks.merge_shards import merge_shard_files

    values = np.ones((2, 1, 2), dtype=np.float64)
    shard_a = _write_shard(tmp_path, 10, values)
    shard_b = _write_shard(tmp_path, 12, values + 1)
    output = tmp_path / "merged.json"
    document, _arrays = merge_shard_files(
        [shard_b, shard_a],
        output,
        expected_seed=10,
        expected_bootstrap_iters=4,
    )
    consistency = document["benchmarks"]["linregress_consistency"]
    record = consistency["precision_metrics"]
    assert not Path(record["path"]).is_absolute()
    assert validate_artifact_record(output, record).is_file()
    assert document["config"]["bootstrap_iters"] == 4
    assert consistency["bootstrap_iterations"] == 4
    assert consistency["result_shape"] == [4, 1, 2]
    assert document["shard_merge"]["seed_ranges"] == [
        {"start": 10, "count": 2},
        {"start": 12, "count": 2},
    ]
    assert document["shard_merge"]["total_bootstrap_iterations"] == 4
    assert document["completion"]["status"] == "complete"


def test_merge_writes_one_completion_record(tmp_path, monkeypatch):
    from benchmarks import merge_shards

    values = np.ones((2, 1, 2), dtype=np.float64)
    shard_a = _write_shard(tmp_path, 10, values)
    shard_b = _write_shard(tmp_path, 12, values + 1)
    output = tmp_path / "merged.json"
    writes = []
    original = merge_shards.atomic_write_json

    def record_write(path, document):
        writes.append(Path(path))
        return original(path, document)

    monkeypatch.setattr(merge_shards, "atomic_write_json", record_write)
    merge_shards.merge_shard_files([shard_a, shard_b], output)

    assert writes == [output]


def test_campaign_shards_dry_run_emits_contiguous_commands(tmp_path, capsys):
    from benchmarks.campaign import main

    code = main(
        [
            "--output-dir",
            str(tmp_path),
            "--fly-data-root",
            "/data/fly",
            "--shards",
            "4",
            "--bootstrap-iters",
            "64",
            "--seed",
            "1000",
            "--dry-run",
        ]
    )
    assert code == 0
    output = capsys.readouterr().out
    assert output.count("--bootstrap-iters 16") == 4
    assert "--seed 1000" in output
    assert "--seed 1016" in output
    assert "--seed 1032" in output
    assert "--seed 1048" in output
    assert "--save-precision-artifacts" in output


def test_sharded_smoke_matches_monolithic_float64(tmp_path):
    from benchmarks.benchmark_neural_backends import main as bench_main
    from benchmarks.merge_shards import (
        arrays_from_npz,
        compare_merged,
        merge_shard_files,
    )

    rng = np.random.default_rng(0)
    fixture = tmp_path / "fixture.npz"
    np.savez(
        fixture,
        source=rng.normal(size=(6, 24, 16)),
        target=rng.normal(size=(6, 24, 8)),
    )

    def run(output, seed, boots):
        metrics = Path(output).with_suffix(".metrics.npz")
        code = bench_main(
            [
                "--suite",
                "smoke",
                "--backend",
                "cpu",
                "--dtype",
                "float64",
                "--fixture",
                str(fixture),
                "--workloads",
                "consistency",
                "--workers",
                "1",
                "--bootstrap-iters",
                str(boots),
                "--seed",
                str(seed),
                "--split-seed",
                "0",
                "--output",
                str(output),
                "--save-precision-artifacts",
                str(metrics),
            ]
        )
        assert code == 0
        document = json.loads(Path(output).read_text())
        document["completion"] = {
            "status": "complete",
            "artifact_schema_version": 1,
        }
        from benchmarks.artifacts import atomic_write_json

        atomic_write_json(output, document)
        return metrics

    mono = tmp_path / "mono.json"
    shard_a = tmp_path / "shard0.json"
    shard_b = tmp_path / "shard1.json"
    merged = tmp_path / "merged.json"
    run(mono, 1000, 4)
    run(shard_a, 1000, 2)
    run(shard_b, 1002, 2)
    _, merged_arrays = merge_shard_files([shard_a, shard_b], merged)
    report = compare_merged(
        merged_arrays, arrays_from_npz(mono.with_suffix(".metrics.npz")), atol=1e-12
    )
    assert report["identical_finite_masks"]
    assert report["passed"]


def test_campaign_forwards_ridge_batch_and_precision_flags(tmp_path):
    from benchmarks.campaign import assign_seed_ranges, build_command, build_parser

    args = build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--fly-data-root",
            "/data/fly",
            "--ridge-max-batch-size",
            "all",
            "--warmup",
            "--save-precision",
        ]
    )
    assignment = build_command(
        args.python,
        args.benchmark_module,
        assign_seed_ranges(1000, 2, 1)[0],
        args,
    )
    command = assignment["command"]
    assert "--ridge-max-batch-size" in command
    assert "all" in command
    assert "--warmup" in command
    assert "--save-precision-artifacts" in command


def test_configuration_manifest_is_exact_stable_and_rejects_auto():
    from benchmarks.config_manifest import (
        build_configuration_manifest,
        scientific_grid,
        validate_configuration_manifest,
    )

    arguments = {
        "seed_ranges": [{"start": 1000, "count": 200}],
        "folds": [{"train": [2, 3], "test": [0, 1]}],
        "target_indices": [0, 4, 8],
        "repository": {
            "commit": "abc",
            "dirty": False,
            "source_sha256": "0" * 64,
        },
        "environment": {"python": "3.11", "numpy": "1.26"},
    }
    first = build_configuration_manifest(**arguments)
    second = build_configuration_manifest(**arguments)
    assert first == second
    assert len(first["configurations"]) == 16
    assert [
        (cell["percentile"], cell["alpha"]) for cell in first["configurations"]
    ] == [
        (percentile, alpha)
        for percentile in (0, 50, 75, 100)
        for alpha in (0.01, 1.0, 100.0, 10000.0)
    ]
    assert all(cell["identity"] is False for cell in scientific_grid())
    assert all(cell["solver"] == "auto" for cell in scientific_grid())
    assert validate_configuration_manifest(first) is first
    with pytest.raises(ValueError, match="backend=auto"):
        build_configuration_manifest(**arguments, backend_requested="auto")

    tampered = copy.deepcopy(first)
    tampered["configurations"][0]["percentile"] = 25
    with pytest.raises(ValueError, match="checksum"):
        validate_configuration_manifest(tampered)


def test_campaign_consumes_a_named_manifest_cell(tmp_path, capsys):
    from benchmarks.artifacts import atomic_write_json
    from benchmarks.campaign import main
    from benchmarks.config_manifest import (
        blocked_folds,
        build_configuration_manifest,
        repository_identity,
    )

    configuration_manifest = tmp_path / "configuration.json"
    manifest = build_configuration_manifest(
        seed_ranges=[{"start": 1000, "count": 200}],
        folds=blocked_folds(39, 5),
        target_indices="all",
        repository=repository_identity(),
        environment={"python": "test"},
    )
    atomic_write_json(configuration_manifest, manifest)
    assert (
        main(
            [
                "--output-dir",
                str(tmp_path),
                "--fly-data-root",
                "/data/fly",
                "--configuration-manifest",
                str(configuration_manifest),
                "--config-id",
                "p000_a0p01",
                "--dry-run",
            ]
        )
        == 0
    )
    command = capsys.readouterr().out
    assert "--backend cuda" in command
    assert "--dtype float64" in command
    assert "--compute-dtype float64" in command
    assert "--percentile 0" in command
    assert "--alpha 0.01" in command
    assert "--solver auto" in command
    campaign_manifest = json.loads(
        (tmp_path / "anchor_p000_a0p01_r0_manifest.json").read_text()
    )
    assert campaign_manifest["configuration_manifest_checksum"] == manifest["checksum"]
    assert campaign_manifest["configuration_id"] == "p000_a0p01"


def test_campaign_forwards_exact_manifest_target_ids(tmp_path, capsys):
    from benchmarks.artifacts import atomic_write_json
    from benchmarks.campaign import main
    from benchmarks.config_manifest import (
        blocked_folds,
        build_configuration_manifest,
        repository_identity,
    )

    path = tmp_path / "configuration.json"
    manifest = build_configuration_manifest(
        seed_ranges=[{"start": 1000, "count": 200}],
        folds=blocked_folds(39, 5),
        target_indices=[0, 4, 8],
        repository=repository_identity(),
        environment={"python": "test"},
    )
    atomic_write_json(path, manifest)

    assert (
        main(
            [
                "--output-dir",
                str(tmp_path),
                "--fly-data-root",
                "/data/fly",
                "--configuration-manifest",
                str(path),
                "--config-id",
                "p100_a1",
                "--dry-run",
            ]
        )
        == 0
    )
    command = capsys.readouterr().out
    assert "--target-indices 0,4,8" in command
    assert "--target-limit" not in command


def test_artifact_helpers_are_atomic_relative_and_checksum_verified(tmp_path):
    from benchmarks.artifacts import (
        artifact_record,
        atomic_savez,
        atomic_write_json,
        validate_artifact_record,
    )

    result_path = tmp_path / "result.json"
    metrics_path = tmp_path / "result.metrics.npz"
    atomic_savez(metrics_path, {"value": np.arange(4)})
    record = artifact_record(result_path, metrics_path)
    assert record["path"] == "result.metrics.npz"
    assert validate_artifact_record(result_path, record) == metrics_path

    atomic_write_json(result_path, {"state": "complete"})
    with pytest.raises(ValueError):
        atomic_write_json(result_path, {"invalid": float("nan")})
    assert json.loads(result_path.read_text()) == {"state": "complete"}

    metrics_path.write_bytes(metrics_path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="size mismatch"):
        validate_artifact_record(result_path, record)


def _campaign_assignment(tmp_path, bootstrap_iters=2, job_index=0, seed=1000):
    from benchmarks.campaign import build_command, build_parser

    args = build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--fly-data-root",
            "/data/fly",
            "--backend",
            "cpu",
            "--dtype",
            "float64",
            "--bootstrap-iters",
            str(bootstrap_iters),
            "--percentile",
            "0",
            "--alpha",
            "1",
            "--save-precision",
        ]
    )
    args.device_ids = None
    return build_command(
        args.python,
        args.benchmark_module,
        {
            "job_index": job_index,
            "seed": seed,
            "bootstrap_iters": bootstrap_iters,
        },
        args,
    )


def _campaign_document(assignment):
    expected = assignment["expected"]
    shape = [assignment["bootstrap_iters"], 5, 2]
    document = _shard_record(
        Path(assignment["output"]).parent,
        assignment["seed"],
        np.ones(shape),
    )["document"]
    document["config"].update(
        {
            "bootstrap_iters": assignment["bootstrap_iters"],
            "percentile": expected["percentile"],
            "splits": 5,
            "split_strategy": "blocked",
        }
    )
    document["workloads"] = ["consistency"]
    document["backend_requested"] = expected["backend_requested"]
    document["backend_effective"] = expected["backend_expected"]
    document["backend_effective_by_workload"] = {
        "consistency": expected["backend_expected"]
    }
    document["dtypes"] = {
        "requested": dict(expected["dtypes"]),
        "effective": {
            "preparation": {
                "source": expected["dtypes"]["preparation"],
                "target": expected["dtypes"]["preparation"],
            },
            "mapper_input": {
                "source": expected["dtypes"]["mapper_input"],
                "target": expected["dtypes"]["mapper_input"],
            },
        },
    }
    folds = [
        {
            "fold_index": index,
            "train": [item for item in range(5) if item != index],
            "test": [index],
        }
        for index in range(5)
    ]
    document["dataset"].update(
        {
            "origin": "brezovec_fly",
            "fixture": None,
            "source": {
                "dtype": expected["dtypes"]["preparation"],
                "shape": [5, 6],
            },
            "target": {
                "dtype": expected["dtypes"]["preparation"],
                "shape": [4, 5, 2],
            },
            "split_strategy": "blocked",
            "selected_fold_indices": [0, 1, 2, 3, 4],
            "train_test_indices": expected["folds"] or folds,
        }
    )
    document["dataset"]["provenance"] = {
        "root": expected["fly_data_root"],
        "source_fly": expected["source_fly"],
        "target_fly": expected["target_fly"],
        "direction": expected["direction"],
        "region": expected["region"],
        "source_events": {"event_source": expected["event_source"]},
        "target_events": {"event_source": expected["event_source"]},
        "target_selection": {
            "strategy": "all",
            "selected_count": 2,
            "indices": None,
        },
    }
    consistency = document["benchmarks"]["linregress_consistency"]
    consistency.update(
        {
            "bootstrap_iterations": assignment["bootstrap_iters"],
            "workers": expected["workers"],
            "train_test_splits": 5,
            "result_shape": shape,
            "backend_requested": expected["backend_requested"],
            "backend_effective": expected["backend_expected"],
            "ridge_parameters": {
                "alpha": expected["alpha"],
                "fit_intercept": True,
                "solver": expected["solver"],
            },
            "dtypes": {
                "input_source": expected["dtypes"]["mapper_input"],
                "input_target": expected["dtypes"]["mapper_input"],
                "correlation": expected["dtypes"]["correlation"],
            },
        }
    )
    consistency["mapping"] = copy.deepcopy(expected["mapping"])
    document["metadata"]["repository"] = dict(expected["repository"])
    document["metadata"]["python"] = expected["environment"]["python"]
    document["metadata"]["platform"] = expected["environment"].get("platform")
    document["metadata"]["libraries"] = {
        name: value
        for name, value in expected["environment"].items()
        if name not in {"python", "platform"}
    }
    return document


def _write_completed_assignment(assignment):
    from benchmarks import campaign
    from benchmarks.artifacts import artifact_record, atomic_savez, atomic_write_json

    shape = [assignment["bootstrap_iters"], 5, 2]
    arrays = _canonical_metric_arrays(np.ones(shape))
    payload = {
        f"{split}_{name}": values
        for split, block in arrays.items()
        for name, values in block.items()
    }
    atomic_savez(assignment["metrics_output"], payload)
    document = _campaign_document(assignment)
    consistency = document["benchmarks"]["linregress_consistency"]
    consistency["precision_metrics"] = artifact_record(
        assignment["output"], assignment["metrics_output"]
    )
    document["completion"] = {
        "status": "complete",
        "artifact_schema_version": 1,
        "resumable": True,
    }
    document["campaign"] = campaign._campaign_metadata(assignment["expected"])
    atomic_write_json(assignment["output"], document)


def test_campaign_skips_only_a_valid_completed_pair(tmp_path, monkeypatch):
    from benchmarks import campaign

    assignment = _campaign_assignment(tmp_path)
    _write_completed_assignment(assignment)

    def should_not_run(*_args, **_kwargs):
        raise AssertionError("valid completed work should be skipped")

    monkeypatch.setattr(campaign.subprocess, "Popen", should_not_run)
    assert campaign.run_local([assignment], 1) == []
    assert assignment["status"] == "skipped"


def test_campaign_promotes_metrics_before_committing_completed_json(tmp_path):
    from benchmarks import campaign
    from benchmarks.artifacts import artifact_record, atomic_savez, atomic_write_json

    assignment = _campaign_assignment(tmp_path)
    shape = [assignment["bootstrap_iters"], 5, 2]
    payload = {
        f"{split}_{name}": values
        for split, block in _canonical_metric_arrays(np.ones(shape)).items()
        for name, values in block.items()
    }
    atomic_savez(assignment["staging_metrics"], payload)
    document = _campaign_document(assignment)
    consistency = document["benchmarks"]["linregress_consistency"]
    consistency["precision_metrics"] = artifact_record(
        assignment["staging_output"], assignment["staging_metrics"]
    )
    atomic_write_json(assignment["staging_output"], document)
    campaign._finalize_assignment(assignment)

    assert not Path(assignment["staging_output"]).exists()
    assert not Path(assignment["staging_metrics"]).exists()
    assert Path(assignment["metrics_output"]).is_file()
    completed = campaign.validate_completed_assignment(assignment)
    assert completed["completion"]["status"] == "complete"


def test_campaign_reruns_a_corrupt_completed_pair(tmp_path, monkeypatch):
    from benchmarks import campaign

    assignment = _campaign_assignment(tmp_path)
    _write_completed_assignment(assignment)
    metrics_path = Path(assignment["metrics_output"])
    metrics_path.write_bytes(metrics_path.read_bytes() + b"corrupt")
    launches = []

    class FailedProcess:
        def __init__(self, command, env):
            launches.append((command, env))

        def poll(self):
            return 1

    monkeypatch.setattr(campaign.subprocess, "Popen", FailedProcess)
    failures = campaign.run_local([assignment], 1)
    assert launches
    assert failures == [assignment]
    assert assignment["status"] == "failed"


@pytest.mark.parametrize("failure", ("truncated_json", "missing_npz", "stale_config"))
def test_campaign_rejects_incomplete_or_stale_completed_work(tmp_path, failure):
    from benchmarks import campaign

    assignment = _campaign_assignment(tmp_path)
    _write_completed_assignment(assignment)
    if failure == "truncated_json":
        Path(assignment["output"]).write_text("{")
    elif failure == "missing_npz":
        Path(assignment["metrics_output"]).unlink()
    else:
        document = json.loads(Path(assignment["output"]).read_text())
        document["config"]["percentile"] = 50.0
        from benchmarks.artifacts import atomic_write_json

        atomic_write_json(assignment["output"], document)

    with pytest.raises((OSError, ValueError)):
        campaign.validate_completed_assignment(assignment)


@pytest.mark.parametrize("change", ("missing", "additional"))
def test_campaign_rejects_noncanonical_metric_keys(tmp_path, change):
    from benchmarks import campaign
    from benchmarks.artifacts import artifact_record, atomic_savez, atomic_write_json

    assignment = _campaign_assignment(tmp_path)
    _write_completed_assignment(assignment)
    with np.load(assignment["metrics_output"], allow_pickle=False) as loaded:
        payload = {name: np.asarray(loaded[name]) for name in loaded.files}
    if change == "missing":
        payload.pop("test_r_xy_n_sb")
    else:
        payload["unexpected"] = np.ones(1)
    atomic_savez(assignment["metrics_output"], payload)
    document = json.loads(Path(assignment["output"]).read_text())
    document["benchmarks"]["linregress_consistency"]["precision_metrics"] = (
        artifact_record(assignment["output"], assignment["metrics_output"])
    )
    atomic_write_json(assignment["output"], document)

    with pytest.raises(ValueError, match="keys differ"):
        campaign.validate_completed_assignment(assignment)


def test_campaign_validates_recorded_input_identity(tmp_path):
    from benchmarks import campaign
    from benchmarks.artifacts import atomic_write_json, sha256_file

    assignment = _campaign_assignment(tmp_path)
    _write_completed_assignment(assignment)
    input_path = tmp_path / "events.npz"
    input_path.write_bytes(b"original")
    stat = input_path.stat()
    document = json.loads(Path(assignment["output"]).read_text())
    document["dataset"]["provenance"]["source_events"]["window_file"] = {
        "path": str(input_path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(input_path),
    }
    atomic_write_json(assignment["output"], document)
    assert campaign.validate_completed_assignment(assignment)

    input_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="input .* changed"):
        campaign.validate_completed_assignment(assignment)


def test_campaign_reruns_only_the_invalid_assignment(tmp_path, monkeypatch):
    from benchmarks import campaign
    from benchmarks.artifacts import atomic_write_json

    valid = _campaign_assignment(tmp_path, job_index=0, seed=1000)
    stale = _campaign_assignment(tmp_path, job_index=1, seed=1002)
    _write_completed_assignment(valid)
    _write_completed_assignment(stale)
    document = json.loads(Path(stale["output"]).read_text())
    document["dataset"]["provenance"]["target_fly"] = "fly_stale"
    atomic_write_json(stale["output"], document)
    launches = []

    class FailedProcess:
        def __init__(self, command, env):
            launches.append(command)

        def poll(self):
            return 1

    monkeypatch.setattr(campaign.subprocess, "Popen", FailedProcess)
    assert campaign.run_local([valid, stale], 2) == [stale]
    assert valid["status"] == "skipped"
    assert len(launches) == 1
    assert "--seed" in launches[0] and "1002" in launches[0]


def test_campaign_polls_all_children_without_head_of_line_waiting(monkeypatch):
    from benchmarks import campaign

    assignments = [
        {"command": ["slow"], "status": "pending"},
        {"command": ["fast"], "status": "pending"},
    ]
    finalized = []

    class Process:
        def __init__(self, command, env):
            self.name = command[0]
            self.polls = 0

        def poll(self):
            self.polls += 1
            if self.name == "fast" or self.polls >= 3:
                return 0
            return None

    monkeypatch.setattr(
        campaign,
        "validate_completed_assignment",
        lambda _: (_ for _ in ()).throw(FileNotFoundError()),
    )
    monkeypatch.setattr(
        campaign,
        "_finalize_assignment",
        lambda item: finalized.append(item["command"][0]),
    )
    monkeypatch.setattr(campaign.subprocess, "Popen", Process)
    assert campaign.run_local(assignments, 2) == []
    assert finalized == ["fast", "slow"]


def test_campaign_rejects_auto_backend(tmp_path):
    from benchmarks.campaign import main

    with pytest.raises(SystemExit):
        main(
            [
                "--output-dir",
                str(tmp_path),
                "--backend",
                "auto",
                "--dry-run",
            ]
        )
