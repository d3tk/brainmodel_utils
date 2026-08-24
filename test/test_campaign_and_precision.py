from pathlib import Path

import numpy as np

from benchmarks.campaign import assign_seed_ranges
from benchmarks.precision import compare_precision, jaccard
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
    found, bouts = _event_files("/missing", "fly_087", "right", events_dir=tmp_path)
    assert found == window
    assert found.is_file()


def test_protocol_flags_default_to_ips_cv_pair():
    from benchmarks.campaign import build_parser, protocol_flags

    args = build_parser().parse_args(["--output-dir", "/tmp"])
    flags = protocol_flags(args)
    assert "--source-fly" in flags and "fly_100" in flags
    assert "--target-fly" in flags and "fly_099" in flags
    assert "--events-dir" in flags


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
    identity = resolve_gpu_identity(
        0, inventory=inventory, cuda_visible_devices="2,1"
    )
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
    from benchmarks.campaign import assign_seed_ranges, build_parser, main

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


def test_merge_shards_concatenates_bootstrap_axis_and_masks(tmp_path):
    from benchmarks.merge_shards import compare_merged, merge_raw_arrays, write_arrays

    def shard(seed, values):
        arrays = {
            "train": {"r_xy_n_sb": np.asarray(values, dtype=np.float64)},
            "test": {"r_xy_n_sb": np.asarray(values, dtype=np.float64) + 1.0},
        }
        return {
            "path": tmp_path / "s{}.json".format(seed),
            "document": {
                "backend_requested": "cpu",
                "seed": seed,
                "config": {"percentile": 99.0, "splits": 2},
                "dataset": {
                    "split_strategy": "blocked",
                    "source": {"dtype": "float64", "shape": [4, 6]},
                    "target": {"dtype": "float64", "shape": [4, 6]},
                    "train_test_indices": [{"train": [0, 1], "test": [2, 2]}],
                },
                "benchmarks": {
                    "linregress_consistency": {
                        "mapping": {"map_type": "percentile"},
                        "bootstrap_iterations": values.shape[0],
                        "total_seconds": 1.0,
                    }
                },
            },
            "arrays": arrays,
            "seed": seed,
            "bootstrap_iters": values.shape[0],
            "seconds": 1.0,
        }

    first = np.array([[[1.0, np.nan]], [[2.0, 3.0]]])
    second = np.array([[[4.0, 5.0]], [[np.nan, 6.0]]])
    _, merged = merge_raw_arrays([shard(20, second), shard(10, first)])
    assert merged["test"]["r_xy_n_sb"].shape[0] == 4
    np.testing.assert_allclose(
        merged["train"]["r_xy_n_sb"][:, 0, 0], [1.0, 2.0, 4.0, np.nan], equal_nan=True
    )
    reference = {
        "train": {"r_xy_n_sb": merged["train"]["r_xy_n_sb"].copy()},
        "test": {"r_xy_n_sb": merged["test"]["r_xy_n_sb"].copy()},
    }
    report = compare_merged(merged, reference, atol=1e-12)
    assert report["passed"]
    assert report["identical_finite_masks"]


def test_campaign_shards_dry_run_emits_contiguous_commands(tmp_path, capsys):
    from benchmarks.campaign import main

    code = main(
        [
            "--output-dir",
            str(tmp_path),
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


def test_precision_gates_catch_metric_and_mask_failures():
    reference = {
        "test_r_xy_n_sb": np.array([[[0.4, 0.5]]]),
        "train_r_xy_n_sb": np.array([[[0.6, 0.7]]]),
        "selected_source_indices": np.array([[0, 1, -1], [2, 3, -1]]),
    }
    close = {
        "test_r_xy_n_sb": np.array([[[0.4004, 0.5002]]]),
        "train_r_xy_n_sb": np.array([[[0.6001, 0.7001]]]),
        "selected_source_indices": np.array([[0, 1, -1], [2, 3, -1]]),
    }
    far = {
        "test_r_xy_n_sb": np.array([[[0.41, 0.52]]]),
        "train_r_xy_n_sb": np.array([[[0.61, 0.72]]]),
        "selected_source_indices": np.array([[0, 4, -1], [2, 3, -1]]),
    }
    passed = compare_precision(close, reference)
    failed = compare_precision(far, reference)
    assert passed["gates_passed"]
    assert not failed["gates_passed"]
    assert "canonical_metric" in failed["failures"]
    assert jaccard([0, 1], [0, 1]) == 1.0
    assert jaccard([0, 1], [0, 4]) == 1.0 / 3.0
