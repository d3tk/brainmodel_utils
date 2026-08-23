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
