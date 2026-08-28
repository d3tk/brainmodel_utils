import json
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import benchmarks.benchmark_neural_backends as benchmark
from benchmarks.artifacts import validate_artifact_record
from benchmarks.benchmark_neural_backends import (
    NEURAL_SERIES_PATH,
    build_parser,
    load_fly_data,
    make_train_test_splits,
    run_benchmarks,
)
from benchmarks.precision import (
    CONSISTENCY_METRICS,
    load_precision_diagnostic,
    validate_precision_artifact,
)


def _write_nwb(root, fly_id, timestamps):
    number = fly_id.split("_")[-1]
    directory = root / f"sub-fly-{number}"
    directory.mkdir(parents=True)
    path = directory / f"sub-fly-{number}_ses-test_behavior+image+ophys.nwb"
    with h5py.File(str(path), "w") as nwb:
        series = nwb.create_group(NEURAL_SERIES_PATH)
        series.create_dataset("data", data=np.zeros((12, 1), dtype=np.float32))
        if timestamps is None:
            starting = series.create_dataset("starting_time", data=0.0)
            starting.attrs["rate"] = 1.0
        else:
            series.create_dataset("timestamps", data=timestamps)
    return path


def _write_event_files(root, fly_id, validated_times, historical_times):
    pilot = root / "inter_animal_consistency" / "two_fly_pilot"
    windows = pilot / "event_tensors" / "short"
    events = pilot / "events"
    windows.mkdir(parents=True, exist_ok=True)
    events.mkdir(parents=True, exist_ok=True)
    relative_times = np.linspace(-1.0, 1.5, 6)
    np.savez(
        str(windows / f"{fly_id}_right.npz"),
        event_times=np.asarray(validated_times),
        relative_times=relative_times,
    )
    rows = ["direction,event_time_seconds"]
    rows.extend(f"right,{value}" for value in historical_times)
    rows.append("left,4.0")
    (events / f"{fly_id}_turn_bouts.csv").write_text("\n".join(rows) + "\n")


def _make_fly_fixture(tmp_path):
    root = Path(tmp_path) / "000727"
    root.mkdir()
    (root / "dandiset.yaml").write_text(
        "identifier: DANDI:000727\n"
        "version: 0.240106.0043\n"
        "citation: https://doi.org/10.48324/dandi.000727/0.240106.0043\n"
    )
    source_fly = "fly_001"
    target_fly = "fly_002"
    _write_nwb(root, source_fly, None)
    _write_nwb(root, target_fly, np.arange(12, dtype=np.float64))

    sample = np.arange(12, dtype=np.float32)
    with h5py.File(str(root / "roi_activity.h5"), "w") as roi_file:
        source = roi_file.create_group(source_fly)
        source.create_dataset("IPS_L", data=np.column_stack((sample, sample**2)))
        source.create_dataset("IPS_R", data=np.sin(sample)[:, None])
        target = roi_file.create_group(target_fly)
        target.create_dataset("IPS_L", data=np.column_stack((sample**2, sample)))
        target.create_dataset("IPS_R", data=np.cos(sample)[:, None])

    _write_event_files(root, source_fly, [0, 3, 4, 5], [2, 3, 4, 5, 6])
    _write_event_files(root, target_fly, [2.5, 3.5, 4.5, 5.5], [2.5, 3.5, 4.5, 5.5])
    return root, source_fly, target_fly


def _learned_fit_command(output, learned_fit, metrics=None):
    command = [
        "--suite",
        "smoke",
        "--backend",
        "cpu",
        "--dtype",
        "float64",
        "--workloads",
        "consistency",
        "--consistency-map",
        "percentile",
        "--percentile",
        "50",
        "--stimuli",
        "20",
        "--sources",
        "8",
        "--targets",
        "3",
        "--trials",
        "4",
        "--latent-dim",
        "4",
        "--signal-sources",
        "2",
        "--bootstrap-iters",
        "1",
        "--splits",
        "2",
        "--monitor-interval",
        "0.01",
        "--output",
        str(output),
        "--save-learned-fit",
        str(learned_fit),
    ]
    if metrics is not None:
        command.extend(["--save-precision-artifacts", str(metrics)])
    return command


def test_cpu_smoke_benchmark():
    args = build_parser().parse_args(
        [
            "--suite",
            "smoke",
            "--backend",
            "cpu",
            "--monitor-interval",
            "0.01",
        ]
    )

    result = run_benchmarks(args)

    assert result["schema_version"] == 3
    assert result["backend_requested"] == "cpu"
    assert result["backend_effective"] == "cpu"
    assert result["backend_effective_by_workload"] == {
        "transfer": "cpu",
        "ridge": "cpu",
        "percentile": "cpu",
        "consistency": "cpu",
    }
    repository = result["metadata"]["repository"]
    if repository["commit"] is not None:
        assert len(repository["commit"]) == 40
    assert repository["dirty"] in (True, False, None)
    assert set(result["benchmarks"]) == {
        "host_device_transfer",
        "ridge_mapper",
        "percentile_mapper",
        "linregress_consistency",
    }
    assert result["benchmarks"]["host_device_transfer"]["measured"] is False
    assert result["dtypes"]["requested"] == {
        "preparation": "float32",
        "mapper_input": "float32",
        "correlation": "float32",
        "compute_override_explicit": False,
        "correlation_override_explicit": False,
        "mixed_precision_allowed": False,
    }
    assert result["dtypes"]["effective"]["preparation"] == {
        "source": "float32",
        "target": "float32",
    }
    assert result["dataset"]["source"]["sha256_scope"] == "full_array_bytes"
    assert result["benchmarks"]["ridge_mapper"]["resources"]["cpu"]["total_seconds"] > 0
    percentile = result["benchmarks"]["percentile_mapper"]
    preprocessing_seconds = percentile["correlation_and_source_selection_seconds"]
    assert preprocessing_seconds > 0
    assert percentile["ridge_fit_seconds"] > 0
    assert preprocessing_seconds + percentile["ridge_fit_seconds"] == pytest.approx(
        percentile["fit_seconds"]
    )
    assert percentile["correlation_and_source_selection_measurement"] == (
        "correlation, percentile threshold, and source selection"
    )
    consistency = result["benchmarks"]["linregress_consistency"]
    assert consistency["result_shape"] == [1, 2, 6]
    assert np.isfinite(consistency["mean_test_r_xy_n_sb"])
    summaries = consistency["metric_summaries"]
    expected_metrics = {
        "denom_sb",
        "r_xx",
        "r_xx_sb",
        "r_xy",
        "r_xy_n_sb",
        "r_yy",
        "r_yy_sb",
    }
    assert set(summaries["train"]) == expected_metrics
    assert set(summaries["test"]) == expected_metrics
    assert (
        len(summaries["test"]["r_xy_n_sb"]["bootstrap_mean_target_median_by_fold"]) == 2
    )
    assert (
        consistency["mean_test_r_xy_n_sb"]
        == summaries["test"]["r_xy_n_sb"]["global"]["mean"]
    )
    json.dumps(result, allow_nan=False)


def test_consistency_saves_strict_learned_fit_and_metric_artifacts(tmp_path):
    output = tmp_path / "result.json"
    learned_fit = tmp_path / "learned_fit.npz"
    metrics = tmp_path / "result.metrics.npz"

    assert benchmark.main(_learned_fit_command(output, learned_fit, metrics)) == 0

    document = json.loads(output.read_text())
    consistency = document["benchmarks"]["linregress_consistency"]
    learned_record = consistency["learned_fit"]
    assert validate_artifact_record(output, learned_record["arrays"]) == learned_fit
    assert validate_artifact_record(
        output, learned_record["metadata"]
    ) == learned_fit.with_suffix(".json")
    assert validate_artifact_record(output, consistency["precision_metrics"]) == (
        metrics
    )
    assert not Path(learned_record["arrays"]["path"]).is_absolute()
    assert not Path(consistency["precision_metrics"]["path"]).is_absolute()

    arrays, metadata = load_precision_diagnostic(learned_fit)
    assert validate_precision_artifact(arrays, metadata) == []
    assert metadata["backend_requested"] == "cpu"
    assert metadata["backend_effective"] == "cpu"
    assert metadata["bootstrap_index"] == 0
    assert metadata["fold_index"] == 0
    assert metadata["seed"] == {
        "bootstrap_start": 1234,
        "bootstrap": 1234,
        "split": 0,
    }
    assert metadata["configuration"]["mapping"] == {
        "map_type": "percentile",
        "percentile": 50.0,
        "identity": False,
        "regression_type": "Ridge",
        "regression_kwargs": {"alpha": 1.0, "fit_intercept": True},
        "selection_cutoff_tolerance": {
            "epsilon_multiplier": 8,
            "scale": "max(1, abs(percentile_threshold))",
        },
    }
    np.testing.assert_array_equal(arrays["target_indices"], [0, 1, 2])
    for role in ("full", "half_1", "half_2"):
        assert arrays[f"{role}_correlations"].shape == (8, 3)
        assert arrays[f"{role}_coefficients"].shape[0] == 3
        assert arrays[f"{role}_intercepts"].shape == (3,)
        assert arrays[f"{role}_train_prediction"].shape[-1] == 3
        assert arrays[f"{role}_test_prediction"].shape[-1] == 3
    for split in ("train", "test"):
        for name in CONSISTENCY_METRICS:
            assert arrays[f"{split}_{name}"].shape == (3,)

    with np.load(metrics, allow_pickle=False) as metric_payload:
        assert set(metric_payload.files) == {
            f"{split}_{name}"
            for split in ("train", "test")
            for name in CONSISTENCY_METRICS
        }
        for split in ("train", "test"):
            for name in CONSISTENCY_METRICS:
                key = f"{split}_{name}"
                np.testing.assert_array_equal(arrays[key], metric_payload[key][0, 0])


def test_learned_fit_selects_exact_bootstrap_fold_and_targets(tmp_path):
    output = tmp_path / "result.json"
    learned_fit = tmp_path / "learned_fit.npz"
    metrics = tmp_path / "metrics.npz"
    command = _learned_fit_command(output, learned_fit, metrics)
    command.extend(
        [
            "--targets",
            "5",
            "--bootstrap-iters",
            "3",
            "--splits",
            "3",
            "--learned-bootstrap-index",
            "2",
            "--learned-fold-index",
            "1",
            "--learned-fit-target-limit",
            "3",
        ]
    )

    assert benchmark.main(command) == 0

    document = json.loads(output.read_text())
    consistency = document["benchmarks"]["linregress_consistency"]
    assert consistency["result_shape"] == [3, 3, 5]
    arrays, metadata = load_precision_diagnostic(learned_fit)
    assert metadata["bootstrap_index"] == 2
    assert metadata["fold_index"] == 1
    assert metadata["seed"]["bootstrap"] == 1236
    assert metadata["seed"]["split"] == 0
    selection = metadata["dataset"]["diagnostic_target_selection"]
    assert selection == {
        "strategy": "evenly_spaced",
        "full_target_count": 5,
        "selected_target_count": 3,
        "positions": [0, 2, 4],
        "original_target_ids": [0, 2, 4],
    }
    np.testing.assert_array_equal(arrays["target_indices"], [0, 2, 4])
    expected_splits = make_train_test_splits("random", 20, 3, 0.8, 0)
    np.testing.assert_array_equal(arrays["train_indices"], expected_splits[1]["train"])
    np.testing.assert_array_equal(arrays["test_indices"], expected_splits[1]["test"])
    with np.load(metrics, allow_pickle=False) as metric_payload:
        for split in ("train", "test"):
            for name in CONSISTENCY_METRICS:
                key = f"{split}_{name}"
                np.testing.assert_array_equal(
                    arrays[key], metric_payload[key][2, 1, [0, 2, 4]]
                )


def test_learned_fit_selects_explicit_target_ids(tmp_path):
    output = tmp_path / "result.json"
    learned_fit = tmp_path / "learned_fit.npz"
    command = _learned_fit_command(output, learned_fit)
    command.extend(
        [
            "--targets",
            "6",
            "--learned-target-indices",
            "1,4",
        ]
    )

    assert benchmark.main(command) == 0

    arrays, metadata = load_precision_diagnostic(learned_fit)
    np.testing.assert_array_equal(arrays["target_indices"], [1, 4])
    assert metadata["dataset"]["diagnostic_target_selection"] == {
        "strategy": "explicit",
        "full_target_count": 6,
        "selected_target_count": 2,
        "positions": [1, 4],
        "original_target_ids": [1, 4],
    }


def test_target_index_parser_requires_sorted_unique_values():
    parser = build_parser()
    args = parser.parse_args(["--fly-data-root", "/data", "--target-indices", "1,4,9"])
    assert args.target_indices == (1, 4, 9)
    with pytest.raises(SystemExit):
        parser.parse_args(["--target-indices", "4,1"])


def test_fold_index_runs_only_the_requested_generated_fold():
    args = build_parser().parse_args(
        [
            "--suite",
            "smoke",
            "--backend",
            "cpu",
            "--workloads",
            "consistency",
            "--bootstrap-iters",
            "1",
            "--splits",
            "3",
            "--fold-index",
            "2",
            "--monitor-interval",
            "0.01",
        ]
    )

    result = run_benchmarks(args)

    assert result["dataset"]["selected_fold_indices"] == [2]
    assert result["dataset"]["train_test_indices"][0]["fold_index"] == 2
    assert result["benchmarks"]["linregress_consistency"]["result_shape"] == [
        1,
        1,
        6,
    ]
    expected = make_train_test_splits("random", 48, 3, 0.8, 0)[2]
    np.testing.assert_array_equal(
        result["dataset"]["train_test_indices"][0]["train"], expected["train"]
    )
    np.testing.assert_array_equal(
        result["dataset"]["train_test_indices"][0]["test"], expected["test"]
    )


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--splits", "2", "--fold-index", "2"], "fold index"),
        (
            ["--bootstrap-iters", "2", "--learned-bootstrap-index", "2"],
            "learned bootstrap index",
        ),
        (["--splits", "2", "--learned-fold-index", "2"], "learned fold index"),
    ],
)
def test_precision_cell_indices_are_validated_before_data_load(
    monkeypatch, arguments, message
):
    args = build_parser().parse_args(["--suite", "smoke"] + arguments)
    monkeypatch.setattr(
        benchmark,
        "make_correlated_data",
        lambda *args, **kwargs: pytest.fail("data loaded before index validation"),
    )

    with pytest.raises(ValueError, match=message):
        run_benchmarks(args)


def test_learned_dataset_omits_dynamic_preparation_time():
    dataset = {
        "source": {"sha256": "source"},
        "target": {"sha256": "target"},
        "provenance": {
            "preparation_seconds": 3.25,
            "roi_h5": {"sha256": "roi"},
        },
    }
    selection = {
        "strategy": "all",
        "full_target_count": 2,
        "selected_target_count": 2,
        "positions": [0, 1],
        "original_target_ids": [0, 1],
    }

    learned = benchmark._learned_fit_dataset(dataset, selection)

    assert "preparation_seconds" not in learned["provenance"]
    assert learned["provenance"]["roi_h5"]["sha256"] == "roi"
    assert learned["source"]["sha256"] == "source"
    assert learned["target"]["sha256"] == "target"
    assert learned["diagnostic_target_selection"] == selection
    assert dataset["provenance"]["preparation_seconds"] == 3.25


def test_failed_learned_fit_does_not_publish_result_json(monkeypatch, tmp_path):
    output = tmp_path / "result.json"
    learned_fit = tmp_path / "learned_fit.npz"

    def fail_diagnostic(*args, **kwargs):
        raise RuntimeError("diagnostic write failed")

    monkeypatch.setattr(benchmark, "save_precision_diagnostic", fail_diagnostic)

    with pytest.raises(SystemExit):
        benchmark.main(_learned_fit_command(output, learned_fit))

    assert not output.exists()


def test_dtype_infers_correlation_from_mapper_input():
    args = build_parser().parse_args(
        [
            "--suite",
            "smoke",
            "--backend",
            "cpu",
            "--dtype",
            "float64",
            "--workloads",
            "percentile",
            "--monitor-interval",
            "0.01",
        ]
    )

    result = run_benchmarks(args)

    assert result["dtypes"]["requested"]["mapper_input"] == "float64"
    assert result["dtypes"]["requested"]["correlation"] == "float64"
    effective = result["dtypes"]["effective"]
    assert effective["mapper_input"] == {
        "source": "float64",
        "target": "float64",
    }
    assert effective["by_workload"]["percentile_mapper"]["correlation"] == ("float64")


@pytest.mark.parametrize(
    "override",
    [
        ["--compute-dtype", "float32"],
        ["--correlation-dtype", "float32"],
    ],
)
def test_dtype_rejects_implicit_mixed_precision_before_data_load(monkeypatch, override):
    args = build_parser().parse_args(
        ["--dtype", "float64", "--workloads", "ridge"] + override
    )
    monkeypatch.setattr(
        benchmark,
        "make_correlated_data",
        lambda *args, **kwargs: pytest.fail("data made before dtype validation"),
    )

    with pytest.raises(ValueError, match="allow-mixed-precision"):
        run_benchmarks(args)


def test_mixed_precision_requires_explicit_opt_in():
    args = build_parser().parse_args(
        [
            "--dtype",
            "float64",
            "--compute-dtype",
            "float32",
            "--correlation-dtype",
            "float64",
            "--allow-mixed-precision",
        ]
    )

    plan = benchmark._precision_plan(args)

    assert plan["preparation"] == np.dtype("float64")
    assert plan["mapper_input"] == np.dtype("float32")
    assert plan["correlation"] == np.dtype("float64")
    assert plan["allow_mixed_precision"]


def test_percentile_phase_includes_threshold_and_source_selection(monkeypatch):
    clock = [0.0]

    def advance(seconds):
        clock[0] += seconds

    def gather_correlations(mapper, X, Y):
        advance(2.0)
        mapper._corrs = np.zeros((X.shape[1], Y.shape[1]))

    def gather_thresholds(mapper):
        advance(3.0)
        mapper._thresholds = np.zeros(mapper._n_targets)

    def get_sources(mapper, target_idx):
        advance(5.0)
        selected = np.zeros(mapper._n_source, dtype=bool)
        selected[target_idx] = True
        return selected

    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(
        benchmark.PercentileNeuralMap,
        "_gather_correlations",
        gather_correlations,
    )
    monkeypatch.setattr(
        benchmark.PercentileNeuralMap,
        "_gather_thresholds",
        gather_thresholds,
    )
    monkeypatch.setattr(
        benchmark.PercentileNeuralMap,
        "_get_sources",
        get_sources,
    )

    mapper = benchmark.TimedPercentileNeuralMap(
        "cpu", percentile=100, identity=True, backend="cpu"
    )
    mapper.benchmark_phase = "fit"
    mapper.fit(np.ones((4, 2)), np.ones((4, 2)))

    assert mapper.correlation_and_source_selection_seconds == 15.0


def test_consistency_metric_summary_uses_canonical_fold_aggregation():
    metric = np.asarray(
        [
            [[1, 2, np.nan], [2, 4, 6]],
            [[3, 4, np.nan], [4, np.nan, 8]],
        ],
        dtype=np.float64,
    )

    summary = benchmark._consistency_metric_summary(metric)

    assert summary["bootstrap_mean_target_median_by_fold"] == [2.5, 4.0]
    assert summary["median_across_folds"] == 3.25
    assert summary["global"]["finite_count"] == 9
    assert summary["global"]["total_count"] == 12
    assert summary["global"]["finite_fraction"] == 0.75
    assert summary["global"]["mean"] == pytest.approx(34 / 9)
    assert summary["global"]["median"] == 4.0

    empty = benchmark._consistency_metric_summary(
        np.full((2, 3, 4), np.nan, dtype=np.float64)
    )
    assert empty["bootstrap_mean_target_median_by_fold"] == [None, None, None]
    assert empty["median_across_folds"] is None
    assert empty["global"]["mean"] is None


def test_selected_source_report_omits_large_per_target_lists():
    small = benchmark._selected_source_report([2, 4, 6])
    assert small["selected_sources_per_target"] == [2, 4, 6]
    assert small["selected_source_count_summary"]["median"] == 4.0
    assert small["selected_source_count_summary"]["per_target_values_recorded"]

    large = benchmark._selected_source_report(
        np.arange(benchmark.SELECTED_SOURCE_DETAIL_LIMIT + 1)
    )
    assert "selected_sources_per_target" not in large
    assert not large["selected_source_count_summary"]["per_target_values_recorded"]
    assert large["selected_source_count_summary"]["target_count"] == 257
    assert large["selected_source_count_summary"]["minimum"] == 0
    assert large["selected_source_count_summary"]["maximum"] == 256


def test_reference_result_compares_matching_runs(tmp_path):
    identity = {
        "seed": 10,
        "split_seed": 2,
        "config": {"bootstrap_iters": 5, "splits": 2},
        "dataset": {
            "origin": "fixture",
            "provenance": None,
            "source": {"shape": [8, 4], "dtype": "float32", "bytes": 128},
            "target": {"shape": [3, 8, 2], "dtype": "float32", "bytes": 192},
            "train_test_indices": [{"train": [0, 1], "test": [2, 3]}],
        },
    }

    def result(backend_name, seconds, metric):
        value = dict(identity)
        value["backend_effective"] = backend_name
        value["benchmarks"] = {
            "linregress_consistency": {
                "total_seconds": seconds,
                "metric_summaries": {
                    "method": "canonical",
                    "train": {"r_xy_n_sb": {"median_across_folds": metric + 0.1}},
                    "test": {"r_xy_n_sb": {"median_across_folds": metric}},
                },
            }
        }
        return value

    current = result("cuda", 4.0, 0.75)
    reference = result("cpu", 10.0, 0.70)
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(reference))

    comparison = benchmark._reference_comparison(current, path)

    assert comparison["configuration_matches"]
    assert comparison["backend_effective"] == "cpu"
    timing = comparison["benchmark_total_seconds"]["linregress_consistency"]
    assert timing["speedup"] == 2.5
    metric = comparison["consistency_metrics"]["test"]["r_xy_n_sb"]
    assert metric["difference"] == pytest.approx(0.05)
    assert metric["absolute_difference"] == pytest.approx(0.05)

    reference["config"] = {"bootstrap_iters": 4, "splits": 2}
    path.write_text(json.dumps(reference))
    comparison = benchmark._reference_comparison(current, path)
    assert not comparison["configuration_matches"]
    assert comparison["mismatched_fields"] == ["config"]
    assert comparison["benchmark_total_seconds"] == {}


def test_production_blocked_split_boundaries():
    splits = make_train_test_splits("blocked", 39, 5, 0.8, 0)

    assert [(int(split["test"][0]), int(split["test"][-1])) for split in splits] == [
        (0, 7),
        (8, 15),
        (16, 23),
        (24, 31),
        (32, 38),
    ]


def test_blocked_splits_require_at_least_two_folds():
    with pytest.raises(ValueError, match="at least two"):
        make_train_test_splits("blocked", 39, 1, 0.8, 0)


def test_load_fly_data_event_sources_and_target_slice(tmp_path):
    root, source_fly, target_fly = _make_fly_fixture(tmp_path)

    source, target, provenance = load_fly_data(
        root,
        source_fly,
        target_fly,
        "right",
        "IPS",
        "validated",
        np.dtype("float32"),
        target_limit=2,
    )

    assert source.shape == (4, 6, 3)
    assert target.shape == (4, 6, 2)
    queries = np.asarray([0, 3, 4, 5])[:, None] + np.linspace(-1.0, 1.5, 6)
    expected = np.interp(queries, np.arange(12), np.arange(12))
    np.testing.assert_allclose(source[:, :, 0], expected)
    assert provenance["roi_order"] == ["IPS_L", "IPS_R"]
    assert provenance["source_events"]["event_count"] == 4
    assert provenance["target_selection"] == {
        "is_slice": True,
        "full_count": 3,
        "selected_count": 2,
        "strategy": "evenly_spaced",
        "indices": [0, 2],
    }
    assert provenance["source_nwb"]["timestamp_storage"] == "implicit"
    assert provenance["target_nwb"]["timestamp_storage"] == "explicit"
    assert provenance["dandiset"]["identifier"] == "DANDI:000727"
    assert provenance["dandiset"]["version"] == "0.240106.0043"
    assert provenance["dandiset"]["doi"] == ("10.48324/dandi.000727/0.240106.0043")
    assert provenance["interpolation_queries"]["source"] == {
        "query_count": 24,
        "clamped_query_count": 2,
        "before_start_count": 2,
        "after_end_count": 0,
        "events_with_clamped_queries": 1,
    }
    assert provenance["interpolation_queries"]["target"]["clamped_query_count"] == 0

    source, target, provenance = load_fly_data(
        root,
        source_fly,
        target_fly,
        "right",
        "IPS",
        "historical",
        np.dtype("float32"),
    )

    assert source.shape == (5, 6, 3)
    assert target.shape == (4, 6, 3)
    assert provenance["source_events"]["event_count"] == 5
    assert provenance["source_events"]["event_file"]["path"].endswith(
        "fly_001_turn_bouts.csv"
    )
    assert len(provenance["source_events"]["event_file"]["sha256"]) == 64


@pytest.mark.parametrize(
    ("region", "width", "roi_order"),
    [("IPS_L", 2, ["IPS_L"]), ("IPS_R", 1, ["IPS_R"])],
)
def test_load_fly_data_selects_individual_ips_regions(
    tmp_path, region, width, roi_order
):
    root, source_fly, target_fly = _make_fly_fixture(tmp_path)

    source, target, provenance = load_fly_data(
        root,
        source_fly,
        target_fly,
        "right",
        region,
        "validated",
        np.dtype("float64"),
    )

    assert source.shape == (4, 6, width)
    assert target.shape == (4, 6, width)
    assert source.dtype == target.dtype == np.float64
    assert provenance["roi_order"] == roi_order
    assert build_parser().parse_args(["--region", region]).region == region


def test_nwb_lookup_rejects_ambiguous_sessions(tmp_path):
    root, source_fly, _ = _make_fly_fixture(tmp_path)
    number = source_fly.split("_")[-1]
    duplicate = root / f"sub-fly-{number}"
    duplicate /= f"sub-fly-{number}_ses-second_behavior+image+ophys.nwb"
    duplicate.touch()

    with pytest.raises(ValueError, match="multiple NWB"):
        benchmark._nwb_for_fly(root, source_fly)


def test_backend_resolution_is_per_workload(monkeypatch):
    monkeypatch.setattr(
        benchmark.neural_backends,
        "resolve_backend",
        lambda backend, _regression_type, regression_kwargs: "cuda",
    )
    monkeypatch.setattr(
        benchmark.neural_backends,
        "resolve_torch_backend",
        lambda backend: "cpu",
    )
    monkeypatch.setattr(
        benchmark,
        "_resolve_transfer_backend",
        lambda backend, gpu_index: "cuda",
    )

    resolved = benchmark._resolve_workload_backends(
        "cuda",
        ["transfer", "ridge", "percentile", "consistency"],
        "ridge",
        {"alpha": 1.0},
        0,
    )

    assert resolved == {
        "transfer": "cuda",
        "ridge": "cuda",
        "percentile": "cpu",
        "consistency": "cuda",
    }


def test_benchmark_parser_rejects_auto_backend():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--backend", "auto"])


def test_explicit_cuda_preflight_fails_before_fly_data_load(monkeypatch, tmp_path):
    args = build_parser().parse_args(
        [
            "--backend",
            "cuda",
            "--fly-data-root",
            str(tmp_path),
            "--workloads",
            "ridge",
        ]
    )
    calls = []
    monkeypatch.setattr(
        benchmark,
        "_resolve_workload_backends",
        lambda *args, **kwargs: calls.append("resolve") or {"ridge": "cuda"},
    )

    def reject_device(*args, **kwargs):
        calls.append("device")
        raise RuntimeError("CUDA device preflight failed")

    monkeypatch.setattr(benchmark, "_select_cuda_device", reject_device)
    monkeypatch.setattr(
        benchmark,
        "load_fly_data",
        lambda *args, **kwargs: pytest.fail("fly data loaded before CUDA preflight"),
    )

    with pytest.raises(RuntimeError, match="preflight failed"):
        run_benchmarks(args)

    assert calls == ["resolve", "device"]


def test_explicit_cuda_rejects_any_non_cuda_workload_before_data_load(
    monkeypatch, tmp_path
):
    args = build_parser().parse_args(
        [
            "--backend",
            "cuda",
            "--fly-data-root",
            str(tmp_path),
            "--workloads",
            "transfer",
            "ridge",
            "percentile",
            "consistency",
        ]
    )
    monkeypatch.setattr(
        benchmark,
        "_resolve_workload_backends",
        lambda *args, **kwargs: {
            "transfer": "cuda",
            "ridge": "cuda",
            "percentile": "cpu",
            "consistency": "cuda",
        },
    )
    monkeypatch.setattr(
        benchmark,
        "_select_cuda_device",
        lambda *args, **kwargs: pytest.fail(
            "selected device after failed postcondition"
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "load_fly_data",
        lambda *args, **kwargs: pytest.fail("loaded data after failed postcondition"),
    )

    with pytest.raises(RuntimeError, match="percentile"):
        run_benchmarks(args)


def test_explicit_ridge_resolution_does_not_require_torch(monkeypatch):
    monkeypatch.setattr(
        benchmark.neural_backends,
        "resolve_backend",
        lambda backend, _regression_type, regression_kwargs: "cuda",
    )
    monkeypatch.setattr(
        benchmark.neural_backends,
        "resolve_torch_backend",
        lambda backend: pytest.fail("Ridge resolution queried PyTorch"),
    )

    assert benchmark._resolve_workload_backends(
        "cuda", ["ridge"], "ridge", {"alpha": 1.0}, 0
    ) == {"ridge": "cuda"}


def test_cuda_unsupported_solver_raises(monkeypatch):
    monkeypatch.setattr(
        benchmark.neural_backends,
        "cuda_available",
        lambda: pytest.fail("unsupported solver queried CUDA availability"),
    )

    with pytest.raises(ValueError, match="solver"):
        benchmark._resolve_workload_backends(
            "cuda", ["ridge"], "ridge", {"solver": "sag"}, 0
        )


def test_gpu_index_selects_cupy_device(monkeypatch):
    selected = []

    class Device:
        def __init__(self, index):
            self.index = index

        def use(self):
            selected.append(self.index)

    cupy = SimpleNamespace(
        cuda=SimpleNamespace(
            Device=Device,
            runtime=SimpleNamespace(getDeviceCount=lambda: 3),
        )
    )
    monkeypatch.setitem(sys.modules, "cupy", cupy)

    benchmark._select_cuda_device(2, require_torch=False)

    assert selected == [2]


def test_cuda_worker_guard_applies_only_to_consistency(monkeypatch):
    args = build_parser().parse_args(
        [
            "--suite",
            "smoke",
            "--backend",
            "cuda",
            "--workers",
            "4",
            "--gpu-index",
            "2",
            "--workloads",
            "ridge",
        ]
    )
    selected = []
    benchmark_call = []
    monkeypatch.setattr(
        benchmark,
        "_resolve_workload_backends",
        lambda *args, **kwargs: {"ridge": "cuda"},
    )
    monkeypatch.setattr(
        benchmark,
        "_select_cuda_device",
        lambda gpu_index, require_torch: selected.append((gpu_index, require_torch)),
    )
    monkeypatch.setattr(
        benchmark,
        "_cuda_probe",
        lambda gpu_index: {"available": True, "device_index": gpu_index},
    )
    monkeypatch.setattr(benchmark, "_metadata", lambda args, _probe: {})
    monkeypatch.setattr(
        benchmark,
        "benchmark_ridge",
        lambda *arguments: benchmark_call.append(arguments) or {},
    )

    result = run_benchmarks(args)

    assert result["backend_effective"] == "cuda"
    assert result["backend_effective_by_workload"] == {"ridge": "cuda"}
    assert selected == [(2, False)]
    assert benchmark_call[0][3:5] == ("cuda", "cuda")


def test_cuda_consistency_still_rejects_multiple_workers(monkeypatch):
    args = build_parser().parse_args(
        [
            "--suite",
            "smoke",
            "--backend",
            "cuda",
            "--workers",
            "2",
            "--workloads",
            "consistency",
        ]
    )
    monkeypatch.setattr(
        benchmark,
        "_resolve_workload_backends",
        lambda *args, **kwargs: {"consistency": "cuda"},
    )
    monkeypatch.setattr(
        benchmark,
        "_select_cuda_device",
        lambda *args, **kwargs: pytest.fail("device selected before worker validation"),
    )

    with pytest.raises(ValueError, match="workers 1"):
        run_benchmarks(args)


def test_real_fly_blocked_percentile_consistency_only(tmp_path):
    root, source_fly, target_fly = _make_fly_fixture(tmp_path)
    args = build_parser().parse_args(
        [
            "--suite",
            "smoke",
            "--fly-data-root",
            str(root),
            "--source-fly",
            source_fly,
            "--target-fly",
            target_fly,
            "--event-source",
            "historical",
            "--target-limit",
            "2",
            "--split-strategy",
            "blocked",
            "--splits",
            "3",
            "--consistency-map",
            "percentile",
            "--percentile",
            "99",
            "--bootstrap-iters",
            "1",
            "--seed",
            "1000",
            "--workloads",
            "consistency",
            "--monitor-interval",
            "0.01",
        ]
    )

    result = run_benchmarks(args)

    assert result["dataset"]["origin"] == "brezovec_fly"
    assert result["dataset"]["split_strategy"] == "blocked"
    assert [split["test"] for split in result["dataset"]["train_test_indices"]] == [
        [0, 1],
        [2, 3],
        [4, 5],
    ]
    assert set(result["benchmarks"]) == {"linregress_consistency"}
    consistency = result["benchmarks"]["linregress_consistency"]
    assert consistency["mapping"]["map_type"] == "percentile"
    assert consistency["mapping"]["map_kwargs"]["percentile"] == 99
    assert consistency["mapping"]["map_kwargs"]["identity"] is False
    assert consistency["result_shape"] == [1, 3, 2]
    json.dumps(result, allow_nan=False)
