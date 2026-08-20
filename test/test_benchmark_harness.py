import json

import numpy as np

from benchmarks.benchmark_neural_backends import build_parser, run_benchmarks


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

    assert result["backend_effective"] == "cpu"
    assert set(result["benchmarks"]) == {
        "host_device_transfer",
        "ridge_mapper",
        "percentile_mapper",
        "linregress_consistency",
    }
    assert result["benchmarks"]["host_device_transfer"]["measured"] is False
    assert result["benchmarks"]["ridge_mapper"]["resources"]["cpu"]["total_seconds"] > 0
    percentile = result["benchmarks"]["percentile_mapper"]
    assert percentile["correlation_seconds"] > 0
    assert percentile["source_selection_seconds"] > 0
    assert percentile["ridge_fit_seconds"] > 0
    consistency = result["benchmarks"]["linregress_consistency"]
    assert consistency["result_shape"] == [1, 2, 6]
    assert np.isfinite(consistency["mean_test_r_xy_n_sb"])
    json.dumps(result, allow_nan=False)
