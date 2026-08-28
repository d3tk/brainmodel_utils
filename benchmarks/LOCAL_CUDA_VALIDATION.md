# Local CUDA validation

This report summarizes bounded correctness testing on the workstation. Raw JSON and NPZ artifacts remain outside the repository; every artifact records its repository commit, source-tree checksum, input provenance, requested and effective backend, and requested and effective dtypes.

## Environment

- NVIDIA RTX 3080, 10 GiB VRAM
- 6 physical / 12 logical CPU cores and approximately 63 GiB host RAM
- Python 3.12, CuPy 14.2, and CUDA-enabled PyTorch
- Real `fly_087` and `fly_089` data from DANDI dataset `000727`
- CPU-only validation environment with no CuPy, PyTorch, or cuML

## Automated checks

- CUDA environment: 185 tests passed
- Dedicated CUDA marker: 59 tests passed
- CPU-only environment: 128 tests passed and 57 CUDA-only tests skipped
- Ruff formatting and lint, Vulture, shell syntax, wheel import smoke, and `git diff --check` passed
- Explicit unavailable or unsupported CUDA configurations failed before data loading; campaign launchers rejected `backend="auto"`

## Float64 agreement

The first real-data anchor used `fly_087 -> fly_089`, right-turn historical events, combined IPS, five blocked folds, two bootstraps, 128 evenly spaced targets, percentile 100, and alpha 1. CPU and CUDA selected exactly the same sources and finite masks. Learned coefficient, intercept, and prediction differences were at roundoff scale, and the finite test `r_xy_n_sb` mean differed by approximately `6.1e-16`.

The complete percentile `0/50/75/100` by alpha `0.01/1/100/10000` grid was then run with one bootstrap, five folds, and 32 real targets. All 16 CPU/CUDA pairs had identical finite masks. Across unnormalized canonical metrics, the aggregate p95 absolute difference was `3.33e-15`; the largest observed absolute difference was `9.95e-13`. The worst normalized value differed by `1.47e-10`, only 1.1% of its combined absolute-plus-relative tolerance.

Promoted five-bootstrap, 1,024-target runs at alpha 1 passed for both mapping extremes:

- Percentile 0 selected all 16,636 sources, matched direct multi-target Ridge, and used the shared-design CUDA solve. On a stratified 128-target learned subset, source sets were exact, maximum learned-array error was below `1e-15`, unnormalized metric p95 error was `2.22e-14`, and finite test-mean error was `1.52e-14`.
- Percentile 100 retained only the maximum-correlation source or numerical ties. Source sets were exact, maximum learned-array error was below `1e-15`, and all metric gates passed.

The percentile-0 and percentile-100 alpha extremes (`0.01` and `10000`) also passed learned-map comparison. Separate real-data checks covered both fly directions, left and right events, historical and validated events, `IPS_L`, `IPS_R`, combined IPS, a two-dimensional model-like source, and unequal trial counts.

These workstation timings are diagnostic, not server performance claims. The promoted percentile-0 CPU/CUDA runs took approximately 8.3/1.2 minutes, while percentile 100 took approximately 2.4/0.34 minutes; peak device use remained below 3.2 GiB.

## Float32 characterization

Float32 remains report-only. On the 128-target anchor, CPU-float32 and CUDA-float32 retained identical source selections and finite masks. Their unnormalized metric p95 difference was approximately `2.1e-5`, with a finite test-mean difference of `4.3e-7`. Relative to float64, CPU and CUDA had similar unnormalized p95 changes of approximately `6.6e-5`; normalized metrics can amplify small denominator changes, reaching absolute differences of `0.017` on CPU and `0.027` on CUDA.

The historical `0.54` outlier was a float32 normalized metric with a denominator near `0.0054`, not evidence of a CUDA-specific float64 error. Its exact learned-map trace requires the server-only `fly_100 -> fly_099` fixture.

## Sharding and recovery

A four-bootstrap monolithic CUDA result matched two merged two-bootstrap shards bit for bit, including seed order and all metric arrays. The same held for a percentile-0 smoke run. Recovery tests verified valid skips and selective reruns for missing or truncated JSON, corrupt or missing NPZ, and stale configuration; incompatible, gapped, overlapping, duplicated, or structurally invalid shards refused to merge.

## Local conclusion

The local correctness and reliability gates pass. Full-target float64 performance, controlled 40–48 GiB GPU scaling, the historical float32 outlier trace, and two-GPU routing remain server acceptance work.
