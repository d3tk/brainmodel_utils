# Neural backend benchmarks

Run the deterministic CPU smoke workload from the repository root:

```bash
python -m benchmarks.benchmark_neural_backends --suite smoke --backend cpu
```

`cpu` is the unchanged default. `cuda` requires the requested CUDA implementation, while `auto` resolves each selected workload independently and falls back to CPU when its CUDA capability or configuration is unavailable:

```bash
python -m benchmarks.benchmark_neural_backends --suite smoke --backend auto
python -m benchmarks.benchmark_neural_backends \
  --suite smoke \
  --backend cuda \
  --workers 1
```

The CUDA dependencies remain optional and are imported lazily. CUDA transfer measurements need CuPy, Ridge needs CuPy and cuML, and percentile correlation needs CUDA-enabled PyTorch as well. The environment tested on the RTX 3080 used Python 3.12, RAPIDS 25.10, `cupy-cuda12x`, PyTorch, and `scikit-learn<1.8`; the exact installation command and backend rules are in the [main README](../README.md#optional-cuda-backend).

CUDA Ridge accepts a scalar `alpha`, `fit_intercept`, and `solver` omitted or set to `"auto"` or `"svd"`. With `auto`, unsupported settings are reported as an effective CPU workload; explicit `cuda` raises a clear error. Because cuML 25.10 lacks the required native multi-target behavior, the common `solver="auto"`, `alpha > 0` path uses a batched CuPy closed-form Ridge implementation. CUDA percentile correlation uses float32 with TF32 disabled for stable source rankings, and the harness sets `store_correlations=False` to avoid retaining the full source-by-target matrix.

For consistency, a CUDA workload requires `--workers 1`; CPU consistency can still use multiple joblib workers. The guard does not restrict standalone Ridge, percentile, or transfer measurements.

## Suites and output

`synthetic` is a medium correlated workload. The bundled `production` suite is a production-shaped synthetic scaling proxy (39 stimuli, 27,118 sources, 200 bootstraps, and five folds), not a claim to reproduce an exact experiment. Every shape and repetition count can be overridden:

```bash
python -m benchmarks.benchmark_neural_backends \
  --suite synthetic \
  --backend cpu \
  --stimuli 512 \
  --sources 1024 \
  --targets 128 \
  --trials 20 \
  --bootstrap-iters 10 \
  --splits 5 \
  --workers 4 \
  --output benchmark.json
```

Select components with `--workloads transfer ridge percentile consistency`; omitting the option runs all four. Results record both the requested backend and the effective backend for each workload.

Each JSON result contains mapper phase timings, warmed H2D/D2H timings, consistency timing, shapes, split indices, CPU/thread/library metadata, repository commit and dirty state, process-tree peak RSS and CPU time, and best-effort whole-device VRAM and utilization. Whole-device GPU readings can include unrelated processes.

The older bundled CPU sweeps remain schema 1 historical outputs. New results use schema 2, where percentile preprocessing reports correlation, percentile-threshold calculation, and source selection as one phase because the optimized CUDA path performs them together.

Matched schema-2 synthetic results are available for [CPU](results/cpu_synthetic_matched_workers1.json) and [CUDA](results/cuda_synthetic_matched_workers1.json). On this small workload, GPU setup dominates the standalone mapper timings, while the warmed full consistency workflow is 23.37x faster and its largest canonical metric difference is `3.34e-7`.

Consistency output includes every train/test metric. Its canonical per-fold value is the target median after taking each target's mean over bootstrap iterations; finite counts, fraction, mean, and median are recorded globally. Standalone percentile runs retain the per-target selected-source counts for at most 256 targets and use compact count statistics for larger runs.

Pass a matching prior result to compare total runtimes and canonical consistency metrics in the new JSON:

```bash
python -m benchmarks.benchmark_neural_backends \
  --suite production \
  --backend cuda \
  --workers 1 \
  --workloads consistency \
  --reference-result benchmarks/results/cpu_reference.json \
  --output benchmarks/results/cuda_result.json
```

The reported speedup is reference seconds divided by current seconds. Comparisons are emitted only when seeds, configuration, input summaries, splits, and real-data identity match; otherwise the JSON lists the mismatched fields.

## Real fly-data protocol

An `.npz` fixture may provide `source` and `target` arrays. `source` can be `stimuli x sources` or `trials x stimuli x sources`; `target` must be `trials x stimuli x targets`.

The mounted Brezovec store can instead be read directly without creating a multi-gigabyte fixture. The production comparison uses:

| Setting | Value |
| ------- | ----- |
| Dandiset | `DANDI:000727`, version `0.240106.0043`, [DOI](https://doi.org/10.48324/dandi.000727/0.240106.0043) |
| Animals | source `fly_087`, target `fly_089` |
| Events | right turns, historical `535/36` source/target events |
| Region | `IPS_L` then `IPS_R`, 16,636 units per fly |
| Stimuli | 39 relative-time bins |
| Splits | five contiguous blocked folds |
| Mapping | percentile 99, Ridge `alpha=1` |
| Repetitions | 200 split-half bootstraps, seed 1000 |

Run the CPU reference with eight workers:

```bash
python -m benchmarks.benchmark_neural_backends \
  --suite production \
  --fly-data-root /mnt/z/brezovec_fly_data/000727 \
  --source-fly fly_087 \
  --target-fly fly_089 \
  --direction right \
  --region IPS \
  --event-source historical \
  --split-strategy blocked \
  --splits 5 \
  --consistency-map percentile \
  --percentile 99 \
  --alpha 1 \
  --bootstrap-iters 200 \
  --seed 1000 \
  --backend cpu \
  --workers 8 \
  --workloads consistency \
  --output benchmarks/results/cpu_fly087_to_fly089_right_ips_boot200_workers8.json
```

Then run the matched CUDA protocol with one worker by changing the last part of the command to:

```bash
  --backend cuda \
  --workers 1 \
  --workloads consistency \
  --reference-result benchmarks/results/cpu_fly087_to_fly089_right_ips_boot200_workers8.json \
  --output benchmarks/results/cuda_fly087_to_fly089_right_ips_boot200_workers1.json
```

`historical` reads all pilot CSV turns. `validated` uses the boundary-filtered NPZ windows instead (`532/36` events). Both modes interpolate the two ROIs onto the saved relative-time grid. For a short validation run, `--target-limit 128` selects evenly spaced targets and marks the output as a slice; omit it for the full 16,636-target protocol. The five blocked test folds are `0–7`, `8–15`, `16–23`, `24–31`, and `32–38`.

Provenance includes the DANDI identifier, version, DOI, file paths, sizes and mtimes, NWB timestamp representation, ROI shapes/order, event source and counts, interpolation-clamp counts, target selection, exact split indices, and repository commit/dirty state.

## RTX 3080 production result

| Run | Wall time | Peak RSS | Peak VRAM |
| --- | ---: | ---: | ---: |
| [CPU, 8 workers](results/cpu_fly087_to_fly089_right_ips_boot200_workers8.json) | 3:31:39.5 | 40.33 GiB (38.70 GiB increase) | n/a |
| [RTX 3080 CUDA, 1 worker](results/cuda_fly087_to_fly089_right_ips_boot200_workers1.json) | 0:37:32.0 | 6.43 GiB (4.18 GiB increase) | 1.73 GiB whole-device (0.58 GiB increase) |

The host was an Intel Core i7-12700KF with 12 logical CPUs and an NVIDIA GeForce RTX 3080 10 GiB. CUDA was **5.64x faster**. The canonical test `r_xy_n_sb` difference (CUDA minus CPU) was **+0.001316**; the corresponding global finite-value mean difference was **-0.000107**. The largest canonical metric difference was 0.001316, within the full-pipeline parity tolerance of 0.005. Peak CUDA utilization was 87%.

## `cuml.accel` experiment

RAPIDS 25.10 `cuml.accel` did not accelerate this API: its profiler reported zero GPU Ridge calls because both the multi-target mapper and the per-target `(samples, 1)` fits were rejected as multi-output. On the warmed smoke workload it preserved predictions exactly but was 1.81× slower for Ridge and 1.70× slower for percentile mapping, so the explicit backend remains the supported GPU path.

## Server campaign

Schema 3 records GPU identity by UUID (so `CUDA_VISIBLE_DEVICES` remapping is not confused with `nvidia-smi` index 0), correlation/Ridge batch settings, phase timings, and utilization time series. Independent comparisons are launched with unique seed ranges and output paths:

There is no debug, array, or CPU partition. All campaign jobs use `general` / `normal` GPU nodes (`--gres=gpu:1`). CPU comparisons still run on those nodes via `--backend cpu`.

```bash
sbatch benchmarks/slurm/probe.sh
BACKEND=cpu WORKERS=8 BOOTSTRAP_ITERS=20 TARGET_LIMIT=128 \
  sbatch --partition=general --qos=normal benchmarks/slurm/run_one.sh
```

`python -m benchmarks.campaign` assigns non-overlapping seeds. `python -m benchmarks.precision` gates are also imported by `benchmarks.precision.compare_precision`. CUDA correlation chunks and Ridge workspace bytes are CLI flags: `--correlation-target-batch-size` and `--ridge-workspace-bytes`. `--compute-dtype` casts at the mapper boundary; `--dtype` still changes data preparation as well.

Screen JSON from Babel `general` is under [results/campaign](results/campaign/), with a short write-up in [results/campaign/SCREEN.md](results/campaign/SCREEN.md). Those 5-bootstrap × 128-target jobs are packing probes, not full IPS comparisons.
