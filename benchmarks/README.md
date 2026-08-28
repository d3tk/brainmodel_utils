# Neural backend benchmarks

Run the deterministic CPU smoke workload from the repository root:

```bash
python -m benchmarks.benchmark_neural_backends --suite smoke --backend cpu
```

`cpu` is the unchanged default. `cuda` requires the requested CUDA implementation and raises if the configuration or runtime is unsupported:

```bash
python -m benchmarks.benchmark_neural_backends \
  --suite smoke \
  --backend cuda \
  --workers 1
```

The CUDA dependencies remain optional and are imported lazily. CUDA transfer measurements and Ridge need CuPy; percentile correlation additionally needs CUDA-enabled PyTorch. The environment tested on the RTX 3080 used Python 3.12, `cupy-cuda12x`, PyTorch, and `scikit-learn<1.8`; the exact installation command and backend rules are in the [main README](../README.md#optional-cuda-backend).

CUDA Ridge accepts a finite scalar `alpha > 0`, boolean `fit_intercept`, and `solver` omitted or set to `"auto"`. Unsupported CUDA settings raise a clear error. Single-, multi-target, and grouped fits use a batched CuPy closed-form Ridge implementation. CUDA percentile correlation follows the input dtype unless explicitly overridden, with TF32 disabled for stable source rankings; the harness sets `store_correlations=False` to avoid retaining the full source-by-target matrix.

Correlations within eight dtype epsilons of a percentile threshold are treated as numerical ties. CPU and CUDA use the same rule. The learned-fit artifact records the threshold and nearest values above and below it, so a cutoff can be audited independently of the selected-source set.

Each CUDA consistency process requires `--workers 1`; CPU consistency can still use multiple joblib workers. Use `--shards N` below to divide one comparison's bootstrap iterations among multiple CUDA processes on a GPU. The worker guard does not restrict standalone Ridge, percentile, or transfer measurements.

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

The older bundled sweeps are historical artifacts under earlier schemas. Current schema-3 results identify GPUs by UUID, record requested and effective dtypes and backends, preserve exact target and fold indices, and include phase/resource telemetry. Historical float32 or mixed-card timings are not acceptance evidence for a true-float64 change.

Consistency output includes every train/test metric. Its canonical per-fold value is the target median after taking each target's mean over bootstrap iterations; finite counts, fraction, mean, and median are recorded globally. Standalone percentile runs retain the per-target selected-source counts for at most 256 targets and use compact count statistics for larger runs.

Pass a matching prior result to compare total runtimes and canonical consistency metrics in the new JSON:

```bash
python -m benchmarks.benchmark_neural_backends \
  --suite production \
  --backend cuda \
  --workers 1 \
  --workloads consistency \
  --reference-result benchmarks/results/cpu_reference.json \
  --output /path/to/artifacts/cuda_result.json
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
| Mapping grid | percentiles `0`, `50`, `75`, `100` × Ridge alphas `0.01`, `1`, `100`, `10000` |
| Repetitions | fixed, recorded bootstrap seed ranges |

Run a small true-float64 CPU reference:

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
  --percentile 100 \
  --alpha 1 \
  --bootstrap-iters 2 \
  --seed 1000 \
  --backend cpu \
  --workers 1 \
  --workloads consistency \
  --target-limit 128 \
  --dtype float64 \
  --compute-dtype float64 \
  --correlation-dtype float64 \
  --output /path/to/artifacts/cpu.json \
  --save-precision-artifacts /path/to/artifacts/cpu.metrics.npz \
  --save-learned-fit /path/to/artifacts/cpu.learned.npz
```

Then run the matched CUDA protocol with one worker by changing the last part of the command to:

```bash
  --backend cuda \
  --workers 1 \
  --workloads consistency \
  --reference-result /path/to/artifacts/cpu.json \
  --output /path/to/artifacts/cuda.json \
  --save-precision-artifacts /path/to/artifacts/cuda.metrics.npz \
  --save-learned-fit /path/to/artifacts/cuda.learned.npz
```

`historical` reads all pilot CSV turns. `validated` uses the boundary-filtered NPZ windows instead (`532/36` events). Both modes interpolate the two ROIs onto the saved relative-time grid. For a short validation run, `--target-limit 128` selects evenly spaced targets and marks the output as a slice; omit it for the full 16,636-target protocol. The five blocked test folds are `0–7`, `8–15`, `16–23`, `24–31`, and `32–38`.

Provenance includes the DANDI identifier, version, DOI, file paths, sizes and mtimes, NWB timestamp representation, ROI shapes/order, event source and counts, interpolation-clamp counts, target selection, exact split indices, and repository commit, dirty state, and source-tree checksum.

## Learned-map precision

`--save-precision-artifacts` stores all fourteen canonical train/test metric arrays. `--save-learned-fit` additionally stores one requested bootstrap/fold cell with correlations, thresholds, cutoff neighbors, selected source IDs, aligned coefficients, intercepts, predictions, finite masks, input checksums, target IDs, splits, dtypes, backend, and repository identity. NPZ payloads contain numeric arrays only and load with `allow_pickle=False`.

Compare matched float64 learned fits with:

```bash
python -m benchmarks.precision \
  /path/to/artifacts/cuda.learned.npz \
  /path/to/artifacts/cpu.learned.npz \
  --output /path/to/artifacts/cpu-vs-cuda.json
```

Float64 acceptance requires valid structure and provenance, exact selected-source sets and finite masks, mapping/prediction agreement at `rtol=1e-10` and `atol=1e-12`, unnormalized metric p95 error at most `1e-12`, finite test-mean `r_xy_n_sb` error at most `1e-10`, and normalized values within `atol=1e-8 + rtol=1e-10 × abs(reference)`. Cutoff-neighbor values are diagnostic because a one-ULP threshold change can change which value is described as “nearest below” without changing the tolerance-aware selection.

Use `--report-only` plus explicit `--allow-provenance-mismatch` entries for CPU/GPU float32-to-float64 characterization. Report mode still fails malformed or incomparable artifacts; it only stops numerical acceptance thresholds from determining the process exit code.

## `cuml.accel` experiment

RAPIDS 25.10 `cuml.accel` did not accelerate this API: its profiler reported zero GPU Ridge calls because both the multi-target mapper and the per-target `(samples, 1)` fits were rejected as multi-output. On the warmed smoke workload it preserved predictions exactly but was 1.81× slower for Ridge and 1.70× slower for percentile mapping, so the explicit backend remains the supported GPU path.

## Manifests, sharding, and server campaigns

`python -m benchmarks.config_manifest` generates the canonical sixteen-cell scientific grid and binds one execution profile: exact seed range, folds, target IDs, requested/expected backend, dtypes, environment, commit, dirty state, and source checksum. Different validation stages use separate signed execution manifests but must share the same `grid_checksum`; scripts must not reconstruct the scientific grid independently.

```bash
python -m benchmarks.config_manifest \
  --output /path/to/artifacts/cuda-f64-selected-targets.json \
  --seed 1000 \
  --bootstrap-iters 2 \
  --stimuli 39 \
  --folds 5 \
  --target-indices 0,536,1073

python -m benchmarks.campaign \
  --configuration-manifest /path/to/artifacts/cuda-f64-selected-targets.json \
  --config-id p100_a1 \
  --output-dir /path/to/artifacts/run \
  --save-precision
```

A completed assignment is skipped only when its JSON and metric NPZ both validate against the full configuration, biological/file provenance, exact shapes and metric keys, repository fingerprint, seed range, and checksums. The metric NPZ is promoted first; the atomic completion JSON is published last. Missing, truncated, corrupt, stale, or incompatible work reruns or refuses to merge.

`--shards N` divides one comparison's contiguous bootstrap seed range across `N` independent `workers=1` CUDA processes and merges raw arrays in seed order. `--jobs N` instead launches independent comparisons and measures throughput. Neither option is native distributed Ridge, and one CUDA process still uses one physical GPU. `--devices` can route independent processes across visible GPUs.

Submit a manifest's sixteen cells with a concurrency cap of eight on `general` or twenty-four on `preempt`. The runner requires the data root and animal IDs explicitly; set `EVENTS_DIR` as well when the selected event files live outside the data root:

```bash
PARTITION=general MAX_IN_FLIGHT=8 \
FLY_DATA_ROOT=/path/to/000727 \
SOURCE_FLY=fly_100 TARGET_FLY=fly_099 \
  benchmarks/slurm/submit_manifest_grid.sh \
  /path/to/artifacts/cuda-f64-selected-targets.json \
  /path/to/artifacts/server-run
```

Keep generated JSON, NPZ, and Slurm logs outside the repository. Timing reports must distinguish one-process latency, bootstrap-sharded latency for one comparison, and independent-comparison throughput; mixed GPU models are operational observations rather than controlled scaling evidence.
