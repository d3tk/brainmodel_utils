# CUDA benchmark report

The CUDA backend was validated against the CPU implementation in float64 on real fly data.

## 🧪 Protocol

| Field | Value |
| --- | --- |
| Dataset | Brezovec fly data |
| Comparison | `fly_100` → `fly_099`, right IPS |
| Events | 624 source and 571 target events |
| Time bins | 30 |
| Source and target units | 27,118 each |
| Temporal folds | 5 |
| Precision | Float64 |
| Mapping | Percentile selection, `identity=False`, Ridge `solver="auto"` |
| Primary GPU | NVIDIA L40S 48 GB |
| CUDA process workers | `num_parallel_jobs=1` |

## ✅ CPU/CUDA agreement

The correctness grid checked percentiles 0, 50, 75, and 100 with Ridge alphas 0.01, 1, 100, and 10,000.

| Check | Result |
| --- | --- |
| Learned mapping, 1 bootstrap and 1,024 targets | All 16 CPU/CUDA pairs passed; no source-selection or finite-mask mismatches |
| Learned parameters | Maximum coefficient difference `1.36e-15`; maximum intercept difference `2.74e-16` |
| Learned predictions | Maximum train difference `4.88e-15`; maximum test difference `7.49e-16` |
| Consistency metrics, 2 bootstraps and 1,024 targets | Worst test-metric p95 absolute difference `6.78e-14`; no finite-mask mismatch |
| Percentile 0 | Selected all 27,118 sources and matched direct multi-target Ridge exactly at alpha 1 |
| Percentile 100 | Selected 1–6 tied sources per target; this was learned Ridge, not identity |
| Monolithic versus two shards | Merged raw metric arrays were numerically identical |

Across the one-bootstrap grid, the largest metric p95 absolute difference was `1.24e-13` and the largest absolute difference between global finite means was `6.46e-13`. The checks include the learned coefficients, intercepts, predictions, selected sources, finite masks, and final metrics rather than comparing only aggregate scores.

## 💾 Full-target runs

All four tested percentile shapes completed at alpha 1 with 50 bootstraps, five folds, and all 27,118 targets on an L40S.

| Percentile | External shards | Elapsed | Whole-device peak |
| ---: | ---: | ---: | ---: |
| 100 | 5 | 5.6 min | 7.0 GiB |
| 75 | 2 | 2.22 h | 20.5 GiB |
| 50 | 2 | 4.56 h | 34.5 GiB |
| 0 | 1 | 1.55 h | 19.3 GiB |

## ⏱️ Same-node CPU reference

The matched reference used percentile 0, alpha 1, one bootstrap, five folds, float64, all 27,118 targets, and one worker on the same node:

| Backend | Compute time |
| --- | ---: |
| CPU | 2,213 s |
| CUDA L40S | 112.8 s |

CUDA was 19.6 times faster for this shared all-source mapping shape. CPU peak host RSS was 57.0 GiB; the CUDA run used 47.8 GiB peak host RSS and 19.3 GiB peak device memory.

## 🌐 Sharding
To speed up latency for single comparisons, you can either shard the bootstraps/folds across multiple GPUs or the same GPU.

Two external shards routed to distinct L40S GPUs merged bitwise-identically to the same two shards run on one GPU. This validates device assignment and raw-array reconstruction, not native distributed Ridge. The [CUDA usage guide](README.md) explains the distinction between `num_parallel_jobs` and external shards.

The simplest way to do this is to split the bootstrap seeds among GPU workers or GPUs. You can do the same with the temporal folds. Then you merge the results when they all finish.

## 🧮 Float32 checks

Float32 was evaluated in four sampled percentile-0 and percentile-100 cells at alpha 0.01 and 10,000. CPU and CUDA were close to each other in those cells, but float32 was not always close to float64. At percentile 100 and alpha 10,000, the float32-versus-float64 consistency metrics had p95 absolute error about `0.0386`, maximum absolute error about `33.6`, and finite-mask mismatch about `0.00117` on both backends.

Float64 is therefore the validated recommendation. A float32 analysis should be checked against float64 over its intended percentiles and regularization strengths.

## 🚧 Coverage limits

- Full-target 50-bootstrap timing covered alpha 1 for each percentile; the other alphas were covered by the smaller correctness grid.
- Timing and shard screens were not repeated, so the reported values are reference measurements rather than performance distributions.
- A full 50-bootstrap CPU percentile-0 run was not performed; the same-node CPU comparison was limited to one bootstrap.
- Fold/Bootstrap sharding is implemented in examples but its not native to the library itself. 

The immutable [server closeout archive](https://github.com/d3tk/brainmodel_utils/tree/2e581eb557dc7ec782dec925d7412097a0913d6f/docs/cuda_server_closeout) contains the machine-readable acceptance report, manifests, comparison summaries, inventory, and checksums included with the closeout commit.
