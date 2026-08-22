# Babel `general` screen (2026-08-21)

These jobs are **not** full IPS comparisons. They are 5-bootstrap, 128-target slices used to test packing, event loading, and CUDA bring-up.

## Why this is seconds, not ~2 hours

IPS nested CV (`autonomous-drosophila`) at percentile 99 maps **all IPS voxels** (~27k targets) with **200 bootstraps** and 5 blocked folds. On Babel that is the 2-hour class of work: 1.77 h at 32 CPUs, 7.16 h at 12 CPUs, for one fly pair × direction.

This screen used the same mapping family (percentile 99, Ridge α=1, 5 blocked folds) but:

| | IPS CV “one comparison” | This screen |
|---|---|---|
| Targets | ~16.6k–27k IPS voxels | **128** |
| Bootstraps | 200 | **5** |
| Target events | typically hundreds | **36** |
| Wall (CPU) | ~2 h | **6–22 s** |

Work scales roughly with `bootstraps × targets × folds × event load`. 128/16636 × 5/200 ≈ **1/5200** of a full 8-worker protocol. 5.8 s × 5200 ≈ 8 h, the same order as the local 8-worker full run (3.5 h) and IPS CV.

The local full protocol (16,636 targets, 200 bootstraps, 8 CPU workers) took **3:31:39**. That is the apples-to-apples number, not the 6-second slice.

## Protocol-matched 39-bin pair (`-15` to `5.216` s)

Events: 535 source / 36 target right turns from IPS-CV npz files (historical CSVs are not on the cluster DANDIet).

| | CPU 8 workers `10186694` | CUDA 1 worker `10186695` |
|---|---:|---:|
| Consistency wall | 5.80 s | 40.4 s |
| Prep | 5.56 s | 5.55 s |
| Peak RSS | 29.9 GiB | 7.0 GiB |
| mean test `r_xy_n_sb` | 0.066692 | 0.066690 |
| Finite fraction | 0.493 | 0.493 |
| GPU | n/a | L40, 1.32 GiB, mean util 0.4% |

Test-metric medians agree to ~0. CUDA is slower on this tiny slice (setup dominates). That does not contradict the local full-protocol 5.64× CUDA win.

## 30-bin IPS-CV grid CPU packing (not protocol-matched)

| Workers | Wall | Peak RSS | mean test `r_xy_n_sb` |
|---:|---:|---:|---:|
| 1 | 22.4 s | 4.3 GiB | 0.357 |
| 4 | 13.5 s | 18.9 GiB | 0.357 |
| 8 | 9.2 s | 23.3 GiB | 0.349 |
| 16 | 8.5 s | 23.6 GiB | 0.357 |

Worker 2 used a mixed 39-bin grid; ignore it for scaling.
