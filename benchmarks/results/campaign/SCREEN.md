# Babel `general` screen (2026-08-21)

Campaign status is in [CAMPAIGN.md](../../CAMPAIGN.md). The four questions stay separate: one-job latency is not comparisons per GPU-hour. Babel uses the IPS nested-CV packing (`fly_100` → `fly_099`, 27,118 IPS voxels, 30 bins). The written `fly_087` 39-bin protocol is local evidence only.

Stage 1 (submitted after the 200-bootstrap trio): 2-bootstrap CUDA batching with the Ridge 1024-target cap lifted, CPU 64-bootstrap workers `{1,2,4,12,16,25}`, and a 20-bootstrap 1,024-target precision screen. The finished CUDA-1 job used 2.13 GiB of 48 GiB at 19% mean SM util; extra VRAM cannot help one comparison until that cap is removed.

Tiny jobs below are **not** full IPS comparisons. They are 5-bootstrap, 128-target slices used to test packing, event loading, and CUDA bring-up.

## Full IPS-CV-matched comparison (2026-08-22)

fly_100 → fly_099, right turns, all IPS voxels, native 30-bin grid (−15 to 0.4 s), 624 / 571 events, 200 bootstraps × 5 blocked folds × 27,118 targets, percentile 99, Ridge α=1. Same mapping as autonomous-drosophila IPS CV (not the 128-target screen).

| | CPU 8 `10190371` | CUDA 1 `10190372` | CPU 32 `10190370` |
|---|---:|---:|---:|
| Consistency wall | 2:40:07 | 1:14:26 | 0:45:56 |
| Peak RSS | 103 GiB | 11 GiB host / 2.13 GiB GPU | 401 GiB |
| mean test `r_xy_n_sb` | 0.543274 | 0.543533 | 0.543274 |
| median test `r_xy_n_sb` | 0.355993 | 0.355992 | 0.355993 |
| Finite fraction | 0.591 | 0.591 | 0.591 |
| Comparisons / hour (one job) | 0.37 | 0.81 | 1.31 |
| GPU | n/a | L40S, mean util 19%, peak 76% | n/a |

JSON: `ips_cv_f100_t099_right_b200_r0_j0_{cpu_w8,cuda_w1,cpu_w32}.json`. CPU-8 and CPU-32 scores match exactly. CUDA medians agree to ~1e-6. These are **one-comparison latency** numbers, not independent comparisons per node-hour. CUDA ridge 45.3 min, correlation 7.5 min.

CPU-8 is the packing closest to the ~2 h IPS CV jobs. CUDA is 2.15× faster than that packing for this one comparison; CPU-32 is 3.49× faster than CPU-8 (4× workers).

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
