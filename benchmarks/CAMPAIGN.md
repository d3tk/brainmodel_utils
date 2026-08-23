# Babel campaign tracker

The written protocol in the original plan used `fly_087` → `fly_089`, 39 bins, and 16,636 `IPS_L`+`IPS_R` units. On Babel the production IPS nested-CV comparison is `fly_100` → `fly_099`, right turns, native 30-bin −15 to 0.4 s events (624 / 571), and all 27,118 IPS voxels. That is the campaign anchor here. The local 5.64× CUDA number remains local RTX 3080 evidence, not a Babel result.

Every job still requests `--gres=gpu:1` on `general`. Do not use `--exclusive`: these nodes have 8–10 GPUs. Do not fold one-job latency into comparisons per allocation-hour.

## Four questions

1. Fastest one full comparison: CPU-32 46:12, CUDA-1 1:14:47, CPU-8 2:40:21 on the 200-bootstrap IPS-CV anchor. Missing CPU 12/16/25 and a tuned CUDA batch setting.
2. Most independent comparisons per GPU-hour: not measured. Needs a fixed 32-CPU + 1-GPU allocation running `1×32`, `2×16`, `4×8`, `8×4` (CPU) and CUDA process concurrency.
3. Extra GPU memory: the finished CUDA-1 job peaked at 2.13 GiB of 48 GiB, mean SM util 19%. Ridge was ~61% of wall time. The implementation capped Ridge batches at 1024 targets, so a larger workspace could not help one comparison. Stage 1 lifts that cap and sweeps correlation chunks and Ridge workspaces on 2 bootstraps.
4. Precision: CPU vs CUDA medians already agree to ~1e-6 on the full 200-bootstrap run, but that is not the gated precision matrix. Stage 1 runs 20 bootstraps on 1,024 targets for CPU float64, CPU float32, CUDA hybrid float32, and true CUDA float64.

## Stage 1 jobs

| Set | What | Protocol |
|---|---|---|
| GPU batch sweep | 1-boot cold, then 2-boot correlation `{128,512,2048,8192,all}` and Ridge `{256MiB…8GiB}` with max batch `all` | all IPS voxels |
| CPU 64-boot | workers `{1,2,4,12,16,25}` | all IPS voxels |
| Precision | 20 boots, 1,024 targets, `dtype=float64`, compute/correlation casts | first 1,024 targets |

Promote only after screens: CUDA 20-bootstrap A/B (default vs Ridge-all 4 GiB), then 200-bootstrap optimized CUDA if that gap holds. CPU float64 precision needs ≥64 GiB. Repeat CPU32 and optimized CUDA three times before throughput packing (`1×32` / `2×16` / `4×8` vs CUDA concurrency).

## Stage 2 jobs

CPU 16 workers won the 64-bootstrap screen but did not replace CPU-32 at 200 bootstraps (45:56). CUDA 2-bootstrap winner: Ridge batch=`all`, 4 GiB workspace, default correlation chunk. Correlation=`all` used 14.5 GiB without a real speedup.

| Job | What |
|---|---|
| `10192036` | CPU 64-boot 1-worker (leave running) |
| `10193368` | CUDA 20-boot default |
| `10193369` | CUDA 20-boot Ridge-all + 4 GiB, correlation 512 |
| `10193370` | CPU float64 precision, 96 GiB, 1,024 targets, metric dump |

Do not submit 200-boot CUDA, three-fold repeats, or throughput until the 20-boot A/B is in.

## Throughput screen (64 bootstraps)

Identical allocation each time: 32 CPUs, 1 GPU, 448 GiB. 64 bootstraps per comparison. Campaign `--jobs` / `--max-concurrent` runs concurrent comparisons with unique seeds.

| Config | Status |
|---|---|
| CPU 1×32 / 2×16 / 4×8 | 64-boot screen |
| CUDA 1× / 2× / 4× / 8× | 64-boot, Ridge-all 4 GiB, workers=1 |

The 20-boot packing jobs were cancelled. 200-boot packing only for the CPU and CUDA throughput winners.
