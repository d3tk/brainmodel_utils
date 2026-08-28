# Full-size IPS benchmark analysis

## Scope

Pulled from `fork/cuda` at commit `b84707e`. The measurements themselves were
produced from commit `978603a` with a dirty working tree.

The three runs use the same recorded workload and provenance:

- `fly_100` to `fly_099`, right turns, IPS
- 624 source events and 571 target events
- 30 time bins from -15.0 to 0.4 seconds
- all 27,118 source and target voxels
- 200 bootstrap seeds, 1000 through 1199
- five blocked folds
- percentile 99 and Ridge `alpha=1`
- float32 source and target arrays

This is a full-size IPS-CV workload, but it is not the same biological pair or
shape as the local `fly_087` to `fly_089` benchmark. CPU and CUDA comparisons
within this server campaign are matched; direct speed comparisons with the
local RTX 3080 run are not.

## Headline result

For one comparison, 32 CPUs are fastest. One L40S beats an eight-CPU packing,
but does not beat 32 CPUs.

| Configuration | Consistency wall | Comparisons/hour | Relative result |
| --- | ---: | ---: | --- |
| CPU, 8 workers | 9,607.40 s (2:40:07) | 0.3747 | baseline |
| L40S CUDA, 1 process | 4,466.13 s (1:14:26) | 0.8061 | 2.151x faster than CPU-8 |
| CPU, 32 workers | 2,756.16 s (0:45:56) | 1.3062 | 1.620x faster than CUDA |

Data preparation took 6.16, 6.28, and 6.53 seconds, respectively, so including
preparation does not change the ordering.

The results answer single-comparison latency only. They do not measure a fixed
node's throughput when it runs several smaller CPU jobs or several CUDA jobs
concurrently.

There are no CPU-only nodes on this server, so all three configurations
necessarily occupied one GPU allocation. Under that operational constraint,
CPU-32 currently also uses the least GPU allocation time: 0.766 GPU-hours per
comparison, versus 1.241 for CUDA and 2.669 for CPU-8. CPU-32 requests more CPU
and memory, so a complete cost comparison must include those resources too.

## CPU scaling

Increasing workers from 8 to 32 produced a 3.4858x speedup for a 4x worker
increase, or 87.14% strong-scaling efficiency. Latency fell by 71.31%.

| Resource measure | CPU-8 | CPU-32 |
| --- | ---: | ---: |
| Mean active cores | 7.90 | 28.63 |
| Allocated-core saturation | 98.77% | 89.46% |
| Process CPU time | 21.09 CPU-hours | 21.92 CPU-hours |
| Allocated core-hours | 21.35 | 24.50 |
| Summed process-tree peak RSS | 103.48 GiB | 401.48 GiB |

The scaling is close to the workload's scheduling ceiling. Eight workers run
25 bootstrap waves; 32 workers require seven waves, with only eight tasks in
the last wave. The largest possible speedup from this task packing is
`25 / 7 = 3.571x`; the observed speedup reaches 97.6% of that ceiling.

CPU-32 therefore buys lower latency at 14.75% more allocated core-hours. CPU-8
is more allocation-efficient. The RSS values sum every process's resident set
and can double-count shared pages; CPU-32 reports more summed RSS than its 384G
Slurm request. Use cgroup memory or `sacct MaxRSS` before turning these values
into node-packing limits.

The requested 25-worker point was not run. Since 200 divides evenly by 25, an
equal-task extrapolation from the two measurements suggests approximately
51--53 minutes, probably faster than the observed CUDA run. This is a
prediction, not a measurement.

## GPU utilization and memory

The L40S was neither memory-bound nor saturated over the full run.

| GPU measurement | Value |
| --- | ---: |
| Total device memory | 44.99 GiB usable |
| Peak whole-device memory | 2.13 GiB (4.74%) |
| Peak process memory | 1.56 GiB |
| Whole-device increase | 1.14 GiB |
| Mean utilization | 18.58% |
| p95 / peak utilization | 69% / 76% |
| Mean / p95 power | 103.1 W / 121.8 W |
| Power limit | 350 W |
| Peak temperature | 32 C |

The single process would fit comfortably on a 12 GiB card. More VRAM alone
would not accelerate the current default configuration. The large unused
capacity is evidence for testing concurrent comparisons and larger resident
batches, not evidence that either will scale linearly.

The approximate measured GPU energy was 0.128 kWh, using mean device power
times wall time. This excludes the host and cannot be compared with CPU energy
because CPU or node power was not recorded.

### Phase breakdown

| Phase | Time | Fraction of CUDA wall |
| --- | ---: | ---: |
| Ridge | 2,715.48 s (45:15) | 60.80% |
| Correlation and source selection | 448.25 s (7:28) | 10.04% |
| CPU metrics | 14.66 s | 0.33% |
| Uninstrumented | 1,287.73 s (21:28) | 28.83% |

Correlation-only optimization has an absolute best-case speedup of about
1.11x, even if that phase became free. It cannot close the 1,710-second gap to
CPU-32. Ridge and the uninstrumented work are the important targets.

The Ridge phase includes CPU construction of padded target batches, host-to-
device transfer, batched solves, device-to-host transfer, and model creation;
it is not pure GPU kernel time. The uninstrumented portion includes split-half
work, predictions, and orchestration.

Each bootstrap runs five folds and creates three fitted mappers per fold, for
3,000 mapper fits. With the default 512-target correlation chunk, each fit
uses 53 correlation chunks. Ridge uses a 512 MiB workspace and is capped at
1,024 targets per batch; its effective consistency batch was not persisted in
the JSON. This fragmentation and repeated transfer pattern are consistent
with the low mean utilization and bursty p95 utilization.

The telemetry reports mean PCIe receive and transmit rates of approximately
784,000 and 186,000 KiB/s. These are sampled rates, not byte counters, but they
support investigating repeated transfers and synchronization before adding
more arithmetic kernels.

## Numerical agreement

CPU-8 and CPU-32 metric summaries are byte-for-byte identical. CUDA aggregate
agreement with CPU is also strong:

| Comparison | CUDA minus CPU |
| --- | ---: |
| Global finite mean, test `r_xy_n_sb` | +0.000259315 |
| Median across fold-specific target medians | -0.000000764 |
| Global finite median, test `r_xy_n_sb` | +0.000004235 |
| Largest absolute difference among 70 fold aggregates | 0.000009898 |

The observed aggregate differences satisfy the planned `0.005` canonical
threshold and `0.001` primary global-mean threshold by wide margins. CUDA has
nine additional finite test cells out of 27,118,000 and two additional finite
train cells.

This is strong summary-level float32 parity, but not a complete precision
validation. The files do not contain raw metric arrays, finite masks, selected
sources, predictions, coefficients, or precision artifacts. A net difference
of nine finite cells does not reveal the true mask mismatch, and no float64
reference was run. Formal interchangeability and float32 error relative to
float64 therefore remain untested.

## Validity and reproducibility limits

- There is one timing run per configuration and no confidence interval.
- The jobs ran on three different nodes. CPU models match, but node variation
  is not estimated.
- CPU used NumPy/scikit-learn/SciPy 2.0.2/1.8.0/1.17.1; CUDA used
  2.2.6/1.7.2/1.18.0. Backend and environment differences are confounded.
- Every result records commit `978603a` with `dirty=true`; the dirty diff or
  hash was not saved.
- The shared manifest was overwritten and now records only the CPU-32 command.
- Every job must request a GPU because the server has no CPU-only nodes. CPU
  runs therefore consume GPU allocation time even though they leave the device
  idle; this is an operational constraint, not a benchmark mistake.
- No 25-worker CPU, fixed-32-core multi-job, CUDA batch, CUDA concurrency,
  float64, or determinism sweep is present.

## Supported decisions

1. **Lowest observed single-comparison latency:** CPU-32, at 45:56.
2. **When only eight CPUs are assigned:** one L40S is 2.15x faster.
3. **GPU memory:** the default single job is not VRAM-limited; 32--48 GiB is
   mainly a concurrency or batching opportunity.
4. **Accuracy:** aggregate CPU/CUDA float32 agreement is excellent for this fly
   pair, but the raw precision gates and float64 comparison remain open.
5. **Required GPU-slot time:** CPU-32 currently wins at 0.766 GPU-hours per
   comparison. Highest queue throughput and total resource cost are still not
   determined.

## Minimum next campaign

1. In one allocation containing one GPU, 32 CPUs, and fixed memory, compare
   CUDA concurrency `1, 2, 4, 6` with CPU packing `1x32`, `2x16`, and `4x8`.
   Use 20 bootstraps for screening and confirm each winner at 200.
2. Run CPU-25 for the missing single-comparison latency and efficiency point.
3. Tune Ridge batching and transfer residency before prioritizing correlation
   chunks. Record the effective Ridge batch in consistency output.
4. Run the precision subset from a clean commit and one locked environment,
   saving raw metrics, masks, selections, and predictions for CPU-float64,
   CPU-float32, CUDA-float32, and true CUDA-float64.
5. Repeat the promoted CPU and CUDA configurations three times on controlled
   nodes before making a production timing claim.

## Result files

- [CPU, 8 workers](ips_cv_f100_t099_right_b200_r0_j0_cpu_w8.json)
- [CPU, 32 workers](ips_cv_f100_t099_right_b200_r0_j0_cpu_w32.json)
- [CUDA, one L40S process](ips_cv_f100_t099_right_b200_r0_j0_cuda_w1.json)
- [Campaign summary](SCREEN.md)
