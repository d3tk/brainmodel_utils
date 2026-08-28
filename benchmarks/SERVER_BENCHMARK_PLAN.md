# Server benchmark plan

## Purpose

The server campaign must answer four separate questions:

1. What is the fastest configuration for one full comparison?
2. What configuration completes the most independent comparisons per hour?
3. Does additional usable GPU memory improve one comparison, increase safe
   concurrency, or both?
4. What numerical precision is fast enough without changing the scientific
   result?

Do not combine single-job latency and aggregate throughput into one claim. A
32-worker CPU job may minimize latency while several smaller CPU jobs produce
more comparisons per hour. The same distinction applies to GPU batching within
one process and concurrent processes on one GPU.

The server no longer has CPU-only nodes: every allocation must request a GPU.
Treat one GPU allocation as the primary operational unit even for CPU runs.
The fairest throughput comparison therefore gives CPU packing and CUDA
concurrency the same allocation, such as one GPU plus 32 CPUs and a fixed
memory limit, and measures how many comparisons that allocation completes.

## Anchor workload and current evidence

Use the existing real-data protocol as the anchor throughout the campaign:

- DANDI `000727`, `fly_087` to `fly_089`
- right turns from the historical event source
- `IPS_L` followed by `IPS_R`, 16,636 targets
- 39 time bins and five blocked folds
- percentile 99, Ridge `alpha=1`
- 200 split-half bootstraps, starting at seed 1000

The exact command and provenance fields are documented in
[README.md](README.md#real-fly-data-protocol).

On the current i7-12700KF and RTX 3080 10 GiB host, the eight-worker CPU run
took 3:31:39.5 and the one-worker CUDA run took 37:32.0, a 5.64x speedup. The
CUDA run peaked at only 1.73 GiB of whole-device VRAM, 0.58 GiB above its
starting level. This already shows that the workload is not intrinsically
limited to one comparison on a 10 or 12 GiB device.

The implementation currently fixes correlation chunks at 512 targets and caps
the grouped-Ridge workspace at 512 MiB. A larger-memory GPU therefore cannot
automatically make one comparison faster. More memory can help only if larger
batches are faster, or if it permits useful process concurrency.

## Prerequisites

Complete these harness changes before using expensive server allocations:

1. Resolve monitored GPUs by UUID. Under `CUDA_VISIBLE_DEVICES`, compute device
   zero may be a different physical device from `nvidia-smi` device zero.
2. Expose and record the correlation target chunk and Ridge workspace budget,
   including the effective batch sizes selected at runtime.
3. Record phase timings and a utilization time series, not only peaks. Capture
   average and p95 GPU utilization, per-process and whole-device VRAM, power,
   temperature, clocks, PCIe traffic, host RSS, faults, and swap where the
   platform supports them.
4. Add a campaign launcher for repetitions and concurrent independent
   comparisons. Every child must have a unique seed range and output path.
5. Add precision artifacts and a cross-precision comparator. Summary medians
   alone cannot diagnose changed feature selection or prediction error.

The current benchmark remains suitable for a single-device smoke run while
these additions are being made, but its GPU telemetry must not be used for a
multi-GPU or remapped-device conclusion.

## Experimental controls

- Pin the fork commit, environment or container digest, input-file provenance,
  split indices, and seeds.
- Capture sockets, physical and logical cores, NUMA topology, RAM, local disk,
  GPU UUIDs, VRAM, PCIe topology, driver, clocks, and power limits before the
  first run.
- Use an exclusive node where possible. Randomize or interleave the order of
  finalist configurations so thermal and cluster-time trends do not favor one
  backend.
- Request the same GPU count for CPU and CUDA comparisons. For the primary
  throughput test, also hold requested CPUs and memory fixed; record both GPU
  allocation-hours and CPU core-hours because CPU runs leave the required GPU
  idle.
- Set `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and
  `NUMEXPR_NUM_THREADS` to `1` for process-parallel CPU runs. Pin processes to
  physical cores and record whether SMT is enabled.
- Stage the input and joblib temporary files on local NVMe. Report data
  preparation time, measured consistency time, and total scheduler wall time
  separately.
- Use one untimed warm-up before warm measurements. Measure cold invocation
  latency separately rather than mixing it into steady-state results.
- Run one repetition for screening. Run each promoted configuration three
  times; extend to five if runtime CV exceeds 5% or clocks show a thermal
  trend.

## Experiment matrix

| Question | Sweep | Primary result |
| --- | --- | --- |
| CPU single-job scaling | workers `1, 2, 4, 8, 12, 16, 25, 32`, plus socket and physical-core boundaries | latency, bootstraps/hour, speedup, parallel efficiency |
| CPU fixed-capacity throughput | one GPU allocation plus 32 CPUs, divided as `1x32`, `2x16`, `4x8`, or `8x4` | comparisons/allocation-hour and p95 job latency |
| GPU batching | correlation chunks `128, 512, 2048, 8192, all`; Ridge budgets `256, 512 MiB, 1, 2, 4, 8 GiB` | phase latency versus peak VRAM |
| One-GPU concurrency | processes `1, 2, 4, 8`, then continue adaptively on larger devices | comparisons/GPU-hour, p50/p95 latency, energy/comparison |
| Multi-GPU throughput | one process per GPU at `1, 2, 4, ...` GPUs, then best safe concurrency per GPU | throughput efficiency |
| Precision | CPU float64, CPU float32, stabilized CUDA float32, CUDA hybrid float64, true CUDA float64 | paired numerical error and runtime |

### CPU scaling

Use three stages to avoid spending many hours on configurations that cannot be
competitive:

1. Run five bootstraps and 128 targets at every worker count to validate
   affinity, memory reporting, and output identity.
2. Run 64 bootstraps with all targets at every worker count as a memory and
   scaling screen. Treat this only as a screen because task granularity differs
   by worker count.
3. Run the full 200 bootstraps at workers `8, 12, 16, 25, 32` and any additional
   server-specific winner. The 25- and 32-worker points are mandatory for the
   stated question.

Report wall time, bootstraps/hour, CPU-core-hours, average CPU utilization,
peak RSS, major faults, and swap. Use rate as the primary scaling statistic if
a full one-worker reference would be prohibitively long. For 200 independent
tasks, 25 workers receive exactly eight tasks each; 32 workers receive six or
seven, so scheduling granularity alone caps ideal 32-worker occupancy at
`200 / (32 * 7) = 89.3%`.

The local eight-worker run used 40.33 GiB peak RSS, including a 38.70 GiB
increase. A linear planning estimate is about 123 GiB at 25 workers and 156 GiB
at 32 workers. With 20% headroom, reserve approximately 153 GiB and 196 GiB,
respectively, until server measurements replace this estimate. Increase worker
count gradually and stop before 80% RAM, any swap, or an OOM.

For throughput, give all configurations one required GPU allocation, the same
32-core budget, and enough comparisons to keep it occupied: one 32-worker job,
two 16-worker jobs, four eight-worker jobs, and eight four-worker jobs. Compare
makespan for the same set of comparisons. Stop testing larger single jobs
after two consecutive throughput gains below 5%, or parallel efficiency falls
below 30%, after the mandatory 25- and 32-worker results are complete.

### GPU memory and batching

First profile one or two bootstraps to divide time among correlation and source
selection, Ridge, prediction, CPU metrics, and transfers. On one GPU, sweep the
correlation chunk while holding the Ridge budget fixed, then sweep the Ridge
budget using the best correlation setting. Test only the best combinations
with 20 bootstraps and all targets, then promote Pareto-optimal settings to the
full 200-bootstrap run.

Stop increasing a batch after two latency improvements below 3%, at 80--85% of
VRAM, or on OOM. Extra memory improves single-job latency only if a setting that
needs the extra memory is at least 10% faster in repeated full runs and still
passes the precision gates.

A comparison between a 12 GiB and a 32 GiB card does not isolate memory: GPU
architecture, bandwidth, FP32 and FP64 throughput, clocks, and PCIe also
change. Use tunable memory budgets on the same card to test the memory
hypothesis. Report cross-card measurements as overall hardware scaling.

### Concurrent comparisons

Run each CUDA comparison in its own process with `--workers 1`. Screen
concurrency `1, 2, 4, 8` using 20 bootstraps and all targets; on a 32 GiB device,
continue to `12` and `16` only while memory and throughput justify it. Confirm
concurrency one and the best safe value with the full workload. CUDA MPS is an
optional follow-up, not part of the baseline.

Measure aggregate comparisons/hour, makespan, median and p95 per-job latency,
fairness, average and p95 utilization, per-process and whole-device VRAM, host
RSS, power, and energy/comparison. Stop when device or host memory exceeds 85%,
an OOM, Xid, or throttling event occurs, p95 latency exceeds twice the
single-process latency, or two concurrency increments each add less than 10%
throughput.

For homogeneous multi-GPU servers, begin with one process per GPU. Then apply
the best safe single-GPU concurrency to every device. Independent comparisons
should be the default unit of distribution. Splitting one comparison across
GPUs is a later latency optimization; it requires a shard-and-merge format
because averaging the existing JSON medians would be incorrect.

## Precision and equivalence

Use one immutable float64 input fixture and identical splits and bootstrap seeds
for every profile. Cast at the mapper boundary to isolate compute precision;
also run a separate end-to-end dtype test because the current `--dtype` option
changes data preparation as well as mapping.

Test these profiles in order:

1. CPU float64 reference.
2. CPU float32.
3. CUDA float32 with the current stabilization and any adaptive Ridge
   promotion recorded.
4. Current CUDA float64, explicitly labeled hybrid because percentile
   correlation is still float32.
5. True CUDA float64 correlation and Ridge as a diagnostic implementation.
6. TF32 or mixed precision only if earlier tests expose a meaningful speed or
   capacity opportunity. Do not use pure float16 normal-equation Ridge as a
   production candidate.

Start with adversarial unit fixtures covering near-collinear sources, constant
columns, large offsets, small variance, ties around the percentile cutoff, and
several condition numbers. Then run 20 bootstraps on 1,024 stratified real
targets, followed by one full 200-bootstrap comparison for every viable
profile. Repeat one identical CUDA case three times to test determinism.

Persist enough paired data to compare correlation error, threshold margins,
selected-source counts and Jaccard agreement, predictions, finite masks, and
all seven train/test consistency metrics. Coefficient differences are
diagnostic but not a primary gate because correlated features can change
coefficients without changing predictions.

Pre-register these production gates:

- maximum absolute difference in every canonical consistency metric at most
  `0.005`;
- absolute difference in the global finite mean of test `r_xy_n_sb` at most
  `0.001`;
- finite-mask mismatch fraction at most `1e-5`;
- any source-selection disagreement must be reported against its distance from
  the percentile cutoff.

Report paired target-level error quantiles and sign changes even when the gates
pass. Compare numerical error with between-bootstrap sampling variability so a
small backend delta is not confused with biological uncertainty. Promote TF32
or mixed precision only if it passes the same scientific gates and improves
latency or safe capacity by at least 10%.

## Secondary tests

Run these only after the anchor decisions are stable:

- target counts `128, 1,024, 4,096, 16,636` and bootstrap counts `10, 50, 200`
  to find the CPU/GPU crossover;
- percentile `95, 99, 99.5` and Ridge `alpha=0.1, 1, 10` to vary feature count
  and numerical conditioning;
- one-socket versus whole-node NUMA placement and physical cores versus SMT;
- sustained runs for allocator growth, thermal throttling, and failure
  recovery;
- two or three additional fly-pair, direction, or region comparisons before
  claiming generality;
- energy and server-cost per comparison when power and allocation prices are
  available.

## Decision report

The campaign ends with three recommendations, all conditional on the numerical
gates:

1. the lowest-latency configuration for one full comparison;
2. the highest-throughput configuration for a queue of comparisons;
3. the lowest energy or cost per comparison.

For the primary CPU-versus-GPU claim, compare the median full-workload CUDA
latency with the best full-workload CPU latency, including the mandatory 25-
and 32-worker CPU results. For throughput and cost, compare configurations
within the same required GPU allocation and report GPU allocation-hours as
well as CPU core-hours. Preserve every raw JSON result, a run manifest,
resource inventory, scheduler job ID, environment lock, and an aggregate table
used to make the recommendation.
