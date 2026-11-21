"""Runtime and peak-memory benchmark: reference vs. optimized consistency.

Each measurement runs in a fresh subprocess so peak RSS is not polluted by
earlier runs. Reported per run:

* wall_s          wall-clock time of the consistency call only (data generation
                  and imports excluded)
* fits            regression fits performed (counted for every variant)
* peak_rss_mb     process high-water mark (ru_maxrss; bytes on macOS, KiB on
                  Linux, converted here)
* tree_peak_mb    sampled peak RSS of the process plus its children (covers
                  joblib/loky workers when --jobs > 1)
* cpu_s           user+system CPU seconds of the process tree during the call
* cores_busy      cpu_s / wall_s (average CPU cores in use)
* delta_peak_mb   peak_rss_mb minus RSS just before the call (input data and
                  library footprint removed)
* max_abs_diff /
  bitwise         against the reference run of the same tier and seeds, when
                  the reference variant is part of the same invocation

Thread counts for BLAS/OpenMP are pinned to 1 in every worker.

Examples:
    python benchmarks/bench_consistency.py --tier small --boots 3 --repeats 3
    python benchmarks/bench_consistency.py --tier memory --boots 1 \
        --variants reference cached batched --batch-sizes 64 256 1024
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
REPO = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO / "benchmarks" / "results"

# Workload tiers (Phase 5). max_iter=None keeps the production 10000; larger
# tiers lower it so local runs finish (memory and fit counts do not depend on
# it, runtime per fit does).
TIERS = {
    "small": dict(n_source_voxels=128, n_target_voxels=128, trials=40, max_iter=None),
    "medium": dict(n_source_voxels=1024, n_target_voxels=1024, trials=40, max_iter=50),
    "wide": dict(n_source_voxels=27118, n_target_voxels=64, trials=20, max_iter=50),
    "memory": dict(n_source_voxels=27118, n_target_voxels=4096, trials=20, max_iter=5),
}
VARIANTS = ("reference", "vectorized", "cache_only", "cached", "batched")


# ---------------------------------------------------------------------------
# worker side
# ---------------------------------------------------------------------------
def _rss_bytes():
    import psutil

    return psutil.Process().memory_info().rss


def _maxrss_bytes():
    import resource

    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return v if sys.platform == "darwin" else v * 1024


class TreeSampler(threading.Thread):
    def __init__(self, interval=0.05):
        super().__init__(daemon=True)
        import psutil

        self.proc = psutil.Process()
        self.interval = interval
        self.peak = 0
        self.cpu = {}  # pid -> last seen user+system CPU seconds
        self.cpu_start = self._cpu_now()
        self._stop_event = threading.Event()

    def _cpu_now(self):
        import psutil

        seen = {}
        try:
            procs = [self.proc] + self.proc.children(recursive=True)
        except psutil.Error:
            procs = [self.proc]
        for p in procs:
            try:
                t = p.cpu_times()
                seen[p.pid] = t.user + t.system
            except psutil.Error:
                pass
        return seen

    def cpu_seconds(self):
        """CPU time used by the process tree since the sampler was created.

        Children that exit between samples lose at most one interval of CPU.
        """
        total = 0.0
        for pid, t in self.cpu.items():
            total += t - self.cpu_start.get(pid, 0.0)
        return total

    def run(self):
        import psutil

        while not self._stop_event.is_set():
            total = 0
            try:
                procs = [self.proc] + self.proc.children(recursive=True)
                for p in procs:
                    try:
                        total += p.memory_info().rss
                    except psutil.Error:
                        pass
            except psutil.Error:
                pass
            self.peak = max(self.peak, total)
            self.cpu.update(self._cpu_now())
            time.sleep(self.interval)

    def stop(self):
        self._stop_event.set()
        self.join()
        self.cpu.update(self._cpu_now())


def worker(spec):
    import warnings

    import numpy as np

    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(REPO / "test"))
    import synthetic_consistency_data as sd
    from brainmodel_utils.metrics import consistency_fast as F
    from brainmodel_utils.metrics import consistency_reference as R
    from brainmodel_utils.neural_mappers import PipelineNeuralMap

    tier = TIERS[spec["tier"]]
    source, target = sd.make_fly_pair(
        n_source_trials=tier["trials"],
        n_target_trials=tier["trials"],
        n_source_voxels=tier["n_source_voxels"],
        n_target_voxels=tier["n_target_voxels"],
    )
    max_iter = spec.get("max_iter") or tier["max_iter"] or sd.ENET_MAX_ITER
    kw = dict(
        source=source,
        target=target,
        map_kwargs=sd.enet_map_kwargs(max_iter=max_iter),
        num_bootstrap_iters=spec["boots"],
        num_parallel_jobs=spec["jobs"],
        start_seed=42,
        splits=sd.make_blocked_splits(),
    )

    fits = {"n": 0}
    original_fit = PipelineNeuralMap.fit

    def counting_fit(self, *a, **k):
        fits["n"] += 1
        return original_fit(self, *a, **k)

    PipelineNeuralMap.fit = counting_fit

    variant = spec["variant"]
    if variant == "reference":
        call = lambda: R.get_linregress_consistency(**kw)  # noqa: E731
    else:
        opts = {
            "vectorized": dict(vectorize=True, cache_full=False, target_batch_size=0),
            "cache_only": dict(vectorize=False, cache_full=True, target_batch_size=0),
            "cached": dict(vectorize=True, cache_full=True, target_batch_size=0),
            "batched": dict(vectorize=True, cache_full=True, target_batch_size=spec["batch"]),
        }[variant]
        call = lambda: F.compute_linregress_consistency(**kw, **opts)  # noqa: E731

    import psutil

    swap_before = psutil.swap_memory().used
    rss_before = _rss_bytes()
    sampler = TreeSampler()
    sampler.start()
    t0 = time.perf_counter()
    results = call()
    wall = time.perf_counter() - t0
    sampler.stop()
    out = {
        **spec,
        "max_iter": max_iter,
        "wall_s": wall,
        "fits": fits["n"] if spec["jobs"] == 1 else None,
        "rss_before_mb": rss_before / 2**20,
        "peak_rss_mb": _maxrss_bytes() / 2**20,
        "tree_peak_mb": sampler.peak / 2**20,
        "swap_delta_mb": (psutil.swap_memory().used - swap_before) / 2**20,
        "cpu_s": sampler.cpu_seconds(),
    }
    out["cores_busy"] = out["cpu_s"] / wall
    out["delta_peak_mb"] = out["peak_rss_mb"] - out["rss_before_mb"]
    if spec.get("save"):
        np.savez(spec["save"], **{f"{p}/{k}": np.asarray(v) for p, d in results.items() for k, v in d.items()})
    print("RESULT " + json.dumps(out), flush=True)


# ---------------------------------------------------------------------------
# parent side
# ---------------------------------------------------------------------------
def run_one(spec):
    pythonpath = os.pathsep.join(filter(None, [str(REPO), os.environ.get("PYTHONPATH")]))
    env = {**os.environ, **{v: "1" for v in THREAD_VARS}, "PYTHONWARNINGS": "ignore",
           "PYTHONPATH": pythonpath}
    proc = subprocess.run(
        [sys.executable, __file__, "--worker", json.dumps(spec)],
        env=env, capture_output=True, text=True,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])
    raise RuntimeError(f"worker failed:\n{proc.stdout}\n{proc.stderr}")


def compare(ref_path, other_path):
    import numpy as np

    with np.load(ref_path) as a, np.load(other_path) as b:
        assert sorted(a.files) == sorted(b.files)
        bitwise, worst = True, 0.0
        for k in a.files:
            x, y = a[k], b[k]
            if x.shape != y.shape or x.dtype != y.dtype or not np.array_equal(np.isnan(x), np.isnan(y)):
                return False, float("inf")
            if not np.array_equal(x, y, equal_nan=True):
                bitwise = False
            fin = np.isfinite(x)
            if fin.any():
                worst = max(worst, float(np.max(np.abs(x[fin].astype(float) - y[fin]))))
        return bitwise, worst


def environment():
    import numpy, scipy, sklearn, joblib, threadpoolctl  # noqa: E401

    return {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "joblib": joblib.__version__,
        "threadpoolctl": threadpoolctl.__version__,
        "cpu_count": os.cpu_count(),
        "git": subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip(),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker")
    ap.add_argument("--tier", choices=list(TIERS), default="small")
    ap.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[64, 128, 256, 512, 1024])
    ap.add_argument("--boots", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--max-iter", type=int, default=None, help="override the tier's max_iter")
    ap.add_argument("--out", type=Path, default=None, help="JSON lines output file")
    args = ap.parse_args()
    if args.worker:
        worker(json.loads(args.worker))
        return

    out = args.out or RESULTS_DIR / f"{args.tier}_b{args.boots}_j{args.jobs}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    env = environment()
    tmp = Path(tempfile.mkdtemp(prefix="bench_consistency_"))
    specs = []
    for variant in args.variants:
        batches = args.batch_sizes if variant == "batched" else [0]
        for batch in batches:
            specs.append(dict(tier=args.tier, variant=variant, batch=batch, boots=args.boots,
                              jobs=args.jobs, max_iter=args.max_iter))

    ref_file = None
    rows = []
    with out.open("a") as fh:
        for spec in specs:
            runs = []
            for rep in range(args.repeats):
                save = tmp / f"{spec['variant']}_{spec['batch']}.npz" if rep == 0 else None
                res = run_one({**spec, "repeat": rep, "save": str(save) if save else None})
                if save and spec["variant"] == "reference":
                    ref_file = save
                if save and ref_file and spec["variant"] != "reference":
                    res["bitwise"], res["max_abs_diff"] = compare(ref_file, save)
                res["env"] = env
                fh.write(json.dumps(res) + "\n")
                fh.flush()
                runs.append(res)
            row = {
                "variant": spec["variant"] + (f"-{spec['batch']}" if spec["batch"] else ""),
                "median_wall_s": statistics.median(r["wall_s"] for r in runs),
                "fits": runs[0]["fits"],
                "peak_rss_mb": statistics.median(r["peak_rss_mb"] for r in runs),
                "delta_peak_mb": statistics.median(r["delta_peak_mb"] for r in runs),
                "tree_peak_mb": statistics.median(r["tree_peak_mb"] for r in runs),
                "median_cpu_s": statistics.median(r["cpu_s"] for r in runs),
                "cores_busy": statistics.median(r["cores_busy"] for r in runs),
                "bitwise": runs[0].get("bitwise"),
                "max_abs_diff": runs[0].get("max_abs_diff"),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)

    base = next((r for r in rows if r["variant"] == "reference"), None)
    print(f"\nTier {args.tier}: {TIERS[args.tier]} boots={args.boots} jobs={args.jobs} repeats={args.repeats}")
    print(f"{'variant':16s} {'wall_s':>9s} {'speedup':>8s} {'cpu_s':>9s} {'cores':>6s} {'fits':>6s} {'peakRSS':>9s} {'Δpeak':>9s} {'tree':>9s}  equivalence")
    for r in rows:
        speed = f"{base['median_wall_s'] / r['median_wall_s']:.2f}x" if base else "-"
        eq = "reference" if r["variant"] == "reference" else (
            "bitwise" if r["bitwise"] else f"max|Δ|={r['max_abs_diff']:.2e}" if r["bitwise"] is not None else "-")
        print(f"{r['variant']:16s} {r['median_wall_s']:9.2f} {speed:>8s} {r['median_cpu_s']:9.2f} {r['cores_busy']:6.2f} {str(r['fits']):>6s} "
              f"{r['peak_rss_mb']:8.0f}M {r['delta_peak_mb']:8.0f}M {r['tree_peak_mb']:8.0f}M  {eq}")
    print(f"\nraw results appended to {out}")


if __name__ == "__main__":
    main()
