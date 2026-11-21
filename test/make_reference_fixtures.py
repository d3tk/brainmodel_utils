"""Run the unmodified reference implementation and save its complete outputs.

    python test/make_reference_fixtures.py --boots 3 10 50

Writes test/fixtures/reference_primary_b{N}.npz holding every metric for both
the train and test partitions, plus the software versions and wall time.
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import synthetic_consistency_data as sd  # noqa: E402

FIXTURE_DIR = HERE / "fixtures"
START_SEED = 42


def fixture_path(boots):
    return FIXTURE_DIR / f"reference_primary_b{boots}.npz"


def environment():
    import scipy
    import sklearn

    return {
        "machine": platform.machine(),
        "system": platform.system(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
    }


def run_kwargs(boots):
    source, target = sd.primary()
    return dict(
        source=source,
        target=target,
        map_kwargs=sd.enet_map_kwargs(),
        num_bootstrap_iters=boots,
        num_parallel_jobs=1,
        start_seed=START_SEED,
        metric="pearsonr",
        splits=sd.make_blocked_splits(),
    )


def save(results, path, meta):
    arrays = {f"{p}/{k}": np.asarray(v) for p, d in results.items() for k, v in d.items()}
    arrays["__meta__"] = np.array(json.dumps(meta))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load(path):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["__meta__"]))
        results = {"train": {}, "test": {}}
        for key in meta["keys"]:
            p, k = key.split("/")
            results[p][k] = data[key]
    return results, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boots", type=int, nargs="+", default=[3])
    args = parser.parse_args()
    from brainmodel_utils.metrics.consistency_reference import get_linregress_consistency

    for boots in args.boots:
        t0 = time.perf_counter()
        results = get_linregress_consistency(**run_kwargs(boots))
        wall = time.perf_counter() - t0
        meta = {
            "boots": boots,
            "start_seed": START_SEED,
            "dataset": "primary",
            "map_kwargs": sd.enet_map_kwargs(),
            "wall_seconds": wall,
            "keys": [f"{p}/{k}" for p, d in results.items() for k in d],
            "env": environment(),
        }
        save(results, fixture_path(boots), meta)
        print(f"boots={boots}: {wall:.1f}s -> {fixture_path(boots)}", flush=True)


if __name__ == "__main__":
    main()
