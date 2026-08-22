"""Launch independent benchmark comparisons with unique seeds and outputs."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_FLY_DATA_ROOT = (
    "/data/group_data/neuroagents_lab/neural_datasets/brezovec_drosophila_data/000727"
)
DEFAULT_EVENTS_DIR = (
    "/data/group_data/neuroagents_lab/autonomous-drosophila/outputs/"
    "interanimal/turn_m15_p0p4_v1/events"
)


def protocol_flags(args):
    flags = [
        "--suite",
        "production",
        "--fly-data-root",
        args.fly_data_root,
        "--source-fly",
        args.source_fly,
        "--target-fly",
        args.target_fly,
        "--direction",
        args.direction,
        "--region",
        args.region,
        "--event-source",
        args.event_source,
        "--split-strategy",
        "blocked",
        "--splits",
        "5",
        "--consistency-map",
        "percentile",
        "--percentile",
        str(args.percentile),
        "--alpha",
        str(args.alpha),
        "--workloads",
        "consistency",
    ]
    if args.events_dir:
        flags.extend(["--events-dir", args.events_dir])
    return flags


def assign_seed_ranges(base_seed, bootstrap_iters, n_jobs, replica=0):
    start = int(base_seed) + int(replica) * int(bootstrap_iters) * int(n_jobs)
    return [
        {
            "job_index": index,
            "seed": start + index * int(bootstrap_iters),
            "bootstrap_iters": int(bootstrap_iters),
        }
        for index in range(int(n_jobs))
    ]


def output_path(directory, stem, job_index, replica, backend, workers):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "{}_r{}_j{}_{}_w{}.json".format(
        stem, replica, job_index, backend, workers
    )


def build_command(python, benchmark_module, assignment, args):
    output = output_path(
        args.output_dir,
        args.stem,
        assignment["job_index"],
        args.replica,
        args.backend,
        args.workers,
    )
    command = [
        python,
        "-m",
        benchmark_module,
        *protocol_flags(args),
        "--backend",
        args.backend,
        "--workers",
        str(args.workers),
        "--bootstrap-iters",
        str(assignment["bootstrap_iters"]),
        "--seed",
        str(assignment["seed"]),
        "--output",
        str(output),
    ]
    if args.target_limit is not None:
        command.extend(["--target-limit", str(args.target_limit)])
    if args.correlation_target_batch_size is not None:
        command.extend(
            [
                "--correlation-target-batch-size",
                str(args.correlation_target_batch_size),
            ]
        )
    if args.ridge_workspace_bytes is not None:
        command.extend(["--ridge-workspace-bytes", str(args.ridge_workspace_bytes)])
    if args.dtype is not None:
        command.extend(["--dtype", args.dtype])
    if args.compute_dtype is not None:
        command.extend(["--compute-dtype", args.compute_dtype])
    if args.correlation_dtype is not None:
        command.extend(["--correlation-dtype", args.correlation_dtype])
    assignment = dict(assignment)
    assignment["output"] = str(output)
    assignment["command"] = command
    return assignment


def run_local(assignments, max_concurrent):
    pending = list(assignments)
    running = []
    failures = []
    while pending or running:
        while pending and len(running) < max_concurrent:
            assignment = pending.pop(0)
            process = subprocess.Popen(assignment["command"])
            running.append((assignment, process))
        if not running:
            break
        assignment, process = running.pop(0)
        code = process.wait()
        assignment["returncode"] = code
        if code != 0:
            failures.append(assignment)
    return failures


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stem", default="anchor")
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--bootstrap-iters", type=int, default=200)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--replica", type=int, default=0)
    parser.add_argument("--max-concurrent", type=int, default=1)
    parser.add_argument("--fly-data-root", default=os.environ.get("FLY_DATA_ROOT", DEFAULT_FLY_DATA_ROOT))
    parser.add_argument("--events-dir", default=os.environ.get("EVENTS_DIR", DEFAULT_EVENTS_DIR))
    parser.add_argument("--source-fly", default=os.environ.get("SOURCE_FLY", "fly_100"))
    parser.add_argument("--target-fly", default=os.environ.get("TARGET_FLY", "fly_099"))
    parser.add_argument("--direction", default=os.environ.get("DIRECTION", "right"))
    parser.add_argument("--region", default="IPS")
    parser.add_argument("--event-source", default="historical")
    parser.add_argument("--percentile", type=float, default=99.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--target-limit", type=int)
    parser.add_argument("--correlation-target-batch-size")
    parser.add_argument("--ridge-workspace-bytes", type=int)
    parser.add_argument("--dtype")
    parser.add_argument("--compute-dtype")
    parser.add_argument("--correlation-dtype")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--benchmark-module", default="benchmarks.benchmark_neural_backends"
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    assignments = [
        build_command(args.python, args.benchmark_module, assignment, args)
        for assignment in assign_seed_ranges(
            args.seed, args.bootstrap_iters, args.jobs, args.replica
        )
    ]
    manifest = Path(args.output_dir) / "{}_r{}_manifest.json".format(
        args.stem, args.replica
    )
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(manifest, "w") as stream:
        json.dump(
            {
                "backend": args.backend,
                "workers": args.workers,
                "replica": args.replica,
                "assignments": assignments,
            },
            stream,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")
    if args.dry_run:
        for assignment in assignments:
            print(" ".join(assignment["command"]))
        return 0
    failures = run_local(assignments, args.max_concurrent)
    if failures:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
