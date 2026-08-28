"""Reproducible neural mapper and consistency benchmarks."""

import argparse
import copy
import csv
import hashlib
import inspect
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import scipy
import sklearn

from benchmarks.artifacts import artifact_record, atomic_savez, atomic_write_json
from benchmarks.config_manifest import repository_identity
from benchmarks.precision import CONSISTENCY_METRICS, save_precision_diagnostic
from benchmarks.telemetry import (
    GpuSampler,
    collect_host_inventory,
    host_faults_and_swap,
    summarize_gpu_samples,
)
from brainmodel_utils.metrics.consistency import get_linregress_consistency
from brainmodel_utils.metrics.utils import generic_trial_avg, get_splithalves
from brainmodel_utils.neural_mappers import (
    PercentileNeuralMap,
    PipelineNeuralMap,
    SKLinearNeuralMap,
)
from brainmodel_utils.neural_mappers import backends as neural_backends
from brainmodel_utils.neural_mappers.profiling import record_phases
from brainmodel_utils.neural_mappers.utils import generate_train_test_splits

SUITES = {
    "smoke": {
        "stimuli": 48,
        "sources": 24,
        "targets": 6,
        "trials": 6,
        "latent_dim": 8,
        "signal_sources": 4,
        "bootstrap_iters": 1,
        "splits": 2,
    },
    "synthetic": {
        "stimuli": 256,
        "sources": 256,
        "targets": 32,
        "trials": 12,
        "latent_dim": 32,
        "signal_sources": 16,
        "bootstrap_iters": 5,
        "splits": 3,
    },
    "production": {
        "stimuli": 39,
        "sources": 27118,
        "targets": 128,
        "trials": 20,
        "latent_dim": 64,
        "signal_sources": 32,
        "bootstrap_iters": 200,
        "splits": 5,
    },
}

WORKLOADS = ("transfer", "ridge", "percentile", "consistency")
WORKLOAD_KEYS = {
    "transfer": "host_device_transfer",
    "ridge": "ridge_mapper",
    "percentile": "percentile_mapper",
    "consistency": "linregress_consistency",
}
REGION_ROIS = {
    "IPS": ("IPS_L", "IPS_R"),
    "IPS_L": ("IPS_L",),
    "IPS_R": ("IPS_R",),
}
NEURAL_SERIES_PATH = "processing/ophys/TwoPhotonSeriesFunctionalGreenProcessed"
SELECTED_SOURCE_DETAIL_LIMIT = 256
PROVENANCE_HASH_LIMIT_BYTES = 64 * 1024**2
BENCHMARK_SCHEMA_VERSION = 3

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def _read_text(path):
    try:
        with open(path, "r") as stream:
            return stream.read()
    except OSError:
        return None


def _cpu_model():
    cpuinfo = _read_text("/proc/cpuinfo")
    if cpuinfo:
        for line in cpuinfo.splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def _affinity_count():
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return None


def _peak_rss_bytes():
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (ImportError, ValueError):
        return None
    if sys.platform == "darwin":
        return int(peak)
    return int(peak * 1024)


def _resource_cpu_times():
    try:
        import resource

        own = resource.getrusage(resource.RUSAGE_SELF)
        children = resource.getrusage(resource.RUSAGE_CHILDREN)
    except (ImportError, ValueError):
        return None
    return {
        "self_user_seconds": own.ru_utime,
        "self_system_seconds": own.ru_stime,
        "children_user_seconds": children.ru_utime,
        "children_system_seconds": children.ru_stime,
    }


def _cpu_usage(start, end, wall_seconds, process_tree_seconds=None):
    if start is None or end is None:
        return None
    usage = {name: end[name] - start[name] for name in start}
    resource_total = sum(usage.values())
    total = (
        max(process_tree_seconds, resource_total)
        if process_tree_seconds is not None
        else resource_total
    )
    usage.update(
        {
            "total_seconds": total,
            "utilization_percent": 100.0 * total / wall_seconds,
            "resource_total_seconds": resource_total,
            "process_tree_seconds": process_tree_seconds,
            "scope": "sampled_process_tree_or_resource_max"
            if process_tree_seconds is not None
            else "self_and_terminated_children",
        }
    )
    return usage


def _pid_rss_bytes(pid):
    statm = _read_text(f"/proc/{pid}/statm")
    if statm:
        return int(statm.split()[1]) * os.sysconf("SC_PAGE_SIZE")
    if pid == os.getpid():
        return _peak_rss_bytes()
    return None


def _pid_cpu_ticks(pid):
    stat = _read_text(f"/proc/{pid}/stat")
    if not stat:
        return None
    close_paren = stat.rfind(")")
    if close_paren < 0:
        return None
    fields = stat[close_paren + 2 :].split()
    if len(fields) < 13:
        return None
    return int(fields[11]) + int(fields[12])


def _process_tree_pids(root_pid):
    pending = [root_pid]
    seen = set()
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        pending.extend(_child_pids(pid))
    return seen


def _child_pids(pid):
    children = _read_text(f"/proc/{pid}/task/{pid}/children")
    if not children:
        return []
    return [int(value) for value in children.split()]


def _process_tree_rss_bytes(root_pid):
    pending = [root_pid]
    seen = set()
    total = 0
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        rss = _pid_rss_bytes(pid)
        if rss is not None:
            total += rss
        pending.extend(_child_pids(pid))
    return total or None


def _run_nvidia_smi(arguments):
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    try:
        completed = subprocess.run(  # noqa: UP022 -- Python 3.6 support
            [executable] + arguments,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,  # noqa: UP021 -- Python 3.6 support
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _gpu_inventory():
    output = _run_nvidia_smi(
        [
            "--query-gpu=index,name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return []
    devices = []
    for row in csv.reader(output.splitlines(), skipinitialspace=True):
        if len(row) != 5:
            continue
        devices.append(
            {
                "index": int(row[0]),
                "name": row[1],
                "uuid": row[2],
                "driver_version": row[3],
                "memory_total_bytes": int(float(row[4]) * 1024 * 1024),
            }
        )
    return devices


def _cuda_probe(gpu_index):
    result = {
        "available": False,
        "device_index": gpu_index,
        "reason": None,
    }
    try:
        import cupy
    except (ImportError, OSError, RuntimeError) as error:
        result["reason"] = f"CuPy unavailable: {error}"
    else:
        try:
            device_count = int(cupy.cuda.runtime.getDeviceCount())
        except (cupy.cuda.runtime.CUDARuntimeError, RuntimeError) as error:
            result["reason"] = f"CuPy CUDA runtime unavailable: {error}"
        else:
            result.update(
                {
                    "available": gpu_index < device_count,
                    "device_count": device_count,
                    "cupy_version": getattr(cupy, "__version__", None),
                }
            )
            if gpu_index >= device_count:
                result["reason"] = (
                    f"CUDA device index {gpu_index} is unavailable; {device_count} devices are visible"
                )
    try:
        import cuml
    except (ImportError, OSError, RuntimeError) as error:
        result["cuml_reason"] = str(error)
    else:
        result["cuml_version"] = getattr(cuml, "__version__", None)
    try:
        import torch
    except (ImportError, OSError, RuntimeError) as error:
        result["torch_reason"] = str(error)
    else:
        torch_available = bool(torch.cuda.is_available())
        result.update(
            {
                "torch_cuda_available": torch_available,
                "torch_device_count": int(torch.cuda.device_count())
                if torch_available
                else 0,
                "torch_version": getattr(torch, "__version__", None),
            }
        )
    return result


def _resolve_transfer_backend(backend, gpu_index):
    if backend == "cpu":
        return "cpu"
    try:
        import cupy

        device_count = int(cupy.cuda.runtime.getDeviceCount())
    except (ImportError, OSError, RuntimeError) as error:
        raise RuntimeError(f"CUDA transfer backend is unavailable: {error}")
    if gpu_index < device_count:
        return "cuda"
    raise RuntimeError(
        f"CUDA device index {gpu_index} is unavailable; {device_count} devices are visible"
    )


def _resolve_workload_backends(
    backend, workloads, consistency_map, regression_kwargs, gpu_index
):
    neural_backends.validate_backend(backend)
    resolved = {}
    ridge_backend = None
    percentile_backend = None

    for workload in workloads:
        if workload == "transfer":
            resolved[workload] = _resolve_transfer_backend(backend, gpu_index)
            continue
        if ridge_backend is None:
            ridge_backend = neural_backends.resolve_backend(
                backend, "Ridge", regression_kwargs
            )
        needs_percentile = workload == "percentile" or (
            workload == "consistency" and consistency_map == "percentile"
        )
        if needs_percentile:
            if percentile_backend is None:
                percentile_backend = ridge_backend
                if percentile_backend == "cuda":
                    percentile_backend = neural_backends.resolve_torch_backend(backend)
            resolved[workload] = percentile_backend
        else:
            resolved[workload] = ridge_backend
    return resolved


def _verify_explicit_backend(requested_backend, workloads, effective_backends):
    if requested_backend != "cuda":
        return
    non_cuda = [
        workload for workload in workloads if effective_backends.get(workload) != "cuda"
    ]
    if non_cuda:
        raise RuntimeError(
            "explicit CUDA benchmark resolved non-CUDA workloads: "
            + ", ".join(non_cuda)
        )


def _select_cuda_device(gpu_index, require_torch):
    if require_torch:
        import torch

        if gpu_index >= torch.cuda.device_count():
            raise ValueError(
                f"PyTorch CUDA device index {gpu_index} is unavailable; {torch.cuda.device_count()} devices are visible"
            )
        torch.cuda.set_device(gpu_index)

    import cupy

    device_count = int(cupy.cuda.runtime.getDeviceCount())
    if gpu_index >= device_count:
        raise ValueError(
            f"CUDA device index {gpu_index} is unavailable; {device_count} devices are visible"
        )
    cupy.cuda.Device(gpu_index).use()


class _GpuReader:
    def __init__(self, device_index):
        self.device_index = device_index
        self.sampler = GpuSampler(device_index)
        self.method = self.sampler.method
        self.error = self.sampler.error
        self.identity = self.sampler.identity

    def sample(self, pids=None):
        sample = self.sampler.sample(pids=pids) or {}
        self.error = self.sampler.error
        return sample

    def close(self):
        self.sampler.close()
        self.error = self.sampler.error


class ResourceMonitor:
    def __init__(self, interval, monitor_gpu, gpu_index):
        self.interval = interval
        self.root_pid = os.getpid()
        self.gpu_reader = _GpuReader(gpu_index) if monitor_gpu else None
        self.stop_event = threading.Event()
        self.thread = None
        self.samples = 0
        self.rss_start = _process_tree_rss_bytes(self.root_pid)
        self.peak_driver_rss = 0
        self.peak_tree_rss = 0
        self.gpu_start_memory = None
        self.peak_gpu_memory = None
        self.peak_gpu_utilization = None
        self.peak_process_gpu_memory = None
        self.gpu_samples = []
        self.host_samples = []
        self.error = None
        self.cpu_start = None
        self.cpu_tick_baseline = {}
        self.cpu_tick_latest = {}
        self.wall_start = None
        self.host_start = host_faults_and_swap(self.root_pid)

    def _sample(self):
        driver_rss = _pid_rss_bytes(self.root_pid)
        tree_rss = _process_tree_rss_bytes(self.root_pid)
        if driver_rss is not None:
            self.peak_driver_rss = max(self.peak_driver_rss, driver_rss)
        if tree_rss is not None:
            self.peak_tree_rss = max(self.peak_tree_rss, tree_rss)
        for pid in _process_tree_pids(self.root_pid):
            ticks = _pid_cpu_ticks(pid)
            if ticks is None:
                continue
            self.cpu_tick_baseline.setdefault(pid, 0)
            self.cpu_tick_latest[pid] = ticks
        elapsed = (
            None if self.wall_start is None else time.perf_counter() - self.wall_start
        )
        if self.gpu_reader is not None:
            sample = (
                self.gpu_reader.sample(pids=_process_tree_pids(self.root_pid)) or {}
            )
            sample["elapsed_seconds"] = elapsed
            self.gpu_samples.append(sample)
            memory = sample.get("device_memory_used_bytes")
            utilization = sample.get("utilization_percent")
            process_memory = sample.get("process_memory_used_bytes")
            if memory is not None:
                if self.gpu_start_memory is None:
                    self.gpu_start_memory = memory
                if self.peak_gpu_memory is None:
                    self.peak_gpu_memory = memory
                self.peak_gpu_memory = max(self.peak_gpu_memory, memory)
            if utilization is not None:
                if self.peak_gpu_utilization is None:
                    self.peak_gpu_utilization = utilization
                self.peak_gpu_utilization = max(self.peak_gpu_utilization, utilization)
            if process_memory is not None:
                if self.peak_process_gpu_memory is None:
                    self.peak_process_gpu_memory = process_memory
                self.peak_process_gpu_memory = max(
                    self.peak_process_gpu_memory, process_memory
                )
        host = host_faults_and_swap(self.root_pid)
        host["elapsed_seconds"] = elapsed
        host["rss_bytes"] = tree_rss
        self.host_samples.append(host)
        self.samples += 1

    def _run(self):
        while not self.stop_event.wait(self.interval):
            try:
                self._sample()
            except (OSError, ValueError) as error:
                self.error = str(error)

    def start(self):
        self.cpu_start = _resource_cpu_times()
        self.wall_start = time.perf_counter()
        for pid in _process_tree_pids(self.root_pid):
            ticks = _pid_cpu_ticks(pid)
            if ticks is not None:
                self.cpu_tick_baseline[pid] = ticks
                self.cpu_tick_latest[pid] = ticks
        self._sample()
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=max(3.0, self.interval * 2))
        try:
            self._sample()
        except (OSError, ValueError) as error:
            self.error = str(error)
        wall_seconds = time.perf_counter() - self.wall_start
        clock_ticks = os.sysconf("SC_CLK_TCK")
        process_tree_seconds = sum(
            max(0, ticks - self.cpu_tick_baseline[pid])
            for pid, ticks in self.cpu_tick_latest.items()
        ) / float(clock_ticks)
        cpu = _cpu_usage(
            self.cpu_start,
            _resource_cpu_times(),
            wall_seconds,
            process_tree_seconds,
        )
        if self.gpu_reader is not None:
            self.gpu_reader.close()
        peak_tree = self.peak_tree_rss or None
        baseline = self.rss_start
        gpu_summary = summarize_gpu_samples(self.gpu_samples)
        identity = self.gpu_reader.identity if self.gpu_reader is not None else None
        gpu_result = {
            "sampled": self.gpu_reader is not None,
            "method": self.gpu_reader.method if self.gpu_reader is not None else None,
            "device_index": self.gpu_reader.device_index
            if self.gpu_reader is not None
            else None,
            "identity": identity,
            "samples": self.samples if self.gpu_reader is not None else 0,
            "memory_scope": "uuid_resolved_device",
            "memory_start_bytes": self.gpu_start_memory,
            "peak_memory_used_bytes": self.peak_gpu_memory,
            "peak_memory_increase_bytes": _positive_delta(
                self.peak_gpu_memory, self.gpu_start_memory
            ),
            "peak_process_memory_used_bytes": self.peak_process_gpu_memory,
            "peak_utilization_percent": self.peak_gpu_utilization,
            "time_series": gpu_summary,
            "series": _compact_gpu_series(self.gpu_samples),
            "error": self.error
            or (self.gpu_reader.error if self.gpu_reader is not None else None),
        }
        host_end = host_faults_and_swap(self.root_pid)
        return {
            "memory_scope": "driver_and_descendant_processes",
            "rss_start_bytes": baseline,
            "peak_rss_bytes": peak_tree,
            "peak_rss_increase_bytes": _positive_delta(peak_tree, baseline),
            "peak_driver_rss_bytes": self.peak_driver_rss or None,
            "peak_child_rss_bytes": max(0, (peak_tree or 0) - self.peak_driver_rss)
            if peak_tree is not None
            else None,
            "host_start": self.host_start,
            "host_end": host_end,
            "gpu": gpu_result,
            "monitor_error": self.error,
            "wall_seconds": wall_seconds,
            "cpu": cpu,
        }


def _compact_gpu_series(samples, limit=256):
    if not samples:
        return []
    stride = max(1, math.ceil(len(samples) / float(limit)))
    keys = (
        "elapsed_seconds",
        "utilization_percent",
        "device_memory_used_bytes",
        "process_memory_used_bytes",
        "power_watts",
        "temperature_c",
        "sm_clock_mhz",
        "pcie_tx_kib_per_sec",
        "pcie_rx_kib_per_sec",
    )
    return [{key: sample.get(key) for key in keys} for sample in samples[::stride]]


def _positive_delta(value, baseline):
    if value is None or baseline is None:
        return None
    return max(0, value - baseline)


def _synchronize(effective_backend):
    if effective_backend != "cuda":
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except (ImportError, RuntimeError):
        pass
    try:
        import cupy

        cupy.cuda.get_current_stream().synchronize()
    except (ImportError, RuntimeError):
        pass


def _seconds(callable_, effective_backend):
    _synchronize(effective_backend)
    start = time.perf_counter()
    value = callable_()
    _synchronize(effective_backend)
    return value, time.perf_counter() - start


def _supports_backend(mapper_class):
    return "backend" in inspect.signature(mapper_class.__init__).parameters


def _mapper_options(
    mapper_class,
    requested_backend,
    effective_backend,
    regression_kwargs,
    extra=None,
):
    options = {
        "regression_type": "Ridge",
        "regression_kwargs": regression_kwargs,
    }
    if extra:
        options.update(extra)
    if _supports_backend(mapper_class):
        options["backend"] = requested_backend
    elif effective_backend != "cpu":
        raise RuntimeError(
            f"{mapper_class.__name__} does not expose the requested backend API"
        )
    return options


def _effective_mapper_backend(mapper, expected_backend):
    effective = getattr(mapper, "_resolved_backend", None)
    if effective is None:
        raise RuntimeError("mapper did not report an effective backend after fitting")
    if effective != expected_backend:
        raise RuntimeError(
            f"mapper resolved backend {effective}, expected {expected_backend}"
        )
    return effective


def _dtype_name(dtype):
    return str(np.dtype(dtype))


def _precision_plan(args):
    preparation = np.dtype(args.dtype)
    compute_option = getattr(args, "compute_dtype", None)
    mapper_input = np.dtype(compute_option) if compute_option else preparation
    correlation_option = getattr(args, "correlation_dtype", None)
    correlation = np.dtype(correlation_option) if correlation_option else mapper_input
    allow_mixed = bool(getattr(args, "allow_mixed_precision", False))
    mismatches = []
    if mapper_input != preparation:
        mismatches.append(
            f"--compute-dtype {_dtype_name(mapper_input)} differs from "
            f"--dtype {_dtype_name(preparation)}"
        )
    if correlation != mapper_input:
        mismatches.append(
            f"--correlation-dtype {_dtype_name(correlation)} differs from mapper "
            f"input dtype {_dtype_name(mapper_input)}"
        )
    if mismatches and not allow_mixed:
        raise ValueError(
            "; ".join(mismatches) + "; pass --allow-mixed-precision for diagnostics"
        )
    return {
        "allow_mixed_precision": allow_mixed,
        "preparation": preparation,
        "mapper_input": mapper_input,
        "correlation": correlation,
        "compute_explicit": compute_option is not None,
        "correlation_explicit": correlation_option is not None,
    }


def _validate_artifact_options(args, workloads):
    metrics_path = getattr(args, "save_precision_artifacts", None)
    learned_fit_path = getattr(args, "save_learned_fit", None)
    if (metrics_path or learned_fit_path) and args.output == "-":
        raise ValueError("precision artifacts require --output to be a JSON path")
    if args.output != "-":
        output = Path(args.output).resolve()
        artifacts = [path for path in (metrics_path, learned_fit_path) if path]
        if any(Path(path).resolve() == output for path in artifacts):
            raise ValueError("precision artifact paths must differ from --output")
        if (
            learned_fit_path
            and Path(learned_fit_path).with_suffix(".json").resolve() == output
        ):
            raise ValueError("learned-fit metadata path must differ from --output")
    if learned_fit_path and "consistency" not in workloads:
        raise ValueError("--save-learned-fit requires the consistency workload")
    if learned_fit_path and args.consistency_map != "percentile":
        raise ValueError("--save-learned-fit requires --consistency-map percentile")
    learned_fit_target_limit = getattr(args, "learned_fit_target_limit", None)
    if learned_fit_target_limit is not None and not learned_fit_path:
        raise ValueError("--learned-fit-target-limit requires --save-learned-fit")
    learned_target_indices = getattr(args, "learned_target_indices", None)
    if learned_target_indices is not None and not learned_fit_path:
        raise ValueError("--learned-target-indices requires --save-learned-fit")
    if learned_fit_target_limit is not None and learned_target_indices is not None:
        raise ValueError(
            "--learned-fit-target-limit and --learned-target-indices are mutually exclusive"
        )
    if not learned_fit_path and (
        getattr(args, "learned_bootstrap_index", 0) != 0
        or getattr(args, "learned_fold_index", 0) != 0
    ):
        raise ValueError("learned cell selectors require --save-learned-fit")


def _percentile_extras(args, correlation_dtype=None):
    extras = {}
    batch_size = getattr(args, "correlation_target_batch_size", None)
    if batch_size is not None:
        extras["correlation_target_batch_size"] = (
            "all" if str(batch_size).lower() == "all" else int(batch_size)
        )
    workspace = getattr(args, "ridge_workspace_bytes", None)
    if workspace is not None:
        extras["ridge_workspace_bytes"] = workspace
    correlation_dtype = getattr(args, "correlation_dtype", None) or correlation_dtype
    if correlation_dtype is not None:
        extras["correlation_dtype"] = _dtype_name(correlation_dtype)
    max_batch_size = getattr(args, "ridge_max_batch_size", None)
    if max_batch_size is not None:
        extras["ridge_max_batch_size"] = (
            "all" if str(max_batch_size).lower() == "all" else int(max_batch_size)
        )
    return extras


class TimedPercentileNeuralMap(PercentileNeuralMap):
    def __init__(self, benchmark_backend, *args, **kwargs):
        self.benchmark_backend = benchmark_backend
        self.correlation_and_source_selection_seconds = 0.0
        self.benchmark_phase = None
        super().__init__(*args, **kwargs)

    def _gather_correlations(self, X, Y):
        _synchronize(self.benchmark_backend)
        start = time.perf_counter()
        super()._gather_correlations(X, Y)
        _synchronize(self.benchmark_backend)
        self.correlation_and_source_selection_seconds += time.perf_counter() - start

    def _gather_thresholds(self):
        start = time.perf_counter()
        super()._gather_thresholds()
        self.correlation_and_source_selection_seconds += time.perf_counter() - start

    def _get_sources(self, target_idx):
        start = time.perf_counter()
        sources = super()._get_sources(target_idx)
        elapsed = time.perf_counter() - start
        if self.benchmark_phase == "fit":
            self.correlation_and_source_selection_seconds += elapsed
        return sources


def _standardize_columns(array):
    array = array - array.mean(axis=0, keepdims=True)
    scale = array.std(axis=0, keepdims=True)
    scale[scale == 0] = 1
    return array / scale


def make_correlated_data(config, seed, dtype):
    rng = np.random.RandomState(seed)
    latent = rng.standard_normal((config["stimuli"], config["latent_dim"]))
    loadings = rng.standard_normal((config["latent_dim"], config["sources"]))
    source = np.dot(latent, loadings) / math.sqrt(config["latent_dim"])
    source += 0.35 * rng.standard_normal(source.shape)
    source = _standardize_columns(source)

    weights = np.zeros((config["sources"], config["targets"]))
    for target_idx in range(config["targets"]):
        selected = rng.choice(
            config["sources"], size=config["signal_sources"], replace=False
        )
        weights[selected, target_idx] = rng.standard_normal(config["signal_sources"])
    target_mean = np.dot(source, weights) / math.sqrt(config["signal_sources"])
    target_mean = _standardize_columns(target_mean)
    target = target_mean[np.newaxis, :, :] + config["noise_std"] * rng.standard_normal(
        (config["trials"], config["stimuli"], config["targets"])
    )
    return source.astype(dtype), target.astype(dtype)


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path_record(path):
    path = Path(path).resolve()
    stat = path.stat()
    record = {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if stat.st_size <= PROVENANCE_HASH_LIMIT_BYTES:
        record.update({"sha256": _file_sha256(path), "sha256_scope": "full_file"})
    else:
        record.update({"sha256": None, "sha256_scope": "omitted_size_limit"})
    return record


def _dandiset_record(root):
    path = Path(root) / "dandiset.yaml"
    if not path.is_file():
        return None
    contents = _read_text(path)
    if contents is None:
        raise ValueError(f"could not read dandiset metadata: {path}")
    identifier = re.search(r"^identifier:\s*(\S+)", contents, flags=re.MULTILINE)
    version = re.search(r"^version:\s*(\S+)", contents, flags=re.MULTILINE)
    doi = re.search(r"https://doi\.org/(10\.48324/dandi\.[^\s]+)", contents)
    return {
        "file": _path_record(path),
        "identifier": identifier.group(1) if identifier else None,
        "version": version.group(1) if version else None,
        "doi": doi.group(1) if doi else None,
    }


def _nwb_for_fly(root, fly_id):
    number = fly_id.split("_", 1)[-1]
    pattern = f"sub-fly-{number}/sub-fly-*_ses-*_behavior+image+ophys.nwb"
    matches = sorted(Path(root).glob(pattern))
    if not matches:
        raise ValueError(f"no NWB file found for {fly_id} under {root}")
    if len(matches) > 1:
        raise ValueError(f"multiple NWB files found for {fly_id} under {root}")
    return matches[0]


def _load_neural_timestamps(nwb_path):
    with h5py.File(str(nwb_path), "r") as nwb:
        if NEURAL_SERIES_PATH not in nwb:
            raise ValueError(f"NWB file {nwb_path} is missing {NEURAL_SERIES_PATH}")
        series = nwb[NEURAL_SERIES_PATH]
        if "data" not in series:
            raise ValueError("NWB neural series is missing its data array")
        length = int(series["data"].shape[0])
        if length < 2:
            raise ValueError("NWB neural series must contain at least two samples")
        if "timestamps" in series:
            timestamps = np.asarray(series["timestamps"], dtype=np.float64)
            storage = "explicit"
            starting_time = None
            rate = None
        else:
            if "starting_time" not in series:
                raise ValueError("NWB neural series has no timestamps or starting_time")
            starting = series["starting_time"]
            if "rate" not in starting.attrs:
                raise ValueError("NWB neural starting_time has no rate attribute")
            starting_time = float(starting[()])
            rate = float(starting.attrs["rate"])
            if rate <= 0:
                raise ValueError("NWB neural sampling rate must be positive")
            timestamps = starting_time + np.arange(length, dtype=np.float64) / rate
            storage = "implicit"
    if timestamps.shape != (length,):
        raise ValueError("NWB neural timestamps and data have different lengths")
    if not np.isfinite(timestamps).all() or not np.all(np.diff(timestamps) > 0):
        raise ValueError("NWB neural timestamps must be finite and increasing")
    return timestamps, {
        "file": _path_record(nwb_path),
        "series_path": NEURAL_SERIES_PATH,
        "timestamp_storage": storage,
        "samples": length,
        "starting_time": starting_time,
        "rate": rate,
    }


PROTOCOL_RELATIVE_DT = 0.532
PROTOCOL_RELATIVE_START = -15.0
PROTOCOL_RELATIVE_BINS = 39
DEFAULT_CLUSTER_EVENTS_DIR = Path(
    "/data/group_data/neuroagents_lab/autonomous-drosophila/outputs/"
    "interanimal/turn_m15_p0p4_v1/events"
)


def protocol_relative_times():
    return PROTOCOL_RELATIVE_START + PROTOCOL_RELATIVE_DT * np.arange(
        PROTOCOL_RELATIVE_BINS, dtype=np.float64
    )


def _event_files(root, fly_id, direction, events_dir=None):
    candidates = []
    extra = []
    if events_dir:
        extra.append(Path(events_dir))
    env_dir = os.environ.get("EVENTS_DIR")
    if env_dir:
        extra.append(Path(env_dir))
    extra.append(DEFAULT_CLUSTER_EVENTS_DIR)
    extra.append(Path(root) / "inter_animal_consistency" / "two_fly_pilot")
    seen = set()
    for base in extra:
        base = Path(base)
        key = str(base)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            {
                "window": base
                / "event_tensors"
                / "short"
                / f"{fly_id}_{direction}.npz",
                "bouts": base / "events" / f"{fly_id}_turn_bouts.csv",
            }
        )
        candidates.append(
            {
                "window": base / f"{fly_id}_{direction}.npz",
                "bouts": base / f"{fly_id}_turn_bouts.csv",
            }
        )
    for candidate in candidates:
        if candidate["window"].is_file():
            return candidate["window"], candidate["bouts"]
    first = candidates[0]
    return first["window"], first["bouts"]


def _historical_event_times(path, direction):
    with open(str(path), "r", newline="") as stream:
        rows = csv.DictReader(stream)
        if rows.fieldnames is None or not {
            "direction",
            "event_time_seconds",
        }.issubset(rows.fieldnames):
            raise ValueError(f"event CSV is missing required columns: {path}")
        values = [
            float(row["event_time_seconds"])
            for row in rows
            if row["direction"] == direction
        ]
    return np.asarray(values, dtype=np.float64)


def _load_events(root, fly_id, direction, event_source, events_dir=None):
    if event_source not in ("validated", "historical"):
        raise ValueError(f"unknown event source: {event_source}")
    window_path, bouts_path = _event_files(
        root, fly_id, direction, events_dir=events_dir
    )
    if not window_path.is_file():
        raise ValueError(f"event window does not exist: {window_path}")
    with np.load(str(window_path), allow_pickle=False) as window:
        files = set(window.files)
        if "relative_times" not in files and "event_times" not in files:
            raise ValueError(
                f"event window is missing event_times/relative_times: {window_path}"
            )
        relative_times = (
            np.asarray(window["relative_times"], dtype=np.float64)
            if "relative_times" in files
            else protocol_relative_times()
        )
        npz_event_times = (
            np.asarray(window["event_times"], dtype=np.float64)
            if "event_times" in files
            else None
        )
        if event_source == "validated":
            if npz_event_times is None:
                raise ValueError(f"event window is missing event_times: {window_path}")
            event_times = npz_event_times
        else:
            if bouts_path.is_file():
                event_times = _historical_event_times(bouts_path, direction)
            elif npz_event_times is not None:
                event_times = npz_event_times
            else:
                raise ValueError(f"event CSV does not exist: {bouts_path}")
        if os.environ.get("PROTOCOL_RELATIVE_TIMES", "").lower() in {
            "1",
            "true",
            "yes",
        }:
            relative_times = protocol_relative_times()
    if event_times.size < 2:
        raise ValueError(f"{fly_id} has fewer than two {direction} events")
    if relative_times.size < 2:
        raise ValueError("event window must contain at least two relative times")
    if not np.isfinite(event_times).all() or not np.isfinite(relative_times).all():
        raise ValueError("event times must be finite")
    provenance = {
        "event_source": event_source,
        "event_count": int(event_times.size),
        "window_file": _path_record(window_path),
    }
    if event_source == "historical":
        if bouts_path.is_file():
            provenance["event_file"] = _path_record(bouts_path)
        else:
            provenance["event_times_from"] = "npz"
    return event_times, relative_times, provenance


def _region_shape(roi_h5, fly_id, roi_names):
    shapes = []
    with h5py.File(str(roi_h5), "r") as roi_file:
        if fly_id not in roi_file:
            raise ValueError(f"ROI HDF5 has no group for {fly_id}")
        group = roi_file[fly_id]
        for name in roi_names:
            if name not in group:
                raise ValueError(f"ROI HDF5 has no dataset {fly_id}/{name}")
            dataset = group[name]
            if dataset.ndim != 2 or dataset.shape[1] < 1:
                raise ValueError(f"ROI dataset {fly_id}/{name} must be time by voxel")
            shapes.append((name, int(dataset.shape[0]), int(dataset.shape[1])))
    timepoints = {shape[1] for shape in shapes}
    if len(timepoints) != 1:
        raise ValueError(f"ROI datasets for {fly_id} have different lengths")
    return shapes


def _parse_target_indices(value):
    try:
        indices = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "target indices must be comma-separated integers"
        ) from error
    if not indices or any(index < 0 for index in indices):
        raise argparse.ArgumentTypeError("target indices must be non-negative")
    if any(
        left >= right
        for left, right in zip(  # noqa: RUF007 -- Python 3.6 support
            indices, indices[1:]
        )
    ):
        raise argparse.ArgumentTypeError("target indices must be sorted and unique")
    return indices


def _target_indices(total, limit, explicit_indices=None):
    if limit is not None and explicit_indices is not None:
        raise ValueError("target limit and target indices are mutually exclusive")
    if explicit_indices is not None:
        selected = np.asarray(explicit_indices, dtype=np.int64)
        if selected.ndim != 1 or selected.size < 1:
            raise ValueError("target indices must be a non-empty vector")
        if selected[0] < 0 or selected[-1] >= total:
            raise ValueError(f"target index must be between 0 and {total - 1}")
        if not np.all(np.diff(selected) > 0):
            raise ValueError("target indices must be sorted and unique")
        return selected, selected.tolist()
    if limit is None or limit == total:
        return np.arange(total, dtype=np.int64), None
    if limit < 1 or limit > total:
        raise ValueError(f"target limit must be between 1 and {total}")
    selected = np.floor(np.linspace(0, total - 1, limit)).astype(np.int64)
    return selected, selected.tolist()


def _interpolation_coordinates(timestamps, event_times, relative_times, dtype):
    queries = (event_times[:, None] + relative_times[None, :]).reshape(-1)
    before_start = queries < timestamps[0]
    after_end = queries > timestamps[-1]
    clamped = before_start | after_end
    upper = np.searchsorted(timestamps, queries, side="right")
    upper = np.clip(upper, 1, timestamps.size - 1)
    lower = upper - 1
    weights = (queries - timestamps[lower]) / (timestamps[upper] - timestamps[lower])
    weights = np.asarray(np.clip(weights, 0, 1), dtype=dtype)
    clamped_events = clamped.reshape(event_times.size, relative_times.size).any(axis=1)
    return (
        lower,
        upper,
        weights,
        {
            "query_count": int(queries.size),
            "clamped_query_count": int(clamped.sum()),
            "before_start_count": int(before_start.sum()),
            "after_end_count": int(after_end.sum()),
            "events_with_clamped_queries": int(clamped_events.sum()),
        },
    )


def _read_roi_columns(dataset, columns, dtype):
    if columns.size == 1 or np.all(np.diff(columns) == 1):
        values = dataset[:, int(columns[0]) : int(columns[-1]) + 1]
    else:
        values = dataset[:, columns]
    return np.asarray(values, dtype=dtype)


def _interpolate_region(
    roi_h5,
    fly_id,
    roi_names,
    timestamps,
    event_times,
    relative_times,
    dtype,
    selected_indices=None,
):
    shapes = _region_shape(roi_h5, fly_id, roi_names)
    if shapes[0][1] != timestamps.size:
        raise ValueError("ROI activity and NWB timestamps have different lengths")
    total = sum(shape[2] for shape in shapes)
    if selected_indices is None:
        selected_indices = np.arange(total, dtype=np.int64)
    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    if selected_indices.ndim != 1 or selected_indices.size < 1:
        raise ValueError("selected target indices must be a non-empty vector")
    if selected_indices[0] < 0 or selected_indices[-1] >= total:
        raise ValueError("selected target index is outside the region")
    if not np.all(np.diff(selected_indices) > 0):
        raise ValueError("selected target indices must be increasing and unique")

    lower, upper, weights, interpolation_info = _interpolation_coordinates(
        timestamps, event_times, relative_times, dtype
    )
    output = np.empty(
        (event_times.size, relative_times.size, selected_indices.size), dtype=dtype
    )
    flat = output.reshape(-1, selected_indices.size)
    destination = 0
    region_start = 0
    with h5py.File(str(roi_h5), "r") as roi_file:
        group = roi_file[fly_id]
        for name, _, width in shapes:
            mask = (selected_indices >= region_start) & (
                selected_indices < region_start + width
            )
            local = selected_indices[mask] - region_start
            dataset = group[name]
            block_size = dataset.chunks[1] if dataset.chunks is not None else 512
            block_size = max(1, min(int(block_size), 2048))
            for start in range(0, local.size, block_size):
                columns = local[start : start + block_size]
                traces = _read_roi_columns(dataset, columns, dtype)
                values = traces[lower]
                values += (traces[upper] - values) * weights[:, None]
                stop = destination + columns.size
                flat[:, destination:stop] = values
                destination = stop
            region_start += width
    if destination != selected_indices.size:
        raise ValueError("failed to load every selected ROI column")
    return output, shapes, interpolation_info


def _validate_benchmark_arrays(source, target):
    if source.ndim not in (2, 3):
        raise ValueError("source must have 2 or 3 dimensions")
    if target.ndim != 3:
        raise ValueError("target must have shape trials x stimuli x targets")
    if source.ndim == 3 and source.shape[0] < 2:
        raise ValueError("trial-based source must contain at least two trials")
    if target.shape[0] < 2:
        raise ValueError("target must contain at least two trials")
    if source.shape[-2] != target.shape[1]:
        raise ValueError("source and target stimulus dimensions differ")
    if target.shape[1] < 2 or source.shape[-1] < 1 or target.shape[-1] < 1:
        raise ValueError("arrays have an empty feature or stimulus dimension")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("arrays must contain only finite values")


def load_fly_data(
    root,
    source_fly,
    target_fly,
    direction,
    region,
    event_source,
    dtype,
    target_limit=None,
    target_indices=None,
    events_dir=None,
):
    started = time.perf_counter()
    root = Path(root).resolve()
    roi_h5 = root / "roi_activity.h5"
    if not roi_h5.is_file():
        raise ValueError(f"ROI HDF5 does not exist: {roi_h5}")
    if region not in REGION_ROIS:
        raise ValueError(f"unsupported region: {region}")
    roi_names = REGION_ROIS[region]
    source_events, source_relative, source_event_info = _load_events(
        root, source_fly, direction, event_source, events_dir=events_dir
    )
    target_events, target_relative, target_event_info = _load_events(
        root, target_fly, direction, event_source, events_dir=events_dir
    )
    if not np.array_equal(source_relative, target_relative):
        raise ValueError("source and target use different relative-time grids")

    source_nwb = _nwb_for_fly(root, source_fly)
    target_nwb = _nwb_for_fly(root, target_fly)
    source_timestamps, source_nwb_info = _load_neural_timestamps(source_nwb)
    target_timestamps, target_nwb_info = _load_neural_timestamps(target_nwb)
    target_shapes = _region_shape(roi_h5, target_fly, roi_names)
    full_target_count = sum(shape[2] for shape in target_shapes)
    selected_targets, selected_target_list = _target_indices(
        full_target_count, target_limit, target_indices
    )

    source, source_shapes, source_interpolation = _interpolate_region(
        roi_h5,
        source_fly,
        roi_names,
        source_timestamps,
        source_events,
        source_relative,
        dtype,
    )
    target, _, target_interpolation = _interpolate_region(
        roi_h5,
        target_fly,
        roi_names,
        target_timestamps,
        target_events,
        target_relative,
        dtype,
        selected_indices=selected_targets,
    )
    _validate_benchmark_arrays(source, target)
    provenance = {
        "root": str(root),
        "dandiset": _dandiset_record(root),
        "roi_h5": _path_record(roi_h5),
        "source_fly": source_fly,
        "target_fly": target_fly,
        "direction": direction,
        "region": region,
        "roi_order": list(roi_names),
        "source_roi_shapes": [list(shape) for shape in source_shapes],
        "target_roi_shapes": [list(shape) for shape in target_shapes],
        "source_events": source_event_info,
        "target_events": target_event_info,
        "source_nwb": source_nwb_info,
        "target_nwb": target_nwb_info,
        "relative_times": {
            "count": int(source_relative.size),
            "start": float(source_relative[0]),
            "stop": float(source_relative[-1]),
        },
        "target_selection": {
            "is_slice": selected_target_list is not None,
            "full_count": full_target_count,
            "selected_count": int(selected_targets.size),
            "strategy": (
                "explicit"
                if target_indices is not None
                else "evenly_spaced"
                if selected_target_list is not None
                else "all"
            ),
            "indices": selected_target_list,
        },
        "interpolation": "linear with endpoint clamping",
        "interpolation_queries": {
            "source": source_interpolation,
            "target": target_interpolation,
        },
        "preparation_seconds": time.perf_counter() - started,
    }
    return source, target, provenance


def load_fixture(path, dtype):
    with np.load(path, allow_pickle=False) as fixture:
        missing = {"source", "target"}.difference(fixture.files)
        if missing:
            raise ValueError(
                "fixture is missing arrays: {}".format(", ".join(sorted(missing)))
            )
        source = np.asarray(fixture["source"], dtype=dtype)
        target = np.asarray(fixture["target"], dtype=dtype)
    _validate_benchmark_arrays(source, target)
    return source, target


def make_train_test_splits(strategy, num_stimuli, num_splits, train_frac, seed):
    if strategy == "random":
        if math.ceil(train_frac * num_stimuli) >= num_stimuli:
            raise ValueError("train_frac leaves no samples for prediction")
        return generate_train_test_splits(
            num_stim=num_stimuli,
            num_splits=num_splits,
            train_frac=train_frac,
            start_seed=seed,
        )
    if num_splits < 2:
        raise ValueError("blocked splits require at least two splits")
    if num_stimuli < num_splits:
        raise ValueError("blocked splits require at least one stimulus per split")
    indices = np.arange(num_stimuli, dtype=np.int64)
    folds = np.array_split(indices, num_splits)
    return [
        {
            "train": np.concatenate(
                [
                    block
                    for block_index, block in enumerate(folds)
                    if block_index != fold
                ]
            ),
            "test": folds[fold],
        }
        for fold in range(num_splits)
    ]


def _validate_fold_index(fold_index, num_splits):
    if fold_index is not None and not 0 <= fold_index < num_splits:
        raise ValueError(f"fold index must be between 0 and {num_splits - 1}")


def _validate_learned_cell(bootstrap_index, fold_index, config):
    if not 0 <= bootstrap_index < config["bootstrap_iters"]:
        raise ValueError(
            "learned bootstrap index must be between 0 and "
            f"{config['bootstrap_iters'] - 1}"
        )
    if not 0 <= fold_index < config["splits"]:
        raise ValueError(
            f"learned fold index must be between 0 and {config['splits'] - 1}"
        )


def _select_folds(splits, fold_index):
    _validate_fold_index(fold_index, len(splits))
    if fold_index is None:
        indices = list(range(len(splits)))
    else:
        indices = [fold_index]
    return [splits[index] for index in indices], indices


def _averaged_source(source):
    if source.ndim == 3:
        return source.mean(axis=0)
    return source


def _array_summary(array):
    array = np.asarray(array)
    summary = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "bytes": int(array.nbytes),
    }
    if array.nbytes <= PROVENANCE_HASH_LIMIT_BYTES:
        contiguous = np.ascontiguousarray(array)
        summary.update(
            {
                "sha256": hashlib.sha256(contiguous.view(np.uint8)).hexdigest(),
                "sha256_scope": "full_array_bytes",
            }
        )
    else:
        summary.update({"sha256": None, "sha256_scope": "omitted_size_limit"})
    return summary


def _attribute_dtype(objects, attribute):
    dtypes = set()
    for obj in objects:
        value = getattr(obj, attribute, None)
        if value is not None:
            dtypes.add(str(np.asarray(value).dtype))
    dtypes = sorted(dtypes)
    if not dtypes:
        return None
    return dtypes[0] if len(dtypes) == 1 else dtypes


def _ridge_dtype_report(mapper):
    direct = getattr(mapper, "_mapper", None)
    models = [direct] if direct is not None else list(getattr(mapper, "_mappers", []))
    return {
        "coefficient": _attribute_dtype(models, "coef_"),
        "intercept": _attribute_dtype(models, "intercept_"),
    }


def _metric_dtype_report(result):
    return {
        split: {
            name: str(np.asarray(values).dtype)
            for name, values in sorted(result[split].items())
        }
        for split in ("train", "test")
    }


def _prediction_summary(prediction):
    prediction = np.asarray(prediction)
    return {
        "shape": list(prediction.shape),
        "dtype": str(prediction.dtype),
        "checksum": float(np.asarray(prediction, dtype=np.float64).sum()),
        "mean": float(np.asarray(prediction, dtype=np.float64).mean()),
        "std": float(np.asarray(prediction, dtype=np.float64).std()),
    }


def _selected_source_report(counts):
    counts = np.asarray(counts, dtype=np.int64)
    report = {
        "selected_source_count_summary": {
            "target_count": int(counts.size),
            "minimum": int(counts.min()),
            "maximum": int(counts.max()),
            "mean": float(counts.mean()),
            "median": float(np.median(counts)),
            "total": int(counts.sum()),
            "per_target_values_recorded": counts.size <= SELECTED_SOURCE_DETAIL_LIMIT,
        }
    }
    if counts.size <= SELECTED_SOURCE_DETAIL_LIMIT:
        report["selected_sources_per_target"] = counts.tolist()
    return report


def _finite_summary(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return {
        "finite_count": int(finite.size),
        "total_count": int(values.size),
        "finite_fraction": float(finite.size) / float(values.size),
        "mean": float(finite.mean()) if finite.size else None,
        "median": float(np.median(finite)) if finite.size else None,
    }


def _consistency_metric_summary(metric):
    metric = np.asarray(metric, dtype=np.float64)
    if metric.ndim != 3:
        raise ValueError("consistency metrics must be bootstrap by fold by target")
    finite = np.isfinite(metric)
    counts = finite.sum(axis=0)
    bootstrap_means = np.full(counts.shape, np.nan, dtype=np.float64)
    np.divide(
        np.where(finite, metric, 0).sum(axis=0),
        counts,
        out=bootstrap_means,
        where=counts > 0,
    )
    by_fold = []
    for values in bootstrap_means:
        values = values[np.isfinite(values)]
        by_fold.append(float(np.median(values)) if values.size else None)
    finite_folds = [value for value in by_fold if value is not None]
    return {
        "bootstrap_mean_target_median_by_fold": by_fold,
        "median_across_folds": (
            float(np.median(finite_folds)) if finite_folds else None
        ),
        "global": _finite_summary(metric),
    }


def _consistency_metric_summaries(result):
    summaries = {
        "method": "bootstrap nanmean per target, then target nanmedian per fold"
    }
    for split in ("train", "test"):
        summaries[split] = {
            name: _consistency_metric_summary(metric)
            for name, metric in sorted(result[split].items())
        }
    return summaries


def benchmark_ridge(
    X_train,
    Y_train,
    X_test,
    backend,
    effective_backend,
    regression_kwargs,
    monitor_interval,
    gpu_index,
):
    monitor = ResourceMonitor(monitor_interval, effective_backend == "cuda", gpu_index)
    monitor.start()
    try:
        total_start = time.perf_counter()
        mapper = SKLinearNeuralMap(
            **_mapper_options(
                SKLinearNeuralMap, backend, effective_backend, regression_kwargs
            )
        )
        _, fit_seconds = _seconds(
            lambda: mapper.fit(X_train, Y_train), effective_backend
        )
        mapper_backend = _effective_mapper_backend(mapper, effective_backend)
        prediction, predict_seconds = _seconds(
            lambda: mapper.predict(X_test), effective_backend
        )
        total_seconds = time.perf_counter() - total_start
    finally:
        resources = monitor.stop()
    return {
        "backend_requested": backend,
        "backend_effective": mapper_backend,
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "total_seconds": total_seconds,
        "train_source_shape": list(X_train.shape),
        "train_target_shape": list(Y_train.shape),
        "test_source_shape": list(X_test.shape),
        "number_of_targets": int(Y_train.shape[1]),
        "ridge_parameters": dict(regression_kwargs),
        "prediction": _prediction_summary(prediction),
        "dtypes": {
            "input_source": str(np.asarray(X_train).dtype),
            "input_target": str(np.asarray(Y_train).dtype),
            "ridge": _ridge_dtype_report(mapper),
            "prediction": str(np.asarray(prediction).dtype),
        },
        "resources": resources,
    }


def benchmark_percentile(
    X_train,
    Y_train,
    X_test,
    backend,
    effective_backend,
    regression_kwargs,
    percentile,
    monitor_interval,
    gpu_index,
    extras=None,
):
    options = _mapper_options(
        PercentileNeuralMap,
        backend,
        effective_backend,
        regression_kwargs,
        extra=extras,
    )
    options.update({"percentile": percentile, "identity": False})
    if effective_backend == "cuda":
        options["store_correlations"] = False
    monitor = ResourceMonitor(monitor_interval, effective_backend == "cuda", gpu_index)
    monitor.start()
    try:
        total_start = time.perf_counter()
        mapper = TimedPercentileNeuralMap(effective_backend, **options)
        mapper.benchmark_phase = "fit"
        _, fit_seconds = _seconds(
            lambda: mapper.fit(X_train, Y_train), effective_backend
        )
        mapper_backend = _effective_mapper_backend(mapper, effective_backend)
        mapper.benchmark_phase = None
        prediction, predict_seconds = _seconds(
            lambda: mapper.predict(X_test), effective_backend
        )
        total_seconds = time.perf_counter() - total_start
    finally:
        resources = monitor.stop()
    preprocessing_seconds = mapper.correlation_and_source_selection_seconds
    ridge_fit = max(0.0, fit_seconds - preprocessing_seconds)
    result = {
        "backend_requested": backend,
        "backend_effective": mapper_backend,
        "correlation_and_source_selection_seconds": preprocessing_seconds,
        "ridge_fit_seconds": ridge_fit,
        "predict_seconds": predict_seconds,
        "fit_seconds": fit_seconds,
        "total_seconds": total_seconds,
        "correlation_and_source_selection_measurement": (
            "correlation, percentile threshold, and source selection"
        ),
        "ridge_fit_measurement": (
            "fit minus correlation, percentile threshold, and source selection"
        ),
        "percentile": percentile,
        "ridge_parameters": dict(regression_kwargs),
        "prediction": _prediction_summary(prediction),
        "dtypes": {
            "input_source": str(np.asarray(X_train).dtype),
            "input_target": str(np.asarray(Y_train).dtype),
            "correlation": (
                _dtype_name(mapper._correlation_dtype)
                if mapper_backend == "cuda" and mapper._correlation_dtype is not None
                else str(np.result_type(X_train, Y_train))
            ),
            "ridge": _ridge_dtype_report(mapper),
            "prediction": str(np.asarray(prediction).dtype),
        },
        "resources": resources,
    }
    result.update(_selected_source_report(mapper._num_source_units))
    if getattr(mapper, "_cuda_batch_report", None) is not None:
        result["cuda_batching"] = mapper._cuda_batch_report
    return result


def _save_precision_metrics(path, result, owner_path):
    payload = {}
    for split in ("train", "test"):
        block = result[split]
        for name, values in block.items():
            payload[f"{split}_{name}"] = np.asarray(values)
    atomic_savez(path, payload)
    return artifact_record(owner_path, path)


def _precision_cell_metrics(
    result,
    bootstrap_index=0,
    fold_position=0,
    target_positions=None,
):
    if target_positions is None:
        target_positions = slice(None)
    metrics = {"train": {}, "test": {}}
    for split, split_metrics in metrics.items():
        for name in CONSISTENCY_METRICS:
            values = np.asarray(result[split][name])
            if values.ndim != 3 or values.shape[0] < 1 or values.shape[1] < 1:
                raise ValueError(
                    "consistency result cannot provide precision cell metrics"
                )
            if bootstrap_index >= values.shape[0] or fold_position >= values.shape[1]:
                raise ValueError("requested precision metric cell is unavailable")
            split_metrics[name] = values[
                bootstrap_index, fold_position, target_positions
            ]
    return metrics


def _precision_trial_views(source, target, seed):
    source_full = generic_trial_avg(source) if source.ndim == 3 else source
    source_half_1, source_half_2 = get_splithalves(source, seed=seed)
    target_full = generic_trial_avg(target)
    target_half_1, target_half_2 = get_splithalves(target, seed=seed)
    return OrderedDict(
        (
            ("full", (source_full, target_full)),
            ("half_1", (source_half_1, target_half_1)),
            ("half_2", (source_half_2, target_half_2)),
        )
    )


def _save_learned_fit_diagnostic(
    path,
    owner_path,
    source,
    target,
    split,
    seed,
    mapper_options,
    effective_backend,
    metrics,
    target_indices,
    metadata,
):
    diagnostic_options = dict(mapper_options)
    diagnostic_options["store_correlations"] = True
    train_indices = np.asarray(split["train"], dtype=np.int64)
    test_indices = np.asarray(split["test"], dtype=np.int64)
    fits = OrderedDict()
    role_dtypes = {}
    for role, (role_source, role_target) in _precision_trial_views(
        source, target, seed
    ).items():
        mapper = PipelineNeuralMap(map_type="percentile", map_kwargs=diagnostic_options)
        X_train = role_source[train_indices]
        X_test = role_source[test_indices]
        Y_train = role_target[train_indices]
        Y_test = role_target[test_indices]
        mapper.fit(X_train, Y_train)
        fitted = mapper.map
        _effective_mapper_backend(fitted, effective_backend)
        fits[role] = {
            "mapper": mapper,
            "X_train": X_train,
            "X_test": X_test,
            "Y_train": Y_train,
            "Y_test": Y_test,
        }
        role_dtypes[role] = {
            "input_source": str(np.asarray(X_train).dtype),
            "input_target": str(np.asarray(Y_train).dtype),
            "correlation": str(np.asarray(fitted._corrs).dtype),
            "ridge": _ridge_dtype_report(fitted),
        }

    diagnostic_metadata = dict(metadata)
    diagnostic_dtypes = dict(diagnostic_metadata["dtypes"])
    diagnostic_dtypes.update(
        {
            "learned_fit": role_dtypes,
            "metrics": {
                split_name: {
                    name: str(np.asarray(values).dtype)
                    for name, values in split_metrics.items()
                }
                for split_name, split_metrics in metrics.items()
            },
        }
    )
    diagnostic_metadata["dtypes"] = diagnostic_dtypes
    return save_precision_diagnostic(
        path,
        fits,
        metrics,
        target_indices,
        train_indices,
        test_indices,
        diagnostic_metadata,
        owner_path=owner_path,
    )


def benchmark_consistency(
    source,
    target,
    splits,
    backend,
    effective_backend,
    regression_kwargs,
    bootstrap_iters,
    workers,
    seed,
    monitor_interval,
    gpu_index,
    consistency_map,
    percentile,
    extras=None,
    warmup=False,
    precision_metrics_path=None,
    precision_owner_path=None,
    learned_fit_path=None,
    learned_fit_target_indices=None,
    learned_fit_target_positions=None,
    learned_bootstrap_index=0,
    learned_fold_position=0,
    learned_fit_metadata=None,
):
    if consistency_map == "ridge":
        mapper_options = _mapper_options(
            SKLinearNeuralMap, backend, effective_backend, regression_kwargs
        )
        map_kwargs = {"map_type": "sklinear", "map_kwargs": mapper_options}
    else:
        mapper_options = _mapper_options(
            PercentileNeuralMap,
            backend,
            effective_backend,
            regression_kwargs,
            extra=extras,
        )
        mapper_options.update({"percentile": percentile, "identity": False})
        if effective_backend == "cuda":
            mapper_options["store_correlations"] = False
        map_kwargs = {"map_type": "percentile", "map_kwargs": mapper_options}
    neural_backends.set_last_cuda_batch_report(None)
    if warmup:
        get_linregress_consistency(
            source=source,
            target=target,
            map_kwargs=map_kwargs,
            num_bootstrap_iters=1,
            num_parallel_jobs=workers,
            start_seed=seed,
            metric="pearsonr",
            splits=splits,
        )
    monitor = ResourceMonitor(monitor_interval, effective_backend == "cuda", gpu_index)
    monitor.start()
    try:
        with record_phases() as phase_times:
            result, total_seconds = _seconds(
                lambda: get_linregress_consistency(
                    source=source,
                    target=target,
                    map_kwargs=map_kwargs,
                    num_bootstrap_iters=bootstrap_iters,
                    num_parallel_jobs=workers,
                    start_seed=seed,
                    metric="pearsonr",
                    splits=splits,
                ),
                effective_backend,
            )
    finally:
        resources = monitor.stop()
    summaries = _consistency_metric_summaries(result)
    primary = summaries["test"]["r_xy_n_sb"]
    metric = np.asarray(result["test"]["r_xy_n_sb"])
    payload = {
        "backend_requested": backend,
        "backend_effective": effective_backend,
        "total_seconds": total_seconds,
        "bootstrap_iterations": bootstrap_iters,
        "train_test_splits": len(splits),
        "workers": workers,
        "metric": "pearsonr",
        "mapping": map_kwargs,
        "result_shape": list(metric.shape),
        "metric_summaries": summaries,
        "finite_result_fraction": primary["global"]["finite_fraction"],
        "mean_test_r_xy_n_sb": primary["global"]["mean"],
        "ridge_parameters": dict(regression_kwargs),
        "selection_cutoff_tolerance": {
            "epsilon_multiplier": neural_backends.CORRELATION_CUTOFF_EPS_MULTIPLIER,
            "scale": "max(1, abs(percentile_threshold))",
        },
        "dtypes": {
            "input_source": str(np.asarray(source).dtype),
            "input_target": str(np.asarray(target).dtype),
            "correlation": (
                (
                    _dtype_name(extras["correlation_dtype"])
                    if effective_backend == "cuda"
                    else str(np.result_type(source, target))
                )
                if consistency_map == "percentile"
                and extras is not None
                and "correlation_dtype" in extras
                else None
            ),
            "ridge": {"coefficient": None, "intercept": None},
            "prediction": None,
            "metric": _metric_dtype_report(result),
        },
        "resources": resources,
        "phase_seconds": dict(phase_times),
        "warmup": bool(warmup),
    }
    report = neural_backends.last_cuda_batch_report()
    if report is not None:
        payload["cuda_batching"] = report
    if learned_fit_path:
        if (
            learned_fit_metadata is None
            or learned_fit_target_indices is None
            or learned_fit_target_positions is None
        ):
            raise ValueError("learned-fit diagnostics require provenance metadata")
        diagnostic_target = target[..., learned_fit_target_positions]
        diagnostic_seed = seed + learned_bootstrap_index
        payload["learned_fit"] = _save_learned_fit_diagnostic(
            learned_fit_path,
            precision_owner_path,
            source,
            diagnostic_target,
            splits[learned_fold_position],
            diagnostic_seed,
            mapper_options,
            effective_backend,
            _precision_cell_metrics(
                result,
                bootstrap_index=learned_bootstrap_index,
                fold_position=learned_fold_position,
                target_positions=learned_fit_target_positions,
            ),
            learned_fit_target_indices,
            learned_fit_metadata,
        )
    if precision_metrics_path:
        payload["precision_metrics"] = _save_precision_metrics(
            precision_metrics_path, result, precision_owner_path
        )
    return payload


def benchmark_transfers(
    host_array,
    requested_backend,
    effective_backend,
    monitor_interval,
    gpu_index,
    repeats,
):
    summary = {
        "backend_requested": requested_backend,
        "backend_effective": effective_backend,
        "measured": effective_backend == "cuda",
        "bytes_per_direction": int(host_array.nbytes),
        "dtype": str(np.asarray(host_array).dtype),
        "dtypes": {"input_source": str(np.asarray(host_array).dtype)},
        "repeats": repeats,
    }
    if effective_backend != "cuda":
        summary["reason"] = "effective backend is CPU"
        return summary

    import cupy

    host_array = np.ascontiguousarray(host_array)
    warmup = cupy.asarray(host_array[:1])
    cupy.asnumpy(warmup)
    cupy.cuda.get_current_stream().synchronize()
    monitor = ResourceMonitor(monitor_interval, True, gpu_index)
    monitor.start()
    h2d_seconds = []
    d2h_seconds = []
    try:
        restored = None
        for _ in range(repeats):
            device, elapsed = _seconds(
                lambda: cupy.asarray(host_array), effective_backend
            )
            h2d_seconds.append(elapsed)
            restored, elapsed = _seconds(
                lambda device=device: cupy.asnumpy(device), effective_backend
            )
            d2h_seconds.append(elapsed)
    finally:
        resources = monitor.stop()
    h2d_median = float(np.median(h2d_seconds))
    d2h_median = float(np.median(d2h_seconds))
    summary.update(
        {
            "warmup_performed": True,
            "h2d_seconds": h2d_seconds,
            "d2h_seconds": d2h_seconds,
            "h2d_median_seconds": h2d_median,
            "d2h_median_seconds": d2h_median,
            "h2d_gib_per_second": host_array.nbytes / h2d_median / 1024**3,
            "d2h_gib_per_second": host_array.nbytes / d2h_median / 1024**3,
            "roundtrip_checksum": float(np.asarray(restored, dtype=np.float64).sum()),
            "resources": resources,
        }
    )
    return summary


def _optional_library_versions():
    distributions = {
        "torch": ("torch",),
        "cupy": ("cupy", "cupy-cuda11x", "cupy-cuda12x"),
        "cuml": ("cuml", "cuml-cu11", "cuml-cu12"),
    }
    versions = {}
    try:
        from importlib import metadata

        get_version = metadata.version
        package_not_found = metadata.PackageNotFoundError
    except ImportError:
        try:
            import pkg_resources
        except ImportError:
            return versions

        def get_version(name):
            return pkg_resources.get_distribution(name).version

        package_not_found = pkg_resources.DistributionNotFound
    for library, candidates in distributions.items():
        for distribution in candidates:
            try:
                versions[library] = str(get_version(distribution))
                break
            except package_not_found:
                pass
    return versions


def _git_metadata():
    root = Path(__file__).resolve().parents[1]
    identity = repository_identity(root)
    identity["root"] = str(root)
    return identity


def _metadata(args, cuda_probe):
    libraries = {
        "h5py": h5py.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }
    libraries.update(_optional_library_versions())
    for library in ("torch", "cupy", "cuml"):
        version = cuda_probe.get(f"{library}_version")
        if version is not None:
            libraries[library] = str(version)
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": {
            "model": _cpu_model(),
            "logical_cpus": os.cpu_count(),
            "affinity_cpus": _affinity_count(),
        },
        "threads": {
            "benchmark_workers": args.workers,
            "environment": {
                name: os.environ.get(name)
                for name in THREAD_ENV_VARS
                if os.environ.get(name) is not None
            },
        },
        "libraries": libraries,
        "repository": _git_metadata(),
        "cuda_probe": cuda_probe,
        "gpu_devices": _gpu_inventory(),
        "host_inventory": collect_host_inventory(),
        "process_peak_rss_bytes_at_start": _peak_rss_bytes(),
    }


def _comparison_identity(result):
    dataset = result["dataset"]
    provenance = dataset.get("provenance") or {}
    dandiset = provenance.get("dandiset") or {}
    source_events = provenance.get("source_events") or {}
    return {
        "seed": result["seed"],
        "split_seed": result["split_seed"],
        "config": result["config"],
        "dataset_origin": dataset["origin"],
        "source": dataset["source"],
        "target": dataset["target"],
        "train_test_indices": dataset.get("train_test_indices"),
        "dandiset_identifier": dandiset.get("identifier"),
        "dandiset_version": dandiset.get("version"),
        "source_fly": provenance.get("source_fly"),
        "target_fly": provenance.get("target_fly"),
        "direction": provenance.get("direction"),
        "region": provenance.get("region"),
        "event_source": source_events.get("event_source"),
        "target_selection": provenance.get("target_selection"),
    }


def _reference_timing_comparison(current, reference):
    comparisons = {}
    common = set(current["benchmarks"]).intersection(reference["benchmarks"])
    for name in sorted(common):
        current_seconds = current["benchmarks"][name].get("total_seconds")
        reference_seconds = reference["benchmarks"][name].get("total_seconds")
        if current_seconds is None or reference_seconds is None:
            continue
        if (
            current_seconds <= 0
            or not np.isfinite([current_seconds, reference_seconds]).all()
        ):
            continue
        comparisons[name] = {
            "current_seconds": float(current_seconds),
            "reference_seconds": float(reference_seconds),
            "speedup": float(reference_seconds) / float(current_seconds),
        }
    return comparisons


def _reference_metric_comparison(current, reference):
    name = WORKLOAD_KEYS["consistency"]
    current_summary = current["benchmarks"].get(name, {}).get("metric_summaries")
    reference_summary = reference["benchmarks"].get(name, {}).get("metric_summaries")
    if current_summary is None or reference_summary is None:
        return {}
    comparisons = {}
    for split in ("train", "test"):
        comparisons[split] = {}
        common = set(current_summary[split]).intersection(reference_summary[split])
        for metric in sorted(common):
            current_value = current_summary[split][metric]["median_across_folds"]
            reference_value = reference_summary[split][metric]["median_across_folds"]
            if current_value is None or reference_value is None:
                continue
            difference = float(current_value) - float(reference_value)
            comparisons[split][metric] = {
                "current": float(current_value),
                "reference": float(reference_value),
                "difference": difference,
                "absolute_difference": abs(difference),
            }
    return comparisons


def _reference_comparison(current, path):
    try:
        with open(path, "r") as stream:
            reference = json.load(stream)
    except (OSError, ValueError) as error:
        raise ValueError(f"could not read reference result {path}: {error}")
    if not isinstance(reference, dict) or "benchmarks" not in reference:
        raise ValueError("reference result is not a benchmark result")

    current_identity = _comparison_identity(current)
    try:
        reference_identity = _comparison_identity(reference)
    except (KeyError, TypeError) as error:
        raise ValueError(f"reference result lacks comparison metadata: {error}")
    mismatches = [
        name
        for name, value in current_identity.items()
        if value != reference_identity[name]
    ]
    comparison = {
        "file": _path_record(path),
        "backend_effective": reference.get("backend_effective"),
        "configuration_matches": not mismatches,
        "mismatched_fields": mismatches,
        "speedup_definition": "reference seconds divided by current seconds",
        "benchmark_total_seconds": {},
        "consistency_metrics": {},
    }
    if not mismatches:
        comparison["benchmark_total_seconds"] = _reference_timing_comparison(
            current, reference
        )
        comparison["consistency_metrics"] = _reference_metric_comparison(
            current, reference
        )
    return comparison


def _suite_config(args):
    config = dict(SUITES[args.suite])
    overrides = {
        "stimuli": args.stimuli,
        "sources": args.sources,
        "targets": args.targets,
        "trials": args.trials,
        "latent_dim": args.latent_dim,
        "signal_sources": args.signal_sources,
        "bootstrap_iters": args.bootstrap_iters,
        "splits": args.splits,
    }
    for name, value in overrides.items():
        if value is not None:
            config[name] = value
    config.update(
        {
            "noise_std": args.noise_std,
            "train_frac": args.train_frac,
            "percentile": args.percentile,
        }
    )
    return config


def _validate_config(config, synthetic):
    positive = (
        "stimuli",
        "sources",
        "targets",
        "trials",
        "latent_dim",
        "signal_sources",
        "bootstrap_iters",
        "splits",
    )
    for name in positive:
        if config[name] < 1:
            raise ValueError(f"{name} must be positive")
    if config["stimuli"] < 4:
        raise ValueError("stimuli must be at least 4")
    if config["trials"] < 2:
        raise ValueError("trials must be at least 2")
    if synthetic:
        if config["latent_dim"] > config["sources"]:
            raise ValueError("latent_dim cannot exceed sources")
        if config["signal_sources"] > config["sources"]:
            raise ValueError("signal_sources cannot exceed sources")
    if not 0 < config["train_frac"] < 1:
        raise ValueError("train_frac must be between 0 and 1")
    if not 0 <= config["percentile"] <= 100:
        raise ValueError("percentile must be between 0 and 100")


def _dataset_target_indices(dataset):
    count = int(dataset["target"]["shape"][-1])
    provenance = dataset.get("provenance") or {}
    selection = provenance.get("target_selection") or {}
    indices = selection.get("indices")
    if indices is None:
        return np.arange(count, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)
    if indices.shape != (count,):
        raise ValueError("target-selection provenance does not match target data")
    return indices


def _learned_fit_target_selection(dataset, target_limit, explicit_indices=None):
    target_indices = _dataset_target_indices(dataset)
    if target_limit is not None and explicit_indices is not None:
        raise ValueError(
            "learned target limit and learned target indices are mutually exclusive"
        )
    if explicit_indices is None:
        positions, _ = _target_indices(target_indices.size, target_limit)
        strategy = "all" if positions.size == target_indices.size else "evenly_spaced"
    else:
        selected_ids = np.asarray(explicit_indices, dtype=np.int64)
        if selected_ids.ndim != 1 or selected_ids.size < 1:
            raise ValueError("learned target indices must be a non-empty vector")
        if selected_ids[0] < 0 or not np.all(np.diff(selected_ids) > 0):
            raise ValueError("learned target indices must be sorted and unique")
        positions = np.searchsorted(target_indices, selected_ids)
        valid = positions < target_indices.size
        valid[valid] &= target_indices[positions[valid]] == selected_ids[valid]
        if not valid.all():
            missing = selected_ids[~valid]
            raise ValueError(
                "learned target indices are not in the main target selection: "
                + ", ".join(str(value) for value in missing)
            )
        strategy = "explicit"
    selected = target_indices[positions]
    selection = {
        "strategy": strategy,
        "full_target_count": int(target_indices.size),
        "selected_target_count": int(positions.size),
        "positions": positions.tolist(),
        "original_target_ids": selected.tolist(),
    }
    return positions, selected, selection


def _learned_fit_dataset(dataset, target_selection):
    learned_dataset = copy.deepcopy(dataset)
    provenance = learned_dataset.get("provenance")
    if isinstance(provenance, dict):
        provenance.pop("preparation_seconds", None)
    learned_dataset["diagnostic_target_selection"] = dict(target_selection)
    return learned_dataset


def _learned_fit_provenance(
    args,
    result,
    regression_kwargs,
    effective_backend,
    percentile_extras,
    bootstrap_index,
    fold_index,
    target_selection,
):
    effective_dtypes = result["dtypes"]["effective"]
    return {
        "configuration": {
            "suite": result["suite"],
            "workloads": list(result["workloads"]),
            "config": dict(result["config"]),
            "metric": "pearsonr",
            "mapping": {
                "map_type": "percentile",
                "percentile": result["config"]["percentile"],
                "identity": False,
                "regression_type": "Ridge",
                "regression_kwargs": dict(regression_kwargs),
                "selection_cutoff_tolerance": {
                    "epsilon_multiplier": (
                        neural_backends.CORRELATION_CUTOFF_EPS_MULTIPLIER
                    ),
                    "scale": "max(1, abs(percentile_threshold))",
                },
            },
        },
        "dataset": _learned_fit_dataset(result["dataset"], target_selection),
        "seed": {
            "bootstrap_start": args.seed,
            "bootstrap": args.seed + bootstrap_index,
            "split": args.split_seed,
        },
        "bootstrap_index": bootstrap_index,
        "fold_index": fold_index,
        "dtypes": {
            "requested": dict(result["dtypes"]["requested"]),
            "effective": {
                "preparation": dict(effective_dtypes["preparation"]),
                "mapper_input": dict(effective_dtypes["mapper_input"]),
            },
        },
        "backend_requested": args.backend,
        "backend_effective": effective_backend,
        "repository": result["metadata"]["repository"],
        "execution": {
            "workers": args.workers,
            "gpu_index": args.gpu_index,
            "warmup": bool(getattr(args, "warmup", False)),
            "store_correlations": True,
            "percentile_options": dict(percentile_extras),
        },
    }


def run_benchmarks(args):
    if getattr(args, "inventory_only", False):
        return {
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "inventory_only": True,
            "metadata": _metadata(args, _cuda_probe(args.gpu_index)),
        }
    config = _suite_config(args)
    requested_fold_index = getattr(args, "fold_index", None)
    _validate_fold_index(requested_fold_index, config["splits"])
    learned_bootstrap_index = getattr(args, "learned_bootstrap_index", 0)
    learned_fold_index = getattr(args, "learned_fold_index", 0)
    _validate_learned_cell(learned_bootstrap_index, learned_fold_index, config)
    precision = _precision_plan(args)
    if args.fixture and args.fly_data_root:
        raise ValueError("--fixture and --fly-data-root are mutually exclusive")
    if args.target_limit is not None and not args.fly_data_root:
        raise ValueError("--target-limit requires --fly-data-root")
    if getattr(args, "target_indices", None) is not None and not args.fly_data_root:
        raise ValueError("--target-indices requires --fly-data-root")

    workloads = []
    for workload in args.workloads:
        if workload not in WORKLOADS:
            raise ValueError(f"unknown workload: {workload}")
        if workload not in workloads:
            workloads.append(workload)
    if not workloads:
        raise ValueError("at least one workload is required")
    _validate_artifact_options(args, workloads)
    if (
        getattr(args, "save_learned_fit", None)
        and requested_fold_index is not None
        and learned_fold_index != requested_fold_index
    ):
        raise ValueError(
            "--learned-fold-index must match --fold-index when both are used"
        )
    percentile_extras = _percentile_extras(args, precision["correlation"])

    regression_kwargs = {
        "alpha": args.alpha,
        "fit_intercept": args.fit_intercept,
    }
    if args.solver is not None:
        regression_kwargs["solver"] = args.solver

    effective_backends = _resolve_workload_backends(
        args.backend,
        workloads,
        args.consistency_map,
        regression_kwargs,
        args.gpu_index,
    )
    _verify_explicit_backend(args.backend, workloads, effective_backends)
    if (
        "consistency" in workloads
        and effective_backends["consistency"] == "cuda"
        and args.workers != 1
    ):
        raise ValueError("CUDA consistency benchmarks require --workers 1")

    cuda_workloads = [
        workload
        for workload, resolved in effective_backends.items()
        if resolved == "cuda"
    ]
    if cuda_workloads:
        require_torch = "percentile" in cuda_workloads or (
            "consistency" in cuda_workloads and args.consistency_map == "percentile"
        )
        _select_cuda_device(args.gpu_index, require_torch)
    cuda_probe = (
        _cuda_probe(args.gpu_index)
        if args.backend != "cpu"
        else {
            "available": None,
            "device_index": args.gpu_index,
            "reason": "not probed for CPU benchmark",
        }
    )

    data_provenance = None
    if args.fly_data_root:
        if args.region not in REGION_ROIS:
            raise ValueError(f"unsupported region: {args.region}")
        source, target, data_provenance = load_fly_data(
            args.fly_data_root,
            args.source_fly,
            args.target_fly,
            args.direction,
            args.region,
            args.event_source,
            precision["preparation"],
            target_limit=args.target_limit,
            target_indices=getattr(args, "target_indices", None),
            events_dir=getattr(args, "events_dir", None),
        )
        data_origin = "brezovec_fly"
    elif args.fixture:
        source, target = load_fixture(args.fixture, precision["preparation"])
        data_origin = "fixture"
        data_provenance = {"file": _path_record(args.fixture)}
    else:
        _validate_config(config, synthetic=True)
        source, target = make_correlated_data(
            config, args.seed, precision["preparation"]
        )
        data_origin = "correlated_synthetic"
    preparation_dtypes = {
        "source": str(source.dtype),
        "target": str(target.dtype),
    }
    if precision["mapper_input"] != source.dtype:
        source = np.asarray(source, dtype=precision["mapper_input"])
        target = np.asarray(target, dtype=precision["mapper_input"])
    config.update(
        {
            "stimuli": target.shape[1],
            "sources": source.shape[-1],
            "targets": target.shape[-1],
            "trials": target.shape[0],
            "split_strategy": args.split_strategy,
            "consistency_map": args.consistency_map,
        }
    )
    _validate_config(config, synthetic=data_origin == "correlated_synthetic")
    backend_values = set(effective_backends.values())
    effective_backend = (
        next(iter(backend_values)) if len(backend_values) == 1 else "mixed"
    )

    generated_splits = make_train_test_splits(
        args.split_strategy,
        config["stimuli"],
        config["splits"],
        config["train_frac"],
        args.split_seed,
    )
    splits, fold_indices = _select_folds(generated_splits, requested_fold_index)
    learned_fold_position = (
        fold_indices.index(learned_fold_index)
        if learned_fold_index in fold_indices
        else None
    )
    first_split = splits[0]
    source_average = None
    target_average = None
    X_train = None
    Y_train = None
    X_test = None
    if any(name in workloads for name in ("transfer", "ridge", "percentile")):
        source_average = _averaged_source(source)
    if any(name in workloads for name in ("ridge", "percentile")):
        target_average = target.mean(axis=0)
        X_train = source_average[first_split["train"]]
        Y_train = target_average[first_split["train"]]
        X_test = source_average[first_split["test"]]
    result = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "suite": args.suite,
        "backend_requested": args.backend,
        "backend_effective": effective_backend,
        "backend_effective_by_workload": effective_backends,
        "dtypes": {
            "requested": {
                "preparation": _dtype_name(precision["preparation"]),
                "mapper_input": _dtype_name(precision["mapper_input"]),
                "correlation": _dtype_name(precision["correlation"]),
                "compute_override_explicit": precision["compute_explicit"],
                "correlation_override_explicit": precision["correlation_explicit"],
                "mixed_precision_allowed": precision["allow_mixed_precision"],
            },
            "effective": {
                "preparation": preparation_dtypes,
                "mapper_input": {
                    "source": str(source.dtype),
                    "target": str(target.dtype),
                },
                "by_workload": {},
            },
        },
        "seed": args.seed,
        "split_seed": args.split_seed,
        "workloads": workloads,
        "config": config,
        "metadata": _metadata(args, cuda_probe),
        "dataset": {
            "origin": data_origin,
            "fixture": os.path.abspath(args.fixture) if args.fixture else None,
            "provenance": data_provenance,
            "source": _array_summary(source),
            "target": _array_summary(target),
            "split_strategy": args.split_strategy,
            "selected_fold_indices": fold_indices,
            "train_test_indices": [
                {
                    "fold_index": fold_index,
                    "train": split["train"].astype(int).tolist(),
                    "test": split["test"].astype(int).tolist(),
                }
                for fold_index, split in zip(fold_indices, splits)
            ],
            "standalone_train_samples": int(first_split["train"].size),
            "standalone_test_samples": int(first_split["test"].size),
        },
        "benchmarks": {},
    }
    if "transfer" in workloads:
        transfer_backend = effective_backends["transfer"]
        result["benchmarks"][WORKLOAD_KEYS["transfer"]] = benchmark_transfers(
            source_average,
            args.backend,
            transfer_backend,
            args.monitor_interval,
            args.gpu_index,
            args.transfer_repeats,
        )
    if "ridge" in workloads:
        ridge_backend = effective_backends["ridge"]
        result["benchmarks"][WORKLOAD_KEYS["ridge"]] = benchmark_ridge(
            X_train,
            Y_train,
            X_test,
            args.backend,
            ridge_backend,
            regression_kwargs,
            args.monitor_interval,
            args.gpu_index,
        )
    if "percentile" in workloads:
        percentile_backend = effective_backends["percentile"]
        result["benchmarks"][WORKLOAD_KEYS["percentile"]] = benchmark_percentile(
            X_train,
            Y_train,
            X_test,
            args.backend,
            percentile_backend,
            regression_kwargs,
            config["percentile"],
            args.monitor_interval,
            args.gpu_index,
            extras=percentile_extras,
        )
    if "consistency" in workloads:
        consistency_backend = effective_backends["consistency"]
        learned_fit_path = getattr(args, "save_learned_fit", None)
        learned_target_positions = None
        learned_target_indices = None
        learned_target_selection = None
        if learned_fit_path:
            (
                learned_target_positions,
                learned_target_indices,
                learned_target_selection,
            ) = _learned_fit_target_selection(
                result["dataset"],
                getattr(args, "learned_fit_target_limit", None),
                getattr(args, "learned_target_indices", None),
            )
        learned_fit_metadata = (
            _learned_fit_provenance(
                args,
                result,
                regression_kwargs,
                consistency_backend,
                percentile_extras,
                learned_bootstrap_index,
                learned_fold_index,
                learned_target_selection,
            )
            if learned_fit_path
            else None
        )
        result["benchmarks"][WORKLOAD_KEYS["consistency"]] = benchmark_consistency(
            source,
            target,
            splits,
            args.backend,
            consistency_backend,
            regression_kwargs,
            config["bootstrap_iters"],
            args.workers,
            args.seed,
            args.monitor_interval,
            args.gpu_index,
            args.consistency_map,
            config["percentile"],
            extras=percentile_extras,
            warmup=getattr(args, "warmup", False),
            precision_metrics_path=getattr(args, "save_precision_artifacts", None),
            precision_owner_path=args.output,
            learned_fit_path=learned_fit_path,
            learned_fit_target_indices=learned_target_indices,
            learned_fit_target_positions=learned_target_positions,
            learned_bootstrap_index=learned_bootstrap_index,
            learned_fold_position=learned_fold_position or 0,
            learned_fit_metadata=learned_fit_metadata,
        )
    result["dtypes"]["effective"]["by_workload"] = {
        name: payload.get("dtypes", {})
        for name, payload in sorted(result["benchmarks"].items())
    }
    result["metadata"]["process_peak_rss_bytes_at_end"] = _peak_rss_bytes()
    if args.reference_result:
        result["reference_comparison"] = _reference_comparison(
            result, args.reference_result
        )
    return result


def build_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark Ridge, percentile mapping, and neural consistency."
    )
    parser.add_argument("--suite", choices=sorted(SUITES), default="smoke")
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--fixture", help="NPZ containing source and target arrays")
    parser.add_argument("--fly-data-root")
    parser.add_argument(
        "--events-dir",
        help="optional directory with {fly}_{direction}.npz event files",
    )
    parser.add_argument("--source-fly", default="fly_087")
    parser.add_argument("--target-fly", default="fly_089")
    parser.add_argument("--direction", choices=("left", "right"), default="right")
    parser.add_argument("--region", choices=sorted(REGION_ROIS), default="IPS")
    parser.add_argument(
        "--event-source", choices=("validated", "historical"), default="validated"
    )
    parser.add_argument("--target-limit", type=int)
    parser.add_argument(
        "--target-indices",
        type=_parse_target_indices,
        help="sorted comma-separated global target indices for fly data",
    )
    parser.add_argument(
        "--split-strategy", choices=("random", "blocked"), default="random"
    )
    parser.add_argument(
        "--fold-index",
        type=int,
        help="run only this generated train/test fold",
    )
    parser.add_argument(
        "--consistency-map", choices=("ridge", "percentile"), default="ridge"
    )
    parser.add_argument(
        "--workloads", nargs="+", choices=WORKLOADS, default=list(WORKLOADS)
    )
    parser.add_argument("--output", default="-", help="JSON path, or - for stdout")
    parser.add_argument("--reference-result", help="prior benchmark JSON to compare")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--solver")
    parser.add_argument("--fit-intercept", action="store_true", default=True)
    parser.add_argument(
        "--no-fit-intercept", action="store_false", dest="fit_intercept"
    )
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--percentile", type=float, default=95.0)
    parser.add_argument("--noise-std", type=float, default=0.2)
    parser.add_argument("--monitor-interval", type=float, default=0.05)
    parser.add_argument("--transfer-repeats", type=int, default=3)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument(
        "--correlation-target-batch-size",
        default="512",
        help="CUDA correlation target chunk, or 'all'",
    )
    parser.add_argument("--ridge-workspace-bytes", type=int)
    parser.add_argument(
        "--ridge-max-batch-size",
        help="CUDA Ridge target cap, or 'all' to let workspace control batch size",
    )
    parser.add_argument(
        "--correlation-dtype",
        choices=("float32", "float64"),
        help="correlation compute dtype; defaults to the mapper input dtype",
    )
    parser.add_argument(
        "--compute-dtype",
        choices=("float32", "float64"),
        help="diagnostic mapper-input override; normally --dtype is authoritative",
    )
    parser.add_argument(
        "--allow-mixed-precision",
        action="store_true",
        help="allow diagnostic dtype overrides that differ from --dtype",
    )
    parser.add_argument("--save-precision-artifacts")
    parser.add_argument(
        "--save-learned-fit",
        help="NPZ path for one learned-fit diagnostic cell",
    )
    parser.add_argument(
        "--learned-fit-target-limit",
        type=int,
        help="evenly spaced target count saved only in the learned-fit artifact",
    )
    parser.add_argument(
        "--learned-target-indices",
        type=_parse_target_indices,
        help="sorted comma-separated global target IDs saved in the learned-fit artifact",
    )
    parser.add_argument("--learned-bootstrap-index", type=int, default=0)
    parser.add_argument("--learned-fold-index", type=int, default=0)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--stimuli", type=int)
    parser.add_argument("--sources", type=int)
    parser.add_argument("--targets", type=int)
    parser.add_argument("--trials", type=int)
    parser.add_argument("--latent-dim", type=int)
    parser.add_argument("--signal-sources", type=int)
    parser.add_argument("--bootstrap-iters", type=int)
    parser.add_argument("--splits", type=int)
    return parser


def _validate_args(parser, args):
    if args.fixture and args.fly_data_root:
        parser.error("--fixture and --fly-data-root are mutually exclusive")
    if args.target_limit is not None and args.target_limit < 1:
        parser.error("--target-limit must be positive")
    if args.target_limit is not None and not args.fly_data_root:
        parser.error("--target-limit requires --fly-data-root")
    if args.target_indices is not None and not args.fly_data_root:
        parser.error("--target-indices requires --fly-data-root")
    if args.target_limit is not None and args.target_indices is not None:
        parser.error("--target-limit and --target-indices are mutually exclusive")
    if args.fold_index is not None and args.fold_index < 0:
        parser.error("--fold-index cannot be negative")
    if args.learned_fit_target_limit is not None and args.learned_fit_target_limit < 1:
        parser.error("--learned-fit-target-limit must be positive")
    if args.learned_bootstrap_index < 0:
        parser.error("--learned-bootstrap-index cannot be negative")
    if args.learned_fold_index < 0:
        parser.error("--learned-fold-index cannot be negative")
    if args.workers == 0:
        parser.error("--workers cannot be zero")
    if args.monitor_interval <= 0:
        parser.error("--monitor-interval must be positive")
    if args.transfer_repeats < 1:
        parser.error("--transfer-repeats must be positive")
    if args.gpu_index < 0:
        parser.error("--gpu-index cannot be negative")
    batch_size = getattr(args, "correlation_target_batch_size", None)
    if batch_size is not None and str(batch_size).lower() != "all":
        try:
            if int(batch_size) < 1:
                parser.error(
                    "--correlation-target-batch-size must be positive or 'all'"
                )
        except (TypeError, ValueError):
            parser.error("--correlation-target-batch-size must be an integer or 'all'")
    workspace = getattr(args, "ridge_workspace_bytes", None)
    if workspace is not None and workspace < 1:
        parser.error("--ridge-workspace-bytes must be positive")
    try:
        _precision_plan(args)
        _validate_artifact_options(args, args.workloads)
    except ValueError as error:
        parser.error(str(error))


def _write_result_json(output, result):
    if output == "-":
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return
    atomic_write_json(output, result)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    try:
        result = run_benchmarks(args)
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    _write_result_json(args.output, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
