"""Reproducible neural mapper and consistency benchmarks."""

import argparse
import csv
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
from pathlib import Path

import h5py
import numpy as np
import scipy
import sklearn

from brainmodel_utils.metrics.consistency import get_linregress_consistency
from brainmodel_utils.neural_mappers import PercentileNeuralMap, SKLinearNeuralMap
from brainmodel_utils.neural_mappers import backends as neural_backends
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
REGION_ROIS = {"IPS": ("IPS_L", "IPS_R")}
NEURAL_SERIES_PATH = "processing/ophys/TwoPhotonSeriesFunctionalGreenProcessed"
SELECTED_SOURCE_DETAIL_LIMIT = 256

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
        if backend == "cuda":
            raise RuntimeError(f"CUDA transfer backend is unavailable: {error}")
        return "cpu"
    if gpu_index < device_count:
        return "cuda"
    if backend == "cuda":
        raise RuntimeError(
            f"CUDA device index {gpu_index} is unavailable; {device_count} devices are visible"
        )
    return "cpu"


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
        self.method = None
        self.error = None
        self._pynvml = None
        self._handle = None
        try:
            import pynvml
        except ImportError as error:
            if shutil.which("nvidia-smi") is not None:
                self.method = "nvidia-smi"
            else:
                self.error = str(error)
            return
        try:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self._pynvml = pynvml
            self.method = "nvml"
        except (pynvml.NVMLError, OSError) as error:
            if shutil.which("nvidia-smi") is not None:
                self.method = "nvidia-smi"
            else:
                self.error = str(error)

    def sample(self):
        if self.method == "nvml":
            try:
                memory = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                utilization = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            except self._pynvml.NVMLError as error:
                self.error = str(error)
                return None, None
            return int(memory.used), float(utilization.gpu)
        if self.method == "nvidia-smi":
            output = _run_nvidia_smi(
                [
                    f"--id={self.device_index}",
                    "--query-gpu=memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ]
            )
            if output:
                try:
                    row = next(csv.reader([output], skipinitialspace=True))
                    return int(float(row[0]) * 1024 * 1024), float(row[1])
                except (IndexError, ValueError) as error:
                    self.error = str(error)
        return None, None

    def close(self):
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except self._pynvml.NVMLError as error:
                self.error = str(error)


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
        self.error = None
        self.cpu_start = None
        self.cpu_tick_baseline = {}
        self.cpu_tick_latest = {}
        self.wall_start = None

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
        if self.gpu_reader is not None:
            memory, utilization = self.gpu_reader.sample()
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
        gpu_result = {
            "sampled": self.gpu_reader is not None,
            "method": self.gpu_reader.method if self.gpu_reader is not None else None,
            "device_index": self.gpu_reader.device_index
            if self.gpu_reader is not None
            else None,
            "samples": self.samples if self.gpu_reader is not None else 0,
            "memory_scope": "whole_device",
            "memory_start_bytes": self.gpu_start_memory,
            "peak_memory_used_bytes": self.peak_gpu_memory,
            "peak_memory_increase_bytes": _positive_delta(
                self.peak_gpu_memory, self.gpu_start_memory
            ),
            "peak_utilization_percent": self.peak_gpu_utilization,
            "error": self.error
            or (self.gpu_reader.error if self.gpu_reader is not None else None),
        }
        return {
            "memory_scope": "driver_and_descendant_processes",
            "rss_start_bytes": baseline,
            "peak_rss_bytes": peak_tree,
            "peak_rss_increase_bytes": _positive_delta(peak_tree, baseline),
            "peak_driver_rss_bytes": self.peak_driver_rss or None,
            "peak_child_rss_bytes": max(0, (peak_tree or 0) - self.peak_driver_rss)
            if peak_tree is not None
            else None,
            "gpu": gpu_result,
            "monitor_error": self.error,
            "wall_seconds": wall_seconds,
            "cpu": cpu,
        }


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
    mapper_class, requested_backend, effective_backend, regression_kwargs
):
    options = {
        "regression_type": "Ridge",
        "regression_kwargs": regression_kwargs,
    }
    if _supports_backend(mapper_class):
        options["backend"] = requested_backend
    elif effective_backend != "cpu":
        raise RuntimeError(
            f"{mapper_class.__name__} does not expose the requested backend API"
        )
    return options


class TimedPercentileNeuralMap(PercentileNeuralMap):
    def __init__(self, benchmark_backend, *args, **kwargs):
        self.benchmark_backend = benchmark_backend
        self.correlation_seconds = 0.0
        self.source_selection_seconds = {"fit": 0.0, "predict": 0.0}
        self.benchmark_phase = None
        super().__init__(*args, **kwargs)

    def _gather_correlations(self, X, Y):
        _synchronize(self.benchmark_backend)
        start = time.perf_counter()
        super()._gather_correlations(X, Y)
        _synchronize(self.benchmark_backend)
        self.correlation_seconds += time.perf_counter() - start

    def _get_sources(self, target_idx):
        start = time.perf_counter()
        sources = super()._get_sources(target_idx)
        elapsed = time.perf_counter() - start
        if self.benchmark_phase in self.source_selection_seconds:
            self.source_selection_seconds[self.benchmark_phase] += elapsed
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


def _path_record(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


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


def _event_files(root, fly_id, direction):
    pilot = Path(root) / "inter_animal_consistency" / "two_fly_pilot"
    window = pilot / "event_tensors" / "short" / f"{fly_id}_{direction}.npz"
    bouts = pilot / "events" / f"{fly_id}_turn_bouts.csv"
    return window, bouts


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


def _load_events(root, fly_id, direction, event_source):
    if event_source not in ("validated", "historical"):
        raise ValueError(f"unknown event source: {event_source}")
    window_path, bouts_path = _event_files(root, fly_id, direction)
    if not window_path.is_file():
        raise ValueError(f"event window does not exist: {window_path}")
    with np.load(str(window_path), allow_pickle=False) as window:
        missing = {"event_times", "relative_times"}.difference(window.files)
        if missing:
            raise ValueError(
                "event window is missing arrays: {}".format(", ".join(sorted(missing)))
            )
        relative_times = np.asarray(window["relative_times"], dtype=np.float64)
        if event_source == "validated":
            event_times = np.asarray(window["event_times"], dtype=np.float64)
        else:
            if not bouts_path.is_file():
                raise ValueError(f"event CSV does not exist: {bouts_path}")
            event_times = _historical_event_times(bouts_path, direction)
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
        provenance["event_file"] = _path_record(bouts_path)
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


def _target_indices(total, limit):
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
        root, source_fly, direction, event_source
    )
    target_events, target_relative, target_event_info = _load_events(
        root, target_fly, direction, event_source
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
        full_target_count, target_limit
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
            "strategy": "evenly_spaced" if selected_target_list is not None else "all",
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


def _averaged_source(source):
    if source.ndim == 3:
        return source.mean(axis=0)
    return source


def _array_summary(array):
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "bytes": int(array.nbytes),
    }


def _prediction_summary(prediction):
    return {
        "shape": list(prediction.shape),
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
        prediction, predict_seconds = _seconds(
            lambda: mapper.predict(X_test), effective_backend
        )
        total_seconds = time.perf_counter() - total_start
    finally:
        resources = monitor.stop()
    return {
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "total_seconds": total_seconds,
        "train_source_shape": list(X_train.shape),
        "train_target_shape": list(Y_train.shape),
        "test_source_shape": list(X_test.shape),
        "number_of_targets": int(Y_train.shape[1]),
        "ridge_parameters": dict(regression_kwargs),
        "prediction": _prediction_summary(prediction),
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
):
    options = _mapper_options(
        PercentileNeuralMap, backend, effective_backend, regression_kwargs
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
        mapper.benchmark_phase = "predict"
        prediction, predict_seconds = _seconds(
            lambda: mapper.predict(X_test), effective_backend
        )
        total_seconds = time.perf_counter() - total_start
    finally:
        resources = monitor.stop()
    selection_fit = mapper.source_selection_seconds["fit"]
    ridge_fit = max(0.0, fit_seconds - mapper.correlation_seconds - selection_fit)
    result = {
        "correlation_seconds": mapper.correlation_seconds,
        "source_selection_seconds": selection_fit,
        "ridge_fit_seconds": ridge_fit,
        "predict_seconds": predict_seconds,
        "predict_source_selection_seconds": mapper.source_selection_seconds["predict"],
        "fit_seconds": fit_seconds,
        "total_seconds": total_seconds,
        "ridge_fit_measurement": "fit minus correlation and source selection",
        "percentile": percentile,
        "ridge_parameters": dict(regression_kwargs),
        "prediction": _prediction_summary(prediction),
        "resources": resources,
    }
    result.update(_selected_source_report(mapper._num_source_units))
    return result


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
):
    if consistency_map == "ridge":
        mapper_options = _mapper_options(
            SKLinearNeuralMap, backend, effective_backend, regression_kwargs
        )
        map_kwargs = {"map_type": "sklinear", "map_kwargs": mapper_options}
    else:
        mapper_options = _mapper_options(
            PercentileNeuralMap, backend, effective_backend, regression_kwargs
        )
        mapper_options.update({"percentile": percentile, "identity": False})
        if effective_backend == "cuda":
            mapper_options["store_correlations"] = False
        map_kwargs = {"map_type": "percentile", "map_kwargs": mapper_options}
    monitor = ResourceMonitor(monitor_interval, effective_backend == "cuda", gpu_index)
    monitor.start()
    try:
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
    return {
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
        "resources": resources,
    }


def benchmark_transfers(
    host_array, effective_backend, monitor_interval, gpu_index, repeats
):
    summary = {
        "measured": effective_backend == "cuda",
        "bytes_per_direction": int(host_array.nbytes),
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

    def git_output(*arguments):
        try:
            completed = subprocess.run(  # noqa: UP022 -- Python 3.6 support
                ["git", "-C", str(root)] + list(arguments),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,  # noqa: UP021 -- Python 3.6 support
                check=False,
            )
        except OSError:
            return None
        if completed.returncode != 0:
            return None
        return completed.stdout.strip()

    commit = git_output("rev-parse", "HEAD")
    status = git_output("status", "--porcelain")
    return {
        "root": str(root),
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
    }


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


def run_benchmarks(args):
    config = _suite_config(args)
    dtype = np.dtype(args.dtype)
    if args.fixture and args.fly_data_root:
        raise ValueError("--fixture and --fly-data-root are mutually exclusive")
    if args.target_limit is not None and not args.fly_data_root:
        raise ValueError("--target-limit requires --fly-data-root")

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
            dtype,
            target_limit=args.target_limit,
        )
        data_origin = "brezovec_fly"
    elif args.fixture:
        source, target = load_fixture(args.fixture, dtype)
        data_origin = "fixture"
        data_provenance = {"file": _path_record(args.fixture)}
    else:
        _validate_config(config, synthetic=True)
        source, target = make_correlated_data(config, args.seed, dtype)
        data_origin = "correlated_synthetic"
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

    workloads = []
    for workload in args.workloads:
        if workload not in WORKLOADS:
            raise ValueError(f"unknown workload: {workload}")
        if workload not in workloads:
            workloads.append(workload)
    if not workloads:
        raise ValueError("at least one workload is required")

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
    backend_values = set(effective_backends.values())
    effective_backend = (
        next(iter(backend_values)) if len(backend_values) == 1 else "mixed"
    )

    splits = make_train_test_splits(
        args.split_strategy,
        config["stimuli"],
        config["splits"],
        config["train_frac"],
        args.split_seed,
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
        "schema_version": 1,
        "suite": args.suite,
        "backend_requested": args.backend,
        "backend_effective": effective_backend,
        "backend_effective_by_workload": effective_backends,
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
            "train_test_indices": [
                {
                    "train": split["train"].astype(int).tolist(),
                    "test": split["test"].astype(int).tolist(),
                }
                for split in splits
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
            ridge_backend,
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
            percentile_backend,
            percentile_backend,
            regression_kwargs,
            config["percentile"],
            args.monitor_interval,
            args.gpu_index,
        )
    if "consistency" in workloads:
        consistency_backend = effective_backends["consistency"]
        result["benchmarks"][WORKLOAD_KEYS["consistency"]] = benchmark_consistency(
            source,
            target,
            splits,
            consistency_backend,
            consistency_backend,
            regression_kwargs,
            config["bootstrap_iters"],
            args.workers,
            args.seed,
            args.monitor_interval,
            args.gpu_index,
            args.consistency_map,
            config["percentile"],
        )
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
    parser.add_argument("--backend", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--fixture", help="NPZ containing source and target arrays")
    parser.add_argument("--fly-data-root")
    parser.add_argument("--source-fly", default="fly_087")
    parser.add_argument("--target-fly", default="fly_089")
    parser.add_argument("--direction", choices=("left", "right"), default="right")
    parser.add_argument("--region", choices=sorted(REGION_ROIS), default="IPS")
    parser.add_argument(
        "--event-source", choices=("validated", "historical"), default="validated"
    )
    parser.add_argument("--target-limit", type=int)
    parser.add_argument(
        "--split-strategy", choices=("random", "blocked"), default="random"
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
    if args.workers == 0:
        parser.error("--workers cannot be zero")
    if args.monitor_interval <= 0:
        parser.error("--monitor-interval must be positive")
    if args.transfer_repeats < 1:
        parser.error("--transfer-repeats must be positive")
    if args.gpu_index < 0:
        parser.error("--gpu-index cannot be negative")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    try:
        result = run_benchmarks(args)
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    if args.output == "-":
        print(rendered)
    else:
        with open(args.output, "w") as stream:
            stream.write(rendered)
            stream.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
