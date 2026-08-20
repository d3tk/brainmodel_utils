"""Reproducible neural mapper and consistency benchmarks."""

import argparse
import csv
import inspect
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
import scipy
import sklearn

from brainmodel_utils.metrics.consistency import get_linregress_consistency
from brainmodel_utils.neural_mappers import PercentileNeuralMap, SKLinearNeuralMap
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


def _cuda_probe():
    result = {"available": False, "reason": None}
    try:
        import cupy
    except (ImportError, OSError) as error:
        result["reason"] = "CuPy unavailable: {}".format(error)
        return result
    try:
        import cuml
    except (ImportError, OSError) as error:
        result["reason"] = "cuML unavailable: {}".format(error)
        return result
    try:
        import torch
    except (ImportError, OSError) as error:
        result["reason"] = "PyTorch unavailable: {}".format(error)
        return result
    try:
        device_count = int(cupy.cuda.runtime.getDeviceCount())
    except (cupy.cuda.runtime.CUDARuntimeError, RuntimeError) as error:
        result["reason"] = "CuPy CUDA runtime unavailable: {}".format(error)
        return result
    if not torch.cuda.is_available():
        result["reason"] = "PyTorch reports CUDA unavailable"
        return result
    result.update(
        {
            "available": device_count > 0,
            "device_count": device_count,
            "cupy_version": getattr(cupy, "__version__", None),
            "cuml_version": getattr(cuml, "__version__", None),
            "torch_version": getattr(torch, "__version__", None),
        }
    )
    if device_count == 0:
        result["reason"] = "CUDA reported no visible devices"
    return result


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
        import cupy

        cupy.cuda.get_current_stream().synchronize()
    except (ImportError, RuntimeError):
        pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
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


def load_fixture(path, dtype):
    with np.load(path, allow_pickle=False) as fixture:
        missing = {"source", "target"}.difference(fixture.files)
        if missing:
            raise ValueError(
                "fixture is missing arrays: {}".format(", ".join(sorted(missing)))
            )
        source = np.asarray(fixture["source"], dtype=dtype)
        target = np.asarray(fixture["target"], dtype=dtype)
    if source.ndim not in (2, 3):
        raise ValueError("fixture source must have 2 or 3 dimensions")
    if target.ndim != 3:
        raise ValueError("fixture target must have shape trials x stimuli x targets")
    source_stimuli = source.shape[-2]
    if source_stimuli != target.shape[1]:
        raise ValueError("fixture source and target stimulus dimensions differ")
    if target.shape[0] < 2:
        raise ValueError("fixture target must contain at least two trials")
    if target.shape[1] < 2 or source.shape[-1] < 1 or target.shape[-1] < 1:
        raise ValueError("fixture arrays have an empty feature or stimulus dimension")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("fixture arrays must contain only finite values")
    return source, target


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
    return {
        "correlation_seconds": mapper.correlation_seconds,
        "source_selection_seconds": selection_fit,
        "ridge_fit_seconds": ridge_fit,
        "predict_seconds": predict_seconds,
        "predict_source_selection_seconds": mapper.source_selection_seconds["predict"],
        "fit_seconds": fit_seconds,
        "total_seconds": total_seconds,
        "ridge_fit_measurement": "fit minus correlation and source selection",
        "percentile": percentile,
        "selected_sources_per_target": list(mapper._num_source_units),
        "ridge_parameters": dict(regression_kwargs),
        "prediction": _prediction_summary(prediction),
        "resources": resources,
    }


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
):
    mapper_options = _mapper_options(
        SKLinearNeuralMap, backend, effective_backend, regression_kwargs
    )
    map_kwargs = {"map_type": "sklinear", "map_kwargs": mapper_options}
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
    metric = np.asarray(result["test"]["r_xy_n_sb"])
    finite = metric[np.isfinite(metric)]
    return {
        "total_seconds": total_seconds,
        "bootstrap_iterations": bootstrap_iters,
        "train_test_splits": len(splits),
        "workers": workers,
        "metric": "pearsonr",
        "result_shape": list(metric.shape),
        "finite_result_fraction": float(finite.size) / float(metric.size),
        "mean_test_r_xy_n_sb": float(finite.mean()) if finite.size else None,
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


def _metadata(args, cuda_probe):
    libraries = {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }
    libraries.update(_optional_library_versions())
    for library in ("torch", "cupy", "cuml"):
        version = cuda_probe.get("{}_version".format(library))
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
        "cuda_probe": cuda_probe,
        "gpu_devices": _gpu_inventory(),
        "process_peak_rss_bytes_at_start": _peak_rss_bytes(),
    }


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


def _validate_config(config):
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
    _validate_config(config)
    dtype = np.dtype(args.dtype)
    if args.fixture:
        source, target = load_fixture(args.fixture, dtype)
        config.update(
            {
                "stimuli": target.shape[1],
                "sources": source.shape[-1],
                "targets": target.shape[-1],
                "trials": target.shape[0],
            }
        )
        data_origin = "fixture"
    else:
        source, target = make_correlated_data(config, args.seed, dtype)
        data_origin = "correlated_synthetic"

    cuda_probe = (
        _cuda_probe()
        if args.backend != "cpu"
        else {
            "available": None,
            "reason": "not probed for CPU benchmark",
        }
    )
    if args.backend == "cuda" and not cuda_probe["available"]:
        raise RuntimeError(
            "CUDA backend requested but unavailable: {}".format(
                cuda_probe.get("reason")
            )
        )
    effective_backend = (
        "cuda"
        if args.backend == "cuda"
        or (args.backend == "auto" and cuda_probe["available"])
        else "cpu"
    )
    if effective_backend == "cuda" and args.workers != 1:
        raise ValueError("CUDA consistency benchmarks require --workers 1")

    train_samples = int(math.ceil(config["train_frac"] * config["stimuli"]))
    if train_samples >= config["stimuli"]:
        raise ValueError("train_frac leaves no samples for standalone prediction")

    source_average = _averaged_source(source)
    target_average = target.mean(axis=0)
    splits = generate_train_test_splits(
        num_stim=config["stimuli"],
        num_splits=config["splits"],
        train_frac=config["train_frac"],
        start_seed=args.split_seed,
    )
    first_split = splits[0]
    X_train = source_average[first_split["train"]]
    Y_train = target_average[first_split["train"]]
    X_test = source_average[first_split["test"]]
    regression_kwargs = {
        "alpha": args.alpha,
        "fit_intercept": args.fit_intercept,
    }
    if args.solver is not None:
        regression_kwargs["solver"] = args.solver

    result = {
        "schema_version": 1,
        "suite": args.suite,
        "backend_requested": args.backend,
        "backend_effective": effective_backend,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "config": config,
        "metadata": _metadata(args, cuda_probe),
        "dataset": {
            "origin": data_origin,
            "fixture": os.path.abspath(args.fixture) if args.fixture else None,
            "source": _array_summary(source),
            "target": _array_summary(target),
            "standalone_train_samples": int(X_train.shape[0]),
            "standalone_test_samples": int(X_test.shape[0]),
        },
        "benchmarks": {},
    }
    result["benchmarks"]["host_device_transfer"] = benchmark_transfers(
        source_average,
        effective_backend,
        args.monitor_interval,
        args.gpu_index,
        args.transfer_repeats,
    )
    result["benchmarks"]["ridge_mapper"] = benchmark_ridge(
        X_train,
        Y_train,
        X_test,
        args.backend,
        effective_backend,
        regression_kwargs,
        args.monitor_interval,
        args.gpu_index,
    )
    result["benchmarks"]["percentile_mapper"] = benchmark_percentile(
        X_train,
        Y_train,
        X_test,
        args.backend,
        effective_backend,
        regression_kwargs,
        config["percentile"],
        args.monitor_interval,
        args.gpu_index,
    )
    result["benchmarks"]["linregress_consistency"] = benchmark_consistency(
        source,
        target,
        splits,
        args.backend,
        effective_backend,
        regression_kwargs,
        config["bootstrap_iters"],
        args.workers,
        args.seed,
        args.monitor_interval,
        args.gpu_index,
    )
    result["metadata"]["process_peak_rss_bytes_at_end"] = _peak_rss_bytes()
    return result


def build_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark Ridge, percentile mapping, and neural consistency."
    )
    parser.add_argument("--suite", choices=sorted(SUITES), default="smoke")
    parser.add_argument("--backend", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--fixture", help="NPZ containing source and target arrays")
    parser.add_argument("--output", default="-", help="JSON path, or - for stdout")
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
