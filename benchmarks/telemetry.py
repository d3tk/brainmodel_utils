"""Host inventory and GPU identity/time-series telemetry for benchmarks."""

import csv
import os
import shutil
import subprocess
import time
from pathlib import Path


def _read_text(path):
    try:
        with open(path, "r") as stream:
            return stream.read()
    except OSError:
        return None


def _run(command, timeout=2):
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * (q / 100.0)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    fraction = rank - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def _mean(values):
    if not values:
        return None
    return float(sum(values) / float(len(values)))


def _positive_delta(value, baseline):
    if value is None or baseline is None:
        return None
    return max(0, value - baseline)


def nvidia_smi_inventory():
    output = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.total,pci.bus_id,clocks.max.sm,clocks.max.mem,power.limit",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return []
    devices = []
    for row in csv.reader(output.splitlines(), skipinitialspace=True):
        if len(row) < 5:
            continue
        record = {
            "nvml_index": int(row[0]),
            "name": row[1],
            "uuid": row[2],
            "driver_version": row[3],
            "memory_total_bytes": int(float(row[4]) * 1024 * 1024),
        }
        if len(row) > 5:
            record["pci_bus_id"] = row[5]
        if len(row) > 6 and row[6] not in ("", "[N/A]"):
            record["max_sm_clock_mhz"] = float(row[6])
        if len(row) > 7 and row[7] not in ("", "[N/A]"):
            record["max_memory_clock_mhz"] = float(row[7])
        if len(row) > 8 and row[8] not in ("", "[N/A]"):
            record["power_limit_watts"] = float(row[8])
        devices.append(record)
    return devices


def parse_cuda_visible_devices(value):
    if value is None or value.strip() == "":
        return None
    tokens = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if token.startswith("GPU-") or token.startswith("MIG-"):
            tokens.append(token)
        else:
            tokens.append(int(token))
    return tokens


def resolve_gpu_identity(compute_index, inventory=None, cuda_visible_devices=None):
    inventory = list(inventory if inventory is not None else nvidia_smi_inventory())
    visible = parse_cuda_visible_devices(
        os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices is None
        else cuda_visible_devices
    )
    uuid = None
    nvml_index = None
    if visible:
        if compute_index >= len(visible):
            raise ValueError(
                "compute GPU index {} is outside CUDA_VISIBLE_DEVICES={}".format(
                    compute_index, visible
                )
            )
        token = visible[compute_index]
        if isinstance(token, int):
            nvml_index = token
            for device in inventory:
                if device["nvml_index"] == token:
                    uuid = device["uuid"]
                    break
        else:
            uuid = token
            for device in inventory:
                if device["uuid"] == token:
                    nvml_index = device["nvml_index"]
                    break
    elif inventory:
        if compute_index >= len(inventory):
            raise ValueError(
                "compute GPU index {} is unavailable; {} devices are visible".format(
                    compute_index, len(inventory)
                )
            )
        uuid = inventory[compute_index]["uuid"]
        nvml_index = inventory[compute_index]["nvml_index"]
    matched = None
    for device in inventory:
        if uuid is not None and device["uuid"] == uuid:
            matched = device
            break
        if uuid is None and device["nvml_index"] == compute_index:
            matched = device
            uuid = device["uuid"]
            nvml_index = device["nvml_index"]
            break
    identity = {
        "compute_index": int(compute_index),
        "uuid": uuid,
        "nvml_index": nvml_index,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices is None
        else cuda_visible_devices,
        "name": matched["name"] if matched else None,
        "driver_version": matched.get("driver_version") if matched else None,
        "memory_total_bytes": matched.get("memory_total_bytes") if matched else None,
        "pci_bus_id": matched.get("pci_bus_id") if matched else None,
        "max_sm_clock_mhz": matched.get("max_sm_clock_mhz") if matched else None,
        "max_memory_clock_mhz": matched.get("max_memory_clock_mhz") if matched else None,
        "power_limit_watts": matched.get("power_limit_watts") if matched else None,
    }
    return identity


def collect_host_inventory():
    cpuinfo = _read_text("/proc/cpuinfo") or ""
    meminfo = _read_text("/proc/meminfo") or ""
    model = None
    physical_ids = set()
    core_ids = set()
    siblings = None
    cores_per_socket = None
    for line in cpuinfo.splitlines():
        lower = line.lower()
        if lower.startswith("model name") and model is None:
            model = line.split(":", 1)[1].strip()
        elif lower.startswith("physical id"):
            physical_ids.add(line.split(":", 1)[1].strip())
        elif lower.startswith("core id"):
            core_ids.add(line.split(":", 1)[1].strip())
        elif lower.startswith("siblings") and siblings is None:
            siblings = int(line.split(":", 1)[1].strip())
        elif lower.startswith("cpu cores") and cores_per_socket is None:
            cores_per_socket = int(line.split(":", 1)[1].strip())
    logical = os.cpu_count()
    smt = None
    if siblings is not None and cores_per_socket:
        smt = siblings > cores_per_socket
    numa = _run(["lscpu", "-p=NODE"])
    numa_nodes = None
    if numa:
        nodes = {
            line.split(",")[0]
            for line in numa.splitlines()
            if line and not line.startswith("#")
        }
        numa_nodes = len(nodes) if nodes else None
    mem_total = None
    swap_total = None
    for line in meminfo.splitlines():
        if line.startswith("MemTotal:"):
            mem_total = int(line.split()[1]) * 1024
        elif line.startswith("SwapTotal:"):
            swap_total = int(line.split()[1]) * 1024
    disk = shutil.disk_usage(os.getcwd())
    local_tmp = os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR")
    return {
        "hostname": os.uname().nodename if hasattr(os, "uname") else None,
        "cpu_model": model,
        "sockets": len(physical_ids) or None,
        "physical_cores_estimate": len(core_ids) * max(len(physical_ids), 1)
        if core_ids
        else None,
        "logical_cpus": logical,
        "cores_per_socket": cores_per_socket,
        "siblings_per_socket": siblings,
        "smt_enabled": smt,
        "numa_nodes": numa_nodes,
        "memory_total_bytes": mem_total,
        "swap_total_bytes": swap_total,
        "cwd_disk_total_bytes": int(disk.total),
        "cwd_disk_free_bytes": int(disk.free),
        "local_tmp": local_tmp,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS")
        or os.environ.get("SLURM_STEP_GPUS"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpus": nvidia_smi_inventory(),
    }


def _nvml_bytes(value):
    if value is None or int(value) < 0:
        return None
    return int(value)


class GpuSampler:
    def __init__(self, compute_index):
        self.compute_index = compute_index
        self.identity = {
            "compute_index": compute_index,
            "uuid": None,
            "nvml_index": compute_index,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
        self.method = None
        self.error = None
        self._pynvml = None
        self._handle = None
        try:
            self.identity = resolve_gpu_identity(compute_index)
        except (OSError, ValueError) as error:
            self.error = str(error)
        uuid = self.identity.get("uuid")
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
            if uuid:
                try:
                    self._handle = pynvml.nvmlDeviceGetHandleByUUID(
                        uuid.encode("utf-8") if not isinstance(uuid, bytes) else uuid
                    )
                except pynvml.NVMLError:
                    self._handle = pynvml.nvmlDeviceGetHandleByIndex(
                        self.identity.get("nvml_index") or compute_index
                    )
            else:
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(compute_index)
            self._pynvml = pynvml
            self.method = "nvml"
        except (pynvml.NVMLError, OSError) as error:
            if shutil.which("nvidia-smi") is not None:
                self.method = "nvidia-smi"
            else:
                self.error = str(error)

    def _nvml_optional(self, getter, *args):
        try:
            return getter(*args)
        except Exception as error:  # noqa: BLE001 - NVML feature availability varies
            self.error = str(error)
            return None

    def sample(self, pids=None):
        pids = set(pids or ())
        if self.method == "nvml" and self._handle is not None:
            return self._sample_nvml(pids)
        if self.method == "nvidia-smi":
            return self._sample_nvidia_smi(pids)
        return {}

    def _sample_nvml(self, pids):
        pynvml = self._pynvml
        handle = self._handle
        memory = self._nvml_optional(pynvml.nvmlDeviceGetMemoryInfo, handle)
        utilization = self._nvml_optional(pynvml.nvmlDeviceGetUtilizationRates, handle)
        power = self._nvml_optional(pynvml.nvmlDeviceGetPowerUsage, handle)
        temperature = self._nvml_optional(
            pynvml.nvmlDeviceGetTemperature, handle, pynvml.NVML_TEMPERATURE_GPU
        )
        sm_clock = self._nvml_optional(
            pynvml.nvmlDeviceGetClockInfo, handle, pynvml.NVML_CLOCK_SM
        )
        mem_clock = self._nvml_optional(
            pynvml.nvmlDeviceGetClockInfo, handle, pynvml.NVML_CLOCK_MEM
        )
        pcie_tx = self._nvml_optional(
            pynvml.nvmlDeviceGetPcieThroughput, handle, pynvml.NVML_PCIE_UTIL_TX_BYTES
        )
        pcie_rx = self._nvml_optional(
            pynvml.nvmlDeviceGetPcieThroughput, handle, pynvml.NVML_PCIE_UTIL_RX_BYTES
        )
        process_memory = 0
        process_memory_available = False
        for getter_name in (
            "nvmlDeviceGetComputeRunningProcesses_v3",
            "nvmlDeviceGetComputeRunningProcesses_v2",
            "nvmlDeviceGetComputeRunningProcesses",
        ):
            getter = getattr(pynvml, getter_name, None)
            if getter is None:
                continue
            processes = self._nvml_optional(getter, handle)
            if processes is None:
                continue
            for process in processes:
                used = getattr(process, "usedGpuMemory", None)
                if used is None or int(used) < 0:
                    continue
                process_memory_available = True
                if not pids or int(process.pid) in pids:
                    process_memory += int(used)
            break
        sample = {
            "device_memory_used_bytes": int(memory.used) if memory is not None else None,
            "utilization_percent": float(utilization.gpu)
            if utilization is not None
            else None,
            "process_memory_used_bytes": process_memory
            if process_memory_available
            else None,
            "power_watts": power / 1000.0 if power is not None else None,
            "temperature_c": float(temperature) if temperature is not None else None,
            "sm_clock_mhz": float(sm_clock) if sm_clock is not None else None,
            "memory_clock_mhz": float(mem_clock) if mem_clock is not None else None,
            "pcie_tx_kib_per_sec": float(pcie_tx) if pcie_tx is not None else None,
            "pcie_rx_kib_per_sec": float(pcie_rx) if pcie_rx is not None else None,
        }
        return sample

    def _sample_nvidia_smi(self, pids):
        uuid = self.identity.get("uuid")
        selector = ["--id={}".format(uuid or self.compute_index)]
        output = _run(
            [
                "nvidia-smi",
            ]
            + selector
            + [
                "--query-gpu=uuid,memory.used,utilization.gpu,power.draw,temperature.gpu,clocks.sm,clocks.mem",
                "--format=csv,noheader,nounits",
            ]
        )
        sample = {}
        if output:
            try:
                row = next(csv.reader([output], skipinitialspace=True))
                if not self.identity.get("uuid") and row:
                    self.identity["uuid"] = row[0]
                sample = {
                    "device_memory_used_bytes": int(float(row[1]) * 1024 * 1024),
                    "utilization_percent": float(row[2]),
                    "power_watts": None
                    if row[3] in ("", "[N/A]")
                    else float(row[3]),
                    "temperature_c": None
                    if row[4] in ("", "[N/A]")
                    else float(row[4]),
                    "sm_clock_mhz": None
                    if len(row) < 6 or row[5] in ("", "[N/A]")
                    else float(row[6 - 1]),
                    "memory_clock_mhz": None
                    if len(row) < 7 or row[6] in ("", "[N/A]")
                    else float(row[6]),
                    "process_memory_used_bytes": None,
                    "pcie_tx_kib_per_sec": None,
                    "pcie_rx_kib_per_sec": None,
                }
            except (IndexError, ValueError) as error:
                self.error = str(error)
        apps = _run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory,gpu_uuid",
                "--format=csv,noheader,nounits",
            ]
        )
        if apps:
            total = 0
            found = False
            for row in csv.reader(apps.splitlines(), skipinitialspace=True):
                if len(row) < 2:
                    continue
                try:
                    pid = int(row[0])
                    used = int(float(row[1]) * 1024 * 1024)
                except ValueError:
                    continue
                gpu_uuid = row[2] if len(row) > 2 else None
                if uuid and gpu_uuid and gpu_uuid != uuid:
                    continue
                if not pids or pid in pids:
                    total += used
                    found = True
            if found:
                sample["process_memory_used_bytes"] = total
        return sample

    def close(self):
        # Leave NVML initialized; shutdown is process-wide and would
        # invalidate other readers in the same benchmark process.
        return


def summarize_gpu_samples(samples):
    def collect(name):
        return [sample[name] for sample in samples if sample.get(name) is not None]

    utilization = collect("utilization_percent")
    device_memory = collect("device_memory_used_bytes")
    process_memory = collect("process_memory_used_bytes")
    power = collect("power_watts")
    temperature = collect("temperature_c")
    return {
        "samples": len(samples),
        "mean_utilization_percent": _mean(utilization),
        "p95_utilization_percent": _percentile(utilization, 95) if utilization else None,
        "peak_utilization_percent": max(utilization) if utilization else None,
        "mean_device_memory_used_bytes": _mean(device_memory),
        "peak_device_memory_used_bytes": max(device_memory) if device_memory else None,
        "mean_process_memory_used_bytes": _mean(process_memory),
        "peak_process_memory_used_bytes": max(process_memory)
        if process_memory
        else None,
        "mean_power_watts": _mean(power),
        "p95_power_watts": _percentile(power, 95) if power else None,
        "peak_temperature_c": max(temperature) if temperature else None,
        "mean_sm_clock_mhz": _mean(collect("sm_clock_mhz")),
        "mean_memory_clock_mhz": _mean(collect("memory_clock_mhz")),
        "mean_pcie_tx_kib_per_sec": _mean(collect("pcie_tx_kib_per_sec")),
        "mean_pcie_rx_kib_per_sec": _mean(collect("pcie_rx_kib_per_sec")),
    }


def host_faults_and_swap(pid):
    status = _read_text("/proc/{}/status".format(pid)) or ""
    vm_swap = None
    for line in status.splitlines():
        if line.startswith("VmSwap:"):
            vm_swap = int(line.split()[1]) * 1024
    stat = _read_text("/proc/{}/stat".format(pid))
    minflt = None
    majflt = None
    if stat:
        close_paren = stat.rfind(")")
        fields = stat[close_paren + 2 :].split()
        if len(fields) >= 10:
            minflt = int(fields[7])
            majflt = int(fields[9])
    meminfo = _read_text("/proc/meminfo") or ""
    swap_free = None
    swap_total = None
    for line in meminfo.splitlines():
        if line.startswith("SwapTotal:"):
            swap_total = int(line.split()[1]) * 1024
        elif line.startswith("SwapFree:"):
            swap_free = int(line.split()[1]) * 1024
    return {
        "vm_swap_bytes": vm_swap,
        "minor_faults": minflt,
        "major_faults": majflt,
        "host_swap_used_bytes": None
        if swap_total is None or swap_free is None
        else max(0, swap_total - swap_free),
    }


def write_inventory(path, inventory):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    with open(path, "w") as stream:
        json.dump(inventory, stream, indent=2, sort_keys=True)
        stream.write("\n")
