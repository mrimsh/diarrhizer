"""Sample OS/native memory and CPU plus whole-device NVIDIA memory for offline benchmarks."""
import math
import os
import platform
import statistics
import threading
import time
from importlib.metadata import version


# [SEMANTIC-BEGIN] QUALITY:RESOURCE_SENSORS
# @purpose: Read full native process-tree RAM/CPU and driver-wide framebuffer memory without loading CUDA
# @description: psutil RSS includes native allocators; shared pages may be counted in multiple processes.
#   Windows VMS is also recorded as private commit, separately from resident RAM.
#   NVML is device-wide (all applications), includes CTranslate2/context, and is never attributed to our PID.
# @inputs: current process and descendants; all physical NVIDIA devices
# @outputs: OS snapshots and GPU identity/driver counters in bytes
# @sideEffects: Queries OS/driver; optional bench dependencies; nvmlInit/nvmlShutdown
# @errors: Missing psutil requires installing .[bench]; unavailable GPU counters are reported, never zero-filled
# @see: QUALITY:RESOURCE_MONITOR, QUALITY:PERFORMANCE_SUMMARY
class ProcessReader:
    def __init__(self):
        try:
            import psutil
        except ImportError as exc:
            raise RuntimeError("Resource measurement requires: python -m pip install '.[bench]'") from exc
        self.psutil = psutil
        self.root = psutil.Process()
        self.logical_cpus = psutil.cpu_count() or 1
        try:
            self.affinity_cpus = max(1, len(self.root.cpu_affinity()))
        except (AttributeError, psutil.Error):
            self.affinity_cpus = self.logical_cpus

    def read(self):
        entries, lost = [], 0
        for process in [self.root, *self.root.children(recursive=True)]:
            try:
                with process.oneshot():
                    identity = (process.pid, process.create_time())
                    memory, cpu = process.memory_info(), process.cpu_times()
                    entries.append({"identity": identity, "rss": memory.rss, "cpu": cpu.user + cpu.system,
                                    "private_commit": memory.vms if os.name == "nt" else None})
            except self.psutil.Error:
                lost += 1  # Short-lived children can exit during enumeration.
        vm = self.psutil.virtual_memory()
        return {"processes": entries, "lost": lost, "system_total": vm.total,
                "system_used": vm.used, "system_available": vm.available}


class GpuReader:
    def __init__(self):
        import pynvml
        self.api = pynvml
        self.initialized = False
        try:
            pynvml.nvmlInit()
            self.initialized = True
            self.driver = pynvml.nvmlSystemGetDriverVersion()
            self.devices = []
            for index in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                self.devices.append((handle, {"index": index, "uuid": pynvml.nvmlDeviceGetUUID(handle),
                                               "name": pynvml.nvmlDeviceGetName(handle)}))
        except Exception:
            self.close()
            raise

    def read(self):
        devices = []
        for handle, identity in self.devices:
            try:
                memory = self.api.nvmlDeviceGetMemoryInfo(handle)
                try:
                    utilization = self.api.nvmlDeviceGetUtilizationRates(handle).gpu
                except self.api.NVMLError:
                    utilization = None
                devices.append({**identity, "total_bytes": memory.total, "used_bytes": memory.used,
                                "utilization_percent": utilization})
            except self.api.NVMLError as exc:
                devices.append({**identity, "error": type(exc).__name__})
        return devices

    def close(self):
        if self.initialized:
            self.api.nvmlShutdown()
            self.initialized = False
# [SEMANTIC-END] QUALITY:RESOURCE_SENSORS


# [SEMANTIC-BEGIN] QUALITY:RESOURCE_MONITOR
# @purpose: Measure pipeline wall time, sampled RAM/VRAM peaks and observed process-tree CPU work
# @description: Baseline before timed call, final sample after timing; sampler overhead remains in workload.
#   Peaks are sampled lower bounds; child CPU persists after exit and identities include creation time.
#   Telemetry errors preserve workload exceptions/results, with coverage and warnings in the report.
# @inputs: interval_seconds (0.02..10); injectable OS/GPU readers for offline regressions
# @outputs: performance report; failed workload exceptions retain benchmark_performance
# @sideEffects: Background sampler thread; OS/driver queries; no model or CUDA initialization
# @errors: ValueError on invalid interval; missing bench dependency fails before running models
# @see: QUALITY:RESOURCE_SENSORS, QUALITY:BENCHMARK, QUALITY:DER_SUITE
class ResourceMonitor:
    def __init__(self, interval_seconds=.1, *, process_reader=None, gpu_factory=GpuReader):
        if type(interval_seconds) not in (int, float) or not math.isfinite(interval_seconds) or not .02 <= interval_seconds <= 10:
            raise ValueError("Sampling interval must be finite, between 0.02 and 10 seconds")
        self.interval = interval_seconds
        self.reader = process_reader if process_reader is not None else ProcessReader()
        self.gpu_factory = gpu_factory
        self.gpu = None
        self.samples, self.warnings = [], []
        self.cpu_initial, self.cpu_latest = {}, {}
        self.stop = threading.Event()
        self.wall_seconds = None

    def warn(self, message):
        if message not in self.warnings:
            self.warnings.append(message)

    def sample(self):
        started = time.perf_counter()
        row = {"at": started, "ram": None, "gpu": []}
        try:
            row["ram"] = self.reader.read()
            for process in row["ram"]["processes"]:
                key = tuple(process["identity"])
                if key not in self.cpu_initial:
                    self.cpu_initial[key] = process["cpu"] if not self.samples or key[1] < self.epoch else 0
                self.cpu_latest[key] = max(self.cpu_latest.get(key, 0), process["cpu"])
        except Exception as exc:
            self.warn(f"RAM/CPU sample unavailable: {type(exc).__name__}")
        if self.gpu is not None:
            try:
                row["gpu"] = self.gpu.read()
                if any("error" in device for device in row["gpu"]):
                    self.warn("Some GPU samples unavailable; inspect device sample coverage")
            except Exception as exc:
                self.warn(f"GPU sample unavailable: {type(exc).__name__}")
        row["read_seconds"] = time.perf_counter() - started
        self.samples.append(row)

    def __enter__(self):
        self.epoch = time.time()
        try:
            self.gpu = self.gpu_factory()
        except Exception as exc:
            self.warn(f"NVIDIA telemetry unavailable: {type(exc).__name__}")
        self.sample()
        self.started = time.perf_counter()
        self.thread = threading.Thread(target=self._loop, name="benchmark-resources", daemon=True)
        self.thread.start()
        return self

    def _loop(self):
        while not self.stop.wait(self.interval):
            self.sample()

    def __exit__(self, exc_type, exc, traceback):
        self.wall_seconds = time.perf_counter() - self.started
        self.stop.set()
        self.thread.join()
        self.sample()
        if self.gpu is not None:
            try:
                self.gpu.close()
            except Exception as close_error:
                self.warn(f"GPU telemetry close failed: {type(close_error).__name__}")
        if exc is not None:
            try:
                exc.benchmark_performance = self.report()
            except Exception:
                pass  # Never replace the original workload exception with a telemetry error.
        return False

    def report(self, audio_seconds=None):
        if self.wall_seconds is None:
            raise RuntimeError("Read performance after the monitored call has finished")
        ram = [sample["ram"] for sample in self.samples if sample["ram"] is not None]
        rss = [sum(p["rss"] for p in sample["processes"]) for sample in ram if sample["processes"]]
        commits = [sum(p["private_commit"] for p in sample["processes"]) for sample in ram
                   if sample["processes"] and all(p.get("private_commit") is not None for p in sample["processes"])]
        cpu = sum(max(0, latest - self.cpu_initial[key]) for key, latest in self.cpu_latest.items())
        devices = {}
        for sample in self.samples:
            for device in sample["gpu"]:
                group = devices.setdefault(device["uuid"], {"identity": device, "values": [], "failed": 0})
                if "error" in device:
                    group["failed"] += 1
                else:
                    group["values"].append(device)
        gpu_reports = []
        for group in devices.values():
            values, identity = group["values"], group["identity"]
            used = [v["used_bytes"] for v in values]
            utilization = [v["utilization_percent"] for v in values if v["utilization_percent"] is not None]
            if used and values[0]["utilization_percent"] is not None and values[0]["utilization_percent"] > 20:
                self.warn(f"GPU {identity['uuid']} was busy at baseline; concurrent workloads affect performance and VRAM")
            gpu_reports.append({**{k: identity[k] for k in ("index", "uuid", "name")},
                                "valid_samples": len(values), "failed_samples": group["failed"],
                                "total_bytes": values[0]["total_bytes"] if values else None,
                                "start_used_bytes": used[0] if used else None, "end_used_bytes": used[-1] if used else None,
                                "peak_used_bytes": max(used) if used else None,
                                "peak_increase_bytes": max(used) - used[0] if used else None,
                                "start_utilization_percent": values[0]["utilization_percent"] if values else None,
                                "utilization_sample_mean_percent": statistics.mean(utilization) if utilization else None})
        gaps = [b["at"] - a["at"] for a, b in zip(self.samples, self.samples[1:])]
        return {"schema_version": 1, "wall_seconds": self.wall_seconds, "audio_seconds": audio_seconds,
                "rtf": self.wall_seconds / audio_seconds if audio_seconds else None,
                "audio_seconds_per_wall_second": audio_seconds / self.wall_seconds if audio_seconds and self.wall_seconds else None,
                "cpu": {"observed_tree_seconds": cpu if rss else None,
                        "average_cores": cpu / self.wall_seconds if rss and self.wall_seconds else None,
                        "affinity_normalized_percent": 100 * cpu / self.wall_seconds / self.reader.affinity_cpus if rss and self.wall_seconds else None,
                        "logical_cpus": self.reader.logical_cpus, "affinity_cpus": self.reader.affinity_cpus},
                "ram": {"scope": "process plus observed descendants; summed RSS may double count shared pages",
                        "start_rss_bytes": rss[0] if rss else None, "end_rss_bytes": rss[-1] if rss else None,
                        "peak_rss_bytes": max(rss) if rss else None,
                        "peak_increase_bytes": max(rss) - rss[0] if rss else None,
                        "peak_private_commit_bytes": max(commits) if commits else None,
                        "valid_samples": len(rss), "lost_process_samples": sum(r["lost"] for r in ram),
                        "observed_processes": len(self.cpu_latest),
                        "system_total_bytes": ram[0]["system_total"] if ram else None,
                        "system_used_peak_bytes": max(r["system_used"] for r in ram) if ram else None,
                        "system_available_min_bytes": min(r["system_available"] for r in ram) if ram else None},
                "gpu": {"scope": "whole physical NVIDIA device, all applications; not per-process VRAM",
                        "driver": getattr(self.gpu, "driver", None), "devices": gpu_reports},
                "sampling": {"requested_interval_seconds": self.interval, "samples": len(self.samples),
                             "max_gap_seconds": max(gaps) if gaps else None,
                             "max_read_seconds": max(s["read_seconds"] for s in self.samples)},
                "host": {"platform": platform.platform(), "processor": platform.processor(),
                         "psutil_version": version("psutil"), "nvidia_ml_py_version": version("nvidia-ml-py"),
                         "pid": os.getpid()},
                "warnings": self.warnings,
                "measurement_note": "Sampled peaks, not guaranteed maxima; short children/spikes may be missed. Baseline/final queries excluded from wall time."}
# [SEMANTIC-END] QUALITY:RESOURCE_MONITOR


# [SEMANTIC-BEGIN] QUALITY:PERFORMANCE_SUMMARY
# @purpose: Summarize successful benchmark performance without summing memory peaks or averaging RTFs
# @description: Weighted RTF; max observed memory; per-case wall median/p95 expose repeated-run spread.
# @inputs: completed/failed and legacy run reports
# @outputs: aggregate and per-case performance with measurement coverage
# @sideEffects: None
# @see: QUALITY:RESOURCE_MONITOR, QUALITY:SUITE, QUALITY:DER_SUITE
def summarize_performance(runs):
    completed = [r for r in runs if r["status"] == "completed"]
    measured = [r for r in completed if r.get("performance")]
    if not measured:
        return {"measured_completed": 0, "unmeasured_completed": len(completed)}
    values = [r["performance"] for r in measured]
    wall = sum(p["wall_seconds"] for p in values)
    audio = sum(p["audio_seconds"] or 0 for p in values)
    peaks = [p["ram"]["peak_rss_bytes"] for p in values if p["ram"]["peak_rss_bytes"] is not None]
    commits = [p["ram"]["peak_private_commit_bytes"] for p in values if p["ram"].get("peak_private_commit_bytes") is not None]
    cpu = [p["cpu"]["observed_tree_seconds"] for p in values if p.get("cpu", {}).get("observed_tree_seconds") is not None]
    groups, gpus = {}, {}
    for run in measured:
        groups.setdefault(run.get("case", "single"), []).append(run["performance"]["wall_seconds"])
        for gpu in run["performance"]["gpu"]["devices"]:
            if gpu["peak_used_bytes"] is not None:
                gpus[gpu["uuid"]] = max(gpus.get(gpu["uuid"], 0), gpu["peak_used_bytes"])
    return {"measured_completed": len(measured), "unmeasured_completed": len(completed) - len(measured),
            "wall_seconds": wall, "audio_seconds": audio, "rtf": wall / audio if audio else None,
            "audio_seconds_per_wall_second": audio / wall if wall else None,
            "peak_rss_bytes": max(peaks) if peaks else None,
            "peak_private_commit_bytes": max(commits) if commits else None,
            "ram_measured_completed": len(peaks), "cpu_measured_completed": len(cpu),
            "observed_tree_cpu_seconds": sum(cpu) if cpu else None,
            "gpu_peak_used_bytes_by_uuid": gpus,
            "by_case": {name: {"runs": len(times), "wall_median_seconds": statistics.median(times),
                               "wall_p95_seconds": sorted(times)[math.ceil(.95 * len(times)) - 1]}
                        for name, times in groups.items()}}
# [SEMANTIC-END] QUALITY:PERFORMANCE_SUMMARY
