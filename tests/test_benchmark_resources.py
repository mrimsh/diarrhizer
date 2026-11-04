"""Deterministic resource accounting and native process-tree smoke, without models or a GPU."""
import subprocess
import sys
from types import SimpleNamespace

import pytest

import benchmark_resources as resources


def ram(entries):
    return {"processes": [{"identity": (pid, born), "cpu": cpu, "rss": rss} for pid, born, cpu, rss in entries],
            "lost": 0, "system_total": 10000, "system_used": 5000, "system_available": 5000}


class Reader:
    logical_cpus = affinity_cpus = 4
    def __init__(self, snapshots):
        self.snapshots = iter(snapshots)
    def read(self):
        snapshot = next(self.snapshots)
        if isinstance(snapshot, Exception):
            raise snapshot
        return snapshot


class Gpu:
    driver = "synthetic"
    def __init__(self, values):
        self.values, self.closed = iter(values), False
    def read(self):
        value = next(self.values)
        return [{"uuid": "GPU-test", "index": 0, "name": "synthetic", "used_bytes": value,
                 "total_bytes": 10000, "utilization_percent": 0}]
    def close(self):
        self.closed = True


def test_native_tree_cpu_and_concurrent_memory_peaks_are_accounted(monkeypatch):
    monkeypatch.setattr(resources.time, "time", lambda: 1000)
    reader = Reader([ram([(1, 100, 10, 100)]), ram([(1, 100, 12, 150), (2, 1001, 3, 200)]),
                     ram([(1, 100, 14, 90)])])
    gpu = Gpu([1000, 3000, 1500])
    with resources.ResourceMonitor(10, process_reader=reader, gpu_factory=lambda: gpu) as monitor:
        monitor.sample()
    report = monitor.report(20)
    assert report["cpu"]["observed_tree_seconds"] == 7
    assert report["ram"]["peak_rss_bytes"] == 350  # Concurrent sum, not sum of lifetime peaks.
    assert report["ram"]["start_rss_bytes"] == 100 and report["ram"]["end_rss_bytes"] == 90
    assert report["ram"]["peak_increase_bytes"] == 250 and report["ram"]["observed_processes"] == 2
    assert report["gpu"]["devices"][0]["peak_used_bytes"] == 3000
    assert report["gpu"]["devices"][0]["peak_increase_bytes"] == 2000
    assert report["rtf"] == monitor.wall_seconds / 20
    assert report["audio_seconds_per_wall_second"] == 20 / monitor.wall_seconds
    assert gpu.closed and not monitor.thread.is_alive()


def test_pid_reuse_does_not_merge_child_cpu_counters(monkeypatch):
    monkeypatch.setattr(resources.time, "time", lambda: 1000)
    reader = Reader([ram([(1, 100, 10, 100)]), ram([(2, 1001, 3, 200)]), ram([(2, 1002, 1, 200)])])
    with resources.ResourceMonitor(10, process_reader=reader, gpu_factory=lambda: Gpu([0, 0, 0])) as monitor:
        monitor.sample()
    assert monitor.report()["cpu"]["observed_tree_seconds"] == 4


def test_unavailable_counters_are_null_and_do_not_replace_workload_error():
    reader = Reader([OSError("unavailable"), OSError("unavailable")])
    def no_gpu():
        raise OSError("no driver")
    with pytest.raises(ValueError, match="workload") as caught:
        with resources.ResourceMonitor(10, process_reader=reader, gpu_factory=no_gpu) as monitor:
            raise ValueError("workload")
    result = caught.value.benchmark_performance
    assert result["ram"]["peak_rss_bytes"] is None and result["cpu"]["observed_tree_seconds"] is None
    assert result["gpu"]["devices"] == [] and len(result["warnings"]) == 2
    assert not monitor.thread.is_alive()


def test_telemetry_failure_and_cleanup_do_not_break_successful_workload():
    gpu = Gpu([0])
    def broken_close():
        raise RuntimeError("cleanup")
    gpu.close = broken_close
    with resources.ResourceMonitor(10, process_reader=Reader([ram([(1, 1, 0, 10)]), ram([(1, 1, 1, 20)])]),
                                   gpu_factory=lambda: gpu) as monitor:
        result = 42
    assert result == 42 and monitor.report()["ram"]["peak_rss_bytes"] == 20
    assert len(monitor.report()["warnings"]) == 2


@pytest.mark.parametrize("interval", [0, -.1, .001, 11, float("nan"), float("inf"), True])
def test_invalid_sampling_interval_is_rejected_before_reading_sensors(interval):
    with pytest.raises(ValueError):
        resources.ResourceMonitor(interval, process_reader=object())


def test_nvml_memory_remains_available_when_utilization_is_unsupported():
    class Error(Exception):
        pass
    def unsupported(handle):
        raise Error()
    gpu = object.__new__(resources.GpuReader)
    gpu.devices = [(object(), {"index": 0, "uuid": "GPU-test", "name": "test"})]
    gpu.api = SimpleNamespace(NVMLError=Error,
                              nvmlDeviceGetMemoryInfo=lambda h: SimpleNamespace(total=100, used=50),
                              nvmlDeviceGetUtilizationRates=unsupported)
    result = gpu.read()[0]
    assert result["used_bytes"] == 50 and result["utilization_percent"] is None


def test_summary_weights_time_and_does_not_sum_memory_or_include_failures():
    def row(wall, audio, rss, vram):
        return {"status": "completed", "case": "same", "performance": {
            "wall_seconds": wall, "audio_seconds": audio,
            "ram": {"peak_rss_bytes": rss}, "gpu": {"devices": [{"uuid": "GPU-test", "peak_used_bytes": vram}]}}}
    rows = [row(2, 10, 100, 300), row(4, 10, 150, 250),
            {**row(100, 10, 10000, 10000), "status": "failed"}, {"status": "completed"}]
    result = resources.summarize_performance(rows)
    assert result["rtf"] == .3 and result["peak_rss_bytes"] == 150
    assert result["gpu_peak_used_bytes_by_uuid"] == {"GPU-test": 300}
    assert result["by_case"]["same"]["wall_median_seconds"] == 3
    assert result["by_case"]["same"]["wall_p95_seconds"] == 4
    assert result["measured_completed"] == 2 and result["unmeasured_completed"] == 1


def test_native_reader_includes_child_resident_memory():
    reader = resources.ProcessReader()
    child = subprocess.Popen([sys.executable, "-c",
                              "import sys; data=bytearray(48*1024*1024); data[::4096]=b'X'*(len(data)//4096); print('ready',flush=True); sys.stdin.readline()"],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "ready"
        snapshot = reader.read()
        # Windows venv python.exe can be a small launcher with the interpreter as its child.
        family = {child.pid, *(p.pid for p in reader.psutil.Process(child.pid).children(recursive=True))}
        entries = [p for p in snapshot["processes"] if p["identity"][0] in family]
        assert entries
        entry = max(entries, key=lambda p: p.get("private_commit") or p["rss"])
        assert entry["rss"] > 0  # OS may trim/page working sets under ambient memory pressure.
        if resources.os.name == "nt":
            assert entry["private_commit"] >= 48 * 1024 * 1024
    finally:
        child.communicate("done\n", timeout=10)
