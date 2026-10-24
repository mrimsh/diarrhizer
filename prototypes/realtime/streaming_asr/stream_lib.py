"""Shared harness for the streaming-ASR variants: a real-time feeder, an event log and a resource monitor.

A variant gets the audio the way a live source would deliver it: 100 ms chunks, each handed over only when it is
"complete" (wall clock = audio time at speed 1.0). The variant works in its own thread, so a slow decode makes the
chunks queue up and shows up as latency - exactly what happens in a live call.
"""
import json, os, queue, subprocess, threading, time
import numpy as np
import psutil
import soundfile as sf

SR = 16000
CHUNK_S = 0.1


def load16(path):
    x, sr = sf.read(path, dtype="float32")
    assert sr == SR, f"{path}: {sr} Hz, expected 16000"
    return x if x.ndim == 1 else x.mean(1)


def gpu_sample():
    """(utilisation %, memory MiB) of GPU 0 via nvidia-smi."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.strip().split(",")
        return float(out[0]), float(out[1])
    except Exception:
        return float("nan"), float("nan")


class Monitor:
    """Samples CPU/RSS of this process and global GPU load every `every` seconds."""

    def __init__(self, every=2.0):
        self.every, self.rows, self._stop = every, [], threading.Event()
        self.proc = psutil.Process()
        self.gpu0 = gpu_sample()[1]          # memory already taken by other processes before we load anything
        self.proc.cpu_percent(None)

    def start(self):
        self._t0 = time.perf_counter()
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()
        return self

    def _run(self):
        while not self._stop.wait(self.every):
            g = gpu_sample()
            self.rows.append((time.perf_counter() - self._t0, self.proc.cpu_percent(None), self.proc.memory_info().rss / 2**20, g[0], g[1]))

    def stop(self):
        self._stop.set(); self._th.join()
        a = np.array(self.rows) if self.rows else np.zeros((1, 5))
        return {"cpu_pct_of_one_core": {"mean": round(float(a[:, 1].mean()), 1), "max": round(float(a[:, 1].max()), 1)},
                "rss_mb": {"mean": round(float(a[:, 2].mean()), 0), "max": round(float(a[:, 2].max()), 0)},
                "gpu_util_pct_total": {"mean": round(float(np.nanmean(a[:, 3])), 1), "max": round(float(np.nanmax(a[:, 3])), 1)},
                "gpu_mem_mib_total": {"max": round(float(np.nanmax(a[:, 4])), 0), "before_load": self.gpu0},
                "gpu_mem_mib_added_by_us": round(float(np.nanmax(a[:, 4])) - self.gpu0, 0)}


class Log:
    """Events a variant emits: finished text with the audio span it covers."""

    def __init__(self):
        self.t0 = None
        self.events = []
        self.fed = 0.0

    def emit(self, text, start, end, **kw):
        if text.strip():
            self.events.append(dict(wall=round(time.perf_counter() - self.t0, 3), fed=round(self.fed, 2),
                                    start=round(float(start), 2), end=round(float(end), 2), text=text.strip(), **kw))


def run_stream(x, worker, speed=1.0, log=None, load_s=None):
    """Feed x (16 kHz mono) in real time. `worker(q, log)` runs in its own thread, reads (audio_end_time, chunk) tuples from
    the queue until it gets None, and calls log.emit(). Returns the Log."""
    log = log or Log()
    q = queue.Queue()
    th = threading.Thread(target=worker, args=(q, log), daemon=True)
    n = int(SR * CHUNK_S)
    log.t0 = time.perf_counter()
    th.start()
    for i in range(0, len(x), n):
        c = x[i:i + n]
        t_end = (i + len(c)) / SR
        d = log.t0 + t_end / speed - time.perf_counter()
        if d > 0:
            time.sleep(d)
        log.fed = t_end
        q.put((t_end, c))
    q.put(None)
    th.join()
    log.total_wall = round(time.perf_counter() - log.t0, 2)
    return log


def save(path, log, meta, resources):
    json.dump({"meta": meta, "resources": resources, "total_wall_s": getattr(log, "total_wall", None), "events": log.events},
              open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
