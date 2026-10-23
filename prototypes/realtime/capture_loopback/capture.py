"""Record system output (WASAPI loopback) and the microphone into two aligned mono WAVs.

    python capture.py --seconds 60 --out ../_local/run1 [--backend pyaw|soundcard] [--follow-devices]

Both tracks share one timeline: T0 is taken before the streams open, every chunk is placed at
(arrival time - T0), so a late first chunk becomes leading silence and a stretch with no packets
(WASAPI loopback sends nothing while no app renders sound) becomes zeros. Both files end up with
the same length and the same sample rate and can be stacked into one stereo file
(left = microphone = "me", right = loopback = "others") or fed to the pipeline separately.

Besides the WAVs the run writes stats.csv (every --stats-every s) and summary.json.
This is a research prototype, not part of the package (nothing here is imported from src/).
"""
import argparse, csv, json, os, queue, sys, threading, time
import numpy as np
import psutil
import soundfile as sf

RATE = 48000          # both tracks are stored at this rate
TOL_S = 0.15          # jitter tolerated before a gap is zero-filled / an overlap is trimmed
sys.stdout.reconfigure(encoding="utf-8")


class Track:
    """One mono WAV on the shared timeline. push() is called from the capture callback/thread."""

    def __init__(self, name, path, t0):
        self.name, self.t0 = name, t0
        self.raw = 0          # frames delivered by the device (before any padding) - used for drift
        self.written = 0      # frames in the file so far (after padding)
        self.padded = 0       # zero frames inserted
        self.pad_events = 0
        self.dropped = 0
        self.overflows = 0    # PortAudio input-overflow flags
        self.blocks = 0
        self.peak = 0.0
        self.sq = 0.0
        self._q = queue.Queue()
        self._f = sf.SoundFile(path, "w", samplerate=RATE, channels=1, subtype="PCM_16")
        self._th = threading.Thread(target=self._writer, daemon=True)
        self._th.start()

    def _writer(self):
        while True:
            x = self._q.get()
            if x is None:
                return
            self._f.write(x)

    def _put(self, x):
        self._q.put(x)
        self.written += len(x)

    def push(self, x, t_arrival):
        n = len(x)
        self.raw += n
        self.blocks += 1
        self.peak = max(self.peak, float(np.abs(x).max()))
        self.sq += float(np.dot(x, x))
        gap = (t_arrival - self.t0) * RATE - n - self.written     # frames of missing time before this chunk
        if gap > TOL_S * RATE:
            g = int(round(gap))
            self._put(np.zeros(g, dtype=np.float32))
            self.padded += g
            self.pad_events += 1
        elif gap < -TOL_S * RATE:
            d = min(n, int(-gap))
            x = x[d:]
            self.dropped += d
        if len(x):
            self._put(x)

    def finish(self, t_end):
        want = int(round((t_end - self.t0) * RATE))
        if want > self.written:
            self.padded += want - self.written
            self._put(np.zeros(want - self.written, dtype=np.float32))
        self._q.put(None)
        self._th.join()
        self._f.close()


# ---------------------------------------------------------------- backends

def default_ids():
    """(default output id/name, default input id/name) - queried fresh each call, so changes are visible."""
    import soundcard as sc
    sp, mic = sc.default_speaker(), sc.default_microphone()
    return (str(sp.id), sp.name), (str(mic.id), mic.name)


class PyawBackend:
    """PyAudioWPatch: callback streams. Loopback sends no callbacks during silence."""
    name = "pyaw"

    def __init__(self, loop_track, mic_track, log):
        self.lt, self.mt, self.log = loop_track, mic_track, log
        self.p = self.sl = self.sm = None

    def open(self):
        import pyaudiowpatch as pa
        self.p = p = pa.PyAudio()
        lb = p.get_default_wasapi_loopback()
        wasapi = p.get_host_api_info_by_type(pa.paWASAPI)
        mic = p.get_device_info_by_index(wasapi["defaultInputDevice"])

        def mk(info, track):
            ch = int(info["maxInputChannels"])
            rate = int(info["defaultSampleRate"])

            def cb(data, frames, ti, status):
                x = np.frombuffer(data, dtype=np.float32).reshape(-1, ch).mean(axis=1)
                if rate != RATE:
                    x = np.interp(np.arange(0, len(x), rate / RATE), np.arange(len(x)), x).astype(np.float32)
                if status & pa.paInputOverflow:
                    track.overflows += 1
                track.push(x, time.perf_counter())
                return (None, pa.paContinue)

            s = p.open(format=pa.paFloat32, channels=ch, rate=rate, input=True,
                       input_device_index=info["index"], frames_per_buffer=rate // 20, stream_callback=cb)
            self.log(f"{track.name}: {info['name']} {rate} Hz x{ch}")
            return s

        self.sl, self.sm = mk(lb, self.lt), mk(mic, self.mt)

    def alive(self):
        return self.sl.is_active() and self.sm.is_active()

    def close(self):
        for s in (self.sl, self.sm):
            try:
                s.stop_stream(); s.close()
            except Exception:
                pass
        if self.p:
            self.p.terminate()


class SoundcardBackend:
    """soundcard: blocking record() in one thread per source; loopback fills silence by itself."""
    name = "soundcard"

    def __init__(self, loop_track, mic_track, log):
        self.lt, self.mt, self.log = loop_track, mic_track, log
        self.stop = threading.Event()
        self.threads = []

    def open(self):
        import soundcard as sc
        spk, mic = sc.default_speaker(), sc.default_microphone()
        lb = sc.get_microphone(str(spk.id), include_loopback=True)
        self.stop.clear()
        self.threads = []
        for dev, track in ((lb, self.lt), (mic, self.mt)):
            self.log(f"{track.name}: {dev.name} x{dev.channels}")
            th = threading.Thread(target=self._loop, args=(dev, track), daemon=True)
            th.start()
            self.threads.append(th)

    def _loop(self, dev, track):
        n = RATE // 20
        try:
            with dev.recorder(samplerate=RATE, blocksize=n) as rec:
                while not self.stop.is_set():
                    a = rec.record(numframes=n)
                    track.push(np.ascontiguousarray(a.mean(axis=1), dtype=np.float32), time.perf_counter())
        except Exception as e:
            self.log(f"{track.name}: thread ended: {type(e).__name__}: {e}")

    def alive(self):
        return all(t.is_alive() for t in self.threads)

    def close(self):
        self.stop.set()
        for t in self.threads:
            t.join(timeout=3)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--out", default="../_local/run")
    ap.add_argument("--backend", choices=["pyaw", "soundcard"], default="pyaw")
    ap.add_argument("--follow-devices", action="store_true",
                    help="reopen both streams when the default output/input device changes or a stream dies")
    ap.add_argument("--stats-every", type=float, default=5)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    events = []
    t0 = time.perf_counter()

    def log(msg):
        events.append({"t": round(time.perf_counter() - t0, 3), "msg": msg})
        print(f"[{events[-1]['t']:8.2f}] {msg}", flush=True)

    lt = Track("loopback", os.path.join(a.out, "loopback.wav"), t0)
    mt = Track("mic", os.path.join(a.out, "mic.wav"), t0)
    be = {"pyaw": PyawBackend, "soundcard": SoundcardBackend}[a.backend](lt, mt, log)
    be.open()
    cur = default_ids()
    log(f"start: backend={be.name} default out={cur[0][1]!r} in={cur[1][1]!r}")

    proc = psutil.Process()
    proc.cpu_percent(None)
    rows = []
    last_stat = t0
    restarts = 0
    end = t0 + a.seconds
    try:
        while True:
            now = time.perf_counter()
            if now >= end:
                break
            time.sleep(min(1.0, end - now))
            if a.follow_devices:
                try:
                    ids = default_ids()
                except Exception as e:
                    log(f"device query failed: {e}")
                    ids = cur
                if ids != cur or not be.alive():
                    log(f"device change / dead stream: {cur[0][1]!r}/{cur[1][1]!r} -> {ids[0][1]!r}/{ids[1][1]!r}; reopening")
                    be.close()
                    for i in range(10):
                        try:
                            be.open(); break
                        except Exception as e:
                            log(f"reopen attempt {i+1} failed: {e}"); time.sleep(1)
                    cur, restarts = ids, restarts + 1
            if time.perf_counter() - last_stat >= a.stats_every:
                last_stat = time.perf_counter()
                cpu = proc.cpu_percent(None)
                rows.append([round(last_stat - t0, 3), mt.raw, lt.raw, mt.written, lt.written,
                             round(cpu, 2), proc.memory_info().rss // 1024])
    except KeyboardInterrupt:
        log("interrupted")
    t_end = time.perf_counter()
    be.close()
    lt.finish(t_end)
    mt.finish(t_end)

    with open(os.path.join(a.out, "stats.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "mic_raw", "loop_raw", "mic_written", "loop_written", "cpu_pct_of_one_core", "rss_kb"])
        w.writerows(rows)
    dur = t_end - t0
    summ = {"backend": be.name, "duration_s": round(dur, 3), "rate": RATE, "restarts": restarts, "events": events}
    for t in (lt, mt):
        summ[t.name] = {"raw_frames": t.raw, "written": t.written, "padded_frames": t.padded,
                        "pad_events": t.pad_events, "dropped_frames": t.dropped, "overflows": t.overflows,
                        "blocks": t.blocks, "peak": round(t.peak, 5),
                        "rms": round((t.sq / max(t.raw, 1)) ** 0.5, 6)}
    ct = proc.cpu_times()
    summ["cpu_total_pct_of_one_core"] = round(100 * (ct.user + ct.system) / dur, 3)   # exact, whole run
    cpu = [r[5] for r in rows]
    if cpu:
        summ["cpu_pct_of_one_core"] = {"mean": round(float(np.mean(cpu)), 2), "max": round(float(np.max(cpu)), 2)}
        summ["rss_mb_end"] = round(rows[-1][6] / 1024, 1)
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in summ.items() if k != "events"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
