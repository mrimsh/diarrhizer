"""Variant B: whisper_streaming (ufal) with the LocalAgreement-2 policy on faster-whisper.

    python variant_ufal.py <material> [--model large-v3] [--min-chunk 1.0] [--trim 15] [--vad] [--lang ru] [--out run.json]

The repository is cloned into ../_local/whisper_streaming (not a dependency, not installed): `git clone --depth 1
https://github.com/ufal/whisper_streaming`. Every --min-chunk seconds the whole not-yet-trimmed buffer (up to --trim s)
is decoded again; a word is shown only after two consecutive hypotheses agree on it (committed output = no flicker).
librosa is only used by its file loader, which we do not call, so a stub module replaces it (no extra install).
"""
import argparse, os, sys, time, types
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "_local", "whisper_streaming")))
import importlib.machinery
_stub = types.ModuleType("librosa"); _stub.__spec__ = importlib.machinery.ModuleSpec("librosa", None)
sys.modules.setdefault("librosa", _stub)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from stream_lib import Monitor, load16, run_stream, save, SR
from whisper_online import FasterWhisperASR, OnlineASRProcessor

ap = argparse.ArgumentParser()
ap.add_argument("material"); ap.add_argument("--model", default="large-v3"); ap.add_argument("--min-chunk", type=float, default=1.0)
ap.add_argument("--trim", type=float, default=15.0); ap.add_argument("--vad", action="store_true"); ap.add_argument("--lang", default="ru")
ap.add_argument("--out"); ap.add_argument("--speed", type=float, default=1.0)
a = ap.parse_args()

mon = Monitor().start()
t_load = time.perf_counter()
asr = FasterWhisperASR(a.lang, modelsize=a.model, logfile=open(os.devnull, "w"))
if a.vad:
    asr.use_vad()
online = OnlineASRProcessor(asr, buffer_trimming=("segment", a.trim), logfile=open(os.devnull, "w"))
load_s = round(time.perf_counter() - t_load, 1)
x = load16(a.material)
asr.transcribe(x[:SR * 3])              # warm-up


def worker(q, log):
    online.init()
    acc, done = [], False
    while not done:
        item = q.get()
        if item is None:
            break
        acc.append(item[1])
        while True:                                    # take everything that has queued up while we were decoding
            try:
                item = q.get_nowait()
            except Exception:
                break
            if item is None:
                done = True
                break
            acc.append(item[1])
        if sum(len(c) for c in acc) >= a.min_chunk * SR or done:
            online.insert_audio_chunk(np.concatenate(acc)); acc = []
            o = online.process_iter()
            if o and o[0] is not None:
                log.emit(o[2], o[0], o[1])
    if acc:
        online.insert_audio_chunk(np.concatenate(acc))
    o = online.finish()
    if o and o[0] is not None:
        log.emit(o[2], o[0], o[1])


log = run_stream(x, worker, speed=a.speed)
res = mon.stop()
meta = dict(variant="ufal", material=os.path.basename(a.material), model=a.model, min_chunk=a.min_chunk, trim=a.trim, vad=a.vad,
            lang=a.lang, model_load_s=load_s, audio_s=round(len(x) / SR, 1))
out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.material)), "..", "stream_runs", os.path.basename(a.material).replace(".wav", "") + ".ufal.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
save(out, log, meta, res)
print(f"{meta['material']}: {len(log.events)} segments, wall {log.total_wall}s, audio {meta['audio_s']}s, resources {res}")
