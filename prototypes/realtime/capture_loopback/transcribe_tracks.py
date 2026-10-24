"""Timeline + Whisper transcripts of the two tracks of a run (offline check of 'who is on which track').

    python transcribe_tracks.py <run dir> [--model large-v3] [--find "фраза" ...]

Prints a per-second activity timeline, then the transcript of mic.wav and of loopback.wav with times,
and for every --find word reports in which track it was recognised (a phrase of mine found in the
loopback track = my voice came back through the client; a phrase of the other side found in the mic = leak).
Uses faster-whisper from the project environment; HF_HUB_OFFLINE=1 is advised (models come from the cache).
"""
import argparse, os, re, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # numpy/scipy + ctranslate2 both ship an OpenMP runtime
import numpy as np, soundfile as sf
from scipy.signal import resample_poly

sys.stdout.reconfigure(encoding="utf-8")
ap = argparse.ArgumentParser()
ap.add_argument("run"); ap.add_argument("--model", default="large-v3"); ap.add_argument("--lang", default="ru")
ap.add_argument("--find", nargs="*", default=[])
a = ap.parse_args()
sr, thr = 48000, 0.003
m, _ = sf.read(os.path.join(a.run, "mic.wav"), dtype="float32"); l, _ = sf.read(os.path.join(a.run, "loopback.wav"), dtype="float32")
n = min(len(m), len(l)); m, l = m[:n], l[:n]
rms = lambda x: [float(np.sqrt(np.mean(x[i * sr:(i + 1) * sr] ** 2))) for i in range(len(x) // sr)]
rm, rl = rms(m), rms(l)
line = lambda r: "".join("#" if v > thr else "." for v in r)
print("sec  " + "".join(f"{i:<10d}" for i in range(0, len(rm), 10)))
print("mic ", line(rm)); print("loop", line(rl))
from faster_whisper import WhisperModel
model = WhisperModel(a.model, device="cuda", compute_type="float16")
found = {}
for name, x in (("MIC", m), ("LOOPBACK", l)):
    y = resample_poly(x, 1, 3).astype("float32")
    segs, _ = model.transcribe(y, language=a.lang, vad_filter=True, beam_size=5, condition_on_previous_text=False)
    print(f"\n=== {name}")
    for s in segs:
        print("[%6.1f-%6.1f] %s" % (s.start, s.end, s.text.strip()))
        for w in a.find:
            if re.search(w, s.text, re.I):
                found.setdefault((name, w), []).append(round(s.start, 1))
print()
for w in a.find:
    print(f"'{w}': mic at {found.get(('MIC', w), [])}, loopback at {found.get(('LOOPBACK', w), [])}")
