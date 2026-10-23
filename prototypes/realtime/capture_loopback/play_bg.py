"""Play material to the default output so there is something to capture.

    python play_bg.py speech [seconds] [gain]   # loops ../_local/speech48k.wav
    python play_bg.py noise  [seconds] [gain]   # very quiet white noise (default gain 0.003 = -50 dBFS)
Loopback taps the digital mix, so a tiny digital level still counts as "sound is playing".
"""
import sys, time
import numpy as np, soundfile as sf, sounddevice as sd

kind = sys.argv[1]
secs = float(sys.argv[2]) if len(sys.argv) > 2 else 20
if kind == "speech":
    gain = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
    data, sr = sf.read("../_local/speech48k.wav", dtype="float32")
    data = np.tile(data, (int(secs * sr // len(data)) + 2, 1))[: int(secs * sr)] * gain
else:
    gain = float(sys.argv[3]) if len(sys.argv) > 3 else 0.003
    sr = 48000
    data = (np.random.default_rng(1).standard_normal((int(secs * sr), 2)) * gain).astype("float32")
sd.play(data, sr)
sd.wait()
