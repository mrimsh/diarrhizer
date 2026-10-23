"""Quiet noise that follows the default output device (re-resolved every second), so loopback has a signal after a switch.

    python play_follow.py [seconds] [gain]
"""
import sys, time
import numpy as np
import soundcard as sc

secs = float(sys.argv[1]) if len(sys.argv) > 1 else 60
gain = float(sys.argv[2]) if len(sys.argv) > 2 else 0.003
x = (np.random.default_rng(1).standard_normal((48000, 2)) * gain).astype("float32")
end = time.time() + secs
last = None
while time.time() < end:
    try:
        spk = sc.default_speaker()
        if spk.name != last:
            print("playing to:", spk.name, flush=True); last = spk.name
        spk.play(x, samplerate=48000)
    except Exception as e:
        print("play failed:", type(e).__name__, e, flush=True); time.sleep(0.5)
