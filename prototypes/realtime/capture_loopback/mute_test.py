"""Guided mute test: records mic + loopback for 80 s and prints what to do at each moment.

    python mute_test.py <out dir>

Phases (keep talking the whole time, in a Meet call):
  0-15 s  everything on        15-30 s mute ONLY in Meet     30-45 s everything on
  45-60 s mute ONLY with the button on the microphone        60-80 s everything on
Afterwards the script prints, per phase, how many mic seconds are exact digital zero and the mic level.
"""
import json, os, subprocess, sys, time
import numpy as np, soundfile as sf

out = sys.argv[1] if len(sys.argv) > 1 else "../_local/mute_test"
phases = [(0, 15, "ВСЁ ВКЛЮЧЕНО, говорите"), (15, 30, "ВЫКЛЮЧИТЕ МИКРОФОН ТОЛЬКО В MEET (кнопка на микрофоне не нажата), говорите"),
          (30, 45, "ВКЛЮЧИТЕ обратно, говорите"), (45, 60, "НАЖМИТЕ КНОПКУ НА МИКРОФОНЕ (в Meet микрофон включён), говорите"),
          (60, 80, "ОТПУСТИТЕ кнопку, всё включено, говорите")]
sys.stdout.reconfigure(encoding="utf-8")
p = subprocess.Popen([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "capture.py"),
                      "--seconds", "80", "--out", out], stdout=subprocess.DEVNULL)
t0 = time.perf_counter()          # capture.py starts its own T0 ~0.3 s later (python start-up); phase edges are +-0.5 s
cues = []
for a, b, text in phases:
    while time.perf_counter() - t0 < a:
        time.sleep(0.05)
    cues.append({"t": round(time.perf_counter() - t0, 2), "phase": text})
    print(f"\n[{a:2d}-{b:2d} с] {text}", flush=True)
p.wait()
json.dump(cues, open(os.path.join(out, "cues.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
m, sr = sf.read(os.path.join(out, "mic.wav"), dtype="int16")
print("\nРезультат (первые/последние 1.5 с каждой фазы отброшены из-за неточности границ):")
for a, b, text in phases:
    x = m[int((a + 1.5) * sr):int((b - 1.5) * sr)]
    zero = (x == 0).mean() if len(x) else float("nan")
    rms = float(np.sqrt(np.mean((x.astype("float32") / 32768) ** 2))) if len(x) else float("nan")
    print(f"  {a:2d}-{b:2d} с: доля нулевых отсчётов {zero:6.1%}, RMS {rms:.5f}  — {text.split(',')[0]}")
