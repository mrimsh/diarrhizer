"""Who is where: activity of the 'me' (mic) and 'others' (loopback) tracks and the bleed between them.

    python analyze_tracks.py <mic.wav> <loopback.wav> [--win 1.0] [--thr 0.003]

Per window (default 1 s):
  active_mic / active_loop    RMS above the threshold
  both                        both active (overlap: the two tracks speak at once)
  echo                        mic window contains a delayed copy of the loopback window:
                              peak of the normalised cross-correlation within +-300 ms > 0.35 (speakers -> mic path)
Prints shares of the active windows. Use it on a test call with yourself to see whether the loopback
signal also lands in the microphone track (open speakers) or not (headphones).
"""
import argparse
import numpy as np
import soundfile as sf

ap = argparse.ArgumentParser()
ap.add_argument("mic"); ap.add_argument("loop")
ap.add_argument("--win", type=float, default=1.0)
ap.add_argument("--thr", type=float, default=0.003)
ap.add_argument("--xthr", type=float, default=0.35)
a = ap.parse_args()
m, sr = sf.read(a.mic, dtype="float32"); l, sr2 = sf.read(a.loop, dtype="float32")
assert sr == sr2
n = min(len(m), len(l)); m, l = m[:n], l[:n]
W = int(a.win * sr); L = int(0.3 * sr)

def ncc_peak(x, y):
    """max |normalised cross-correlation| of x against y for lags in [-L, L]"""
    x = x - x.mean(); y = y - y.mean()
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    if nx < 1e-9 or ny < 1e-9:
        return 0.0
    N = 1 << int(np.ceil(np.log2(len(x) + len(y))))
    c = np.fft.irfft(np.fft.rfft(x, N) * np.conj(np.fft.rfft(y, N)), N)
    c = np.concatenate([c[-L:], c[:L + 1]])
    return float(np.abs(c).max() / (nx * ny))

tot = am = al = both = echo = 0
for i in range(0, n - W + 1, W):
    xm, xl = m[i:i + W], l[i:i + W]
    rm, rl = np.sqrt(np.mean(xm ** 2)), np.sqrt(np.mean(xl ** 2))
    tot += 1; am += rm > a.thr; al += rl > a.thr
    if rm > a.thr and rl > a.thr:
        both += 1
        echo += ncc_peak(xm, xl) > a.xthr
print(f"duration {n/sr:.1f}s, windows {tot} x {a.win}s, thr {a.thr}")
print(f"mic active {am} ({am/max(tot,1):.0%}), loopback active {al} ({al/max(tot,1):.0%}), both {both}")
print(f"windows with a coherent copy of loopback inside mic (echo): {echo} of {both} 'both active'")
