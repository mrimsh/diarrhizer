"""Concatenate materials (wav, 48k wav, stems by speaker id, json, ref) into one longer material.

    python concat_materials.py <out name> <part1> <part2> ...      # parts are names in ../_local/materials
Stems of speakers with the same letter are joined on the same track (letters are re-used across parts on purpose:
it keeps the number of speakers at 3 while the material grows).
"""
import glob, json, os, sys
import numpy as np, soundfile as sf
M = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_local", "materials"))
out, parts = sys.argv[1], sys.argv[2:]
gap = 1.5
def cat(suffix, sr):
    segs = []
    for p in parts:
        x, _ = sf.read(os.path.join(M, p + suffix), dtype="float32"); segs += [x, np.zeros(int(gap * sr), "float32")]
    sf.write(os.path.join(M, out + suffix), np.concatenate(segs[:-1]), sr, subtype="PCM_16")
cat(".wav", 16000); cat(".48k.wav", 48000)
turns, t0 = [], 0.0
for p in parts:
    j = json.load(open(os.path.join(M, p + ".json"), encoding="utf-8"))
    turns += [dict(r, start=round(r["start"] + t0, 2), end=round(r["end"] + t0, 2)) for r in j["turns"]]
    t0 += j["duration_s"] + gap
json.dump({"duration_s": round(t0 - gap, 2), "turns": turns}, open(os.path.join(M, out + ".json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
open(os.path.join(M, out + ".ref.txt"), "w", encoding="utf-8").write(" ".join(r["text"] for r in sorted(turns, key=lambda r: r["start"])))
for s in sorted({r["speaker"] for r in turns}):
    segs = []
    for p in parts:
        f = os.path.join(M, f"{p}.stem_{s}.wav")
        n = int(round(json.load(open(os.path.join(M, p + ".json")))["duration_s"] * 16000))
        x = sf.read(f, dtype="float32")[0] if os.path.exists(f) else np.zeros(n, "float32")
        segs += [x, np.zeros(int(gap * 16000), "float32")]
    sf.write(os.path.join(M, f"{out}.stem_{s}.wav"), np.concatenate(segs[:-1]), 16000, subtype="PCM_16")
print(out, round((t0 - gap) / 60, 2), "min,", len(turns), "turns")
