"""Clock drift of the two tracks from stats.csv written by capture.py.

    python analyze_drift.py <run dir>

raw frames per track are counted before any zero padding, so   frames/48000 - t   is that device's clock error
against the PC's QPC clock (perf_counter). The *difference* of the two is what makes the tracks slide against each
other. Valid only for runs where the loopback had a signal the whole time (otherwise the pyaw loopback has no packets).
"""
import csv, json, sys, os
import numpy as np

d = sys.argv[1]
rows = list(csv.DictReader(open(os.path.join(d, "stats.csv"))))
t = np.array([float(r["t_s"]) for r in rows])
mic = np.array([float(r["mic_raw"]) for r in rows]) / 48000
loop = np.array([float(r["loop_raw"]) for r in rows]) / 48000
skip = t > 10                                   # ignore start-up (first packet latency)
def fit(y):
    a, b = np.polyfit(t[skip], y[skip], 1)      # y ~ a*t + b ; a-1 = rate error
    return (a - 1) * 1e6, b
ppm_m, off_m = fit(mic)
ppm_l, off_l = fit(loop)
diff = (loop - mic)[skip]
res = {"run": d, "points": int(skip.sum()), "span_s": round(float(t[-1] - t[skip][0]), 1),
       "mic_ppm_vs_qpc": round(ppm_m, 1), "loop_ppm_vs_qpc": round(ppm_l, 1),
       "loop_minus_mic_ppm": round(ppm_l - ppm_m, 1),
       "loop_minus_mic_ms_start": round(float(diff[0]) * 1000, 1),
       "loop_minus_mic_ms_end": round(float(diff[-1]) * 1000, 1),
       "loop_minus_mic_ms_change": round(float(diff[-1] - diff[0]) * 1000, 1),
       "loop_minus_mic_ms_min_max": [round(float(diff.min()) * 1000, 1), round(float(diff.max()) * 1000, 1)],
       "mic_ms_vs_wall_end": round(float(mic[-1] - t[-1]) * 1000, 1), "loop_ms_vs_wall_end": round(float(loop[-1] - t[-1]) * 1000, 1)}
cpu = [float(r["cpu_pct_of_one_core"]) for r in rows]
res["cpu_sampled_pct_of_one_core_mean"] = round(float(np.mean(cpu)), 2)
s = os.path.join(d, "summary.json")
if os.path.exists(s):
    j = json.load(open(s, encoding="utf-8"))
    res["cpu_total_pct_of_one_core"] = j.get("cpu_total_pct_of_one_core")
    res["rss_mb_end"] = j.get("rss_mb_end")
    for k in ("loopback", "mic"):
        res[k] = {x: j[k][x] for x in ("padded_frames", "pad_events", "dropped_frames", "overflows", "blocks")}
print(json.dumps(res, ensure_ascii=False, indent=1))
