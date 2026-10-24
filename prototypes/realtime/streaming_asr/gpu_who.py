r"""Which processes use the GPU (Windows 'GPU Engine' counters, every 5 s). Summarises a typeperf CSV: top processes by average utilisation.

    typeperf "\GPU Engine(*)\Utilization Percentage" -si 5 -sc 100 -o who.csv -y
    python gpu_who.py who.csv
"""
import csv, re, sys, collections
import psutil
rows = list(csv.reader(open(sys.argv[1], encoding="utf-8", errors="ignore")))
hdr, data = rows[0], rows[1:]
pid_cols = collections.defaultdict(list)
for i, h in enumerate(hdr):
    m = re.search(r"pid_(\d+)", h)
    if m:
        pid_cols[int(m.group(1))].append(i)
tot = collections.defaultdict(list)
for r in data:
    for pid, cols in pid_cols.items():
        v = 0.0
        for c in cols:
            try: v += float(r[c])
            except Exception: pass
        tot[pid].append(v)
def name(p):
    try: return psutil.Process(p).name()
    except Exception: return "?"
agg = collections.defaultdict(lambda: [0.0, 0.0, 0])
for pid, v in tot.items():
    a = agg[name(pid)]; a[0] += sum(v) / max(len(v), 1); a[1] = max(a[1], max(v)); a[2] += 1
for n, (mean, mx, k) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:8]:
    print(f"{n:28s} mean {mean:6.1f}%  peak {mx:6.1f}%  ({k} pids)")
print("samples:", len(data))
