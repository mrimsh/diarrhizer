"""OPTIONAL real-provider check of the three context modes (full / window / incremental) on an INVENTED transcript.

    python live_check.py <result.json of a synthetic material> --live "<provider name>" --invented-transcript-ok [--every 90] [--modes full,incremental]

Not run by default, never by the simulator. It sends the (invented, synthetic) transcript text of ../_local/ref_runs/... to the provider
you name - through the repository's own adapter (diarrhizer.adapters.llm, read-only import; the provider comes from summary_providers.json,
keys from the environment) - so it refuses to start unless
  * --live NAME is given (explicit consent for this run),
  * --invented-transcript-ok is given (you confirm the transcript is the invented test material), and
  * the transcript file lies under prototypes/realtime/_local/ref_runs (the pipeline outputs of the synthetic Б2 materials).
Per call it prints the REAL usage (input/output/cached tokens), latency and cost estimate, and at the end the share of "fact words"
(see sim.py) kept in the last summary of each mode - the real counterpart of the stub's capacity model.
"""
import argparse, os, sys, time
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "src")))      # read-only use of the product adapters
import sim

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("transcript")
ap.add_argument("--live", required=True, metavar="PROVIDER", help="name of a provider from summary_providers.json")
ap.add_argument("--invented-transcript-ok", action="store_true")
ap.add_argument("--every", type=float, default=90.0, help="seconds of audio between summaries")
ap.add_argument("--modes", default="full,window,incremental")
ap.add_argument("--window-min", type=float, default=3.0)
ap.add_argument("--summary-tokens", type=int, default=400)
a = ap.parse_args()
real = os.path.realpath(a.transcript)
allowed = os.path.realpath(os.path.join(HERE, "..", "_local", "ref_runs"))
if not a.invented_transcript_ok or not real.startswith(allowed + os.sep):
    raise SystemExit("refusing: pass --invented-transcript-ok and a transcript under prototypes/realtime/_local/ref_runs (synthetic material only)")

from diarrhizer.adapters.llm import complete
from diarrhizer.summary.providers import ProviderStore

provider = ProviderStore.load().get(a.live)
reps = sim.load_transcript(a.transcript, 0.0)
stub = sim.StubModel(reps, 3.0)
facts = sim._facts(reps, stub)
head = sim.TEMPLATE_HEAD + f"Длина ответа — не больше {a.summary_tokens} токенов. "
end = max(r.end for r in reps)
checkpoints = [t for t in range(int(a.every), int(end) + int(a.every), int(a.every))]
print(f"provider={provider.name} model={provider.model or '(default)'} checkpoints={len(checkpoints)} modes={a.modes}")
final = {}
for mode in a.modes.split(","):
    prev, cut, total_in, total_out, total_lat, cost = "", 0.0, 0, 0, 0.0, 0.0
    for t in checkpoints:
        seen = [r for r in reps if r.end <= t]
        if mode == "full":
            body = "\n".join(r.line for r in seen)
            prompt = head + "\n\n<transcript>\n" + body + "\n</transcript>\n\nЗаметки:"
        elif mode == "window":
            body = "\n".join(r.line for r in seen if r.end >= t - a.window_min * 60)
            prompt = head + "\n\n<transcript>\n" + body + "\n</transcript>\n\nЗаметки:"
        else:
            new = "\n".join(r.line for r in seen if r.end > cut)
            prompt = head + "\n\n<previous_summary>\n" + prev + "\n</previous_summary>\n\n<new_text>\n" + new + "\n</new_text>\n\nОбновлённые заметки:"
        t0 = time.perf_counter()
        reply = complete(provider, prompt)
        lat = time.perf_counter() - t0
        prev, cut = reply.text, t
        u = reply.usage
        total_in += u.input_tokens if u else 0; total_out += u.output_tokens if u else 0; total_lat += lat
        cost += reply.cost_usd or 0.0
        print(f"[{mode:11s} t={t:4d}s] in={u.input_tokens if u else '?'} out={u.output_tokens if u else '?'} cached={u.cached_tokens if u else '?'} latency={lat:.1f}s")
    final[mode] = prev
    cov = sim._coverage(facts, prev)
    print(f"== {mode}: calls={len(checkpoints)} in={total_in} out={total_out} mean latency={total_lat / len(checkpoints):.1f}s cost~{cost:.4f} USD "
          f"fact coverage all={cov['all']:.3f} first_half={cov['first_half']:.3f} last_5min={cov['last_5min']:.3f}")
print("\n--- last summary of each mode ---")
for mode, text in final.items():
    print(f"\n[{mode}]\n{text}")
