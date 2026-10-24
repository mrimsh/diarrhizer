"""Grid of settings x materials x model profiles -> table (JSON + markdown). Stub model only; no network.

    python run_grid.py [--out ../_local/live_grid.json]
"""
import argparse, dataclasses, glob, json, os, sys
import sim

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "..", "_local", "ref_runs")
MATERIALS = {"dialog 4.9 мин": "dialog", "митинг 7.7 мин": "meeting_noisy", "звонок 32 мин": "long32"}


def transcript(name):
    return sorted(glob.glob(os.path.join(REF, f"{name}.48k_*", "export", "result.json")))[-1]


def P(**kw):
    return dataclasses.replace(sim.Policy(), **kw)

# name -> Policy. Every line differs from "balanced" in what the name says.
GRID = {
    "A. каждые 30 с, весь текст":                 P(interval_s=30, pause_s=None, silence_ms=0, max_wait_s=0, min_new_words=1, context="full", min_interval_s=0),
    "B. каждые 60 с, весь текст":                 P(interval_s=60, pause_s=None, silence_ms=0, max_wait_s=0, min_new_words=1, context="full", min_interval_s=0),
    "C. каждые 60 с, окно 5 мин":                 P(interval_s=60, pause_s=None, silence_ms=0, max_wait_s=0, min_new_words=1, context="window", window_min=5, min_interval_s=0),
    "D. каждые 60 с, инкрементально":             P(interval_s=60, pause_s=None, silence_ms=0, max_wait_s=0, min_new_words=1, context="incremental", min_interval_s=0),
    "E. по 150 новых слов, инкрементально":       P(interval_s=None, pause_s=None, new_words=150, silence_ms=1200, max_wait_s=15, min_new_words=40),
    "F. по паузе 6 с, инкрементально":            P(interval_s=None, pause_s=6, silence_ms=1200, max_wait_s=15, min_new_words=40),
    "G. сбалансированно (120 с или пауза 6 с)":   sim.PRESETS["balanced"],
    "H. G, но весь текст":                        P(context="full"),
    "I. G, но окно 5 мин":                        P(context="window", window_min=5),
    "J. G, busy=skip":                            P(busy="skip"),
    "F2. по паузе 2 с, инкрементально":          P(interval_s=None, pause_s=2, silence_ms=1200, max_wait_s=15, min_new_words=40),
    "G2. 120 с или пауза 2 с":                    P(pause_s=2.0),
    "G3. G + длинная инструкция 1500 ток.":        P(template_tokens=1500),
    "A'. каждые 30 с, весь текст, busy=skip":     P(interval_s=30, pause_s=None, silence_ms=0, max_wait_s=0, min_new_words=1, context="full", min_interval_s=0, busy="skip"),
    "L'. часто, busy=skip":                       P(**{**dataclasses.asdict(sim.PRESETS["frequent"]), "busy": "skip"}),
    "K. экономно":                                sim.PRESETS["economical"],
    "L. часто":                                   sim.PRESETS["frequent"],
    "M. вручную (каждые 5 мин)":                  P(interval_s=None, pause_s=None, silence_ms=0, max_wait_s=0, min_new_words=1, manual_at=tuple(range(300, 4000, 300)), context="incremental"),
}
PROFILES = {  # label -> (latency profile, prices, cache)
    "api-fast, цены репо, кэш без скидки": ("api-fast", "repo-example", "openai-auto"),
    "cli-slow, цены репо, кэш без скидки": ("cli-slow", "repo-example", "openai-auto"),
    "api-fast, кэш -50 %": ("api-fast", "cache-50", "openai-auto"),
    "api-fast, anthropic-подобные цены, явный кэш": ("api-fast", "anthropic-like", "anthropic"),
    "api-fast, anthropic-подобные цены, без кэша": ("api-fast", "anthropic-like", "none"),
    "cli-codex (измерено), без цены токенов": ("cli-codex", "none", "openai-auto"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "_local", "live_grid.json"))
    a = ap.parse_args()
    out = {}
    for stretch in (1.0, 4.0):
        for mlabel, name in MATERIALS.items():
            reps = sim.load_transcript(transcript(name), 4.0, stretch)
            for plabel, (lat, prices, cache) in PROFILES.items():
                model = sim.make_model(lat, prices, cache)
                for gname, pol in GRID.items():
                    if model.overhead_tokens:             # a CLI's fixed input eats the default token_budget fast (see docs 4.6): measure the cadence itself
                        pol = dataclasses.replace(pol, token_budget=None)
                    out[f"{mlabel} | {plabel} | {gname} | x{stretch:g}"] = sim.run(reps, pol, model)
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("runs:", len(out), "->", a.out)


if __name__ == "__main__":
    main()
