"""Markdown tables for docs/realtime.md from the grid written by run_grid.py (../_local/live_grid.json)."""
import json, os, sys
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
g = json.load(open(os.path.join(HERE, "..", "_local", "live_grid.json"), encoding="utf-8"))
R = {tuple(k.split(" | ")): v for k, v in g.items()}      # (material, profile, setting, stretch)
FAST, SLOW = "api-fast, цены репо, кэш без скидки", "cli-slow, цены репо, кэш без скидки"
C50, ANT, ANT0 = "api-fast, кэш -50 %", "api-fast, anthropic-подобные цены, явный кэш", "api-fast, anthropic-подобные цены, без кэша"
LONG = "звонок 32 мин"

def find(mat, prof, prefix, st="x1"):
    for (m, p, n, s), v in R.items():
        if m == mat and p == prof and s == st and n.startswith(prefix):
            return n, v
    raise KeyError((mat, prof, prefix, st))

def row(cells): print("| " + " | ".join(str(c) for c in cells) + " |")

print("### T1: 32-минутный звонок, api-fast, цены репо (gpt-4o-mini: 0,15 / 0,6 $ за 1 М), паузы как в материале")
row(["Настройки", "вызовов", "вход, тыс. ток.", "выход, тыс. ток.", "$/час", "возраст саммари, с (ср. / p95)", "отставание реплики, с (ср.)"])
row(["---"] * 7)
for p in ("A.", "B.", "C.", "D.", "E.", "G.", "H.", "I.", "K.", "L.", "M."):
    n, v = find(LONG, FAST, p)
    row([n, v["calls"], round(v["in_tokens"] / 1000, 1), round(v["out_tokens"] / 1000, 1), v["cost_usd_per_hour"], f'{v["age_s_mean"]} / {v["age_s_p95"]}', v["replica_lag_s_mean"]])

print("\n### T2: кэш префикса, 32-минутный звонок ($/час и доля входных токенов из кэша)")
row(["Контекст", "цены репо, кэш без скидки", "кэш −50 % (доля из кэша)", "anthropic-подобные, без кэша", "anthropic-подобные, явный кэш (доля из кэша)"])
row(["---"] * 5)
for p, label in (("B.", "весь текст, каждые 60 с"), ("H.", "весь текст, сбалансированно"), ("I.", "окно 5 мин, сбалансированно"), ("G.", "инкрементально, сбалансированно"), ("G3", "инкрементально + инструкция 1500 ток.")):
    a = find(LONG, FAST, p)[1]; b = find(LONG, C50, p)[1]; c = find(LONG, ANT0, p)[1]; d = find(LONG, ANT, p)[1]
    row([label, a["cost_usd_per_hour"], f'{b["cost_usd_per_hour"]} ({b["cache_hit_share_of_input"]:.0%})', c["cost_usd_per_hour"], f'{d["cost_usd_per_hour"]} ({d["cache_hit_share_of_input"]:.0%})'])

print("\n### T3: перекрытие вызовов (busy), 32 мин: api-fast против cli-slow")
row(["Настройки", "профиль", "вызовов", "пропущено", "ожиданий (шагов 0,5 с)", "возраст, с (ср. / p95)", "$/час"])
row(["---"] * 7)
for p in ("A.", "A'", "L.", "L'"):
    for prof in (FAST, SLOW):
        n, v = find(LONG, prof, p)
        row([n, "api-fast" if prof == FAST else "cli-slow", v["calls"], v["skipped_busy"], v["defer_wait_steps"], f'{v["age_s_mean"]} / {v["age_s_p95"]}', v["cost_usd_per_hour"]])

print("\n### T4: чувствительность к паузам (32 мин, api-fast): паузы как в материале (x1, 0,25–0,7 с) и растянутые в 4 раза (x4)")
row(["Настройки", "паузы", "вызовов", "возраст, с (ср. / p95)", "$/час"])
row(["---"] * 5)
for p in ("F.", "F2", "G.", "G2", "L."):
    for st in ("x1", "x4"):
        n, v = find(LONG, FAST, p, st)
        row([n, st, v["calls"], f'{v["age_s_mean"]} / {v["age_s_p95"]}', v["cost_usd_per_hour"]])

print("\n### T5: пресеты на трёх материалах (api-fast, цены репо / anthropic-подобные с явным кэшем), паузы x4")
row(["Пресет", "материал", "вызовов", "вызовов/час", "возраст, с (ср. / p95)", "$/час (репо)", "$/час (anthropic-подобные, кэш)"])
row(["---"] * 7)
for p in ("K.", "G.", "L."):
    for mat in ("dialog 4.9 мин", "митинг 7.7 мин", LONG):
        n, v = find(mat, FAST, p, "x4"); w = find(mat, ANT, p, "x4")[1]
        row([n, mat, v["calls"], v["calls_per_hour"], f'{v["age_s_mean"]} / {v["age_s_p95"]}', v["cost_usd_per_hour"], w["cost_usd_per_hour"]])

print("\n### T6: ёмкость summary (заглушка), митинг 7,7 мин (тексты не повторяются): доля «фактов» в последнем саммари")
row(["Настройки", "итоговое покрытие", "один вызов по всему тексту, та же ёмкость", "относительно (1,0 = без потерь)", "покрытие фактов первой половины", "последние 5 мин"])
row(["---"] * 6)
for p in ("D.", "G.", "H.", "I.", "L."):
    n, v = find("митинг 7.7 мин", FAST, p)
    row([n, v["fact_coverage_final"], v["fact_coverage_ideal_same_capacity"], v["coverage_vs_ideal"], v["fact_coverage_first_half"], v["fact_coverage_last_5min"]])

CODEX = "cli-codex (измерено), без цены токенов"
print("\n### T7: подписочный CLI (профиль измерен на Codex: ≈ 12,5 тыс. токенов фиксированного входа на запрос, ≈ 2 с + 30 ток/с), паузы x4")
row(["Пресет", "материал", "вызовов/час", "токенов/час (вход + выход), тыс.", "из них из кэша, %", "вызовов до потолка 400 тыс. токенов", "возраст, с (ср. / p95)"])
row(["---"] * 7)
for p in ("K.", "G.", "L."):
    for mat in ("dialog 4.9 мин", "митинг 7.7 мин", LONG):
        n, v = find(mat, CODEX, p, "x4"); h = 3600 / v["audio_s"]
        per_call = (v["in_tokens"] + v["out_tokens"]) / max(v["calls"], 1)
        row([n, mat, v["calls_per_hour"], round((v["in_tokens"] + v["out_tokens"]) * h / 1000), f'{v["cache_hit_share_of_input"]:.0%}', round(400000 / per_call), f'{v["age_s_mean"]} / {v["age_s_p95"]}'])
print("\n### T8: те же настройки, 32 мин, режимы контекста на профиле подписочного CLI (x1)")
row(["Настройки", "вызовов", "вход, тыс. ток.", "из кэша, %", "возраст, с (ср. / p95)"])
row(["---"] * 5)
for p in ("B.", "C.", "D.", "H.", "G.", "I."):
    n, v = find(LONG, CODEX, p)
    row([n, v["calls"], round(v["in_tokens"] / 1000), f'{v["cache_hit_share_of_input"]:.0%}', f'{v["age_s_mean"]} / {v["age_s_p95"]}'])
