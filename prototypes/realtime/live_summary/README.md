# prototypes/realtime/live_summary

Step B4: when to call a language model during a call. A simulator of the trigger policy on a replayed transcript with a **stub** model.
Findings: [`docs/realtime.md`](../../../docs/realtime.md), section "Живая саммаризация". Nothing here is imported from `src/` by the simulator
(only the optional `live_check.py` reads the product adapters); profile `cli-codex` was fitted to its first real run (Codex CLI, 2026-10-25), no new dependencies, no network by default.

| file | what it does |
|---|---|
| `sim.py` | the simulator: `Policy` (all trigger / limit / context settings), stub `Model` (latency, tokens, prefix cache, prices), `run()` -> calls, tokens, cost, cache hits, staleness, replica lag, fact coverage; presets `economical` / `balanced` / `frequent` |
| `run_grid.py` | the grid of settings x materials x model profiles x pause stretch -> `../_local/live_grid.json` |
| `report_tables.py` | markdown tables for the docs from that grid |
| `selfcheck.py` | sanity checks (limits are respected, cost rule = `Provider.estimate_cost`) |
| `live_check.py` | OPTIONAL: the same three context modes against a real provider on an INVENTED transcript; refuses to start without `--live NAME --invented-transcript-ok` and a transcript from `../_local/ref_runs` |

```
python selfcheck.py
python sim.py ..\_local\ref_runs\<job>\export\result.json --preset balanced --verbose
python sim.py <result.json> --set interval_s=45 --set context=full --latency cli-slow --prices anthropic-like --cache anthropic
python sim.py <result.json> --pause-stretch 4        # pauses between speech x4 (the synthetic materials only have 0.25-0.7 s pauses)
python run_grid.py && python report_tables.py
```

Inputs are the pipeline results (`export/result.json`) of the synthetic step-B2 materials (`streaming_asr/make_materials.py`); real calls were not used.
Assumptions that are NOT measurements are listed in the docs, section 4.1 (model latency profiles, cache lifetime, the two price sets other than the repository example).
