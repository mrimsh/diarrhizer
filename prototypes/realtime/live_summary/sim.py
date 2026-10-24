"""Simulator of live summarization during a call: a trigger policy on top of a replayed transcript and a STUB model.

    python sim.py <pipeline result.json> [--preset balanced] [--set key=value ...] [--json out.json]
    python sim.py --list-presets

The transcript is the output of the Diarrhizer pipeline (segments with start/end/speaker/text). It is "played" in virtual time
in steps of `dt` seconds (no real sleeping; `--speed N` makes it sleep dt/N per step, for a visual check). Text of a phrase
becomes available `asr_latency_s` after the phrase ends (step B2: ~4 s for the pseudo-stream), the speech activity (VAD) is
available at once. The model is a stub: latency = base + per-1k-input + per-output-token, input/output tokens are estimated from
characters, the summary is an extractive compression to `summary_tokens` (see StubModel) - enough to count calls, tokens, cost,
cache hits and staleness, NOT to judge the wording of a real summary. A real provider is called only with --live (live_check.py).

Policy (everything is a field of Policy; defaults = the proposed "balanced" preset):
  triggers (any of them makes a call "wanted"): interval_s, new_words, new_turns, pause_s, manual_at
  end of phrase: a wanted call waits for silence_ms of silence, at most max_wait_s, then it is sent anyway
  min_new_words: not enough new text -> no call (manual calls ignore it)
  context: full | window (window_min minutes) | incremental ({{previous_summary}} + {{new_text}})
  busy: while a call is running a new one is `defer`red (sent as soon as the running one ends, all new text coalesced) or `skip`ped
  limits: min_interval_s between call starts, max_calls_per_hour, max_session_calls, token_budget, cost_budget_usd
"""
import argparse, bisect, dataclasses, json, math, re, sys, time
from dataclasses import dataclass, field

sys.stdout.reconfigure(encoding="utf-8")

# ------------------------------------------------------------------ settings


@dataclass
class Policy:
    # triggers
    interval_s: float | None = 120.0
    new_words: int | None = None
    new_turns: int | None = None
    pause_s: float | None = 6.0
    manual_at: tuple = ()
    # end of phrase
    silence_ms: int = 1200
    max_wait_s: float = 15.0
    # how much text
    min_new_words: int = 40
    context: str = "incremental"          # full | window | incremental
    window_min: float = 5.0
    # overlap protection
    busy: str = "defer"                   # defer | skip
    # limits
    min_interval_s: float = 20.0
    max_calls_per_hour: int = 60
    max_session_calls: int | None = None
    token_budget: int | None = 400_000
    cost_budget_usd: float | None = None
    # model side (separate from the ordinary summarization's provider/model)
    summary_tokens: int = 600             # size of the summary the model writes
    template_tokens: int = 300            # size of the instruction part of the template


@dataclass
class Model:
    """Stub model: what a call costs in time, and how tokens / cache are counted."""
    latency_base_s: float = 1.0
    latency_per_1k_in_s: float = 0.4
    out_tokens_per_s: float = 40.0
    chars_per_token: float = 3.0          # = CHARS_PER_TOKEN of summary/chat.py (the product's own estimate); real tokenizers differ by model
    cache: str = "openai-auto"            # openai-auto (no write fee, >=1024 tokens, 128-token steps) | anthropic (explicit breakpoint, write fee) | none
    cache_ttl_s: float = 300.0
    overhead_tokens: int = 0              # fixed input the tool adds to every request (a subscription CLI's own system prompt / tools)
    price_in: float = 0.15                # USD per million tokens
    price_out: float = 0.60
    price_cache_read: float | None = None  # None: cache tokens are charged at price_in (as Provider.estimate_cost does)
    price_cache_write: float | None = None


LATENCY = {"api-fast": dict(latency_base_s=1.0, latency_per_1k_in_s=0.4, out_tokens_per_s=40.0),
           "cli-slow": dict(latency_base_s=8.0, latency_per_1k_in_s=0.8, out_tokens_per_s=25.0),   # ASSUMED profiles
           # MEASURED with live_check.py on the Codex CLI (12 calls, 2026-10-25): ~12.5k tokens of fixed input per request (11.3k of it cached),
           # latency ~ 2 s + output / 30 tokens/s (11-15 s for 200-410 output tokens; one cold first call took 25.8 s and is not in the fit)
           "cli-codex": dict(latency_base_s=2.0, latency_per_1k_in_s=0.0, out_tokens_per_s=30.0, overhead_tokens=12500)}
PRICES = {  # USD per 1M tokens. "repo-example" = the example provider of summary_providers.example.json (openai/gpt-4o-mini via OpenRouter);
            # the other two are ASSUMPTIONS to show where the cache matters, not quotes: plug in the real prices of a provider.
    "none": dict(price_in=0.0, price_out=0.0),        # subscription CLI: no price per token, tokens still count against the plan's windows
    "repo-example": dict(price_in=0.15, price_out=0.60),
    "cache-50": dict(price_in=0.15, price_out=0.60, price_cache_read=0.075),
    "anthropic-like": dict(price_in=3.0, price_out=15.0, price_cache_read=0.30, price_cache_write=3.75),
}
PRESETS = {
    "economical": Policy(interval_s=300, pause_s=10, silence_ms=2000, max_wait_s=30, min_new_words=120, context="incremental",
                         min_interval_s=120, max_calls_per_hour=20, summary_tokens=500),
    "balanced": Policy(),
    "frequent": Policy(interval_s=45, new_words=None, pause_s=3, silence_ms=800, max_wait_s=10, min_new_words=20, context="incremental",
                       min_interval_s=15, max_calls_per_hour=120, summary_tokens=700),
}

# ------------------------------------------------------------------ transcript


@dataclass
class Replica:
    start: float
    end: float
    speaker: str
    text: str
    avail: float = 0.0

    @property
    def line(self) -> str:
        return f"{self.speaker}: {self.text}"

    @property
    def words(self) -> int:
        return len(self.text.split())


def load_transcript(path: str, asr_latency_s: float, pause_stretch: float = 1.0) -> list[Replica]:
    """pause_stretch > 1 stretches every pause (> 0.3 s) between speech by that factor, speech itself unchanged: the synthetic
    materials only have 0.25-0.7 s pauses between turns, real calls have longer ones, so this is a sensitivity knob for the
    pause-based settings (pause_s, silence_ms), not a model of real pauses."""
    data = json.load(open(path, encoding="utf-8"))
    out = [Replica(float(s["start"]), float(s["end"]), str(s.get("speaker_id", "?")), str(s["text"]).strip(), 0.0)
           for s in data["segments"] if str(s.get("text", "")).strip()]
    if pause_stretch != 1.0:
        extra = []                                           # (original time where a pause ends, extra seconds added by the stretch)
        iv = speech_intervals(out)
        for (a0, b0), (a1, _) in zip(iv, iv[1:]):
            extra.append((a1, (a1 - b0) * (pause_stretch - 1.0)))
        shift = lambda t: t + sum(e for te, e in extra if te <= t + 1e-9)
        out = [Replica(shift(r.start), shift(r.end), r.speaker, r.text, 0.0) for r in out]
    for r in out:
        r.avail = r.end + asr_latency_s
    return sorted(out, key=lambda r: r.avail)


def speech_intervals(reps: list[Replica], gap: float = 0.3) -> list[tuple[float, float]]:
    iv = []
    for r in sorted(reps, key=lambda r: r.start):
        if iv and r.start - iv[-1][1] <= gap:
            iv[-1] = (iv[-1][0], max(iv[-1][1], r.end))
        else:
            iv.append((r.start, r.end))
    return iv


# ------------------------------------------------------------------ stub model (extractive, fixed capacity)

_STOP = set("это как так что чтобы тоже если или для при про над под без все всё ещё уже было были будет будем нужно можно очень мне нам вам они она оно его ее её тут там где когда потом тогда сейчас просто давайте хорошо ладно спасибо".split())


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-zа-яё0-9]+", text.lower().replace("ё", "е")) if len(w) >= 5 and w not in _STOP]


class StubModel:
    """Writes the summary as the most informative units that fit `summary_tokens`.

    A unit is a phrase (full / window) or a line of the previous summary plus the new phrases (incremental). Informative = rare words
    of the whole transcript (IDF, an oracle on purpose: the stub should be a *good* summarizer, so that what is lost in incremental mode is
    lost because of the capacity limit and not because of a dumb stub). Real summaries paraphrase; this is a capacity model, not a quality one.
    """

    def __init__(self, reps: list[Replica], cpt: float):
        self.cpt = cpt
        df: dict[str, int] = {}
        for r in reps:
            for w in set(_words(r.text)):
                df[w] = df.get(w, 0) + 1
        n = max(len(reps), 1)
        self.idf = {w: math.log(1 + n / c) for w, c in df.items()}

    def score(self, text: str) -> float:
        ws = set(_words(text))
        return sum(self.idf.get(w, 0.0) for w in ws) / math.sqrt(max(len(text) / self.cpt, 1.0))

    def summarize(self, units: list[tuple[float, str]], budget_tokens: int) -> list[tuple[float, str]]:
        """units: (time of its latest phrase end, text). Returns the kept units in chronological order."""
        ranked = sorted(units, key=lambda u: -self.score(u[1]))
        kept, used = [], 0
        for u in ranked:
            t = math.ceil(len(u[1]) / self.cpt)
            if used + t <= budget_tokens:
                kept.append(u); used += t
        return sorted(kept, key=lambda u: u[0])


# ------------------------------------------------------------------ the session


TEMPLATE_HEAD = ("Ты ведёшь краткие заметки по ходу рабочего созвона. Отвечай по-русски, коротко, списком. "
                 "Выделяй решения, поручения с исполнителем и сроком, цифры и открытые вопросы. Не придумывай того, чего нет в тексте. ")


def template_text(tokens: int, cpt: float) -> str:
    base = TEMPLATE_HEAD
    need = int(tokens * cpt)
    while len(base) < need:
        base += "Если в тексте есть противоречие, отметь его отдельным пунктом. "
    return base[:need]


def lcp(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


@dataclass
class Call:
    start: float
    end: float
    in_tokens: int
    out_tokens: int
    cached_read: int
    cache_write: int
    cost: float
    included_end: float
    new_words: int


def run(reps: list[Replica], pol: Policy, model: Model, asr_latency_s: float = 4.0, dt: float = 0.5, speed: float = 0.0, verbose: bool = False) -> dict:
    stub = StubModel(reps, model.chars_per_token)
    cpt = model.chars_per_token
    tmpl = template_text(pol.template_tokens, cpt)
    iv = speech_intervals(reps)
    iv_starts = [a for a, _ in iv]
    end_audio = max(r.end for r in reps)
    horizon = end_audio + asr_latency_s + 60

    def silence_at(t: float) -> float:
        i = bisect.bisect_right(iv_starts, t) - 1
        if i < 0:
            return t
        a, b = iv[i]
        return 0.0 if t <= b else t - b

    avail_i = 0
    seen: list[Replica] = []                 # text available so far
    cutoff = 0                               # index in `seen` already sent to the model (incremental / new-text counting)
    new_words_cnt = new_turns_cnt = 0
    summary_units: list[tuple[float, str]] = []      # incremental: the previous summary
    shown_end = None                         # latest phrase end included in the summary that is on screen
    shown_hist: list[tuple[float, float]] = []       # (time summary appeared, its included_end)
    calls: list[Call] = []
    running: dict | None = None
    last_call_start = -1e9
    interval_ref = 0.0
    want_since: float | None = None
    blocked = 0; skipped = 0; deferred_waits = 0
    manual = sorted(pol.manual_at)
    recent: list[tuple[float, str]] = []     # (call time, prompt) for the cache model
    tokens_used = 0; cost_used = 0.0
    call_starts: list[float] = []
    lag_per_replica: dict[int, float] = {}

    t = 0.0
    while t <= horizon:
        # a running call finishes
        if running and running["end"] <= t:
            c = running["call"]; calls.append(c)
            summary_units = running["units_after"]
            shown_end = c.included_end
            shown_hist.append((c.end, c.included_end))
            for idx in running["idxs"]:
                lag_per_replica.setdefault(idx, c.end - seen[idx].end)
            running = None
        # text arrives
        while avail_i < len(reps) and reps[avail_i].avail <= t:
            seen.append(reps[avail_i]); new_words_cnt += reps[avail_i].words; new_turns_cnt += 1; avail_i += 1
        # is a call wanted?
        fire_manual = bool(manual and manual[0] <= t)
        sil = silence_at(t)
        wanted = fire_manual
        if pol.interval_s is not None and t - interval_ref >= pol.interval_s and new_words_cnt > 0:
            wanted = True
        if pol.new_words is not None and new_words_cnt >= pol.new_words:
            wanted = True
        if pol.new_turns is not None and new_turns_cnt >= pol.new_turns:
            wanted = True
        if pol.pause_s is not None and sil >= pol.pause_s and new_words_cnt > 0:
            wanted = True
        if wanted and want_since is None:
            want_since = t
        if not wanted:
            want_since = None
        if wanted:
            waited_enough = sil >= pol.silence_ms / 1000 or (t - want_since) >= pol.max_wait_s
            enough_text = new_words_cnt >= pol.min_new_words or fire_manual
            spaced = t - last_call_start >= pol.min_interval_s or fire_manual
            hour_ok = sum(1 for s in call_starts if t - s < 3600) < pol.max_calls_per_hour
            session_ok = pol.max_session_calls is None or len(call_starts) < pol.max_session_calls
            budget_ok = (pol.token_budget is None or tokens_used < pol.token_budget) and (pol.cost_budget_usd is None or cost_used < pol.cost_budget_usd)
            if waited_enough and enough_text and spaced:
                if running:
                    if pol.busy == "skip":
                        skipped += 1; interval_ref = t; want_since = None
                        if fire_manual: manual.pop(0)
                    else:
                        deferred_waits += 1                      # stays wanted, goes out right after the running call (coalesced)
                elif not (hour_ok and session_ok and budget_ok):
                    blocked += 1
                    if fire_manual: manual.pop(0)
                else:
                    # ---- build the prompt
                    if pol.context == "full":
                        body = "\n".join(r.line for r in seen); idxs = list(range(len(seen)))
                        prompt = tmpl + "\n\n<transcript>\n" + body + "\n</transcript>\n\nЗаметки:"
                        units = [(r.end, r.line) for r in seen]
                        stable_end = len(tmpl) + len("\n\n<transcript>\n") + len(body) + len("\n</transcript>")
                    elif pol.context == "window":
                        lo = t - pol.window_min * 60
                        idxs = [i for i, r in enumerate(seen) if r.end >= lo]
                        body = "\n".join(seen[i].line for i in idxs)
                        prompt = tmpl + "\n\n<transcript>\n" + body + "\n</transcript>\n\nЗаметки:"
                        units = [(seen[i].end, seen[i].line) for i in idxs]
                        stable_end = len(tmpl) + len("\n\n<transcript>\n") + len(body) + len("\n</transcript>")
                    else:                                         # incremental: {{previous_summary}} + {{new_text}}
                        idxs = list(range(cutoff, len(seen)))
                        prev = "\n".join(u[1] for u in summary_units)
                        new = "\n".join(seen[i].line for i in idxs)
                        prompt = tmpl + "\n\n<previous_summary>\n" + prev + "\n</previous_summary>\n\n<new_text>\n" + new + "\n</new_text>\n\nОбновлённые заметки:"
                        units = list(summary_units) + [(seen[i].end, seen[i].line) for i in idxs]
                        stable_end = len(tmpl)
                    if model.overhead_tokens:
                        pre = "§" * int(model.overhead_tokens * cpt)
                        prompt = pre + prompt; stable_end += len(pre)
                    in_tok = math.ceil(len(prompt) / cpt)
                    kept = stub.summarize(units, pol.summary_tokens)
                    out_tok = min(pol.summary_tokens, max(40, in_tok // 2)) if kept else 0
                    out_tok = min(out_tok, sum(math.ceil(len(u[1]) / cpt) for u in kept) or out_tok)
                    # ---- cache: longest common prefix with a recent prompt
                    read = write = 0
                    if model.cache != "none":
                        recent = [(ts, p) for ts, p in recent if t - ts <= model.cache_ttl_s]
                        best = max((lcp(prompt, p) for _, p in recent), default=0)
                        tok = best / cpt
                        if model.cache == "openai-auto":
                            read = int(tok // 128 * 128) if tok >= 1024 else 0
                        else:  # anthropic: explicit breakpoint at the end of the stable part, min 1024 tokens
                            read = int(min(tok, stable_end / cpt)) if tok >= 1024 else 0
                            bp = stable_end / cpt
                            write = int(max(0, bp - read)) if bp >= 1024 else 0
                    recent.append((t, prompt))
                    cost = _cost(model, in_tok, out_tok, read, write)
                    lat = model.latency_base_s + model.latency_per_1k_in_s * in_tok / 1000 + out_tok / model.out_tokens_per_s
                    inc_end = max(seen[i].end for i in idxs) if idxs else (shown_end or 0.0)
                    if pol.context == "incremental":
                        inc_end = max(inc_end, shown_end or 0.0)
                    c = Call(t, t + lat, in_tok, out_tok, read, write, cost, inc_end, sum(seen[i].words for i in idxs))
                    running = {"end": t + lat, "call": c, "units_after": kept, "idxs": idxs}
                    last_call_start = t; call_starts.append(t); interval_ref = t; want_since = None
                    tokens_used += in_tok + out_tok; cost_used += cost
                    cutoff = len(seen); new_words_cnt = 0; new_turns_cnt = 0
                    if fire_manual: manual.pop(0)
                    if verbose:
                        print(f"[{t:7.1f}s] call #{len(call_starts)} in={in_tok} out={out_tok} cache_read={read} lat={lat:.1f}s new_words={c.new_words}")
        if speed > 0:
            time.sleep(dt / speed)
        t += dt

    # ---- metrics
    if running:                       # a call still open at the horizon (should not happen)
        calls.append(running["call"])
    dur = end_audio
    first = shown_hist[0][0] if shown_hist else None
    # time-weighted age of the summary on screen: now - latest phrase end that the screen summary includes
    age = []
    cur = None; hist = sorted(shown_hist)
    k = 0
    tt = first or 0.0
    while first is not None and tt <= end_audio:
        while k < len(hist) and hist[k][0] <= tt:
            cur = hist[k][1]; k += 1
        age.append(tt - cur if cur is not None else 0.0)
        tt += 1.0
    lags = sorted(lag_per_replica.values())
    # information kept in the last summary: share of "fact words" (the 2 rarest per phrase) present in it
    final_units = summary_units
    facts = _facts(reps, stub)
    covered = _coverage(facts, " ".join(u[1] for u in final_units))
    ideal = _coverage(facts, " ".join(u[1] for u in stub.summarize([(r.end, r.line) for r in reps], pol.summary_tokens)))   # one call over everything, same capacity
    n_in = sum(c.in_tokens for c in calls); n_out = sum(c.out_tokens for c in calls); n_read = sum(c.cached_read for c in calls)
    total_cost = sum(c.cost for c in calls)
    return {
        "calls": len(calls), "skipped_busy": skipped, "defer_wait_steps": deferred_waits, "blocked_by_limits": blocked,
        "in_tokens": n_in, "out_tokens": n_out, "cached_read_tokens": n_read, "cache_write_tokens": sum(c.cache_write for c in calls),
        "cache_hit_share_of_input": round(n_read / n_in, 3) if n_in else 0.0,
        "cost_usd": round(total_cost, 4), "cost_usd_per_hour": round(total_cost / dur * 3600, 3) if dur else 0.0,
        "calls_per_hour": round(len(calls) / dur * 3600, 1) if dur else 0.0,
        "age_s_mean": round(sum(age) / len(age), 1) if age else None, "age_s_p95": round(sorted(age)[int(len(age) * .95) - 1], 1) if age else None,
        "replica_lag_s_mean": round(sum(lags) / len(lags), 1) if lags else None, "replica_lag_s_p95": round(lags[int(len(lags) * .95) - 1], 1) if lags else None,
        "replicas_never_included": len(reps) - len(lag_per_replica),
        "fact_coverage_final": round(covered["all"], 3), "fact_coverage_first_half": round(covered["first_half"], 3),
        "fact_coverage_last_5min": round(covered["last_5min"], 3),
        "fact_coverage_ideal_same_capacity": round(ideal["all"], 3), "coverage_vs_ideal": round(covered["all"] / ideal["all"], 3) if ideal["all"] else None,
        "max_in_tokens": max((c.in_tokens for c in calls), default=0), "audio_s": round(dur, 1),
    }


def _cost(m: Model, in_tok: int, out_tok: int, read: int, write: int) -> float:
    """Same rule as Provider.estimate_cost in src/diarrhizer/summary/providers.py (not imported on purpose)."""
    r = read if m.price_cache_read is not None else 0
    w = write if m.price_cache_write is not None else 0
    plain = max(in_tok - r - w, 0)
    return (plain * m.price_in + r * (m.price_cache_read or 0.0) + w * (m.price_cache_write or 0.0) + out_tok * m.price_out) / 1_000_000


def _facts(reps: list[Replica], stub: StubModel) -> list[tuple[float, str]]:
    out = []
    for r in reps:
        ws = sorted(set(_words(r.text)), key=lambda w: -stub.idf.get(w, 0))[:2]
        out += [(r.end, w) for w in ws]
    return out


def _coverage(facts: list[tuple[float, str]], summary: str) -> dict:
    sw = " ".join(_words(summary))
    stems = {w[:5] for w in sw.split()}
    end = max(t for t, _ in facts)
    def share(sel):
        sel = list(sel)
        return sum(1 for _, w in sel if w[:5] in stems) / len(sel) if sel else 0.0
    return {"all": share(facts), "first_half": share(f for f in facts if f[0] <= end / 2), "last_5min": share(f for f in facts if f[0] >= end - 300)}


# ------------------------------------------------------------------ CLI


def apply_sets(obj, sets: list[str]):
    for s in sets or []:
        k, _, v = s.partition("=")
        if not hasattr(obj, k):
            raise SystemExit(f"unknown setting {k!r}; known: {', '.join(f.name for f in dataclasses.fields(obj))}")
        cur = getattr(obj, k)
        if v.lower() in ("none", "null"):
            val = None
        elif k == "manual_at":
            val = tuple(float(x) for x in v.split(",") if x)
        elif isinstance(cur, bool):
            val = v.lower() in ("1", "true", "yes")
        elif isinstance(cur, int) and not isinstance(cur, bool):
            val = int(float(v))
        elif isinstance(cur, float) or cur is None:
            try:
                val = float(v)
            except ValueError:
                val = v
        else:
            val = v
        setattr(obj, k, val)


def make_model(latency: str, prices: str, cache: str) -> Model:
    return Model(cache=cache, **LATENCY[latency], **PRICES[prices])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("transcript", nargs="?")
    ap.add_argument("--preset", default="balanced", choices=list(PRESETS))
    ap.add_argument("--set", action="append", help="policy field, e.g. --set interval_s=30 --set context=full")
    ap.add_argument("--model-set", action="append", help="stub model field, e.g. --model-set chars_per_token=3")
    ap.add_argument("--latency", default="api-fast", choices=list(LATENCY))
    ap.add_argument("--prices", default="repo-example", choices=list(PRICES))
    ap.add_argument("--cache", default="openai-auto", choices=["openai-auto", "anthropic", "none"])
    ap.add_argument("--asr-latency", type=float, default=4.0)
    ap.add_argument("--pause-stretch", type=float, default=1.0, help="stretch pauses between speech (sensitivity of pause-based settings)")
    ap.add_argument("--speed", type=float, default=0.0, help="0 = as fast as possible; N = sleep dt/N per simulated step")
    ap.add_argument("--list-presets", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()
    if a.list_presets:
        for k, v in PRESETS.items():
            print(k, json.dumps(dataclasses.asdict(v), ensure_ascii=False))
        return
    pol = dataclasses.replace(PRESETS[a.preset]); apply_sets(pol, a.set)
    model = make_model(a.latency, a.prices, a.cache); apply_sets(model, a.model_set)
    reps = load_transcript(a.transcript, a.asr_latency, a.pause_stretch)
    res = run(reps, pol, model, a.asr_latency, speed=a.speed, verbose=a.verbose)
    out = {"policy": dataclasses.asdict(pol), "model": dataclasses.asdict(model), "result": res}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    if a.json:
        json.dump(out, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
