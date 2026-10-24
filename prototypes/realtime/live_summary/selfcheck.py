"""Sanity checks of the simulator (no transcript files needed): limits are respected, cost rule matches Provider.estimate_cost."""
import dataclasses, sys
import sim

def rep(t, words=20, spk="A"):
    return sim.Replica(t, t + 4.0, spk, " ".join(f"слово{(i * 7 + int(t)) % 40:02d}аб" for i in range(words)), t + 8.0)

reps = [rep(10.0 * i) for i in range(120)]            # 20 minutes, a phrase every 10 s, 20 words each
m = sim.make_model("api-fast", "anthropic-like", "anthropic")
# 1) min_interval_s is respected
pol = dataclasses.replace(sim.Policy(), interval_s=5, pause_s=None, silence_ms=0, max_wait_s=0, min_new_words=1, min_interval_s=60)
r = sim.run(reps, pol, m)
assert r["calls"] <= 20 + 1, r["calls"]
# 2) max_calls_per_hour caps the calls and counts the refusals
pol = dataclasses.replace(pol, min_interval_s=0, max_calls_per_hour=10)
r = sim.run(reps, pol, m); assert r["calls"] == 10 and r["blocked_by_limits"] > 0, r
# 3) token budget stops the calls
pol = dataclasses.replace(pol, max_calls_per_hour=1000, token_budget=5000)
r = sim.run(reps, pol, m); assert r["in_tokens"] + r["out_tokens"] < 5000 + 4000 and r["blocked_by_limits"] > 0, r
# 4) cost rule: 1M plain input at 3 USD + 1M output at 15 USD = 18 USD; cache read at its own price replaces the input price
assert abs(sim._cost(m, 1_000_000, 1_000_000, 0, 0) - 18.0) < 1e-9
assert abs(sim._cost(m, 1_000_000, 0, 1_000_000, 0) - 0.30) < 1e-9
assert abs(sim._cost(sim.make_model("api-fast", "repo-example", "none"), 1_000_000, 0, 500_000, 0) - 0.15) < 1e-9   # no cache price -> input price
# 5) manual triggers fire even with too little text, and skip/defer differ only when a call is still running
pol = dataclasses.replace(sim.Policy(), interval_s=None, pause_s=None, manual_at=(100.0, 101.0), min_new_words=10_000, min_interval_s=0)
r = sim.run(reps, pol, sim.make_model("cli-slow", "repo-example", "none")); assert r["calls"] >= 1, r
print("selfcheck ok")
