"""Score a streaming run: WER/CER against (1) the Diarrhizer pipeline transcript and (2) the true script text, and the latency.

    python eval_stream.py <stream run .json> <pipeline result.json> <material .ref.txt> [--json out.json]

Latency (the main number): every phrase of the reference (= a segment of the pipeline transcript) is looked up in the stream's
output via word alignment (difflib on normalised tokens); latency = wall time at which the stream showed the phrase's last matched
word minus the phrase end in the audio (the pipeline's word alignment gives the end). Wall time starts with the first chunk, so at
speed 1.0 audio time == wall time. Words are compared in lower case without punctuation, ё=е, numbers spelled out (num2words).
"""
import argparse, difflib, json, re, sys
import numpy as np
import jiwer
from num2words import num2words

sys.stdout.reconfigure(encoding="utf-8")


def norm_text(s):
    s = s.lower().replace("ё", "е")
    s = re.sub(r"(\d+)[.,](\d+)", lambda m: f"{num2words(int(m.group(1)), lang='ru')} точка {num2words(int(m.group(2)), lang='ru')}", s)
    s = re.sub(r"(\d+)\s*%", lambda m: f"{m.group(1)} процентов", s)
    s = re.sub(r"\d+", lambda m: num2words(int(m.group(0)), lang="ru"), s)
    s = re.sub(r"[^\w\s]", " ", s).replace("_", " ")
    return s.split()


def pct(v, p):
    return round(float(np.percentile(v, p)), 2) if len(v) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("pipeline"); ap.add_argument("script"); ap.add_argument("--json")
    a = ap.parse_args()
    run = json.load(open(a.run, encoding="utf-8"))
    pipe = json.load(open(a.pipeline, encoding="utf-8"))
    script = open(a.script, encoding="utf-8").read()
    ev = run["events"]

    hyp_tokens, hyp_wall = [], []
    for e in ev:
        toks = norm_text(e["text"])
        hyp_tokens += toks; hyp_wall += [e["wall"]] * len(toks)
    ref_words, ref_end, seg_of = [], [], []
    for si, s in enumerate(pipe["segments"]):
        for w in s.get("words", []):
            t = norm_text(w["word"])
            for tok in t:
                ref_words.append(tok); ref_end.append(w.get("end", s["end"])); seg_of.append(si)
    pipe_text = " ".join(ref_words)
    script_tokens = norm_text(script)
    hyp_text = " ".join(hyp_tokens)

    out = {"run": a.run, "meta": run["meta"], "resources": run["resources"], "total_wall_s": run["total_wall_s"],
           "segments_shown": len(ev), "words_shown": len(hyp_tokens)}
    out["wer_vs_pipeline"] = round(jiwer.wer(pipe_text, hyp_text), 4)
    out["cer_vs_pipeline"] = round(jiwer.cer(pipe_text, hyp_text), 4)
    out["wer_vs_script"] = round(jiwer.wer(" ".join(script_tokens), hyp_text), 4)
    out["cer_vs_script"] = round(jiwer.cer(" ".join(script_tokens), hyp_text), 4)
    out["pipeline_vs_script_wer"] = round(jiwer.wer(" ".join(script_tokens), pipe_text), 4)

    sm = difflib.SequenceMatcher(a=ref_words, b=hyp_tokens, autojunk=False)
    match_of_ref = {}
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            match_of_ref[blk.a + k] = blk.b + k
    word_lat = [hyp_wall[j] - ref_end[i] for i, j in match_of_ref.items()]
    phr_lat = []
    nseg = max(seg_of) + 1 if seg_of else 0
    last_idx = {}
    for i, s in enumerate(seg_of):
        last_idx[s] = i
    for s in range(nseg):
        idxs = [i for i in range(len(seg_of)) if seg_of[i] == s and i in match_of_ref]
        if not idxs:
            continue
        i = idxs[-1]
        phr_lat.append(hyp_wall[match_of_ref[i]] - ref_end[last_idx[s]])
    out["matched_words_pct"] = round(100 * len(match_of_ref) / max(len(ref_words), 1), 1)
    out["phrase_latency_s"] = {"n": len(phr_lat), "mean": round(float(np.mean(phr_lat)), 2) if phr_lat else None, "median": pct(phr_lat, 50),
                               "p95": pct(phr_lat, 95), "max": round(float(np.max(phr_lat)), 2) if phr_lat else None}
    out["word_latency_s"] = {"n": len(word_lat), "mean": round(float(np.mean(word_lat)), 2) if word_lat else None, "p95": pct(word_lat, 95)}
    fed_lag = [e["wall"] - e["fed"] for e in ev]
    out["emit_lag_behind_fed_audio_s"] = {"mean": round(float(np.mean(fed_lag)), 2) if fed_lag else None, "max": round(float(np.max(fed_lag)), 2) if fed_lag else None}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    if a.json:
        json.dump(out, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
