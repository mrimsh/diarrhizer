"""Run hash-verified synthetic cases sequentially; retain failures and aggregate edit counts."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from benchmark_asr import run_benchmark
from benchmark_resources import summarize_performance
from diarrhizer.env_file import load_project_env
from diarrhizer.utils import write_json_atomic


# [SEMANTIC-BEGIN] QUALITY:SUITE
# @purpose: Compare models/devices on a hash-verified synthetic corpus and preserve failed cases
# @description: Runs sequential fresh jobs; aggregates edit counts, not mean WER; silence/overlap stay unscored.
#   Incrementally writes results; nonzero exit on failed execution or text/alignment integrity failure.
#   Per-model resource summaries keep device scope, measurement coverage and repeated-case timing spread.
# @inputs: corpus manifest, model/device, repeats and optional two-speaker diarization
# @outputs: per-case reports and aggregate WER/CER/RTF
# @sideEffects: Model loading and job/report writes; optional local HF credential loading
# @errors: ValueError on invalid corpus; per-case errors retained in report, exit 1
# @see: QUALITY:BENCHMARK, QUALITY:CORPUS, QUALITY:PERFORMANCE_SUMMARY
def select_cases(manifest_path, only=None):
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported corpus schema")
    known = {case["id"] for case in manifest["cases"]}
    if len(known) != len(manifest["cases"]) or only and set(only) - known:
        raise ValueError("Duplicate or unknown case IDs")
    cases = [dict(case) for case in manifest["cases"] if not only or case["id"] in only]
    if not cases:
        raise ValueError("No selected cases")
    for case in cases:
        audio = (path.parent / case["audio"]).resolve()
        if not audio.is_relative_to(path.parent):
            raise ValueError("Corpus audio must stay within its directory")
        with audio.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != case["sha256"]:
            raise ValueError(f"Audio hash mismatch: {case['id']}; regenerate the corpus")
        case["audio_path"] = audio
    return cases


def aggregate(runs):
    groups = {}
    for run in runs:
        group = groups.setdefault(run["model"], {"completed": 0, "failed": 0, "word_edits": 0,
                                                "reference_words": 0, "char_edits": 0, "reference_chars": 0,
                                                "wall_seconds": 0.0, "audio_seconds": 0.0})
        if run["status"] == "failed":
            group["failed"] += 1
            continue
        group["completed"] += 1
        for key in ("wall_seconds", "audio_seconds"):
            group[key] += run[key]
        for key in ("word_edits", "reference_words", "char_edits", "reference_chars"):
            group[key] += (run["metrics"] or {}).get(key, 0)
    for model, group in groups.items():
        group["wer"] = group["word_edits"] / group["reference_words"] if group["reference_words"] else None
        group["cer"] = group["char_edits"] / group["reference_chars"] if group["reference_chars"] else None
        group["rtf"] = group["wall_seconds"] / group["audio_seconds"] if group["audio_seconds"] else None
        group["performance"] = summarize_performance([r for r in runs if r["model"] == model])
    return groups


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--models", nargs="+", default=["tiny"])
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--sample-interval", type=float, default=.1)
    parser.add_argument("--diarize", action="store_true", help="Two-speaker synthetic cases only; loads local HF credentials")
    args = parser.parse_args(argv)
    if not 1 <= args.repeat <= 20:
        parser.error("--repeat must be between 1 and 20")
    if not math.isfinite(args.sample_interval) or not .02 <= args.sample_interval <= 10:
        parser.error("--sample-interval must be in 0.02..10 seconds")
    cases = select_cases(args.corpus, args.only)
    if args.diarize:
        if any(len({u["speaker_id"] for u in c["utterances"]}) != 2 for c in cases):
            parser.error("--diarize requires --only cases with exactly two reference speakers")
        load_project_env()
    report = {"schema_version": 1, "device": args.device, "diarize": args.diarize,
              "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(), "runs": []}
    for model in args.models:
        for repeat in range(1, args.repeat + 1):
            for case in cases:
                run = {"case": case["id"], "model": model, "repeat": repeat}
                try:
                    run.update(run_benchmark(case["audio_path"], case["reference_text"], args.out,
                                             device=args.device, model=model, language=case["language"],
                                             reference_words=case["reference_words"],
                                             score_text=case.get("score_text", True), diarize=args.diarize,
                                             sample_interval=args.sample_interval))
                    run["status"] = "completed"
                    run["integrity_ok"] = (run["export_preserves_normalized_text"]
                                           and run["asr_metadata"].get("alignment_status") != "failed")
                except Exception as exc:
                    run.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                               performance=getattr(exc, "benchmark_performance", None))
                report["runs"].append(run)
                report["aggregate"] = aggregate(report["runs"])
                write_json_atomic(args.report, report)
                print(f"{case['id']} / {model} / {repeat}: {run['status']}", flush=True)
    return int(any(r["status"] == "failed" or not r.get("integrity_ok", False) for r in report["runs"]))
# [SEMANTIC-END] QUALITY:SUITE


if __name__ == "__main__":
    raise SystemExit(main())
