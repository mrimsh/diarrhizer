"""Run duration-based DER on a hash-verified RTTM/UEM corpus, without ASR or oracle speaker counts."""
import argparse
import json
import math
import time
import wave
from pathlib import Path

from diarization_metrics import aggregate_der, read_hypothesis, read_rttm, read_uem, score_der
from prepare_der_corpus import digest
from diarrhizer.artifacts import provenance
from diarrhizer.audio_formats import WAV_FORMAT
from diarrhizer.env_file import load_project_env
from diarrhizer.pipeline.runner import run_pipeline
from diarrhizer.pipeline.settings import build_stages
from diarrhizer.utils import write_json_atomic


# [SEMANTIC-BEGIN] QUALITY:DER_SUITE
# @purpose: Evaluate raw diarization on independently annotated human speech, including overlap and silence
# @description: Validates all selected audio/reference/UEM hashes before inference. Runs convert+diarize
#   with explicit speaker bounds (default 1..10); no reference count leakage, ASR dependency or job-cache reuse.
#   Preserves per-case failures and aggregates speaker-second components with one scoring policy.
# @inputs: speaker_diarization corpus manifest, device, speaker bounds, collar, skip_overlap, repeats
# @outputs: per-recording raw hypotheses/DER reports, weighted corpus DER and explicit failure counts
# @sideEffects: Loads local HF credentials/models; creates jobs and incremental atomic report
# @errors: ValueError on invalid corpus/policy; exit 1 on any failed case
# @see: QUALITY:DER, QUALITY:DER_CORPUS, STAGE:DIARIZE
def load_cases(path, only=None):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1
            or manifest.get("task") != "speaker_diarization" or not isinstance(manifest.get("cases"), list)):
        raise ValueError("Expected speaker_diarization corpus schema 1 with explicit RTTM/UEM")
    ids = [case["id"] for case in manifest["cases"]]
    if (not ids or any(not isinstance(uri, str) or not uri.strip() for uri in ids)
            or len(set(ids)) != len(ids) or only and set(only) - set(ids)):
        raise ValueError("Empty, duplicate or unknown recording IDs")
    cases = []
    for source in manifest["cases"]:
        if only and source["id"] not in only:
            continue
        case = dict(source)
        for key in ("audio", "reference", "uem"):
            target = (path.parent / case[key]).resolve()
            if not target.is_relative_to(path.parent):
                raise ValueError(f"Corpus {key} path escapes its directory")
            if digest(target) != case[key + "_sha256"]:
                raise ValueError(f"Corpus {key} hash mismatch: {case['id']}")
            case[key + "_path"] = target
        duration = case["duration_seconds"]
        if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
            raise ValueError("Audio duration must be finite and positive")
        with wave.open(str(case["audio_path"])) as stream:
            if abs(stream.getnframes() / stream.getframerate() - duration) > 1 / stream.getframerate():
                raise ValueError("Manifest duration does not match WAV")
        case["reference_annotation"] = read_rttm(case["reference_path"], case["id"])
        case["uem_timeline"] = read_uem(case["uem_path"], case["id"])
        if (case["uem_timeline"].extent().end > duration + 1e-7
                or case["reference_annotation"].get_timeline().extent().end > duration + 1e-7):
            raise ValueError("Reference/UEM extends beyond audio duration")
        cases.append(case)
    return cases


def run_case(case, output, *, device="cpu", min_speakers=1, max_speakers=10, collar=0.0, skip_overlap=False):
    started = time.perf_counter()
    result = run_pipeline(input_path=case["audio_path"], out_dir=output,
                          stages=build_stages(("convert", "diarize")), device=device,
                          min_speakers=min_speakers, max_speakers=max_speakers,
                          audio_format=WAV_FORMAT, audio_profile="raw")
    seconds = time.perf_counter() - started
    hypothesis_path = Path(result["job_dir"]) / "diar" / "diarization.json"
    score = score_der(case["reference_annotation"], read_hypothesis(hypothesis_path, case["id"]),
                      case["uem_timeline"], collar=collar, skip_overlap=skip_overlap)
    return {"der": score, "job_dir": result["job_dir"], "hypothesis_sha256": digest(hypothesis_path),
            "wall_seconds": seconds, "audio_seconds": case["duration_seconds"],
            "rtf": seconds / case["duration_seconds"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--min-speakers", type=int, default=1)
    parser.add_argument("--max-speakers", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--collar", type=float, default=0.0, help="Total width centered on reference boundaries, in seconds")
    parser.add_argument("--skip-overlap", action="store_true", help="Diagnostic comparison only; default includes overlap")
    args = parser.parse_args(argv)
    if not (1 <= args.min_speakers <= args.max_speakers and 1 <= args.repeat <= 20):
        parser.error("Require 1 <= min-speakers <= max-speakers and repeat in 1..20")
    if not math.isfinite(args.collar) or args.collar < 0:
        parser.error("--collar must be finite and nonnegative")
    cases = load_cases(args.corpus, args.only)
    load_project_env()
    report = {"schema_version": 1, "task": "speaker_diarization", "provenance": provenance(),
              "corpus_sha256": digest(args.corpus), "device": args.device,
              "settings": {"min_speakers": args.min_speakers, "max_speakers": args.max_speakers,
                           "audio_profile": "raw", "collar_seconds": args.collar, "skip_overlap": args.skip_overlap},
              "measurement_note": "Fresh convert+diarize job; includes model load; reference speaker count is not supplied",
              "runs": []}
    for repeat in range(1, args.repeat + 1):
        for case in cases:
            run = {"case": case["id"], "repeat": repeat,
                   **{key + "_sha256": case[key + "_sha256"] for key in ("audio", "reference", "uem")}}
            try:
                run.update(run_case(case, args.out, device=args.device, min_speakers=args.min_speakers,
                                    max_speakers=args.max_speakers, collar=args.collar, skip_overlap=args.skip_overlap))
                run["status"] = "completed"
            except Exception as exc:
                run.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            report["runs"].append(run)
            completed = [r for r in report["runs"] if r["status"] == "completed"]
            report["aggregate"] = aggregate_der([r["der"] for r in completed])
            report["coverage"] = {"planned": len(cases) * args.repeat, "completed": len(completed),
                                  "failed": len(report["runs"]) - len(completed)}
            write_json_atomic(args.report, report)
            print(f"{case['id']} / {repeat}: {run['status']}", flush=True)
    return int(any(run["status"] == "failed" for run in report["runs"]))
# [SEMANTIC-END] QUALITY:DER_SUITE


if __name__ == "__main__":
    raise SystemExit(main())
