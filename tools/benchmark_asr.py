"""Small ASR baseline runner. Use only recordings you may process; never sends audio to an LLM."""

import argparse
import hashlib
import itertools
import math
import time
import unicodedata
import wave
from pathlib import Path

from diarrhizer.artifacts import SCHEMA_VERSION, provenance
from diarrhizer.audio_formats import WAV_FORMAT
from diarrhizer.pipeline.runner import run_pipeline
from diarrhizer.pipeline.settings import build_stages
from diarrhizer.utils import read_json, write_json_atomic


# [SEMANTIC-BEGIN] QUALITY:METRICS
# @purpose: Compare ASR text and matched-word timing/speakers with explicit synthetic references
# @description: WER/CER use edit counts; timing/WDER use only exact lexical matches and report coverage.
#   Speaker labels are optimally permuted (up to 8); WDER is conditional on matched words, not DER.
#   Silence reports insertions without inventing a zero-denominator WER; overlap has no single reference order.
# @inputs: reference/hypothesis text, word markers, audio duration
# @outputs: edit counts/rates, onset errors and conditional speaker error
# @sideEffects: None
# @errors: ValueError for an empty lexical reference or excessive word/speaker counts
def matched_words(reference, hypothesis):
    """Exact word pairs on a minimum-edit alignment; stable diagonal/deletion/insertion tie order."""
    def tokens(words):
        return [{**w, "word": token} for w in words for token in normalize(w["word"]).split()]
    ref, hyp = tokens(reference), tokens(hypothesis)
    if max(len(ref), len(hyp)) > 2000:
        raise ValueError("Word-marker scoring is limited to 2000 tokens per synthetic case")
    rows = [list(range(len(hyp) + 1))]
    for i, expected in enumerate(ref, 1):
        row = [i]
        for j, actual in enumerate(hyp, 1):
            row.append(min(row[-1] + 1, rows[-1][j] + 1,
                           rows[-1][j - 1] + (expected["word"] != actual["word"])))
        rows.append(row)
    pairs, i, j = [], len(ref), len(hyp)
    while i or j:
        if i and j and rows[i][j] == rows[i-1][j-1] + (ref[i-1]["word"] != hyp[j-1]["word"]):
            if ref[i-1]["word"] == hyp[j-1]["word"]:
                pairs.append((ref[i-1], hyp[j-1]))
            i, j = i - 1, j - 1
        elif i and rows[i][j] == rows[i-1][j] + 1:
            i -= 1
        else:
            j -= 1
    return list(reversed(pairs)), len(ref)


def word_metrics(reference, hypothesis, *, speakers=False):
    pairs, total = matched_words(reference, hypothesis)
    def finite(value):
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    errors = sorted(abs(a["start"] - b["start"]) for a, b in pairs
                    if finite(a.get("start")) and finite(b.get("start")))
    result = {"reference_words": total, "matched_words": len(pairs),
              "lexical_coverage": len(pairs) / total if total else None,
              "timed_matches": len(errors), "timing_coverage": len(errors) / total if total else None,
              "onset_mae_seconds": sum(errors) / len(errors) if errors else None,
              "onset_p95_seconds": errors[math.ceil(.95 * len(errors)) - 1] if errors else None,
              "speaker_error_rate": None, "speaker_mapping": None}
    labelled = [(a.get("speaker_id"), b.get("speaker_id")) for a, b in pairs if a.get("speaker_id")]
    if speakers and labelled:
        refs = sorted({a for a, _ in labelled})
        hyps = sorted({b for _, b in labelled if b})
        if max(len(refs), len(hyps)) > 8:
            raise ValueError("Synthetic speaker scoring supports at most 8 speakers")
        targets = refs + [None] * max(0, len(hyps) - len(refs))
        best, mapping = -1, {}
        for assignment in itertools.permutations(targets, len(hyps)):
            candidate = dict(zip(hyps, assignment))
            correct = sum(candidate.get(b) == a for a, b in labelled)
            if correct > best:
                best, mapping = correct, candidate
        result.update(speaker_error_rate=1 - best / len(labelled), speaker_mapping=mapping,
                      speaker_scored_words=len(labelled))
    return result


def normalize(text):
    """NFKC + casefold, punctuation to spaces; keep Unicode letters and numbers."""
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join("".join(c if c.isalnum() else " " for c in text).split())


def distance(reference, hypothesis):
    """Levenshtein edits, one row of memory; counts substitutions, insertions and deletions."""
    row = list(range(len(hypothesis) + 1))
    for i, expected in enumerate(reference, 1):
        following = [i]
        for j, actual in enumerate(hypothesis, 1):
            following.append(min(following[-1] + 1, row[j] + 1, row[j - 1] + (expected != actual)))
        row = following
    return row[-1]


def score(reference, hypothesis):
    expected, actual = normalize(reference), normalize(hypothesis)
    if not expected:
        raise ValueError("Reference must contain letters or numbers")
    words, chars = expected.split(), expected.replace(" ", "")
    word_edits = distance(words, actual.split())
    char_edits = distance(chars, actual.replace(" ", ""))
    return {"wer": word_edits / len(words), "cer": char_edits / len(chars),
            "word_edits": word_edits, "reference_words": len(words),
            "char_edits": char_edits, "reference_chars": len(chars)}


def missing_times(words):
    def valid(word):
        start, end = word.get("start"), word.get("end")
        return (type(start) in (float, int) and type(end) in (float, int)
                and math.isfinite(start) and math.isfinite(end) and 0 <= start <= end)
    return sum(not valid(word) for word in words) / len(words) if words else None
# [SEMANTIC-END] QUALITY:METRICS


# [SEMANTIC-BEGIN] QUALITY:BENCHMARK
# @purpose: Run a fresh synthetic case and retain comparable metrics plus effective model/load provenance
# @description: Includes conversion, model load, ASR/alignment and export in wall time. No cached job reuse.
#   Optional diarization scores speakers on matched words; overlap text/timing scores are explicitly omitted.
# @inputs: audio, reference text/markers, model/device/language, output directory
# @outputs: report with hashes, edit counts, timings, warnings and partial PyTorch memory measurement
# @sideEffects: Creates pipeline job; loads models; optional diarization uses locally configured HF credentials
# @see: QUALITY:METRICS, ADAPTER:ALIGNMENT_LOAD, PIPELINE:EXECUTE
def run_benchmark(audio, reference, out, *, device="cpu", model="tiny", language="ru",
                  reference_words=None, score_text=True, diarize=False):
    audio = Path(audio)
    if normalize(reference):
        score(reference, "")
    with audio.open("rb") as source:
        audio_sha = hashlib.file_digest(source, "sha256").hexdigest()
    torch = None
    if device == "cuda":
        import torch
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    names = ("convert", "transcribe", "diarize", "merge", "export") if diarize else ("convert", "transcribe", "merge", "export")
    result = run_pipeline(input_path=audio, out_dir=out, stages=build_stages(names),
                          device=device, language=language, asr_model=model, audio_format=WAV_FORMAT,
                          min_speakers=2 if diarize else 1, max_speakers=2 if diarize else 1,
                          asr_compute_type="float16" if device == "cuda" else "int8")
    seconds = time.perf_counter() - started
    job = Path(result["job_dir"])
    with wave.open(str(job / "audio" / "normalized.wav"), "rb") as wav:
        duration = wav.getnframes() / wav.getframerate()
    asr = read_json(job / "asr" / "transcript.json")
    exported = read_json(job / "export" / "result.json")
    hypothesis = " ".join(seg["text"] for seg in asr["segments"])
    final_text = " ".join(seg["text"] for seg in exported["segments"])
    report = {
        "schema_version": SCHEMA_VERSION, "provenance": provenance(),
        "audio_sha256": audio_sha, "reference_sha256": hashlib.sha256(reference.encode()).hexdigest(),
        "job_dir": str(job), "device": device, "model": model, "language": language, "diarize": diarize,
        "normalization": "NFKC/casefold; punctuation to spaces; CER excludes spaces; no number expansion",
        "metrics": score(reference, hypothesis) if score_text and normalize(reference) else None,
        "hypothesis_words": len(normalize(hypothesis).split()),
        "unscored_reason": None if score_text else "overlapping speech has no single text order",
        "silence_insertions": len(normalize(hypothesis).split()) if not normalize(reference) else None,
        "audio_seconds": duration, "wall_seconds": seconds, "rtf": seconds / duration if duration else None,
        "words_without_times_fraction": missing_times(asr.get("words", [])),
        "export_preserves_normalized_text": normalize(hypothesis) == normalize(final_text),
        "word_metrics": word_metrics(reference_words, [w for s in exported["segments"] for w in s.get("words", [])],
                                     speakers=diarize) if reference_words and score_text else None,
        "pytorch_peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch else None,
        "memory_note": "PyTorch allocator only; excludes CTranslate2/CUDA context, not total VRAM or RAM",
        "measurement_note": "Fresh job includes model loading; WDER is conditional on matched words, not DER",
        "asr_metadata": asr.get("metadata", {}),
    }
    return report
# [SEMANTIC-END] QUALITY:BENCHMARK


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--model", default="tiny")
    parser.add_argument("--lang", default="ru")
    args = parser.parse_args(argv)
    reference = args.reference.read_text(encoding="utf-8-sig")
    score(reference, "")  # Validate before inference.
    report = run_benchmark(args.audio, reference, args.out, device=args.device, model=args.model, language=args.lang)
    write_json_atomic(args.report, report)
    print(f"WER={report['metrics']['wer']:.4f}, CER={report['metrics']['cer']:.4f}, RTF={report['rtf']:.3f}")
    print(args.report.resolve())


if __name__ == "__main__":
    main()
