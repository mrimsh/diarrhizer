"""Small ASR baseline runner. Use only recordings you may process; never sends audio to an LLM."""

import argparse
import hashlib
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
        return (isinstance(start, (float, int)) and isinstance(end, (float, int))
                and math.isfinite(start) and math.isfinite(end) and 0 <= start <= end)
    return sum(not valid(word) for word in words) / len(words) if words else None


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
    with args.audio.open("rb") as source:
        audio_sha = hashlib.file_digest(source, "sha256").hexdigest()
    torch = None
    if args.device == "cuda":
        import torch
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = run_pipeline(input_path=args.audio, out_dir=args.out,
                          stages=build_stages(("convert", "transcribe", "merge", "export")),
                          device=args.device, language=args.lang, asr_model=args.model, audio_format=WAV_FORMAT,
                          asr_compute_type="float16" if args.device == "cuda" else "int8")
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
        "job_dir": str(job), "device": args.device, "model": args.model, "language": args.lang,
        "normalization": "NFKC/casefold; punctuation to spaces; CER excludes spaces; no number expansion",
        "metrics": score(reference, hypothesis), "audio_seconds": duration,
        "wall_seconds": seconds, "rtf": seconds / duration if duration else None,
        "words_without_times_fraction": missing_times(asr.get("words", [])),
        "export_preserves_normalized_text": normalize(hypothesis) == normalize(final_text),
        "pytorch_peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch else None,
        "memory_note": "PyTorch allocator only; excludes CTranslate2/CUDA context, not total VRAM or RAM",
        "measurement_note": "Fresh ASR-only job including model load, alignment and exports; no DER measurement",
        "asr_metadata": asr.get("metadata", {}),
    }
    write_json_atomic(args.report, report)
    print(f"WER={report['metrics']['wer']:.4f}, CER={report['metrics']['cer']:.4f}, RTF={report['rtf']:.3f}")
    print(args.report.resolve())


if __name__ == "__main__":
    main()
