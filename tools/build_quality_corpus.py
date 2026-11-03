"""Compose deterministic quality fixtures from the local System.Speech units (no recordings)."""
import argparse
import hashlib
import json
import platform
import wave
from pathlib import Path

import numpy as np

RATE = 16000
SPEC = Path(__file__).parent / "fixtures" / "corpus-spec.json"


# [SEMANTIC-BEGIN] QUALITY:CORPUS
# @purpose: Build synthetic RU/EN cases with reproducible transforms and independent word-onset references
# @description: Retains SAPI markers and voice/source/spec hashes; seeded noise uses whole-wave RMS.
#   Speaker spans are utterance envelopes, not speech-activity annotations suitable for DER.
# @inputs: checked-in corpus spec and locally synthesized PCM/marker units
# @outputs: ten WAV cases and corpus.json with text, onsets, speaker labels and hashes
# @sideEffects: Reads units; writes fixtures under the chosen output directory
# @errors: ValueError on wrong PCM format, voice, text or missing markers
# @see: QUALITY:METRICS, QUALITY:SUITE
def read_wave(path):
    with wave.open(str(path), "rb") as source:
        if (source.getnchannels(), source.getsampwidth(), source.getframerate()) != (1, 2, RATE):
            raise ValueError(f"Expected mono PCM16 at {RATE} Hz: {path}")
        return np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(np.float64) / 32768


def compose(case, units):
    pieces, words, spans, texts = [], [], [], []
    cursor = 0.5
    for part in case["parts"]:
        audio, metadata = units[part["unit"]]
        start = float(part.get("at", cursor + part.get("gap_before", 0)))
        offset = round(start * RATE)
        start = offset / RATE
        end = start + len(audio) / RATE
        speaker = part.get("speaker", "A")
        pieces.append((offset, audio))
        words.extend({"word": w["word"], "start": start + w["start"], "speaker_id": speaker}
                     for w in metadata["words"])
        spans.append({"start": start, "end": end, "speaker_id": speaker, "voice": metadata["voice"],
                      "text": metadata["text"]})
        texts.append(metadata["text"])
        cursor = max(cursor, end)
    length = round(float(case.get("duration", cursor + .5)) * RATE)
    signal = np.zeros(length, dtype=np.float64)
    for offset, audio in pieces:
        signal[offset:offset + len(audio)] += audio
    if "snr_db" in case:
        noise = np.random.default_rng(case["seed"]).standard_normal(length)
        noise *= np.sqrt(np.mean(signal ** 2) / np.mean(noise ** 2)) / (10 ** (case["snr_db"] / 20))
        signal += noise
    signal *= 10 ** (case.get("gain_db", 0) / 20)
    signal *= min(1.0, .98 / max(np.max(np.abs(signal)), 1e-12))
    if case.get("stereo"):
        signal = np.column_stack((signal, signal * .5))
    pcm = np.rint(signal * 32767).astype("<i2")
    return pcm, {"reference_text": " ".join(texts), "reference_words": words, "utterances": spans}


def build(root, spec_path=SPEC):
    root = Path(root)
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    units = {}
    source_hashes = {}
    for unit in spec["units"]:
        path = root / "units" / (unit["id"] + ".wav")
        metadata_path = path.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        if metadata["text"] != unit["text"] or metadata["voice"] != unit["voice"] or not metadata["words"]:
            raise ValueError(f"Unit differs from spec or has no TTS word markers: {unit['id']}")
        units[unit["id"]] = read_wave(path), metadata
        source_hashes[unit["id"]] = {"audio": hashlib.sha256(path.read_bytes()).hexdigest(),
                                    "markers": hashlib.sha256(metadata_path.read_bytes()).hexdigest()}
    cases = []
    for case in spec["cases"]:
        pcm, reference = compose(case, units)
        path = root / (case["id"] + ".wav")
        with wave.open(str(path), "wb") as output:
            output.setnchannels(2 if pcm.ndim == 2 else 1)
            output.setsampwidth(2)
            output.setframerate(RATE)
            output.writeframes(pcm.tobytes())
        cases.append({**case, **reference, "audio": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "duration": len(pcm) / RATE})
    manifest = {"schema_version": 1, "generator": "System.Speech + deterministic PCM composition",
                "platform": platform.platform(), "numpy_version": np.__version__, "source_hashes": source_hashes,
                "spec_sha256": hashlib.sha256(Path(spec_path).read_bytes()).hexdigest(), "cases": cases,
                "reference_note": "SAPI word-onset markers; utterance spans include silence, not DER annotations"}
    (root / "corpus.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
# [SEMANTIC-END] QUALITY:CORPUS


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = build(args.root)
    print(f"Generated {len(result['cases'])} cases: {args.root / 'corpus.json'}")
