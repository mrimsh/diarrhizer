"""Reference annotations and DER scoring for the online-diarization experiments.

Reference = voice activity of each synthetic stem (Silero VAD per speaker stem, gaps < 0.3 s merged), so overlaps are exact.
Hypotheses: Diarrhizer pipeline (offline pyannote), diart (online), and the channel split of step B1 ("me" vs "others").
"""
import json, os, sys
import numpy as np
import soundfile as sf
import torch
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

SR = 16000


def vad_segments(x, thr=0.5, min_silence_ms=300, min_speech_ms=200):
    from silero_vad import load_silero_vad, get_speech_timestamps
    m = load_silero_vad()
    ts = get_speech_timestamps(torch.from_numpy(x), m, sampling_rate=SR, threshold=thr, min_silence_duration_ms=min_silence_ms,
                               min_speech_duration_ms=min_speech_ms, speech_pad_ms=50)
    return [(t["start"] / SR, t["end"] / SR) for t in ts]


def ref_from_stems(material_dir, name, labels=None, merge=None):
    """merge: dict stem letter -> label, e.g. {"A": "me", "B": "others", "C": "others"}"""
    ann = Annotation()
    for f in sorted(os.listdir(material_dir)):
        if f.startswith(name + ".stem_") and f.endswith(".wav"):
            letter = f[len(name) + 6:-4]
            x, _ = sf.read(os.path.join(material_dir, f), dtype="float32")
            for s, e in vad_segments(x):
                ann[Segment(s, e), f"{letter}"] = (merge or {}).get(letter, letter)
    return ann.support(0.0) if False else ann


def hyp_from_pipeline(result_json, merge=None):
    ann = Annotation()
    for s in json.load(open(result_json, encoding="utf-8"))["segments"]:
        for w in s.get("words", [{"start": s["start"], "end": s["end"], "speaker_id": s["speaker_id"]}]):
            if "start" in w and "end" in w:
                ann[Segment(w["start"], w["end"]), w.get("speaker_id", s["speaker_id"])] = w.get("speaker_id", s["speaker_id"])
    return ann.support(0.4)


def der(ref, hyp, collar=0.25, skip_overlap=False):
    m = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
    d = m(ref, hyp, detailed=True)
    tot = d["total"]
    return {"DER": round(d["diarization error rate"], 4), "missed": round(d["missed detection"] / tot, 4),
            "false_alarm": round(d["false alarm"] / tot, 4), "confusion": round(d["confusion"] / tot, 4), "ref_speech_s": round(tot, 1)}


def relabel(ann, mapping):
    out = Annotation()
    for seg, trk, lab in ann.itertracks(yield_label=True):
        out[seg, trk] = mapping.get(lab, lab)
    return out
