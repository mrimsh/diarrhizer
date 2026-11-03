"""Overlap-aware DER on explicit RTTM/UEM references, independent of ASR and merge output."""
import argparse
import hashlib
import json
import math
from importlib.metadata import version
from pathlib import Path

from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.metrics.identification import IdentificationErrorRate

from diarrhizer.utils import write_json_atomic


# [SEMANTIC-BEGIN] QUALITY:DIARIZATION_REFERENCE
# @purpose: Read explicit speaker activity and evaluated regions without guessing silence or dropping overlap
# @description: One recording/channel per input; finite positive intervals only. Same-speaker duplicates
#   are unioned, distinct simultaneous speakers remain separate. UEM is mandatory and nonempty.
# @inputs: RTTM, raw diar/diarization.json or interval dictionaries; UEM; recording ID
# @outputs: canonical pyannote Annotation and Timeline
# @sideEffects: Reads local reference/hypothesis files
# @errors: ValueError on malformed, non-finite, wrong-ID/channel or empty-UEM inputs
# @see: QUALITY:DER, QUALITY:DER_SUITE
def interval(start, end):
    if (type(start) not in (int, float) or type(end) not in (int, float)
            or not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end):
        raise ValueError(f"Invalid interval: {start!r}, {end!r}")
    return Segment(start, end)


def annotation(segments, uri):
    result = Annotation(uri=uri)
    for track, item in enumerate(segments):
        label = item.get("speaker")
        if not isinstance(label, str) or not label.strip():
            raise ValueError("Each interval needs a nonempty speaker label")
        result[interval(item["start"], item["end"]), track] = label
    return result.support()  # Union duplicate/overlapping intervals of the same speaker only.


def read_rttm(path, uri):
    segments = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if (len(fields) != 10 or fields[0] != "SPEAKER" or fields[1] != uri or fields[2] != "1"
                or fields[7] == "<NA>"):
            raise ValueError(f"{path}:{number}: expected SPEAKER for {uri!r}, channel 1, ten fields and a speaker label")
        start, duration = float(fields[3]), float(fields[4])
        segments.append({"start": start, "end": start + duration, "speaker": fields[7]})
    return annotation(segments, uri)


def read_uem(path, uri):
    regions = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 4 or fields[0] != uri or fields[1] != "1":
            raise ValueError(f"{path}:{number}: expected UEM for {uri!r}, channel 1, four fields")
        regions.append(interval(float(fields[2]), float(fields[3])))
    if not regions:
        raise ValueError("UEM must explicitly identify nonempty evaluated regions")
    return Timeline(regions, uri=uri).support()


def read_hypothesis(path, uri):
    path = Path(path)
    if path.suffix.lower() == ".rttm":
        return read_rttm(path, uri)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("stage") != "diarize":
        raise ValueError("Expected raw diar/diarization.json (stage=diarize), not merged/exported words")
    return annotation(data["segments"], uri)
# [SEMANTIC-END] QUALITY:DIARIZATION_REFERENCE


COMPONENTS = {"false_alarm_seconds": "false alarm", "missed_speech_seconds": "missed detection",
              "confusion_seconds": "confusion", "reference_speaker_seconds": "total", "correct_seconds": "correct"}


def rates(components):
    total = components["reference_speaker_seconds"]
    errors = sum(components[k] for k in ("false_alarm_seconds", "missed_speech_seconds", "confusion_seconds"))
    return {**components, "der": errors / total if total else None}


# [SEMANTIC-BEGIN] QUALITY:DER
# @purpose: Measure full duration-based DER, including simultaneous reference speakers and non-speech
# @description: Uses the pinned pyannote.metrics Hungarian mapping after UEM/collar exclusions.
#   The same recording-wide mapping scores overlap and non-overlap; never remap each subset optimistically.
#   Collar is total boundary width (half on each side); default zero, overlap included. Zero denominator is null.
# @inputs: canonical reference/hypothesis annotations, explicit UEM, collar and skip_overlap
# @outputs: DER components in speaker-seconds, mapping, scored durations and additive overlap breakdown
# @sideEffects: None
# @errors: ValueError on invalid scoring policy or empty UEM
# @see: QUALITY:DIARIZATION_REFERENCE, QUALITY:DER_SUITE
def score_der(reference, hypothesis, uem, *, collar=0.0, skip_overlap=False):
    if type(collar) not in (int, float) or not math.isfinite(collar) or collar < 0:
        raise ValueError("Collar must be a finite nonnegative number of seconds")
    if type(skip_overlap) is not bool:
        raise ValueError("skip_overlap must be boolean")
    if uem is None or not uem:
        raise ValueError("DER requires an explicit nonempty UEM")
    reference, hypothesis, uem = reference.support(), hypothesis.support(), uem.support()
    metric = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
    raw_overlap = reference.crop(uem).get_overlap().duration()
    ref, hyp, evaluated = metric.uemify(reference, hypothesis, uem=uem, collar=collar,
                                       skip_overlap=skip_overlap, returns_uem=True)
    # Disjoint label types prevent an unmapped hypothesis label accidentally matching a reference label.
    ref_names = {label: f"reference_{i}" for i, label in enumerate(ref.labels())}
    hyp_names = {label: i for i, label in enumerate(hyp.labels())}
    ref = ref.rename_labels(mapping=ref_names)
    hyp = hyp.rename_labels(mapping=hyp_names)
    mapping = metric.optimal_mapping(ref, hyp)
    mapped = hyp.rename_labels(mapping=mapping)
    identification = IdentificationErrorRate(collar=0.0, skip_overlap=False)

    def components(regions):
        details = identification.compute_components(ref, mapped, uem=regions)
        return rates({**{key: float(details[value]) for key, value in COMPONENTS.items()},
                      "evaluated_seconds": regions.duration()})

    overlap = ref.get_overlap()
    non_overlap = overlap.gaps(support=evaluated)
    inverse_ref = {value: key for key, value in ref_names.items()}
    return {"policy": {"collar_seconds": float(collar), "skip_overlap": skip_overlap,
                       "collar_convention": "total width centered on each reference boundary",
                       "mapping": "recording-wide Hungarian after exclusions", "scorer": "pyannote.metrics",
                       "scorer_version": version("pyannote.metrics"), "core_version": version("pyannote.core")},
            **components(evaluated), "overlap": components(overlap), "non_overlap": components(non_overlap),
            "reference_overlap_seconds_before_exclusions": raw_overlap,
            "speaker_mapping": {name: inverse_ref[mapping[key]] for name, key in hyp_names.items() if key in mapping},
            "unmapped_hypothesis_speakers": [name for name, key in hyp_names.items() if key not in mapping]}


def aggregate_der(scores):
    if not scores:
        return None
    if any(score["policy"] != scores[0]["policy"] for score in scores):
        raise ValueError("Cannot aggregate different DER scoring policies/versions")
    keys = (*COMPONENTS, "evaluated_seconds")
    def summed(items):
        return rates({key: sum(item[key] for item in items) for key in keys})
    return {"policy": scores[0]["policy"], "recordings": len(scores), **summed(scores),
            "overlap": summed([score["overlap"] for score in scores]),
            "non_overlap": summed([score["non_overlap"] for score in scores]),
            "reference_overlap_seconds_before_exclusions": sum(s["reference_overlap_seconds_before_exclusions"] for s in scores)}
# [SEMANTIC-END] QUALITY:DER


# [SEMANTIC-BEGIN] QUALITY:DER_CLI
# @purpose: Re-score saved raw hypotheses without loading a speech model
# @inputs: single-recording reference RTTM, hypothesis RTTM/JSON, UEM, scoring policy
# @outputs: JSON DER report with input hashes and scorer versions
# @sideEffects: Reads inputs and writes report
# @errors: ValueError for invalid reference/hypothesis/policy
# @see: QUALITY:DER, QUALITY:DIARIZATION_REFERENCE
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--hypothesis", type=Path, required=True)
    parser.add_argument("--uem", type=Path, required=True)
    parser.add_argument("--uri", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--collar", type=float, default=0.0)
    parser.add_argument("--skip-overlap", action="store_true")
    args = parser.parse_args(argv)
    report = score_der(read_rttm(args.reference, args.uri), read_hypothesis(args.hypothesis, args.uri),
                       read_uem(args.uem, args.uri), collar=args.collar, skip_overlap=args.skip_overlap)
    report["inputs"] = {"uri": args.uri}
    for key in ("reference", "hypothesis", "uem"):
        with getattr(args, key).open("rb") as stream:
            report["inputs"][key + "_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    write_json_atomic(args.report, report)
    print(json.dumps({key: report[key] for key in ("der", "false_alarm_seconds", "missed_speech_seconds", "confusion_seconds")}))
# [SEMANTIC-END] QUALITY:DER_CLI


if __name__ == "__main__":
    main()
