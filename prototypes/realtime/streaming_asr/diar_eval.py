"""Who-speaks-when experiments on the synthetic materials: offline pipeline vs online diart vs the channel split of step B1.

    python diar_eval.py <material> <pipeline result.json> [--diart hyp.json] [--diart-others hyp.json]

Reference: VAD of every speaker stem.  "Channel split" = left channel carries speaker A ("me"), right channel carries everyone else
("others"); its hypothesis is the VAD of each channel, which is what a per-channel diarization-free setup would give.
--diart-others: a diart run on the "others" channel only (how well several voices inside one track are told apart).
"""
import argparse, json, os, sys
import numpy as np
import soundfile as sf
from pyannote.core import Annotation, Segment
from diar_tools import SR, der, hyp_from_pipeline, ref_from_stems, relabel, vad_segments

sys.stdout.reconfigure(encoding="utf-8")
M = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_local", "materials"))


def ann_from_diart(path, offset=0.0):
    j = json.load(open(path, encoding="utf-8"))
    ann = Annotation()
    for s, e, l in j["segments"]:
        ann[Segment(s + offset, e + offset), l] = l
    return ann, j


def best_mapping_der(ref, hyp, **kw):
    """DER with the optimal one-to-one label mapping (pyannote does it inside DiarizationErrorRate)."""
    return der(ref, hyp, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("material"); ap.add_argument("pipeline"); ap.add_argument("--diart"); ap.add_argument("--diart-others")
    a = ap.parse_args()
    ref = ref_from_stems(M, a.material)
    out = {"material": a.material, "ref_speakers": sorted({l for _, _, l in ref.itertracks(yield_label=True)})}
    out["offline_pipeline"] = {"with_overlap": der(ref, hyp_from_pipeline(a.pipeline)), "skip_overlap": der(ref, hyp_from_pipeline(a.pipeline), skip_overlap=True)}
    if a.diart:
        h, j = ann_from_diart(a.diart)
        out["diart_online"] = {"with_overlap": der(ref, h), "skip_overlap": der(ref, h, skip_overlap=True),
                               "rtf": j["rtf"], "device": j["device"], "step": j["step"], "latency": j["latency"], "speakers_found": j["speakers_found"]}
    # channel split: A on the left, everyone else summed on the right
    stems = {l: sf.read(os.path.join(M, f"{a.material}.stem_{l}.wav"), dtype="float32")[0] for l in out["ref_speakers"]}
    n = max(len(v) for v in stems.values())
    left = np.pad(stems["A"], (0, n - len(stems["A"]))) if "A" in stems else np.zeros(n, "float32")
    right = sum(np.pad(v, (0, n - len(v))) for k, v in stems.items() if k != "A") if len(stems) > 1 else np.zeros(n, "float32")
    merged_ref = relabel(ref, {k: ("me" if k == "A" else "others") for k in out["ref_speakers"]})
    hyp = Annotation()
    for lab, x in (("me", left), ("others", right)):
        for s, e in vad_segments(x.astype("float32")):
            hyp[Segment(s, e), f"{lab}"] = lab
    out["channel_split_me_vs_others"] = {"with_overlap": der(merged_ref, hyp), "skip_overlap": der(merged_ref, hyp, skip_overlap=True)}
    if a.diart_others:
        oth = Annotation()
        for seg, trk, lab in ref.itertracks(yield_label=True):
            if lab != "A":
                oth[seg, trk] = lab
        h, j = ann_from_diart(a.diart_others)
        out["diart_on_others_channel"] = {"speakers_in_ref": sorted({l for _, _, l in oth.itertracks(yield_label=True)}),
                                          "with_overlap": der(oth, h), "speakers_found": j["speakers_found"], "rtf": j["rtf"]}
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
