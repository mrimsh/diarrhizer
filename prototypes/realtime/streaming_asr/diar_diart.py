"""Online diarization with diart on a file (the file is processed as fast as the model allows; real-time factor is reported).

    python diar_diart.py <material.wav 16k> [--seg pyannote/segmentation] [--emb pyannote/embedding] [--device cpu|cuda]
                         [--step 0.5] [--latency 0.5] [--max-speakers 4] [--out hyp.json]

HF token: read from the environment (HF_TOKEN / HUGGINGFACE_HUB_TOKEN) or from the repository's .env - never printed or stored.
diart 0.9.2 is installed with --no-deps next to pyannote.audio 3.3.2; it needs a shim for matplotlib.cm.get_cmap (removed in
matplotlib 3.9) that is applied here, in this script only.
"""
import argparse, json, os, sys, time, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
import matplotlib, matplotlib.cm, matplotlib.pyplot
if not hasattr(matplotlib.cm, "get_cmap"):
    matplotlib.cm.get_cmap = lambda name=None, lut=None: matplotlib.pyplot.get_cmap(name, lut)
HERE = os.path.dirname(os.path.abspath(__file__))
env = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".env"))
if os.path.exists(env) and not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")):
    for ln in open(env, encoding="utf-8"):
        k, _, v = ln.strip().partition("=")
        if k in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN") and v:
            os.environ[k] = v.strip().strip('"').strip("'")
tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or True

ap = argparse.ArgumentParser()
ap.add_argument("material"); ap.add_argument("--seg", default="pyannote/segmentation"); ap.add_argument("--emb", default="pyannote/embedding")
ap.add_argument("--device", default="cpu"); ap.add_argument("--step", type=float, default=0.5); ap.add_argument("--latency", type=float, default=0.5)
ap.add_argument("--max-speakers", type=int, default=4); ap.add_argument("--out")
a = ap.parse_args()

import torch
from diart import SpeakerDiarization, SpeakerDiarizationConfig
from diart.inference import StreamingInference
from diart.models import EmbeddingModel, SegmentationModel
from diart.sources import FileAudioSource

t_load = time.perf_counter()
cfg = SpeakerDiarizationConfig(segmentation=SegmentationModel.from_pretrained(a.seg, use_hf_token=tok),
                               embedding=EmbeddingModel.from_pretrained(a.emb, use_hf_token=tok),
                               step=a.step, latency=a.latency, max_speakers=a.max_speakers, device=torch.device(a.device))
pipe = SpeakerDiarization(cfg)
load_s = time.perf_counter() - t_load
src = FileAudioSource(a.material, cfg.sample_rate, block_duration=a.step)
inf = StreamingInference(pipe, src, do_plot=False, show_progress=False)
t0 = time.perf_counter()
pred = inf()
wall = time.perf_counter() - t0
import soundfile as sf
dur = sf.info(a.material).duration
segs = [(round(s.start, 2), round(s.end, 2), str(l)) for s, _, l in pred.itertracks(yield_label=True)]
res = {"material": os.path.basename(a.material), "seg": a.seg, "emb": a.emb, "device": a.device, "step": a.step, "latency": a.latency,
       "audio_s": round(dur, 1), "compute_s": round(wall, 1), "rtf": round(wall / dur, 3), "model_load_s": round(load_s, 1),
       "speakers_found": len(set(l for *_, l in segs)), "segments": segs}
if a.out:
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
print({k: v for k, v in res.items() if k != "segments"})
