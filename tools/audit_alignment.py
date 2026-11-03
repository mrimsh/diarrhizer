"""Compare cached original/canonical HF loading and packaged VAD; set HF_HUB_OFFLINE=1 for offline use."""
import argparse
import gc
import hashlib
from pathlib import Path

import torch
from transformers import Wav2Vec2ForCTC

from diarrhizer.adapters.alignment import load_alignment_model
from diarrhizer.artifacts import provenance
from diarrhizer.utils import write_json_atomic


# [SEMANTIC-BEGIN] QUALITY:CHECKPOINT_AUDIT
# @purpose: Distinguish false key-name warnings from actual missing alignment/VAD weights
# @description: Compares norm tensors and deterministic logits; verifies packaged VAD hash and every tensor.
#   Original-loader and VAD version warnings remain visible; no cache rewrite or filtering.
# @inputs: cached default RU alignment and the WhisperX packaged VAD checkpoint
# @outputs: JSON evidence with revisions, package versions, equality checks and training metadata
# @sideEffects: Model loads, CPU/GPU allocation, report write; disables TF32 for this process
# @errors: Exits nonzero if hash, weights or predictions differ
# @see: ADAPTER:ALIGNMENT_LOAD, QUALITY:BENCHMARK
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    import whisperx.alignment as alignment
    import whisperx.vad as vad
    from pyannote.audio import Model

    name = alignment.DEFAULT_ALIGN_MODELS_HF["ru"]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    original, info = Wav2Vec2ForCTC.from_pretrained(name, output_loading_info=True, local_files_only=True)
    original.eval().to(args.device)
    norm = {k: v.detach().cpu().clone() for k, v in original.state_dict().items() if "parametrizations.weight" in k}
    signal = torch.linspace(-.1, .1, 16000, device=args.device).unsqueeze(0)
    with torch.inference_mode():
        original_logits = original(signal).logits.cpu()
    del original
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()
    corrected, metadata = load_alignment_model("ru", args.device)
    with torch.inference_mode():
        corrected_logits = corrected(signal).logits.cpu()
    norm_equal = all(torch.equal(v, corrected.state_dict()[k].cpu()) for k, v in norm.items())
    delta = (original_logits - corrected_logits).abs().max().item()
    del corrected
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()

    checkpoint = Path(vad.__file__).parent / "assets" / "pytorch_model.bin"
    with checkpoint.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    if sha != vad.VAD_SEGMENTATION_URL.split("/")[-2]:
        raise ValueError("Unexpected packaged VAD checkpoint")
    # Trusted packaged checkpoint, hash verified against WhisperX's own constant; includes Lightning objects.
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = Model.from_pretrained(checkpoint, strict=True)
    loaded = model.state_dict()
    unequal = [k for k, v in state["state_dict"].items() if k not in loaded or not torch.equal(v, loaded[k])]
    unexpected = sorted(set(loaded) - set(state["state_dict"]))
    report = {"provenance": provenance(), "device": args.device, "alignment_model": metadata["model_info"],
              "original_loading_info": info, "weight_norm_equal_before_after": norm_equal,
              "max_logit_delta": delta, "probe": "one-second deterministic ramp; not an accuracy benchmark",
              "vad": {"sha256": sha, "strict_load": True, "unequal_keys": unequal,
                      "unexpected_keys": unexpected, "tensor_count": len(loaded),
                      "training_versions": {k: str(v) for k, v in state.get("pyannote.audio", {}).get("versions", {}).items()},
                      "lightning_checkpoint_version": state.get("pytorch-lightning_version")}}
    write_json_atomic(args.report, report)
    print(f"norm_equal={norm_equal}, max_logit_delta={delta}, VAD unequal={len(unequal)}, unexpected={len(unexpected)}")
    if not norm_equal or delta != 0 or unequal or unexpected:
        raise SystemExit(1)
# [SEMANTIC-END] QUALITY:CHECKPOINT_AUDIT


if __name__ == "__main__":
    main()
