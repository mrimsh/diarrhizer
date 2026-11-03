"""Validated WhisperX alignment loading for the supported Transformers/PyTorch stack."""

import copy


# [SEMANTIC-BEGIN] ADAPTER:ALIGNMENT_LOAD
# @purpose: Load alignment weights without false legacy weight-norm warnings or untrained parameters
# @description: Canonicalize complete legacy g/v pairs before Transformers' key inventory and validate
#   the loaded tensors. The local subclass adapts the pinned 4.39.3 loader; no global monkeypatch,
#   warning filter or checkpoint rewrite. Real missing/unexpected/mismatched weights reject alignment.
#   Legacy sharded checkpoints are refused rather than reporting an unverified conversion.
# @inputs: language, device; WhisperX default model registry and HF/torchaudio cache
# @outputs: model and WhisperX metadata with model/revision/load audit
# @sideEffects: Loads/downloads weights through the existing backends; CPU/GPU allocation
# @errors: ValueError/RuntimeError on incompatible checkpoint; caller preserves ASR on failure
# @see: ADAPTER:WHISPERX_ASR, STAGE:TRANSCRIBE
def canonicalize_weight_norm(state_dict, expected_keys):
    """Return a shallow copy and old→new names, only for complete, unambiguous expected pairs."""
    renamed = {}
    expected = set(expected_keys)
    for key in state_dict:
        if not key.endswith((".weight_g", ".weight_v")):
            continue
        prefix = key.rsplit(".", 1)[0]
        pairs = [(prefix + ".weight_g", prefix + ".parametrizations.weight.original0"),
                 (prefix + ".weight_v", prefix + ".parametrizations.weight.original1")]
        if not all(new in expected for _, new in pairs):
            continue
        if any(old not in state_dict or new in state_dict for old, new in pairs):
            raise ValueError(f"Incomplete or ambiguous weight-norm pair: {prefix}")
        renamed.update(pairs)
    result = copy.copy(state_dict)
    for old, new in renamed.items():
        result[new] = result.pop(old)
    return result, renamed


def validated_ctc_class():
    """Lazily build the local compatibility subclass; importing this module loads no ML libraries."""
    import torch
    from transformers import Wav2Vec2ForCTC

    class ValidatedWav2Vec2ForCTC(Wav2Vec2ForCTC):
        @classmethod
        def _load_pretrained_model(cls, model, state_dict, loaded_keys, *args, **kwargs):
            if state_dict is None and any(k.endswith((".weight_g", ".weight_v")) for k in loaded_keys):
                raise ValueError("Legacy sharded alignment checkpoint cannot be validated; use an unsharded checkpoint")
            renamed = {}
            if state_dict is not None:
                state_dict, renamed = canonicalize_weight_norm(state_dict, model.state_dict())
                loaded_keys = [renamed.get(key, key) for key in loaded_keys]
            result = super()._load_pretrained_model(model, state_dict, loaded_keys, *args, **kwargs)
            # Parent return contract is pinned/tested against Transformers 4.39.3.
            loaded, missing, unexpected, mismatched, _offload, errors = result
            if missing or unexpected or mismatched or errors:
                raise ValueError(f"Alignment checkpoint mismatch: missing={missing}, unexpected={unexpected}, "
                                 f"mismatched={mismatched}, errors={errors}")
            actual = loaded.state_dict()
            for key in renamed.values():
                if not torch.equal(actual[key].cpu(), state_dict[key].cpu()):
                    raise ValueError(f"Alignment weight was not restored exactly: {key}")
            loaded.alignment_load_audit = {"weight_norm_renamed": renamed, "all_keys_matched": True,
                                           "renamed_tensors_verified": len(renamed)}
            return result
    return ValidatedWav2Vec2ForCTC


def load_alignment_model(language_code: str, device: str):
    import whisperx.alignment as backend

    if language_code in backend.DEFAULT_ALIGN_MODELS_TORCH:
        model, metadata = backend.load_align_model(language_code, device)
        metadata["model_info"] = {
            "name": backend.DEFAULT_ALIGN_MODELS_TORCH[language_code], "backend": "torchaudio",
            "revision": None, "loading": {"validation": "torchaudio bundle loader"},
        }
        return model, metadata
    name = backend.DEFAULT_ALIGN_MODELS_HF.get(language_code)
    if name is None:
        raise ValueError(f"No default alignment model for language: {language_code}")
    from transformers import Wav2Vec2Config, Wav2Vec2Processor

    config = Wav2Vec2Config.from_pretrained(name)
    revision = getattr(config, "_commit_hash", None)
    processor = Wav2Vec2Processor.from_pretrained(name, revision=revision)
    model = validated_ctc_class().from_pretrained(name, config=config, revision=revision).eval().to(device)
    return model, {
        "language": language_code, "type": "huggingface",
        "dictionary": {char.lower(): code for char, code in processor.tokenizer.get_vocab().items()},
        "model_info": {"name": name, "backend": "huggingface", "revision": revision,
                       "loading": model.alignment_load_audit},
    }
# [SEMANTIC-END] ADAPTER:ALIGNMENT_LOAD
