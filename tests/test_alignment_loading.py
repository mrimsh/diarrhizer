"""Real tiny CTC checkpoints: no downloads, random-weight acceptance or warning suppression."""
import logging

import pytest
import torch
from transformers import Wav2Vec2Config, Wav2Vec2ForCTC

from diarrhizer.adapters.alignment import canonicalize_weight_norm, validated_ctc_class


@pytest.fixture
def checkpoint(tmp_path):
    config = Wav2Vec2Config(hidden_size=8, num_hidden_layers=1, num_attention_heads=2, intermediate_size=16,
                           conv_dim=(8, 8, 8), conv_stride=(2, 2, 2), conv_kernel=(3, 3, 3),
                           num_conv_pos_embeddings=4, num_conv_pos_embedding_groups=2, vocab_size=5)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(13)
        original = Wav2Vec2ForCTC(config).eval()
    config.save_pretrained(tmp_path)
    state = original.state_dict()
    legacy = {k.replace("parametrizations.weight.original0", "weight_g").replace(
        "parametrizations.weight.original1", "weight_v"): v.clone() for k, v in state.items()}
    return original, legacy, tmp_path


def test_legacy_weights_are_exact_and_predictions_unchanged_without_false_warnings(checkpoint, caplog):
    original, legacy, folder = checkpoint
    torch.save(legacy, folder / "pytorch_model.bin")
    caplog.set_level(logging.WARNING, logger="transformers.modeling_utils")
    loaded = validated_ctc_class().from_pretrained(folder, local_files_only=True)
    assert all(torch.equal(value, loaded.state_dict()[key]) for key, value in original.state_dict().items())
    assert loaded.alignment_load_audit["renamed_tensors_verified"] == 2
    assert "newly initialized" not in caplog.text and "were not used" not in caplog.text
    with torch.inference_mode():
        signal = torch.linspace(-1, 1, 400).unsqueeze(0)
        torch.testing.assert_close(original(signal).logits, loaded(signal).logits, rtol=0, atol=0)


def test_modern_checkpoint_needs_no_conversion(checkpoint):
    original, _, folder = checkpoint
    original.save_pretrained(folder, safe_serialization=True)
    loaded = validated_ctc_class().from_pretrained(folder, local_files_only=True)
    assert loaded.alignment_load_audit["weight_norm_renamed"] == {}


def test_legacy_sharded_checkpoint_is_explicitly_rejected(checkpoint):
    original, _, _ = checkpoint
    with pytest.raises(ValueError, match="Legacy sharded"):
        validated_ctc_class()._load_pretrained_model(original, None, ["conv.weight_g", "conv.weight_v"])


@pytest.mark.parametrize("damage", ["missing", "unexpected", "shape", "half_pair", "ambiguous"])
def test_incomplete_or_incompatible_checkpoint_is_never_used(checkpoint, damage):
    original, legacy, folder = checkpoint
    g = next(key for key in legacy if key.endswith("weight_g"))
    if damage == "missing":
        del legacy["lm_head.weight"]
    elif damage == "unexpected":
        legacy["unknown.weight"] = torch.zeros(1)
    elif damage == "shape":
        legacy["lm_head.weight"] = torch.zeros(1, 1)
    elif damage == "half_pair":
        del legacy[g]
    else:
        legacy[g.replace("weight_g", "parametrizations.weight.original0")] = legacy[g]
    torch.save(legacy, folder / "pytorch_model.bin")
    with pytest.raises((ValueError, RuntimeError)):
        validated_ctc_class().from_pretrained(folder, local_files_only=True)


def test_conversion_does_not_mutate_checkpoint_and_ignores_unrelated_names(checkpoint):
    original, legacy, _ = checkpoint
    keys = set(legacy)
    converted, renamed = canonicalize_weight_norm(legacy, original.state_dict())
    assert set(legacy) == keys
    assert all(converted[new] is legacy[old] for old, new in renamed.items())
    assert canonicalize_weight_norm({"unrelated.weight_g": 1}, ["other.weight"])[0] == {"unrelated.weight_g": 1}
