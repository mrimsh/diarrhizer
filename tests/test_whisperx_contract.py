"""Backend-boundary regressions using WhisperX 3.3.1's real call signatures."""

import sys
from types import SimpleNamespace

import pytest

from diarrhizer.adapters.whisperx import WhisperXAdapter


@pytest.fixture
def backend(monkeypatch, tmp_path):
    calls = []
    transcriptions = []

    class Model:
        def transcribe(self, audio, batch_size=None, num_workers=0, language=None,
                       task=None, chunk_size=30, print_progress=False, combined_progress=False):
            transcriptions.append(language)
            return {"language": language or "en", "segments": [{"start": 0, "end": 1, "text": "hello"}]}

    def load_model(whisper_arch, device, device_index=0, compute_type="float16",
                   asr_options=None, language=None, vad_model=None, vad_options=None,
                   model=None, task="transcribe", download_root=None, local_files_only=False, threads=4):
        calls.append({"model": whisper_arch, "device": device, "compute_type": compute_type,
                      "language": language, "asr_options": asr_options})
        return Model()

    module = SimpleNamespace(load_model=load_model, load_audio=lambda _: "synthetic waveform",
                             load_align_model=lambda *args: (object(), {}),
                             align=lambda segments, *args, **kwargs: {"segments": segments})
    monkeypatch.setitem(sys.modules, "whisperx", module)
    audio = tmp_path / "synthetic.wav"
    audio.write_bytes(b"mock decoder input")
    return module, calls, transcriptions, audio


def test_decoding_options_go_to_load_model_and_overrides_reload_consistently(backend):
    _, calls, transcriptions, audio = backend
    adapter = WhisperXAdapter(device="cpu", language="ru", beam_size=9, initial_prompt="Names 2026")
    first = adapter.transcribe(audio)
    adapter.transcribe(audio)
    assert len(calls) == 1
    assert calls[0]["asr_options"] == {"beam_size": 9, "initial_prompt": "Names 2026"}
    assert calls[0]["language"] == "ru" and calls[0]["compute_type"] == "int8"
    assert first["effective_settings"]["beam_size"] == 9
    adapter.transcribe(audio, compute_type="int8_float16", beam_size=2, initial_prompt="")
    assert calls[-1]["asr_options"] == {"beam_size": 2, "initial_prompt": ""}
    assert calls[-1]["compute_type"] == "int8_float16"
    adapter.transcribe(audio)
    assert len(calls) == 3 and calls[-1] == calls[0]
    assert transcriptions == ["ru"] * 4
    adapter.unload()


def test_unsupported_controls_are_disclosed_as_effective_values(backend, caplog):
    _, _, _, audio = backend
    result = WhisperXAdapter(device="cpu", temperature=0.7, condition_on_previous_text=True,
                             vad_filter=False, vad_min_silence_ms=300).transcribe(audio)
    effective = result["effective_settings"]
    assert effective["temperature"] == 0.0
    assert effective["condition_on_previous_text"] is False
    assert effective["vad_filter"] is True and effective["vad_min_silence_ms"] is None
    assert all(any(name in warning for warning in result["warnings"])
               for name in ("temperature", "previous text", "vad_filter", "vad_min_silence_ms"))
    assert "unsupported" in caplog.text


def test_alignment_failure_preserves_raw_text_and_reports_quality(backend):
    module, _, _, audio = backend
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic alignment failure")
    module.align = fail
    result = WhisperXAdapter(device="cpu").transcribe(audio)
    assert result["text"] == "hello"
    assert result["alignment_status"] == "failed"
    assert any("alignment failure" in message for message in result["warnings"])


def test_backend_type_error_is_not_retried_with_options_silently_removed(backend):
    module, calls, _, audio = backend
    def incompatible(*args, **kwargs):
        calls.append(kwargs)
        raise TypeError("unexpected keyword argument 'asr_options'")
    module.load_model = incompatible
    with pytest.raises(RuntimeError, match="asr_options"):
        WhisperXAdapter(device="cpu").transcribe(audio)
    assert len(calls) == 1


@pytest.mark.parametrize("segments", [[], [{"start": 0, "end": 1, "text": ""}]])
def test_alignment_cannot_erase_asr_segments(backend, segments):
    module, _, _, audio = backend
    module.align = lambda *args, **kwargs: {"segments": segments}
    result = WhisperXAdapter(device="cpu").transcribe(audio)
    assert result["segments"][0]["text"] == "hello"
    assert result["alignment_status"] == "partial"
