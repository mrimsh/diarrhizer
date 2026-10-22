"""Offscreen tests of the restart dialog (GUI:RESTART_DIALOG).

QSettings goes to a temp INI file, the audio format store to a temp path and
the HF cache listing is stubbed, so nothing here reads or writes the real ones.
"""

import dataclasses
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6", reason="GUI extra not installed")

from PySide6.QtWidgets import QDialog, QMessageBox  # noqa: E402

from diarrhizer.audio_formats import BUILTIN_FORMATS, ENV_AUDIO_FORMATS_FILE, AudioFormat  # noqa: E402
from diarrhizer_gui.job_scan import restart_settings  # noqa: E402
from diarrhizer_gui.screens import new_job_screen, restart_dialog  # noqa: E402

RECORDED = {
    "language": "ru",
    "device": "cpu",
    "min_speakers": 2,
    "max_speakers": 3,
    "asr_model": "small",
    "asr_compute_type": "int8",
    "asr_beam_size": 3,
    "asr_temperature": 0.25,  # more precision than the spin box shows
    "asr_condition_on_previous_text": False,
    "asr_initial_prompt": "Диаррайзер, ООО Ромашка",
    "asr_vad_filter": True,
    "asr_vad_min_silence_ms": 700,
    "audio_profile": "voice-call",
    "min_turn_duration": 0.45,
    "audio_format": BUILTIN_FORMATS["opus-24k"],
}


@pytest.fixture
def make_dialog(qapp, ini_settings, tmp_path, monkeypatch):
    _path, factory = ini_settings
    monkeypatch.setenv(ENV_AUDIO_FORMATS_FILE, str(tmp_path / "audio_formats.json"))
    monkeypatch.setattr(new_job_screen.model_cache, "list_cached_models", lambda: [])

    def make(stage="convert", mode="full", recorded=None, probe=None):
        return restart_dialog.RestartDialog(
            None,
            "intro",
            restart_settings(mode, stage),
            RECORDED if recorded is None else recorded,
            factory(),
            probe,
        )

    return make


def test_an_untouched_dialog_reports_no_changes(make_dialog):
    dialog = make_dialog("convert")
    # Every setting of every stage is offered ...
    every_setting = ["device"] + [key for _title, keys in restart_dialog.SECTIONS for key in keys]
    assert all(dialog.field(key) is not None for key in every_setting)
    # ... and showing them changes nothing, even where a field has to round the recorded value.
    assert dialog.field("asr_temperature").value() == pytest.approx(0.3, abs=0.051)
    assert dialog.overrides() == {}


def test_only_the_changed_settings_are_reported(make_dialog):
    dialog = make_dialog("transcribe")
    dialog.field("asr_model").setCurrentText("large-v3")
    dialog.field("asr_beam_size").setValue(5)
    dialog.field("asr_initial_prompt").setPlainText("")
    dialog.field("asr_vad_filter").setChecked(False)

    assert dialog.overrides() == {
        "asr_model": "large-v3",
        "asr_beam_size": 5,
        "asr_initial_prompt": None,  # cleared on purpose, not "unchanged"
        "asr_vad_filter": False,
    }


def test_blank_language_means_auto(make_dialog):
    dialog = make_dialog("transcribe")
    dialog.field("language").setCurrentText("  ")
    assert dialog.overrides() == {"language": "auto"}


@pytest.mark.parametrize(
    "stage, mode, offered, not_offered",
    [
        ("export", "full", [], ["asr_model", "device", "min_speakers", "min_turn_duration", "audio_profile"]),
        ("merge", "full", ["min_turn_duration"], ["asr_model", "device", "min_speakers"]),
        ("diarize", "full", ["device", "min_speakers", "max_speakers", "min_turn_duration"], ["asr_model", "language"]),
        ("transcribe", "full", ["asr_model", "language", "device", "min_turn_duration"], ["min_speakers", "audio_profile"]),
        ("convert", "asr_only", ["audio_profile", "audio_format", "asr_model"], ["min_speakers", "max_speakers"]),
    ],
)
def test_only_the_settings_of_the_recomputed_stages_are_offered(
    make_dialog, stage, mode, offered, not_offered
):
    dialog = make_dialog(stage, mode)
    assert all(dialog.field(key) is not None for key in offered)
    assert all(dialog.field(key) is None for key in not_offered)


def test_the_recorded_storage_format_is_selected_with_the_recorded_keep_wav(make_dialog):
    # opus-24k keeps the working WAV by default; this job was run with keep_wav off.
    recorded = {**RECORDED, "audio_format": dataclasses.replace(BUILTIN_FORMATS["opus-24k"], keep_wav=False)}
    dialog = make_dialog("convert", recorded=recorded)

    combo = dialog.field("audio_format")
    assert combo.count() == len(BUILTIN_FORMATS)  # no extra "as in the job" entry
    assert "Opus" in combo.currentText()
    keep_wav = dialog.field("audio_format_keep_wav")
    assert keep_wav.isEnabled() and not keep_wav.isChecked()
    assert dialog.overrides() == {}


def test_changing_the_storage_format_gives_the_resolved_format(make_dialog):
    dialog = make_dialog("convert")
    combo = dialog.field("audio_format")
    combo.setCurrentIndex(next(i for i in range(combo.count()) if "FLAC" in combo.itemText(i)))
    dialog.field("audio_format_keep_wav").setChecked(False)

    overrides = dialog.overrides()
    assert overrides == {"audio_format": dataclasses.replace(BUILTIN_FORMATS["flac"], keep_wav=False)}


def test_a_storage_format_gone_from_the_store_stays_available(make_dialog):
    gone = AudioFormat(name="my-opus", codec="libopus", extension="opus", bitrate="16k", keep_wav=False)
    dialog = make_dialog("convert", recorded={**RECORDED, "audio_format": gone})

    combo = dialog.field("audio_format")
    assert combo.count() == len(BUILTIN_FORMATS) + 1
    assert combo.currentText() == "my-opus — как в задании"
    assert dialog.overrides() == {}


def test_a_job_from_before_formats_existed_starts_on_wav(make_dialog):
    legacy = {key: value for key, value in RECORDED.items() if key != "audio_format"}
    dialog = make_dialog("convert", recorded=legacy)
    assert dialog.field("audio_format").currentText().startswith("WAV")
    assert dialog.overrides() == {}


def test_missing_recorded_settings_start_from_the_pipeline_defaults(make_dialog):
    dialog = make_dialog("transcribe", recorded={})
    assert dialog.field("asr_model").currentText() == "large-v3"
    assert dialog.field("asr_beam_size").value() == 5
    assert dialog.field("asr_vad_min_silence_ms").value() == 1000
    assert dialog.overrides() == {}


def test_cuda_is_switched_off_when_the_environment_has_none(make_dialog):
    probe = SimpleNamespace(cuda_available=False, cuda_message="нет CUDA")
    dialog = make_dialog("transcribe", recorded={**RECORDED, "device": "cuda"}, probe=probe)

    device = dialog.field("device")
    assert device.currentText() == "cpu"
    assert not device.model().item(0).isEnabled()
    assert dialog.overrides() == {"device": "cpu"}


def test_accept_refuses_a_speaker_minimum_above_the_maximum(make_dialog, monkeypatch):
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda _parent, _title, text, *_args: warnings.append(text))
    dialog = make_dialog("diarize")
    dialog.field("min_speakers").setValue(5)
    dialog.field("max_speakers").setValue(2)

    dialog.accept()

    assert dialog.result() != QDialog.DialogCode.Accepted
    assert "Минимум спикеров" in warnings[0]

    dialog.field("max_speakers").setValue(5)
    dialog.accept()
    assert dialog.result() == QDialog.DialogCode.Accepted


def test_accept_refuses_an_empty_model(make_dialog, monkeypatch):
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda _parent, _title, text, *_args: warnings.append(text))
    dialog = make_dialog("transcribe")
    dialog.field("asr_model").setCurrentText("  ")

    dialog.accept()

    assert dialog.result() != QDialog.DialogCode.Accepted
    assert "модель" in warnings[0]
