"""Tests for diarrhizer_gui.custom_models (the remembered list of user-added
ASR model repo ids).

Runs against an INI-format QSettings pointed at a tmp_path file - QSettings
needs no QApplication, so these stay headless and never touch the real
registry/settings store of the machine running them.
"""

import pytest

pytest.importorskip("PySide6", reason="GUI extra not installed")

from PySide6.QtCore import QSettings  # noqa: E402

from diarrhizer_gui import custom_models, settings_keys  # noqa: E402

PRESETS = ["tiny", "base", "large-v3"]


@pytest.fixture
def settings(tmp_path):
    store = QSettings(str(tmp_path / "test.ini"), QSettings.Format.IniFormat)
    yield store
    store.sync()


def test_empty_store_returns_empty_list(settings):
    assert custom_models.load_custom_models(settings) == []


def test_add_then_load_round_trips(settings):
    assert custom_models.add_custom_model(settings, "bond005/whisper-podlodka-turbo-ct2") is True
    assert custom_models.load_custom_models(settings) == ["bond005/whisper-podlodka-turbo-ct2"]


def test_single_entry_survives_a_reopened_store(tmp_path):
    path = str(tmp_path / "test.ini")
    first = QSettings(path, QSettings.Format.IniFormat)
    custom_models.add_custom_model(first, "bzikst/faster-whisper-podlodka-turbo")
    first.sync()

    reopened = QSettings(path, QSettings.Format.IniFormat)
    assert custom_models.load_custom_models(reopened) == ["bzikst/faster-whisper-podlodka-turbo"]


def test_add_preserves_insertion_order(settings):
    custom_models.add_custom_model(settings, "a/one")
    custom_models.add_custom_model(settings, "b/two")
    custom_models.add_custom_model(settings, "c/three")
    assert custom_models.load_custom_models(settings) == ["a/one", "b/two", "c/three"]


@pytest.mark.parametrize("value", ["", "   "])
def test_blank_ids_are_rejected(settings, value):
    assert custom_models.add_custom_model(settings, value) is False
    assert custom_models.load_custom_models(settings) == []


def test_duplicate_is_not_added_twice(settings):
    assert custom_models.add_custom_model(settings, "a/one") is True
    assert custom_models.add_custom_model(settings, "  a/one  ") is False
    assert custom_models.load_custom_models(settings) == ["a/one"]


def test_remove_drops_only_the_named_entry(settings):
    custom_models.add_custom_model(settings, "a/one")
    custom_models.add_custom_model(settings, "b/two")
    custom_models.remove_custom_model(settings, "a/one")
    assert custom_models.load_custom_models(settings) == ["b/two"]


def test_remove_unknown_entry_is_a_no_op(settings):
    custom_models.add_custom_model(settings, "a/one")
    custom_models.remove_custom_model(settings, "nope/nope")
    assert custom_models.load_custom_models(settings) == ["a/one"]


def test_load_normalises_a_list_valued_store(settings):
    # Another Qt backend (or a hand-edited store) can hand back a list here.
    settings.setValue(settings_keys.CUSTOM_ASR_MODELS, ["a/one", "b/two"])
    assert custom_models.load_custom_models(settings) == ["a/one", "b/two"]


def test_asr_model_choices_appends_custom_after_presets(settings):
    custom_models.add_custom_model(settings, "bond005/whisper-podlodka-turbo-ct2")
    assert custom_models.asr_model_choices(settings, PRESETS) == [
        *PRESETS,
        "bond005/whisper-podlodka-turbo-ct2",
    ]


def test_asr_model_choices_does_not_duplicate_a_preset(settings):
    custom_models.add_custom_model(settings, "large-v3")
    assert custom_models.asr_model_choices(settings, PRESETS) == PRESETS


def test_asr_model_choices_leaves_presets_untouched(settings):
    custom_models.add_custom_model(settings, "a/one")
    custom_models.asr_model_choices(settings, PRESETS)
    assert PRESETS == ["tiny", "base", "large-v3"]


class _Info:
    def __init__(self, repo_id, is_ctranslate2):
        self.repo_id = repo_id
        self.is_ctranslate2 = is_ctranslate2


def test_detect_cached_asr_models_keeps_only_ct2_whisper_repos():
    cached = [
        _Info("bzikst/faster-whisper-podlodka-turbo", True),
        _Info("openai/whisper-large-v3", False),  # transformers format
        _Info("pyannote/segmentation-3.0", False),
        _Info("jonatasgrosman/wav2vec2-large-xlsr-53-russian", True),  # aligner
    ]
    assert custom_models.detect_cached_asr_models(cached) == [
        "bzikst/faster-whisper-podlodka-turbo"
    ]


def test_detect_cached_asr_models_skips_repos_behind_presets():
    cached = [_Info("Systran/faster-whisper-small", True), _Info("a/faster-whisper-x", True)]
    assert custom_models.detect_cached_asr_models(cached, ["small"]) == ["a/faster-whisper-x"]


def test_asr_model_choices_appends_extra_without_duplicates(settings):
    custom_models.add_custom_model(settings, "a/one")
    assert custom_models.asr_model_choices(settings, PRESETS, ["a/one", "b/two"]) == [
        *PRESETS,
        "a/one",
        "b/two",
    ]
