"""Channel choices reach queued jobs and remain editable on restarts."""

import pytest

pytest.importorskip("PySide6")

from diarrhizer.audio_formats import ENV_AUDIO_FORMATS_FILE
from diarrhizer_gui.child_tasks import build_stages
from diarrhizer_gui.job_scan import restart_settings
from diarrhizer_gui.screens import new_job_screen
from diarrhizer_gui.screens.restart_dialog import RestartDialog


def test_channel_choices_queue_and_reopen_without_token(qapp, ini_settings, tmp_path, monkeypatch):
    _, factory = ini_settings
    monkeypatch.setattr(new_job_screen, "QSettings", factory)
    monkeypatch.setenv(ENV_AUDIO_FORMATS_FILE, str(tmp_path / "formats.json"))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    source = tmp_path / "call.wav"
    source.write_bytes(b"test")
    screen = new_job_screen.NewJobScreen()
    screen.add_paths([source])
    screen._mode_combo.setCurrentIndex(screen._mode_combo.findData("me_others"))
    screen._my_channel_combo.setCurrentIndex(screen._my_channel_combo.findData("right"))
    assert screen._run_button.isEnabled()
    assert not screen._min_speakers.isEnabled()
    screen._one_on_one_checkbox.setChecked(False)
    assert not screen._run_button.isEnabled() and "HF-токен" in screen._warning_label.text()
    screen._one_on_one_checkbox.setChecked(True)
    queued = []
    screen.runs_requested.connect(queued.extend)
    screen._emit_run_requested()
    assert len(queued) == 1 and queued[0].mode == "me_others"
    assert queued[0].kwargs["channel_mode"] == "me-others"
    assert queued[0].kwargs["my_channel"] == "right"
    assert queued[0].kwargs["one_on_one"] is True
    assert queued[0].kwargs["min_speakers"] == queued[0].kwargs["max_speakers"] == 1
    again = new_job_screen.NewJobScreen()
    assert again._mode_combo.currentData() == "me_others"
    assert again._my_channel_combo.currentData() == "right"
    assert again._one_on_one_checkbox.isChecked()
    assert [s.NAME for s in build_stages("me_others")] == ["convert", "transcribe", "diarize", "merge", "export"]
    screen.close()
    again.close()


def test_restart_keeps_channel_options_and_changes_one_on_one(qapp, ini_settings):
    _, factory = ini_settings
    dialog = RestartDialog(None, "Перезапуск", restart_settings("me_others", "diarize"),
                           {"channel_mode": "me-others", "my_channel": "right", "one_on_one": True,
                            "device": "cpu"}, factory("test", "test"))
    assert dialog.overrides() == {}
    dialog._widgets["one_on_one"].setChecked(False)
    assert dialog.overrides() == {"one_on_one": False}
    assert dialog._problem() is None
    dialog.close()


def test_track_choices_are_validated_queued_and_remembered(qapp, ini_settings, tmp_path, monkeypatch):
    _, factory = ini_settings
    monkeypatch.setattr(new_job_screen, 'QSettings', factory)
    monkeypatch.setenv(ENV_AUDIO_FORMATS_FILE, str(tmp_path / 'formats.json'))
    source = tmp_path / 'tracks.mkv'
    source.write_bytes(b'test')
    screen = new_job_screen.NewJobScreen()
    screen.add_paths([source])
    screen._mode_combo.setCurrentIndex(screen._mode_combo.findData('me_others'))
    screen._channel_source_combo.setCurrentIndex(screen._channel_source_combo.findData('tracks'))
    screen._my_track_spin.setValue(2)
    assert not screen._run_button.isEnabled()
    assert 'разные аудиодорожки' in screen._warning_label.text()
    screen._my_track_spin.setValue(1)
    screen._mix_track_spin.setValue(3)
    assert screen._run_button.isEnabled()
    queued = []
    screen.runs_requested.connect(queued.extend)
    screen._emit_run_requested()
    assert queued[0].kwargs['channel_source'] == 'tracks'
    assert [queued[0].kwargs[k] for k in ('my_track', 'others_track', 'mix_track')] == [1, 2, 3]
    again = new_job_screen.NewJobScreen()
    assert again._channel_source_combo.currentData() == 'tracks'
    assert again._mix_track_spin.value() == 3
    assert not again._tracks_options.isHidden() and again._stereo_options.isHidden()
    screen.close()
    again.close()


def test_conversion_restart_exposes_audio_track_selection(qapp, ini_settings):
    _, factory = ini_settings
    dialog = RestartDialog(None, 'Перезапуск', restart_settings('me_others', 'convert'),
                           {'channel_mode': 'me-others', 'channel_source': 'tracks', 'my_track': 1,
                            'others_track': 2, 'mix_track': 3, 'one_on_one': True, 'device': 'cpu'},
                           factory('test', 'test'))
    assert dialog.overrides() == {}
    dialog._widgets['my_track'].setValue(2)
    assert dialog._problem() is not None
    dialog._widgets['my_track'].setValue(3)
    assert dialog._problem() is None and dialog.overrides() == {'my_track': 3}
    dialog.close()
