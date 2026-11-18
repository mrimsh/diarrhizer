"""The New Job preview reads actual stream metadata and follows the current file."""

import subprocess

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEventLoop, QTimer

from diarrhizer.adapters.ffmpeg import media_probe_command, resolve_ffmpeg_path
from diarrhizer_gui.screens import new_job_screen


def _wait_for_preview(preview, expected):
    loop = QEventLoop()
    timer = QTimer()
    timer.setInterval(20)

    def check():
        if expected in preview.toPlainText():
            loop.quit()

    timer.timeout.connect(check)
    deadline = QTimer()
    deadline.setSingleShot(True)
    deadline.timeout.connect(loop.quit)
    if expected not in preview.toPlainText():
        timer.start()
        deadline.start(5000)
        loop.exec()
    assert expected in preview.toPlainText()
    return preview.toPlainText()


def test_current_file_shows_audio_ordinals_instead_of_container_indexes(qapp, ini_settings, tmp_path, monkeypatch):
    _, settings = ini_settings
    monkeypatch.setattr(new_job_screen, "QSettings", settings)
    ffmpeg = resolve_ffmpeg_path()
    if not ffmpeg:
        pytest.skip("FFmpeg not installed")
    stereo = tmp_path / "Стерео.wav"
    multitrack = tmp_path / "Две дорожки.mkv"
    try:
        media_probe_command(stereo)
    except FileNotFoundError:
        pytest.skip("FFprobe not installed")

    subprocess.run([ffmpeg, "-y", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=stereo:d=2",
                    str(stereo)], check=True, capture_output=True, timeout=15)
    subprocess.run([ffmpeg, "-y", "-f", "lavfi", "-i", "color=size=16x16:rate=1:duration=2",
                    "-i", str(stereo), "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono:d=2",
                    "-map", "0:v", "-map", "1:a", "-map", "2:a", "-c:v", "ffv1", "-c:a", "pcm_s16le",
                    "-metadata:s:a:0", "title=Микрофон", "-metadata:s:a:1", "title=Собеседник",
                    str(multitrack)], check=True, capture_output=True, timeout=15)

    screen = new_job_screen.NewJobScreen()
    screen.add_paths([stereo, multitrack])
    preview = screen._media_info
    text = _wait_for_preview(preview, "Аудиодорожек: 1")
    assert "Длительность: 00:00:02" in text
    assert "Размер:" in text
    assert "1. PCM_S16LE · 16 кГц · 2 канала (стерео)" in text

    screen._files.setCurrentRow(1)
    text = _wait_for_preview(preview, "Аудиодорожек: 2")
    assert "1. PCM_S16LE · 16 кГц · 2 канала (стерео) · Микрофон" in text
    assert "2. PCM_S16LE · 48 кГц · 1 канал (моно) · Собеседник" in text
    assert stereo.name not in text and multitrack.name in text

    screen._files.clear()
    assert preview.toPlainText() == ""
    screen.close()


def test_probe_failure_does_not_block_queuing(qapp, ini_settings, tmp_path, monkeypatch):
    _, settings = ini_settings
    monkeypatch.setattr(new_job_screen, "QSettings", settings)

    def unavailable(path):
        raise FileNotFoundError("FFprobe not installed")

    monkeypatch.setattr("diarrhizer_gui.screens.media_info.media_probe_command", unavailable)
    source = tmp_path / "call.wav"
    source.write_bytes(b"preview cannot read this")
    screen = new_job_screen.NewJobScreen()
    screen._mode_combo.setCurrentIndex(screen._mode_combo.findData("asr_only"))
    screen.add_paths([source])
    _wait_for_preview(screen._media_info, "Сведения недоступны")
    assert screen._run_button.isEnabled()
    screen.close()
