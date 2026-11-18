"""Metadata preview for the current media file."""

import json
import math
from pathlib import Path

from PySide6.QtCore import QProcess, QTimer
from PySide6.QtWidgets import QPlainTextEdit

from diarrhizer.adapters.ffmpeg import media_probe_command
from diarrhizer.export.turns import format_timestamp


# [SEMANTIC-BEGIN] GUI:MEDIA_INFO
# @purpose: Show file properties and the numbers to use when selecting voice tracks
# @description: Inspect just the current file with an asynchronous FFprobe process. A brief
#   debounce and cancellation prevent old selections from updating the preview; a timeout
#   bounds slow probes. Streams are numbered from one, excluding video. File metadata is
#   rendered as plain text. A probe failure is informational and does not block job creation.
# @inputs: current local media path, or None when no file is selected
# @outputs: read-only preview or a concise failure message
# @sideEffects: Starts and may kill FFprobe; never changes the file or job settings
# @errors: Missing FFprobe, unreadable files and timeouts are shown in the preview
# @see: GUI:NEW_JOB, ADAPTER:MEDIA_PROBE
class MediaInfo(QPlainTextEdit):
    def __init__(self) -> None:
        super().__init__()
        self.setReadOnly(True)
        self.setAccessibleName("Сведения об аудиофайле")
        self.setPlaceholderText("Выберите файл, чтобы увидеть его аудиодорожки.")
        self._source: Path | None = None
        self._active: Path | None = None
        self._process = QProcess(self)
        self._process.finished.connect(self._finished)
        self._process.errorOccurred.connect(self._process_error)
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(120)
        self._debounce.timeout.connect(self._start)
        self._deadline = QTimer(self)
        self._deadline.setSingleShot(True)
        self._deadline.timeout.connect(self._timed_out)
        self._display("")

    def show_file(self, path: str | None) -> None:
        source = Path(path) if path is not None else None
        if source == self._source:
            return
        self._source = source
        self._active = None
        self._debounce.stop()
        self._deadline.stop()
        if self._process.state() != QProcess.ProcessState.NotRunning:
            self._process.kill()
        if source is None:
            self._display("")
        else:
            self._display("Читаем свойства файла…")
            self._debounce.start()

    def _display(self, body: str) -> None:
        text = f"{self._source.name}\n{body}" if self._source else body
        self.setPlainText(text)
        lines = max(2, min(text.count("\n") + 1, 7))
        self.setFixedHeight(lines * self.fontMetrics().lineSpacing() + 14)

    def _start(self) -> None:
        if self._source is None:
            return
        if self._process.state() != QProcess.ProcessState.NotRunning:
            self._debounce.start()
            return
        try:
            program, *args = media_probe_command(self._source)
        except OSError:
            self._display("Сведения недоступны: проверьте установку FFmpeg и FFprobe.")
            return
        self._active = self._source
        self._deadline.start(10_000)
        self._process.start(program, args)

    def _finished(self, code: int, status: QProcess.ExitStatus) -> None:
        if self._active is None:
            return
        self._active = None
        self._deadline.stop()
        if code == 0 and status == QProcess.ExitStatus.NormalExit:
            try:
                metadata = json.loads(bytes(self._process.readAllStandardOutput()))
                self._display(_format_metadata(metadata))
                return
            except (ValueError, TypeError, AttributeError):
                pass
        self._display("Не удалось прочитать сведения об аудио.")

    def _process_error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart and self._active is not None:
            self._active = None
            self._deadline.stop()
            self._display("Не удалось запустить FFprobe.")

    def _timed_out(self) -> None:
        self._active = None
        self._process.kill()
        self._display("Не удалось прочитать сведения об аудио за 10 секунд.")


def _number(raw) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0
    return value if math.isfinite(value) and value > 0 else 0


def _format_metadata(metadata: dict) -> str:
    streams = metadata.get("streams", [])
    media = metadata.get("format", {})
    summary = [f"Аудиодорожек: {len(streams)}"]
    if duration := _number(media.get("duration")):
        summary.append(f"Длительность: {format_timestamp(duration)}")
    if size := _number(media.get("size")):
        summary.append(f"Размер: {size / (1024 * 1024):.1f} МиБ" if size >= 1024 * 1024 else
                       f"Размер: {size / 1024:.1f} КиБ")
    lines = [" · ".join(summary)]
    for number, stream in enumerate(streams, 1):
        codec = stream.get("codec_name")
        properties = [codec.upper() if codec else "кодек неизвестен"]
        if rate := _number(stream.get("sample_rate")):
            properties.append(f"{rate / 1000:g} кГц")
        if count := _number(stream.get("channels")):
            channels = {1: "1 канал (моно)", 2: "2 канала (стерео)"}.get(count, f"{count:g} кан.")
            layout = stream.get("channel_layout")
            if count > 2 and layout and layout != "unknown":
                channels += f" ({layout})"
            properties.append(channels)
        tags = {key.lower(): value for key, value in stream.get("tags", {}).items()}
        for key in ("title", "language"):
            if tag := " ".join(str(tags.get(key, "")).split())[:100]:
                properties.append(tag)
        lines.append(f"{number}. " + " · ".join(properties))
    return "\n".join(lines)
# [SEMANTIC-END] GUI:MEDIA_INFO
