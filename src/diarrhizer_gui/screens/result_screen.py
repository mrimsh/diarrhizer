"""Result screen: transcript view + speaker renaming + re-export.

Pure filesystem read of merged/segments.json - same "no heavy imports at
module level" rule as the other screens (only PySide6 + stdlib here).
Re-export goes through the same PipelineWorker/QThread machinery as a new
job; this screen only builds the kwargs and emits a signal, it never touches
QThread itself (MainWindow owns that, same as NewJobScreen).
"""

import html
import json
import os
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from diarrhizer_gui import settings_keys


def _resolve_name(speaker_id: str, names: dict) -> str:
    return names.get(speaker_id, speaker_id)


def _format_timestamp(seconds: float) -> str:
    """Same HH:MM:SS shape as export/markdown_export.py's _format_timestamp -
    that function is private to the core package, so this is a 3-line
    equivalent rather than reaching into a non-public helper.
    """
    seconds = seconds or 0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def unique_speakers(segments: list) -> list:
    """Unique speaker_ids in order of first appearance."""
    seen = []
    for seg in segments:
        sid = seg.get("speaker_id", "Speaker_00")
        if sid not in seen:
            seen.append(sid)
    return seen


def build_transcript_html(segments: list, names: dict) -> str:
    if not segments:
        return "<p>Нет сегментов.</p>"

    parts = []
    for seg in segments:
        start = _format_timestamp(seg.get("start", 0))
        end = _format_timestamp(seg.get("end", 0))
        speaker_id = seg.get("speaker_id", "Speaker_00")
        name = html.escape(_resolve_name(speaker_id, names))
        text = html.escape(seg.get("text", "").strip())
        parts.append(
            f'<p style="margin:4px 0;">'
            f'<span style="color:#7d8394;">[{start} → {end}]</span> '
            f'<b style="color:#a85520;">{name}:</b> {text}</p>'
        )
    return "".join(parts)


class ResultScreen(QWidget):
    back_requested = Signal()
    reexport_requested = Signal(dict)

    def __init__(self) -> None:
        super().__init__()
        self._settings = QSettings("Diarrhizer", "DiarrhizerGUI")
        self._job_dir: Optional[Path] = None
        self._segments: list = []
        self._names: dict = {}
        self._pipeline_config: dict = {}
        self._speaker_fields: dict = {}

        self._title_label = QLabel("Просмотр результата")
        self._title_label.setStyleSheet("font-size: 18px; font-weight: 600;")

        self._open_folder_button = QPushButton("Открыть папку")
        self._open_folder_button.clicked.connect(self._open_folder)
        self._open_md_button = QPushButton("Открыть result.md")
        self._open_md_button.clicked.connect(lambda: self._open_file(self._result_md_path))
        self._open_json_button = QPushButton("Открыть result.json")
        self._open_json_button.clicked.connect(lambda: self._open_file(self._result_json_path))

        header = QHBoxLayout()
        header.addWidget(self._title_label)
        header.addStretch(1)
        header.addWidget(self._open_folder_button)
        header.addWidget(self._open_md_button)
        header.addWidget(self._open_json_button)

        self._speakers_layout = QVBoxLayout()
        self._speakers_layout.setSpacing(4)

        self._save_button = QPushButton("Сохранить и реэкспортировать")
        self._save_button.clicked.connect(self._emit_reexport)

        self._transcript = QTextEdit()
        self._transcript.setReadOnly(True)

        self._back_button = QPushButton("Назад к истории")
        self._back_button.clicked.connect(self.back_requested)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.addLayout(header)
        layout.addWidget(QLabel("Спикеры:"))
        layout.addLayout(self._speakers_layout)
        layout.addWidget(self._save_button, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self._transcript, stretch=1)
        layout.addWidget(self._back_button, alignment=Qt.AlignmentFlag.AlignRight)

        self._result_md_path: Optional[Path] = None
        self._result_json_path: Optional[Path] = None

    def load(self, job_dir: Path) -> None:
        self._job_dir = job_dir

        segments_path = job_dir / "merged" / "segments.json"
        read_error = None
        try:
            data = json.loads(segments_path.read_text(encoding="utf-8"))
            self._segments = data.get("segments", [])
        except (OSError, json.JSONDecodeError) as exc:
            self._segments = []
            read_error = str(exc)

        speakers_path = job_dir / "speakers.json"
        self._names = {}
        if speakers_path.exists():
            try:
                self._names = json.loads(speakers_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self._names = {}

        meta = {}
        meta_path = job_dir / "meta" / "run.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                meta = {}
        self._pipeline_config = meta.get("pipeline_config", {})

        input_path = meta.get("input_path")
        self._title_label.setText(Path(input_path).name if input_path else job_dir.name)

        self._result_md_path = job_dir / "export" / "result.md"
        self._result_json_path = job_dir / "export" / "result.json"
        self._open_md_button.setEnabled(self._result_md_path.exists())
        self._open_json_button.setEnabled(self._result_json_path.exists())

        self._rebuild_speaker_panel()
        self._render_transcript()

        if read_error:
            self._transcript.setPlainText(
                f"Не удалось прочитать merged/segments.json: {read_error}"
            )

    def _rebuild_speaker_panel(self) -> None:
        while self._speakers_layout.count():
            item = self._speakers_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._speaker_fields = {}

        for speaker_id in unique_speakers(self._segments):
            label = QLabel(speaker_id)
            label.setFixedWidth(100)
            field = QLineEdit(self._names.get(speaker_id, speaker_id))
            field.textChanged.connect(self._render_transcript)
            self._speaker_fields[speaker_id] = field

            row_widget = QWidget()
            row = QHBoxLayout(row_widget)
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(label)
            row.addWidget(field)
            self._speakers_layout.addWidget(row_widget)

    def _current_names(self) -> dict:
        return {sid: field.text() for sid, field in self._speaker_fields.items()}

    def _render_transcript(self) -> None:
        self._transcript.setHtml(build_transcript_html(self._segments, self._current_names()))

    def _open_folder(self) -> None:
        if self._job_dir is not None:
            os.startfile(str(self._job_dir))

    def _open_file(self, path: Optional[Path]) -> None:
        if path is not None and path.exists():
            os.startfile(str(path))

    def _emit_reexport(self) -> None:
        if self._job_dir is None:
            return

        names = self._current_names()
        speakers_path = self._job_dir / "speakers.json"
        speakers_path.write_text(json.dumps(names, ensure_ascii=False, indent=2), encoding="utf-8")
        self._names = names

        default_out = str(Path.cwd() / "out")
        out_dir = self._settings.value(settings_keys.OUT_DIR, default_out)

        kwargs = {
            "job_dir": self._job_dir,
            "out_dir": out_dir,
            "from_stage": "export",
            "to_stage": "export",
            "speakers": names,
            # ExportStage.is_cache_valid() only compares file mtimes
            # (segments.json vs result.md/result.json) - it has no way to
            # see that `speakers` changed, since that's an in-memory dict,
            # not a tracked artifact. Without forcing, a re-export after a
            # rename is silently a no-op (confirmed: this is also true of
            # the bare CLI's own documented --from-stage export --speakers
            # workflow in README.md, not something this GUI introduced).
            "force_stage": "export",
        }
        # Forward the original run's language/device/speaker-range so the
        # regenerated result.md/result.json header reflects reality instead
        # of run_pipeline()'s own defaults (see plan: this is a real gap in
        # the bare CLI --job-dir --from-stage export workflow too).
        for key in ("language", "device", "min_speakers", "max_speakers"):
            value = self._pipeline_config.get(key)
            if value is not None:
                kwargs[key] = value

        self.reexport_requested.emit(kwargs)
