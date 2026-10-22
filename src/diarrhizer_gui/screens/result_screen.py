"""Result screen: the transcript of one job, speaker naming, playback.

- The transcript is shown as speaker turns (EXPORT:SPEAKER_TURNS), one
  colour per speaker; every timestamp is a link that plays the recording
  from that moment, and "▶" next to a speaker plays a sample of their voice -
  which is what makes "who is Speaker_01?" answerable.
- Names are saved to speakers.json and the exports are re-rendered right
  here, in-process: export is pure Python and takes milliseconds, so there
  is no need to go through the job queue for it.
- "Пересчитать спикеров" re-runs diarization with another speaker range
  (and adds speakers to a recognition-only job) through the job queue,
  keeping the transcript (GUI:JOB_PLANS).

Only PySide6 + light diarrhizer modules at import time; QtMultimedia is
imported on first use and the player simply stays hidden without it.
"""

import html
import json
import os
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QKeySequence, QShortcut, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTextBrowser,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from diarrhizer.audio_formats import find_job_audio
from diarrhizer.export.plain_transcript import format_plain_transcript
from diarrhizer.export.speakers import normalize_speaker_id
from diarrhizer.export.turns import format_timestamp, format_turn_line, group_turns, speaking_time
from diarrhizer.utils import write_json_atomic
from diarrhizer_gui.job_queue import QueuedRun
from diarrhizer_gui.job_scan import read_speaker_names, rediarize_plan, set_aside_speaker_names
from diarrhizer_gui.screens.new_job_screen import hf_token_present

SPEAKER_COLORS = [
    "#a85520", "#2c6fbb", "#2f7d52", "#8e44ad", "#c0392b", "#16808a", "#7a6a1f", "#d35400",
]
SAMPLE_SECONDS = 8.0
MAX_SEARCH_MATCHES = 5000

# (menu label, path relative to the job folder)
OPEN_TARGETS = [
    ("Протокол (result.md)", "export/result.md"),
    ("Текст (result.txt)", "export/result.txt"),
    ("Субтитры (result.srt)", "export/result.srt"),
    ("Данные (result.json)", "export/result.json"),
    ("Текст без спикеров (asr/transcript.txt)", "asr/transcript.txt"),
]


def unique_speakers(segments: list) -> list:
    """Unique normalized speaker_ids in order of first appearance."""
    seen = []
    for seg in segments:
        sid = normalize_speaker_id(seg.get("speaker_id") or "Speaker_00")
        if sid not in seen:
            seen.append(sid)
    return seen


def speaker_colors(turns: list) -> dict:
    """Display name -> colour, in order of first appearance."""
    colors: dict = {}
    for turn in turns:
        name = turn["speaker_name"]
        if name not in colors:
            colors[name] = SPEAKER_COLORS[len(colors) % len(SPEAKER_COLORS)]
    return colors


def build_transcript_html(segments: list, names: dict) -> str:
    """Speaker turns as HTML; each timestamp links to "t:<seconds>"."""
    if not segments:
        return "<p>Нет сегментов.</p>"
    turns = group_turns(segments, names)
    colors = speaker_colors(turns)
    parts = []
    for turn in turns:
        start = turn["start"]
        name = html.escape(turn["speaker_name"])
        text = html.escape(turn["text"])
        parts.append(
            f'<p style="margin:0 0 8px 0;">'
            f'<a href="t:{start:.3f}" style="color:#7d8394; text-decoration:none;">'
            f"[{format_timestamp(start)}]</a> "
            f'<b style="color:{colors[turn["speaker_name"]]};">{name}:</b> {text}</p>'
        )
    return "".join(parts)


def transcript_plain_text(segments: list, names: dict) -> str:
    """The transcript as result.txt shows it, for the clipboard."""
    return "\n\n".join(format_turn_line(turn) for turn in group_turns(segments, names)) + "\n"


def sample_range(segments: list, speaker_id: str) -> Optional[tuple]:
    """(start, end) of a speaker's longest segment, capped at SAMPLE_SECONDS."""
    best = None
    for seg in segments:
        if normalize_speaker_id(seg.get("speaker_id") or "Speaker_00") != speaker_id:
            continue
        start = float(seg.get("start") or 0.0)
        end = float(seg.get("end") or start)
        if best is None or end - start > best[1] - best[0]:
            best = (start, end)
    if best is None:
        return None
    return best[0], min(best[1], best[0] + SAMPLE_SECONDS)


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


# [SEMANTIC-BEGIN] GUI:REEXPORT
# @purpose: Re-render a finished job's exports with new speaker names, in the GUI process
# @description: Runs run_pipeline() with only ExportStage (pure Python, no torch) and force_stage="export",
#   passing the job's recorded language/device/speaker range so the regenerated headers match the original
#   run. Takes milliseconds, so the Result screen calls it directly instead of queueing a job.
# @inputs: job_dir, speakers mapping {Speaker_XX: name}
# @outputs: run_pipeline() result dict; export/result.* rewritten
# @sideEffects: Writes export files
# @errors: FileNotFoundError (no merged/segments.json or no recorded input), RuntimeError, ValueError
# @see: STAGE:EXPORT, PIPELINE:RUNNER
def reexport(job_dir: Path, speakers: dict) -> dict:
    from diarrhizer.pipeline.runner import run_pipeline
    from diarrhizer.pipeline.stages.export import ExportStage

    config = _read_json(job_dir / "meta" / "run.json").get("pipeline_config") or {}
    recorded = {
        key: config[key]
        for key in ("language", "device", "min_speakers", "max_speakers")
        if config.get(key) is not None
    }
    return run_pipeline(
        out_dir=job_dir.parent,
        job_dir=job_dir,
        stages=[ExportStage()],
        speakers=speakers or None,
        force_stage="export",
        **recorded,
    )
# [SEMANTIC-END] GUI:REEXPORT


class RediarizeDialog(QDialog):
    def __init__(self, parent, has_diarization: bool, current_count: int) -> None:
        super().__init__(parent)
        self.setWindowTitle("Пересчитать спикеров" if has_diarization else "Разделить по спикерам")
        intro = QLabel(
            "Распознанный текст сохранится — заново выполнятся только диаризация, объединение и "
            "экспорт (обычно это несколько минут).\n\n"
            + (
                "Имена спикеров будут сброшены: новая диаризация нумерует спикеров заново "
                "(прежние имена сохранятся в speakers.before-rediarize.json)."
                if has_diarization
                else "Задание выполнялось в режиме «Только распознавание» — теперь реплики "
                "будут разделены по спикерам."
            )
        )
        intro.setWordWrap(True)
        self.min_spin = QSpinBox()
        self.min_spin.setRange(1, 20)
        self.max_spin = QSpinBox()
        self.max_spin.setRange(1, 20)
        count = max(1, current_count) if has_diarization else 2
        self.min_spin.setValue(count)
        self.max_spin.setValue(count)
        row = QHBoxLayout()
        row.addWidget(self.min_spin)
        row.addWidget(QLabel("–"))
        row.addWidget(self.max_spin)
        row.addStretch(1)
        hint = QLabel("Если число участников известно точно, укажите его в обоих полях.")
        hint.setStyleSheet("color: #7d8394;")

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Запустить")
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)

        form = QFormLayout()
        form.addRow("Спикеры (мин–макс):", row)
        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addLayout(form)
        layout.addWidget(hint)
        layout.addWidget(buttons)

    def _accept(self) -> None:
        if self.min_spin.value() > self.max_spin.value():
            QMessageBox.warning(self, self.windowTitle(), "Минимум не может быть больше максимума.")
            return
        self.accept()


class ResultScreen(QWidget):
    back_requested = Signal()
    runs_requested = Signal(list)  # [QueuedRun]

    def __init__(self) -> None:
        super().__init__()
        self._job_dir: Optional[Path] = None
        self._segments: list = []
        self._names: dict = {}
        self._meta: dict = {}
        self._speaker_fields: dict = {}
        self._has_diarization = False
        self._matches: list = []
        self._match_index = -1

        # player state (QtMultimedia is created lazily, see _ensure_player)
        self._player = None
        self._audio_output = None
        self._stop_at_ms: Optional[int] = None
        self._slider_dragging = False

        # --- header ---
        self._title_label = QLabel("Просмотр результата")
        self._title_label.setStyleSheet("font-size: 18px; font-weight: 600;")
        self._title_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._subtitle_label = QLabel()
        self._subtitle_label.setStyleSheet("color: #7d8394;")

        self._open_menu = QMenu(self)
        open_button = QToolButton()
        open_button.setText("Открыть файл")
        open_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        open_button.setMenu(self._open_menu)
        self._open_menu.aboutToShow.connect(self._fill_open_menu)

        folder_button = QPushButton("Папка задания")
        folder_button.clicked.connect(self._open_folder)

        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        title_box.addWidget(self._title_label)
        title_box.addWidget(self._subtitle_label)
        header.addLayout(title_box, stretch=1)
        header.addWidget(open_button)
        header.addWidget(folder_button)

        # --- speakers ---
        speakers_title = QLabel("Спикеры")
        speakers_title.setStyleSheet("font-weight: 600; padding-top: 6px;")
        self._speakers_grid = QGridLayout()
        self._speakers_grid.setHorizontalSpacing(8)
        self._speakers_grid.setVerticalSpacing(4)
        speakers_widget = QWidget()
        speakers_widget.setLayout(self._speakers_grid)
        self._speakers_scroll = QScrollArea()
        self._speakers_scroll.setWidgetResizable(True)
        self._speakers_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._speakers_scroll.setWidget(speakers_widget)
        self._speakers_scroll.setMaximumHeight(170)

        self._speakers_hint = QLabel(
            "▶ — послушать фрагмент речи спикера. Одинаковое имя у двух спикеров объединяет их реплики "
            "(диаризация иногда делит одного человека надвое)."
        )
        self._speakers_hint.setWordWrap(True)
        self._speakers_hint.setStyleSheet("color: #7d8394;")

        self._save_button = QPushButton("Сохранить имена")
        self._save_button.setToolTip("Записать speakers.json и сразу обновить result.md/.txt/.srt/.json")
        self._save_button.clicked.connect(self._save_names)
        self._rediarize_button = QPushButton("Пересчитать спикеров…")
        self._rediarize_button.clicked.connect(self._rediarize)
        self._status_label = QLabel()
        self._status_label.setWordWrap(True)
        actions_row = QHBoxLayout()
        actions_row.addWidget(self._save_button)
        actions_row.addWidget(self._rediarize_button)
        actions_row.addWidget(self._status_label, stretch=1)

        # --- search / copy ---
        self._search_field = QLineEdit()
        self._search_field.setPlaceholderText("Поиск по тексту (Ctrl+F)…")
        self._search_field.setClearButtonEnabled(True)
        self._search_field.returnPressed.connect(lambda: self._step_match(1))
        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(200)
        self._search_timer.timeout.connect(self._update_search)
        self._search_field.textChanged.connect(self._search_timer.start)
        prev_button = QPushButton("▲")
        prev_button.setFixedWidth(32)
        prev_button.setToolTip("Предыдущее совпадение")
        prev_button.clicked.connect(lambda: self._step_match(-1))
        next_button = QPushButton("▼")
        next_button.setFixedWidth(32)
        next_button.setToolTip("Следующее совпадение (Enter)")
        next_button.clicked.connect(lambda: self._step_match(1))
        self._match_label = QLabel()
        self._match_label.setStyleSheet("color: #7d8394;")
        copy_button = QPushButton("Копировать текст")
        copy_button.setToolTip("Весь протокол с текущими именами — в буфер обмена")
        copy_button.clicked.connect(self._copy_transcript)
        search_row = QHBoxLayout()
        search_row.addWidget(self._search_field, stretch=1)
        search_row.addWidget(prev_button)
        search_row.addWidget(next_button)
        search_row.addWidget(self._match_label)
        search_row.addSpacing(12)
        search_row.addWidget(copy_button)
        QShortcut(QKeySequence.StandardKey.Find, self, self._focus_search)

        # --- transcript ---
        self._transcript = QTextBrowser()
        self._transcript.setOpenLinks(False)
        self._transcript.setOpenExternalLinks(False)
        self._transcript.anchorClicked.connect(self._on_anchor)
        self._transcript.setStyleSheet("QTextBrowser { font-size: 13px; padding: 6px; }")
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(250)
        self._render_timer.timeout.connect(self._render_transcript)

        # --- player ---
        self._play_button = QPushButton("▶")
        self._play_button.setFixedWidth(36)
        self._play_button.setToolTip("Воспроизвести / пауза. Щелчок по времени в тексте — перейти к реплике")
        self._play_button.clicked.connect(self._toggle_play)
        self._position_label = QLabel("00:00:00 / 00:00:00")
        self._position_label.setStyleSheet("color: #565c6b; font-family: Consolas;")
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.sliderPressed.connect(self._on_slider_pressed)
        self._slider.sliderReleased.connect(self._on_slider_released)
        self._player_bar = QWidget()
        player_row = QHBoxLayout(self._player_bar)
        player_row.setContentsMargins(0, 0, 0, 0)
        player_row.addWidget(self._play_button)
        player_row.addWidget(self._position_label)
        player_row.addWidget(self._slider, stretch=1)
        self._player_bar.hide()
        self._player_note = QLabel()
        self._player_note.setStyleSheet("color: #7d8394;")
        self._player_note.hide()

        back_button = QPushButton("Назад к истории")
        back_button.clicked.connect(self.back_requested)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.addLayout(header)
        layout.addWidget(speakers_title)
        layout.addWidget(self._speakers_scroll)
        layout.addWidget(self._speakers_hint)
        layout.addLayout(actions_row)
        layout.addLayout(search_row)
        layout.addWidget(self._transcript, stretch=1)
        layout.addWidget(self._player_bar)
        layout.addWidget(self._player_note)
        layout.addWidget(back_button, alignment=Qt.AlignmentFlag.AlignRight)

    # --- loading ---------------------------------------------------------------

    def load(self, job_dir: Path) -> None:
        self._job_dir = job_dir
        self._set_status("", "#565c6b")

        read_error = None
        try:
            data = json.loads((job_dir / "merged" / "segments.json").read_text(encoding="utf-8"))
            self._segments = data.get("segments", [])
        except (OSError, json.JSONDecodeError) as exc:
            self._segments = []
            read_error = str(exc)

        self._names = read_speaker_names(job_dir)

        self._meta = _read_json(job_dir / "meta" / "run.json")
        self._has_diarization = (job_dir / "diar" / "diarization.json").exists()
        input_path = self._meta.get("input_path")
        self._title_label.setText(Path(input_path).name if input_path else job_dir.name)
        self._subtitle_label.setText(self._subtitle())
        self._rediarize_button.setText(
            "Пересчитать спикеров…" if self._has_diarization else "Разделить по спикерам…"
        )
        self._rediarize_button.setToolTip(
            "Заново выполнить диаризацию с другим числом спикеров, не распознавая текст заново"
            if self._has_diarization
            else "Добавить разделение по спикерам к заданию «Только распознавание»"
        )
        self._rediarize_button.setEnabled((job_dir / "asr" / "transcript.json").exists())

        self._rebuild_speaker_panel()
        self._render_transcript()
        self._update_save_state()
        self._load_audio()

        if read_error:
            self._transcript.setPlainText(f"Не удалось прочитать merged/segments.json: {read_error}")

    def _subtitle(self) -> str:
        config = self._meta.get("pipeline_config") or {}
        parts = []
        name = self._job_dir.name if self._job_dir else ""
        if self._segments:
            duration = max(float(seg.get("end") or 0) for seg in self._segments)
            parts.append(f"длительность {format_timestamp(duration)}")
        if config.get("language"):
            parts.append(f"язык {config['language']}")
        if config.get("asr_model"):
            parts.append(f"модель {config['asr_model']}")
        if self._has_diarization:
            parts.append(f"спикеров {len(unique_speakers(self._segments))}")
        else:
            parts.append("без разделения по спикерам")
        return " · ".join([name] + parts)

    def _rebuild_speaker_panel(self) -> None:
        while self._speakers_grid.count():
            item = self._speakers_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._speaker_fields = {}

        totals = speaking_time(self._segments)
        total = sum(totals.values()) or 1.0
        for row, speaker_id in enumerate(unique_speakers(self._segments)):
            play = QPushButton("▶")
            play.setFixedWidth(32)
            play.setToolTip("Послушать фрагмент речи этого спикера")
            play.clicked.connect(lambda checked=False, sid=speaker_id: self._play_sample(sid))
            play.setEnabled(False)  # _load_audio() enables it once a source is loaded

            id_label = QLabel(speaker_id)
            id_label.setMinimumWidth(90)

            field = QLineEdit(self._names.get(speaker_id, ""))
            field.setPlaceholderText(f"имя для {speaker_id}")
            field.textChanged.connect(self._on_names_edited)
            self._speaker_fields[speaker_id] = field

            seconds = totals.get(speaker_id, 0.0)
            stats = QLabel(f"{format_timestamp(seconds)} · {seconds / total:.0%}")
            stats.setStyleSheet("color: #7d8394;")
            stats.setToolTip("Сколько времени говорит спикер и его доля в разговоре")

            self._speakers_grid.addWidget(play, row, 0)
            self._speakers_grid.addWidget(id_label, row, 1)
            self._speakers_grid.addWidget(field, row, 2)
            self._speakers_grid.addWidget(stats, row, 3)
        self._speakers_grid.setColumnStretch(2, 1)

    def _current_names(self) -> dict:
        return {
            sid: field.text().strip()
            for sid, field in self._speaker_fields.items()
            if field.text().strip()
        }

    def _on_names_edited(self) -> None:
        self._render_timer.start()
        self._update_save_state()

    def _update_save_state(self) -> None:
        dirty = self._current_names() != self._names
        self._save_button.setEnabled(dirty and self._job_dir is not None)
        self._save_button.setText("Сохранить имена •" if dirty else "Сохранить имена")

    def _render_transcript(self) -> None:
        scroll = self._transcript.verticalScrollBar().value()
        self._transcript.setHtml(build_transcript_html(self._segments, self._current_names()))
        self._transcript.verticalScrollBar().setValue(scroll)
        if self._search_field.text():
            self._update_search(keep_position=True)

    # --- open / copy -------------------------------------------------------------

    def _fill_open_menu(self) -> None:
        self._open_menu.clear()
        if self._job_dir is None:
            return
        for label, rel_path in OPEN_TARGETS:
            path = self._job_dir / rel_path
            action = self._open_menu.addAction(label)
            if rel_path == "asr/transcript.txt":
                action.setEnabled((self._job_dir / "asr" / "transcript.json").exists())
                action.triggered.connect(self._open_plain_text)
            else:
                action.setEnabled(path.exists())
                action.triggered.connect(lambda checked=False, p=path: self._open_file(p))
        self._open_menu.addSeparator()
        audio = find_job_audio(self._job_dir)
        action = self._open_menu.addAction("Аудио задания")
        action.setEnabled(audio is not None)
        if audio is not None:
            action.triggered.connect(lambda checked=False, p=audio: self._open_file(p))
        original = Path(self._meta["input_path"]) if self._meta.get("input_path") else None
        action = self._open_menu.addAction("Исходный файл")
        action.setEnabled(original is not None and original.exists())
        if original is not None:
            action.triggered.connect(lambda checked=False, p=original: self._open_file(p))

    def _open_folder(self) -> None:
        if self._job_dir is not None:
            os.startfile(str(self._job_dir))

    def _open_plain_text(self) -> None:
        """Open asr/transcript.txt, building it from transcript.json first when
        the job predates that file (or it was deleted)."""
        if self._job_dir is None:
            return
        txt_path = self._job_dir / "asr" / "transcript.txt"
        if not txt_path.exists():
            data = _read_json(self._job_dir / "asr" / "transcript.json")
            try:
                txt_path.write_text(format_plain_transcript(data.get("segments", [])), encoding="utf-8")
            except OSError:
                return
        os.startfile(str(txt_path))

    def _open_file(self, path: Optional[Path]) -> None:
        if path is not None and path.exists():
            os.startfile(str(path))

    def _copy_transcript(self) -> None:
        if not self._segments:
            return
        QApplication.clipboard().setText(transcript_plain_text(self._segments, self._current_names()))
        self._set_status("Текст скопирован в буфер обмена.", "#2f7d52")

    # --- saving / re-running ---------------------------------------------------------

    def _save_names(self) -> None:
        if self._job_dir is None:
            return
        names = self._current_names()
        try:
            write_json_atomic(self._job_dir / "speakers.json", names)
            reexport(self._job_dir, names)
        except Exception as exc:  # show anything - the names stay on screen to retry
            QMessageBox.warning(self, "Не удалось сохранить", f"{type(exc).__name__}: {exc}")
            return
        self._names = names
        self._update_save_state()
        self._set_status("Имена сохранены, файлы result.md/.txt/.srt/.json обновлены.", "#2f7d52")

    def _rediarize(self) -> None:
        if self._job_dir is None:
            return
        if not hf_token_present():
            QMessageBox.warning(
                self,
                self._rediarize_button.text().rstrip("…"),
                "Для диаризации нужен HF-токен — задайте его на экране «Настройки».",
            )
            return
        dialog = RediarizeDialog(self, self._has_diarization, len(unique_speakers(self._segments)))
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        plan = rediarize_plan(self._job_dir, dialog.min_spin.value(), dialog.max_spin.value())
        if plan is None:
            QMessageBox.warning(self, dialog.windowTitle(), "У задания нет распознанного текста.")
            return
        mode, kwargs = plan
        # The old names belong to the old speaker ids.
        set_aside_speaker_names(self._job_dir)
        label =f"{self._title_label.text()} (спикеры {kwargs['min_speakers']}–{kwargs['max_speakers']})"
        self.runs_requested.emit([QueuedRun(mode=mode, kwargs=kwargs, label=label)])

    # --- search -----------------------------------------------------------------

    def _focus_search(self) -> None:
        self._search_field.setFocus()
        self._search_field.selectAll()

    def _update_search(self, keep_position: bool = False) -> None:
        needle = self._search_field.text()
        previous = self._match_index
        self._matches = []
        if needle:
            document = self._transcript.document()
            cursor = QTextCursor(document)
            while len(self._matches) < MAX_SEARCH_MATCHES:
                cursor = document.find(needle, cursor)
                if cursor.isNull():
                    break
                self._matches.append(QTextCursor(cursor))
        if not self._matches:
            self._match_index = -1
        elif keep_position and 0 <= previous < len(self._matches):
            self._match_index = previous
        else:
            self._match_index = 0
        self._show_matches(scroll=not keep_position)

    def _step_match(self, step: int) -> None:
        if not self._matches:
            self._update_search()
            return
        self._match_index = (self._match_index + step) % len(self._matches)
        self._show_matches(scroll=True)

    def _show_matches(self, scroll: bool) -> None:
        selections = []
        for index, cursor in enumerate(self._matches):
            selection = QTextEdit.ExtraSelection()
            selection.cursor = cursor
            selection.format.setBackground(QColor("#f5a623" if index == self._match_index else "#ffe58a"))
            selections.append(selection)
        self._transcript.setExtraSelections(selections)
        if self._search_field.text():
            self._match_label.setText(
                f"{self._match_index + 1} из {len(self._matches)}" if self._matches else "не найдено"
            )
        else:
            self._match_label.setText("")
        if scroll and self._match_index >= 0:
            cursor = QTextCursor(self._matches[self._match_index])
            cursor.clearSelection()
            self._transcript.setTextCursor(cursor)
            self._transcript.ensureCursorVisible()

    # --- playback -----------------------------------------------------------------

    def _ensure_player(self) -> bool:
        if self._player is not None:
            return True
        try:
            from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
        except ImportError:
            return False
        self._audio_output = QAudioOutput(self)
        self._player = QMediaPlayer(self)
        self._player.setAudioOutput(self._audio_output)
        self._player.positionChanged.connect(self._on_position)
        self._player.durationChanged.connect(self._on_duration)
        self._player.playbackStateChanged.connect(self._on_playback_state)
        self._player.errorOccurred.connect(self._on_player_error)
        return True

    def _audio_source(self) -> Optional[Path]:
        if self._job_dir is None:
            return None
        audio = find_job_audio(self._job_dir)
        if audio is not None:
            return audio
        original = self._meta.get("input_path")
        if original and Path(original).exists():
            return Path(original)
        return None

    def _load_audio(self) -> None:
        source = self._audio_source()
        if source is None:
            self._player_bar.hide()
            self._player_note.setText("Аудио задания не найдено — воспроизведение недоступно.")
            self._player_note.show()
            self._set_sample_buttons_enabled(False)
            return
        if not self._ensure_player():
            self._player_bar.hide()
            self._player_note.setText("Воспроизведение недоступно: в установке PySide6 нет QtMultimedia.")
            self._player_note.show()
            self._set_sample_buttons_enabled(False)
            return
        self._player.stop()
        self._stop_at_ms = None
        self._player.setSource(QUrl.fromLocalFile(str(source)))
        self._player_note.hide()
        self._player_bar.show()
        self._set_sample_buttons_enabled(True)

    def _set_sample_buttons_enabled(self, enabled: bool) -> None:
        for row in range(self._speakers_grid.rowCount()):
            item = self._speakers_grid.itemAtPosition(row, 0)
            if item is not None and item.widget() is not None:
                item.widget().setEnabled(enabled)

    def _play_from(self, seconds: float, until: Optional[float] = None) -> None:
        if self._player is None or self._player_bar.isHidden():
            return
        self._player.setPosition(int(seconds * 1000))
        self._stop_at_ms = int(until * 1000) if until is not None else None
        self._player.play()

    def _play_sample(self, speaker_id: str) -> None:
        sample = sample_range(self._segments, speaker_id)
        if sample is not None:
            self._play_from(*sample)

    def _on_anchor(self, url: QUrl) -> None:
        target = url.toString()
        if target.startswith("t:"):
            try:
                self._play_from(float(target[2:]))
            except ValueError:
                pass

    def _toggle_play(self) -> None:
        if self._player is None:
            return
        from PySide6.QtMultimedia import QMediaPlayer

        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
        else:
            self._stop_at_ms = None
            self._player.play()

    def _on_position(self, position: int) -> None:
        if self._stop_at_ms is not None and position >= self._stop_at_ms:
            self._stop_at_ms = None
            self._player.pause()
        if not self._slider_dragging:
            self._slider.setValue(position)
        duration = self._player.duration() if self._player is not None else 0
        self._position_label.setText(
            f"{format_timestamp(position / 1000)} / {format_timestamp(duration / 1000)}"
        )

    def _on_duration(self, duration: int) -> None:
        self._slider.setRange(0, max(0, duration))
        self._on_position(self._player.position())

    def _on_playback_state(self, state) -> None:
        from PySide6.QtMultimedia import QMediaPlayer

        self._play_button.setText("⏸" if state == QMediaPlayer.PlaybackState.PlayingState else "▶")

    def _on_player_error(self, _error, message: str) -> None:
        self._player_note.setText(f"Не удалось воспроизвести аудио: {message}")
        self._player_note.show()

    def _on_slider_pressed(self) -> None:
        self._slider_dragging = True

    def _on_slider_released(self) -> None:
        self._slider_dragging = False
        if self._player is not None:
            self._stop_at_ms = None
            self._player.setPosition(self._slider.value())

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        if self._player is None:
            return
        if event.spontaneous():
            # The window was minimized - keep the position, just stop the sound.
            self._player.pause()
            return
        # Left the screen: release the file, or Windows would keep it locked
        # and deleting this job from History would fail. load() sets it again.
        self._player.stop()
        self._player.setSource(QUrl())
        self._player_bar.hide()
        self._set_sample_buttons_enabled(False)

    # --- misc -----------------------------------------------------------------------

    def _set_status(self, text: str, color: str) -> None:
        self._status_label.setText(text)
        self._status_label.setStyleSheet(f"color: {color};")
