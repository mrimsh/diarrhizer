"""New Job screen: pick one or more media files and the parameters to run
them with. Every file becomes its own job; MainWindow runs them one after
another (GUI:JOB_QUEUE). Resuming a job and partial re-runs live on the
History and Result screens; audio storage formats are edited on Settings.

The parameters of the last started job are remembered (QSettings, under
settings_keys.NEW_JOB_PREFIX), so a typical "same settings, new file" run
is just: add files -> Run. Output folder, device and model start from the
Settings defaults.

Imports nothing heavy: CUDA availability comes from the shared
EnvironmentProbe (checked in a child process), not from importing torch here.
"""

import os
from pathlib import Path

from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from diarrhizer.audio_formats import AudioFormat, AudioFormatStore, store_path
from diarrhizer.diagnostics import models as model_cache
from diarrhizer.media import MEDIA_EXTENSIONS, expand_media_paths, is_media_file
from diarrhizer_gui import custom_models, settings_keys
from diarrhizer_gui.job_queue import QueuedRun

MEDIA_FILTER = (
    "Медиафайлы (" + " ".join(f"*{ext}" for ext in MEDIA_EXTENSIONS) + ");;Все файлы (*.*)"
)

# Built-in faster-whisper presets. Models the user warmed up themselves on the
# Models screen are appended at show time - see _reload_asr_models().
ASR_MODELS = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]

MODES = [
    ("Полный цикл (распознавание + спикеры)", "full"),
    ("Только распознавание (без диаризации)", "asr_only"),
]

# Codes Whisper understands; the field stays editable for any other code.
LANGUAGES = [
    "auto", "ru", "en", "uk", "be", "kk", "uz", "az", "hy", "ka",
    "de", "fr", "es", "it", "pt", "pl", "cs", "tr", "zh", "ja", "ko", "ar", "he",
]

AUDIO_PROFILES = [
    ("raw", "Без обработки (по умолчанию)"),
    ("voice-call", "Телефонный звонок — полосовой фильтр 300 Гц–7 кГц"),
    ("denoise-light", "Лёгкое шумоподавление"),
    ("split-stereo", "Доп. L/R-файлы (диаризация всё равно по моно-миксу)"),
]

# Russian labels for the built-in storage formats (diarrhizer.audio_formats);
# user profiles show their own name and description.
AUDIO_FORMAT_LABELS = {
    "wav": "WAV 16 кГц моно — без сжатия, только рабочий файл (≈115 МБ/ч)",
    "flac": "FLAC — сжатие без потерь (≈50–75 МБ/ч)",
    "mp3-q5": "MP3 VBR -q:a 5 (≈15–20 МБ/ч)",
    "opus-24k": "Opus 24 кбит/с (≈11 МБ/ч)",
}

COMPUTE_TYPES = [
    ("авто", None),
    ("float16", "float16"),
    ("int8_float16", "int8_float16"),
    ("int8", "int8"),
]

# Advanced ASR/merge defaults - the same values as PipelineConfig's.
ADVANCED_DEFAULTS = {
    "compute_type": "",
    "beam_size": 5,
    "temperature": 0.0,
    "condition": True,
    "vad": True,
    "vad_silence_ms": 1000,
    "min_turn": 0.4,
}


def audio_format_label(fmt: AudioFormat) -> str:
    if fmt.name in AUDIO_FORMAT_LABELS:
        return AUDIO_FORMAT_LABELS[fmt.name]
    return f"{fmt.name} — {fmt.description}" if fmt.description else fmt.name


def audio_format_hint(fmt: AudioFormat) -> str:
    """One line on what ends up on disk and roughly how big it is."""
    estimate = fmt.estimated_mb_per_hour()
    size = f"≈{estimate:.0f} МБ/ч" if estimate is not None else "размер зависит от записи"
    if not fmt.writes_archive:
        return f"На диске остаётся audio/normalized.wav · {size}"
    return (
        f"Архив audio/archive.{fmt.extension} · {fmt.codec}, {fmt.sample_rate} Гц, "
        f"{fmt.channels} кан. · {size}"
    )


def load_audio_format_store() -> tuple[AudioFormatStore, str]:
    """The profile store, or built-ins only plus an error message if the file is broken."""
    try:
        return AudioFormatStore.load(), ""
    except ValueError as e:
        return AudioFormatStore(path=store_path()), f"Профили аудио не загружены: {e}"


def available_asr_models(settings: QSettings) -> list:
    """Presets, models added on the Models screen and ones found in the HF
    cache - what every model dropdown offers."""
    return custom_models.asr_model_choices(
        settings,
        ASR_MODELS,
        custom_models.detect_cached_asr_models(model_cache.list_cached_models(), ASR_MODELS),
    )


def plural(count: int, one: str, few: str, many: str) -> str:
    """Russian plural form: 1 файл, 2 файла, 5 файлов."""
    if count % 10 == 1 and count % 100 != 11:
        return one
    if 2 <= count % 10 <= 4 and not 12 <= count % 100 <= 14:
        return few
    return many


def hf_token_present() -> bool:
    return bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))


class NewJobScreen(QWidget):
    runs_requested = Signal(list)  # [QueuedRun]

    def __init__(self, probe=None) -> None:
        super().__init__()
        self._settings = QSettings("Diarrhizer", "DiarrhizerGUI")
        self._probe = probe
        self._busy = False

        title = QLabel("Новое задание")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")

        # --- files ---
        self._files = QListWidget()
        self._files.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._files.setMinimumHeight(90)
        self._files.setMaximumHeight(160)
        self._files.model().rowsInserted.connect(self._validate)
        self._files.model().rowsRemoved.connect(self._validate)
        QShortcut(QKeySequence.StandardKey.Delete, self._files, self._remove_selected,
                  context=Qt.ShortcutContext.WidgetShortcut)

        add_files = QPushButton("Добавить файлы…")
        add_files.clicked.connect(self._browse_files)
        add_folder = QPushButton("Добавить папку…")
        add_folder.setToolTip("Все медиафайлы из выбранной папки (без вложенных папок)")
        add_folder.clicked.connect(self._browse_folder)
        self._remove_button = QPushButton("Убрать")
        self._remove_button.clicked.connect(self._remove_selected)
        self._clear_button = QPushButton("Очистить")
        self._clear_button.clicked.connect(self._files.clear)
        files_buttons = QHBoxLayout()
        files_buttons.addWidget(add_files)
        files_buttons.addWidget(add_folder)
        files_buttons.addStretch(1)
        files_buttons.addWidget(self._remove_button)
        files_buttons.addWidget(self._clear_button)

        files_hint = QLabel(
            "Файлы и папки можно перетащить прямо в окно. Каждый файл — отдельное задание; "
            "они обрабатываются по очереди с одинаковыми параметрами."
        )
        files_hint.setWordWrap(True)
        files_hint.setStyleSheet("color: #7d8394;")

        files_box = QVBoxLayout()
        files_box.setSpacing(4)
        files_box.addWidget(self._files)
        files_box.addLayout(files_buttons)
        files_box.addWidget(files_hint)

        self._out_field = QLineEdit()
        self._out_field.setReadOnly(True)
        out_browse = QPushButton("Обзор…")
        out_browse.clicked.connect(self._browse_out)
        out_row = QHBoxLayout()
        out_row.addWidget(self._out_field, stretch=1)
        out_row.addWidget(out_browse)

        self._mode_combo = QComboBox()
        for label, key in MODES:
            self._mode_combo.addItem(label, key)
        self._mode_combo.currentIndexChanged.connect(self._validate)

        self._language_combo = QComboBox()
        self._language_combo.setEditable(True)
        self._language_combo.addItems(LANGUAGES)
        self._language_combo.setToolTip(
            "auto — определить автоматически по началу записи.\n"
            "Если язык известен, лучше указать его: так надёжнее на коротких записях."
        )

        self._min_speakers = QSpinBox()
        self._min_speakers.setRange(1, 20)
        self._max_speakers = QSpinBox()
        self._max_speakers.setRange(1, 20)
        self._min_speakers.valueChanged.connect(self._validate)
        self._max_speakers.valueChanged.connect(self._validate)
        speakers_row = QHBoxLayout()
        speakers_row.addWidget(self._min_speakers)
        speakers_row.addWidget(QLabel("–"))
        speakers_row.addWidget(self._max_speakers)
        speakers_hint = QLabel("если число участников известно, укажите его в обоих полях")
        speakers_hint.setStyleSheet("color: #7d8394;")
        speakers_row.addWidget(speakers_hint)
        speakers_row.addStretch(1)

        self._model_combo = QComboBox()
        self._model_combo.setEditable(True)
        self._model_combo.addItems(ASR_MODELS)
        self._model_combo.setToolTip(
            "Больше модель — точнее, но медленнее и требует больше видеопамяти.\n"
            "large-v3 — лучшее качество; small/medium — быстрее, особенно на CPU."
        )

        self._device_combo = QComboBox()
        self._device_combo.addItem("cuda")
        self._device_combo.addItem("cpu")

        self._audio_profile_combo = QComboBox()
        for value, label in AUDIO_PROFILES:
            self._audio_profile_combo.addItem(label, value)

        # Storage format: the pipeline always processes the lossless working
        # WAV; this picks the archive copy kept next to it and whether the WAV
        # itself survives the job.
        self._audio_formats: dict = {}
        self._audio_format_combo = QComboBox()
        self._audio_format_combo.currentIndexChanged.connect(self._on_audio_format_changed)
        self._keep_wav_checkbox = QCheckBox("Оставить рабочий WAV")
        self._keep_wav_checkbox.setToolTip(
            "Рабочий audio/normalized.wav (без сжатия) нужен только на время обработки.\n"
            "Если его не оставлять, после завершения задания на диске остаётся\n"
            "только архивная копия в выбранном формате."
        )
        audio_format_row = QHBoxLayout()
        audio_format_row.addWidget(self._audio_format_combo, stretch=1)
        audio_format_row.addWidget(self._keep_wav_checkbox)
        self._audio_format_hint = QLabel()
        self._audio_format_hint.setStyleSheet("color: #7d8394;")
        self._audio_format_hint.setWordWrap(True)

        form = QFormLayout()
        form.addRow("Файлы:", files_box)
        form.addRow("Папка результатов:", out_row)
        form.addRow("Режим:", self._mode_combo)
        form.addRow("Язык:", self._language_combo)
        form.addRow("Спикеры (мин–макс):", speakers_row)
        form.addRow("Модель ASR:", self._model_combo)
        form.addRow("Устройство:", self._device_combo)
        form.addRow("Предобработка звука:", self._audio_profile_combo)
        form.addRow("Хранение аудио:", audio_format_row)
        form.addRow("", self._audio_format_hint)

        self._advanced_toggle = QPushButton("▸ Продвинутые параметры")
        self._advanced_toggle.setCheckable(True)
        self._advanced_toggle.setStyleSheet(
            "QPushButton { text-align: left; border: none; font-weight: 600; padding: 4px 0; }"
        )
        self._advanced_toggle.toggled.connect(self._toggle_advanced)

        self._compute_type_combo = QComboBox()
        for label, value in COMPUTE_TYPES:
            self._compute_type_combo.addItem(label, value)

        self._beam_size_spin = QSpinBox()
        self._beam_size_spin.setRange(1, 20)

        self._temperature_spin = QDoubleSpinBox()
        self._temperature_spin.setRange(0.0, 1.0)
        self._temperature_spin.setSingleStep(0.1)
        self._temperature_spin.setDecimals(1)

        self._condition_checkbox = QCheckBox("Учитывать предыдущий текст (стабильнее декодирование)")
        self._vad_checkbox = QCheckBox("Включить VAD-фильтр (отсекать тишину)")

        self._vad_silence_spin = QSpinBox()
        self._vad_silence_spin.setRange(0, 10000)
        self._vad_silence_spin.setSingleStep(100)
        self._vad_silence_spin.setSuffix(" мс")

        self._min_turn_spin = QDoubleSpinBox()
        self._min_turn_spin.setRange(0.0, 5.0)
        self._min_turn_spin.setSingleStep(0.1)
        self._min_turn_spin.setDecimals(2)
        self._min_turn_spin.setSuffix(" с")
        self._min_turn_spin.setToolTip(
            "Сегмент ASR, задевающий смену говорящего, режется на реплики.\n"
            "Отрезки короче этого порога, зажатые между двумя отрезками\n"
            "одного и того же другого спикера, считаются дрожанием\n"
            "диаризации и приклеиваются обратно. 0 — резать на каждой смене."
        )

        self._prompt_edit = QPlainTextEdit()
        self._prompt_edit.setPlaceholderText(
            "Глоссарий/начальный промпт (необязательно) — термины, имена, "
            "аббревиатуры для более точного распознавания. Запоминается между запусками."
        )
        self._prompt_edit.setFixedHeight(70)

        reset_advanced = QPushButton("Сбросить к значениям по умолчанию")
        reset_advanced.setToolTip("Глоссарий не трогается")
        reset_advanced.clicked.connect(self._reset_advanced)

        advanced_form = QFormLayout()
        advanced_form.addRow("Compute type:", self._compute_type_combo)
        advanced_form.addRow("Beam size:", self._beam_size_spin)
        advanced_form.addRow("Temperature:", self._temperature_spin)
        advanced_form.addRow(self._condition_checkbox)
        advanced_form.addRow(self._vad_checkbox)
        advanced_form.addRow("Мин. тишина для VAD:", self._vad_silence_spin)
        advanced_form.addRow("Промпт/глоссарий:", self._prompt_edit)
        # Everything above is ASR; this one belongs to the merge stage, so it
        # gets its own heading rather than reading as another Whisper knob.
        diarization_header = QLabel("Диаризация")
        diarization_header.setStyleSheet("font-weight: 600; padding-top: 8px;")
        advanced_form.addRow(diarization_header)
        advanced_form.addRow("Мин. длительность реплики:", self._min_turn_spin)
        advanced_form.addRow(reset_advanced)

        self._advanced_container = QWidget()
        self._advanced_container.setLayout(advanced_form)
        self._advanced_container.setVisible(False)

        self._warning_label = QLabel()
        self._warning_label.setStyleSheet("color: #b23b35;")
        self._warning_label.setWordWrap(True)
        self._warning_label.hide()

        self._run_button = QPushButton("Запустить")
        self._run_button.setMinimumHeight(32)
        self._run_button.setStyleSheet("font-weight: 600;")
        self._run_button.clicked.connect(self._emit_run_requested)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.addWidget(title)
        layout.addLayout(form)
        layout.addWidget(self._advanced_toggle)
        layout.addWidget(self._advanced_container)
        layout.addWidget(self._warning_label)
        layout.addWidget(self._run_button)
        layout.addStretch(1)

        # The advanced section makes the screen taller than a small window.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(content)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        self._restore_last_used()
        self.apply_defaults()
        if self._probe is not None:
            self._probe.finished.connect(self._on_probe_finished)
            if self._probe.results is not None:
                self._on_probe_finished(self._probe.results)
        self._validate()

    # --- public API ----------------------------------------------------------

    def apply_defaults(self) -> None:
        """Output folder, model and device from the Settings defaults."""
        default_out = str(Path.cwd() / "out")
        self._out_field.setText(self._settings.value(settings_keys.OUT_DIR, default_out))

        self._reload_asr_models()
        default_model = self._settings.value(settings_keys.DEFAULT_ASR_MODEL, "") or "large-v3"
        self._model_combo.setCurrentText(default_model)

        default_device = self._settings.value(settings_keys.DEFAULT_DEVICE, "")
        if not default_device:
            cuda = self._probe.cuda_available if self._probe is not None else None
            default_device = "cpu" if cuda is False else "cuda"
        self._device_combo.setCurrentText(default_device)
        self._enforce_device()

        self._reload_audio_formats()

    def add_paths(self, paths, only_media: bool = True) -> int:
        """Add files (folders expand to their media files); returns how many were new."""
        existing = {self._files.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self._files.count())}
        added = 0
        for path in expand_media_paths(paths):
            if not path.is_file() or (only_media and not is_media_file(path)):
                continue
            key = str(path)
            if key in existing:
                continue
            existing.add(key)
            entry = QListWidgetItem(f"{path.name}    ({path.parent})")
            entry.setData(Qt.ItemDataRole.UserRole, key)
            entry.setToolTip(key)
            self._files.addItem(entry)
            added += 1
        return added

    def set_busy(self, busy: bool) -> None:
        """While a batch runs, new jobs go to the end of the queue."""
        self._busy = busy
        self._update_run_button()

    # --- Qt events -----------------------------------------------------------

    def showEvent(self, event) -> None:
        super().showEvent(event)
        # Lists may have changed elsewhere (Models/Settings); keep the choices.
        self._reload_asr_models()
        self._reload_audio_formats()
        self._validate()

    # --- persistence ---------------------------------------------------------

    def _key(self, name: str) -> str:
        return settings_keys.NEW_JOB_PREFIX + name

    def _restore_last_used(self) -> None:
        value = self._settings.value
        mode = value(self._key("mode"), "full", type=str)
        self._mode_combo.setCurrentIndex(max(0, self._mode_combo.findData(mode)))
        self._language_combo.setCurrentText(value(self._key("language"), "auto", type=str) or "auto")
        self._min_speakers.setValue(value(self._key("min_speakers"), 1, type=int))
        self._max_speakers.setValue(value(self._key("max_speakers"), 10, type=int))
        profile = value(self._key("audio_profile"), "raw", type=str)
        self._audio_profile_combo.setCurrentIndex(max(0, self._audio_profile_combo.findData(profile)))
        self._prompt_edit.setPlainText(value(self._key("prompt"), "", type=str))

        defaults = ADVANCED_DEFAULTS
        self._set_compute_type(value(self._key("compute_type"), defaults["compute_type"], type=str))
        self._beam_size_spin.setValue(value(self._key("beam_size"), defaults["beam_size"], type=int))
        self._temperature_spin.setValue(value(self._key("temperature"), defaults["temperature"], type=float))
        self._condition_checkbox.setChecked(value(self._key("condition"), defaults["condition"], type=bool))
        self._vad_checkbox.setChecked(value(self._key("vad"), defaults["vad"], type=bool))
        self._vad_silence_spin.setValue(
            value(self._key("vad_silence_ms"), defaults["vad_silence_ms"], type=int)
        )
        self._min_turn_spin.setValue(value(self._key("min_turn"), defaults["min_turn"], type=float))

    def _remember(self) -> None:
        set_value = self._settings.setValue
        set_value(self._key("mode"), self._mode_combo.currentData())
        set_value(self._key("language"), self._language_combo.currentText().strip())
        set_value(self._key("min_speakers"), self._min_speakers.value())
        set_value(self._key("max_speakers"), self._max_speakers.value())
        set_value(self._key("audio_profile"), self._audio_profile_combo.currentData())
        set_value(self._key("prompt"), self._prompt_edit.toPlainText())
        set_value(self._key("compute_type"), self._compute_type_combo.currentData() or "")
        set_value(self._key("beam_size"), self._beam_size_spin.value())
        set_value(self._key("temperature"), self._temperature_spin.value())
        set_value(self._key("condition"), self._condition_checkbox.isChecked())
        set_value(self._key("vad"), self._vad_checkbox.isChecked())
        set_value(self._key("vad_silence_ms"), self._vad_silence_spin.value())
        set_value(self._key("min_turn"), self._min_turn_spin.value())

    def _reset_advanced(self) -> None:
        defaults = ADVANCED_DEFAULTS
        self._set_compute_type(defaults["compute_type"])
        self._beam_size_spin.setValue(defaults["beam_size"])
        self._temperature_spin.setValue(defaults["temperature"])
        self._condition_checkbox.setChecked(defaults["condition"])
        self._vad_checkbox.setChecked(defaults["vad"])
        self._vad_silence_spin.setValue(defaults["vad_silence_ms"])
        self._min_turn_spin.setValue(defaults["min_turn"])

    def _set_compute_type(self, stored: str) -> None:
        for index in range(self._compute_type_combo.count()):
            if (self._compute_type_combo.itemData(index) or "") == (stored or ""):
                self._compute_type_combo.setCurrentIndex(index)
                return

    # --- environment -----------------------------------------------------------

    def _on_probe_finished(self, _results) -> None:
        self._enforce_device()

    def _enforce_device(self) -> None:
        """Disable cuda once the probe says it is unavailable."""
        if self._probe is None or self._probe.cuda_available is None:
            return
        cuda_ok = bool(self._probe.cuda_available)
        cuda_item = self._device_combo.model().item(0)
        cuda_item.setEnabled(cuda_ok)
        cuda_item.setToolTip("" if cuda_ok else self._probe.cuda_message)
        if not cuda_ok and self._device_combo.currentText() == "cuda":
            self._device_combo.setCurrentText("cpu")

    # --- lists -----------------------------------------------------------------

    def _reload_audio_formats(self) -> None:
        """Rebuild the storage-format list from audio_formats.json (profiles may
        have been edited on the Settings screen); keep the current choice if it
        still exists, else select the default profile.
        """
        current = self._audio_format_combo.currentData()
        store, error = load_audio_format_store()
        self._audio_formats = store.formats()
        self._audio_format_combo.blockSignals(True)
        self._audio_format_combo.clear()
        for name, fmt in self._audio_formats.items():
            self._audio_format_combo.addItem(audio_format_label(fmt), name)
        wanted = current if current in self._audio_formats else store.default().name
        self._audio_format_combo.setCurrentIndex(max(0, self._audio_format_combo.findData(wanted)))
        self._audio_format_combo.blockSignals(False)
        if current != wanted or current is None:
            self._on_audio_format_changed()
        if error:
            self._audio_format_hint.setText(error)
            self._audio_format_hint.setStyleSheet("color: #b23b35;")

    def _on_audio_format_changed(self) -> None:
        fmt = self._audio_formats.get(self._audio_format_combo.currentData())
        if fmt is None:
            return
        # A WAV-only format has nothing else to keep, so the WAV always stays.
        self._keep_wav_checkbox.setEnabled(fmt.writes_archive)
        self._keep_wav_checkbox.setChecked(fmt.keeps_wav)
        self._audio_format_hint.setText(audio_format_hint(fmt))
        self._audio_format_hint.setStyleSheet("color: #7d8394;")

    def _reload_asr_models(self) -> None:
        """Rebuild the model dropdown from presets + models added on the Models
        screen, so a model warmed up mid-session shows up without a restart.
        """
        choices = available_asr_models(self._settings)
        if [self._model_combo.itemText(i) for i in range(self._model_combo.count())] == choices:
            return
        # clear() wipes the edit line too, so restore whatever was typed there.
        current = self._model_combo.currentText()
        self._model_combo.clear()
        self._model_combo.addItems(choices)
        self._model_combo.setCurrentText(current)

    # --- actions ---------------------------------------------------------------

    def _toggle_advanced(self, checked: bool) -> None:
        self._advanced_container.setVisible(checked)
        arrow = "▾" if checked else "▸"
        self._advanced_toggle.setText(f"{arrow} Продвинутые параметры")

    def _browse_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Медиафайлы", self._last_dir(), MEDIA_FILTER)
        if paths:
            self._settings.setValue(self._key("last_dir"), str(Path(paths[0]).parent))
            # Picked explicitly in the dialog: accept unusual extensions too.
            self.add_paths(paths, only_media=False)

    def _browse_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Папка с записями", self._last_dir())
        if folder:
            self._settings.setValue(self._key("last_dir"), folder)
            if not self.add_paths([folder]):
                self._show_problems(["в папке нет медиафайлов"])

    def _last_dir(self) -> str:
        return self._settings.value(self._key("last_dir"), "", type=str)

    def _remove_selected(self) -> None:
        for entry in self._files.selectedItems():
            self._files.takeItem(self._files.row(entry))

    def _browse_out(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Папка результатов", self._out_field.text())
        if chosen:
            self._out_field.setText(chosen)
            self._settings.setValue(settings_keys.OUT_DIR, chosen)

    def _selected_files(self) -> list:
        return [self._files.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self._files.count())]

    def _validate(self, *_args) -> bool:
        problems = []
        if not self._files.count():
            problems.append("добавьте хотя бы один медиафайл")
        if self._min_speakers.value() > self._max_speakers.value():
            problems.append("минимум спикеров не может быть больше максимума")
        if self._mode_combo.currentData() == "full" and not hf_token_present():
            problems.append(
                "для полного цикла нужен HF-токен (экран «Настройки») — "
                "или выберите режим «Только распознавание»"
            )
        self._show_problems(problems)
        self._update_run_button()
        return not problems

    def _show_problems(self, problems: list) -> None:
        if problems:
            self._warning_label.setText(" · ".join(problems))
            self._warning_label.show()
            self._run_button.setEnabled(False)
        else:
            self._warning_label.hide()
            self._run_button.setEnabled(True)

    def _update_run_button(self) -> None:
        count = self._files.count()
        if self._busy:
            text = "Добавить в очередь"
        else:
            text = "Запустить"
        if count > 1:
            text += f" ({count} {plural(count, 'файл', 'файла', 'файлов')})"
        self._run_button.setText(text)

    def _emit_run_requested(self) -> None:
        if not self._validate():
            return

        mode = self._mode_combo.currentData()
        common = {
            "out_dir": self._out_field.text(),
            "min_speakers": self._min_speakers.value(),
            "max_speakers": self._max_speakers.value(),
            "language": self._language_combo.currentText().strip() or "auto",
            "device": self._device_combo.currentText(),
            "asr_model": self._model_combo.currentText().strip(),
            "audio_profile": self._audio_profile_combo.currentData(),
            # Pass the resolved profile, not its name: the job must use what
            # was on screen even if audio_formats.json changes before it runs.
            "audio_format": self._audio_formats[self._audio_format_combo.currentData()],
            "keep_wav": self._keep_wav_checkbox.isChecked(),
            "asr_compute_type": self._compute_type_combo.currentData(),
            "asr_beam_size": self._beam_size_spin.value(),
            "asr_temperature": self._temperature_spin.value(),
            "asr_condition_on_previous_text": self._condition_checkbox.isChecked(),
            "asr_vad_filter": self._vad_checkbox.isChecked(),
            "asr_vad_min_silence_ms": self._vad_silence_spin.value(),
            "min_turn_duration": self._min_turn_spin.value(),
        }
        prompt = self._prompt_edit.toPlainText().strip()
        if prompt:
            common["asr_initial_prompt"] = prompt

        runs = [
            QueuedRun(mode=mode, kwargs={**common, "input_path": path}, label=Path(path).name)
            for path in self._selected_files()
        ]
        self._remember()
        self._files.clear()
        self.runs_requested.emit(runs)
