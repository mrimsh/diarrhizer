"""New Job screen: a deliberately small subset of the full CLI parameter
surface (see the GUI Blueprint) - just enough to start a brand-new job in
one of two modes. Resuming a job and force/from-stage/to-stage are later
passes; advanced ASR params, audio profiles and audio storage formats are
covered here (formats themselves are edited on the Settings screen).

Only imports diarrhizer.diagnostics.doctor at module level (light, no torch
at import time - same as the Doctor screen). Anything from
diarrhizer.pipeline/diarrhizer.adapters stays inside pipeline_worker.py,
imported lazily when a run actually starts.
"""

from pathlib import Path

from PySide6.QtCore import QSettings, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from diarrhizer.audio_formats import AudioFormat, AudioFormatStore, store_path
from diarrhizer.diagnostics import doctor
from diarrhizer_gui import custom_models, settings_keys

MEDIA_FILTER = "Медиафайлы (*.mp3 *.wav *.m4a *.mp4 *.mkv *.webm);;Все файлы (*.*)"

# Built-in faster-whisper presets. Models the user warmed up themselves on the
# Models screen are appended at show time - see _reload_asr_models().
ASR_MODELS = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]

MODES = [
    ("Полный цикл", "full"),
    ("Только распознавание (без диаризации)", "asr_only"),
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


COMPUTE_TYPES = [
    ("авто", None),
    ("float16", "float16"),
    ("int8_float16", "int8_float16"),
    ("int8", "int8"),
]


class NewJobScreen(QWidget):
    run_requested = Signal(str, dict)  # mode, run_pipeline() kwargs

    def __init__(self) -> None:
        super().__init__()
        self._settings = QSettings("Diarrhizer", "DiarrhizerGUI")

        title = QLabel("Новое задание")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")

        self._input_field = QLineEdit()
        self._input_field.setReadOnly(True)
        input_browse = QPushButton("Обзор…")
        input_browse.clicked.connect(self._browse_input)
        input_row = QHBoxLayout()
        input_row.addWidget(self._input_field, stretch=1)
        input_row.addWidget(input_browse)

        self._out_field = QLineEdit()
        self._out_field.setReadOnly(True)
        out_browse = QPushButton("Обзор…")
        out_browse.clicked.connect(self._browse_out)
        out_row = QHBoxLayout()
        out_row.addWidget(self._out_field, stretch=1)
        out_row.addWidget(out_browse)

        self._mode_combo = QComboBox()
        for label, _key in MODES:
            self._mode_combo.addItem(label)

        self._language_combo = QComboBox()
        self._language_combo.setEditable(True)
        self._language_combo.addItems(["auto", "ru", "en"])

        self._min_speakers = QSpinBox()
        self._min_speakers.setRange(1, 20)
        self._min_speakers.setValue(1)
        self._max_speakers = QSpinBox()
        self._max_speakers.setRange(1, 20)
        self._max_speakers.setValue(10)
        self._min_speakers.valueChanged.connect(self._validate)
        self._max_speakers.valueChanged.connect(self._validate)
        speakers_row = QHBoxLayout()
        speakers_row.addWidget(self._min_speakers)
        speakers_row.addWidget(QLabel("–"))
        speakers_row.addWidget(self._max_speakers)
        speakers_row.addStretch(1)

        self._model_combo = QComboBox()
        self._model_combo.setEditable(True)
        self._model_combo.addItems(ASR_MODELS)

        self._device_combo = QComboBox()
        self._device_combo.addItem("cuda")
        self._device_combo.addItem("cpu")
        _, cuda_ok, cuda_message = doctor.check_cuda()
        if not cuda_ok:
            cuda_item = self._device_combo.model().item(0)
            cuda_item.setEnabled(False)
            cuda_item.setToolTip(cuda_message)
        self._cuda_available = cuda_ok

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
        form.addRow("Файл:", input_row)
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
        self._beam_size_spin.setValue(5)

        self._temperature_spin = QDoubleSpinBox()
        self._temperature_spin.setRange(0.0, 1.0)
        self._temperature_spin.setSingleStep(0.1)
        self._temperature_spin.setDecimals(1)
        self._temperature_spin.setValue(0.0)

        self._condition_checkbox = QCheckBox("Учитывать предыдущий текст (стабильнее декодирование)")
        self._condition_checkbox.setChecked(True)

        self._vad_checkbox = QCheckBox("Включить VAD-фильтр (отсекать тишину)")
        self._vad_checkbox.setChecked(True)

        self._vad_silence_spin = QSpinBox()
        self._vad_silence_spin.setRange(0, 10000)
        self._vad_silence_spin.setSingleStep(100)
        self._vad_silence_spin.setValue(1000)
        self._vad_silence_spin.setSuffix(" мс")

        self._min_turn_spin = QDoubleSpinBox()
        self._min_turn_spin.setRange(0.0, 5.0)
        self._min_turn_spin.setSingleStep(0.1)
        self._min_turn_spin.setDecimals(2)
        self._min_turn_spin.setValue(0.4)
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
            "аббревиатуры для более точного распознавания"
        )
        self._prompt_edit.setFixedHeight(70)

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

        self._advanced_container = QWidget()
        self._advanced_container.setLayout(advanced_form)
        self._advanced_container.setVisible(False)

        self._warning_label = QLabel()
        self._warning_label.setStyleSheet("color: #b23b35;")
        self._warning_label.hide()

        self._run_button = QPushButton("Запустить")
        self._run_button.clicked.connect(self._emit_run_requested)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.addWidget(title)
        layout.addLayout(form)
        layout.addWidget(self._advanced_toggle)
        layout.addWidget(self._advanced_container)
        layout.addWidget(self._warning_label)
        layout.addWidget(self._run_button)
        layout.addStretch(1)

        self._apply_defaults()
        self._validate()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._apply_defaults()

    def _apply_defaults(self) -> None:
        default_out = str(Path.cwd() / "out")
        self._out_field.setText(self._settings.value(settings_keys.OUT_DIR, default_out))

        self._reload_asr_models()
        default_model = self._settings.value(settings_keys.DEFAULT_ASR_MODEL, "") or "large-v3"
        self._model_combo.setCurrentText(default_model)

        default_device = self._settings.value(settings_keys.DEFAULT_DEVICE, "")
        if not default_device:
            default_device = "cuda" if self._cuda_available else "cpu"
        self._device_combo.setCurrentText(default_device)

        self._reload_audio_formats()

    def _reload_audio_formats(self) -> None:
        """Rebuild the storage-format list from audio_formats.json (profiles may
        have been edited on the Settings screen) and select the default profile.
        """
        store, error = load_audio_format_store()
        self._audio_formats = store.formats()
        self._audio_format_combo.blockSignals(True)
        self._audio_format_combo.clear()
        for name, fmt in self._audio_formats.items():
            self._audio_format_combo.addItem(audio_format_label(fmt), name)
        self._audio_format_combo.setCurrentIndex(
            max(0, self._audio_format_combo.findData(store.default().name))
        )
        self._audio_format_combo.blockSignals(False)
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
        screen. Called from showEvent(), so a model warmed up mid-session shows
        up here without restarting the app.
        """
        choices = custom_models.asr_model_choices(self._settings, ASR_MODELS)
        if [self._model_combo.itemText(i) for i in range(self._model_combo.count())] == choices:
            return
        # clear() wipes the edit line too, so restore whatever was typed there.
        current = self._model_combo.currentText()
        self._model_combo.clear()
        self._model_combo.addItems(choices)
        self._model_combo.setCurrentText(current)

    def _toggle_advanced(self, checked: bool) -> None:
        self._advanced_container.setVisible(checked)
        arrow = "▾" if checked else "▸"
        self._advanced_toggle.setText(f"{arrow} Продвинутые параметры")

    def _browse_input(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Медиафайл", "", MEDIA_FILTER)
        if path:
            self._input_field.setText(path)
            self._validate()

    def _browse_out(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Папка результатов", self._out_field.text())
        if chosen:
            self._out_field.setText(chosen)
            self._settings.setValue(settings_keys.OUT_DIR, chosen)

    def _validate(self) -> bool:
        problems = []
        if not self._input_field.text():
            problems.append("выберите медиафайл")
        if self._min_speakers.value() > self._max_speakers.value():
            problems.append("минимум спикеров не может быть больше максимума")

        if problems:
            self._warning_label.setText(" · ".join(problems))
            self._warning_label.show()
            self._run_button.setEnabled(False)
            return False

        self._warning_label.hide()
        self._run_button.setEnabled(True)
        return True

    def _emit_run_requested(self) -> None:
        if not self._validate():
            return

        mode = MODES[self._mode_combo.currentIndex()][1]
        kwargs = {
            "input_path": self._input_field.text(),
            "out_dir": self._out_field.text(),
            "min_speakers": self._min_speakers.value(),
            "max_speakers": self._max_speakers.value(),
            "language": self._language_combo.currentText(),
            "device": self._device_combo.currentText(),
            "asr_model": self._model_combo.currentText(),
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
            kwargs["asr_initial_prompt"] = prompt
        self.run_requested.emit(mode, kwargs)
