"""Restart dialog: confirm restarting one stage of a job from History, after
looking over - and optionally changing - the settings the stages that will be
recomputed run with.

The fields are the run_pipeline() settings of those stages (STAGE_SETTINGS,
as filtered by job_scan.restart_settings) and start from what the job
recorded; the form looks like New Job's advanced section, reusing its choice
lists. Only PySide6 + light diarrhizer modules at import time.
"""

import dataclasses
from typing import Any, Callable, Optional

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from diarrhizer.audio_formats import AudioFormat, resolve_audio_format
from diarrhizer.pipeline.runner import PipelineConfig
from diarrhizer_gui.screens.new_job_screen import (
    AUDIO_PROFILES,
    COMPUTE_TYPES,
    LANGUAGES,
    audio_format_hint,
    audio_format_label,
    available_asr_models,
    load_audio_format_store,
)

# (heading, settings under it) in pipeline order; device has no heading - it
# belongs to two stages and sits at the top.
SECTIONS = [
    ("Конвертация", ("audio_profile", "audio_format")),
    (
        "Распознавание",
        (
            "language",
            "asr_model",
            "asr_compute_type",
            "asr_beam_size",
            "asr_temperature",
            "asr_condition_on_previous_text",
            "asr_vad_filter",
            "asr_vad_min_silence_ms",
            "asr_initial_prompt",
        ),
    ),
    ("Диаризация", ("min_speakers", "max_speakers")),
    ("Объединение", ("min_turn_duration",)),
]

MIN_TURN_TOOLTIP = (
    "Сегмент ASR, задевающий смену говорящего, режется на реплики.\n"
    "Отрезки короче этого порога, зажатые между двумя отрезками\n"
    "одного и того же другого спикера, считаются дрожанием\n"
    "диаризации и приклеиваются обратно. 0 — резать на каждой смене."
)


# [SEMANTIC-BEGIN] GUI:RESTART_DIALOG
# @purpose: Confirm a stage restart and let the user change the settings of the stages it recomputes
# @description: Shows `intro` (what will be redone) and one field per setting in `settings`
#   (job_scan.restart_settings), pre-filled from `recorded` (the plan's kwargs: what the job ran with;
#   PipelineConfig defaults for anything the record lacks). overrides() returns only the settings the
#   user changed, judged against the widgets' own initial state - so a recorded value a spin box has to
#   round (temperature 0.25 -> 0.3) is not reported as a change by merely opening the dialog. The
#   storage format is resolved to an AudioFormat (with keep_wav applied) like New Job does, so the run
#   uses what was on screen even if audio_formats.json changes before it starts. A job whose recorded
#   format is no longer in the profile store keeps it as an extra "как в задании" choice. cuda is
#   disabled when the environment probe says it is unavailable. accept() refuses an empty model and a
#   speaker minimum above the maximum.
# @inputs: intro text, settings keys, recorded kwargs, QSettings (custom model list), optional EnvironmentProbe
# @outputs: overrides() -> {run_pipeline setting: new value}, accepted/rejected dialog result
# @sideEffects: Reads audio_formats.json and the HF cache listing to fill the choices; writes nothing
# @errors: None (invalid input is shown in a message box and the dialog stays open)
# @see: GUI:JOB_PLANS, CONFIG:PIPELINE, CONFIG:AUDIO_FORMATS
class RestartDialog(QDialog):
    def __init__(
        self,
        parent,
        intro: str,
        settings: list,
        recorded: dict,
        qsettings: QSettings,
        probe=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Перезапустить этап")
        self.intro_text = intro
        self._keys = set(settings)
        self._recorded = recorded
        self._qsettings = qsettings
        self._probe = probe
        self._widgets: dict = {}
        self._getters: dict[str, Callable[[], Any]] = {}
        self._formats: list = []

        intro_label = QLabel(intro)
        intro_label.setWordWrap(True)

        form = QFormLayout()
        if "device" in self._keys:
            self._add_device(form)
        for title, keys in SECTIONS:
            shown = [key for key in keys if key in self._keys]
            if not shown:
                continue
            header = QLabel(title)
            header.setStyleSheet("font-weight: 600; padding-top: 8px;")
            form.addRow(header)
            for key in shown:
                self._add_field(form, key)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Перезапустить")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(intro_label)
        if self._getters:
            hint = QLabel(
                "Настройки взяты из задания. Изменённые значения сохранятся в нём: "
                "их увидят история и повторные запуски."
            )
            hint.setWordWrap(True)
            hint.setStyleSheet("color: #7d8394;")
            body = QWidget()
            body.setLayout(form)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            scroll.setWidget(body)
            layout.addWidget(hint)
            layout.addWidget(scroll, stretch=1)
            self.resize(640, 560)
        else:
            self.setMinimumWidth(460)
        layout.addWidget(buttons)

        self._initial = self._values()
        if "device" in self._initial:
            # Compared with what the job ran on: leaving cuda for cpu because
            # there is no CUDA is a change the run has to be told about.
            self._initial["device"] = str(self._recorded_value("device"))

    # --- public API ----------------------------------------------------------

    def field(self, key: str):
        """The input widget for a setting (None when the dialog does not offer it)."""
        return self._widgets.get(key)

    def overrides(self) -> dict:
        """{setting: new value} for what the user changed."""
        return {key: value for key, value in self._values().items() if value != self._initial[key]}

    def accept(self) -> None:
        problem = self._problem()
        if problem:
            QMessageBox.warning(self, self.windowTitle(), problem)
            return
        super().accept()

    # --- building the form ------------------------------------------------------

    def _recorded_value(self, key: str):
        return self._recorded[key] if key in self._recorded else getattr(PipelineConfig, key)

    def _register(self, key: str, widget, getter: Callable[[], Any]) -> None:
        self._widgets[key] = widget
        self._getters[key] = getter

    def _add_device(self, form: QFormLayout) -> None:
        combo = QComboBox()
        combo.addItem("cuda")
        combo.addItem("cpu")
        combo.setCurrentText(str(self._recorded_value("device")))
        if self._probe is not None and self._probe.cuda_available is False:
            cuda_item = combo.model().item(0)
            cuda_item.setEnabled(False)
            cuda_item.setToolTip(self._probe.cuda_message)
            if combo.currentText() == "cuda":
                combo.setCurrentText("cpu")
        self._register("device", combo, combo.currentText)
        form.addRow("Устройство:", combo)

    def _add_field(self, form: QFormLayout, key: str) -> None:
        value = self._recorded_value(key)
        if key == "audio_profile":
            combo = QComboBox()
            for data, label in AUDIO_PROFILES:
                combo.addItem(label, data)
            self._select_data(combo, value)
            self._register(key, combo, combo.currentData)
            form.addRow("Предобработка звука:", combo)
        elif key == "audio_format":
            self._add_audio_format(form, value)
        elif key == "language":
            combo = QComboBox()
            combo.setEditable(True)
            combo.addItems(LANGUAGES)
            combo.setCurrentText(str(value))
            self._register(key, combo, lambda: combo.currentText().strip() or "auto")
            form.addRow("Язык:", combo)
        elif key == "asr_model":
            combo = QComboBox()
            combo.setEditable(True)
            combo.addItems(available_asr_models(self._qsettings))
            combo.setCurrentText(str(value))
            self._register(key, combo, lambda: combo.currentText().strip())
            form.addRow("Модель ASR:", combo)
        elif key == "asr_compute_type":
            combo = QComboBox()
            for label, data in COMPUTE_TYPES:
                combo.addItem(label, data)
            self._select_data(combo, value)
            self._register(key, combo, combo.currentData)
            form.addRow("Compute type:", combo)
        elif key == "asr_beam_size":
            spin = QSpinBox()
            spin.setRange(1, 20)
            spin.setValue(int(value))
            self._register(key, spin, spin.value)
            form.addRow("Beam size:", spin)
        elif key == "asr_temperature":
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 1.0)
            spin.setSingleStep(0.1)
            spin.setDecimals(1)
            spin.setValue(float(value))
            self._register(key, spin, spin.value)
            form.addRow("Temperature:", spin)
        elif key == "asr_condition_on_previous_text":
            box = QCheckBox("Учитывать предыдущий текст (стабильнее декодирование)")
            box.setChecked(bool(value))
            self._register(key, box, box.isChecked)
            form.addRow(box)
        elif key == "asr_vad_filter":
            box = QCheckBox("Включить VAD-фильтр (отсекать тишину)")
            box.setChecked(bool(value))
            self._register(key, box, box.isChecked)
            form.addRow(box)
        elif key == "asr_vad_min_silence_ms":
            spin = QSpinBox()
            spin.setRange(0, 10000)
            spin.setSingleStep(100)
            spin.setSuffix(" мс")
            spin.setValue(int(value))
            self._register(key, spin, spin.value)
            form.addRow("Мин. тишина для VAD:", spin)
        elif key == "asr_initial_prompt":
            edit = QPlainTextEdit()
            edit.setPlaceholderText("Глоссарий/начальный промпт (необязательно)")
            edit.setFixedHeight(70)
            edit.setPlainText(value or "")
            self._register(key, edit, lambda: edit.toPlainText().strip() or None)
            form.addRow("Промпт/глоссарий:", edit)
        elif key == "min_speakers":
            self._add_speakers(form)
        elif key == "min_turn_duration":
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 5.0)
            spin.setSingleStep(0.1)
            spin.setDecimals(2)
            spin.setSuffix(" с")
            spin.setToolTip(MIN_TURN_TOOLTIP)
            spin.setValue(float(value))
            self._register(key, spin, spin.value)
            form.addRow("Мин. длительность реплики:", spin)
        # max_speakers is added together with min_speakers.

    @staticmethod
    def _select_data(combo: QComboBox, value) -> None:
        """Select the item whose data is `value`; a value the list does not
        know (a custom profile, a hand-edited record) is added, not lost."""
        for index in range(combo.count()):
            if (combo.itemData(index) or "") == (value or ""):
                combo.setCurrentIndex(index)
                return
        combo.addItem(str(value), value)
        combo.setCurrentIndex(combo.count() - 1)

    def _add_speakers(self, form: QFormLayout) -> None:
        low = QSpinBox()
        low.setRange(1, 20)
        low.setValue(int(self._recorded_value("min_speakers")))
        high = QSpinBox()
        high.setRange(1, 20)
        high.setValue(int(self._recorded_value("max_speakers")))
        self._register("min_speakers", low, low.value)
        self._register("max_speakers", high, high.value)
        row = QHBoxLayout()
        row.addWidget(low)
        row.addWidget(QLabel("–"))
        row.addWidget(high)
        row.addStretch(1)
        hint = QLabel("Если число участников известно, укажите его в обоих полях.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #7d8394;")
        form.addRow("Спикеры (мин–макс):", row)
        form.addRow("", hint)

    def _add_audio_format(self, form: QFormLayout, recorded: AudioFormat) -> None:
        store, error = load_audio_format_store()
        self._formats = list(store.formats().values())
        combo = QComboBox()
        for fmt in self._formats:
            combo.addItem(audio_format_label(fmt))
        # Which store profile the job used: keep_wav is a per-run choice, so it
        # is left out of the comparison. A profile edited or deleted since stays
        # available as the job's own, instead of silently becoming another one.
        match = next(
            (
                index
                for index, fmt in enumerate(self._formats)
                if dataclasses.replace(fmt, keep_wav=recorded.keep_wav) == recorded
            ),
            None,
        )
        if match is None:
            self._formats.insert(0, recorded)
            combo.insertItem(0, f"{recorded.name} — как в задании")
            match = 0
        combo.setCurrentIndex(match)

        keep_wav = QCheckBox("Оставить рабочий WAV")
        keep_wav.setToolTip(
            "Рабочий audio/normalized.wav (без сжатия) нужен только на время обработки.\n"
            "Если его не оставлять, после завершения задания на диске остаётся\n"
            "только архивная копия в выбранном формате."
        )
        hint = QLabel()
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #7d8394;")

        def on_format_changed() -> None:
            fmt = self._formats[combo.currentIndex()]
            # A WAV-only format has nothing else to keep, so the WAV always stays.
            keep_wav.setEnabled(fmt.writes_archive)
            keep_wav.setChecked(fmt.keeps_wav)
            hint.setText(audio_format_hint(fmt))

        combo.currentIndexChanged.connect(lambda _index: on_format_changed())
        on_format_changed()
        keep_wav.setChecked(recorded.keeps_wav)  # what the job chose, not the profile's default
        if error:
            hint.setText(error)
            hint.setStyleSheet("color: #b23b35;")

        self._widgets["audio_format_keep_wav"] = keep_wav
        self._register(
            "audio_format",
            combo,
            lambda: resolve_audio_format(self._formats[combo.currentIndex()], keep_wav.isChecked()),
        )
        # Format labels are long: let the list shrink to the dialog instead of
        # widening it, and give the checkbox a row of its own.
        combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(24)
        form.addRow("Хранение аудио:", combo)
        form.addRow("", keep_wav)
        form.addRow("", hint)

    # --- reading the form back --------------------------------------------------

    def _values(self) -> dict:
        return {key: getter() for key, getter in self._getters.items()}

    def _problem(self) -> Optional[str]:
        values = self._values()
        if "asr_model" in values and not values["asr_model"]:
            return "Укажите модель распознавания."
        if "min_speakers" in values and values["min_speakers"] > values["max_speakers"]:
            return "Минимум спикеров не может быть больше максимума."
        return None
# [SEMANTIC-END] GUI:RESTART_DIALOG
