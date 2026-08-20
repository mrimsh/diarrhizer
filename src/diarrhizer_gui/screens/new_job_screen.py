"""New Job screen: a deliberately small subset of the full CLI parameter
surface (see the GUI Blueprint) - just enough to start a brand-new job in
one of two modes. Resuming a job, advanced ASR params, audio profiles other
than "raw", and force/from-stage/to-stage are later passes.

Only imports diarrhizer.diagnostics.doctor at module level (light, no torch
at import time - same as the Doctor screen). Anything from
diarrhizer.pipeline/diarrhizer.adapters stays inside pipeline_worker.py,
imported lazily when a run actually starts.
"""

from pathlib import Path

from PySide6.QtCore import QSettings, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from diarrhizer.diagnostics import doctor
from diarrhizer_gui import settings_keys

MEDIA_FILTER = "Медиафайлы (*.mp3 *.wav *.m4a *.mp4 *.mkv *.webm);;Все файлы (*.*)"

ASR_MODELS = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]

MODES = [
    ("Полный цикл", "full"),
    ("Только распознавание (без диаризации)", "asr_only"),
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
        self._model_combo.setCurrentText("large-v3")

        self._device_combo = QComboBox()
        self._device_combo.addItem("cuda")
        self._device_combo.addItem("cpu")
        _, cuda_ok, cuda_message = doctor.check_cuda()
        if cuda_ok:
            self._device_combo.setCurrentIndex(0)
        else:
            cuda_item = self._device_combo.model().item(0)
            cuda_item.setEnabled(False)
            cuda_item.setToolTip(cuda_message)
            self._device_combo.setCurrentIndex(1)

        form = QFormLayout()
        form.addRow("Файл:", input_row)
        form.addRow("Папка результатов:", out_row)
        form.addRow("Режим:", self._mode_combo)
        form.addRow("Язык:", self._language_combo)
        form.addRow("Спикеры (мин–макс):", speakers_row)
        form.addRow("Модель ASR:", self._model_combo)
        form.addRow("Устройство:", self._device_combo)

        self._warning_label = QLabel()
        self._warning_label.setStyleSheet("color: #b23b35;")
        self._warning_label.hide()

        self._run_button = QPushButton("Запустить")
        self._run_button.clicked.connect(self._emit_run_requested)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.addWidget(title)
        layout.addLayout(form)
        layout.addWidget(self._warning_label)
        layout.addWidget(self._run_button)
        layout.addStretch(1)

        default_out = str(Path.cwd() / "out")
        self._out_field.setText(self._settings.value(settings_keys.OUT_DIR, default_out))

        self._validate()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        default_out = str(Path.cwd() / "out")
        self._out_field.setText(self._settings.value(settings_keys.OUT_DIR, default_out))

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
        }
        self.run_requested.emit(mode, kwargs)
