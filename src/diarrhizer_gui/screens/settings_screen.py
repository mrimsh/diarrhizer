"""Settings screen: default output folder/device/ASR model (QSettings), plus
HF token and FFmpeg path override, both persisted to the repo-root .env via
env_file.py and applied to os.environ immediately so they take effect in the
current session without a restart.

Only imports diarrhizer.diagnostics.doctor at module level (light, no torch
at import time - same pattern as the other screens).
"""

import os

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from diarrhizer.diagnostics import doctor
from diarrhizer_gui import env_file, settings_keys
from diarrhizer_gui.screens.new_job_screen import ASR_MODELS

ENV_PATH = env_file.REPO_ROOT / ".env"


class SettingsScreen(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._settings = QSettings("Diarrhizer", "DiarrhizerGUI")

        title = QLabel("Настройки")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")

        # --- Defaults for new jobs ---
        self._out_field = QLineEdit()
        self._out_field.setReadOnly(True)
        out_browse = QPushButton("Обзор…")
        out_browse.clicked.connect(self._browse_out)
        out_row = QHBoxLayout()
        out_row.addWidget(self._out_field, stretch=1)
        out_row.addWidget(out_browse)

        self._device_combo = QComboBox()
        self._device_combo.addItem("cuda")
        self._device_combo.addItem("cpu")
        _, cuda_ok, cuda_message = doctor.check_cuda()
        if not cuda_ok:
            cuda_item = self._device_combo.model().item(0)
            cuda_item.setEnabled(False)
            cuda_item.setToolTip(cuda_message)
        self._device_combo.currentTextChanged.connect(self._save_device)

        self._model_combo = QComboBox()
        self._model_combo.setEditable(True)
        self._model_combo.addItems(ASR_MODELS)
        self._model_combo.currentTextChanged.connect(self._save_asr_model)

        defaults_form = QFormLayout()
        defaults_form.addRow("Папка результатов по умолчанию:", out_row)
        defaults_form.addRow("Устройство по умолчанию:", self._device_combo)
        defaults_form.addRow("Модель ASR по умолчанию:", self._model_combo)

        # --- HF token ---
        self._hf_field = QLineEdit()
        self._hf_field.setEchoMode(QLineEdit.EchoMode.Password)
        self._hf_toggle = QPushButton("Показать")
        self._hf_toggle.setCheckable(True)
        self._hf_toggle.toggled.connect(self._toggle_hf_visibility)
        hf_save = QPushButton("Сохранить")
        hf_save.clicked.connect(self._save_hf_token)
        hf_row = QHBoxLayout()
        hf_row.addWidget(self._hf_field, stretch=1)
        hf_row.addWidget(self._hf_toggle)
        hf_row.addWidget(hf_save)

        self._hf_status_label = QLabel()
        self._hf_status_label.setStyleSheet("color: #7d8394;")

        # --- FFmpeg path ---
        self._ffmpeg_field = QLineEdit()
        self._ffmpeg_field.setReadOnly(True)
        ffmpeg_browse = QPushButton("Обзор…")
        ffmpeg_browse.clicked.connect(self._browse_ffmpeg)
        ffmpeg_save = QPushButton("Сохранить")
        ffmpeg_save.clicked.connect(self._save_ffmpeg_path)
        ffmpeg_clear = QPushButton("Сбросить")
        ffmpeg_clear.clicked.connect(self._clear_ffmpeg_path)
        ffmpeg_row = QHBoxLayout()
        ffmpeg_row.addWidget(self._ffmpeg_field, stretch=1)
        ffmpeg_row.addWidget(ffmpeg_browse)
        ffmpeg_row.addWidget(ffmpeg_save)
        ffmpeg_row.addWidget(ffmpeg_clear)

        self._ffmpeg_status_label = QLabel()
        self._ffmpeg_status_label.setStyleSheet("color: #7d8394;")

        env_form = QFormLayout()
        env_form.addRow("HF-токен:", hf_row)
        env_form.addRow("", self._hf_status_label)
        env_form.addRow("Путь к FFmpeg:", ffmpeg_row)
        env_form.addRow("", self._ffmpeg_status_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.addWidget(title)
        layout.addLayout(defaults_form)
        layout.addLayout(env_form)
        layout.addStretch(1)

        self._load()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._load()

    def _load(self) -> None:
        import os as _os
        from pathlib import Path

        default_out = str(Path.cwd() / "out")
        self._out_field.setText(self._settings.value(settings_keys.OUT_DIR, default_out))

        device = self._settings.value(settings_keys.DEFAULT_DEVICE, "")
        if device:
            self._device_combo.blockSignals(True)
            self._device_combo.setCurrentText(device)
            self._device_combo.blockSignals(False)

        model = self._settings.value(settings_keys.DEFAULT_ASR_MODEL, "")
        if model:
            self._model_combo.blockSignals(True)
            self._model_combo.setCurrentText(model)
            self._model_combo.blockSignals(False)

        self._hf_field.setText(_os.environ.get("HF_TOKEN", ""))
        self._refresh_hf_status()

        self._ffmpeg_field.setText(_os.environ.get("DIARRHIZER_FFMPEG_PATH", ""))
        self._refresh_ffmpeg_status()

    def _refresh_hf_status(self) -> None:
        _, ok, message = doctor.check_hf_token()
        self._hf_status_label.setText(message)
        self._hf_status_label.setStyleSheet("color: #2f7d52;" if ok else "color: #b23b35;")

    def _refresh_ffmpeg_status(self) -> None:
        _, ok, message = doctor.check_ffmpeg()
        self._ffmpeg_status_label.setText(message)
        self._ffmpeg_status_label.setStyleSheet("color: #2f7d52;" if ok else "color: #b23b35;")

    def _browse_out(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Папка результатов", self._out_field.text())
        if chosen:
            self._out_field.setText(chosen)
            self._settings.setValue(settings_keys.OUT_DIR, chosen)

    def _save_device(self, value: str) -> None:
        self._settings.setValue(settings_keys.DEFAULT_DEVICE, value)

    def _save_asr_model(self, value: str) -> None:
        self._settings.setValue(settings_keys.DEFAULT_ASR_MODEL, value)

    def _toggle_hf_visibility(self, checked: bool) -> None:
        self._hf_field.setEchoMode(
            QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
        )
        self._hf_toggle.setText("Скрыть" if checked else "Показать")

    def _save_hf_token(self) -> None:
        token = self._hf_field.text().strip()
        env_file.write_env_file(ENV_PATH, {"HF_TOKEN": token})
        os.environ["HF_TOKEN"] = token
        self._refresh_hf_status()

    def _browse_ffmpeg(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "ffmpeg.exe", "", "Исполняемые файлы (*.exe);;Все файлы (*.*)"
        )
        if path:
            self._ffmpeg_field.setText(path)

    def _save_ffmpeg_path(self) -> None:
        path = self._ffmpeg_field.text().strip()
        env_file.write_env_file(ENV_PATH, {"DIARRHIZER_FFMPEG_PATH": path})
        if path:
            os.environ["DIARRHIZER_FFMPEG_PATH"] = path
        else:
            os.environ.pop("DIARRHIZER_FFMPEG_PATH", None)
        self._refresh_ffmpeg_status()

    def _clear_ffmpeg_path(self) -> None:
        self._ffmpeg_field.setText("")
        env_file.write_env_file(ENV_PATH, {"DIARRHIZER_FFMPEG_PATH": ""})
        os.environ.pop("DIARRHIZER_FFMPEG_PATH", None)
        self._refresh_ffmpeg_status()
