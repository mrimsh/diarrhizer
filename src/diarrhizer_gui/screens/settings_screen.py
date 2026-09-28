"""Settings screen: default output folder/device/ASR model (QSettings), plus
HF token and FFmpeg path override, both persisted to the repo-root .env via
env_file.py and applied to os.environ immediately so they take effect in the
current session without a restart. Also edits audio storage format profiles
(diarrhizer.audio_formats), persisted to audio_formats.json so the CLI sees
the same profiles and default.

Only imports diarrhizer.diagnostics.doctor at module level (light, no torch
at import time - same pattern as the other screens).
"""

import os
import shlex
from pathlib import Path

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from diarrhizer.audio_formats import AudioFormat, AudioFormatStore
from diarrhizer.diagnostics import doctor
from diarrhizer.diagnostics import models as model_cache
from diarrhizer_gui import custom_models, env_file, settings_keys
from diarrhizer_gui.screens.new_job_screen import (
    ASR_MODELS,
    audio_format_hint,
    audio_format_label,
    load_audio_format_store,
)

ENV_PATH = env_file.REPO_ROOT / ".env"

# Encoder -> usual extension; picking an encoder fills the extension in.
CODEC_EXTENSIONS = {
    "pcm_s16le": "wav",
    "flac": "flac",
    "libmp3lame": "mp3",
    "libopus": "opus",
    "libvorbis": "ogg",
}
SAMPLE_RATES = ["8000", "16000", "22050", "32000", "44100", "48000"]


def _breakable_paths(text: str) -> str:
    """Let a word-wrapped label break long paths after separators (zero-width
    spaces, display only) - QLabel only wraps at whitespace.
    """
    return text.replace("\\", "\\\u200b").replace("/", "/\u200b")


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
        self._hf_status_label.setWordWrap(True)

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
        # Wrapped so a long ffmpeg path can't force the (scrollable) screen wider than the window.
        self._ffmpeg_status_label.setWordWrap(True)

        # --- Model cache directory (HF_HOME) ---
        self._cache_dir_field = QLineEdit()
        self._cache_dir_field.setReadOnly(True)
        cache_dir_browse = QPushButton("Обзор…")
        cache_dir_browse.clicked.connect(self._browse_cache_dir)
        cache_dir_save = QPushButton("Сохранить")
        cache_dir_save.clicked.connect(self._save_cache_dir)
        cache_dir_clear = QPushButton("Сбросить")
        cache_dir_clear.clicked.connect(self._clear_cache_dir)
        cache_dir_row = QHBoxLayout()
        cache_dir_row.addWidget(self._cache_dir_field, stretch=1)
        cache_dir_row.addWidget(cache_dir_browse)
        cache_dir_row.addWidget(cache_dir_save)
        cache_dir_row.addWidget(cache_dir_clear)

        self._cache_dir_status_label = QLabel()
        self._cache_dir_status_label.setStyleSheet("color: #7d8394;")
        self._cache_dir_status_label.setWordWrap(True)

        env_form = QFormLayout()
        env_form.addRow("HF-токен:", hf_row)
        env_form.addRow("", self._hf_status_label)
        env_form.addRow("Путь к FFmpeg:", ffmpeg_row)
        env_form.addRow("", self._ffmpeg_status_label)
        env_form.addRow("Каталог кэша моделей:", cache_dir_row)
        env_form.addRow("", self._cache_dir_status_label)

        formats_title, formats_form = self._build_audio_formats_section()

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.addWidget(title)
        layout.addLayout(defaults_form)
        layout.addLayout(env_form)
        layout.addWidget(formats_title)
        layout.addLayout(formats_form)
        layout.addStretch(1)

        # The formats editor makes this screen taller than the window.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(content)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        self._load()

    def _build_audio_formats_section(self) -> tuple:
        title = QLabel("Хранение аудио")
        title.setStyleSheet("font-size: 15px; font-weight: 600; padding-top: 12px;")

        intro = QLabel(
            "Обработка всегда идёт по рабочему WAV 16 кГц моно. Профиль задаёт архивную копию "
            "(audio/archive.<расширение>) и нужно ли оставлять WAV после завершения задания. "
            "Встроенные профили не меняются: измените поля и нажмите «Сохранить как…»."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #7d8394;")

        self._store: AudioFormatStore | None = None
        self._format_combo = QComboBox()
        self._format_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._format_combo.currentIndexChanged.connect(self._on_format_selected)
        self._format_default_label = QLabel()
        self._format_default_label.setStyleSheet("color: #2f7d52;")
        profile_row = QHBoxLayout()
        profile_row.addWidget(self._format_combo, stretch=1)
        profile_row.addWidget(self._format_default_label)

        self._codec_combo = QComboBox()
        self._codec_combo.setEditable(True)
        self._codec_combo.addItems(list(CODEC_EXTENSIONS))
        self._codec_combo.setToolTip("Кодировщик ffmpeg (-c:a)")
        self._codec_combo.currentTextChanged.connect(self._on_codec_changed)

        self._extension_field = QLineEdit()
        self._extension_field.setToolTip("Расширение архивного файла — определяет контейнер")

        self._quality_field = QLineEdit()
        self._quality_field.setPlaceholderText("напр. 5 — переменный битрейт (-q:a)")
        self._bitrate_field = QLineEdit()
        self._bitrate_field.setPlaceholderText("напр. 32k — постоянный битрейт (-b:a)")
        quality_row = QHBoxLayout()
        quality_row.addWidget(self._quality_field)
        quality_row.addWidget(QLabel("или битрейт:"))
        quality_row.addWidget(self._bitrate_field)

        self._sample_rate_combo = QComboBox()
        self._sample_rate_combo.setEditable(True)
        self._sample_rate_combo.addItems(SAMPLE_RATES)
        self._channels_spin = QSpinBox()
        self._channels_spin.setRange(1, 8)
        rate_row = QHBoxLayout()
        rate_row.addWidget(self._sample_rate_combo)
        rate_row.addWidget(QLabel("Гц   каналов:"))
        rate_row.addWidget(self._channels_spin)
        rate_row.addStretch(1)

        self._extra_args_field = QLineEdit()
        self._extra_args_field.setPlaceholderText("напр. -sample_fmt s16")
        self._keep_wav_box = QCheckBox("Оставлять рабочий WAV после завершения задания")
        self._format_desc_field = QLineEdit()

        self._format_hint = QLabel()
        self._format_hint.setWordWrap(True)

        for signal in (
            self._extension_field.textChanged,
            self._quality_field.textChanged,
            self._bitrate_field.textChanged,
            self._sample_rate_combo.currentTextChanged,
            self._channels_spin.valueChanged,
            self._extra_args_field.textChanged,
            self._keep_wav_box.toggled,
        ):
            signal.connect(self._refresh_format_hint)

        self._format_save_as = QPushButton("Сохранить как…")
        self._format_save_as.clicked.connect(self._save_format_as)
        self._format_save = QPushButton("Сохранить")
        self._format_save.clicked.connect(self._save_format)
        self._format_delete = QPushButton("Удалить")
        self._format_delete.clicked.connect(self._delete_format)
        self._format_make_default = QPushButton("Сделать по умолчанию")
        self._format_make_default.clicked.connect(self._make_format_default)
        buttons_row = QHBoxLayout()
        for button in (
            self._format_save_as, self._format_save, self._format_delete, self._format_make_default
        ):
            buttons_row.addWidget(button)
        buttons_row.addStretch(1)

        self._format_status = QLabel()
        self._format_status.setWordWrap(True)

        form = QFormLayout()
        form.addRow(intro)
        form.addRow("Профиль:", profile_row)
        form.addRow("Кодек:", self._codec_combo)
        form.addRow("Расширение:", self._extension_field)
        form.addRow("Качество:", quality_row)
        form.addRow("Частота:", rate_row)
        form.addRow("Доп. параметры ffmpeg:", self._extra_args_field)
        form.addRow(self._keep_wav_box)
        form.addRow("Описание:", self._format_desc_field)
        form.addRow("", self._format_hint)
        form.addRow(buttons_row)
        form.addRow(self._format_status)
        return title, form

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._load()

    def _load(self) -> None:
        default_out = str(Path.cwd() / "out")
        self._out_field.setText(self._settings.value(settings_keys.OUT_DIR, default_out))

        device = self._settings.value(settings_keys.DEFAULT_DEVICE, "")
        if device:
            self._device_combo.blockSignals(True)
            self._device_combo.setCurrentText(device)
            self._device_combo.blockSignals(False)

        # Rebuilding the item list and setting the text both fire
        # currentTextChanged, which would write a half-applied value back to
        # QSettings - block signals around the whole thing.
        self._model_combo.blockSignals(True)
        self._reload_asr_models()
        model = self._settings.value(settings_keys.DEFAULT_ASR_MODEL, "")
        if model:
            self._model_combo.setCurrentText(model)
        self._model_combo.blockSignals(False)

        self._hf_field.setText(os.environ.get("HF_TOKEN", ""))
        self._refresh_hf_status()

        self._ffmpeg_field.setText(os.environ.get("DIARRHIZER_FFMPEG_PATH", ""))
        self._refresh_ffmpeg_status()

        self._cache_dir_field.setText(os.environ.get("HF_HOME", ""))
        self._refresh_cache_dir_status()

        self._reload_formats()

    def _reload_asr_models(self) -> None:
        """Presets + models added on the Models screen. Caller blocks signals."""
        choices = custom_models.asr_model_choices(self._settings, ASR_MODELS)
        if [self._model_combo.itemText(i) for i in range(self._model_combo.count())] == choices:
            return
        current = self._model_combo.currentText()
        self._model_combo.clear()
        self._model_combo.addItems(choices)
        self._model_combo.setCurrentText(current)

    def _refresh_hf_status(self) -> None:
        _, ok, message = doctor.check_hf_token()
        self._hf_status_label.setText(message)
        self._hf_status_label.setStyleSheet("color: #2f7d52;" if ok else "color: #b23b35;")

    def _refresh_ffmpeg_status(self) -> None:
        _, ok, message = doctor.check_ffmpeg()
        self._ffmpeg_status_label.setText(_breakable_paths(message))
        self._ffmpeg_status_label.setStyleSheet("color: #2f7d52;" if ok else "color: #b23b35;")

    def _refresh_cache_dir_status(self) -> None:
        resolved = model_cache.resolve_cache_dir()
        self._cache_dir_status_label.setText(
            f"Фактически используется: {resolved}. Новые модели скачиваются сюда; "
            "уже скачанные в прежнем расположении сами не переносятся."
        )

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

    def _browse_cache_dir(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, "Каталог кэша моделей", self._cache_dir_field.text()
        )
        if chosen:
            self._cache_dir_field.setText(chosen)

    def _save_cache_dir(self) -> None:
        path = self._cache_dir_field.text().strip()
        env_file.write_env_file(ENV_PATH, {"HF_HOME": path})
        if path:
            os.environ["HF_HOME"] = path
        else:
            os.environ.pop("HF_HOME", None)
        self._refresh_cache_dir_status()

    def _clear_cache_dir(self) -> None:
        self._cache_dir_field.setText("")
        env_file.write_env_file(ENV_PATH, {"HF_HOME": ""})
        os.environ.pop("HF_HOME", None)
        self._refresh_cache_dir_status()

    # --- Audio storage formats ---

    def _reload_formats(self, select: str | None = None) -> None:
        store, error = load_audio_format_store()
        # A file that failed to parse is left alone: saving now would replace
        # the user's (broken but hand-written) profiles with built-ins only.
        self._store = None if error else store
        current = select or self._format_combo.currentData() or store.default_name
        self._format_combo.blockSignals(True)
        self._format_combo.clear()
        for name, fmt in store.formats().items():
            self._format_combo.addItem(audio_format_label(fmt), name)
        self._format_combo.setCurrentIndex(max(0, self._format_combo.findData(current)))
        self._format_combo.blockSignals(False)
        self._on_format_selected()
        if error:
            self._set_format_status(error, ok=False)

    def _selected_format(self) -> AudioFormat | None:
        store = self._store or load_audio_format_store()[0]
        return store.formats().get(self._format_combo.currentData())

    def _on_format_selected(self) -> None:
        fmt = self._selected_format()
        if fmt is None:
            return
        editors = (
            self._codec_combo, self._extension_field, self._quality_field, self._bitrate_field,
            self._sample_rate_combo, self._channels_spin, self._extra_args_field,
            self._keep_wav_box, self._format_desc_field,
        )
        for widget in editors:
            widget.blockSignals(True)
        self._codec_combo.setCurrentText(fmt.codec)
        self._extension_field.setText(fmt.extension)
        self._quality_field.setText("" if fmt.quality is None else f"{fmt.quality:g}")
        self._bitrate_field.setText(fmt.bitrate or "")
        self._sample_rate_combo.setCurrentText(str(fmt.sample_rate))
        self._channels_spin.setValue(fmt.channels)
        self._extra_args_field.setText(shlex.join(fmt.extra_args))
        self._keep_wav_box.setChecked(fmt.keep_wav)
        self._format_desc_field.setText(fmt.description)
        for widget in editors:
            widget.blockSignals(False)

        builtin = AudioFormatStore.is_builtin(fmt.name)
        writable = self._store is not None
        self._format_save.setEnabled(writable and not builtin)
        self._format_delete.setEnabled(writable and not builtin)
        self._format_save_as.setEnabled(writable)
        is_default = writable and self._store.default().name == fmt.name
        self._format_make_default.setEnabled(writable and not is_default)
        self._format_default_label.setText("✓ по умолчанию" if is_default else "")
        self._format_status.clear()
        self._refresh_format_hint()

    def _on_codec_changed(self, codec: str) -> None:
        extension = CODEC_EXTENSIONS.get(codec.strip())
        if extension:
            self._extension_field.setText(extension)
        self._refresh_format_hint()

    def _format_from_fields(self, name: str) -> AudioFormat:
        """Build a format from the editor; ValueError says what is wrong."""
        quality = None
        quality_text = self._quality_field.text().strip().replace(",", ".")
        if quality_text:
            try:
                quality = float(quality_text)
            except ValueError:
                raise ValueError(f"Качество (-q:a) должно быть числом: {quality_text!r}") from None
            if quality.is_integer():
                quality = int(quality)
        try:
            sample_rate = int(self._sample_rate_combo.currentText().strip())
        except ValueError:
            raise ValueError("Частота должна быть целым числом Гц") from None
        try:
            extra_args = shlex.split(self._extra_args_field.text())
        except ValueError as e:
            raise ValueError(f"Доп. параметры ffmpeg: {e}") from None
        return AudioFormat(
            name=name,
            description=self._format_desc_field.text().strip(),
            codec=self._codec_combo.currentText().strip(),
            extension=self._extension_field.text().strip(),
            quality=quality,
            bitrate=self._bitrate_field.text().strip() or None,
            sample_rate=sample_rate,
            channels=self._channels_spin.value(),
            extra_args=extra_args,
            keep_wav=self._keep_wav_box.isChecked(),
        )

    def _refresh_format_hint(self) -> None:
        try:
            fmt = self._format_from_fields(self._format_combo.currentData() or "preview")
        except ValueError as e:
            self._format_hint.setText(str(e))
            self._format_hint.setStyleSheet("color: #b23b35;")
            return
        lines = [audio_format_hint(fmt)]
        if fmt.writes_archive:
            lines.append(f"ffmpeg … {shlex.join(fmt.encoder_args())} archive.{fmt.extension}")
            if not fmt.keep_wav:
                lines.append("После завершения задания рабочий WAV удаляется, остаётся только архив.")
        self._format_hint.setText("\n".join(lines))
        self._format_hint.setStyleSheet("color: #7d8394;")

    def _set_format_status(self, message: str, ok: bool) -> None:
        self._format_status.setText(message)
        self._format_status.setStyleSheet("color: #2f7d52;" if ok else "color: #b23b35;")

    def _store_format(self, fmt: AudioFormat) -> bool:
        try:
            self._store.put(fmt)
            self._store.save()
        except (ValueError, OSError) as e:
            self._set_format_status(str(e), ok=False)
            return False
        return True

    def _save_format_as(self) -> None:
        if self._store is None:
            return
        suggested = self._format_combo.currentData() or ""
        if AudioFormatStore.is_builtin(suggested):
            suggested = f"{suggested}-my"
        name, accepted = QInputDialog.getText(
            self, "Новый профиль", "Имя профиля (без пробелов):", text=suggested
        )
        name = name.strip()
        if not accepted or not name:
            return
        if AudioFormatStore.is_builtin(name):
            self._set_format_status(f"«{name}» — встроенный профиль, выберите другое имя", ok=False)
            return
        if name in self._store.custom and QMessageBox.question(
            self, "Профиль существует", f"Заменить профиль «{name}»?"
        ) != QMessageBox.StandardButton.Yes:
            return
        try:
            fmt = self._format_from_fields(name)
        except ValueError as e:
            self._set_format_status(str(e), ok=False)
            return
        if self._store_format(fmt):
            self._reload_formats(select=name)
            self._set_format_status(f"Профиль «{name}» сохранён в {self._store.path.name}", ok=True)

    def _save_format(self) -> None:
        name = self._format_combo.currentData()
        if self._store is None or not name or AudioFormatStore.is_builtin(name):
            return
        try:
            fmt = self._format_from_fields(name)
        except ValueError as e:
            self._set_format_status(str(e), ok=False)
            return
        if self._store_format(fmt):
            self._reload_formats(select=name)
            self._set_format_status(f"Профиль «{name}» сохранён", ok=True)

    def _delete_format(self) -> None:
        name = self._format_combo.currentData()
        if self._store is None or not name or AudioFormatStore.is_builtin(name):
            return
        if QMessageBox.question(
            self, "Удалить профиль",
            f"Удалить профиль «{name}»? Уже созданные задания не изменятся.",
        ) != QMessageBox.StandardButton.Yes:
            return
        try:
            self._store.remove(name)
            self._store.save()
        except (ValueError, OSError) as e:
            self._set_format_status(str(e), ok=False)
            return
        self._reload_formats(select=self._store.default_name)
        self._set_format_status(f"Профиль «{name}» удалён", ok=True)

    def _make_format_default(self) -> None:
        name = self._format_combo.currentData()
        if self._store is None or not name:
            return
        try:
            self._store.set_default(name)
            self._store.save()
        except (ValueError, OSError) as e:
            self._set_format_status(str(e), ok=False)
            return
        self._reload_formats(select=name)
        self._set_format_status(
            f"«{name}» — профиль по умолчанию для новых заданий (GUI и CLI)", ok=True
        )
