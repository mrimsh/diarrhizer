"""Models screen: UI over diagnostics/models.py's already-complete HF Hub
cache management (scan/warm-up/delete). This screen holds no cache logic of
its own - it only renders CachedModelInfo and drives the warm_up_*/clear_cache
functions.

`diarrhizer.diagnostics.models`'s scanning functions are import-light (only
huggingface_hub) - safe at module level here, same as doctor.py elsewhere.
warm_up_asr_model()/warm_up_diarization_model() defer torch/whisperx imports
to call time internally, but still block and can download gigabytes, so they
run through a small QThread worker, not on the UI thread.
"""

import os
from typing import Optional

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from diarrhizer.diagnostics import doctor
from diarrhizer.diagnostics import models as model_cache

# Mirrors faster_whisper.utils._MODELS (faster-whisper==1.1.0, pinned in
# requirements/constraints-stable.txt) for just the presets offered elsewhere
# in the GUI (see NewJobScreen.ASR_MODELS). Not imported directly: `import
# faster_whisper.utils` pulls in torch (confirmed), which this screen must
# not do at module level. Revisit if the pinned faster-whisper version changes.
ASR_MODEL_REPOS = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
}


def format_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if size < 1024 or unit == "ГБ":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} ГБ"


class ModelActionWorker(QObject):
    finished = Signal()
    failed = Signal(str, str)

    def __init__(self, fn, *args, **kwargs) -> None:
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def run(self) -> None:
        try:
            self._fn(*self._args, **self._kwargs)
            self.finished.emit()
        except Exception as exc:
            self.failed.emit(type(exc).__name__, str(exc))


class ModelsScreen(QWidget):
    ASR_COLUMNS = ["Модель", "Статус", "Размер", ""]
    CACHE_COLUMNS = ["Repo", "Размер", "Использовано", ""]

    def __init__(self) -> None:
        super().__init__()
        self._busy = False
        self._thread: Optional[QThread] = None
        self._worker: Optional[ModelActionWorker] = None
        self._diar_device = "cpu"

        title = QLabel("Модели")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")

        asr_label = QLabel("Модели ASR")
        asr_label.setStyleSheet("font-weight: 600; margin-top: 8px;")

        self._asr_table = QTableWidget(0, len(self.ASR_COLUMNS))
        self._asr_table.setHorizontalHeaderLabels(self.ASR_COLUMNS)
        self._asr_table.verticalHeader().setVisible(False)
        self._asr_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        asr_header = self._asr_table.horizontalHeader()
        asr_header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        asr_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)

        self._custom_model_field = QLineEdit()
        self._custom_model_field.setPlaceholderText(
            "Свой HF repo id (например koekaverna/faster-whisper-podlodka-turbo)"
        )
        self._custom_warm_button = QPushButton("Прогреть")
        self._custom_warm_button.clicked.connect(self._warm_custom_model)
        custom_row = QHBoxLayout()
        custom_row.addWidget(self._custom_model_field, stretch=1)
        custom_row.addWidget(self._custom_warm_button)

        diar_label = QLabel("Диаризация")
        diar_label.setStyleSheet("font-weight: 600; margin-top: 8px;")
        self._diar_status_label = QLabel()
        self._diar_warm_button = QPushButton("Прогреть модель диаризации")
        self._diar_warm_button.clicked.connect(self._warm_diarization)
        diar_row = QHBoxLayout()
        diar_row.addWidget(self._diar_status_label, stretch=1)
        diar_row.addWidget(self._diar_warm_button)

        cache_label = QLabel("Весь кэш")
        cache_label.setStyleSheet("font-weight: 600; margin-top: 8px;")
        self._cache_size_label = QLabel()
        self._cache_size_label.setStyleSheet("color: #7d8394;")
        refresh_button = QPushButton("Обновить")
        refresh_button.clicked.connect(self.refresh)
        cache_header_row = QHBoxLayout()
        cache_header_row.addWidget(cache_label)
        cache_header_row.addWidget(self._cache_size_label)
        cache_header_row.addStretch(1)
        cache_header_row.addWidget(refresh_button)

        self._cache_table = QTableWidget(0, len(self.CACHE_COLUMNS))
        self._cache_table.setHorizontalHeaderLabels(self.CACHE_COLUMNS)
        self._cache_table.verticalHeader().setVisible(False)
        self._cache_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        cache_header = self._cache_table.horizontalHeader()
        cache_header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        cache_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)

        self._status_label = QLabel()
        self._status_label.setWordWrap(True)
        self._status_label.hide()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.addWidget(title)
        layout.addWidget(asr_label)
        layout.addWidget(self._asr_table)
        layout.addLayout(custom_row)
        layout.addWidget(diar_label)
        layout.addLayout(diar_row)
        layout.addLayout(cache_header_row)
        layout.addWidget(self._cache_table, stretch=1)
        layout.addWidget(self._status_label)

        self.refresh()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.refresh()

    def refresh(self) -> None:
        cached = model_cache.list_cached_models()
        cached_by_id = {m.repo_id: m for m in cached}

        self._asr_table.setRowCount(0)
        for row, (alias, repo_id) in enumerate(ASR_MODEL_REPOS.items()):
            self._asr_table.insertRow(row)
            self._asr_table.setItem(row, 0, QTableWidgetItem(alias))
            info = cached_by_id.get(repo_id)
            self._asr_table.setItem(row, 1, QTableWidgetItem("В кэше" if info else "Не скачано"))
            self._asr_table.setItem(row, 2, QTableWidgetItem(format_size(info.size_on_disk) if info else "—"))
            warm_button = QPushButton("Прогреть")
            warm_button.setEnabled(not self._busy)
            warm_button.clicked.connect(lambda checked=False, a=alias: self._warm_asr(a))
            self._asr_table.setCellWidget(row, 3, warm_button)

        _, cuda_ok, _ = doctor.check_cuda()
        self._diar_device = "cuda" if cuda_ok else "cpu"

        _, hf_ok, hf_message = doctor.check_hf_token()
        self._diar_status_label.setText(f"HF-токен: {hf_message}")
        self._diar_warm_button.setEnabled(hf_ok and not self._busy)
        self._diar_warm_button.setToolTip(
            "" if hf_ok else "Нужен HF_TOKEN/HUGGINGFACE_HUB_TOKEN в окружении"
        )

        self._cache_table.setRowCount(0)
        total_size = 0
        for row, info in enumerate(cached):
            self._cache_table.insertRow(row)
            self._cache_table.setItem(row, 0, QTableWidgetItem(info.repo_id))
            self._cache_table.setItem(row, 1, QTableWidgetItem(format_size(info.size_on_disk)))
            self._cache_table.setItem(
                row, 2, QTableWidgetItem(info.last_used.strftime("%Y-%m-%d %H:%M"))
            )
            delete_button = QPushButton("Удалить")
            delete_button.setEnabled(not self._busy)
            delete_button.clicked.connect(lambda checked=False, rid=info.repo_id: self._delete_model(rid))
            self._cache_table.setCellWidget(row, 3, delete_button)
            total_size += info.size_on_disk
        self._cache_size_label.setText(f"({format_size(total_size)})")

    def _warm_asr(self, alias: str) -> None:
        device = "cuda" if doctor.check_cuda()[1] else "cpu"
        self._run_action(
            f"Прогреваю «{alias}»… (без индикатора прогресса — huggingface_hub не отдаёт колбэк, см. docstring warm_up_asr_model)",
            model_cache.warm_up_asr_model,
            alias,
            device=device,
        )

    def _warm_custom_model(self) -> None:
        repo_id = self._custom_model_field.text().strip()
        if not repo_id:
            return
        device = "cuda" if doctor.check_cuda()[1] else "cpu"
        self._run_action(
            f"Прогреваю «{repo_id}»… (без индикатора прогресса)",
            model_cache.warm_up_asr_model,
            repo_id,
            device=device,
        )

    def _warm_diarization(self) -> None:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if not token:
            return
        self._run_action(
            "Прогреваю модель диаризации… (без индикатора прогресса)",
            model_cache.warm_up_diarization_model,
            token,
            device=self._diar_device,
        )

    def _delete_model(self, repo_id: str) -> None:
        model_cache.clear_cache(repo_id)
        self.refresh()

    def _run_action(self, message: str, fn, *args, **kwargs) -> None:
        if self._busy:
            return
        self._set_busy(True, message)

        worker = ModelActionWorker(fn, *args, **kwargs)
        thread = QThread(self)
        worker.moveToThread(thread)
        # Bound methods, not lambdas: finished/failed are emitted from the
        # worker thread, and Qt only auto-detects "this needs a queued,
        # main-thread delivery" for a real QObject bound-method receiver - a
        # plain lambda has no such receiver, so it would run directly on the
        # worker thread and touch widgets from there (this is exactly what
        # produced "QObject::setParent: Cannot set parent, new parent is in
        # a different thread" the first time this shipped).
        worker.finished.connect(self._on_worker_finished)
        worker.failed.connect(self._on_worker_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.started.connect(worker.run)
        thread.finished.connect(self._on_thread_finished)
        thread.finished.connect(thread.deleteLater)

        self._worker = worker
        self._thread = thread
        thread.start()

    def _on_thread_finished(self) -> None:
        # Same lesson as MainWindow's pipeline threading: thread.deleteLater()
        # destroys the C++ QThread independently of this Python reference, so
        # don't keep touching self._thread afterward.
        self._thread = None
        self._worker = None

    def _on_worker_finished(self) -> None:
        self._set_busy(False, "")

    def _on_worker_failed(self, error_type: str, message: str) -> None:
        self._set_busy(False, f"Ошибка ({error_type}): {message}")
        self._status_label.setStyleSheet("color: #b23b35;")

    def _set_busy(self, busy: bool, message: str) -> None:
        # refresh() unconditionally (not just when finishing) so the button
        # enabled/disabled state actually reflects `busy` the moment an
        # action starts, not only after it completes.
        self._busy = busy
        self._status_label.setStyleSheet("color: #565c6b;")
        self._status_label.setText(message)
        self._status_label.setVisible(bool(message))
        self.refresh()
