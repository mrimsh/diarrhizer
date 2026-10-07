"""Doctor screen: renders the same environment checks as `diarrhizer doctor`.

The checks run in a child process (EnvironmentProbe), so this screen never
blocks the window and "Повторить проверку" sees packages installed since
the GUI started. A failed optional check (torchcodec) is a warning.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from diarrhizer.diagnostics.doctor import OPTIONAL_CHECKS

RECOMMENDED_ACTION = (
    "pip install -c requirements/constraints-stable.txt -r requirements/base.txt"
)

# What to do about a failed check, in the GUI's terms.
FAILURE_HINTS = {
    "HF Token": "Нужен для разделения по спикерам: вставьте токен на экране «Настройки». "
    "Без него работает режим «Только распознавание».",
    "FFmpeg": "Установите FFmpeg и добавьте его в PATH или укажите путь к ffmpeg.exe на экране «Настройки».",
    "CUDA": "Обработка пойдёт на CPU (медленнее). Для GPU нужна сборка torch с CUDA — см. README.",
    "cuDNN": "Распознавание на GPU упадёт с ошибкой cuDNN — см. docs/troubleshooting.md или выберите устройство cpu.",
    "torchcodec": "Необязательно: без него аудио для диаризации декодируется чуть медленнее.",
}

# Same light-theme pass/fail pairs as the GUI Blueprint's --ok/--ok-soft and
# --danger/--danger-soft tokens, kept as plain hex pairs (not alpha-blended
# QSS colors) since Qt Style Sheets use #AARRGGBB, not CSS's #RRGGBBAA.
PASS_COLORS = ("#2f7d52", "#e2f0e6")
FAIL_COLORS = ("#b23b35", "#f6e1df")
WARN_COLORS = ("#8f6a15", "#f3ecd6")


class CheckRow(QFrame):
    def __init__(self, name: str, passed: bool, message: str) -> None:
        super().__init__()
        self.setObjectName("CheckRow")
        self.setStyleSheet(
            "QFrame#CheckRow { border: 1px solid #d7dae2; border-radius: 8px; "
            "padding: 10px 12px; background: #ffffff; }"
        )

        optional = not passed and name in OPTIONAL_CHECKS
        if passed:
            (fg, bg), text = PASS_COLORS, "PASS"
        elif optional:
            (fg, bg), text = WARN_COLORS, "WARN"
        else:
            (fg, bg), text = FAIL_COLORS, "FAIL"
        pill = QLabel(text)
        pill.setFixedWidth(56)
        pill.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pill.setStyleSheet(
            f"background: {bg}; color: {fg}; border-radius: 4px; "
            f"font-weight: 600; padding: 2px 0;"
        )

        name_label = QLabel(name)
        name_label.setStyleSheet("font-weight: 600;")

        message_label = QLabel(message)
        message_label.setStyleSheet("color: #565c6b;")
        message_label.setWordWrap(True)
        message_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        text_col = QVBoxLayout()
        text_col.setSpacing(2)
        text_col.addWidget(name_label)
        text_col.addWidget(message_label)
        if not passed and name in FAILURE_HINTS:
            hint = QLabel(FAILURE_HINTS[name])
            hint.setWordWrap(True)
            hint.setStyleSheet(f"color: {fg};")
            text_col.addWidget(hint)

        row = QHBoxLayout(self)
        row.addWidget(pill)
        row.addLayout(text_col, stretch=1)


class DoctorScreen(QWidget):
    def __init__(self, probe) -> None:
        super().__init__()
        self._probe = probe

        title = QLabel("Окружение / Доктор")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")

        self._rerun_button = QPushButton("Повторить проверку")
        self._rerun_button.clicked.connect(self._probe.run)

        header = QHBoxLayout()
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self._rerun_button)

        self._summary_label = QLabel()
        self._summary_label.setWordWrap(True)

        self._results_layout = QVBoxLayout()
        self._results_layout.setSpacing(8)

        results_container = QWidget()
        results_container.setLayout(self._results_layout)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(results_container)

        fg, bg = WARN_COLORS
        self._action_label = QLabel()
        self._action_label.setWordWrap(True)
        self._action_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._action_label.setStyleSheet(
            f"background: {bg}; color: {fg}; border-radius: 6px; padding: 10px 12px;"
        )
        self._action_label.hide()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.addLayout(header)
        layout.addWidget(self._summary_label)
        layout.addWidget(scroll, stretch=1)
        layout.addWidget(self._action_label)

        self._probe.started.connect(self._on_started)
        self._probe.finished.connect(self._show_results)
        if self._probe.results is not None:
            self._show_results(self._probe.results)
        else:
            self._on_started()

    def _clear(self) -> None:
        while self._results_layout.count():
            item = self._results_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _on_started(self) -> None:
        self._rerun_button.setEnabled(False)
        self._summary_label.setText(
            "Проверяю окружение… Первая проверка после запуска может занять до минуты: "
            "загружаются torch и whisperx. Остальные экраны уже доступны."
        )
        self._summary_label.setStyleSheet("color: #565c6b;")

    def _show_results(self, results: list) -> None:
        self._rerun_button.setEnabled(True)
        self._clear()
        for name, passed, message in results:
            self._results_layout.addWidget(CheckRow(name, passed, message))
        self._results_layout.addStretch(1)

        failures = self._probe.failure_count
        warnings = sum(1 for name, passed, _ in results if not passed and name in OPTIONAL_CHECKS)
        if failures:
            text = f"Проблем: {failures}. Подсказки — под каждой проверкой с пометкой FAIL."
            color = FAIL_COLORS[0]
        else:
            text = "Окружение в порядке." + (" Есть необязательные предупреждения." if warnings else "")
            color = PASS_COLORS[0]
        self._summary_label.setText(text)
        self._summary_label.setStyleSheet(f"color: {color}; font-weight: 600;")

        imports_failed = any(
            name == "Critical imports" and not passed for name, passed, _ in results
        )
        if imports_failed:
            self._action_label.setText(f"Рекомендуемое действие:\n{RECOMMENDED_ACTION}")
            self._action_label.show()
        else:
            self._action_label.hide()
