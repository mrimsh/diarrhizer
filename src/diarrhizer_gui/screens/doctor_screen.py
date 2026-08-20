"""Doctor screen: renders the same environment checks as `diarrhizer doctor`.

Calls each check_*() function directly instead of the print-based
run_doctor_checks() - every check in doctor.py already returns
(name, passed, message) independently, so no core changes were needed.
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

from diarrhizer.diagnostics import doctor

CHECKS = [
    doctor.check_python_version,
    doctor.check_ffmpeg,
    doctor.check_torch,
    doctor.check_cuda,
    doctor.check_torchcodec,
    doctor.check_critical_imports,
    doctor.check_hf_token,
]

RECOMMENDED_ACTION = (
    "pip install -c requirements/constraints-stable.txt -r requirements/base.txt"
)

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

        fg, bg = PASS_COLORS if passed else FAIL_COLORS
        pill = QLabel("PASS" if passed else "FAIL")
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

        text_col = QVBoxLayout()
        text_col.setSpacing(2)
        text_col.addWidget(name_label)
        text_col.addWidget(message_label)

        row = QHBoxLayout(self)
        row.addWidget(pill)
        row.addLayout(text_col, stretch=1)


class DoctorScreen(QWidget):
    def __init__(self) -> None:
        super().__init__()

        title = QLabel("Окружение / Доктор")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")

        self._rerun_button = QPushButton("Повторить проверку")
        self._rerun_button.clicked.connect(self._run_checks)

        header = QHBoxLayout()
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self._rerun_button)

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
        layout.addWidget(scroll, stretch=1)
        layout.addWidget(self._action_label)

        self._run_checks()

    def _run_checks(self) -> None:
        while self._results_layout.count():
            item = self._results_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        results = [check() for check in CHECKS]
        for name, passed, message in results:
            self._results_layout.addWidget(CheckRow(name, passed, message))
        self._results_layout.addStretch(1)

        imports_failed = any(
            name == "Critical imports" and not passed for name, passed, _ in results
        )
        if imports_failed:
            self._action_label.setText(f"Рекомендуемое действие:\n{RECOMMENDED_ACTION}")
            self._action_label.show()
        else:
            self._action_label.hide()
