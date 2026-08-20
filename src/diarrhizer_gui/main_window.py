"""Main application window: sidebar navigation + stacked screens."""

from PySide6.QtCore import QSize
from PySide6.QtWidgets import (
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QStackedWidget,
    QWidget,
)

from diarrhizer_gui.screens.doctor_screen import DoctorScreen
from diarrhizer_gui.screens.history_screen import HistoryScreen
from diarrhizer_gui.screens.placeholder_screen import PlaceholderScreen

NAV_ITEMS = [
    ("Доктор", "doctor"),
    ("Модели", "models"),
    ("Новое задание", "new_job"),
    ("История заданий", "history"),
    ("Настройки", "settings"),
]


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Diarrhizer")
        self.resize(1100, 720)

        self._nav = QListWidget()
        self._nav.setObjectName("NavList")
        self._nav.setFixedWidth(220)

        self._stack = QStackedWidget()
        self._build_screens()

        self._nav.currentRowChanged.connect(self._stack.setCurrentIndex)
        self._nav.setCurrentRow(0)

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._nav)
        layout.addWidget(self._stack, stretch=1)
        self.setCentralWidget(central)

    def _build_screens(self) -> None:
        screens = {
            "doctor": DoctorScreen(),
            "models": PlaceholderScreen("Модели"),
            "new_job": PlaceholderScreen("Новое задание"),
            "history": HistoryScreen(),
            "settings": PlaceholderScreen("Настройки"),
        }
        for label, key in NAV_ITEMS:
            item = QListWidgetItem(label)
            item.setSizeHint(QSize(0, 40))
            self._nav.addItem(item)
            self._stack.addWidget(screens[key])
