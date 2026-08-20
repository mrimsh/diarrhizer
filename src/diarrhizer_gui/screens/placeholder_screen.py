"""Stand-in for screens not built yet, so navigation stays fully clickable."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class PlaceholderScreen(QWidget):
    def __init__(self, title: str) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        heading = QLabel(title)
        heading.setStyleSheet("font-size: 18px; font-weight: 600;")
        heading.setAlignment(Qt.AlignmentFlag.AlignCenter)

        note = QLabel("Экран ещё не реализован")
        note.setStyleSheet("color: #7d8394;")
        note.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(heading)
        layout.addWidget(note)
