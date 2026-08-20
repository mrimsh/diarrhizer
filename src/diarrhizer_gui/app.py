"""Entry point for the Diarrhizer desktop GUI."""

import sys

from PySide6.QtWidgets import QApplication

from diarrhizer_gui.main_window import MainWindow

ACCENT = "#a85520"

STYLESHEET = f"""
QListWidget#NavList {{
    border: none;
    outline: none;
    padding: 8px;
    background: #f2f3f6;
}}
QListWidget#NavList::item {{
    padding: 10px 12px;
    border-radius: 6px;
    margin: 2px 0;
}}
QListWidget#NavList::item:selected {{
    background: {ACCENT};
    color: white;
}}
"""


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    app.setOrganizationName("Diarrhizer")
    app.setApplicationName("DiarrhizerGUI")

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
