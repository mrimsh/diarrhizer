"""Entry point for the Diarrhizer desktop GUI."""

import sys

from PySide6.QtWidgets import QApplication

from diarrhizer_gui import env_file
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
QLabel#CreditLabel {{
    padding: 8px;
    color: #8a8f98;
    background: #f2f3f6;
    font-size: 11px;
}}
"""


def main() -> int:
    # Core diarrhizer never auto-loads .env (confirmed: no python-dotenv
    # anywhere in src/diarrhizer) - this is GUI-only, additive behavior, and
    # runs before MainWindow so Doctor/New Job see a persisted HF_TOKEN /
    # DIARRHIZER_FFMPEG_PATH immediately without a manual export first.
    env_file.apply_env_file(env_file.REPO_ROOT / ".env")

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
