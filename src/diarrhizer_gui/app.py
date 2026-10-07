"""Entry point for the Diarrhizer desktop GUI."""

import multiprocessing
import sys

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
QPushButton#JobStatusButton {{
    text-align: left;
    padding: 8px 12px;
    margin: 0 8px 4px 8px;
    border: 1px solid #d7dae2;
    border-radius: 6px;
    background: #ffffff;
    color: #1a1d24;
}}
QPushButton#JobStatusButton:hover {{
    border-color: {ACCENT};
}}
QLabel#CreditLabel {{
    padding: 8px;
    color: #8a8f98;
    background: #f2f3f6;
    font-size: 11px;
}}
QWidget#Sidebar {{
    background: #f2f3f6;
}}
"""


def main() -> int:
    # Imported here, not at module level: pipeline jobs and doctor checks run
    # in spawned child processes, which re-import this module as __mp_main__ -
    # keeping the module itself light keeps every child's startup light too.
    from PySide6.QtWidgets import QApplication

    from diarrhizer_gui import env_file
    from diarrhizer_gui.child_process import kill_all
    from diarrhizer_gui.main_window import MainWindow

    # Same load as the CLI (repo-root .env, then cwd .env; process env wins).
    # Children inherit os.environ, so they see the same token/paths.
    env_file.load_project_env()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    app.setOrganizationName("Diarrhizer")
    app.setApplicationName("DiarrhizerGUI")
    # Pipeline/doctor children are not daemons; never leave one running.
    app.aboutToQuit.connect(kill_all)

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
