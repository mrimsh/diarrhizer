"""Run outside the checkout after installing its wheel; no models, credentials or user settings."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def check_gui():
    root = Path.cwd()
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["DIARRHIZER_CONFIG_DIR"] = str(root / "config")
    for variable in ("DIARRHIZER_AUDIO_FORMATS_FILE", "DIARRHIZER_SUMMARY_TEMPLATES_FILE",
                     "DIARRHIZER_SUMMARY_PROVIDERS_FILE"):
        os.environ.pop(variable, None)
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from diarrhizer import __version__, env_file
    from diarrhizer.summary.providers import ProviderStore
    from diarrhizer_gui.child_process import kill_all
    from diarrhizer_gui.main_window import MainWindow
    from diarrhizer_gui.screens.changelog_dialog import ChangelogDialog

    assert not env_file._SOURCE_CHECKOUT, "An editable install was imported instead of the wheel"
    assert env_file.config_dir() == root / "config"
    env_file.write_env_file(env_file.config_dir() / ".env", {"SMOKE_VALUE": "synthetic"})
    assert env_file.read_env_file(env_file.config_dir() / ".env") == {"SMOKE_VALUE": "synthetic"}
    store = ProviderStore.load()
    store.save()
    assert store.path.parent == root / "config"
    app = QApplication([])
    app.setOrganizationName("DiarrhizerWheelSmoke")
    app.setApplicationName("Smoke")
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(root / "qt"))
    window = MainWindow()
    try:
        window.show()
        app.processEvents()
        assert window.isVisible()
        window.version_button.click()
        history = window.findChild(ChangelogDialog)
        assert history.isVisible() and history.entries[0].version == __version__
        assert history.entries[0].highlights()[0] in history.browser.toPlainText()
        assert not history.newer.isEnabled()
        total = history.selector.count()
        for _ in range(total - 1):
            history.older.click()
        assert not history.older.isEnabled()
        assert "Ранние изменения" in history.browser.toPlainText()
        history.newer.click()
        assert history.selector.currentIndex() == total - 2
        history.selector.setCurrentIndex(0)
        assert history.page.text() == f"1 / {total}"
        assert history.entries[0].title in history.browser.toPlainText()
        history.close()
    finally:
        window.close()
        kill_all()
    print("Installed wheel configuration and release history navigation OK")


if __name__ == "__main__":
    if sys.argv[1:] == ["--check-gui"]:
        check_gui()
    else:
        # Windows can retain Qt directory handles until the GUI process exits.
        with tempfile.TemporaryDirectory(prefix="diarrhizer-wheel-") as workdir:
            subprocess.run([sys.executable, "-I", str(Path(__file__).resolve()), "--check-gui"],
                           cwd=workdir, check=True)
