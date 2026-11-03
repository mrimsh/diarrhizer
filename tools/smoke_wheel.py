"""Run outside the checkout after installing its wheel; no models, credentials or user settings."""
import os
import tempfile
from contextlib import chdir
from pathlib import Path


def main():
    with tempfile.TemporaryDirectory(prefix="diarrhizer-wheel-") as temporary, chdir(temporary):
        root = Path(temporary)
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        os.environ["DIARRHIZER_CONFIG_DIR"] = str(root / "config")
        for variable in ("DIARRHIZER_AUDIO_FORMATS_FILE", "DIARRHIZER_SUMMARY_TEMPLATES_FILE",
                         "DIARRHIZER_SUMMARY_PROVIDERS_FILE"):
            os.environ.pop(variable, None)
        from diarrhizer import env_file
        from diarrhizer.summary.providers import ProviderStore
        from diarrhizer_gui.main_window import MainWindow
        from diarrhizer_gui.child_process import kill_all
        from PySide6.QtCore import QSettings
        from PySide6.QtWidgets import QApplication

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
        finally:
            window.close()
            kill_all()
        print("Installed wheel CLI/GUI configuration smoke OK")


if __name__ == "__main__":
    main()
