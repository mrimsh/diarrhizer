"""Build the standalone Windows launcher from a separate, lightweight build venv."""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


# [SEMANTIC-BEGIN] LAUNCHER:BUILD
# @purpose: Package a standalone Windows launcher that does not need installed Python or a venv
# @description: One-file PyInstaller bundle includes Tk and a clean application source payload for
#   regular pip installation. Only source, requirements and public package metadata are copied;
#   personal .env, recordings, caches and virtual environments never enter the payload.
# @inputs: build Python with requirements/launcher-build.txt installed
# @outputs: dist/Diarrhizer.exe; optional isolated UI smoke
# @sideEffects: Writes build/dist files and invokes PyInstaller
# @errors: Non-Windows build, missing build dependencies or nonzero build/smoke exit
# @see: LAUNCHER:UI, LAUNCHER:SETUP
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("The Windows launcher must be built on Windows.")
    build = ROOT / "build" / "launcher"
    build.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="payload-", dir=build) as temporary:
        payload = Path(temporary) / "application"
        payload.mkdir()
        for name in ("pyproject.toml", "README.md", "LICENSE"):
            shutil.copy2(ROOT / name, payload / name)
        shutil.copytree(ROOT / "requirements", payload / "requirements")
        shutil.copytree(ROOT / "src", payload / "src",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"))
        command = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onefile", "--windowed",
                   "--name", "Diarrhizer", "--paths", str(ROOT / "src"), "--distpath", str(ROOT / "dist"),
                   "--workpath", str(build / "work"), "--specpath", str(build),
                   "--add-data", str(payload) + ";application"]
        for module in ("PySide6", "torch", "torchaudio", "whisperx", "numpy", "pytest"):
            command.extend(("--exclude-module", module))
        subprocess.run([*command, str(ROOT / "Diarrhizer.pyw")], cwd=ROOT, check=True)
    executable = ROOT / "dist" / "Diarrhizer.exe"
    if args.smoke_test:
        env = dict(os.environ)
        for key in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "TCL_LIBRARY", "TK_LIBRARY"):
            env.pop(key, None)
        windows = Path(os.environ.get("SystemRoot", r"C:\Windows"))
        env["PATH"] = str(windows / "System32") + os.pathsep + str(windows)
        with tempfile.TemporaryDirectory(prefix="diarrhizer-exe-smoke-") as directory:
            subprocess.run([str(executable), "--smoke-test"], cwd=directory, env=env, check=True, timeout=60)
    print(f"Standalone launcher: {executable} ({executable.stat().st_size / 1048576:.1f} MiB)")
    return 0
# [SEMANTIC-END] LAUNCHER:BUILD


if __name__ == "__main__":
    raise SystemExit(main())
