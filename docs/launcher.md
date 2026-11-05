# Windows launcher

`Diarrhizer.exe` is the graphical entry point for setup and everyday use on Windows 10/11 x64.
The launcher includes its own Python interpreter and Tk interface; it needs neither an existing
Diarrhizer `.venv` nor a manually installed launcher dependency. The processing application remains
a normal Python installation in one selected virtual environment. There is no separate main-GUI exe.

## First launch

1. Open `Diarrhizer.exe` and review the proposed settings.
2. Choose an environment directory (normally `.venv`), compatible **CPython 3.11/3.12 x64**, **CPU**
   or **CUDA 12.1**, the model-cache directory and the results directory. An empty Python field means
   automatic discovery, including the Windows Python launcher and registered interpreters.
3. Select an existing `ffmpeg.exe`, or leave the field empty to search the process PATH. `ffprobe.exe`
   must be in the same directory. The selected bin directory is added to the GUI and worker PATH only.
4. If needed, allow the optional Python and FFmpeg component downloads. Python is downloaded from
   python.org (about 26 MB); the FFmpeg essentials ZIP comes from gyan.dev (about 110 MB). Both are
   verified with SHA-256 before use. The UI displays the actual download size when the server supplies it.
5. Press **Подготовить окружение** and confirm the displayed paths and installation plan.

The stages are Python preparation, venv preparation, FFmpeg preparation, constrained dependency
installation, diagnostics, and GUI start-file creation. Overall progress counts stages; package
installation has an activity indicator because pip does not provide a reliable overall percentage.
Downloads report byte progress. The detailed output is streamed into the window and a durable log.

An existing venv requires explicit confirmation before packages are installed/updated. It is never
deleted or cleared. An unrelated nonempty directory is rejected. An unsupported or broken existing
interpreter is reported; select a working interpreter and a new empty environment directory rather
than replacing the old environment. Changing the Python minor version inside an existing venv is rejected.

The installer uses `requirements/constraints-stable.txt`, `base.txt` and exactly one of `cpu.txt` /
`cuda-cu121.txt`, then installs the application with its `gui` extra. Torch/torchaudio must match
**2.3.1+cpu** or **2.3.1+cu121**. The application is installed normally, without an editable link to the
launcher's temporary extracted files. A launcher rebuild is needed to distribute changed application code.

NVIDIA hardware/driver detection runs before CUDA installation. Final checks also verify Torch CUDA,
CTranslate2 CUDA device discovery and a small CUDA tensor operation. A detected GPU alone is not
treated as proof that the installed CUDA stack works. CPU remains a supported choice.

## Start, shortcut and files

After successful setup the window offers all three actions:

- **Запустить Diarrhizer** starts the main GUI using the created environment.
- **Создать ярлык…** saves a Windows `.lnk`, defaulting to the actual Windows desktop directory
  (including redirected/OneDrive desktops). Its target is the environment's `pythonw.exe`, with the
  generated `gui_entry.py` as its argument. Activation of `.venv` is unnecessary.
- **Открыть папку запуска GUI** opens the folder containing **`Start Diarrhizer.cmd`**. This is a
  file for launching the main GUI, not an exe of the main application. It also works without activation;
  a console may briefly appear. PowerShell argument encoding preserves Unicode, spaces and `%` in paths.

The GUI receives the configured `HF_HOME`, `TORCH_HOME`, config directory and FFmpeg PATH.
Results/device defaults are applied on the first GUI launch for each saved settings revision;
later changes made in the GUI are preserved. Startup through both the shortcut and launcher uses
the same saved settings. Keep the environment and launcher-data paths in place; after moving them,
adjust the setup and recreate the start files/shortcut.

On subsequent launcher runs, readiness is checked in the saved environment itself: Python version,
venv prefix, main GUI/Qt modules and selected Torch package metadata. Full DLL/driver diagnostics
run during setup or repair. A missing/damaged environment disables GUI launch until it is repaired.

## Recovery and storage

One configuration is kept in `%LOCALAPPDATA%\Diarrhizer\launcher\settings.json`. A bundled release
defaults to `%LOCALAPPDATA%\Diarrhizer\.venv`, `models`, and `out`; configuration files such as `.env`
use `%LOCALAPPDATA%\Diarrhizer`. Source launches default to the checkout's `.venv`, `out` and config
directory. `--state <absolute settings.json>` supports an explicitly selected portable/test location;
the UI does not manage multiple environments.

Each setup plan has an append-only `logs/setup-*.log`. A failure records the failed stage and preserves
completed checkpoints. **Повторить: …** continues at that stage, including after closing/reopening the
launcher. Changing settings starts a fresh plan. **Проверить и восстановить среду** explicitly reruns
all stages and reinstalls the constrained dependencies, which can download packages again. It never
deletes models, results, `.env` or the environment directory. Cancellation stops the child process tree;
the interrupted stage remains retryable. Downloads may take until the current network timeout to stop.
An OS lock prevents two launcher windows from installing into the same saved configuration concurrently.

**Открыть журнал** remains available after failure. Main-GUI startup exceptions go into `logs/gui.log`
and a Windows error dialog. Malformed saved state produces an error and is never overwritten automatically.

Setup diagnostics use `doctor --profile asr`, `pip check`, GUI imports and explicit selected-wheel checks.
They need neither a Hugging Face token nor model downloads. For diarization, set `HF_TOKEN` /
`HUGGINGFACE_HUB_TOKEN` locally or use **Настройки → HF-токен** in the main GUI. The launcher never
stores tokens in its JSON or copies `.env` into its distribution. The **Модели** screen remains the
separate optional model-download step. The launcher gives indicative ASR download sizes: tiny about
80 MB, base about 150 MB, large-v3 about 3.1 GB; alignment and diarization require additional models.
These rounded estimates refer to the [tiny](https://huggingface.co/Systran/faster-whisper-tiny/tree/main),
[base](https://huggingface.co/Systran/faster-whisper-base/tree/main) and
[large-v3](https://huggingface.co/Systran/faster-whisper-large-v3/tree/main) repositories, not total cache usage.

## Build and validate

Build on Windows in a separate lightweight environment:

```powershell
py -3.12 -m venv build/launcher-env
build/launcher-env/Scripts/python.exe -m pip install -r requirements/launcher-build.txt
build/launcher-env/Scripts/python.exe tools/build_launcher.py --smoke-test
```

The result is **`dist/Diarrhizer.exe`**. Distribute that file; no sibling source tree or launcher packages
are required. PyInstaller is a build-only dependency and is excluded from processing requirements.
The payload allowlist is the `src` tree, requirements and public package metadata; recordings, model
caches, `.env`, user settings and virtual environments are excluded. CI builds and opens/closes the
standalone UI without installing the processing stack; it does not upload the executable.

For source development, `Diarrhizer.pyw` or `python -m diarrhizer_launcher` runs with standard-library
Tkinter from an already installed Python. This source convenience is not the distribution method for
a machine without Python. `--smoke-test` uses temporary settings, never installs anything and closes
the UI automatically. Launcher regression tests cover failure/resume, existing-environment protection,
verified downloads and real Windows shortcut creation. Local integration validation also creates a fresh
CPU venv, installs the complete constrained stack, runs diagnostics and opens the actual installed GUI
through the launcher, `.lnk` and `.cmd` with temporary Qt settings. A machine with no Python/FFmpeg
at all, and actual CUDA installation, still need separate clean-machine qualification.

Distribution references: [PyInstaller's bundled interpreter](https://pyinstaller.org/en/stable/operating-mode.html),
[Python 3.12.10 installer](https://www.python.org/downloads/release/python-31210/),
[Python per-user installation options](https://docs.python.org/3.12/using/windows.html#installing-without-ui),
[FFmpeg's Windows download providers](https://ffmpeg.org/download.html),
[Gyan essentials builds and checksums](https://www.gyan.dev/ffmpeg/builds/).
Python 3.12.10 is the last 3.12 release with the traditional official Windows installer;
existing compatible interpreters, including later 3.11/3.12 patches, are preferred over downloading it.
