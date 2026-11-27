"""Independent Tk launcher with offline release notes and environment setup."""

import argparse
import http.client
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
from dataclasses import asdict
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from diarrhizer import __author__, __version__
from diarrhizer.changelog import load_history

from .changelog_window import ChangelogWindow
from .config import (
    LauncherLock,
    Settings,
    State,
    application_source,
    atomic_text,
    launcher_home,
)
from .existing import inspect_environment, remember_environment
from .launch import (
    create_shortcut,
    desktop_folder,
    start_file,
    start_gui,
    write_start_files,
)
from .repository import SOURCE_LABEL, latest_application_info
from .self_update import (
    discard_update,
    latest_launcher,
    prepare_update,
    start_replacement,
    version_numbers,
)
from .setup import STEPS, SetupWorker, check_ready, environment_python
from .system import capture, discover_pythons, gpu_description


# [SEMANTIC-BEGIN] LAUNCHER:UI
# @purpose: Provide graphical first-run setup, recovery and subsequent GUI launch without a venv
# @description: Tk ships inside the standalone bundle. Workers report through a queue; all widget
#   changes stay on the Tk thread. Confirmation describes the actual plan and existing-environment
#   package changes. Closing during setup waits for child shutdown before releasing the single-instance lock.
#   Selecting a venv reads its parameters asynchronously; adopting it is an explicit action. Runtime
#   path edits are saved without pip; interpreter/device changes use the confirmed setup flow.
#   Title/header display the shared application release and sole author without loading the processing stack.
#   Self-update downloads/verifies a release off-thread, then confirms restart before handing off
#   replacement to a hidden helper. The single-instance lock is released before the new exe starts.
#   Startup and hourly probes highlight available updates and identify the commit behind the main version.
#   The latest local release and its history can be read before any probes or environment setup.
# @inputs: one local state file, user paths/device/download choices
# @outputs: setup progress/log, retry/repair controls, GUI/shortcut/start-folder actions
# @sideEffects: Native dialogs, background setup/probes and user-requested shortcuts
# @errors: Localized errors with persistent logs; corrupt state is never replaced automatically
# @see: LAUNCHER:CONFIG, LAUNCHER:SETUP, LAUNCHER:START, LAUNCHER:READY
class Launcher:
    def __init__(self, root: tk.Tk, state_path: Path, source: Path, *, probe: bool = True):
        self.root, self.state_path, self.source = root, state_path, source
        self.state = State.load(state_path)
        self.events = queue.Queue()
        self.worker = None
        self.busy = False
        self.scanning = probe
        self.available = False
        self.closing = False
        self.controls = []
        self.self_updating = False
        self.self_update_handoff = False
        self.self_update_cancel = threading.Event()
        settings = self.state.settings if self.state else Settings.defaults(source, state_path.parent)
        self.variables = {key: (tk.BooleanVar(value=value) if isinstance(value, bool) else tk.StringVar(value=value))
                          for key, value in asdict(settings).items()}
        root.title(f"Лаунчер Diarrhizer {__version__} — запуск и настройка")
        root.geometry("980x820")
        root.minsize(850, 740)
        root.protocol("WM_DELETE_WINDOW", self.close)
        style = ttk.Style(root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Title.TLabel", font=("Segoe UI", 19, "bold"))
        style.configure("Update.TLabel", font=("Segoe UI", 10, "bold"), foreground="#211900", background="#ffe680")
        style.configure("TButton", padding=(9, 5))
        body = self._scrollable_body(root)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(5, weight=1)
        header = ttk.Frame(body)
        header.grid(row=0, column=0, rowspan=2, sticky="ew", pady=(0, 8))
        header.columnconfigure(1, weight=1)
        ttk.Label(header, text=f"Лаунчер Diarrhizer {__version__}", style="Title.TLabel").grid(row=0, column=0, sticky="sw")
        ttk.Label(header, text=__author__).grid(row=1, column=0, sticky="nw")
        self.status = tk.StringVar(value="Поиск совместимого Python и проверка среды…" if probe
                                   else "Подготовьте окружение, затем откройте основной GUI.")
        ttk.Label(header, textvariable=self.status, wraplength=280).grid(row=2, column=0, sticky="nw")
        self.entries = load_history()
        update = ttk.LabelFrame(header, text="Последний коммит приложения · main", padding=(9, 4))
        update.grid(row=0, column=1, rowspan=3, sticky="ew", padx=(12, 0))
        update.columnconfigure(0, weight=1)
        self.history_version = tk.StringVar()
        ttk.Label(update, textvariable=self.history_version, wraplength=350).grid(row=0, column=0, sticky="w")
        self.history_button = ttk.Button(update, text="История в этой сборке", command=self._open_history)
        self.history_button.grid(row=0, column=1, sticky="e")
        self.commit_message = ""
        self.commit_revision = ""
        self.commit_button = ttk.Button(update, text="Сообщение коммита…", state="disabled",
                                       command=self._open_commit)
        self.commit_button.grid(row=1, column=0, sticky="w")

        form = ttk.LabelFrame(body, text="Параметры среды", padding=12)
        form.grid(row=2, column=0, sticky="ew")
        form.columnconfigure(1, weight=1)
        self._path_row(form, 0, "Окружение (.venv)", "venv")
        self.open_button = ttk.Button(form, text="Открыть среду", command=self.open_environment)
        self.open_button.grid(row=0, column=3, padx=(8, 0))
        self.controls.append(self.open_button)
        ttk.Label(form, text="Python 3.11 / 3.12 x64").grid(row=1, column=0, sticky="w", padx=(0, 12))
        self.python_combo = ttk.Combobox(form, textvariable=self.variables["python"])
        self.python_combo.grid(row=1, column=1, sticky="ew", pady=3)
        self.controls.append(self.python_combo)
        button = ttk.Button(form, text="Обзор…", command=lambda: self._browse("python", True))
        button.grid(row=1, column=2, padx=(8, 0))
        self.controls.append(button)
        self._check(form, 2, "Пустое поле — найти Python; при отсутствии скачать Python 3.12.10 с python.org (~26 МБ)",
                    "install_python")
        ttk.Label(form, text="Вычисления").grid(row=3, column=0, sticky="w")
        devices = ttk.Frame(form)
        devices.grid(row=3, column=1, columnspan=2, sticky="w", pady=3)
        for name, value in (("CPU", "cpu"), ("CUDA 12.1 · NVIDIA", "cuda")):
            button = ttk.Radiobutton(devices, text=name, value=value, variable=self.variables["device"])
            button.pack(side="left", padx=(0, 16))
            self.controls.append(button)
        button = ttk.Button(devices, text="Проверить GPU", command=self._probe_gpu)
        button.pack(side="left")
        self.controls.append(button)
        self.gpu = tk.StringVar(value="Проверка NVIDIA GPU…" if probe else "GPU ещё не проверен.")
        ttk.Label(form, textvariable=self.gpu, wraplength=720).grid(row=4, column=1, columnspan=2, sticky="w")
        self._path_row(form, 5, "FFmpeg (пусто = поиск)", "ffmpeg", file=True)
        self._check(form, 6, "Если FFmpeg отсутствует, скачать комплект с ffprobe с gyan.dev (~110 МБ)", "install_ffmpeg")
        self._path_row(form, 7, "Кэш моделей", "cache")
        self._path_row(form, 8, "Результаты обработки", "output")
        self._path_row(form, 9, "Настройки приложения", "config_dir")
        self.installed_folder = None
        self.installed_version = ""
        self.installed_status = "не проверено"
        self.local_probe_id = 0
        self.remote_version = ""
        self.remote_status = "проверяем…" if probe else "не проверено"
        self.source_text = tk.StringVar()
        self.app_label = ttk.Label(form, textvariable=self.source_text, wraplength=900)
        self.app_label.grid(row=11, column=0, columnspan=4, sticky="w", pady=(4, 0))
        ttk.Label(form, text="Зависимости: общие ограничения проекта + CPU / CUDA. Установка может занять несколько ГБ.\n"
                  "Модели — отдельно в GUI: tiny ≈80 МБ, base ≈150 МБ, large-v3 ≈3,1 ГБ; выравнивание и диаризация — дополнительно.",
                  wraplength=900).grid(row=10, column=0, columnspan=4, sticky="w", pady=(8, 0))

        progress = ttk.Frame(body)
        progress.grid(row=3, column=0, sticky="ew", pady=(12, 6))
        progress.columnconfigure(0, weight=1)
        self.stage = tk.StringVar(value="Ожидание")
        ttk.Label(progress, textvariable=self.stage).grid(row=0, column=0, sticky="w")
        self.overall = ttk.Progressbar(progress, maximum=len(STEPS))
        self.overall.grid(row=1, column=0, sticky="ew", pady=4)
        self.detail = ttk.Progressbar(progress, mode="indeterminate")
        self.detail.grid(row=2, column=0, sticky="ew")
        self.download_status = tk.StringVar()
        ttk.Label(progress, textvariable=self.download_status).grid(row=3, column=0, sticky="w")

        actions = ttk.Frame(body)
        actions.grid(row=4, column=0, sticky="ew", pady=(2, 8))
        self.setup_button = ttk.Button(actions, text="Подготовить окружение", command=self.begin)
        self.setup_button.pack(side="left")
        self.repair_button = ttk.Button(actions, text="Проверить и восстановить среду", command=lambda: self.begin(repair=True))
        self.repair_button.pack(side="left", padx=8)
        self.cancel_button = ttk.Button(actions, text="Остановить", command=self.stop)
        self.cancel_button.pack(side="right")
        update_actions = ttk.Frame(body)
        update_actions.grid(row=8, column=0, sticky="w", pady=(6, 0))
        self.update_button = ttk.Button(update_actions, text="Обновить приложение", command=lambda: self.begin(update=True))
        self.update_button.pack(side="left")
        self.self_update_button = ttk.Button(update_actions, text="Обновить лаунчер…", command=self.update_launcher)
        self.self_update_button.pack(side="left", padx=8)
        self.launcher_status = tk.StringVar(value=f"Лаунчер {__version__}")
        self.launcher_label = ttk.Label(update_actions, textvariable=self.launcher_status)
        self.launcher_label.pack(side="left")

        self.log = ScrolledText(body, height=4, wrap="word", font=("Consolas", 9), state="disabled")
        self.log.grid(row=5, column=0, sticky="nsew")
        self.log_path = tk.StringVar(value="Журнал: " + (self.state.log if self.state and self.state.log
                                                       else str(state_path.parent / "logs")))
        footer = ttk.Frame(body)
        footer.grid(row=6, column=0, sticky="ew", pady=(5, 10))
        footer.columnconfigure(0, weight=1)
        ttk.Label(footer, textvariable=self.log_path, wraplength=730).grid(row=0, column=0, sticky="w")
        ttk.Button(footer, text="Открыть журнал", command=self.open_log).grid(row=0, column=1)
        finish = ttk.LabelFrame(body, text="После настройки", padding=10)
        finish.grid(row=7, column=0, sticky="ew")
        self.launch_button = ttk.Button(finish, text="Запустить Diarrhizer", command=self.launch)
        self.launch_button.pack(side="left")
        self.shortcut_button = ttk.Button(finish, text="Создать ярлык…", command=self.shortcut)
        self.shortcut_button.pack(side="left", padx=8)
        self.folder_button = ttk.Button(finish, text="Открыть папку запуска GUI", command=self.open_start_folder)
        self.folder_button.pack(side="left")
        self._append("Файл запуска основного GUI после настройки: " + str(start_file(state_path.parent)))
        if self.state and self.state.error:
            self._append("Предыдущая ошибка: " + self.state.error)
        for variable in self.variables.values():
            variable.trace_add("write", lambda *_: self.refresh())
        self.refresh()
        root.after(100, self.poll)
        root.after(1500, self._report_launcher_update)
        self.history_version.set("Проверяем…" if probe else "Нет данных о коммите")
        if probe:
            threading.Thread(target=self._discover, daemon=True).start()
            if getattr(sys, "frozen", False):
                self.launcher_status.set(f"Лаунчер {__version__} · проверяем обновления…")
            self._refresh_update_info()

    def _refresh_update_info(self):
        if self.closing:
            return
        self.root.after(60 * 60 * 1000, self._refresh_update_info)
        if not self.busy:
            self._check_installed()
        self._version_probe("app_version", latest_application_info)
        if getattr(sys, "frozen", False):
            self._version_probe("exe_version", lambda: latest_launcher(__version__))

    def _open_commit(self):
        window = tk.Toplevel(self.root)
        window.title(self.history_version.get())
        window.transient(self.root)
        body = ScrolledText(window, wrap="word", width=76, height=15)
        body.pack(fill="both", expand=True)
        body.insert("1.0", self.commit_message)
        body.configure(state="disabled")
        ttk.Button(window, text="Закрыть", command=window.destroy).pack(pady=5)
        return window

    @staticmethod
    def _scrollable_body(root):
        # At Windows display scaling even the minimum-size form can be taller than the screen.
        viewport = tk.Canvas(root, highlightthickness=0)
        scroll = ttk.Scrollbar(root, orient="vertical", command=viewport.yview)
        scroll.pack(side="right", fill="y")
        viewport.pack(side="left", fill="both", expand=True)
        viewport.configure(yscrollcommand=scroll.set)
        body = ttk.Frame(viewport, padding=18)
        content = viewport.create_window(0, 0, anchor="nw", window=body)

        def resize(_):
            width = viewport.winfo_width()
            height = max(body.winfo_reqheight(), viewport.winfo_height())
            viewport.itemconfigure(content, width=width, height=height)
            viewport.configure(scrollregion=(0, 0, width, height))

        def wheel(event):
            if not isinstance(event.widget, (tk.Text, ttk.Combobox)):
                viewport.yview_scroll(-int(event.delta / 120), "units")

        viewport.bind("<Configure>", resize)
        body.bind("<Configure>", resize)
        root.bind("<MouseWheel>", wheel)
        return body

    def _open_history(self):
        window = ChangelogWindow(self.root, self.entries)
        window.grab_set()
        return window

    def _version_probe(self, kind, read, **context):
        def check():
            try:
                value = read()
                event = {}
                if kind == "app_version":
                    value, event["revision"], event["message"] = value
                if kind in {"installed_version", "app_version"} and value:
                    version_numbers(value)
                event["value"] = value
            except (OSError, ValueError, SyntaxError, RuntimeError, http.client.HTTPException,
                    subprocess.SubprocessError) as exc:
                event = {"error": str(exc)}
            self.events.put({"kind": kind, **context, **event})
        threading.Thread(target=check, daemon=True).start()

    def _check_installed(self):
        folder = self.variables["venv"].get().strip()
        self.local_probe_id += 1
        self.installed_folder = folder
        self.installed_version = ""
        self.installed_status = "проверяем…"
        self._show_versions()

        def read():
            python = environment_python(folder)
            if not folder or not python.is_file():
                return ""
            script = ("from importlib.util import find_spec; "
                      "print(__import__('diarrhizer').__version__ if find_spec('diarrhizer') else '')")
            return capture([str(python), "-I", "-c", script])
        self._version_probe("installed_version", read, probe_id=self.local_probe_id)

    def _show_versions(self):
        selected = self.variables["venv"].get().strip() == self.installed_folder
        local = self.installed_status if selected else "откройте среду"
        newer = (selected and self.installed_version and self.remote_version
                 and version_numbers(self.remote_version) > version_numbers(self.installed_version))
        self.source_text.set(f"Приложение в среде: {local} · {SOURCE_LABEL}: {self.remote_version or self.remote_status}"
                             + (" · доступно обновление" if newer else ""))
        self.app_label.configure(style="Update.TLabel" if newer else "TLabel")

    def _version_result(self, event):
        kind = event["kind"]
        if kind == "installed_version" and event["probe_id"] != self.local_probe_id:
            return
        failed = "error" in event
        if failed:
            self._append("Проверка версии (" + {"installed_version": "среда", "app_version": "main",
                         "exe_version": "лаунчер"}[kind] + "): " + event["error"])
        value = event.get("value")
        if kind == "exe_version":
            status = "проверка недоступна" if failed else (
                f"доступна версия {value.version}" if value else "обновлений нет")
            self.launcher_status.set(f"Лаунчер {__version__} · {status}")
            self.launcher_label.configure(style="Update.TLabel" if value else "TLabel")
        elif kind == "app_version":
            self.remote_version = value or ""
            self.remote_status = "проверка недоступна"
            self._set_commit(value, event.get("revision", ""), event.get("message", ""))
        else:
            self.installed_version = value or ""
            self.installed_status = value or ("версия неизвестна" if failed else "не установлено")
        self._show_versions()

    def _set_commit(self, version, revision, message):
        self.commit_revision = revision
        self.commit_message = message
        title = f"Diarrhizer {version} · {revision[:12]}" if revision else "Нет данных о коммите"
        self.history_version.set(title)
        self.commit_button.configure(state="normal" if message else "disabled")

    def _path_row(self, parent, row, title, key, *, file=False):
        ttk.Label(parent, text=title).grid(row=row, column=0, sticky="w", padx=(0, 12))
        entry = ttk.Entry(parent, textvariable=self.variables[key])
        entry.grid(row=row, column=1, sticky="ew", pady=3)
        button = ttk.Button(parent, text="Обзор…", command=lambda: self._browse(key, file))
        button.grid(row=row, column=2, padx=(8, 0))
        self.controls.extend((entry, button))
        if key == "venv":
            entry.bind("<Return>", lambda _: self.open_environment())

    def _check(self, parent, row, title, key):
        control = ttk.Checkbutton(parent, text=title, variable=self.variables[key])
        control.grid(row=row, column=1, columnspan=2, sticky="w", pady=(0, 6))
        self.controls.append(control)

    def _browse(self, key, file):
        value = (filedialog.askopenfilename(parent=self.root, filetypes=[("Windows executable", "*.exe")])
                 if file else filedialog.askdirectory(parent=self.root, mustexist=False))
        if value:
            self.variables[key].set(str(Path(value)))
            if key == "venv":
                self.open_environment()

    def open_environment(self):
        if self.busy or self.scanning:
            return
        folder = Path(self.variables["venv"].get().strip())
        self.available = False
        self.scanning = True
        self.status.set("Чтение параметров выбранной среды…")
        defaults = self._settings()
        self._check_installed()
        current = self.state
        self.refresh()

        def inspect():
            try:
                state, ready, reason = inspect_environment(folder, current, defaults)
                self.events.put({"kind": "opened", "folder": str(folder), "state": state,
                                 "ready": ready, "message": reason})
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                self.events.put({"kind": "opened", "folder": str(folder), "state": None,
                                 "ready": False, "message": str(exc)})
        threading.Thread(target=inspect, daemon=True).start()

    def _save_selection(self):
        """Called only after explicit apply/use; inspection alone never changes saved state."""
        settings = self.state.settings
        settings.validate(self.source, self.state_path.parent)
        for folder in (settings.cache, settings.output, settings.config_dir):
            Path(folder).mkdir(parents=True, exist_ok=True)
        self.state.log = (self.state.log if self.state.log and Path(self.state.log).resolve().is_relative_to(
            (self.state_path.parent / "logs").resolve()) else "")
        self.state.save(self.state_path)
        # The environment snapshot is optional for an adopted read-only venv.
        try:
            remember_environment(self.state_path, self.state)
        except OSError:
            self._append("Параметры сохранены в лаунчере; папка среды недоступна для записи копии настроек.")
        write_start_files(self.state_path, self.state)

    def _use_selection(self):
        saved = State.load(self.state_path)
        if saved and saved.settings == self.state.settings and saved.ready:
            return True
        if not messagebox.askyesno("Использовать среду?", "Сохранить выбранную среду и её параметры для последующих "
                                  "запусков? Пакеты изменяться не будут.", parent=self.root, default="no"):
            return False
        self._save_selection()
        return True

    def _discover(self):
        try:
            candidates = discover_pythons()
            gpu = gpu_description()
            ready = check_ready(self.state) if self.state else (False, "Подготовьте окружение для первого запуска.")
            self.events.put({"kind": "discovered", "candidates": candidates, "gpu": gpu, "ready": ready})
        except (OSError, ValueError, RuntimeError) as exc:
            self.events.put({"kind": "discovered", "candidates": [], "gpu": "", "ready": (False, str(exc))})

    def _probe_gpu(self):
        self.gpu.set("Проверка NVIDIA GPU…")
        threading.Thread(target=lambda: self.events.put({"kind": "gpu", "gpu": gpu_description()}), daemon=True).start()

    def _settings(self):
        return Settings(**{key: variable.get().strip() if isinstance(variable, tk.StringVar) else variable.get()
                           for key, variable in self.variables.items()})

    def refresh(self):
        self._show_versions()
        changed = self.state is None or self._settings() != self.state.settings
        idle = not self.busy and not self.scanning and not self.self_updating
        for control in self.controls:
            control.configure(state="normal" if idle else "disabled")
        self.setup_button.configure(state="normal" if idle else "disabled")
        label = "Применить параметры" if self.available and changed else "Подготовить окружение"
        if self.state and self.state.failed_step and not changed:
            title = "Загрузка исходников" if self.state.failed_step == "source" else dict(STEPS).get(
                self.state.failed_step, self.state.failed_step)
            label = "Повторить: " + title
        self.setup_button.configure(text=label)
        self.repair_button.configure(state="normal" if idle and self.state else "disabled")
        self.cancel_button.configure(state="normal" if (self.busy or self.self_updating)
                                     and not self.self_update_handoff else "disabled")
        self.update_button.configure(state="normal" if idle and self.available and not changed else "disabled")
        self.self_update_button.configure(state="normal" if idle else "disabled")
        for button in (self.launch_button, self.shortcut_button, self.folder_button):
            button.configure(state="normal" if idle and self.available and not changed else "disabled")

    def begin(self, *, repair=False, update=False):
        if self.busy or self.scanning or self.self_updating:
            return
        try:
            settings = self._settings()
            settings.validate(self.source, self.state_path.parent)
            folder = Path(settings.venv)
            nonempty = folder.exists() and (not folder.is_dir() or any(folder.iterdir()))
            managed = bool(self.state and self.state.managed
                           and Path(self.state.settings.venv).resolve() == folder.resolve())
            existing = nonempty and not managed
            if existing and not (folder / "pyvenv.cfg").is_file():
                raise ValueError("Папка окружения не пуста и не является venv. Выберите пустую папку.")
            runtime_only = (self.available and self.state and self.state.ready and not repair and not update
                            and all(getattr(settings, key) == getattr(self.state.settings, key)
                                    for key in ("venv", "python", "device")))
            if runtime_only:
                if not messagebox.askyesno("Сохранить параметры?", "Сохранить параметры запуска выбранной среды? "
                                          "Пакеты переустанавливаться не будут.", parent=self.root, default="no"):
                    return
                self.state.settings = settings
                self._save_selection()
                self.refresh()
                self.status.set("Параметры сохранены. Доступны запуск GUI, ярлык и папка запуска.")
                return
            text = (f"Окружение: {settings.venv}\nPython: {settings.python or 'найти / скачать 3.12.10 x64'}\n"
                    f"Режим: {settings.device.upper()}\nКэш: {settings.cache}\nРезультаты: {settings.output}\n\n"
                    "Будут установлены зависимости и выполнена диагностика. Модели не скачиваются.")
            if existing:
                text += "\n\nОкружение уже существует. Разрешить установку / обновление его пакетов?"
            if repair:
                text += "\n\nВосстановление переустановит зависимости; возможна повторная загрузка пакетов."
            text += f"\n\nИсточник приложения: {SOURCE_LABEL}. Актуальная версия будет скачана перед установкой."
            if update:
                text = (f"Скачать и установить актуальный Diarrhizer из {SOURCE_LABEL}?\n"
                        "Библиотеки CPU/CUDA переустанавливаться не будут. Закройте основной GUI перед обновлением.")
            if not messagebox.askyesno("Подтвердить настройку", text, parent=self.root, default="no"):
                return
            if self.state is None or settings != self.state.settings:
                self.state = State(settings, managed=managed)
            repair = (repair or self.state.ready) and not update
            self.available = False
            self.busy = True
            self.worker = SetupWorker(self.state, self.state_path, self.source, self.events.put,
                                      approve_existing=existing, remote=True)
            self.refresh()
            threading.Thread(target=lambda: self.worker.run(repair=repair, application_only=update), daemon=True).start()
        except (ValueError, OSError) as exc:
            messagebox.showerror("Настройка не началась", str(exc), parent=self.root)

    def stop(self):
        if self.self_update_handoff:
            return
        if self.self_updating:
            self.self_update_cancel.set()
            self.status.set("Остановка обновления лаунчера… Текущий exe сохранён.")
            return
        if self.worker:
            self.worker.cancel.set()
            self.status.set("Остановка текущего шага… Журнал и завершённые этапы будут сохранены.")

    def update_launcher(self):
        if self.busy or self.scanning or self.self_updating:
            return
        if os.name != "nt" or not getattr(sys, "frozen", False):
            messagebox.showinfo("Обновление лаунчера", "Самообновление доступно в самостоятельном Diarrhizer.exe. "
                                "Эта копия запущена из Python.", parent=self.root)
            return
        self.self_updating = True
        self.self_update_cancel.clear()
        self.status.set("Проверка опубликованного релиза лаунчера на GitHub…")
        self.detail.configure(mode="indeterminate")
        self.detail.start(12)
        self.refresh()

        def prepare():
            try:
                release = latest_launcher(__version__)
                if release is None:
                    self.events.put({"kind": "launcher_current"})
                    return
                self.events.put({"kind": "launcher_downloading", "version": release.version, "size": release.size})
                prepared = prepare_update(release, Path(sys.executable), self.state_path,
                                          self.events.put, self.self_update_cancel)
                self.events.put({"kind": "launcher_prepared", "prepared": prepared})
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                self.events.put({"kind": "launcher_error", "message": str(exc)})
        threading.Thread(target=prepare, daemon=True).start()

    def _report_launcher_update(self):
        notice = self.state_path.parent / "launcher-updates" / "last-result.json"
        try:
            if not notice.exists():
                return
            data = json.loads(notice.read_text(encoding="utf-8"))
            status = data.get("status") if isinstance(data, dict) else None
            if status not in {"updated", "failed"}:
                return
            os.replace(notice, notice.with_name("last-result.seen.json"))
        except (OSError, ValueError):
            return
        message = (f"Лаунчер обновлён до {__version__}." if status == "updated" else
                   "Обновление лаунчера не выполнено. Подробности: logs/launcher-update.log.")
        self.status.set(message)
        self._append(message)
        if status == "failed":
            messagebox.showwarning("Обновление лаунчера", message, parent=self.root)

    def _launcher_event(self, event):
        kind = event["kind"]
        if kind == "launcher_downloading":
            self.status.set(f"Загрузка лаунчера {event['version']} ({event['size'] / 1048576:.1f} МБ)…")
            return
        self.detail.stop()
        self.download_status.set("")
        if kind == "launcher_handoff":
            self.root.destroy()
            return
        self.self_updating = False
        self.self_update_handoff = False
        self.refresh()
        if kind == "launcher_prepared":
            prepared = event["prepared"]
            if self.closing or self.self_update_cancel.is_set():
                discard_update(prepared)
            elif messagebox.askyesno("Обновить и перезапустить лаунчер?",
                                    f"Лаунчер {prepared.version} загружен и проверен.\n"
                                    "Установить его и перезапустить это окно? Предыдущий exe будет сохранён.\n"
                                    "Окружение, модели, результаты и настройки останутся на месте.",
                                    parent=self.root, default="no"):
                self.self_updating = True
                self.self_update_handoff = True
                self.refresh()
                self.status.set("Подготовка замены и перезапуска лаунчера…")

                def handoff():
                    try:
                        start_replacement(prepared)
                        self.events.put({"kind": "launcher_handoff"})
                    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                        discard_update(prepared)
                        self.events.put({"kind": "launcher_error", "message": str(exc)})
                threading.Thread(target=handoff, daemon=True).start()
                return
            else:
                discard_update(prepared)
                self.status.set("Обновление лаунчера отложено. Текущий exe сохранён.")
        elif kind == "launcher_current":
            self.status.set(f"Лаунчер {__version__}: более нового опубликованного релиза нет.")
        elif kind == "launcher_error":
            message = "Обновление лаунчера не выполнено: " + event["message"]
            self.status.set(message)
            self._append(message)
            log = self.state_path.parent / "logs" / "launcher-update.log"
            try:
                log.parent.mkdir(parents=True, exist_ok=True)
                with log.open("a", encoding="utf-8") as stream:
                    stream.write(message + "\n")
            except OSError:
                pass
        if self.closing:
            self.root.destroy()

    def poll(self):
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            kind = event["kind"]
            if kind in {"app_version", "exe_version", "installed_version"}:
                self._version_result(event)
            elif kind.startswith("launcher_"):
                self._launcher_event(event)
                if kind == "launcher_handoff" or self.closing:
                    return
            elif kind == "opened":
                self.scanning = False
                if event["folder"] == self.variables["venv"].get().strip():
                    self.available = event["ready"]
                    if event["state"]:
                        self.state = event["state"]
                        if getattr(sys, "frozen", False):
                            self.source = application_source(Path(self.state.settings.venv))
                        for key, value in asdict(self.state.settings).items():
                            self.variables[key].set(value)
                        self.log_path.set("Журнал: " + (self.state.log or str(self.state_path.parent / "logs")))
                    self.status.set(event["message"])
                    self.stage.set("Среда готова" if self.available else "Требуется настройка / восстановление")
                    self.overall["value"] = len(STEPS) if self.available else 0
                self.refresh()
            elif kind == "source":
                if event["revision"] != self.commit_revision:
                    self._set_commit(event["version"], event["revision"], "")
                self.remote_version = event["version"]
                self._append(f"{SOURCE_LABEL} · версия {event['version']} · коммит {event['revision'][:12]}")
                self._show_versions()
            elif kind == "log":
                self._append(event["message"])
                if self.state and self.state.log:
                    self.log_path.set("Журнал: " + self.state.log)
            elif kind == "stage":
                self.overall["value"] = event["index"]
                self.stage.set(f"Этап {event['index'] + 1} из {event['total']}: {event['message']}")
                self.detail.stop()
                self.detail.configure(mode="indeterminate")
                self.detail.start(12)
                self.download_status.set("Выполняется…")
            elif kind == "download":
                received, total = event["received"], event["total"]
                if total:
                    self.detail.stop()
                    self.detail.configure(mode="determinate", maximum=total, value=received)
                text = f"Скачано {received / 1048576:.1f} МБ"
                self.download_status.set(text + (f" из {total / 1048576:.1f} МБ" if total else ""))
            elif kind in {"done", "error"}:
                self.busy = False
                self.available = kind == "done" or (event.get("step") == "source" and self.state.ready)
                self.detail.stop()
                self.download_status.set("")
                # Probes/downloads resolve automatic paths. Keep those values on failure too,
                # otherwise a retry would look like changed settings and lose its checkpoints.
                for key, value in asdict(self.state.settings).items():
                    self.variables[key].set(value)
                self._check_installed()
                if kind == "done":
                    self.overall["value"] = len(STEPS)
                    self.stage.set("Настройка завершена")
                    self.status.set("Готово. Запустите Diarrhizer, создайте ярлык или откройте папку запуска GUI.")
                else:
                    title = "Загрузка исходников" if event["step"] == "source" else dict(STEPS).get(event["step"], event["step"])
                    self.stage.set("Не удалось завершить: " + title)
                    self.status.set(event["message"] + " Журнал сохранён; шаг можно повторить.")
                self.refresh()
                if self.closing:
                    self.root.destroy()
                    return
            elif kind == "discovered":
                self.scanning = False
                self.python_combo.configure(values=["", *event["candidates"]])
                self.available, reason = event["ready"]
                self.status.set(reason)
                if self.available:
                    self.stage.set("Среда готова")
                    self.overall["value"] = len(STEPS)
                elif self.state and not self.state.ready:
                    self.overall["value"] = len(self.state.completed)
                self.gpu.set(event["gpu"] or "NVIDIA GPU не обнаружен. Доступен режим CPU.")
                self.refresh()
            elif kind == "gpu":
                self.gpu.set(event["gpu"] or "NVIDIA GPU не обнаружен. Выберите CPU или проверьте драйвер NVIDIA.")
        self.root.after(100, self.poll)

    def _append(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        if int(self.log.index("end-1c").split(".")[0]) > 6000:
            self.log.delete("1.0", "3000.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def launch(self):
        try:
            if not self._use_selection():
                return
            process = start_gui(self.state_path, self.state)
            self.status.set("GUI запускается… Ошибки запуска сохраняются в logs/gui.log.")
            self.root.after(2000, lambda: self.status.set(
                "GUI запущен." if process.poll() is None else "GUI завершился. Если окно не открылось, проверьте logs/gui.log."))
        except (ValueError, OSError) as exc:
            messagebox.showerror("Не удалось запустить GUI", str(exc), parent=self.root)

    def shortcut(self):
        try:
            if not self._use_selection():
                return
        except (OSError, ValueError) as exc:
            messagebox.showerror("Параметры среды", str(exc), parent=self.root)
            return
        destination = filedialog.asksaveasfilename(parent=self.root, title="Ярлык основного GUI Diarrhizer",
                                                 initialdir=str(desktop_folder()), initialfile="Diarrhizer.lnk",
                                                 defaultextension=".lnk", filetypes=[("Windows shortcut", "*.lnk")])
        if destination:
            try:
                create_shortcut(self.state_path, self.state, Path(destination))
                self.status.set("Ярлык основного GUI создан: " + destination)
            except (OSError, ValueError, subprocess.SubprocessError):
                messagebox.showerror("Не удалось создать ярлык", "Проверьте папку назначения и доступ к ней. "
                                     "Выберите имя с расширением .lnk.", parent=self.root)

    def open_start_folder(self):
        try:
            if not self._use_selection():
                return
            write_start_files(self.state_path, self.state)
            os.startfile(self.state_path.parent)
            self.status.set("Start Diarrhizer.cmd — файл запуска основного GUI с настроенным окружением.")
        except OSError as exc:
            messagebox.showerror("Папка запуска", str(exc), parent=self.root)

    def open_log(self):
        target = Path(self.state.log) if self.state and self.state.log else self.state_path.parent / "logs"
        try:
            if not target.exists():
                target = self.state_path.parent
            os.startfile(target)
        except OSError as exc:
            messagebox.showerror("Журнал", str(exc), parent=self.root)

    def close(self):
        if self.self_update_handoff:
            self.status.set("Подготовка перезапуска лаунчера…")
            return
        if self.busy or self.self_updating:
            title = "Остановить обновление лаунчера?" if self.self_updating else "Остановить настройку?"
            if not messagebox.askyesno(title, "Текущий шаг будет прерван. Его можно повторить "
                                       "при следующем запуске.", parent=self.root, default="no"):
                return
            self.closing = True
            self.stop()
        else:
            self.root.destroy()


def main() -> int:
    parser = argparse.ArgumentParser(description="Diarrhizer graphical setup launcher")
    parser.add_argument("--state", type=Path, help="Single configuration file (portable/test use)")
    parser.add_argument("--smoke-test", action="store_true", help="Open/close an isolated UI without setup or downloads")
    parser.add_argument("--update-ack", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    temporary = tempfile.TemporaryDirectory(prefix="diarrhizer-launcher-") if args.smoke_test else None
    state_path = (Path(temporary.name) / "settings.json" if temporary else args.state
                  or launcher_home() / "settings.json").resolve()
    if args.update_ack and (args.update_ack.name != "startup.ack" or not args.update_ack.resolve().is_relative_to(
            state_path.parent / "launcher-updates")):
        parser.error("Invalid launcher startup acknowledgement path")
    root = tk.Tk()
    lock = None
    try:
        lock = LauncherLock(state_path.parent)
        # Frozen DLL search paths must not leak into the separate Python/FFmpeg processes.
        if os.name == "nt":
            import ctypes
            ctypes.windll.kernel32.SetDllDirectoryW(None)
        launcher = Launcher(root, state_path, application_source(), probe=not args.smoke_test)
        if args.update_ack:
            root.after(100, lambda: atomic_text(args.update_ack, "ready"))
        if args.smoke_test:
            root.update_idletasks()
            history = root.nametowidget(launcher.history_button.invoke())
            total = len(launcher.entries)
            assert total and launcher.entries[0].version == __version__
            assert launcher.commit_button.instate(["disabled"])
            assert history.newer.instate(["disabled"])
            for _ in range(total - 1):
                history.older.invoke()
            assert history.older.instate(["disabled"])
            assert "Ранние изменения" in history.browser.get("1.0", "end")
            history.newer.invoke()
            assert history.selector.current() == total - 2
            history.selector.current(0)
            history.selector.event_generate("<<ComboboxSelected>>")
            assert history.page["text"] == f"1 / {total}"
            assert launcher.entries[0].highlights()[0] in history.browser.get("1.0", "end")
            history.destroy()
            root.after(800, root.destroy)
        root.mainloop()
        return 0
    except (OSError, ValueError, RuntimeError, tk.TclError) as exc:
        messagebox.showerror("Diarrhizer — ошибка запуска", str(exc), parent=root)
        root.destroy()
        return 1
    finally:
        if lock:
            lock.close()
        if temporary:
            temporary.cleanup()
# [SEMANTIC-END] LAUNCHER:UI
