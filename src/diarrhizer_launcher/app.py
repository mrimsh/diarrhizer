"""Independent Tk launcher; no Qt, torch or Diarrhizer imports."""

import argparse
import os
import queue
import subprocess
import tempfile
import threading
import tkinter as tk
from dataclasses import asdict
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from diarrhizer import __author__, __version__

from .config import LauncherLock, Settings, State, application_source, launcher_home
from .existing import inspect_environment, remember_environment
from .launch import (
    create_shortcut,
    desktop_folder,
    start_file,
    start_gui,
    write_start_files,
)
from .setup import STEPS, SetupWorker, check_ready
from .system import discover_pythons, gpu_description


# [SEMANTIC-BEGIN] LAUNCHER:UI
# @purpose: Provide graphical first-run setup, recovery and subsequent GUI launch without a venv
# @description: Tk ships inside the standalone bundle. Workers report through a queue; all widget
#   changes stay on the Tk thread. Confirmation describes the actual plan and existing-environment
#   package changes. Closing during setup waits for child shutdown before releasing the single-instance lock.
#   Selecting a venv reads its parameters asynchronously; adopting it is an explicit action. Runtime
#   path edits are saved without pip; interpreter/device changes use the confirmed setup flow.
#   Title/header display the shared application release and sole author without loading the processing stack.
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
        settings = self.state.settings if self.state else Settings.defaults(source, state_path.parent)
        self.variables = {key: (tk.BooleanVar(value=value) if isinstance(value, bool) else tk.StringVar(value=value))
                          for key, value in asdict(settings).items()}
        root.title(f"Diarrhizer {__version__} — запуск и настройка")
        root.geometry("980x820")
        root.minsize(850, 740)
        root.protocol("WM_DELETE_WINDOW", self.close)
        style = ttk.Style(root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Title.TLabel", font=("Segoe UI", 19, "bold"))
        style.configure("TButton", padding=(9, 5))
        body = ttk.Frame(root, padding=18)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(5, weight=1)
        ttk.Label(body, text=f"Diarrhizer {__version__} · {__author__}", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        self.status = tk.StringVar(value="Поиск совместимого Python и проверка среды…" if probe
                                   else "Подготовьте окружение, затем откройте основной GUI.")
        ttk.Label(body, textvariable=self.status, wraplength=920).grid(row=1, column=0, sticky="w", pady=(4, 12))

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

        self.log = ScrolledText(body, height=9, wrap="word", font=("Consolas", 9), state="disabled")
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
        if probe:
            threading.Thread(target=self._discover, daemon=True).start()

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
        changed = self.state is None or self._settings() != self.state.settings
        idle = not self.busy and not self.scanning
        for control in self.controls:
            control.configure(state="normal" if idle else "disabled")
        self.setup_button.configure(state="normal" if idle else "disabled")
        label = "Применить параметры" if self.available and changed else "Подготовить окружение"
        if self.state and self.state.failed_step and not changed:
            title = dict(STEPS).get(self.state.failed_step, self.state.failed_step)
            label = "Повторить: " + title
        self.setup_button.configure(text=label)
        self.repair_button.configure(state="normal" if idle and self.state else "disabled")
        self.cancel_button.configure(state="normal" if self.busy else "disabled")
        for button in (self.launch_button, self.shortcut_button, self.folder_button):
            button.configure(state="normal" if idle and self.available and not changed else "disabled")

    def begin(self, *, repair=False):
        if self.busy or self.scanning:
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
            runtime_only = (self.available and self.state and self.state.ready and not repair
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
            if not messagebox.askyesno("Подтвердить настройку", text, parent=self.root, default="no"):
                return
            if self.state is None or settings != self.state.settings:
                self.state = State(settings, managed=managed)
            repair = repair or self.state.ready
            self.available = False
            self.busy = True
            self.worker = SetupWorker(self.state, self.state_path, self.source, self.events.put,
                                      approve_existing=existing)
            self.refresh()
            threading.Thread(target=lambda: self.worker.run(repair=repair), daemon=True).start()
        except (ValueError, OSError) as exc:
            messagebox.showerror("Настройка не началась", str(exc), parent=self.root)

    def stop(self):
        if self.worker:
            self.worker.cancel.set()
            self.status.set("Остановка текущего шага… Журнал и завершённые этапы будут сохранены.")

    def poll(self):
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            kind = event["kind"]
            if kind == "opened":
                self.scanning = False
                if event["folder"] == self.variables["venv"].get().strip():
                    self.available = event["ready"]
                    if event["state"]:
                        self.state = event["state"]
                        for key, value in asdict(self.state.settings).items():
                            self.variables[key].set(value)
                        self.log_path.set("Журнал: " + (self.state.log or str(self.state_path.parent / "logs")))
                    self.status.set(event["message"])
                    self.stage.set("Среда готова" if self.available else "Требуется настройка / восстановление")
                    self.overall["value"] = len(STEPS) if self.available else 0
                self.refresh()
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
                self.available = kind == "done"
                self.detail.stop()
                self.download_status.set("")
                # Probes/downloads resolve automatic paths. Keep those values on failure too,
                # otherwise a retry would look like changed settings and lose its checkpoints.
                for key, value in asdict(self.state.settings).items():
                    self.variables[key].set(value)
                if kind == "done":
                    self.overall["value"] = len(STEPS)
                    self.stage.set("Настройка завершена")
                    self.status.set("Готово. Запустите Diarrhizer, создайте ярлык или откройте папку запуска GUI.")
                else:
                    self.stage.set("Не удалось завершить: " + dict(STEPS).get(event["step"], event["step"]))
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
        if self.busy:
            if not messagebox.askyesno("Остановить настройку?", "Текущий шаг будет прерван. Его можно повторить "
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
    args = parser.parse_args()
    temporary = tempfile.TemporaryDirectory(prefix="diarrhizer-launcher-") if args.smoke_test else None
    state_path = (Path(temporary.name) / "settings.json" if temporary else args.state
                  or launcher_home() / "settings.json").resolve()
    root = tk.Tk()
    lock = None
    try:
        lock = LauncherLock(state_path.parent)
        # Frozen DLL search paths must not leak into the separate Python/FFmpeg processes.
        if os.name == "nt":
            import ctypes
            ctypes.windll.kernel32.SetDllDirectoryW(None)
        Launcher(root, state_path, application_source(), probe=not args.smoke_test)
        if args.smoke_test:
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
