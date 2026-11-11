"""Project/team selection and portable scope settings (no pipeline imports)."""

from pathlib import Path
from datetime import datetime

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QMessageBox, QPushButton, QVBoxLayout, QWidget,
)

from diarrhizer.projects import (
    PROJECT_FILE, TEAM_FILE, DEFAULT_TEMPLATE, DEFAULT_STORAGE, change_participant,
    ensure_scope, inherited_settings, list_scopes, participants, read_config,
    render_name, rename_scope, save_scope, scope_file, scope_path,
)

STORAGE_LABELS = (("С исходным именем", "original"), ("По шаблону задания", "template"),
                  ("Не сохранять", "discard"))
KIND_LABELS = {"original": "Оригинальный файл", "working": "Рабочий WAV", "archive": "Архивная копия"}


def storage_fields(form):
    fields = {}
    for kind, label in KIND_LABELS.items():
        combo = QComboBox()
        for text, value in STORAGE_LABELS:
            combo.addItem(text, value)
        form.addRow(label + ":", combo)
        fields[kind] = combo
    return fields


def set_storage(fields, values):
    for kind, combo in fields.items():
        combo.setCurrentIndex(max(0, combo.findData(values.get(kind, DEFAULT_STORAGE[kind]))))


# [SEMANTIC-BEGIN] GUI:PROJECT_SETTINGS
# @purpose: Edit directory-local defaults and explicitly manage reusable team participant names
# @description: Project roots and named teams have independent rosters. Rename moves the actual scope;
#   active/queued descendants block edits that move it. Merely opening/cancelling an existing scope is read-only.
# @inputs: scope directory, busy predicate
# @outputs: saved defaults, renamed directory, explicit roster changes
# @sideEffects: Writes scope JSON, renames scope on Save, edits roster only on Add/Remove
# @errors: File/config errors displayed in the dialog
# @see: LIBRARY:SCOPES, GUI:PROJECT_SELECTION
class ScopeDialog(QDialog):
    scope_renamed = Signal(Path, Path)

    def __init__(self, parent, directory: Path, is_busy=None):
        super().__init__(parent)
        self.directory = Path(directory)
        self._is_busy = is_busy or (lambda p: False)
        self.setWindowTitle("Параметры проекта / команды")
        self.resize(580, 560)
        data = read_config(scope_file(self.directory))
        self._name = QLineEdit(self.directory.name)
        self._name.setEnabled(scope_file(self.directory).name in (PROJECT_FILE, TEAM_FILE))
        self._template = QLineEdit(data.get("naming_template", DEFAULT_TEMPLATE))
        self._default_team = QComboBox()
        self._default_team.addItem("")
        self._default_team.addItems(list_scopes(self.directory, teams=True))
        self._default_team.setCurrentText(data.get("default_team", ""))
        form = QFormLayout()
        form.addRow("Название:", self._name)
        form.addRow("Шаблон названия:", self._template)
        hint = QLabel("PROJECT · TEAM · YYYY_MM_DD · HH_MM_SS · CUSTOM · SOURCE\n"
                      "Также: {{project}}, {{team}}, {{date}}, {{time}}, {{custom}}, {{source}}")
        hint.setWordWrap(True)
        form.addRow("", hint)
        if scope_file(self.directory).name == PROJECT_FILE:
            form.addRow("Команда по умолчанию:", self._default_team)
        self._storage = storage_fields(form)
        # Show inherited values; saving makes this scope's choices explicit.
        marker = scope_file(self.directory).name
        if marker == TEAM_FILE:
            defaults = inherited_settings(self.directory.parent.parent, self.directory.parent.name, self.directory.name)
        elif marker == PROJECT_FILE:
            defaults = inherited_settings(self.directory.parent, self.directory.name)
        else:
            defaults = inherited_settings(self.directory)
        self._template.setText(defaults["naming_template"])
        set_storage(self._storage, defaults["storage"])
        self._people = QListWidget()
        self._people.addItems(participants(self.directory))
        self._person = QLineEdit()
        self._person.setPlaceholderText("Новое имя участника")
        add = QPushButton("Добавить")
        add.clicked.connect(lambda: self._change_person(False))
        remove = QPushButton("Удалить выбранное имя")
        remove.clicked.connect(lambda: self._change_person(True))
        row = QHBoxLayout()
        row.addWidget(self._person, 1)
        row.addWidget(add)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(QLabel("Известные участники этой команды / корня:"))
        layout.addWidget(self._people, 1)
        layout.addLayout(row)
        layout.addWidget(remove)
        layout.addWidget(QLabel("Добавление и удаление имён сохраняется сразу; имена в заданиях сохраняются отдельно."))
        layout.addWidget(buttons)

    def _change_person(self, remove):
        current = self._people.currentItem()
        name = current.text() if remove and current else self._person.text().strip() if not remove else ""
        if not name:
            return
        try:
            change_participant(self.directory, name, remove=remove)
            self._people.clear()
            self._people.addItems(participants(self.directory))
            self._person.clear()
        except (OSError, ValueError, RuntimeError) as exc:
            QMessageBox.warning(self, "Участники", str(exc))

    def _save(self):
        try:
            render_name(self._template.text(), project="Проект", team="Команда", custom="Значение", source="Запись")
            data = {"naming_template": self._template.text(),
                    "storage": {k: c.currentData() for k, c in self._storage.items()}}
            if scope_file(self.directory).name == PROJECT_FILE:
                data["default_team"] = self._default_team.currentText()
            if self._name.isEnabled() and self._name.text() != self.directory.name:
                if self._is_busy(self.directory):
                    raise ValueError("В этом проекте/команде есть выполняющиеся или ожидающие задания")
                previous = self.directory
                self.directory = rename_scope(self.directory, self._name.text())
                self.scope_renamed.emit(previous, self.directory)
            from diarrhizer.artifacts import job_write_lock
            with job_write_lock(self.directory):
                save_scope(self.directory, data)
        except (OSError, ValueError, RuntimeError) as exc:
            QMessageBox.warning(self, "Параметры", str(exc))
            return
        self.accept()
# [SEMANTIC-END] GUI:PROJECT_SETTINGS


# [SEMANTIC-BEGIN] GUI:PROJECT_SELECTION
# @purpose: Plan new jobs in existing/new projects and teams with naming preview and per-type retention
# @description: Editable selectors create scopes only when explicitly editing settings or starting a job.
#   Defaults inherit library -> project -> named team. Planning and preview never write job artifacts.
# @inputs: results-folder getter and optional busy predicate
# @outputs: runner kwargs, changed signal
# @sideEffects: Only explicit settings actions write directories/settings
# @errors: Invalid selections returned as ValueError for the parent form to display
# @see: LIBRARY:JOB_IDENTITY, GUI:PROJECT_SETTINGS, GUI:JOB_QUEUE
class ProjectPanel(QWidget):
    changed = Signal()
    scope_renamed = Signal(Path, Path)

    def __init__(self, root_getter, is_busy=None):
        super().__init__()
        self._root_getter = root_getter
        self._is_busy = is_busy
        self._recorded_at = None
        self._source_name = "запись"
        self.project = QComboBox()
        self.team = QComboBox()
        for combo in (self.project, self.team):
            combo.setEditable(True)
            combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.project.setToolTip("Выберите проект или введите название нового; пусто — без проекта")
        self.team.setToolTip("Выберите команду или введите название новой; пусто — корень проекта")
        self.template = QLineEdit(DEFAULT_TEMPLATE)
        self.custom = QLineEdit()
        self.custom.setPlaceholderText("Значение CUSTOM (необязательно)")
        self.preview = QLabel()
        self.preview.setWordWrap(True)
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        for label, combo, kind in (("Проект:", self.project, "project"), ("Команда:", self.team, "team")):
            row = QHBoxLayout()
            row.addWidget(combo, 1)
            button = QPushButton("Параметры…")
            button.clicked.connect(lambda checked=False, k=kind: self._edit(k))
            row.addWidget(button)
            form.addRow(label, row)
        root_button = QPushButton("Параметры папки заданий…")
        root_button.clicked.connect(lambda: self._edit("root"))
        form.addRow("", root_button)
        form.addRow("Шаблон названия:", self.template)
        form.addRow("", QLabel("PROJECT · YYYY_MM_DD · HH_MM_SS · TEAM · CUSTOM · SOURCE"))
        form.addRow("Своё значение:", self.custom)
        self.storage = storage_fields(form)
        form.addRow("Будет создано:", self.preview)
        self.project.currentTextChanged.connect(self._project_changed)
        self.team.currentTextChanged.connect(self._team_changed)
        self.template.textChanged.connect(self._preview)
        self.custom.textChanged.connect(self._preview)
        for combo in self.storage.values():
            combo.currentIndexChanged.connect(self.changed)
        self.refresh()

    def root(self):
        return Path(self._root_getter())

    @staticmethod
    def _fill(combo, values, current):
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("")
        combo.addItems(values)
        combo.setCurrentText(current)
        combo.blockSignals(False)

    def refresh(self, reload_defaults=False):
        previous = (self.template.text(), {k: c.currentData() for k, c in self.storage.items()})
        preserve = not reload_defaults and getattr(self, "_loaded_root", None) == self.root()
        try:
            self._fill(self.project, list_scopes(self.root()), self.project.currentText())
            self._project_changed(self.project.currentText(), preserve_team=True)
            if preserve:
                self.template.setText(previous[0])
                set_storage(self.storage, previous[1])
            self._loaded_root = self.root()
        except (OSError, ValueError) as exc:
            self.preview.setText(str(exc))

    def _project_changed(self, _text, preserve_team=False):
        try:
            project = self.project.currentText().strip()
            directory = scope_path(self.root(), project)
            defaults = inherited_settings(self.root(), project)
            wanted = self.team.currentText() if preserve_team else defaults["default_team"]
            self._fill(self.team, list_scopes(directory, teams=True) if project else [], wanted)
            self.team.setEnabled(bool(project))
            self._team_changed()
        except (OSError, ValueError) as exc:
            self.preview.setText(str(exc))
            self.changed.emit()

    def _team_changed(self, *_):
        try:
            defaults = inherited_settings(self.root(), self.project.currentText().strip(), self.team.currentText().strip())
            self.template.setText(defaults["naming_template"])
            set_storage(self.storage, defaults["storage"])
            self._preview()
        except (OSError, ValueError) as exc:
            self.preview.setText(str(exc))
            self.changed.emit()

    def options(self):
        project = self.project.currentText().strip()
        team = self.team.currentText().strip()
        scope_path(self.root(), project, team)
        render_name(self.template.text(), project=project, team=team, custom=self.custom.text(), source=self._source_name)
        return {"project": project, "team": team, "naming_template": self.template.text(),
                "custom": self.custom.text(), "storage": {k: c.currentData() for k, c in self.storage.items()}}

    def _preview(self, *_):
        try:
            options = self.options()
            name = render_name(options["naming_template"], project=options["project"], team=options["team"],
                               custom=options["custom"], source=self._source_name,
                               date=self._recorded_at.strftime("%Y_%m_%d") if self._recorded_at else None,
                               time=self._recorded_at.strftime("%H_%M_%S") if self._recorded_at else None)
            self.preview.setText(str(scope_path(self.root(), options["project"], options["team"]) / name))
        except (ValueError, OSError) as exc:
            self.preview.setText(str(exc))
        self.changed.emit()

    def set_recording_context(self, source, recorded_at):
        self._source_name = Path(source).stem
        self._recorded_at = datetime.fromisoformat(recorded_at)
        self._preview()

    def _edit(self, kind):
        try:
            project = self.project.currentText().strip() if kind != "root" else ""
            team = self.team.currentText().strip() if kind == "team" else ""
            if kind == "project" and not project:
                raise ValueError("Введите или выберите проект")
            if kind == "team" and not team:
                raise ValueError("Введите или выберите команду; параметры корня доступны у проекта")
            directory = ensure_scope(self.root(), project, team)
            dialog = ScopeDialog(self, directory, self._is_busy)
            dialog.scope_renamed.connect(self.scope_renamed)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                if kind == "project":
                    self.project.setCurrentText(dialog.directory.name)
                elif kind == "team":
                    self.team.setCurrentText(dialog.directory.name)
            self.refresh(reload_defaults=True)
        except (ValueError, OSError, RuntimeError) as exc:
            QMessageBox.warning(self, "Проекты и команды", str(exc))
# [SEMANTIC-END] GUI:PROJECT_SELECTION
