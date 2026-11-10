"""Read-only destination picker for moving a batch of existing jobs."""

from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit,
    QMessageBox, QPlainTextEdit, QVBoxLayout,
)

from diarrhizer.projects import inherited_settings, list_scopes, plan_job_moves, scope_path


# [SEMANTIC-BEGIN] GUI:MOVE_JOBS
# @purpose: Choose an existing/new project and team for selected history jobs without creating folders
# @description: Empty project selects the library root; empty team selects the project root explicitly.
#   Optional renaming previews the destination template with each job's original date and custom value.
#   Cancel never creates directories or renames files. The accepted choice is executed by History.
# @inputs: library root, selection count and initial project/team
# @outputs: destination() project/team names after acceptance
# @sideEffects: Reads directory markers only; cancel leaves the library untouched
# @errors: Invalid destination names and read errors are displayed in the dialog
# @see: LIBRARY:MOVE_JOBS, GUI:HISTORY_TABLE
class MoveJobsDialog(QDialog):
    def __init__(self, parent, root: Path, count: int, project: str = "", team: str = "", *, jobs=()):
        super().__init__(parent)
        self._root = Path(root)
        self._jobs = list(jobs)
        self.setWindowTitle("Переместить задания")
        self.resize(680, 480)
        self._project = QComboBox()
        self._team = QComboBox()
        for combo in (self._project, self._team):
            combo.setEditable(True)
            combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self._project.addItem("")
        self._project.addItems(list_scopes(self._root))
        self._project.setCurrentText(project)
        self._project.setToolTip("Выберите существующий проект или введите новый; пусто — без проекта")
        self._team.setToolTip("Выберите существующую команду или введите новую; пусто — корень проекта")
        self._preview = QLabel()
        self._preview.setWordWrap(True)
        self._rename = QCheckBox("Переименовать задания и сохранённые файлы по шаблону назначения")
        self._template = QLineEdit()
        self._template.setReadOnly(True)
        self._custom = QLineEdit()
        self._custom.setPlaceholderText("Пусто — сохранить своё значение каждого задания")
        self._names_preview = QPlainTextEdit()
        self._names_preview.setReadOnly(True)
        self._names_preview.setMaximumHeight(170)
        form = QFormLayout()
        form.addRow("Проект:", self._project)
        form.addRow("Команда:", self._team)
        form.addRow("Папка назначения:", self._preview)
        form.addRow(self._rename)
        form.addRow("Шаблон назначения:", self._template)
        form.addRow("Новое значение CUSTOM:", self._custom)
        form.addRow("Предпросмотр:", self._names_preview)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Переместить")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Отмена")
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        hint = QLabel("Выберите проект и команду или введите новые названия. Пустая команда означает корень проекта. "
                      "Дата задания сохраняется. При совпадении имён добавится числовой суффикс. "
                      "Шаблон меняется в параметрах проекта/команды. Внешний оригинал не переименовывается.")
        hint.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"Выбрано заданий: {count}"))
        layout.addLayout(form)
        layout.addWidget(hint)
        layout.addWidget(buttons)
        self._project.currentTextChanged.connect(self._load_teams)
        self._team.currentTextChanged.connect(self._update_preview)
        self._rename.toggled.connect(self._update_preview)
        self._custom.textChanged.connect(self._update_preview)
        self._load_teams()
        self._team.setCurrentText(team)

    def destination(self) -> tuple[str, str]:
        project, team = self._project.currentText().strip(), self._team.currentText().strip()
        scope_path(self._root, project, team)
        return project, team

    def _load_teams(self, *_):
        self._team.blockSignals(True)
        self._team.clear()
        self._team.addItem("")
        project = self._project.currentText().strip()
        try:
            directory = scope_path(self._root, project)
            if project:
                self._team.addItems(list_scopes(directory, teams=True))
        except (ValueError, OSError) as exc:
            self._preview.setText(str(exc))
        self._team.setEnabled(bool(project))
        self._team.blockSignals(False)
        self._update_preview()

    def _update_preview(self, *_):
        try:
            project, team = self.destination()
            self._preview.setText(str(scope_path(self._root, project, team)))
            self._template.setText(inherited_settings(self._root, project, team)["naming_template"])
            self._custom.setEnabled(self._rename.isChecked())
            plans = plan_job_moves(self._root, self._jobs, project=project, team=team, **self.rename_options())
            self._names_preview.setPlainText("\n".join(f"{a.name} → {b.name}" for a, b in plans.items()))
        except (ValueError, OSError) as exc:
            self._preview.setText(str(exc))
            self._names_preview.setPlainText(str(exc))

    def rename_options(self):
        return {"rename": self._rename.isChecked(), "custom": self._custom.text() or None}

    def _accept(self):
        try:
            project, team = self.destination()
            plan_job_moves(self._root, self._jobs, project=project, team=team, **self.rename_options())
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "Переместить задания", str(exc))
            return
        self.accept()
# [SEMANTIC-END] GUI:MOVE_JOBS
