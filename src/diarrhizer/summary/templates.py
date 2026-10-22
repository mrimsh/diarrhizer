"""Prompt templates for summarizing a transcript.

A template is a named prompt; the transcript (export/result.md) is attached to
it when a summary is requested (service.build_prompt). Built-in templates are
read-only; user templates and the default choice live in a JSON file shared by
the CLI and the GUI.
"""

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from diarrhizer.summary import store_file
from diarrhizer.utils import write_json_atomic

# Overrides where user templates are stored (default: <repo root>/summary_templates.json).
ENV_TEMPLATES_FILE = "DIARRHIZER_SUMMARY_TEMPLATES_FILE"
TEMPLATES_FILENAME = "summary_templates.json"

MAX_NAME_LENGTH = 80


# [SEMANTIC-BEGIN] CONFIG:SUMMARY_TEMPLATES
# @purpose: Named prompts that tell the model what kind of summary to write from a transcript
# @description: A template is just a prompt (plus an optional description); the transcript is attached by
#   SUMMARY:JOB at request time, so a prompt never has to contain it. Names are free text (spaces and
#   Cyrillic allowed) and are the identity: built-in names cannot be overwritten or deleted, and a default
#   pointing at a deleted template falls back to the first built-in instead of breaking every run.
#   TemplateStore keeps user templates plus the default name in a JSON file; a missing file is an empty
#   store, a malformed one raises ValueError naming the file.
# @inputs: template fields (validated on construction), JSON store path (DIARRHIZER_SUMMARY_TEMPLATES_FILE)
# @outputs: PromptTemplate instances, summary_templates.json
# @sideEffects: TemplateStore.load() reads and save() atomically writes the JSON store
# @errors: ValueError for invalid fields, unknown/built-in names, or an unreadable store file
# @see: SUMMARY:JOB, CONFIG:SUMMARY_PROVIDERS, CLI:SUMMARIZE, GUI:SUMMARY_SETTINGS
@dataclass(frozen=True)
class PromptTemplate:
    """A prompt the transcript is attached to.

    Attributes:
        name: Shown in the GUI and used on the CLI; unique, single line
        prompt: What to do with the transcript, e.g. "Write minutes of this meeting..."
        description: Free-text label shown next to the name
    """

    name: str
    prompt: str
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Template name must not be empty")
        if self.name != self.name.strip() or "\n" in self.name or len(self.name) > MAX_NAME_LENGTH:
            raise ValueError(
                f"Template name must be one line of up to {MAX_NAME_LENGTH} characters "
                f"with no leading/trailing spaces: {self.name!r}"
            )
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError(f"Template '{self.name}': the prompt must not be empty")
        if not isinstance(self.description, str):
            raise ValueError(f"Template '{self.name}': description must be text")

    def to_dict(self, include_name: bool = True) -> dict:
        data = {"description": self.description, "prompt": self.prompt}
        return {"name": self.name, **data} if include_name else data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PromptTemplate":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            # A typo in a hand-edited file ("promt") must not silently produce a template without a prompt.
            raise ValueError(f"Template '{data.get('name')}': unknown field(s) {', '.join(unknown)}")
        return cls(**dict(data))


BUILTIN_TEMPLATES: dict[str, PromptTemplate] = {
    template.name: template
    for template in (
        PromptTemplate(
            name="Краткое резюме",
            description="О чём разговор, ключевые мысли, решения и задачи",
            prompt=(
                "Ниже расшифровка записанного разговора с разметкой по спикерам и временными метками. "
                "Составь по ней краткое резюме в формате Markdown:\n"
                "1. О чём был разговор — 2–3 предложения.\n"
                "2. Ключевые темы и выводы.\n"
                "3. Принятые решения и договорённости.\n"
                "4. Задачи: кто, что и к какому сроку (если срок назван).\n"
                "5. Открытые вопросы.\n\n"
                "Опирайся только на текст расшифровки и ничего не выдумывай; раздел, для которого "
                "в разговоре нет материала, пропусти. Расшифровка получена автоматически и содержит "
                "ошибки распознавания — очевидные искажения имён и терминов исправляй по контексту. "
                "Отвечай на языке разговора."
            ),
        ),
        PromptTemplate(
            name="Протокол встречи",
            description="Участники, повестка, решения и таблица задач",
            prompt=(
                "Ниже расшифровка рабочей встречи с разметкой по спикерам и временными метками. "
                "Оформи протокол встречи в формате Markdown:\n"
                "- Участники (по именам из разметки; если спикер не назван по имени — как есть).\n"
                "- Цель встречи.\n"
                "- Обсуждавшиеся вопросы: по пунктам, у каждого — суть обсуждения и время начала.\n"
                "- Решения.\n"
                "- Задачи — таблицей: ответственный | задача | срок.\n"
                "- Открытые вопросы и риски.\n\n"
                "Не добавляй того, чего не было в разговоре. Расшифровка получена автоматически: "
                "очевидные ошибки распознавания в именах и терминах исправляй по контексту. "
                "Отвечай на языке встречи."
            ),
        ),
        PromptTemplate(
            name="Разбор звонка",
            description="Цель, потребности, возражения, договорённости, что улучшить",
            prompt=(
                "Ниже расшифровка телефонного разговора с разметкой по спикерам и временными метками. "
                "Разбери звонок в формате Markdown:\n"
                "1. Цель звонка и его итог.\n"
                "2. Потребности и проблемы собеседника.\n"
                "3. Возражения и сомнения, и как на них ответили.\n"
                "4. Договорённости и следующие шаги (кто, что, когда).\n"
                "5. Общий тон разговора — с короткими цитатами и временными метками.\n"
                "6. Что можно было сделать лучше.\n\n"
                "Опирайся только на текст расшифровки. Она получена автоматически и содержит "
                "ошибки распознавания — очевидные искажения исправляй по контексту. "
                "Отвечай на языке разговора."
            ),
        ),
    )
}

DEFAULT_TEMPLATE_NAME = next(iter(BUILTIN_TEMPLATES))


def store_path() -> Path:
    """Where user templates live: DIARRHIZER_SUMMARY_TEMPLATES_FILE, else <repo root>/summary_templates.json."""
    return store_file.resolve_path(ENV_TEMPLATES_FILE, TEMPLATES_FILENAME)


@dataclass
class TemplateStore:
    """Built-in templates plus user templates and the default template name."""

    path: Path
    default_name: str = DEFAULT_TEMPLATE_NAME
    custom: dict[str, PromptTemplate] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "TemplateStore":
        """Read the store; a missing file is an empty store, a malformed one raises ValueError."""
        path = store_path() if path is None else Path(path)
        default_name, entries = store_file.read_entries(path, "templates", "summary templates")
        custom = {}
        for name, entry in entries.items():
            if name in BUILTIN_TEMPLATES:
                raise ValueError(f"{path}: template '{name}' reuses a built-in template name")
            try:
                custom[name] = PromptTemplate.from_dict({**entry, "name": name})
            except TypeError as e:
                raise ValueError(f"{path}: template '{name}': {e}") from e
        return cls(path=path, default_name=default_name or DEFAULT_TEMPLATE_NAME, custom=custom)

    def templates(self) -> dict[str, PromptTemplate]:
        """Every selectable template, built-ins first."""
        return {**BUILTIN_TEMPLATES, **self.custom}

    def names(self) -> list[str]:
        return list(self.templates())

    def get(self, name: str) -> PromptTemplate:
        templates = self.templates()
        if name not in templates:
            raise ValueError(f"Unknown summary template '{name}'. Available: {', '.join(templates)}")
        return templates[name]

    def default(self) -> PromptTemplate:
        return self.templates().get(self.default_name, BUILTIN_TEMPLATES[DEFAULT_TEMPLATE_NAME])

    def resolve(self, name: str | None) -> PromptTemplate:
        """The named template, or the default one for None."""
        return self.default() if name is None else self.get(name)

    @staticmethod
    def is_builtin(name: str) -> bool:
        return name in BUILTIN_TEMPLATES

    def put(self, template: PromptTemplate) -> None:
        """Add or replace a user template (in memory - call save() to persist)."""
        if self.is_builtin(template.name):
            raise ValueError(f"'{template.name}' is a built-in template and cannot be overwritten")
        self.custom[template.name] = template

    def remove(self, name: str) -> None:
        if self.is_builtin(name):
            raise ValueError(f"'{name}' is a built-in template and cannot be deleted")
        self.custom.pop(name, None)
        if self.default_name == name:
            self.default_name = DEFAULT_TEMPLATE_NAME

    def set_default(self, name: str) -> None:
        self.get(name)
        self.default_name = name

    def save(self) -> None:
        write_json_atomic(
            self.path,
            {
                "default": self.default_name,
                "templates": {
                    name: template.to_dict(include_name=False) for name, template in self.custom.items()
                },
            },
        )
# [SEMANTIC-END] CONFIG:SUMMARY_TEMPLATES
