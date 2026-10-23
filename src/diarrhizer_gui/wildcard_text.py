"""Words for what the GUI says about the {{wildcards}} of a prompt template.

Pure functions over diarrhizer.summary.wildcards, so the template editor, the summary dialog and the request
preview say the same thing the same way. The list of wildcards is the registry's (WILDCARDS) - nothing here
names one except {{transcript}} and {{note}}, whose rules the messages explain.
"""

from diarrhizer.summary.wildcards import WILDCARDS, Wildcard, find_wildcards


def token(name: str) -> str:
    return "{{" + name + "}}"


def short_description(wildcard: Wildcard) -> str:
    """One line for the editor's list: the registry's Russian text, else its (English) description."""
    return wildcard.description_ru or wildcard.description


def available_text() -> str:
    return ", ".join(token(name) for name in WILDCARDS)


def unknown_message(prompt: str) -> str:
    """What is wrong with the unknown wildcards of a prompt (with their lines); "" when there are none."""
    unknown = []
    for ref in find_wildcards(prompt):
        label = f"{token(ref.name)} (строка {ref.line})"
        if not ref.known and label not in unknown:
            unknown.append(label)
    if not unknown:
        return ""
    noun = "Неизвестный wildcard" if len(unknown) == 1 else "Неизвестные wildcards"
    return (
        f"{noun}: {', '.join(unknown)}. Доступные: {available_text()}. "
        "Чтобы написать «{{» буквально, поставьте перед ней обратную косую черту: \\{{."
    )


def missing_transcript_message(prompt: str) -> str:
    """The warning for a prompt that does not place the transcript; "" when it does."""
    if any(ref.name == "transcript" for ref in find_wildcards(prompt)):
        return ""
    return f"В шаблоне нет {token('transcript')}: расшифровка будет добавлена в конец запроса."


def note_placement_message(prompt: str, note: str) -> str:
    """The warning for a note the template has no place for; "" when the note is empty or has its place."""
    if not note.strip():
        return ""
    names = {ref.name for ref in find_wildcards(prompt)}
    if "note" in names:
        return ""
    where = (
        "в самый конец запроса, после текста шаблона"
        if "transcript" in names
        else "в конец запроса, перед расшифровкой"
    )
    return f"В шаблоне нет {token('note')}: заметка будет добавлена {where}."
