"""Offline release history shared by the Qt application and the standalone Tk launcher."""

import re
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

TITLE = "История изменений"
UNAVAILABLE = "Не удалось прочитать историю изменений (CHANGELOG.md)."
NEWER = "К более новым"
OLDER = "К более старым"
_ENTRY = re.compile(
    r"^## (?:(?P<version>\d+(?:\.\d+)+)(?: — (?P<date>\d{4}-\d{2}-\d{2}))?|Ранние изменения)$",
    re.MULTILINE,
)
_INLINE = re.compile(r"`([^`]+)`|\*\*(.+?)\*\*|\[([^\]]+)\]\((https?://[^ )]+)\)")


# [SEMANTIC-BEGIN] APP:CHANGELOG
# @purpose: Read the same UTF-8 Markdown pages and highlights in every installation
# @description: Only release headings and the early-history heading start a page.
#   Resolve the checkout, frozen payload or wheel resource independently of cwd.
# @outputs: Entries in document order, or an empty tuple for the UI error state
# @sideEffects: Reads local package metadata and the changelog
@dataclass(frozen=True)
class Entry:
    title: str
    version: str
    date: str
    body: str

    def highlights(self, count: int = 3) -> list[str]:
        items = []
        for kind, line in blocks(self.body):
            if kind == "heading" and items:
                break
            if kind == "bullet":
                items.append("".join(label for _, label, _ in inline(line)))
            if len(items) == count:
                break
        return items


def _document() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / "application" / "CHANGELOG.md"
    source = Path(__file__).resolve().parent.parent
    if source.name == "src" and (source.parent / "pyproject.toml").is_file():
        return source.parent / "CHANGELOG.md"
    for file in distribution("diarrhizer").files or ():
        if file.parts[-3:] == ("share", "diarrhizer", "CHANGELOG.md"):
            return file.locate()
    raise FileNotFoundError("CHANGELOG.md")


def load_history(path: Path | None = None) -> tuple[Entry, ...]:
    try:
        document = (path or _document()).read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError, PackageNotFoundError):
        return ()
    sections = []
    end = len(document)
    for heading in reversed(list(_ENTRY.finditer(document))):
        sections.append(Entry(heading[0][3:], heading["version"] or "", heading["date"] or "",
                              document[heading.end():end].strip()))
        end = heading.start()
    return tuple(reversed(sections))


def blocks(markdown: str):
    """Join soft-wrapped lines, preserving lists, headings and paragraph boundaries."""
    kind, lines = "paragraph", []
    for line in markdown.splitlines():
        item = re.match(r"^(#{1,6} |[-*] |\d+\. )(.*)", line)
        if (item or not line.strip()) and lines:
            yield kind, " ".join(lines)
            kind, lines = "paragraph", []
        if item:
            marker, value = item.groups()
            if marker.startswith("#"):
                kind = "heading"
            elif marker[0].isdigit():
                kind, value = "numbered", marker + value
            else:
                kind = "bullet"
            lines.append(value)
        elif line.strip():
            lines.append(line.strip())
    if lines:
        yield kind, " ".join(lines)


def inline(line: str):
    """Yield styled spans for the Markdown used in release notes."""
    start = 0
    for match in _INLINE.finditer(line):
        if match.start() > start:
            yield "", line[start:match.start()], ""
        code, bold, label, url = match.groups()
        if code is not None:
            yield "code", code, ""
        elif bold is not None:
            yield "bold", bold, ""
        else:
            yield "link", label, url
        start = match.end()
    if start < len(line):
        yield "", line[start:], ""
# [SEMANTIC-END] APP:CHANGELOG
