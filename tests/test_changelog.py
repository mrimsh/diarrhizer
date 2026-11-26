"""Check the shared document through both native renderers."""

import re
import tkinter as tk

import pytest

from diarrhizer import __version__, changelog
from diarrhizer_launcher.changelog_window import ChangelogWindow


def _words(text):
    # Qt represents list markers in the document layout, Tk includes them in the text.
    return re.sub(r"^(?:• |\d+\. )", "", text, flags=re.MULTILINE).split()


@pytest.mark.parametrize("contents", [None, b"\xff", b"# Empty"])
def test_unavailable_document(tmp_path, monkeypatch, qapp, contents):
    from diarrhizer_gui.screens.changelog_dialog import ChangelogDialog

    path = tmp_path / "CHANGELOG.md"
    if contents is not None:
        path.write_bytes(contents)
    monkeypatch.setattr(changelog, "_document", lambda: path)
    qt = ChangelogDialog()
    root = tk.Tk()
    root.withdraw()
    try:
        window = ChangelogWindow(root, changelog.load_history())
        assert window.browser.get("1.0", "end-1c") == qt.browser.toPlainText() == changelog.UNAVAILABLE
        assert not any(w.isEnabled() for w in (qt.newer, qt.older, qt.selector))
        assert all(w.instate(["disabled"]) for w in (window.newer, window.older, window.selector))
    finally:
        qt.close()
        root.destroy()


def test_rendered_versions_agree_from_another_working_directory(qapp, tmp_path, monkeypatch):
    from diarrhizer_gui.screens.changelog_dialog import ChangelogDialog

    root = tk.Tk()
    root.withdraw()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "CHANGELOG.md").write_text("## 9.99\n\n- Wrong document", encoding="utf-8")
    qt = ChangelogDialog()
    try:
        window = ChangelogWindow(root, changelog.load_history())
        assert qt.entries == window.entries
        assert qt.entries[0].version == __version__ and qt.entries[-1].title == "Ранние изменения"
        versions = [tuple(map(int, e.version.split("."))) for e in qt.entries if e.version]
        assert versions == sorted(set(versions), reverse=True)
        for index, entry in enumerate(qt.entries):
            qt.selector.setCurrentIndex(index)
            window.show_page(index)
            assert qt.page.text() == window.page["text"]
            assert qt.older.isEnabled() == window.older.instate(["!disabled"]) == (index < len(qt.entries) - 1)
            assert qt.newer.isEnabled() == window.newer.instate(["!disabled"]) == (index > 0)
            assert _words(qt.browser.toPlainText()) == _words(window.browser.get("1.0", "end")), entry.title
    finally:
        qt.close()
        root.destroy()


def test_wrapped_highlights_optional_date_and_older_notes(tmp_path):
    path = tmp_path / "CHANGELOG.md"
    path.write_text("## 3.0\n\n- **Первая** строка\n  с `кодом`.\n- Вторая.\n\n"
                    "#### Подробности\n\nТекст.\n\n## Ранние изменения\n\n- Архив.", encoding="utf-8-sig")
    latest, early = changelog.load_history(path)
    assert latest.version == "3.0" and latest.date == ""
    assert latest.highlights() == ["Первая строка с кодом.", "Вторая."]
    assert "Подробности" in latest.body and early.version == ""
