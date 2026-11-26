"""Manual edits across the result screen, export-only runs and prompt consumers."""

import json
import sys
from types import SimpleNamespace

import pytest

from diarrhizer import cli
from diarrhizer.artifacts import bundle_valid
from diarrhizer.pipeline.runner import run_pipeline
from diarrhizer.pipeline.stages.export import ExportStage
from diarrhizer.pipeline.stages.merge import MergeStage
from diarrhizer.summary.service import read_transcript
from diarrhizer.summary.wildcards import PromptContext
from diarrhizer.transcript import (
    MANUAL,
    SOURCE,
    change_speaker,
    change_text,
    load_transcript,
    merge_segments,
    resolve_conflict,
    save_revision,
    split_segment,
)
from diarrhizer.utils import read_json, write_json_atomic


@pytest.fixture
def job(tmp_path):
    root = tmp_path / "call"
    segments = [
        {"start": 0, "end": 4, "speaker_id": "Speaker_00", "text": "Первый ответ"},
        {"start": 4, "end": 6, "speaker_id": "Speaker_01", "text": "Второй"},
        {"start": 6, "end": 7, "speaker_id": "Speaker_00", "text": "Конец",
         "words": [{"start": 6, "end": 7, "word": "Конец", "speaker_id": "Speaker_00"}]},
    ]
    write_json_atomic(root / SOURCE, {"segments": segments})
    write_json_atomic(root / "meta/run.json", {"input_path": "call.wav", "pipeline_config": {"language": "ru"}})
    write_json_atomic(root / "speakers.json", {"Speaker_01": "Анна"})
    return root


def export(job):
    return run_pipeline(job_dir=job, out_dir=job.parent, stages=[ExportStage()], force_stage="export")


def test_split_merge_and_word_alignment(job):
    first, second, last = load_transcript(job).segments
    for time in (0, 4, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="внутри"):
            split_segment(first, 2, time)
    for cursor in (0, len(first["text"])):
        with pytest.raises(ValueError, match="курсор"):
            split_segment(first, cursor, 1)
    left, right = split_segment(first, 6, 1.25)
    assert (left["start"], left["end"], right["start"], right["end"]) == (0, 1.25, 1.25, 4)
    assert left["speaker_id"] == right["speaker_id"] == first["speaker_id"]
    with pytest.raises(ValueError, match="спикера"):
        merge_segments(right, second)
    merged = merge_segments(second, right, "Speaker_01")
    assert (merged["start"], merged["end"], merged["text"]) == (1.25, 6, "ответ Второй")
    changed = change_speaker(last, "Speaker_01")
    assert changed["words"][0] == {**last["words"][0], "speaker_id": "Speaker_01"}
    assert "words" not in change_text(changed, "Исправлено")
    assert last["words"][0]["speaker_id"] == "Speaker_00"


def test_revision_validation_and_competing_writers(job):
    opened = load_transcript(job)
    for bounds in ({"end": -1}, {"start": 5}, {"end": float("inf")}, {"start": None}):
        with pytest.raises(ValueError):
            save_revision(job, opened, [{**opened.segments[0], **bounds}])
    assert not (job / MANUAL).exists()
    save_revision(job, opened, [change_text(opened.segments[0], "Сохранено")])
    with pytest.raises(RuntimeError, match="изменилась"):
        save_revision(job, opened, opened.segments)
    assert load_transcript(job).segments[0]["text"] == "Сохранено"


@pytest.mark.parametrize("keep_manual", [False, True])
def test_merge_rerun_conflict_and_explicit_resolution(job, keep_manual):
    opened = load_transcript(job)
    manual = [change_text(opened.segments[0], "Отредактировано")]
    save_revision(job, opened, manual)
    export(job)
    # Metadata and whitespace can change even for equivalent runs.
    write_json_atomic(job / SOURCE, {"metadata": {"generation": 2}, "segments": opened.source})
    assert load_transcript(job).segments == manual
    write_json_atomic(job / "asr/transcript.json", {"segments": [{"text": "Новая запись", "start": 0, "end": 9}]})
    with pytest.raises(ValueError, match="Выберите"):
        run_pipeline(job_dir=job, out_dir=job.parent, stages=[MergeStage(), ExportStage()], force=True)
    with pytest.raises(ValueError, match="Выберите"):
        read_transcript(job)
    conflict = load_transcript(job)
    assert conflict.conflict and conflict.manual["segments"] == manual
    resolve_conflict(job, conflict, keep_manual=keep_manual)
    with pytest.raises(ValueError, match="changed"):
        read_transcript(job)
    export(job)
    expected = manual if keep_manual else conflict.source
    assert load_transcript(job).segments == expected
    assert expected[0]["text"] in read_transcript(job)
    if not keep_manual:
        archive, = (job / "edits").glob("transcript-*.json")
        assert read_json(archive)["segments"] == manual


def test_gui_edit_play_split_reassign_merge_save_and_reopen(qapp, job, monkeypatch):
    from PySide6.QtWidgets import QInputDialog

    from diarrhizer.adapters import llm
    from diarrhizer.summary.providers import KIND_CODEX_CLI, Provider
    from diarrhizer.summary.service import summarize_job
    from diarrhizer.summary.templates import PromptTemplate
    from diarrhizer_gui.screens.result_screen import ResultScreen

    original = (job / SOURCE).read_bytes()
    previous_summary = job / "summary/previous.md"
    previous_summary.parent.mkdir()
    previous_summary.write_text("Старое саммари", encoding="utf-8")
    screen = ResultScreen()
    assert screen.load(job)
    # Use the player's public timing API through the real result-screen signal connections.
    played = []
    screen._player = SimpleNamespace(position=lambda: 2125, setPosition=played.append,
                                     play=lambda: played.append("play"))
    screen._player_bar.show()
    screen._edit_button.click()
    editor = screen._editor
    editor.text_field.setPlainText("Да🙂 Теперь")
    cursor = editor.text_field.textCursor()
    cursor.setPosition(4)  # after the emoji in UTF-16
    editor.text_field.setTextCursor(cursor)
    def choose_cut(_parent, _title, _label, initial, start, end, _decimals):
        assert (initial, start, end) == (2.125, 0, 4)
        return 2.25, True
    monkeypatch.setattr(QInputDialog, "getDouble", choose_cut)
    editor.split.click()
    assert editor.table.currentRow() == 1
    assert [s["text"] for s in editor.segments] == ["Да🙂", "Теперь", "Второй", "Конец"]
    editor.play.click()
    assert played == [2250, "play"] and screen._stop_at_ms == 4000
    screen._player = None
    editor.speaker_field.setCurrentIndex(editor.speaker_field.findData("Speaker_01"))
    assert [s["speaker_id"] for s in editor.segments][:2] == ["Speaker_00", "Speaker_01"]
    editor.speaker_field.setCurrentIndex(editor.speaker_field.findData("Speaker_00"))
    def choose_speaker(_parent, _title, _label, labels, _current, _editable):
        return next(label for label in labels if "Анна" in label), True
    monkeypatch.setattr(QInputDialog, "getItem", choose_speaker)
    editor.merge.click()
    assert editor.segments[1]["speaker_id"] == "Speaker_01"
    assert screen._save_edits()
    assert (job / SOURCE).read_bytes() == original
    assert read_json(job / MANUAL)["segments"][-1] == json.loads(original)["segments"][-1]
    reopened = ResultScreen()
    assert reopened.load(job)
    assert "Теперь Второй" in reopened._transcript.toPlainText()
    reopened._edit_button.click()
    reopened._editor.text_field.setPlainText("Не сохранять")
    reopened._editor.cancel_requested.emit()
    assert reopened._segments[0]["text"] == "Да🙂"
    monkeypatch.setattr(sys, "argv", ["diarrhizer", "run", "--job-dir", str(job),
                                      "--from-stage", "export", "--force-stage", "export"])
    assert cli.main() == 0
    for ext in ("md", "txt", "srt"):
        rendered = (job / f"export/result.{ext}").read_text(encoding="utf-8")
        assert all(text in rendered for text in ("Да🙂", "Анна", "Теперь Второй", "Конец"))
        assert "Первый ответ" not in rendered
    rows = read_json(job / "export/result.json")["segments"]
    assert [(s["start"], s["end"], s["speaker_name"]) for s in rows] == [
        (0, 2.25, "Speaker_00"), (2.25, 6, "Анна"), (6, 7, "Speaker_00")]
    assert [s["text"] for s in rows] == ["Да🙂", "Теперь Второй", "Конец"]
    assert PromptContext("", job_dir=job).segments == reopened._segments
    prompts = []
    def complete(_provider, prompt):
        prompts.append(prompt)
        return llm.LLMReply("Саммари исправленной записи")
    monkeypatch.setattr(llm, "complete", complete)
    summarize_job(job, PromptTemplate("test", "{{transcript}}"), Provider(name="test", kind=KIND_CODEX_CLI))
    assert len(prompts) == 1 and "Теперь Второй" in prompts[0] and "Первый ответ" not in prompts[0]
    assert previous_summary.read_text(encoding="utf-8") == "Старое саммари"


def test_failed_export_keeps_editor_and_retry_publishes_all_files(qapp, job, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from diarrhizer import artifacts
    from diarrhizer_gui.screens.result_screen import ResultScreen

    export(job)
    screen = ResultScreen()
    screen.load(job)
    screen._edit_button.click()
    screen._editor.text_field.setPlainText("Повторная попытка")
    errors = []
    monkeypatch.setattr(QMessageBox, "warning", lambda _, title, message: errors.append((title, message)))
    write = artifacts.write_text_atomic
    def fail_one(path, content):
        if path.suffix == ".txt":
            raise PermissionError("TXT locked")
        write(path, content)
    with monkeypatch.context() as failure:
        failure.setattr(artifacts, "write_text_atomic", fail_one)
        assert not screen._save_edits()
        assert not screen._editor.isHidden() and screen._editor.segments[0]["text"] == "Повторная попытка"
        assert errors == [("Обновление экспортов", "TXT locked")]
        assert not bundle_valid(job / "export", "manifest.json")
        with pytest.raises(ValueError, match="incomplete"):
            read_transcript(job)
        reopened = ResultScreen()
        reopened.load(job)
        assert not reopened._retry_button.isHidden()
        assert reopened._segments[0]["text"] == "Повторная попытка"
    assert screen._save_edits()
    assert bundle_valid(job / "export", "manifest.json")
    assert "Повторная попытка" in read_transcript(job)


@pytest.mark.parametrize("use_new", [False, True])
def test_source_changes_while_editing_and_gui_resolves_conflict(qapp, job, monkeypatch, use_new):
    from PySide6.QtWidgets import QMessageBox

    from diarrhizer_gui.screens.result_screen import ResultScreen

    screen = ResultScreen()
    screen.load(job)
    screen._edit_button.click()
    screen._editor.text_field.setPlainText("Правка открытой реплики")
    write_json_atomic(job / SOURCE, {"segments": [{"start": 5, "end": 8, "text": "Новая разметка"}]})
    assert not screen._save_edits()
    assert "Сохранены обе версии" in screen._transcript.toPlainText()
    assert not screen._summary_button.isEnabled()
    label = "Использовать новую исходную" if use_new else "Использовать ручную"
    monkeypatch.setattr(QMessageBox, "exec", lambda box: next(b for b in box.buttons() if b.text() == label).click())
    screen._edit_button.click()
    expected = "Новая разметка" if use_new else "Правка открытой реплики"
    assert expected in screen._transcript.toPlainText()
    assert expected in read_transcript(job)
    assert screen._summary_button.isEnabled()


def test_revision_write_and_render_errors_remain_retryable(qapp, job, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from diarrhizer import transcript
    from diarrhizer_gui.screens.result_screen import ResultScreen

    export(job)
    screen = ResultScreen()
    screen.load(job)
    screen._edit_button.click()
    screen._editor.text_field.setPlainText("Новый текст")
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: None)
    def fail(*args):
        raise OSError("synthetic failure")
    with monkeypatch.context() as failure:
        failure.setattr(transcript, "write_json_atomic", fail)
        assert not screen._save_edits()
        assert "Сохранение ручной версии" in screen._status_label.text()
        assert not (job / MANUAL).exists()
        assert screen._editor.text_field.toPlainText() == "Новый текст"
    with monkeypatch.context() as failure:
        failure.setattr("diarrhizer.pipeline.stages.export.write_bundle", fail)
        assert not screen._save_edits()
        assert bundle_valid(job / "export", "manifest.json")  # Previous complete publication.
        with pytest.raises(ValueError, match="changed"):
            read_transcript(job)
    assert screen._save_edits() and "Новый текст" in read_transcript(job)


@pytest.mark.parametrize("choice", ["Остаться в редакторе", "Отбросить", "Сохранить"])
def test_leaving_dirty_editor(qapp, job, monkeypatch, choice):
    from PySide6.QtGui import QCloseEvent
    from PySide6.QtWidgets import QMessageBox

    from diarrhizer_gui.main_window import MainWindow
    from diarrhizer_gui.screens.result_screen import ResultScreen
    screen = ResultScreen()
    screen.load(job)
    screen._edit_button.click()
    screen._editor.text_field.setPlainText("Черновик")
    monkeypatch.setattr(QMessageBox, "exec", lambda box: next(b for b in box.buttons() if b.text() == choice).click())
    event = QCloseEvent()
    MainWindow.closeEvent(SimpleNamespace(_result=screen, _worker=None), event)
    assert event.isAccepted() == (choice != "Остаться в редакторе")
    assert (job / MANUAL).exists() == (choice == "Сохранить")


def test_sidebar_and_back_respect_unsaved_edits(qapp, job, monkeypatch, ini_settings, summary_stores):
    from PySide6.QtWidgets import QMessageBox

    from diarrhizer_gui.environment import EnvironmentProbe
    from diarrhizer_gui.main_window import ROWS, MainWindow

    for module in ("history_screen", "models_screen", "new_job_screen", "settings_screen", "summary_settings_screen"):
        monkeypatch.setattr(f"diarrhizer_gui.screens.{module}.QSettings", ini_settings[1])
    monkeypatch.setattr(EnvironmentProbe, "run", lambda *_: None)
    window = MainWindow()
    window._show_result(job)
    window._result._edit_button.click()
    window._result._editor.text_field.setPlainText("Черновик")
    choice = "Остаться в редакторе"
    monkeypatch.setattr(QMessageBox, "exec", lambda box: next(b for b in box.buttons() if b.text() == choice).click())
    window._nav.setCurrentRow(ROWS["settings"])
    assert window._stack.currentWidget() is window._result
    assert window._result._editor.text_field.toPlainText() == "Черновик"
    choice = "Отбросить"
    window._result.back_requested.emit()
    assert window._stack.currentWidget() is window._history
    assert not (job / MANUAL).exists()
    window.deleteLater()
