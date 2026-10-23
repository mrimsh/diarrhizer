"""Offscreen tests of the GUI screens and their pure helpers.

Every screen's QSettings is redirected to a temp INI file (ini_settings),
so nothing here touches the real settings store.
"""

import gc
import json
from pathlib import Path

import pytest

pytest.importorskip("PySide6", reason="GUI extra not installed")

from PySide6.QtCore import QCoreApplication, QEvent, QMimeData, QModelIndex, QPointF, Qt, QUrl  # noqa: E402
from PySide6.QtGui import QDragEnterEvent, QDropEvent  # noqa: E402
from PySide6.QtWidgets import QDialog, QMessageBox, QPushButton  # noqa: E402

from diarrhizer.audio_formats import ENV_AUDIO_FORMATS_FILE  # noqa: E402
from diarrhizer_gui import settings_keys  # noqa: E402
from diarrhizer_gui.job_queue import (  # noqa: E402
    CANCELLED,
    DONE,
    PENDING,
    RUNNING,
    JobQueue,
    QueuedRun,
)
from diarrhizer_gui.screens import history_screen, new_job_screen, result_screen  # noqa: E402


@pytest.fixture
def isolated(ini_settings, tmp_path, monkeypatch):
    _path, factory = ini_settings
    monkeypatch.setattr(new_job_screen, "QSettings", factory)
    monkeypatch.setattr(history_screen, "QSettings", factory)
    monkeypatch.setenv(ENV_AUDIO_FORMATS_FILE, str(tmp_path / "audio_formats.json"))
    return factory


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


# --- job queue ---------------------------------------------------------------


def _run(label, **kwargs):
    return QueuedRun(mode="full", kwargs=kwargs, label=label)


def test_queue_runs_in_order_and_starts_a_new_batch_when_idle():
    queue = JobQueue()
    queue.add([_run("a"), _run("b")])
    first = queue.next_pending()
    first.status = RUNNING
    assert queue.current() is first
    assert queue.position(first) == (1, 2)
    first.status = DONE
    queue.add([_run("c")])  # still active (b pending): appended to the batch
    assert [i.label for i in queue.items] == ["a", "b", "c"]
    for item in queue.items:
        item.status = DONE
    queue.add([_run("d")])  # idle: a fresh batch
    assert [i.label for i in queue.items] == ["d"]


def test_cancel_pending_leaves_the_running_item_alone():
    queue = JobQueue()
    queue.add([_run("a"), _run("b"), _run("c")])
    queue.items[0].status = RUNNING
    assert queue.cancel_pending() == 2
    assert [i.status for i in queue.items] == [RUNNING, CANCELLED, CANCELLED]
    assert queue.counts()[CANCELLED] == 2


def test_busy_job_dirs_cover_pending_and_running_jobs(tmp_path):
    queue = JobQueue()
    queue.add([_run("a", job_dir=tmp_path / "a"), _run("b", job_dir=tmp_path / "b"), _run("c")])
    queue.items[0].status = DONE
    assert queue.busy_job_dirs() == {(tmp_path / "b").resolve()}
    assert queue.items[2].status == PENDING


# --- New Job ---------------------------------------------------------------------


def test_new_job_makes_one_run_per_file_and_remembers_the_parameters(qapp, isolated, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    folder = tmp_path / "calls"
    _touch(folder / "a.m4a")
    _touch(folder / "b.amr")
    _touch(folder / "notes.txt")

    screen = new_job_screen.NewJobScreen()
    assert screen.add_paths([folder]) == 2
    assert screen.add_paths([folder]) == 0  # no duplicates
    screen._language_combo.setCurrentText("ru")
    screen._min_speakers.setValue(2)
    screen._max_speakers.setValue(2)
    screen._prompt_edit.setPlainText("ООО Ромашка")
    received = []
    screen.runs_requested.connect(received.append)

    assert screen._run_button.isEnabled()
    assert screen._run_button.text() == "Запустить (2 файла)"
    screen._run_button.click()

    (runs,) = received
    assert [Path(r.kwargs["input_path"]).name for r in runs] == ["a.m4a", "b.amr"]
    for run in runs:
        assert run.mode == "full"
        assert run.kwargs["language"] == "ru"
        assert (run.kwargs["min_speakers"], run.kwargs["max_speakers"]) == (2, 2)
        assert run.kwargs["asr_initial_prompt"] == "ООО Ромашка"
        assert "job_dir" not in run.kwargs  # MainWindow assigns it when the run starts
    assert screen._files.count() == 0

    again = new_job_screen.NewJobScreen()
    assert again._language_combo.currentText() == "ru"
    assert (again._min_speakers.value(), again._max_speakers.value()) == (2, 2)
    assert again._prompt_edit.toPlainText() == "ООО Ромашка"


def test_full_mode_needs_an_hf_token(qapp, isolated, tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    screen = new_job_screen.NewJobScreen()
    screen.add_paths([_touch(tmp_path / "call.wav")])
    screen._mode_combo.setCurrentIndex(screen._mode_combo.findData("full"))
    assert not screen._run_button.isEnabled()
    assert "HF-токен" in screen._warning_label.text()

    screen._mode_combo.setCurrentIndex(screen._mode_combo.findData("asr_only"))
    assert screen._run_button.isEnabled()


def test_busy_screen_queues_instead_of_starting(qapp, isolated, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    screen = new_job_screen.NewJobScreen()
    screen.add_paths([_touch(tmp_path / "call.wav")])
    screen.set_busy(True)
    assert screen._run_button.text() == "Добавить в очередь"


def _file_screen(tmp_path, names):
    screen = new_job_screen.NewJobScreen()
    for name in names:
        screen.add_paths([_touch(tmp_path / name)])
    return screen


def _names(screen):
    return [Path(p).name for p in screen._selected_files()]


def test_new_job_file_order_follows_the_list_and_reaches_the_runs(qapp, isolated, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    screen = _file_screen(tmp_path, ["a.wav", "b.wav", "c.wav"])
    screen._files.model().moveRow(QModelIndex(), 2, QModelIndex(), 0)  # what a drag-and-drop move does
    assert _names(screen) == ["c.wav", "a.wav", "b.wav"]
    # the path travels with the row, and the visible text still matches it
    for i in range(screen._files.count()):
        entry = screen._files.item(i)
        assert entry.data(Qt.ItemDataRole.UserRole) == str(tmp_path / entry.text().split()[0])
    assert screen._run_button.isEnabled()
    received = []
    screen.runs_requested.connect(received.append)
    screen._run_button.click()
    (runs,) = received
    assert [r.label for r in runs] == ["c.wav", "a.wav", "b.wav"]


def test_new_job_up_down_buttons_move_the_selection(qapp, isolated, tmp_path):
    screen = _file_screen(tmp_path, ["a.wav", "b.wav", "c.wav", "d.wav"])
    files = screen._files
    files.item(2).setSelected(True)
    files.item(3).setSelected(True)
    screen._move_selected(-1)
    assert _names(screen) == ["a.wav", "c.wav", "d.wav", "b.wav"]
    assert [e.text().split()[0] for e in files.selectedItems()] == ["c.wav", "d.wav"]
    screen._move_selected(-1)
    screen._move_selected(-1)  # already at the top: stays
    assert _names(screen) == ["c.wav", "d.wav", "a.wav", "b.wav"]
    screen._move_selected(1)
    assert _names(screen) == ["a.wav", "c.wav", "d.wav", "b.wav"]
    screen._move_selected(1)
    screen._move_selected(1)  # at the bottom: stays
    assert _names(screen) == ["a.wav", "b.wav", "c.wav", "d.wav"]
    assert screen._files.count() == 4


def test_new_job_sort_by_name_keeps_paths_and_selection(qapp, isolated, tmp_path):
    screen = _file_screen(tmp_path, ["b.wav", "C.wav", "a.wav"])
    screen._files.item(0).setSelected(True)  # b.wav
    screen._sort_by_name()
    assert _names(screen) == ["a.wav", "b.wav", "C.wav"]
    assert [Path(e.data(Qt.ItemDataRole.UserRole)).name for e in screen._files.selectedItems()] == ["b.wav"]
    assert screen._files.count() == 3


def test_new_job_list_still_takes_files_dropped_from_outside(qapp, isolated, tmp_path):
    screen = _file_screen(tmp_path, ["a.wav"])
    outside = _touch(tmp_path / "other" / "z.wav")
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(outside))])
    enter = QDragEnterEvent(QPointF(5, 5).toPoint(), Qt.DropAction.CopyAction, mime,
                            Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    screen._files.dragEnterEvent(enter)
    assert enter.isAccepted()
    drop = QDropEvent(QPointF(5, 5), Qt.DropAction.CopyAction, mime,
                      Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    screen._files.dropEvent(drop)
    assert _names(screen) == ["a.wav", "z.wav"]


def test_plural():
    assert [new_job_screen.plural(n, "файл", "файла", "файлов") for n in (1, 2, 5, 11, 21, 22, 25)] == [
        "файл", "файла", "файлов", "файлов", "файл", "файла", "файлов"
    ]


# --- History ---------------------------------------------------------------------------


def _unfinished_job(out: Path) -> Path:
    job_dir = out / "call_20260102_030405"
    _write(job_dir / "meta" / "run.json", {
        "input_path": "D:/calls/call.m4a",
        "pipeline_config": {"language": "ru", "min_speakers": 2, "max_speakers": 2},
    })
    _touch(job_dir / "audio" / "normalized.wav")
    _write(job_dir / "asr" / "transcript.json", {"segments": []})
    return job_dir


def test_history_resume_emits_a_run_from_the_first_missing_stage(qapp, isolated, tmp_path):
    out = tmp_path / "out"
    job_dir = _unfinished_job(out)
    isolated().setValue(settings_keys.OUT_DIR, str(out))

    screen = history_screen.HistoryScreen()
    (job,) = screen._jobs
    assert job.resumable
    received = []
    screen.runs_requested.connect(received.append)
    screen._resume(job)

    (runs,) = received
    assert runs[0].kwargs["from_stage"] == "diarize"
    assert runs[0].job_dir == job_dir
    assert "диаризация" in runs[0].label


def test_history_disables_delete_and_resume_for_a_running_job(qapp, isolated, tmp_path):
    out = tmp_path / "out"
    job_dir = _unfinished_job(out)
    isolated().setValue(settings_keys.OUT_DIR, str(out))

    screen = history_screen.HistoryScreen(is_busy=lambda d: d == job_dir)
    actions = screen._table.cellWidget(0, 6)
    buttons = {b.text(): b for b in actions.findChildren(QPushButton)}
    assert not buttons["Удалить"].isEnabled()
    assert not buttons["Продолжить"].isEnabled()


def _complete_job(out: Path) -> Path:
    job_dir = _unfinished_job(out)
    _touch(job_dir / "diar" / "diarization.json")
    _touch(job_dir / "merged" / "segments.json")
    _touch(job_dir / "export" / "result.md")
    return job_dir


def _record_source(job_dir: Path, source: Path) -> None:
    """Make the job's recorded source file `source` (the helpers above name a path that does not exist)."""
    meta = json.loads((job_dir / "meta" / "run.json").read_text(encoding="utf-8"))
    meta["input_path"] = str(source)
    _write(job_dir / "meta" / "run.json", meta)


def _restart_button(screen) -> QPushButton:
    actions = screen._table.cellWidget(0, 6)
    return next(b for b in actions.findChildren(QPushButton) if b.text() == "Перезапустить этап")


def _answer(monkeypatch, answer=QDialog.DialogCode.Accepted, edit=None):
    """Stand in for the restart dialog's event loop: `edit(dialog)` plays the
    user changing settings, `answer` is how it is closed. The dialogs it was
    asked to show are returned."""
    shown = []

    def exec_(dialog):
        shown.append(dialog)
        if edit is not None:
            edit(dialog)
        return answer

    monkeypatch.setattr(history_screen.RestartDialog, "exec", exec_)
    return shown


def _warnings(monkeypatch):
    shown = []
    monkeypatch.setattr(QMessageBox, "warning", lambda _parent, _title, text, *_args: shown.append(text))
    return shown


def test_history_restart_menu_greys_out_stages_without_inputs(qapp, isolated, tmp_path):
    out = tmp_path / "out"
    _unfinished_job(out)  # converted and transcribed, nothing after that
    isolated().setValue(settings_keys.OUT_DIR, str(out))

    screen = history_screen.HistoryScreen()
    button = _restart_button(screen)

    enabled = {action.text(): action.isEnabled() for action in button.menu().actions()}
    assert enabled == {
        "Конвертация": True,
        "Распознавание": True,
        "Диаризация": True,
        "Объединение": False,
        "Экспорт": False,
    }
    assert button.isEnabled()


def test_history_disables_restart_for_a_running_job(qapp, isolated, tmp_path):
    out = tmp_path / "out"
    job_dir = _complete_job(out)
    isolated().setValue(settings_keys.OUT_DIR, str(out))

    assert not _restart_button(history_screen.HistoryScreen(is_busy=lambda d: d == job_dir)).isEnabled()
    assert _restart_button(history_screen.HistoryScreen()).isEnabled()


def test_history_restart_emits_a_forced_run_of_the_stage(qapp, isolated, tmp_path, monkeypatch):
    out = tmp_path / "out"
    job_dir = _complete_job(out)
    isolated().setValue(settings_keys.OUT_DIR, str(out))
    shown_dialogs = _answer(monkeypatch)

    screen = history_screen.HistoryScreen()
    received = []
    screen.runs_requested.connect(received.append)
    screen._restart(screen._jobs[0], "merge")

    (runs,) = received
    (run,) = runs
    assert (run.kwargs["from_stage"], run.kwargs["force_stage"]) == ("merge", "merge")
    assert run.job_dir == job_dir
    assert "объединение" in run.label
    assert "объединение, экспорт" in shown_dialogs[0].intro_text


def test_history_restart_runs_with_the_settings_changed_in_the_dialog(qapp, isolated, tmp_path, monkeypatch):
    out = tmp_path / "out"
    _complete_job(out)
    isolated().setValue(settings_keys.OUT_DIR, str(out))

    def edit(dialog):
        dialog.field("asr_model").setCurrentText("tiny")
        dialog.field("language").setCurrentText("en")

    shown_dialogs = _answer(monkeypatch, edit=edit)

    screen = history_screen.HistoryScreen()
    received = []
    screen.runs_requested.connect(received.append)
    screen._restart(screen._jobs[0], "transcribe")

    (run,) = received[0]
    assert (run.kwargs["asr_model"], run.kwargs["language"]) == ("tiny", "en")
    assert run.kwargs["force_stage"] == "transcribe"
    # What the job was recorded with is the starting point for everything else.
    assert (run.kwargs["min_speakers"], run.kwargs["max_speakers"]) == (2, 2)
    # Only the stages being recomputed are offered: no speaker range for transcription.
    assert shown_dialogs[0].field("min_speakers") is None
    assert shown_dialogs[0].field("asr_model") is not None


def test_history_restart_without_changes_keeps_the_recorded_settings(qapp, isolated, tmp_path, monkeypatch):
    out = tmp_path / "out"
    _complete_job(out)
    isolated().setValue(settings_keys.OUT_DIR, str(out))
    _answer(monkeypatch)

    screen = history_screen.HistoryScreen()
    received = []
    screen.runs_requested.connect(received.append)
    screen._restart(screen._jobs[0], "transcribe")

    (run,) = received[0]
    assert run.kwargs["language"] == "ru"  # the recorded value, as the job had it
    assert "asr_model" not in run.kwargs  # not recorded: run_pipeline's own default applies


def test_history_restart_of_export_offers_no_settings(qapp, isolated, tmp_path, monkeypatch):
    out = tmp_path / "out"
    _complete_job(out)
    isolated().setValue(settings_keys.OUT_DIR, str(out))
    shown_dialogs = _answer(monkeypatch)

    screen = history_screen.HistoryScreen()
    screen._restart(screen._jobs[0], "export")

    assert shown_dialogs[0].overrides() == {}
    assert all(shown_dialogs[0].field(key) is None for key in ("asr_model", "min_turn_duration", "device"))


def test_history_restart_of_diarization_offers_the_speaker_range_and_merge_setting(
    qapp, isolated, tmp_path, monkeypatch
):
    out = tmp_path / "out"
    _complete_job(out)
    isolated().setValue(settings_keys.OUT_DIR, str(out))
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    shown_dialogs = _answer(monkeypatch, edit=lambda dialog: dialog.field("max_speakers").setValue(4))

    screen = history_screen.HistoryScreen()
    received = []
    screen.runs_requested.connect(received.append)
    screen._restart(screen._jobs[0], "diarize")

    dialog = shown_dialogs[0]
    assert all(dialog.field(key) is not None for key in ("min_speakers", "device", "min_turn_duration"))
    assert dialog.field("asr_model") is None
    (run,) = received[0]
    assert (run.kwargs["min_speakers"], run.kwargs["max_speakers"]) == (2, 4)


def test_history_restart_does_nothing_when_declined(qapp, isolated, tmp_path, monkeypatch):
    out = tmp_path / "out"
    job_dir = _complete_job(out)
    _write(job_dir / "speakers.json", {"Speaker_00": "Иван"})
    isolated().setValue(settings_keys.OUT_DIR, str(out))
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    _answer(monkeypatch, QDialog.DialogCode.Rejected)

    screen = history_screen.HistoryScreen()
    received = []
    screen.runs_requested.connect(received.append)
    screen._restart(screen._jobs[0], "diarize")

    assert received == []
    assert (job_dir / "speakers.json").exists()


def test_history_restart_of_diarization_sets_the_old_names_aside(qapp, isolated, tmp_path, monkeypatch):
    out = tmp_path / "out"
    job_dir = _complete_job(out)
    _write(job_dir / "speakers.json", {"Speaker_00": "Иван"})
    isolated().setValue(settings_keys.OUT_DIR, str(out))
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    shown_dialogs = _answer(monkeypatch)

    screen = history_screen.HistoryScreen()
    received = []
    screen.runs_requested.connect(received.append)
    screen._restart(screen._jobs[0], "diarize")

    (runs,) = received
    assert runs[0].kwargs["speakers"] is None
    assert not (job_dir / "speakers.json").exists()
    assert (job_dir / "speakers.before-rediarize.json").exists()
    assert "Имена спикеров будут сброшены" in shown_dialogs[0].intro_text


def test_history_restart_that_redoes_diarization_needs_an_hf_token(qapp, isolated, tmp_path, monkeypatch):
    out = tmp_path / "out"
    job_dir = _complete_job(out)
    _record_source(job_dir, _touch(tmp_path / "call.m4a"))
    _write(job_dir / "speakers.json", {"Speaker_00": "Иван"})
    isolated().setValue(settings_keys.OUT_DIR, str(out))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    shown = _warnings(monkeypatch)
    _answer(monkeypatch)

    screen = history_screen.HistoryScreen()
    received = []
    screen.runs_requested.connect(received.append)
    screen._restart(screen._jobs[0], "diarize")
    screen._restart(screen._jobs[0], "convert")  # new audio makes diarization stale too

    assert received == []
    assert len(shown) == 2 and all("HF-токен" in text for text in shown)
    assert (job_dir / "speakers.json").exists()

    screen._restart(screen._jobs[0], "transcribe")  # diarization stays cached: no token needed
    assert len(received) == 1


def test_history_restart_of_convert_needs_the_source_file(qapp, isolated, tmp_path, monkeypatch):
    out = tmp_path / "out"
    job_dir = _complete_job(out)
    source = tmp_path / "gone.m4a"
    _record_source(job_dir, source)
    isolated().setValue(settings_keys.OUT_DIR, str(out))
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    shown = _warnings(monkeypatch)
    _answer(monkeypatch)

    screen = history_screen.HistoryScreen()
    received = []
    screen.runs_requested.connect(received.append)
    screen._restart(screen._jobs[0], "convert")
    assert received == [] and str(source) in shown[0]

    source.write_bytes(b"x")
    screen._restart(screen._jobs[0], "convert")
    assert len(received) == 1


def test_history_filter_hides_other_files(qapp, isolated, tmp_path):
    out = tmp_path / "out"
    _unfinished_job(out)
    _write(out / "meeting_20260103_000000" / "meta" / "run.json", {"input_path": "D:/m/meeting.mp4"})
    isolated().setValue(settings_keys.OUT_DIR, str(out))

    screen = history_screen.HistoryScreen()
    screen._filter_field.setText("meet")
    hidden = [screen._table.isRowHidden(row) for row in range(screen._table.rowCount())]
    names = [job.input_name for job in screen._jobs]
    assert dict(zip(names, hidden)) == {"meeting.mp4": False, "call.m4a": True}


def _history_job(out: Path, stamp: str, name: str, config=None, stages=0) -> Path:
    """A job folder `<name>_<stamp>` with the first `stages` stages' outputs (5 = finished)."""
    job_dir = out / f"{Path(name).stem}_{stamp}"
    meta = {"input_path": f"D:/calls/{name}"}
    if config is not None:
        meta["pipeline_config"] = config
    _write(job_dir / "meta" / "run.json", meta)
    for artifact in ["audio/normalized.wav", "asr/transcript.json", "diar/diarization.json",
                     "merged/segments.json", "export/result.md"][:stages]:
        _touch(job_dir / artifact)
    return job_dir


@pytest.fixture
def history_out(isolated, tmp_path):
    out = tmp_path / "out"
    cfg = {"language": "ru", "min_speakers": 2, "max_speakers": 2, "asr_model": "small"}
    _history_job(out, "20260105_100000", "alpha.wav", cfg, stages=5)
    _history_job(out, "20260103_100000", "Beta.wav",
                 {"language": "en", "min_speakers": 10, "max_speakers": 10, "asr_model": "large-v3"}, stages=2)
    _history_job(out, "20260104_100000", "gamma.wav", None, stages=0)  # no config: placeholders
    isolated().setValue(settings_keys.OUT_DIR, str(out))
    return out


def _order(screen):
    return [job.input_name for job in screen._jobs]


def _click(screen, column, times=1):
    for _ in range(times):
        screen._table.horizontalHeader().sectionClicked.emit(column)


def test_history_default_order_is_newest_first_with_the_indicator_on_date(qapp, history_out):
    screen = history_screen.HistoryScreen()
    assert _order(screen) == ["alpha.wav", "gamma.wav", "Beta.wav"]
    header = screen._table.horizontalHeader()
    assert (header.sortIndicatorSection(), header.sortIndicatorOrder()) == (0, Qt.SortOrder.DescendingOrder)


def test_history_sorts_by_each_column_with_its_own_type(qapp, history_out):
    screen = history_screen.HistoryScreen()
    _click(screen, 0)  # date: a real time, oldest first
    assert _order(screen) == ["Beta.wav", "gamma.wav", "alpha.wav"]
    _click(screen, 1)  # file, case-insensitive
    assert _order(screen) == ["alpha.wav", "Beta.wav", "gamma.wav"]
    _click(screen, 2)  # language: en, ru; the job without one is last
    assert _order(screen) == ["Beta.wav", "alpha.wav", "gamma.wav"]
    _click(screen, 3)  # speakers as numbers: 2 before 10, no value last
    assert _order(screen) == ["alpha.wav", "Beta.wav", "gamma.wav"]
    _click(screen, 4)  # model: large-v3 < small, no value last
    assert _order(screen) == ["Beta.wav", "alpha.wav", "gamma.wav"]
    _click(screen, 5)  # stages done: 0, 2, 5
    assert _order(screen) == ["gamma.wav", "Beta.wav", "alpha.wav"]


def test_history_second_click_reverses_but_empty_values_stay_last(qapp, history_out):
    screen = history_screen.HistoryScreen()
    _click(screen, 3)
    assert _order(screen) == ["alpha.wav", "Beta.wav", "gamma.wav"]
    _click(screen, 3)
    assert _order(screen) == ["Beta.wav", "alpha.wav", "gamma.wav"]  # 10 before 2, no value still last
    header = screen._table.horizontalHeader()
    assert (header.sortIndicatorSection(), header.sortIndicatorOrder()) == (3, Qt.SortOrder.DescendingOrder)


def test_history_actions_column_is_not_sortable(qapp, history_out):
    screen = history_screen.HistoryScreen()
    _click(screen, 3)
    _click(screen, 6)
    assert _order(screen) == ["alpha.wav", "Beta.wav", "gamma.wav"]
    assert screen._table.horizontalHeader().sortIndicatorSection() == 3


def test_history_sort_is_stable(qapp, isolated, tmp_path):
    out = tmp_path / "out"
    for stamp, name in [("20260101_000000", "a.wav"), ("20260102_000000", "b.wav"), ("20260103_000000", "c.wav")]:
        _history_job(out, stamp, name, {"language": "ru"})
    isolated().setValue(settings_keys.OUT_DIR, str(out))
    screen = history_screen.HistoryScreen()
    _click(screen, 2)  # all equal: the date order (newest first) is kept
    assert _order(screen) == ["c.wav", "b.wav", "a.wav"]
    _click(screen, 2)
    assert _order(screen) == ["c.wav", "b.wav", "a.wav"]


def test_history_rows_match_self_jobs_after_sorting(qapp, history_out):
    screen = history_screen.HistoryScreen()
    _click(screen, 1)
    for row, job in enumerate(screen._jobs):
        assert job.input_name in screen._table.item(row, 1).text()
        assert screen._table.item(row, 0).text() == job.date_label


def test_history_double_click_and_row_actions_use_the_sorted_row(qapp, history_out, monkeypatch):
    other = _history_job(history_out, "20260106_100000", "zeta.wav", {"language": "de"}, stages=5)
    screen = history_screen.HistoryScreen()
    _click(screen, 1)  # alpha, Beta, gamma, zeta
    opened = []
    screen.view_result_requested.connect(opened.append)
    row = _order(screen).index("zeta.wav")
    screen._on_double_click(row, 0)
    assert opened == [other]

    buttons = {b.text(): b for b in screen._table.cellWidget(row, 6).findChildren(QPushButton)}
    buttons["Результат"].click()
    assert opened == [other, other]

    monkeypatch.setattr(QMessageBox, "question", lambda *_a, **_k: QMessageBox.StandardButton.Yes)
    buttons["Удалить"].click()
    assert not other.exists()
    assert _order(screen) == ["alpha.wav", "Beta.wav", "gamma.wav"]  # the sort is kept after the refresh


def test_history_filter_works_with_the_sort(qapp, history_out):
    screen = history_screen.HistoryScreen()
    _click(screen, 1)
    screen._filter_field.setText("a.wav")  # alpha.wav, Beta.wav, gamma.wav -> only names with "a.wav"
    shown = [job.input_name for row, job in enumerate(screen._jobs) if not screen._table.isRowHidden(row)]
    assert shown == ["alpha.wav", "Beta.wav", "gamma.wav"]
    screen._filter_field.setText("alpha")
    shown = [job.input_name for row, job in enumerate(screen._jobs) if not screen._table.isRowHidden(row)]
    assert shown == ["alpha.wav"]
    _click(screen, 1)  # re-sorting keeps the filter applied
    shown = [job.input_name for row, job in enumerate(screen._jobs) if not screen._table.isRowHidden(row)]
    assert shown == ["alpha.wav"]


def test_history_selection_survives_refresh_and_resort(qapp, history_out):
    screen = history_screen.HistoryScreen()
    screen._table.selectRow(_order(screen).index("Beta.wav"))
    screen.refresh()
    _click(screen, 1)
    (selected,) = screen._table.selectionModel().selectedRows()
    assert screen._jobs[selected.row()].input_name == "Beta.wav"


def test_history_remembers_sort_and_widths(qapp, history_out):
    screen = history_screen.HistoryScreen()
    _click(screen, 3, times=2)
    screen._table.horizontalHeader().resizeSection(2, 123)
    screen._table.horizontalHeader().resizeSection(1, 321)
    screen._save_widths()

    again = history_screen.HistoryScreen()
    assert _order(again) == ["Beta.wav", "alpha.wav", "gamma.wav"]
    header = again._table.horizontalHeader()
    assert (header.sortIndicatorSection(), header.sortIndicatorOrder()) == (3, Qt.SortOrder.DescendingOrder)
    again.refresh()  # the widths the user set are not touched by a refresh
    assert (header.sectionSize(2), header.sectionSize(1)) == (123, 321)


def test_history_ignores_unusable_saved_sort_and_widths(qapp, history_out, isolated):
    store = isolated()
    for column, order, widths in [
        ("99", "asc", '{"columns": 7, "widths": {"2": "wide"}}'),   # unknown column; not a width
        ("6", "asc", '{"columns": 5, "widths": {"2": 100}}'),    # actions column; another column count
        ("x", "up", "not json"),
        ("1", "sideways", '{"columns": 7, "widths": {"9": 100}}'),
        ("-1", "asc", '{"columns": 7, "widths": {"2": -5}}'),
    ]:
        store.setValue(settings_keys.HISTORY_SORT_COLUMN, column)
        store.setValue(settings_keys.HISTORY_SORT_ORDER, order)
        store.setValue(settings_keys.HISTORY_COLUMN_WIDTHS, widths)
        store.sync()
        screen = history_screen.HistoryScreen()
        assert _order(screen) == ["alpha.wav", "gamma.wav", "Beta.wav"], (column, order)
        assert screen._user_widths == {}, widths


# --- Result ------------------------------------------------------------------------------

SEGMENTS = [
    {"start": 0.0, "end": 2.0, "speaker_id": "SPEAKER_00", "text": "Привет."},
    {"start": 2.2, "end": 9.0, "speaker_id": "SPEAKER_00", "text": "Как дела?"},
    {"start": 9.5, "end": 20.0, "speaker_id": "SPEAKER_01", "text": "Хорошо <спасибо>."},
]


def test_transcript_html_groups_turns_links_timestamps_and_escapes():
    page = result_screen.build_transcript_html(SEGMENTS, {"Speaker_00": "Иван"})
    assert page.count("<p ") == 2
    assert 'href="t:0.000"' in page and 'href="t:9.500"' in page
    assert "Иван:" in page
    assert "&lt;спасибо&gt;" in page


def test_transcript_plain_text_matches_result_txt_lines():
    text = result_screen.transcript_plain_text(SEGMENTS, {"Speaker_01": "Мария"})
    assert text.splitlines()[0] == "[00:00:00 → 00:00:09] Speaker_00: Привет. Как дела?"
    assert "Мария: Хорошо <спасибо>." in text


def test_sample_range_is_the_longest_segment_capped():
    assert result_screen.sample_range(SEGMENTS, "Speaker_01") == (9.5, 9.5 + result_screen.SAMPLE_SECONDS)
    assert result_screen.sample_range(SEGMENTS, "Speaker_00") == (2.2, 9.0)
    assert result_screen.sample_range(SEGMENTS, "Speaker_05") is None


def _finished_job(tmp_path: Path) -> Path:
    job_dir = tmp_path / "out" / "call_20260101_000000"
    _write(job_dir / "merged" / "segments.json", {"segments": SEGMENTS})
    _write(job_dir / "meta" / "run.json", {
        "input_path": str(tmp_path / "call.wav"),
        "pipeline_config": {"language": "ru", "device": "cpu", "min_speakers": 2, "max_speakers": 2},
    })
    return job_dir


def test_reexport_renders_names_with_the_recorded_settings(tmp_path):
    job_dir = _finished_job(tmp_path)
    result_screen.reexport(job_dir, {"Speaker_00": "Иван"})
    text = (job_dir / "export" / "result.txt").read_text(encoding="utf-8")
    assert "Language: ru" in text
    assert "Иван: Привет. Как дела?" in text
    assert (job_dir / "export" / "result.srt").exists()


def test_result_screen_saves_names_and_reexports(qapp, tmp_path):
    job_dir = _finished_job(tmp_path)
    screen = result_screen.ResultScreen()
    screen.load(job_dir)

    assert set(screen._speaker_fields) == {"Speaker_00", "Speaker_01"}
    assert not screen._save_button.isEnabled()
    screen._speaker_fields["Speaker_01"].setText("Мария")
    assert screen._save_button.isEnabled()
    screen._save_names()

    assert json.loads((job_dir / "speakers.json").read_text(encoding="utf-8")) == {"Speaker_01": "Мария"}
    assert "Мария: Хорошо" in (job_dir / "export" / "result.txt").read_text(encoding="utf-8")
    assert not screen._save_button.isEnabled()
    # No audio in this job: playback is unavailable, not broken.
    assert not screen._player_bar.isVisible()


def _pump(ms: int) -> None:
    from PySide6.QtCore import QEventLoop, QTimer

    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def test_leaving_the_result_screen_releases_the_audio_file(qapp, tmp_path):
    pytest.importorskip("PySide6.QtMultimedia")
    import wave

    job_dir = _finished_job(tmp_path)
    wav_path = job_dir / "audio" / "normalized.wav"
    wav_path.parent.mkdir(parents=True)
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 16000)

    screen = result_screen.ResultScreen()
    screen.load(job_dir)
    screen.show()
    _pump(800)
    assert not screen._player_bar.isHidden()

    screen.hide()
    _pump(300)
    wav_path.unlink()  # would raise PermissionError while the player held it
    assert not wav_path.exists()

    # Take the player down here, on this thread, now. Left to the garbage collector it was destroyed at
    # some arbitrary later moment - possibly inside another thread of another test - and QtMultimedia's
    # FFmpeg threads then aborted the whole test run ("Fatal Python error: Aborted", exit code 3).
    screen.deleteLater()
    del screen
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    gc.collect()
    _pump(300)


def test_result_screen_ignores_unchanged_ids_saved_by_older_versions(qapp, tmp_path):
    job_dir = _finished_job(tmp_path)
    _write(job_dir / "speakers.json", {"Speaker_00": "Speaker_00", "SPEAKER_01": "Мария"})
    screen = result_screen.ResultScreen()
    screen.load(job_dir)
    assert screen._speaker_fields["Speaker_00"].text() == ""
    assert screen._speaker_fields["Speaker_01"].text() == "Мария"
