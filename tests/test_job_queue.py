"""Unit tests for the GUI job queue (pure Python, no Qt)."""

from diarrhizer_gui.job_queue import CANCELLED, DONE, PENDING, RUNNING, JobQueue, QueuedRun


def _run(label, **kwargs):
    return QueuedRun(mode="full", kwargs=kwargs, label=label)


def _labels(queue):
    return [item.label for item in queue.items]


def _batch(*labels):
    """done d1, running r, then the waiting ones."""
    queue = JobQueue()
    runs = [_run("d1"), _run("r")] + [_run(label) for label in labels]
    queue.add(runs)
    runs[0].status = DONE
    runs[1].status = RUNNING
    return queue, {run.label: run for run in runs}


def test_move_pending_reorders_only_the_waiting_items():
    queue, r = _batch("a", "b", "c")
    assert queue.move_pending(r["c"], 0)
    assert _labels(queue) == ["d1", "r", "c", "a", "b"]
    assert queue.move_pending(r["c"], 2)
    assert _labels(queue) == ["d1", "r", "a", "b", "c"]
    assert queue.move_pending(r["b"], 0)
    assert _labels(queue) == ["d1", "r", "b", "a", "c"]


def test_waiting_items_never_go_ahead_of_the_running_or_finished_ones():
    queue, r = _batch("a", "b")
    queue.move_pending(r["b"], -5)
    assert _labels(queue) == ["d1", "r", "b", "a"]
    assert queue.items[0].status == DONE and queue.items[1].status == RUNNING


def test_move_pending_clamps_the_index():
    queue, r = _batch("a", "b", "c")
    assert queue.move_pending(r["a"], 99)
    assert _labels(queue) == ["d1", "r", "b", "c", "a"]
    assert queue.move_pending(r["a"], -1)
    assert _labels(queue) == ["d1", "r", "a", "b", "c"]


def test_move_pending_refuses_items_that_are_not_waiting_or_unknown():
    queue, r = _batch("a", "b")
    before = _labels(queue)
    assert not queue.move_pending(r["r"], 0)
    assert not queue.move_pending(r["d1"], 3)
    assert not queue.move_pending(_run("stranger"), 0)
    r["a"].status = RUNNING  # started meanwhile
    assert not queue.move_pending(r["a"], 5)
    assert _labels(queue) == before


def test_move_pending_keeps_non_waiting_slots_even_when_they_are_interleaved():
    queue = JobQueue()
    a, x, b, c = _run("a"), _run("x"), _run("b"), _run("c")
    queue.add([a, x, b, c])
    x.status = CANCELLED
    queue.move_pending(c, 0)
    assert _labels(queue) == ["c", "x", "a", "b"]


def test_next_pending_follows_the_new_order():
    queue, r = _batch("a", "b")
    r["r"].status = DONE
    assert queue.next_pending() is r["a"]
    queue.move_pending(r["b"], 0)
    assert queue.next_pending() is r["b"]


def test_remove_pending_deletes_the_item_and_keeps_the_numbers_consistent(tmp_path):
    queue = JobQueue()
    runs = [_run("d"), _run("r", job_dir=tmp_path / "r"), _run("a", job_dir=tmp_path / "a"),
            _run("b", job_dir=tmp_path / "b")]
    queue.add(runs)
    runs[0].status, runs[1].status = DONE, RUNNING
    assert queue.remove_pending(runs[2])
    assert _labels(queue) == ["d", "r", "b"]
    assert queue.pending_count() == 1
    assert queue.counts()[PENDING] == 1 and queue.counts()[CANCELLED] == 0
    assert queue.position(runs[3]) == (3, 3)
    assert queue.busy_job_dirs() == {(tmp_path / "r").resolve(), (tmp_path / "b").resolve()}


def test_remove_pending_refuses_running_finished_and_unknown_items():
    queue, r = _batch("a")
    assert not queue.remove_pending(r["r"])
    assert not queue.remove_pending(r["d1"])
    assert not queue.remove_pending(_run("stranger"))
    assert _labels(queue) == ["d1", "r", "a"]
    r["a"].status = RUNNING
    assert not queue.remove_pending(r["a"])
    assert _labels(queue) == ["d1", "r", "a"]


def test_equal_looking_runs_are_told_apart_by_identity():
    queue = JobQueue()
    first, second = _run("same"), _run("same")
    queue.add([first, second])
    assert queue.remove_pending(second)
    assert queue.items == [first] and queue.items[0] is first


def test_add_starts_a_new_batch_after_the_waiting_items_were_removed():
    queue, r = _batch("a")
    r["r"].status = DONE
    assert queue.remove_pending(r["a"])
    assert not queue.active()
    queue.add([_run("new")])
    assert _labels(queue) == ["new"]
