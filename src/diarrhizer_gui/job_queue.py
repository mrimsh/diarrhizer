"""The GUI's job queue: which runs are waiting, which one runs, how each ended.

Pure Python (no Qt) so the bookkeeping is testable on its own; MainWindow
owns one JobQueue and starts a PipelineWorker for each item in turn.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

PENDING = "pending"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
CANCELLED = "cancelled"

STATUS_LABELS = {
    PENDING: "в очереди",
    RUNNING: "выполняется",
    DONE: "готово",
    FAILED: "ошибка",
    CANCELLED: "отменено",
}
STATUS_ICONS = {PENDING: "…", RUNNING: "▶", DONE: "✓", FAILED: "✗", CANCELLED: "■"}


@dataclass
class QueuedRun:
    """One pipeline run: a new file, a resumed job or a partial re-run."""

    mode: str
    kwargs: dict
    label: str
    status: str = PENDING
    detail: str = ""
    job_dir: Optional[Path] = field(default=None)

    def __post_init__(self) -> None:
        if self.job_dir is None and self.kwargs.get("job_dir"):
            self.job_dir = Path(self.kwargs["job_dir"])


# [SEMANTIC-BEGIN] GUI:JOB_QUEUE
# @purpose: Track a batch of pipeline runs executed one after another
# @description: add() appends runs; when the queue was idle it first drops the previous batch, so `items`
#   is always "the batch the Monitor screen shows". Runs execute strictly one at a time (one GPU, and the
#   models are loaded per stage anyway). A failed or cancelled run does not stop the batch;
#   cancel_pending() abandons whatever has not started yet.
# @inputs: QueuedRun items
# @outputs: queue state for MainWindow / MonitorScreen
# @sideEffects: None
# @errors: None
# @see: GUI:PIPELINE_WORKER, GUI:JOB_PLANS
class JobQueue:
    def __init__(self) -> None:
        self.items: list[QueuedRun] = []

    def add(self, runs: Iterable[QueuedRun]) -> None:
        if not self.active():
            self.items = []
        self.items.extend(runs)

    def active(self) -> bool:
        return any(item.status in (PENDING, RUNNING) for item in self.items)

    def current(self) -> Optional[QueuedRun]:
        return next((item for item in self.items if item.status == RUNNING), None)

    def next_pending(self) -> Optional[QueuedRun]:
        return next((item for item in self.items if item.status == PENDING), None)

    def pending_count(self) -> int:
        return sum(1 for item in self.items if item.status == PENDING)

    def cancel_pending(self) -> int:
        count = 0
        for item in self.items:
            if item.status == PENDING:
                item.status = CANCELLED
                count += 1
        return count

    def position(self, item: QueuedRun) -> tuple[int, int]:
        """1-based position of item in the batch, and the batch size."""
        return self.items.index(item) + 1, len(self.items)

    def counts(self) -> dict:
        result = {status: 0 for status in STATUS_LABELS}
        for item in self.items:
            result[item.status] += 1
        return result

    def busy_job_dirs(self) -> set:
        """Job folders a pending or running item will write to."""
        return {
            item.job_dir.resolve()
            for item in self.items
            if item.job_dir is not None and item.status in (PENDING, RUNNING)
        }
# [SEMANTIC-END] GUI:JOB_QUEUE
