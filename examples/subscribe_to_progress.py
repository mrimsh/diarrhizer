"""Manual example: subscribe to the "diarrhizer" logger to observe pipeline
progress the way a GUI would - without parsing any printed text.

Every stage module and pipeline/runner.py report progress via
`logging.getLogger(__name__).info(...)`, passing the stage name through
`extra={"stage": ...}` on a per-stage line (see PIPELINE:RUNNER /
STAGE:CONVERT / etc. in SEMANTIC_TAGS.md-tagged code). Any logging.Handler
attached to the "diarrhizer" logger receives one LogRecord per progress line,
with `record.stage` available whenever the line is tied to a specific stage.

Run with:
    python examples/subscribe_to_progress.py

Uses MergeStage/ExportStage only (pure Python, no torch/ffmpeg dependency)
so this runs anywhere without ML dependencies installed - it seeds a
throwaway job directory with the ASR/diarization artifacts those two stages
expect as input.
"""

import json
import logging
import tempfile
from pathlib import Path

from diarrhizer.pipeline.runner import run_pipeline
from diarrhizer.pipeline.stages.merge import MergeStage
from diarrhizer.pipeline.stages.export import ExportStage


class ProgressHandler(logging.Handler):
    """Stand-in for whatever a GUI would do with progress: route it to a
    per-stage progress bar/log pane instead of re-printing raw text.
    """

    def emit(self, record: logging.LogRecord) -> None:
        stage = getattr(record, "stage", "-")
        print(f"[progress] stage={stage!r}: {record.getMessage()}")


def _seed_job_dir(job_dir: Path) -> None:
    (job_dir / "asr").mkdir(parents=True)
    (job_dir / "diar").mkdir(parents=True)
    (job_dir / "asr" / "transcript.json").write_text(
        json.dumps({
            "segments": [{"start": 0, "end": 1, "text": "Hello world"}],
            "words": [],
        }),
        encoding="utf-8",
    )
    (job_dir / "diar" / "diarization.json").write_text(
        json.dumps({"segments": [{"start": 0, "end": 1, "speaker": "Speaker_00"}]}),
        encoding="utf-8",
    )


def main() -> None:
    logger = logging.getLogger("diarrhizer")
    logger.setLevel(logging.INFO)
    logger.addHandler(ProgressHandler())

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        input_file = tmp_path / "input.wav"
        input_file.write_bytes(b"fake")
        out_dir = tmp_path / "out"
        job_dir = out_dir / "example_job"
        _seed_job_dir(job_dir)

        run_pipeline(
            input_path=input_file,
            out_dir=out_dir,
            stages=[MergeStage(), ExportStage()],
            job_dir=job_dir,
        )


if __name__ == "__main__":
    main()
