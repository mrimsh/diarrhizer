"""Export stage for generating final output files."""

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from diarrhizer.export.json_export import export_to_json
from diarrhizer.export.markdown_export import export_to_markdown
from diarrhizer.export.speakers import normalize_speaker_id
from diarrhizer.export.text_export import export_to_text
from diarrhizer.pipeline.cache import is_stale
from diarrhizer.utils import read_json, write_text_atomic

if TYPE_CHECKING:
    from diarrhizer.pipeline.runner import JobContext, PipelineConfig

logger = logging.getLogger(__name__)


# [SEMANTIC-BEGIN] STAGE:EXPORT
# @purpose: Export merged segments to every registered output format
# @description: Reads merged/segments.json and renders it through each
#   Exporter in ExportStage.EXPORTERS (currently Markdown and JSON). Adding a
#   format (SRT/VTT/HTML/DOCX/...) means appending an Exporter entry to that
#   list - run() and get_output_paths()/get_artifact_paths()/is_cache_valid()
#   all derive from it, so none of them need editing for a new format.
#   Registered outputs are treated as one atomic cache group (matching prior
#   behavior, back when there were only two hardcoded outputs): is_cache_valid()
#   is False, and --force-stage export deletes every output, if even one
#   format is missing or stale, so formats can never drift out of sync with
#   segments.json or with each other.
# @inputs: artifacts/merged/segments.json
# @outputs: one file per Exporter in EXPORTERS (result.md, result.txt, result.json)
# @sideEffects: Reads JSON files, writes export files to disk,
#   logs progress via logging (INFO, extra={"stage": "export"})
# @errors: FileNotFoundError if input artifacts missing
# @see: STAGE:MERGE, EXPORT:MARKDOWN, EXPORT:TEXT, EXPORT:JSON
@dataclass(frozen=True)
class Exporter:
    """A single registered export format.

    export_fn renders segments to text; output_path is where run() writes
    that text, relative to job_dir.
    """

    name: str
    export_fn: Callable[[list[dict[str, Any]], "PipelineConfig", str], str]
    output_path: str


class ExportStage:
    """Stage for exporting processed transcripts to output files."""

    # Stage name for identification
    NAME = "export"

    # Output paths relative to job directory
    EXPORT_DIR = "export"

    # Input artifact path
    INPUT_SEGMENTS = "merged/segments.json"

    # Registered export formats. To add a format, append an Exporter here -
    # no other method in this class needs to change.
    EXPORTERS: tuple[Exporter, ...] = (
        Exporter("markdown", export_to_markdown, "export/result.md"),
        Exporter("text", export_to_text, "export/result.txt"),
        Exporter("json", export_to_json, "export/result.json"),
    )

    def run(self, job: "JobContext") -> dict:
        """Run the export stage.

        Args:
            job: Job context containing input path and configuration

        Returns:
            Dictionary with stage output paths and metadata
        """
        job_dir = job.job_dir
        config = job.config

        # Build input path
        segments_input = job_dir / self.INPUT_SEGMENTS

        logger.info(f"[{self.NAME}] Exporting results", extra={"stage": self.NAME})

        # Check if input exists
        if not segments_input.exists():
            raise FileNotFoundError(
                f"Segments not found: {segments_input}. "
                "Please run the merge stage first."
            )

        # Load input segments
        with open(segments_input, "r", encoding="utf-8") as f:
            segments_data = json.load(f)

        # Extract segments list
        segments = segments_data.get("segments", [])

        # Get input path from config
        input_path = config.input_file

        start_time = datetime.now()

        # Render and write every registered format
        outputs: dict[str, str] = {}
        for exporter in self.EXPORTERS:
            content = exporter.export_fn(segments, config, input_path)
            output_file = job_dir / exporter.output_path
            write_text_atomic(output_file, content)
            outputs[exporter.name] = str(output_file)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        logger.info(f"[{self.NAME}] Completed in {duration:.2f}s", extra={"stage": self.NAME})
        logger.info(f"[{self.NAME}] Segments: {len(segments)}", extra={"stage": self.NAME})
        for exporter in self.EXPORTERS:
            logger.info(
                f"[{self.NAME}] {exporter.name}: {outputs[exporter.name]}",
                extra={"stage": self.NAME},
            )

        return {
            "stage": self.NAME,
            "status": "completed",
            "outputs": outputs,
            "num_segments": len(segments),
            "duration_seconds": duration,
        }

    def get_artifact_paths(self, job_dir: Path) -> dict:
        """Get the expected artifact paths for this stage.

        Args:
            job_dir: Job directory path

        Returns:
            Dictionary of artifact name to path (input segments plus one
            entry per registered Exporter)
        """
        artifacts = {"segments": job_dir / self.INPUT_SEGMENTS}
        artifacts.update(self.get_output_paths(job_dir))
        return artifacts

    def get_output_paths(self, job_dir: Path) -> dict:
        """Get only the artifact paths this stage produces (not its inputs).

        Args:
            job_dir: Job directory path

        Returns:
            Dictionary of Exporter.name to path, one entry per registered
            Exporter in EXPORTERS.
        """
        return {exporter.name: job_dir / exporter.output_path for exporter in self.EXPORTERS}

    def is_cache_valid(self, job: "JobContext") -> bool:
        """Check if every registered output exists, is newer than segments.json,
        and was rendered with the same speaker mapping as this run.
        """
        job_dir = job.job_dir
        artifacts = self.get_artifact_paths(job_dir)
        if is_stale(
            outputs=list(self.get_output_paths(job_dir).values()),
            inputs=[artifacts["segments"]],
        ):
            return False
        return not self._speakers_mismatch(job)

    def _speakers_mismatch(self, job: "JobContext") -> bool:
        data = read_json(job.job_dir / "export" / "result.json")
        if not isinstance(data, dict):
            return True
        stored = (data.get("metadata") or {}).get("speakers")
        expected = job.config.speakers or {}
        if stored is None:
            return bool(expected)
        return _normalized_speakers(stored) != _normalized_speakers(expected)


def _normalized_speakers(mapping: dict) -> dict:
    return {normalize_speaker_id(str(k)): v for k, v in mapping.items()}


# [SEMANTIC-END] STAGE:EXPORT
