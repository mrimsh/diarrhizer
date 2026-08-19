"""Export stage for generating final output files."""

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from diarrhizer.export.markdown_export import export_to_markdown
from diarrhizer.export.json_export import export_to_json
from diarrhizer.pipeline.cache import is_stale
from diarrhizer.utils import write_text_atomic

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
# @outputs: one file per Exporter in EXPORTERS (artifacts/export/result.md, artifacts/export/result.json)
# @sideEffects: Reads JSON files, writes export files to disk,
#   logs progress via logging (INFO, extra={"stage": "export"})
# @errors: FileNotFoundError if input artifacts missing
# @see: STAGE:MERGE, EXPORT:MARKDOWN, EXPORT:JSON
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
        metadata = segments_data.get("metadata", {})

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

    def is_cache_valid(self, job_dir: Path) -> bool:
        """Check if stage output exists and is up to date relative to its input segments.

        All registered formats are one atomic group: if any single one is
        missing or older than segments.json, the whole stage is considered
        stale and every format is re-rendered.

        Args:
            job_dir: Job directory path

        Returns:
            True if every registered output exists and is valid
        """
        artifacts = self.get_artifact_paths(job_dir)
        return not is_stale(
            outputs=list(self.get_output_paths(job_dir).values()),
            inputs=[artifacts["segments"]],
        )


# [SEMANTIC-END] STAGE:EXPORT
