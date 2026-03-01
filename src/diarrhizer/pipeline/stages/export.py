"""Export stage for generating final output files."""

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from diarrhizer.export.markdown_export import export_to_markdown
from diarrhizer.export.json_export import export_to_json

if TYPE_CHECKING:
    from diarrhizer.pipeline.runner import JobContext


# [SEMANTIC-BEGIN] STAGE:EXPORT
# @purpose: Export merged segments to Markdown and JSON output files
# @description: Reads merged/segments.json and produces result.md and result.json
# @inputs: artifacts/merged/segments.json
# @outputs: artifacts/export/result.md, artifacts/export/result.json
# @sideEffects: Reads JSON files, writes export files to disk
# @errors: FileNotFoundError if input artifacts missing
# @see: STAGE:MERGE, EXPORT:MARKDOWN, EXPORT:JSON
class ExportStage:
    """Stage for exporting processed transcripts to output files."""

    # Stage name for identification
    NAME = "export"

    # Output paths relative to job directory
    EXPORT_DIR = "export"
    RESULT_MD = "export/result.md"
    RESULT_JSON = "export/result.json"

    # Input artifact path
    INPUT_SEGMENTS = "merged/segments.json"

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

        # Build output paths
        md_output = job_dir / self.RESULT_MD
        json_output = job_dir / self.RESULT_JSON

        print(f"[{self.NAME}] Exporting results")

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
        input_path = config.get("input_file", str(job.input_path))

        start_time = datetime.now()

        # Export to Markdown
        md_content = export_to_markdown(segments, config, input_path)
        md_output.parent.mkdir(parents=True, exist_ok=True)
        with open(md_output, "w", encoding="utf-8") as f:
            f.write(md_content)

        # Export to JSON
        json_content = export_to_json(segments, config, input_path)
        with open(json_output, "w", encoding="utf-8") as f:
            f.write(json_content)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print(f"[{self.NAME}] Completed in {duration:.2f}s")
        print(f"[{self.NAME}] Segments: {len(segments)}")
        print(f"[{self.NAME}] Markdown: {md_output}")
        print(f"[{self.NAME}] JSON: {json_output}")

        return {
            "stage": self.NAME,
            "status": "completed",
            "output_md": str(md_output),
            "output_json": str(json_output),
            "num_segments": len(segments),
            "duration_seconds": duration,
        }

    def get_artifact_paths(self, job_dir: Path) -> dict:
        """Get the expected artifact paths for this stage.

        Args:
            job_dir: Job directory path

        Returns:
            Dictionary of artifact name to path
        """
        return {
            "segments": job_dir / self.INPUT_SEGMENTS,
            "result_md": job_dir / self.RESULT_MD,
            "result_json": job_dir / self.RESULT_JSON,
        }

    def is_cache_valid(self, job_dir: Path) -> bool:
        """Check if stage output exists and is valid.

        Args:
            job_dir: Job directory path

        Returns:
            True if both outputs exist
        """
        artifacts = self.get_artifact_paths(job_dir)
        return artifacts["result_md"].exists() and artifacts["result_json"].exists()


# [SEMANTIC-END] STAGE:EXPORT
