"""Pipeline runner for orchestrating stage execution."""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, Sequence

# Configure logging
logger = logging.getLogger(__name__)


# [SEMANTIC-BEGIN] PIPELINE:RUNNER
# @purpose: Orchestrate pipeline stages execution with caching and artifact management
# @description: Runs a sequence of stages, manages job directory, handles idempotency. Supports force options to override caching.
# @inputs: input_path, config, out_dir, stages, force, force_stage
# @outputs: Artifacts on disk per stage definitions
# @sideEffects: Creates job directory, writes artifacts to disk, deletes artifacts when force is used
# @errors: RuntimeError, FileNotFoundError
# @see: STAGE:CONVERT, STAGE:TRANSCRIBE, ARTIFACTS:LAYOUT
class StageProtocol(Protocol):
    """Protocol for pipeline stages."""

    NAME: str

    def run(self, job: "JobContext") -> dict:
        """Run the stage."""
        ...

    def is_cache_valid(self, job_dir: Path) -> bool:
        """Check if stage output is cached."""
        ...

    def get_artifact_paths(self, job_dir: Path) -> dict:
        """Get the expected artifact paths for this stage."""
        ...


@dataclass
class JobContext:
    """Job context containing all information needed to run a stage.

    Attributes:
        input_path: Path to input media file
        job_dir: Job output directory
        config: Job configuration (language, device, speakers, etc.)
    """

    input_path: Path
    job_dir: Path
    config: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Ensure paths are Path objects."""
        self.input_path = Path(self.input_path)
        self.job_dir = Path(self.job_dir)


def generate_job_id(input_path: str | Path) -> str:
    """Generate a job ID from input path and timestamp.

    Args:
        input_path: Path to input media file

    Returns:
        Job ID string in format: <filename>_<timestamp>
    """
    input_path = Path(input_path)
    filename = input_path.stem
    # Use timestamp format: YYYYMMDD_HHMMSS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{filename}_{timestamp}"


# [SEMANTIC-END] PIPELINE:RUNNER


def run_pipeline(
    input_path: str | Path,
    out_dir: str | Path,
    stages: Sequence[StageProtocol],
    min_speakers: int = 1,
    max_speakers: int = 10,
    language: str = "auto",
    device: str = "cuda",
    force: bool = False,
    force_stage: str | None = None,
) -> dict:
    """Run the processing pipeline for a media file.

    Args:
        input_path: Path to input media file
        out_dir: Base output directory
        stages: Sequence of pipeline stages to run
        min_speakers: Minimum number of speakers
        max_speakers: Maximum number of speakers
        language: Language code or "auto"
        device: Device to use ("cuda" or "cpu")
        force: If True, recompute all stages regardless of cache
        force_stage: If set, only force a specific stage to recompute

    Returns:
        Dictionary with pipeline execution results
    """
    input_path = Path(input_path)
    out_dir = Path(out_dir)

    # Validate input exists
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Validate speaker range
    if min_speakers > max_speakers:
        raise ValueError(
            f"min_speakers ({min_speakers}) cannot be greater than max_speakers ({max_speakers})"
        )

    # Validate out_dir is accessible
    out_dir = Path(out_dir)
    if not out_dir.exists() and not out_dir.parent.exists():
        raise FileNotFoundError(
            f"Output directory parent does not exist: {out_dir.parent}"
        )

    # Generate job ID and create job directory
    job_id = generate_job_id(input_path)
    job_dir = out_dir / job_id

    # Create output directory if needed
    job_dir.mkdir(parents=True, exist_ok=True)

    # Build job configuration
    config = {
        "min_speakers": min_speakers,
        "max_speakers": max_speakers,
        "language": language,
        "device": device,
        "job_id": job_id,
        "input_file": str(input_path),
        "force": force,
        "force_stage": force_stage,
    }

    # Create job context
    job = JobContext(
        input_path=input_path,
        job_dir=job_dir,
        config=config,
    )

    print(f"=" * 50)
    print(f"Diarrhizer Pipeline")
    print(f"=" * 50)
    print(f"Input: {input_path}")
    print(f"Output: {job_dir}")
    print(f"Job ID: {job_id}")
    print(f"Language: {language}")
    print(f"Device: {device}")
    print(f"Speakers: {min_speakers}-{max_speakers}")
    if force:
        print(f"FORCE: Recomputing all stages")
    elif force_stage:
        print(f"FORCE: Recomputing stage '{force_stage}' only")
    print(f"=" * 50)

    # Run stages sequentially
    results: list[dict] = []
    start_time = datetime.now()

    for stage in stages:
        stage_name = getattr(stage, "NAME", "unknown")
        print(f"\n--- Stage: {stage_name} ---")

        # Determine if this stage should be forced
        should_force = force or (force_stage == stage_name)

        # Check cache before running (skip if not forced and cache is valid)
        if not should_force and stage.is_cache_valid(job_dir):
            # Get the artifact paths for the log message
            artifacts = stage.get_artifact_paths(job_dir)
            # Find the first existing artifact for the log
            artifact_path = None
            if isinstance(artifacts, dict):
                for path in artifacts.values():
                    if isinstance(path, Path) and path.exists():
                        artifact_path = path
                        break
            if artifact_path:
                print(f"Stage {stage_name}: using cached output from {artifact_path}")
            else:
                print(f"Stage {stage_name}: using cached output")
            results.append({
                "stage": stage_name,
                "status": "cached",
            })
            continue

        # If forcing, delete existing outputs first to avoid partial state
        if should_force:
            print(f"Stage {stage_name}: forcing recompute (--force flag)")
            artifacts = stage.get_artifact_paths(job_dir)
            if isinstance(artifacts, dict):
                for path in artifacts.values():
                    if isinstance(path, Path) and path.exists():
                        try:
                            path.unlink()
                            logger.debug(f"Deleted {path}")
                        except OSError as e:
                            logger.warning(f"Could not delete {path}: {e}")
        else:
            print(f"Stage {stage_name}: running...")

        # Run the stage
        try:
            result = stage.run(job)
            results.append(result)
        except Exception as e:
            print(f"[{stage_name}] Error: {e}")
            raise

    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()

    print(f"\n{'=' * 50}")
    print(f"Pipeline completed in {total_duration:.2f}s")
    print(f"Output directory: {job_dir}")
    print(f"{'=' * 50}")

    return {
        "job_id": job_id,
        "job_dir": str(job_dir),
        "stages": results,
        "total_duration_seconds": total_duration,
    }
