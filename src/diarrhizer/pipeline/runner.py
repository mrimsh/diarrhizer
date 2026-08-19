"""Pipeline runner for orchestrating stage execution."""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, Sequence

# Configure logging
logger = logging.getLogger(__name__)


# [SEMANTIC-BEGIN] PIPELINE:RUNNER
# @purpose: Orchestrate pipeline stages execution with caching and artifact management
# @description: Runs a sequence of stages, manages job directory, handles idempotency. Supports
#   force options to override caching, --job-dir to resume an existing job directory (recovering
#   input_path from a prior convert stage's meta/run.json if not given explicitly), and
#   from_stage/to_stage to skip stages outside a range entirely. Stages outside the range are
#   never touched (not even cache-checked); stages inside it still go through the normal
#   is_stale-based caching, not an unconditional recompute.
# @inputs: input_path, config, out_dir, stages, job_dir, force, force_stage, from_stage, to_stage
# @outputs: Artifacts on disk per stage definitions
# @sideEffects: Creates job directory, writes artifacts to disk, deletes artifacts when force is used
# @errors: RuntimeError, FileNotFoundError, ValueError
# @see: STAGE:CONVERT, STAGE:TRANSCRIBE, ARTIFACTS:LAYOUT, PIPELINE:CACHE, CONFIG:PIPELINE
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
        """Get all artifact paths (inputs and outputs) relevant to this stage."""
        ...

    def get_output_paths(self, job_dir: Path) -> dict:
        """Get only the artifact paths this stage produces (not its inputs)."""
        ...


# [SEMANTIC-BEGIN] CONFIG:PIPELINE
# @purpose: Single typed source of truth for pipeline stage configuration
# @description: Replaces the previous untyped config dict carried on JobContext.config.
#   Field defaults here are canonical; run_pipeline()'s own parameter defaults reference
#   these fields directly (same module, so they can never drift). cli.py's argparse
#   defaults are a separate, intentionally-synced literal set - see the comment there.
# @inputs: (none - constructed by run_pipeline from its own parameters)
# @outputs: PipelineConfig instance, carried as JobContext.config
# @sideEffects: None (plain dataclass)
# @errors: None
# @see: PIPELINE:RUNNER, STAGE:CONVERT, STAGE:TRANSCRIBE, STAGE:DIARIZE, EXPORT:MARKDOWN, EXPORT:JSON, CLI:RUN
@dataclass
class PipelineConfig:
    """Pipeline configuration shared by all stages via JobContext.config.

    Attributes:
        job_id: Generated job identifier
        input_file: Path to the original input media file
        min_speakers: Minimum number of speakers
        max_speakers: Maximum number of speakers
        language: Language code or "auto"
        device: Device to use ("cuda" or "cpu")
        force: If True, recompute all stages regardless of cache
        force_stage: If set, only force a specific stage to recompute
        from_stage: If set, skip every stage before this one (its own artifacts
            must already exist on disk, or be reachable via job resume)
        to_stage: If set, skip every stage after this one
        speakers: Optional speaker name mapping {speaker_id: display_name}
        asr_model: WhisperX model size or HF repo
        asr_compute_type: Compute type (float16, int8_float16, int8)
        asr_beam_size: Decoding beam size
        asr_temperature: Decoding temperature
        asr_condition_on_previous_text: Condition on previous text for stable decoding
        asr_initial_prompt: Initial prompt string for ASR
        asr_hotwords_file: Path to hotwords file (not yet implemented)
        asr_vad_filter: Enable VAD filtering
        asr_vad_min_silence_ms: VAD minimum silence in milliseconds
        audio_profile: Audio preprocessing profile
    """

    job_id: str
    input_file: str
    min_speakers: int = 1
    max_speakers: int = 10
    language: str = "auto"
    device: str = "cuda"
    force: bool = False
    force_stage: str | None = None
    from_stage: str | None = None
    to_stage: str | None = None
    speakers: dict | None = None
    asr_model: str = "large-v3"
    asr_compute_type: str | None = None
    asr_beam_size: int = 5
    asr_temperature: float = 0.0
    asr_condition_on_previous_text: bool = True
    asr_initial_prompt: str | None = None
    asr_hotwords_file: str | None = None
    asr_vad_filter: bool = True
    asr_vad_min_silence_ms: int = 1000
    audio_profile: str = "raw"
# [SEMANTIC-END] CONFIG:PIPELINE


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
    config: PipelineConfig

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


def _read_recorded_input_path(job_dir: Path) -> Path | None:
    """Recover the original input file path from a prior convert stage's
    meta/run.json, so resuming a job dir via --job-dir doesn't require
    re-passing the input file. Returns None if it can't be recovered.
    """
    meta_path = job_dir / "meta" / "run.json"
    if not meta_path.exists():
        return None
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    recorded = data.get("input_path")
    return Path(recorded) if recorded else None


def run_pipeline(
    *,
    input_path: str | Path | None = None,
    out_dir: str | Path,
    stages: Sequence[StageProtocol],
    job_dir: str | Path | None = None,
    min_speakers: int = PipelineConfig.min_speakers,
    max_speakers: int = PipelineConfig.max_speakers,
    language: str = PipelineConfig.language,
    device: str = PipelineConfig.device,
    force: bool = PipelineConfig.force,
    force_stage: str | None = PipelineConfig.force_stage,
    from_stage: str | None = PipelineConfig.from_stage,
    to_stage: str | None = PipelineConfig.to_stage,
    speakers: dict | None = PipelineConfig.speakers,
    asr_model: str = PipelineConfig.asr_model,
    asr_compute_type: str | None = PipelineConfig.asr_compute_type,
    asr_beam_size: int = PipelineConfig.asr_beam_size,
    asr_temperature: float = PipelineConfig.asr_temperature,
    asr_condition_on_previous_text: bool = PipelineConfig.asr_condition_on_previous_text,
    asr_initial_prompt: str | None = PipelineConfig.asr_initial_prompt,
    asr_hotwords_file: str | None = PipelineConfig.asr_hotwords_file,
    asr_vad_filter: bool = PipelineConfig.asr_vad_filter,
    asr_vad_min_silence_ms: int = PipelineConfig.asr_vad_min_silence_ms,
    audio_profile: str = PipelineConfig.audio_profile,
) -> dict:
    """Run the processing pipeline for a media file.

    Args:
        input_path: Path to input media file. Optional when job_dir resumes a
            job that already recorded it in meta/run.json (see _read_recorded_input_path).
        out_dir: Base output directory (ignored when job_dir is given)
        stages: Sequence of pipeline stages to run
        job_dir: Resume this existing job directory instead of creating a new
            one from input_path + a timestamp
        min_speakers: Minimum number of speakers
        max_speakers: Maximum number of speakers
        language: Language code or "auto"
        device: Device to use ("cuda" or "cpu")
        force: If True, recompute all stages regardless of cache
        force_stage: If set, only force a specific stage to recompute
        from_stage: If set, skip every stage before this one entirely
        to_stage: If set, skip every stage after this one entirely
        speakers: Optional speaker name mapping {speaker_id: display_name}
        asr_model: WhisperX model size or HF repo
        asr_compute_type: Compute type (float16, int8_float16, int8)
        asr_beam_size: Decoding beam size
        asr_temperature: Decoding temperature
        asr_condition_on_previous_text: Condition on previous text for stable decoding
        asr_initial_prompt: Initial prompt string for ASR
        asr_hotwords_file: Path to hotwords file (not yet implemented)
        asr_vad_filter: Enable VAD filtering
        asr_vad_min_silence_ms: VAD minimum silence in milliseconds
        audio_profile: Audio preprocessing profile

    Returns:
        Dictionary with pipeline execution results
    """
    input_path = Path(input_path) if input_path is not None else None
    out_dir = Path(out_dir)

    # Validate speaker range
    if min_speakers > max_speakers:
        raise ValueError(
            f"min_speakers ({min_speakers}) cannot be greater than max_speakers ({max_speakers})"
        )

    # Resolve the requested stage range against the stages actually supplied
    # (not a hardcoded stage list), so a caller-provided subset - e.g. tests
    # running just [MergeStage(), ExportStage()] - validates and skips
    # correctly too.
    stage_names = [getattr(s, "NAME", None) for s in stages]

    def _stage_index(name: str | None, flag: str) -> int | None:
        if name is None:
            return None
        if name not in stage_names:
            raise ValueError(
                f"Unknown {flag} '{name}'. Available stages: {', '.join(stage_names)}"
            )
        return stage_names.index(name)

    from_index = _stage_index(from_stage, "--from-stage")
    from_index = 0 if from_index is None else from_index
    to_index = _stage_index(to_stage, "--to-stage")
    to_index = len(stage_names) - 1 if to_index is None else to_index
    if stage_names and from_index > to_index:
        raise ValueError(
            f"--from-stage '{from_stage}' comes after --to-stage '{to_stage}' in "
            f"pipeline order ({', '.join(stage_names)})"
        )
    convert_in_range = "convert" in stage_names[from_index:to_index + 1]

    # Resolve the job directory: either an existing one to resume (--job-dir)
    # or a fresh one named from the input file + timestamp. Each branch
    # validates input_path *before* creating anything on disk where possible
    # (matches the pre-existing "fail fast, touch nothing" behavior); the
    # --job-dir branch is the one exception, since recovering input_path from
    # meta/run.json requires the directory to already be readable.
    if job_dir is not None:
        job_dir = Path(job_dir)
        job_dir.mkdir(parents=True, exist_ok=True)

        # Recover the original input path from a prior run's meta/run.json
        # when resuming without re-passing it explicitly.
        if input_path is None:
            input_path = _read_recorded_input_path(job_dir)
            if input_path is None:
                raise FileNotFoundError(
                    f"input_path is required: none was given, and {job_dir / 'meta' / 'run.json'} "
                    "does not record one (the convert stage must run at least once first)."
                )

        # The original media file only needs to exist on disk when the
        # convert stage is actually going to run - later stages only read
        # job_dir artifacts, and export only carries this path through as
        # descriptive text.
        if convert_in_range and not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
    else:
        if not out_dir.exists() and not out_dir.parent.exists():
            raise FileNotFoundError(
                f"Output directory parent does not exist: {out_dir.parent}"
            )
        if input_path is None:
            raise FileNotFoundError(
                "input_path is required to start a new job (or pass job_dir to resume an existing one)"
            )
        # Unlike the --job-dir resume branch, a brand new job has no
        # artifacts yet regardless of stage range, so the input file is
        # always required to exist here - not just when convert is in range.
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        job_dir = out_dir / generate_job_id(input_path)
        job_dir.mkdir(parents=True, exist_ok=True)

    job_id = job_dir.name

    # Build job configuration
    config = PipelineConfig(
        job_id=job_id,
        input_file=str(input_path),
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        language=language,
        device=device,
        force=force,
        force_stage=force_stage,
        from_stage=from_stage,
        to_stage=to_stage,
        speakers=speakers,
        asr_model=asr_model,
        asr_compute_type=asr_compute_type,
        asr_beam_size=asr_beam_size,
        asr_temperature=asr_temperature,
        asr_condition_on_previous_text=asr_condition_on_previous_text,
        asr_initial_prompt=asr_initial_prompt,
        asr_hotwords_file=asr_hotwords_file,
        asr_vad_filter=asr_vad_filter,
        asr_vad_min_silence_ms=asr_vad_min_silence_ms,
        audio_profile=audio_profile,
    )

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
    if from_stage or to_stage:
        print(f"Stage range: {from_stage or stage_names[0]} -> {to_stage or stage_names[-1]}")
    if force:
        print(f"FORCE: Recomputing all stages")
    elif force_stage:
        print(f"FORCE: Recomputing stage '{force_stage}' only")
    print(f"=" * 50)

    # Run stages sequentially
    results: list[dict] = []
    start_time = datetime.now()

    for index, stage in enumerate(stages):
        stage_name = getattr(stage, "NAME", "unknown")

        # Stages outside [from_stage, to_stage] are skipped entirely - not
        # run, not cache-checked - as opposed to stages inside the range,
        # which still go through the normal is_stale-based caching below
        # rather than being unconditionally recomputed.
        if index < from_index or index > to_index:
            print(f"\n--- Stage: {stage_name} (skipped, outside stage range) ---")
            results.append({"stage": stage_name, "status": "skipped"})
            continue

        print(f"\n--- Stage: {stage_name} ---")

        # Determine if this stage should be forced
        should_force = force or (force_stage == stage_name)

        # Check cache before running (skip if not forced and cache is valid)
        if not should_force and stage.is_cache_valid(job_dir):
            # Get this stage's own output paths for the log message. Using
            # get_output_paths() (not get_artifact_paths()) matters here:
            # get_artifact_paths() also includes this stage's *inputs*, which
            # belong to earlier stages.
            outputs = stage.get_output_paths(job_dir)
            # Find the first existing artifact for the log
            artifact_path = None
            if isinstance(outputs, dict):
                for path in outputs.values():
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

        # If forcing, delete this stage's own outputs first to avoid partial
        # state. Deliberately uses get_output_paths(), not get_artifact_paths():
        # the latter also includes this stage's *inputs* (other stages'
        # outputs), which must never be deleted here.
        if should_force:
            print(f"Stage {stage_name}: forcing recompute (--force flag)")
            outputs = stage.get_output_paths(job_dir)
            if isinstance(outputs, dict):
                for path in outputs.values():
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
