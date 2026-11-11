"""Pipeline runner for orchestrating stage execution."""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol, Sequence

# Light import: merge.py pulls in nothing heavier than the stdlib, so importing
# the algorithm's own default here keeps config and implementation in sync
# without dragging torch/whisperx into every runner import.
from diarrhizer.pipeline.stages.merge import DEFAULT_MIN_TURN_DURATION
from diarrhizer.audio_formats import WAV_FORMAT, AudioFormat, resolve_audio_format
from diarrhizer.utils import read_json, write_json_atomic
from diarrhizer.artifacts import locked_job, job_write_lock, recover_bundles
from diarrhizer.pipeline.settings import FULL_STAGES, read_job_settings, recorded_stage_names, build_stages, set_aside_speaker_names
from diarrhizer.projects import JOB_FILE, reserve_job, resolve_input

# Configure logging
logger = logging.getLogger(__name__)


# [SEMANTIC-BEGIN] PIPELINE:RUNNER
# @purpose: Orchestrate pipeline stages execution with caching and artifact management
# @description: Runs a sequence of stages, manages job directory, handles idempotency. Supports
#   force options to override caching, --job-dir to resume an existing job directory (recovering
#   input_path from a prior convert stage's meta/run.json if not given explicitly), and
#   from_stage/to_stage to skip stages outside a range entirely. Stages outside the range are
#   never touched (not even cache-checked); stages inside it still go through the normal
#   is_stale-based caching, not an unconditional recompute. Once a run reaches the last stage, every
#   stage's optional finalize(job) hook runs (regardless of from_stage - it acts on the job's recorded
#   state, e.g. convert dropping a working WAV that the audio format says not to keep); a failure
#   there is logged, not raised, since the job itself already completed. Inputs are validated before
#   stage artifacts are written, and so is every in-range stage's optional preflight(job) hook (a stage
#   that will have to run but cannot - diarization without an HF token - fails in seconds, not after
#   the transcription). new_job_dir() atomically reserves a fresh folder for callers that start several
#   jobs in a row (GUI queue, CLI batch) and pass it as job_dir. The stage names a run was given are
#   carried as PipelineConfig.stages (convert records them for resuming). Convert records the whole job
#   config in meta/run.json; every later stage that actually ran (not cached, not skipped) then
#   updates its own STAGE_SETTINGS keys there, so a stage re-run with other settings (GUI "restart
#   stage", CLI --job-dir --from-stage) leaves the record true to what produced the artifacts.
#   Optional project/team/template/storage reserve portable job.json identities; legacy callers retain
#   timestamp names. Project locks protect scope moves, and retained originals can restore missing sources.
#   Explicit recording wall time is captured in job identity for naming independently of processing time.
# @inputs: input_path, config, out_dir, stages, job_dir, force, force_stage, from_stage, to_stage, audio_format
# @outputs: Artifacts on disk per stage definitions
# @sideEffects: Creates job directory, writes artifacts to disk, deletes artifacts when force is used,
#   finalize hooks may delete intermediate artifacts, reads audio_formats.json for non-built-in format names,
#   rewrites pipeline_config in meta/run.json after a non-convert stage ran with other settings,
#   logs pipeline/stage progress via logging (INFO; per-stage lines carry extra={"stage": <name>})
# @errors: RuntimeError, FileNotFoundError, ValueError
# @see: STAGE:CONVERT, STAGE:TRANSCRIBE, ARTIFACTS:LAYOUT, PIPELINE:CACHE, CONFIG:PIPELINE
class StageProtocol(Protocol):
    """Protocol for pipeline stages."""

    NAME: str

    def run(self, job: "JobContext") -> dict:
        """Run the stage."""
        ...

    def is_cache_valid(self, job: "JobContext") -> bool:
        """Check if stage output is cached for this job's config and artifacts."""
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
#   Field defaults here are canonical. The public runner distinguishes omission from explicit
#   overrides, restores saved settings, then falls back to these defaults; CLI uses the same path.
#   STAGE_SETTINGS next to it names, per stage, the settings that stage's output depends on.
# @inputs: (none - constructed by run_pipeline from its own parameters)
# @outputs: PipelineConfig instance, carried as JobContext.config
# @sideEffects: None (plain dataclass)
# @errors: None
# @see: PIPELINE:RUNNER, STAGE:CONVERT, STAGE:TRANSCRIBE, STAGE:DIARIZE, EXPORT:MARKDOWN, EXPORT:JSON, CLI:RUN, CONFIG:AUDIO_FORMATS
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
        audio_format: How the job's audio is stored: archive codec/quality and
            whether the working WAV is kept (resolved from a profile name by
            run_pipeline; see CONFIG:AUDIO_FORMATS)
        min_turn_duration: Shortest speaker turn, in seconds, the merge stage
            will split out on its own; shorter word runs between two runs of
            the same other speaker are folded back in as diarization jitter
        stages: Names of the stages this run was given, in order (set by
            run_pipeline; recorded in meta/run.json so a job can be resumed
            with the same stage list, e.g. without diarization)
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
    audio_format: AudioFormat = WAV_FORMAT
    min_turn_duration: float = DEFAULT_MIN_TURN_DURATION
    stages: tuple[str, ...] = ()


# run_pipeline() settings each stage's output depends on, by stage name: what a
# stage re-run with other settings may change (the GUI offers exactly these
# when restarting a stage), and what the runner records in meta/run.json after
# the stage ran. Names are run_pipeline() parameters and, except audio_format
# (PipelineConfig holds the resolved AudioFormat, run.json records its name),
# PipelineConfig fields. device is shared by the two GPU stages.
STAGE_SETTINGS: dict[str, tuple[str, ...]] = {
    "convert": ("audio_profile", "audio_format"),
    "transcribe": (
        "language",
        "device",
        "asr_model",
        "asr_compute_type",
        "asr_beam_size",
        "asr_temperature",
        "asr_condition_on_previous_text",
        "asr_initial_prompt",
        "asr_vad_filter",
        "asr_vad_min_silence_ms",
    ),
    "diarize": ("device", "min_speakers", "max_speakers"),
    "merge": ("min_turn_duration",),
    "export": (),
}
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


def new_job_dir(out_dir: str | Path, input_path: str | Path, *, project: str = "", team: str | None = None,
                naming_template: str | None = None, custom: str = "", storage: dict | None = None,
                recorded_at: str | None = None) -> Path:
    """Atomically reserve a fresh job directory, avoiding concurrent name collisions.

    generate_job_id() has one-second resolution, so two files with the same
    name queued back to back (a batch, or a job that failed instantly) could
    otherwise land in - and mix artifacts into - the same folder. Callers that
    start several jobs pass the result to run_pipeline(job_dir=...).
    """
    if project or team or naming_template is not None or storage is not None or custom or recorded_at is not None:
        return reserve_job(Path(out_dir), Path(input_path), project=project, team=team,
                           naming_template=naming_template, custom=custom, storage=storage, recorded_at=recorded_at)
    base = Path(out_dir) / generate_job_id(input_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    candidate = base
    suffix = 2
    while True:
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            candidate = base.with_name(f"{base.name}_{suffix}")
            suffix += 1


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
    recorded = data.get("input_path") if isinstance(data, dict) else None
    source = Path(recorded) if isinstance(recorded, str) and recorded else None
    return resolve_input(job_dir, source) if source is not None else None


def _record_stage_settings(job_dir: Path, stage_name: str, config: PipelineConfig) -> None:
    """Bring pipeline_config in meta/run.json in line with a stage that just ran.

    Convert writes the whole record itself. Any later stage that ran with
    other settings than the record holds (re-run via --from-stage, or the
    GUI's "restart stage") updates its own STAGE_SETTINGS keys, or History,
    resuming and the exports' headers would keep describing the old run. A
    job without a pipeline_config (from before it was recorded) is left alone.
    """
    keys = () if stage_name == "convert" else STAGE_SETTINGS.get(stage_name, ())
    if not keys:
        return
    meta_path = job_dir / "meta" / "run.json"
    data = read_json(meta_path)
    recorded = data.get("pipeline_config") if isinstance(data, dict) else None
    if not isinstance(recorded, dict):
        return
    current = {key: getattr(config, key) for key in keys}
    if stage_name == "diarize" and isinstance(recorded.get("stages"), list):
        current["stages"] = [name for name in FULL_STAGES if name == "diarize" or name in recorded["stages"]]
    if all(key in recorded and recorded[key] == value for key, value in current.items()):
        return
    recorded.update(current)
    try:
        write_json_atomic(meta_path, data)
    except OSError as e:
        # The stage's artifacts are already written; a stale record is not worth failing the job.
        logger.warning(f"Could not update settings in {meta_path}: {e}")


_UNSET = object()
_RESUMABLE_SETTINGS = (
    "min_speakers", "max_speakers", "language", "device", "speakers", "asr_model",
    "asr_compute_type", "asr_beam_size", "asr_temperature", "asr_condition_on_previous_text",
    "asr_initial_prompt", "asr_hotwords_file", "asr_vad_filter", "asr_vad_min_silence_ms",
    "audio_profile", "audio_format", "min_turn_duration",
)


# [SEMANTIC-BEGIN] PIPELINE:EXECUTE
# @purpose: Execute a pipeline using shared, reproducible resume settings
# @description: Omitted parameters restore recorded settings; explicit None clears nullable values.
#   Omitted stages restore the recorded list. Planning and validation precede output changes.
#   Explicit names are saved when export is reached; an empty map prevents stale names returning on resume.
#   Existing jobs are write-locked before settings are read; fresh jobs reserve their name atomically
#   before preflight (empty reservations are removed on failure). Export journals are recovered under the lock before any stage can reuse the bundle.
#   New project/team jobs use portable naming/storage identity; resume keeps that identity unchanged.
#   A supplied recorded_at is for new-job allocation only; metadata edits use the locked rename API.
# @inputs: input/out paths, optional job/stages, explicit configuration overrides
# @outputs: per-stage results and artifacts
# @sideEffects: Reads saved settings, then executes pipeline stages with disk/GPU effects
# @errors: ValueError, OSError, RuntimeError for invalid settings or stage failures
# @see: CONFIG:RESUME, CONFIG:PIPELINE, PIPELINE:RUNNER, PIPELINE:BUILD_STAGES
@locked_job
def run_pipeline(
    *,
    input_path: str | Path | None = None,
    out_dir: str | Path,
    stages: Sequence[StageProtocol] | None = None,
    job_dir: str | Path | None = None,
    min_speakers: int = _UNSET,
    max_speakers: int = _UNSET,
    language: str = _UNSET,
    device: str = _UNSET,
    force: bool = PipelineConfig.force,
    force_stage: str | None = PipelineConfig.force_stage,
    from_stage: str | None = PipelineConfig.from_stage,
    to_stage: str | None = PipelineConfig.to_stage,
    speakers: dict | None = _UNSET,
    asr_model: str = _UNSET,
    asr_compute_type: str | None = _UNSET,
    asr_beam_size: int = _UNSET,
    asr_temperature: float = _UNSET,
    asr_condition_on_previous_text: bool = _UNSET,
    asr_initial_prompt: str | None = _UNSET,
    asr_hotwords_file: str | None = _UNSET,
    asr_vad_filter: bool = _UNSET,
    asr_vad_min_silence_ms: int = _UNSET,
    audio_profile: str = _UNSET,
    audio_format: AudioFormat | str | None = _UNSET,
    keep_wav: bool | None = None,
    min_turn_duration: float = _UNSET,
    project: str = "",
    team: str | None = None,
    naming_template: str | None = None,
    custom: str = "",
    storage: dict | None = None,
    recorded_at: str | None = None,
) -> dict:
    """Run a job with explicit overrides > recorded settings > defaults.

    Omitting a setting preserves it on resume; explicitly passing None resets
    nullable settings (e.g. speakers or initial prompt). An explicit stage list
    remains authoritative, allowing GUI rediarization of an ASR-only job.
    """
    supplied = locals()
    overrides = {key: supplied[key] for key in _RESUMABLE_SETTINGS if supplied[key] is not _UNSET}
    settings = read_job_settings(Path(job_dir), include_speakers="speakers" not in overrides) if job_dir is not None else {}
    settings.update(overrides)
    if stages is None:
        stages = build_stages(recorded_stage_names(Path(job_dir) if job_dir is not None else None))
    return _run_pipeline(input_path=input_path, out_dir=out_dir, stages=stages, job_dir=job_dir,
                         force=force, force_stage=force_stage, from_stage=from_stage,
                         to_stage=to_stage, keep_wav=keep_wav,
                         preserve_speakers="speakers" in overrides, project=project, team=team,
                         naming_template=naming_template, custom=custom, storage=storage, recorded_at=recorded_at, **settings)
# [SEMANTIC-END] PIPELINE:EXECUTE


def _run_pipeline(
    *,
    input_path: str | Path | None = None,
    out_dir: str | Path,
    stages: Sequence[StageProtocol],
    job_dir: str | Path | None = None,
    preserve_speakers: bool = False,
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
    audio_format: AudioFormat | str | None = PipelineConfig.audio_format,
    keep_wav: bool | None = None,
    min_turn_duration: float = PipelineConfig.min_turn_duration,
    project: str = "",
    team: str | None = None,
    naming_template: str | None = None,
    custom: str = "",
    storage: dict | None = None,
    recorded_at: str | None = None,
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
        audio_format: AudioFormat, a format profile name (built-in or from
            audio_formats.json), or None for the store's default profile
        keep_wav: Override the format's keep_wav for this run (None = as the
            profile says)
        min_turn_duration: Shortest speaker turn (seconds) the merge stage will
            split out on its own; 0 splits on every word-level speaker change

    Returns:
        Dictionary with pipeline execution results
    """
    fresh_job = job_dir is None
    input_path = Path(input_path) if input_path is not None else None
    out_dir = Path(out_dir)

    if min_turn_duration < 0:
        raise ValueError(f"min_turn_duration ({min_turn_duration}) cannot be negative")

    # Resolved before anything touches disk: an unknown profile name fails fast.
    audio_format = resolve_audio_format(audio_format, keep_wav)

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
    # validates input_path *before* creating anything on disk (matches the
    # pre-existing "fail fast, touch nothing" behavior), so a missing input
    # never leaves an empty job folder behind in the history.
    if job_dir is not None:
        job_dir = Path(job_dir)
        if input_path is not None:
            input_path = resolve_input(job_dir, input_path)

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
        audio_format=audio_format,
        min_turn_duration=min_turn_duration,
        stages=tuple(name for name in stage_names if name),
    )

    # Create job context
    job = JobContext(
        input_path=input_path,
        job_dir=job_dir,
        config=config,
    )

    if fresh_job:
        job_dir = new_job_dir(out_dir, input_path, project=project, team=team,
                              naming_template=naming_template, custom=custom, storage=storage, recorded_at=recorded_at)
        job_id = job_dir.name
        config.job_id = job_id
        job.job_dir = job_dir

    with job_write_lock(job_dir):
        # Reserve fresh names before preflight so a colliding older job cannot
        # make a stage mistake its cache for this new job's artifacts.
        try:
            for index, stage in enumerate(stages):
                preflight = getattr(stage, "preflight", None)
                if preflight is not None and from_index <= index <= to_index:
                    preflight(job)
        except Exception:
            if fresh_job:
                try:
                    (job_dir / JOB_FILE).unlink(missing_ok=True)
                    job_dir.rmdir()  # Only our empty reservation; never remove contents.
                except OSError:
                    pass
            raise

        job_dir.mkdir(parents=True, exist_ok=True)

        recover_bundles(job_dir / "export")
        logger.info(f"=" * 50)
        logger.info(f"Diarrhizer Pipeline")
        logger.info(f"=" * 50)
        logger.info(f"Input: {input_path}")
        logger.info(f"Output: {job_dir}")
        logger.info(f"Job ID: {job_id}")
        logger.info(f"Language: {language}")
        logger.info(f"Device: {device}")
        logger.info(f"Speakers: {min_speakers}-{max_speakers}")
        logger.info(
            f"Audio format: {audio_format.name}"
            + ("" if audio_format.keeps_wav else " (working WAV not kept)")
        )
        if from_stage or to_stage:
            logger.info(f"Stage range: {from_stage or stage_names[0]} -> {to_stage or stage_names[-1]}")
        if force:
            logger.info(f"FORCE: Recomputing all stages")
        elif force_stage:
            logger.info(f"FORCE: Recomputing stage '{force_stage}' only")
        logger.info(f"=" * 50)

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
                logger.info(
                    f"\n--- Stage: {stage_name} (skipped, outside stage range) ---",
                    extra={"stage": stage_name},
                )
                results.append({"stage": stage_name, "status": "skipped"})
                continue

            logger.info(f"\n--- Stage: {stage_name} ---", extra={"stage": stage_name})

            # Determine if this stage should be forced
            should_force = force or (force_stage == stage_name)

            if stage_name == "export" and preserve_speakers:
                write_json_atomic(job_dir / "speakers.json", config.speakers or {})

            # Check cache before running (skip if not forced and cache is valid)
            if not should_force and stage.is_cache_valid(job):
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
                    logger.info(
                        f"Stage {stage_name}: using cached output from {artifact_path}",
                        extra={"stage": stage_name},
                    )
                else:
                    logger.info(
                        f"Stage {stage_name}: using cached output", extra={"stage": stage_name}
                    )
                results.append({
                    "stage": stage_name,
                    "status": "cached",
                })
                continue

            # Only a stage that actually runs changes the saved name mapping.
            # Pending/cancelled jobs and cached diarization leave it untouched.
            if stage_name == "diarize":
                set_aside_speaker_names(job_dir)
                if not preserve_speakers:
                    config.speakers = None

            # If forcing, delete this stage's own outputs first to avoid partial
            # state. Deliberately uses get_output_paths(), not get_artifact_paths():
            # the latter also includes this stage's *inputs* (other stages'
            # outputs), which must never be deleted here.
            if should_force and stage_name != "export":
                logger.info(
                    f"Stage {stage_name}: forcing recompute (--force flag)",
                    extra={"stage": stage_name},
                )
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
                logger.info(f"Stage {stage_name}: running...", extra={"stage": stage_name})

            # Run the stage
            try:
                result = stage.run(job)
                results.append(result)
            except Exception as e:
                print(f"[{stage_name}] Error: {e}")
                raise
            _record_stage_settings(job_dir, stage_name, config)

        # The job is complete only once the last stage ran (or was cached) - a
        # --to-stage run that stops early leaves intermediate files for later stages.
        if stages and to_index == len(stages) - 1:
            for stage in stages:
                finalize = getattr(stage, "finalize", None)
                if finalize is None:
                    continue
                try:
                    finalize(job)
                except OSError as e:
                    logger.warning(f"Stage {getattr(stage, 'NAME', 'unknown')}: cleanup failed: {e}")

        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()

        logger.info(f"\n{'=' * 50}")
        logger.info(f"Pipeline completed in {total_duration:.2f}s")
        logger.info(f"Output directory: {job_dir}")
        logger.info(f"{'=' * 50}")

        return {
            "job_id": job_id,
            "job_dir": str(job_dir),
            "stages": results,
            "total_duration_seconds": total_duration,
        }
