"""Convert stage for audio normalization."""

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from diarrhizer.adapters.ffmpeg import FFmpegAdapter
from diarrhizer.audio_formats import WAV_FORMAT, AudioFormat, archive_path
from diarrhizer.pipeline.cache import config_mismatch, is_stale
from diarrhizer.utils import read_json, write_json_atomic

if TYPE_CHECKING:
    from diarrhizer.pipeline.runner import JobContext

logger = logging.getLogger(__name__)


# [SEMANTIC-BEGIN] STAGE:CONVERT
# @purpose: Normalize input media to WAV mono 16kHz with optional audio profiles, plus an archive copy
# @description: Uses FFmpeg adapter to convert input audio/video to a standardized format with preprocessing.
#   Every profile, including split-stereo, produces audio/normalized.wav - the working file transcribe/diarize
#   decode - so pipeline shape never depends on audio_profile. split-stereo additionally produces
#   audio/normalized_left.wav / normalized_right.wav as extra artifacts (not consumed downstream).
#   config.audio_format (CONFIG:AUDIO_FORMATS) may add an archive copy of each of those files
#   (audio/archive[_left|_right].<ext>), encoded from the original input with the same filters; with
#   keep_wav=false, finalize() drops the working WAVs once the whole job completed, leaving only archives.
#   get_output_paths()/is_cache_valid() follow what the last run recorded in meta/run.json (profile, format,
#   retention), so a job whose WAV was dropped stays cached. When only the storage format/retention changed,
#   run() reuses the existing WAV (or leaves a dropped one dropped) instead of rewriting it, so its mtime -
#   and with it transcribe/diarize's cache - survives; a new source file, profile change or force rewrites it.
#   Stale audio files from an earlier format/profile are deleted only after the new ones were written.
# @inputs: job.input_path, config.audio_profile, config.audio_format
# @outputs: artifacts/audio/normalized.wav, meta/run.json, optional artifacts/audio/archive.<ext>, and for
#   split-stereo also artifacts/audio/normalized_left.wav / normalized_right.wav (+ archive_left/right.<ext>)
# @sideEffects: Creates output directory structure, writes audio file(s) to disk, deletes superseded audio
#   files, finalize() deletes working WAVs, logs progress via logging (INFO, extra={"stage": "convert"})
# @errors: RuntimeError, FileNotFoundError
# @see: ADAPTER:FFMPEG, CONFIG:AUDIO_FORMATS, ARTIFACTS:JOB_AUDIO, PIPELINE:RUNNER
class ConvertStage:
    """Stage for converting input media to normalized WAV format."""

    # Stage name for identification
    NAME = "convert"

    # Output paths relative to job directory
    AUDIO_DIR = "audio"
    META_DIR = "meta"
    NORMALIZED_WAV = "audio/normalized.wav"
    NORMALIZED_LEFT_WAV = "audio/normalized_left.wav"
    NORMALIZED_RIGHT_WAV = "audio/normalized_right.wav"
    META_RUN_JSON = "meta/run.json"

    # (artifact key suffix, archive filename suffix, split-stereo channel)
    _SPLIT_CHANNELS = (("_left", "_left", 0), ("_right", "_right", 1))

    def __init__(self) -> None:
        """Initialize the convert stage."""
        self._ffmpeg_adapter: FFmpegAdapter | None = None

    @property
    def ffmpeg_adapter(self) -> FFmpegAdapter:
        """Get or create FFmpeg adapter (lazy initialization)."""
        if self._ffmpeg_adapter is None:
            self._ffmpeg_adapter = FFmpegAdapter()
        return self._ffmpeg_adapter

    def run(self, job: "JobContext") -> dict:
        """Run the convert stage.

        Args:
            job: Job context containing input path and configuration

        Returns:
            Dictionary with stage output paths and metadata
        """
        input_path = job.input_path
        job_dir = job.job_dir
        config = job.config

        logger.info(f"[{self.NAME}] Converting: {input_path}", extra={"stage": self.NAME})

        audio_profile = config.audio_profile
        audio_format = config.audio_format
        split = audio_profile == FFmpegAdapter.PROFILE_SPLIT_STEREO

        # Build output paths
        audio_output = job_dir / self.NORMALIZED_WAV
        meta_output = job_dir / self.META_RUN_JSON
        wav_paths = self._wav_paths(job_dir, split)
        archive_targets = self._archive_targets(job_dir, audio_format, split)

        # Whether to (re)run is decided by the pipeline runner (is_cache_valid
        # + force flags), not here - run() always does the work when called.
        # It only decides whether the *working WAV* needs rewriting.

        # Ensure output directories exist
        audio_output.parent.mkdir(parents=True, exist_ok=True)
        meta_output.parent.mkdir(parents=True, exist_ok=True)

        # The WAV's content depends only on source + audio_profile. If neither
        # changed since the last conversion, rewriting it would bump its mtime
        # and needlessly invalidate transcribe/diarize.
        recorded = self._read_recorded_config(job_dir)
        forced = config.force or config.force_stage == self.NAME
        signal_unchanged = (
            not forced
            and recorded.get("audio_profile") == audio_profile
            and not is_stale(outputs=[meta_output], inputs=[input_path])
        )
        if signal_unchanged and all(p.exists() for p in wav_paths.values()):
            wav_action = "reuse"
        elif (
            signal_unchanged
            and not audio_format.keeps_wav
            and not any(p.exists() for p in wav_paths.values())
        ):
            # Retention already dropped the WAV for this very signal - only the
            # archive is being redone, so don't bring the WAV back.
            wav_action = "skip"
        else:
            wav_action = "convert"

        start_time = datetime.now()
        if wav_action == "convert":
            self.ffmpeg_adapter.convert_to_wav(
                input_path=str(input_path),
                output_path=str(audio_output),
                audio_profile=audio_profile,
            )
        else:
            logger.info(
                f"[{self.NAME}] Working WAV "
                + ("kept as is" if wav_action == "reuse" else "not recreated (keep_wav=false)")
                + ": source and audio profile unchanged",
                extra={"stage": self.NAME},
            )

        for path, channel in archive_targets.values():
            self.ffmpeg_adapter.encode(
                input_path=input_path,
                output_path=path,
                encoder_args=audio_format.encoder_args(channels=1 if channel is not None else None),
                audio_profile=audio_profile,
                channel=channel,
            )
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        wav_outputs = [p for p in wav_paths.values() if p.exists()]
        archive_outputs = [path for path, _ in archive_targets.values()]
        self._remove_superseded_audio(job_dir, keep=set(wav_outputs) | set(archive_outputs))

        output_paths = wav_outputs + archive_outputs
        main_output = wav_outputs[0] if wav_outputs else archive_outputs[0]

        # Write metadata
        meta_info = {
            "stage": self.NAME,
            "input_path": str(input_path),
            "output_path": str(main_output),
            "output_paths": [str(p) for p in output_paths],
            "archive_paths": [str(p) for p in archive_outputs],
            "config": {
                "sample_rate": FFmpegAdapter.TARGET_SAMPLE_RATE,
                "channels": FFmpegAdapter.TARGET_CHANNELS,
                "format": "wav",
                "audio_profile": audio_profile,
                "audio_format": audio_format.to_dict(),
            },
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "pipeline_config": {
                "min_speakers": config.min_speakers,
                "max_speakers": config.max_speakers,
                "language": config.language,
                "device": config.device,
                "asr_model": config.asr_model,
                "asr_compute_type": config.asr_compute_type,
                "asr_beam_size": config.asr_beam_size,
                "asr_temperature": config.asr_temperature,
                "audio_profile": audio_profile,
                "audio_format": audio_format.name,
            },
        }

        write_json_atomic(meta_output, meta_info)

        logger.info(f"[{self.NAME}] Completed in {duration:.2f}s", extra={"stage": self.NAME})
        logger.info(f"[{self.NAME}] Profile: {audio_profile}", extra={"stage": self.NAME})
        logger.info(
            f"[{self.NAME}] Audio format: {audio_format.name}"
            + ("" if audio_format.keeps_wav else " (working WAV removed when the job completes)"),
            extra={"stage": self.NAME},
        )
        if len(output_paths) > 1:
            logger.info(
                f"[{self.NAME}] Outputs: {', '.join(str(p) for p in output_paths)}",
                extra={"stage": self.NAME},
            )
        else:
            logger.info(f"[{self.NAME}] Output: {main_output}", extra={"stage": self.NAME})

        return {
            "stage": self.NAME,
            "status": "completed",
            "output_path": str(main_output),
            "output_paths": [str(p) for p in output_paths],
            "duration_seconds": duration,
        }

    def finalize(self, job: "JobContext") -> None:
        """Drop the working WAV(s) once the whole job completed, if the recorded
        format says not to keep them. Called by the runner after the last stage.

        A WAV is only deleted when its archive copy exists, so the job never
        loses its only copy of the audio.
        """
        job_dir = job.job_dir
        recorded = self._read_recorded_config(job_dir)
        audio_format = self._recorded_format(recorded)
        if audio_format.keeps_wav:
            return
        split = recorded.get("audio_profile") == FFmpegAdapter.PROFILE_SPLIT_STEREO
        wav_paths = self._wav_paths(job_dir, split)
        archives = self._archive_targets(job_dir, audio_format, split)
        for key, wav in wav_paths.items():
            archive, _ = archives[key.replace("audio", "archive", 1)]
            if wav.exists() and archive.exists():
                wav.unlink()
                logger.info(
                    f"[{self.NAME}] Removed working WAV {wav.name} (kept {archive.name})",
                    extra={"stage": self.NAME},
                )

    def _wav_paths(self, job_dir: Path, split: bool) -> dict:
        paths = {"audio": job_dir / self.NORMALIZED_WAV}
        if split:
            paths["audio_left"] = job_dir / self.NORMALIZED_LEFT_WAV
            paths["audio_right"] = job_dir / self.NORMALIZED_RIGHT_WAV
        return paths

    def _archive_targets(self, job_dir: Path, audio_format: AudioFormat, split: bool) -> dict:
        """{artifact key: (archive path, split-stereo channel or None)}; empty when no archive."""
        if not audio_format.writes_archive:
            return {}
        ext = audio_format.extension
        targets = {"archive": (archive_path(job_dir, ext), None)}
        if split:
            for key_suffix, file_suffix, channel in self._SPLIT_CHANNELS:
                targets[f"archive{key_suffix}"] = (archive_path(job_dir, ext, file_suffix), channel)
        return targets

    def _remove_superseded_audio(self, job_dir: Path, keep: set) -> None:
        """Delete convert-owned audio a previous format/profile left behind
        (e.g. archive.mp3 after switching to opus, or split-stereo's extras
        after switching to raw) - reclaiming that space is the point of formats.
        """
        audio_dir = job_dir / self.AUDIO_DIR
        for pattern in ("normalized*.wav", "archive*.*"):
            for path in audio_dir.glob(pattern):
                if path not in keep and path.is_file():
                    try:
                        path.unlink()
                        logger.info(
                            f"[{self.NAME}] Removed superseded {path.name}",
                            extra={"stage": self.NAME},
                        )
                    except OSError as e:
                        logger.warning(f"Could not delete {path}: {e}")

    def _read_recorded_config(self, job_dir: Path) -> dict:
        """The "config" block a prior convert run recorded in meta/run.json
        (audio_profile, audio_format, ...), or {} if there is none yet.
        """
        data = read_json(job_dir / self.META_RUN_JSON)
        if not isinstance(data, dict) or not isinstance(data.get("config"), dict):
            return {}
        return data["config"]

    @staticmethod
    def _recorded_format(recorded: dict) -> AudioFormat:
        """The AudioFormat a prior run recorded; jobs from before formats existed
        (or with an unreadable record) only ever had the working WAV.
        """
        data = recorded.get("audio_format")
        if not isinstance(data, dict):
            return WAV_FORMAT
        try:
            return AudioFormat.from_dict(data)
        except (TypeError, ValueError):
            return WAV_FORMAT

    def get_artifact_paths(self, job_dir: Path) -> dict:
        """Get the expected artifact paths for this stage.

        Follows what the last run recorded in meta/run.json: split-stereo's
        per-channel extras, the archive copy of a non-WAV format, and the
        working WAV only while the format keeps it - so a job whose WAV was
        dropped on purpose is not seen as missing an output.

        Args:
            job_dir: Job directory path

        Returns:
            Dictionary of artifact name to path
        """
        recorded = self._read_recorded_config(job_dir)
        audio_format = self._recorded_format(recorded)
        split = recorded.get("audio_profile") == FFmpegAdapter.PROFILE_SPLIT_STEREO

        paths = self._wav_paths(job_dir, split) if audio_format.keeps_wav else {}
        for key, (path, _) in self._archive_targets(job_dir, audio_format, split).items():
            paths[key] = path
        paths["meta"] = job_dir / self.META_RUN_JSON
        return paths

    def get_output_paths(self, job_dir: Path) -> dict:
        """Get only the artifact paths this stage produces.

        Convert is the first stage: it has no job-directory inputs of its
        own (it reads the original external input file), so its outputs are
        the same as its full artifact set.

        Args:
            job_dir: Job directory path

        Returns:
            Dictionary of output artifact name to path
        """
        return self.get_artifact_paths(job_dir)

    def is_cache_valid(self, job: "JobContext") -> bool:
        """Check outputs exist, match the current audio_profile and audio format, and are newer than the source."""
        job_dir = job.job_dir
        recorded = self._read_recorded_config(job_dir)
        recorded_profile = recorded.get("audio_profile")
        if recorded_profile is not None and recorded_profile != job.config.audio_profile:
            return False
        if config_mismatch(
            self._recorded_format(recorded).cache_key(), job.config.audio_format.cache_key()
        ):
            return False
        outputs = list(self.get_output_paths(job_dir).values())
        inputs = [job.input_path] if job.input_path.exists() else []
        return not is_stale(outputs=outputs, inputs=inputs)


# [SEMANTIC-END] STAGE:CONVERT
