"""FFmpeg adapter for audio normalization."""

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional, List

# Environment variable used to override the FFmpeg executable path.
# See resolve_ffmpeg_path() for the full resolution order.
ENV_FFMPEG_PATH = "DIARRHIZER_FFMPEG_PATH"


def resolve_ffmpeg_path(explicit: Optional[str | Path] = None) -> Optional[str]:
    """Resolve the path to the FFmpeg executable.

    Resolution order:
        1. `explicit` argument (e.g. FFmpegAdapter(ffmpeg_path=...))
        2. `DIARRHIZER_FFMPEG_PATH` environment variable
        3. `ffmpeg` found on PATH (shutil.which)

    An override from (1) or (2) that points at a nonexistent file raises
    immediately instead of silently falling back to the next priority
    level - a misconfigured override should fail loudly, not get masked
    by a PATH lookup the user didn't intend to use.

    Returns:
        The resolved path as a string, or None if nothing on PATH resolves
        and no override was given.

    Raises:
        FileNotFoundError: `explicit` or the env var is set but the path
            does not point to an existing file.
    """
    if explicit is not None:
        explicit_path = Path(explicit)
        if not explicit_path.is_file():
            raise FileNotFoundError(
                f"FFmpeg not found at explicit path (ffmpeg_path argument): {explicit_path}"
            )
        return str(explicit_path)

    env_value = os.environ.get(ENV_FFMPEG_PATH)
    if env_value:
        env_path = Path(env_value)
        if not env_path.is_file():
            raise FileNotFoundError(
                f"FFmpeg not found at path from {ENV_FFMPEG_PATH}: {env_path}"
            )
        return str(env_path)

    return shutil.which("ffmpeg")


# [SEMANTIC-BEGIN] ADAPTER:FFMPEG
# @purpose: Wrap FFmpeg calls for audio normalization and format conversion
# @description: Provides a clean interface to FFmpeg for converting media files to WAV with optional profiles.
#   Locates the ffmpeg executable via resolve_ffmpeg_path(): explicit ffmpeg_path arg > DIARRHIZER_FFMPEG_PATH
#   env var > PATH lookup. split-stereo always writes the standard mono downmix to output_path in addition
#   to the per-channel _left/_right files, so every profile leaves the same primary file behind (see
#   _convert_split_stereo).
# @inputs: input_path (str or Path), output_path (str or Path), audio_profile, ffmpeg_path (optional override)
# @outputs: Path to converted audio file, or [output_path, left_path, right_path] for split-stereo
# @sideEffects: Executes FFmpeg subprocess, creates output file(s) on disk, reads DIARRHIZER_FFMPEG_PATH env var
# @errors: RuntimeError if FFmpeg is not found or conversion fails
# @see: STAGE:CONVERT, DIAGNOSTICS:DOCTOR
class FFmpegAdapter:
    """Adapter for FFmpeg audio conversion operations."""

    # Target audio format: WAV, mono, 16kHz
    TARGET_SAMPLE_RATE = 16000
    TARGET_CHANNELS = 1  # Mono

    # Audio profile filter presets
    PROFILE_RAW = "raw"
    PROFILE_VOICE_CALL = "voice-call"
    PROFILE_DENOISE_LIGHT = "denoise-light"
    PROFILE_SPLIT_STEREO = "split-stereo"

    def __init__(self, ffmpeg_path: Optional[str | Path] = None) -> None:
        """Initialize the FFmpeg adapter and verify FFmpeg availability.

        Args:
            ffmpeg_path: Explicit path to the ffmpeg executable. Takes priority
                over the DIARRHIZER_FFMPEG_PATH environment variable and PATH.
        """
        self._ffmpeg_path: Optional[str] = None
        self._verify_ffmpeg(ffmpeg_path)

    def _verify_ffmpeg(self, ffmpeg_path: Optional[str | Path] = None) -> None:
        """Resolve and verify FFmpeg is available (explicit arg > env var > PATH)."""
        try:
            resolved = resolve_ffmpeg_path(ffmpeg_path)
        except FileNotFoundError as e:
            raise RuntimeError(str(e)) from e

        if resolved is None:
            raise RuntimeError(
                "FFmpeg not found. Install FFmpeg and add it to your system PATH, "
                f"set the {ENV_FFMPEG_PATH} environment variable, or pass "
                "ffmpeg_path=... to FFmpegAdapter(). See: https://ffmpeg.org/download.html"
            )
        self._ffmpeg_path = resolved

    @property
    def ffmpeg_path(self) -> str:
        """Get the path to FFmpeg executable."""
        return self._ffmpeg_path or ""

    def convert_to_wav(
        self,
        input_path: str | Path,
        output_path: str | Path,
        audio_profile: str = PROFILE_RAW,
    ) -> Path | List[Path]:
        """Convert input media file to WAV with optional processing profile.

        Args:
            input_path: Path to input media file (any format FFmpeg supports)
            output_path: Path to output WAV file. For split-stereo this is still
                the path downstream stages read (a standard mono downmix is
                written here too), plus two extra per-channel files derived
                from its stem/suffix (see _convert_split_stereo).
            audio_profile: Audio preprocessing profile (raw, voice-call, denoise-light, split-stereo)

        Returns:
            Path to the converted audio file, or for split-stereo a list
            `[output_path, left_path, right_path]` - output_path is the same
            standard mono downmix every other profile produces, so downstream
            stages don't need to know which profile ran; left/right are
            extra per-channel artifacts, not consumed by the rest of the pipeline.

        Raises:
            RuntimeError: If FFmpeg conversion fails
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if audio_profile == self.PROFILE_SPLIT_STEREO:
            return self._convert_split_stereo(input_path, output_path)
        else:
            return self._convert_single_channel(input_path, output_path, audio_profile)

    def _convert_single_channel(
        self,
        input_path: Path,
        output_path: Path,
        audio_profile: str,
    ) -> Path:
        """Convert to single channel WAV with optional audio filters.

        Args:
            input_path: Path to input media file
            output_path: Path to output WAV file
            audio_profile: Audio processing profile

        Returns:
            Path to the converted audio file
        """
        # Build base command
        cmd = [
            self._ffmpeg_path,
            "-y",  # Overwrite output file if exists
            "-i", str(input_path),
            "-ac", str(self.TARGET_CHANNELS),  # Mono
            "-ar", str(self.TARGET_SAMPLE_RATE),  # 16kHz
            "-acodec", "pcm_s16le",  # 16-bit PCM
        ]

        # Add profile-specific audio filters
        afilters = []
        if audio_profile == self.PROFILE_VOICE_CALL:
            afilters.append("lowpass=7000,highpass=200,equalizer=f=3000:width_type=q:w=1:g=3")
        elif audio_profile == self.PROFILE_DENOISE_LIGHT:
            afilters.append("afftdn=nr=12:nt=auto")

        if afilters:
            cmd.extend(["-af", ",".join(afilters)])

        cmd.append(str(output_path))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=3600,  # 1 hour max for large files
            )
            return output_path
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"FFmpeg conversion failed: {e.stderr}"
            ) from e

    def _convert_split_stereo(self, input_path: Path, output_path: Path) -> List[Path]:
        """Split stereo audio into separate left/right channel files, and also
        write the standard mono downmix to output_path itself.

        Every other profile leaves a single mono WAV at output_path that the
        rest of the pipeline (transcribe/diarize) reads unconditionally via a
        hardcoded "audio/normalized.wav" path. split-stereo used to skip that
        file entirely and only write the per-channel extras, which made those
        later stages fail with FileNotFoundError. Writing the same downmix
        here too means split-stereo behaves like every other profile for the
        rest of the pipeline; the per-channel files are additional artifacts,
        not (yet) consumed by transcribe/diarize/merge.

        Args:
            input_path: Path to input media file
            output_path: Path to the standard mono downmix (left/right
                filenames are derived from its stem/suffix)

        Returns:
            List of paths to converted audio files [output_path, left, right]
        """
        stem = output_path.stem
        suffix = output_path.suffix
        parent = output_path.parent

        left_path = parent / f"{stem}_left{suffix}"
        right_path = parent / f"{stem}_right{suffix}"

        # Standard mono downmix, same as the raw profile - lets every
        # downstream stage read audio/normalized.wav regardless of profile.
        cmd_mono = [
            self._ffmpeg_path,
            "-y",
            "-i", str(input_path),
            "-ac", str(self.TARGET_CHANNELS),
            "-ar", str(self.TARGET_SAMPLE_RATE),
            "-acodec", "pcm_s16le",
            str(output_path),
        ]

        # Extract left channel
        cmd_left = [
            self._ffmpeg_path,
            "-y",
            "-i", str(input_path),
            "-map_channel", "0.0.0",  # Left channel
            "-ar", str(self.TARGET_SAMPLE_RATE),
            "-acodec", "pcm_s16le",
            str(left_path),
        ]

        # Extract right channel
        cmd_right = [
            self._ffmpeg_path,
            "-y",
            "-i", str(input_path),
            "-map_channel", "0.0.1",  # Right channel
            "-ar", str(self.TARGET_SAMPLE_RATE),
            "-acodec", "pcm_s16le",
            str(right_path),
        ]

        try:
            subprocess.run(cmd_mono, capture_output=True, text=True, check=True, timeout=3600)
            subprocess.run(cmd_left, capture_output=True, text=True, check=True, timeout=3600)
            subprocess.run(cmd_right, capture_output=True, text=True, check=True, timeout=3600)
            return [output_path, left_path, right_path]
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"FFmpeg split-stereo conversion failed: {e.stderr}"
            ) from e

    def get_audio_info(self, input_path: str | Path) -> dict:
        """Get audio information from a media file.

        Args:
            input_path: Path to input media file

        Returns:
            Dictionary with audio properties (duration, sample_rate, channels, codec, etc.)
        """
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        cmd = [
            self._ffmpeg_path,
            "-i", str(input_path),
            "-f", "null",
            "-",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
            stderr = result.stderr

            # Parse audio info from FFmpeg output
            info: dict = {
                "path": str(input_path),
                "exists": True,
            }

            # Parse duration (format: Duration: HH:MM:SS.ms)
            duration_match = re.search(r'Duration: (\d{2}):(\d{2}):(\d{2})\.(\d{2})', stderr)
            if duration_match:
                hours, minutes, seconds, centiseconds = duration_match.groups()
                total_seconds = (
                    int(hours) * 3600
                    + int(minutes) * 60
                    + int(seconds)
                    + int(centiseconds) / 100
                )
                info["duration_seconds"] = round(total_seconds, 2)

            # Parse audio stream info (format: Stream #0:0: Audio: codec_name, ...)
            audio_stream_match = re.search(r'Stream.*Audio: (\w+)', stderr)
            if audio_stream_match:
                info["codec"] = audio_stream_match.group(1)

            # Parse sample rate (format: ... Hz, ...)
            sample_rate_match = re.search(r'(\d+) Hz', stderr)
            if sample_rate_match:
                info["sample_rate"] = int(sample_rate_match.group(1))

            # Parse channel layout (format: stereo, 5.1(side), etc.)
            channel_match = re.search(r'(stereo|mono|(\d+(\.\d+)? ch))', stderr)
            if channel_match:
                info["channels"] = channel_match.group(1)

            return info
        except Exception as e:
            return {"path": str(input_path), "exists": True, "error": str(e)}


# [SEMANTIC-END] ADAPTER:FFMPEG


# Module-level convenience function
def convert_audio(
    input_path: str | Path,
    output_path: str | Path,
    audio_profile: str = "raw",
) -> Path | List[Path]:
    """Convenience function to convert audio using FFmpeg.

    Args:
        input_path: Path to input media file
        output_path: Path to output WAV file
        audio_profile: Audio preprocessing profile

    Returns:
        Path to the converted audio file or list of paths for split-stereo
    """
    adapter = FFmpegAdapter()
    return adapter.convert_to_wav(input_path, output_path, audio_profile)