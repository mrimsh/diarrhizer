"""Storage formats for a job's imported audio.

The pipeline always processes a lossless working WAV (audio/normalized.wav,
PCM 16 kHz mono). An AudioFormat decides what is *kept* on disk: an optional
archive copy (audio/archive.<ext>) encoded with its own codec, quality, sample
rate and channel count, and whether the working WAV itself survives once the
job completes (keep_wav). Built-in presets are read-only; user profiles and
the default choice live in a JSON file shared by the CLI and the GUI.

Stdlib only (plus env_file/utils): the CLI's argument handling and the GUI's
job scanner import this, and neither may pull in torch/whisperx.
"""

import dataclasses
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from diarrhizer.env_file import REPO_ROOT
from diarrhizer.utils import write_json_atomic

# Overrides where user profiles are stored (default: <repo root>/audio_formats.json).
ENV_AUDIO_FORMATS_FILE = "DIARRHIZER_AUDIO_FORMATS_FILE"

# Job-dir layout, relative to job_dir.
WORKING_WAV = "audio/normalized.wav"
ARCHIVE_STEM = "archive"

_NAME_RE = re.compile(r"^\S+$")
_EXTENSION_RE = re.compile(r"^[a-z0-9]+$")
_BITRATE_RE = re.compile(r"^(\d+(?:\.\d+)?)([kKmM]?)$")
_PCM_BITS_RE = re.compile(r"^pcm_[suf](\d+)")


# [SEMANTIC-BEGIN] CONFIG:AUDIO_FORMATS
# @purpose: Named codec/quality/retention profiles for the audio a job keeps on disk
# @description: An AudioFormat whose encoding equals the working WAV's (WAV_FORMAT) writes no archive
#   copy and always keeps the WAV; any other format writes audio/archive.<ext> and honours keep_wav.
#   Encoder args always start with -vn: some containers accept video, e.g. an .mp4 encoded to .ogg
#   without it keeps a Theora video stream (checked with ffmpeg 9.0). AudioFormatStore holds user profiles plus
#   the default name in a JSON file; built-in names cannot be overwritten or deleted, and a default
#   pointing at a deleted profile falls back to WAV_FORMAT instead of breaking every run.
# @inputs: profile fields (validated on construction), JSON store path (DIARRHIZER_AUDIO_FORMATS_FILE)
# @outputs: AudioFormat instances, ffmpeg encoder argv fragments, audio_formats.json
# @sideEffects: AudioFormatStore.load() reads and save() atomically writes the JSON store
# @errors: ValueError for invalid fields, unknown/duplicate/built-in names, or an unreadable store file
# @see: ADAPTER:FFMPEG, STAGE:CONVERT, CONFIG:PIPELINE, ARTIFACTS:JOB_AUDIO, CLI:RUN
@dataclass(frozen=True)
class AudioFormat:
    """How a job's audio is stored.

    Attributes:
        name: Profile name (no whitespace), used on the CLI and in the GUI
        description: Free-text label shown next to the name
        codec: ffmpeg audio encoder (-c:a), e.g. pcm_s16le, flac, libmp3lame, libopus
        extension: Archive file extension (container), e.g. wav, flac, mp3, opus
        quality: VBR quality (-q:a); encoder-specific scale, e.g. 0 (best) - 9 for libmp3lame
        bitrate: Target bitrate (-b:a), e.g. "24k"; mutually exclusive with quality
        sample_rate: Archive sample rate in Hz (-ar)
        channels: Archive channel count (-ac)
        extra_args: Additional ffmpeg output options, appended verbatim
        keep_wav: Keep the working WAV after the job completes (only meaningful
            when an archive copy is written - otherwise the WAV is the only copy)
    """

    name: str
    description: str = ""
    codec: str = "pcm_s16le"
    extension: str = "wav"
    quality: float | None = None
    bitrate: str | None = None
    sample_rate: int = 16000
    channels: int = 1
    extra_args: tuple[str, ...] = ()
    keep_wav: bool = True

    def __post_init__(self) -> None:
        # JSON hands back lists; a tuple keeps the frozen dataclass hashable.
        object.__setattr__(self, "extra_args", tuple(str(arg) for arg in self.extra_args))
        object.__setattr__(self, "extension", str(self.extension).lstrip(".").lower())
        self._validate()

    def _validate(self) -> None:
        if not isinstance(self.name, str) or not _NAME_RE.match(self.name):
            raise ValueError(f"Audio format name must be non-empty with no spaces: {self.name!r}")
        prefix = f"Audio format '{self.name}'"
        if not isinstance(self.codec, str) or not _NAME_RE.match(self.codec):
            raise ValueError(f"{prefix}: codec must be an ffmpeg encoder name, got {self.codec!r}")
        if not _EXTENSION_RE.match(self.extension):
            raise ValueError(f"{prefix}: extension must be letters/digits, got {self.extension!r}")
        if self.quality is not None and self.bitrate is not None:
            raise ValueError(f"{prefix}: set either quality (-q:a) or bitrate (-b:a), not both")
        if self.quality is not None and (
            isinstance(self.quality, bool) or not isinstance(self.quality, (int, float))
        ):
            raise ValueError(f"{prefix}: quality must be a number, got {self.quality!r}")
        if self.bitrate is not None and (
            not isinstance(self.bitrate, str) or not _BITRATE_RE.match(self.bitrate)
        ):
            raise ValueError(f"{prefix}: bitrate must look like '24k' or '64000', got {self.bitrate!r}")
        if not _is_int(self.sample_rate) or not 1000 <= self.sample_rate <= 384000:
            raise ValueError(f"{prefix}: sample_rate must be 1000-384000 Hz, got {self.sample_rate!r}")
        if not _is_int(self.channels) or not 1 <= self.channels <= 8:
            raise ValueError(f"{prefix}: channels must be 1-8, got {self.channels!r}")
        if not isinstance(self.keep_wav, bool):
            raise ValueError(f"{prefix}: keep_wav must be true/false, got {self.keep_wav!r}")

    def encoder_args(self, channels: int | None = None) -> list[str]:
        """ffmpeg output options for the archive copy (everything between input and output path).

        Args:
            channels: Override the channel count, e.g. 1 for a single extracted
                split-stereo channel.
        """
        args = [
            "-vn",
            "-ac", str(self.channels if channels is None else channels),
            "-ar", str(self.sample_rate),
            "-c:a", self.codec,
        ]
        if self.bitrate is not None:
            args += ["-b:a", self.bitrate]
        if self.quality is not None:
            args += ["-q:a", f"{self.quality:g}"]
        args += list(self.extra_args)
        return args

    def encoding(self) -> dict:
        """The fields that decide the archive's bytes, JSON-shaped (lists, not tuples)."""
        return {
            "codec": self.codec,
            "extension": self.extension,
            "quality": self.quality,
            "bitrate": self.bitrate,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "extra_args": list(self.extra_args),
        }

    @property
    def writes_archive(self) -> bool:
        """False when the archive would just duplicate the working WAV."""
        return self.encoding() != WAV_FORMAT.encoding()

    @property
    def keeps_wav(self) -> bool:
        """Whether the working WAV stays on disk once the job completes."""
        return self.keep_wav or not self.writes_archive

    def cache_key(self) -> dict:
        """What convert's cache compares: encoding plus retention, not name/description."""
        return {**self.encoding(), "keep_wav": self.keeps_wav}

    def estimated_mb_per_hour(self) -> float | None:
        """Archive size per hour of audio, or None when it depends on the content (VBR, lossless)."""
        pcm = _PCM_BITS_RE.match(self.codec)
        if pcm:
            bytes_per_second = self.sample_rate * self.channels * int(pcm.group(1)) / 8
        elif self.bitrate is not None:
            number, unit = _BITRATE_RE.match(self.bitrate).groups()
            multiplier = {"": 1, "k": 1_000, "m": 1_000_000}[unit.lower()]
            bytes_per_second = float(number) * multiplier / 8
        else:
            return None
        return bytes_per_second * 3600 / 1_000_000

    def to_dict(self, include_name: bool = True) -> dict:
        data = {"description": self.description, **self.encoding(), "keep_wav": self.keep_wav}
        return {"name": self.name, **data} if include_name else data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AudioFormat":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            # A typo in a hand-edited file ("quallity") must not silently fall back to a default.
            raise ValueError(
                f"Audio format '{data.get('name')}': unknown field(s) {', '.join(unknown)}"
            )
        return cls(**dict(data))


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


# The working WAV's own encoding - the plain WAV the pipeline has always written.
WAV_FORMAT = AudioFormat(
    name="wav",
    description="Working WAV only: PCM 16 kHz mono, no archive copy (~115 MB/h)",
)

# No AAC/M4A preset on purpose: diarization's torchaudio fallback (libsndfile)
# cannot decode it on Windows, and an archive becomes the pipeline's input once
# the working WAV is dropped. FLAC/MP3/Opus decode through both paths.
BUILTIN_FORMATS: dict[str, AudioFormat] = {
    fmt.name: fmt
    for fmt in (
        WAV_FORMAT,
        AudioFormat(
            name="flac",
            description="Lossless FLAC archive, 16 kHz mono (~50-75 MB/h)",
            codec="flac",
            extension="flac",
            # Without this, ffmpeg picks 24-bit (s32) samples when the source
            # decodes to float (mp3/aac), and the "compressed" FLAC comes out
            # larger than the 16-bit WAV it was supposed to shrink.
            extra_args=("-sample_fmt", "s16"),
            keep_wav=False,
        ),
        AudioFormat(
            name="mp3-q5",
            description="MP3 VBR -q:a 5 archive, 16 kHz mono (~15-20 MB/h)",
            codec="libmp3lame",
            extension="mp3",
            quality=5,
            keep_wav=False,
        ),
        AudioFormat(
            name="opus-24k",
            description="Opus 24 kbit/s archive, 16 kHz mono (~11 MB/h)",
            codec="libopus",
            extension="opus",
            bitrate="24k",
            keep_wav=False,
        ),
    )
}

DEFAULT_FORMAT_NAME = WAV_FORMAT.name


def store_path() -> Path:
    """Where user profiles live: DIARRHIZER_AUDIO_FORMATS_FILE, else <repo root>/audio_formats.json."""
    override = os.environ.get(ENV_AUDIO_FORMATS_FILE)
    return Path(override) if override else REPO_ROOT / "audio_formats.json"


@dataclass
class AudioFormatStore:
    """Built-in presets plus user profiles and the default profile name."""

    path: Path
    default_name: str = DEFAULT_FORMAT_NAME
    custom: dict[str, AudioFormat] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "AudioFormatStore":
        """Read the store; a missing file is an empty store, a malformed one raises ValueError."""
        path = store_path() if path is None else Path(path)
        if not path.exists():
            return cls(path=path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise ValueError(f"Cannot read audio formats file {path}: {e}") from e
        if not isinstance(data, dict) or not isinstance(data.get("profiles", {}), dict):
            raise ValueError(
                f"Audio formats file {path} must be an object with a 'profiles' object"
            )

        custom = {}
        for name, entry in data.get("profiles", {}).items():
            if name in BUILTIN_FORMATS:
                raise ValueError(f"{path}: profile '{name}' reuses a built-in format name")
            if not isinstance(entry, dict):
                raise ValueError(f"{path}: profile '{name}' must be an object")
            try:
                custom[name] = AudioFormat.from_dict({**entry, "name": name})
            except TypeError as e:
                raise ValueError(f"{path}: profile '{name}': {e}") from e
        default_name = data.get("default") or DEFAULT_FORMAT_NAME
        return cls(path=path, default_name=str(default_name), custom=custom)

    def formats(self) -> dict[str, AudioFormat]:
        """Every selectable format, built-ins first."""
        return {**BUILTIN_FORMATS, **self.custom}

    def names(self) -> list[str]:
        return list(self.formats())

    def get(self, name: str) -> AudioFormat:
        formats = self.formats()
        if name not in formats:
            raise ValueError(
                f"Unknown audio format '{name}'. Available: {', '.join(formats)}"
            )
        return formats[name]

    def default(self) -> AudioFormat:
        return self.formats().get(self.default_name, WAV_FORMAT)

    @staticmethod
    def is_builtin(name: str) -> bool:
        return name in BUILTIN_FORMATS

    def put(self, fmt: AudioFormat) -> None:
        """Add or replace a user profile (in memory - call save() to persist)."""
        if self.is_builtin(fmt.name):
            raise ValueError(f"'{fmt.name}' is a built-in format and cannot be overwritten")
        self.custom[fmt.name] = fmt

    def remove(self, name: str) -> None:
        if self.is_builtin(name):
            raise ValueError(f"'{name}' is a built-in format and cannot be deleted")
        self.custom.pop(name, None)
        if self.default_name == name:
            self.default_name = DEFAULT_FORMAT_NAME

    def set_default(self, name: str) -> None:
        self.get(name)
        self.default_name = name

    def save(self) -> None:
        write_json_atomic(
            self.path,
            {
                "default": self.default_name,
                "profiles": {
                    name: fmt.to_dict(include_name=False) for name, fmt in self.custom.items()
                },
            },
        )


def resolve_audio_format(
    value: "AudioFormat | str | None", keep_wav: bool | None = None
) -> AudioFormat:
    """Turn a format name (or None = the store's default) into an AudioFormat.

    Built-in names resolve without reading the store, so a broken
    audio_formats.json never blocks a run that doesn't need it.

    Args:
        value: AudioFormat, profile name, or None for the store's default
        keep_wav: Per-run override of the profile's keep_wav
    """
    if isinstance(value, AudioFormat):
        fmt = value
    elif value in BUILTIN_FORMATS:
        fmt = BUILTIN_FORMATS[value]
    elif value is None:
        fmt = AudioFormatStore.load().default()
    else:
        fmt = AudioFormatStore.load().get(value)
    if keep_wav is not None and keep_wav != fmt.keep_wav:
        fmt = dataclasses.replace(fmt, keep_wav=keep_wav)
    return fmt
# [SEMANTIC-END] CONFIG:AUDIO_FORMATS


# [SEMANTIC-BEGIN] ARTIFACTS:JOB_AUDIO
# @purpose: Locate a job's audio files: the working WAV and the archive copy
# @description: find_job_audio() prefers the working WAV while it exists and falls back to
#   audio/archive.<ext> once retention (keep_wav=false) removed it, so transcribe/diarize re-runs
#   decode whichever copy the job kept. Used by the GUI job scanner too, so it only stats files.
# @inputs: job_dir
# @outputs: Path to the audio to decode, or None when the job has no audio yet
# @sideEffects: Filesystem stat/glob only
# @errors: None
# @see: STAGE:CONVERT, STAGE:TRANSCRIBE, STAGE:DIARIZE, CONFIG:AUDIO_FORMATS
def archive_path(job_dir: str | Path, extension: str, suffix: str = "") -> Path:
    """audio/archive<suffix>.<extension> inside job_dir (suffix e.g. "_left")."""
    return Path(job_dir) / "audio" / f"{ARCHIVE_STEM}{suffix}.{extension}"


def find_job_audio(job_dir: str | Path) -> Path | None:
    """The audio downstream stages should decode, or None if the job has none yet."""
    wav = Path(job_dir) / WORKING_WAV
    if wav.exists():
        return wav
    archives = sorted(
        Path(job_dir).glob(f"audio/{ARCHIVE_STEM}.*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return archives[0] if archives else None
# [SEMANTIC-END] ARTIFACTS:JOB_AUDIO
