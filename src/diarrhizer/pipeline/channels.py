"""Small shared helpers for recordings with the local voice on one stereo channel."""

from pathlib import Path

from diarrhizer.audio_formats import find_job_audio

TRACK_DEFAULTS = {"channel_source": "stereo", "my_track": 1, "others_track": 2, "mix_track": 0}
CHANNEL_DEFAULTS = {"channel_mode": "mixed", "my_channel": "left", "one_on_one": False, **TRACK_DEFAULTS}


# [SEMANTIC-BEGIN] AUDIO:CHANNEL_MODE
# @purpose: Keep channel identity and timestamps consistent across pipeline stages
# @description: Channel mode is explicit; stereo alone never implies separate speakers.
#   Left/right audio is resolved independently, with no fallback to the mixed recording.
#   Combined ASR retains both channel originals, per-channel provenance and remapped flat word ownership.
#   Track mode uses one-based audio-stream ordinals. L/R artifact names remain internal slots;
#   source_track identifies the actual stream. Mix track 0 builds playback from the selected voices.
# @inputs: pipeline config, job directory, optional recorded metadata
# @outputs: validated settings, channel paths, default names and combined ASR view
# @sideEffects: Filesystem stat/glob through find_job_audio
# @errors: ValueError for invalid channel options; FileNotFoundError for missing channel audio
# @see: CONFIG:PIPELINE, STAGE:TRANSCRIBE, STAGE:DIARIZE, STAGE:MERGE
def channel_settings(config) -> dict:
    return {key: getattr(config, key, default) for key, default in CHANNEL_DEFAULTS.items()}


def track_settings(config) -> dict:
    return {key: getattr(config, key, default) for key, default in TRACK_DEFAULTS.items()}


def channel_tracks(config) -> dict:
    return {config.my_channel: config.my_track,
            "right" if config.my_channel == "left" else "left": config.others_track}


def default_channel_speakers(config) -> dict:
    if config.channel_mode != "me-others":
        return {}
    return {"Speaker_00": "Я", **({"Speaker_01": "Собеседник"} if config.one_on_one else {})}


def validate_channel_settings(channel_mode: str, my_channel: str, one_on_one: bool,
                              channel_source: str = "stereo", my_track: int = 1,
                              others_track: int = 2, mix_track: int = 0, *, save_mix: bool = True) -> None:
    if channel_mode not in ("mixed", "me-others"):
        raise ValueError("channel_mode must be mixed or me-others")
    if my_channel not in ("left", "right"):
        raise ValueError("my_channel must be left or right")
    if type(one_on_one) is not bool:
        raise ValueError("one_on_one must be boolean")
    if one_on_one and channel_mode != "me-others":
        raise ValueError("one_on_one requires channel_mode=me-others")
    if type(save_mix) is not bool or (not save_mix and channel_mode != "me-others"):
        raise ValueError("save_mix must be boolean and can only be disabled in me-others mode")
    if channel_source not in ("stereo", "tracks"):
        raise ValueError("channel_source must be stereo or tracks")
    for key, value, minimum in (("my_track", my_track, 1), ("others_track", others_track, 1),
                                 ("mix_track", mix_track, 0)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    if channel_source == "tracks":
        if channel_mode != "me-others":
            raise ValueError("Separate audio tracks require channel_mode=me-others")
        if my_track == others_track:
            raise ValueError("Your voice and others must use different audio tracks")


def channel_audio(job_dir: Path, channel: str) -> Path:
    audio = find_job_audio(job_dir, suffix=f"_{channel}")
    if audio is None:
        raise FileNotFoundError(f"Missing {channel} channel audio; rerun the convert stage")
    return audio


def require_channel_transcript(data: dict) -> dict:
    channels = data.get("channels")
    if (not isinstance(channels, dict) or set(channels) != {"left", "right"}
            or not all(isinstance(result, dict) for result in channels.values())):
        raise ValueError("Channel transcripts are missing; rerun the transcribe stage")
    if (data.get("metadata") or {}).get("channel_mode") != "me-others":
        raise ValueError("Transcript mode differs from this run; rerun the transcribe stage")
    return channels


def combine_channel_transcripts(channels: dict) -> dict:
    """Build a chronological ASR view; channel originals retain local word ownership."""
    indexed = [(channel, index, segment) for channel, result in channels.items()
               for index, segment in enumerate(result.get("segments", []))]
    indexed.sort(key=lambda item: item[2]["start"])
    positions = {(channel, index): pos for pos, (channel, index, _) in enumerate(indexed)}
    segments = [{**segment, "channel": channel,
                 **({"source_track": channels[channel]["source_track"]} if "source_track" in channels[channel] else {})}
                for channel, _, segment in indexed]
    words = []
    for channel, result in channels.items():
        for word in result.get("words", []):
            copied = {**word, "channel": channel}
            if "source_track" in result:
                copied["source_track"] = result["source_track"]
            if "segment_index" in word:
                copied["segment_index"] = positions.get((channel, word["segment_index"]))
            words.append(copied)
    languages = list(dict.fromkeys(result.get("language", "unknown") for result in channels.values()))
    return {
        "channels": channels, "segments": segments, "words": words,
        "text": " ".join(seg.get("text", "").strip() for seg in segments).strip(),
        "language": languages[0] if len(languages) == 1 else "multiple",
        "effective_settings": {channel: result.get("effective_settings", {}) for channel, result in channels.items()},
        "alignment_status": "per-channel",
        "warnings": [f"{channel}: {warning}" for channel, result in channels.items()
                     for warning in result.get("warnings", [])],
    }
# [SEMANTIC-END] AUDIO:CHANNEL_MODE
