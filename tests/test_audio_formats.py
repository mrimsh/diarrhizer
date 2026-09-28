"""Tests for diarrhizer.audio_formats: format validation, ffmpeg argv, the
JSON profile store, name resolution, and locating a job's audio.
"""

import json

import pytest

from diarrhizer import audio_formats
from diarrhizer.audio_formats import (
    BUILTIN_FORMATS,
    ENV_AUDIO_FORMATS_FILE,
    WAV_FORMAT,
    AudioFormat,
    AudioFormatStore,
    archive_path,
    find_job_audio,
    resolve_audio_format,
)


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    """Never read or write the developer's real <repo>/audio_formats.json."""
    path = tmp_path / "audio_formats.json"
    monkeypatch.setenv(ENV_AUDIO_FORMATS_FILE, str(path))
    return path


# --- AudioFormat ------------------------------------------------------------


def test_mp3_q5_preset_matches_the_classic_lame_command_line():
    """-vn -c:a libmp3lame -q:a 5, at the pipeline's 16 kHz mono."""
    assert BUILTIN_FORMATS["mp3-q5"].encoder_args() == [
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "libmp3lame", "-q:a", "5",
    ]


def test_bitrate_and_extra_args_are_passed_through():
    fmt = AudioFormat(
        name="opus-hq", codec="libopus", extension="opus", bitrate="48k",
        sample_rate=48000, channels=2, extra_args=["-application", "voip"],
    )
    assert fmt.encoder_args() == [
        "-vn", "-ac", "2", "-ar", "48000", "-c:a", "libopus", "-b:a", "48k",
        "-application", "voip",
    ]
    assert fmt.encoder_args(channels=1)[1:3] == ["-ac", "1"]


def test_flac_preset_pins_16_bit_samples():
    """Without -sample_fmt s16 ffmpeg writes 24-bit FLAC from a float decode,
    which came out larger than the WAV it was meant to shrink.
    """
    args = BUILTIN_FORMATS["flac"].encoder_args()
    assert args[args.index("-sample_fmt") + 1] == "s16"


def test_every_encoder_argv_drops_video():
    for fmt in BUILTIN_FORMATS.values():
        assert fmt.encoder_args()[0] == "-vn"


@pytest.mark.parametrize(
    "fields, message",
    [
        ({"name": "has space"}, "no spaces"),
        ({"name": "x", "quality": 5, "bitrate": "32k"}, "not both"),
        ({"name": "x", "bitrate": "fast"}, "bitrate"),
        ({"name": "x", "quality": "5"}, "quality"),
        ({"name": "x", "channels": 0}, "channels"),
        ({"name": "x", "sample_rate": 16000.0}, "sample_rate"),
        ({"name": "x", "extension": "m p3"}, "extension"),
        ({"name": "x", "keep_wav": "no"}, "keep_wav"),
    ],
)
def test_invalid_fields_are_rejected(fields, message):
    with pytest.raises(ValueError, match=message):
        AudioFormat(**fields)


def test_extension_is_normalized():
    assert AudioFormat(name="x", codec="flac", extension=".FLAC").extension == "flac"


def test_only_a_non_wav_encoding_writes_an_archive():
    assert WAV_FORMAT.writes_archive is False
    assert BUILTIN_FORMATS["mp3-q5"].writes_archive is True
    # Still PCM WAV, but a different rate - that is a real archive, not a duplicate.
    assert AudioFormat(name="wav8k", sample_rate=8000).writes_archive is True


def test_wav_only_format_always_keeps_the_wav():
    assert AudioFormat(name="w", keep_wav=False).keeps_wav is True
    assert BUILTIN_FORMATS["mp3-q5"].keeps_wav is False


def test_cache_key_ignores_name_and_description():
    a = AudioFormat(name="a", description="one", codec="libmp3lame", extension="mp3", quality=5)
    b = AudioFormat(name="b", description="two", codec="libmp3lame", extension="mp3", quality=5)
    assert a.cache_key() == b.cache_key()


def test_size_estimate_only_where_it_is_deterministic():
    assert WAV_FORMAT.estimated_mb_per_hour() == pytest.approx(115.2)
    assert BUILTIN_FORMATS["opus-24k"].estimated_mb_per_hour() == pytest.approx(10.8)
    assert BUILTIN_FORMATS["mp3-q5"].estimated_mb_per_hour() is None  # VBR
    assert BUILTIN_FORMATS["flac"].estimated_mb_per_hour() is None  # content-dependent


def test_dict_round_trip():
    fmt = BUILTIN_FORMATS["flac"]
    assert AudioFormat.from_dict(fmt.to_dict()) == fmt


def test_from_dict_rejects_unknown_fields():
    with pytest.raises(ValueError, match="quallity"):
        AudioFormat.from_dict({"name": "x", "quallity": 5})


# --- AudioFormatStore -------------------------------------------------------


def test_missing_file_is_an_empty_store_defaulting_to_wav(isolated_store):
    store = AudioFormatStore.load()
    assert store.path == isolated_store
    assert store.default() == WAV_FORMAT
    assert store.names() == list(BUILTIN_FORMATS)


def test_profiles_and_default_round_trip(isolated_store):
    store = AudioFormatStore.load()
    mine = AudioFormat(name="архив", description="мой", codec="libmp3lame", extension="mp3", bitrate="32k")
    store.put(mine)
    store.set_default("архив")
    store.save()

    reloaded = AudioFormatStore.load()
    assert reloaded.get("архив") == mine
    assert reloaded.default() == mine
    on_disk = json.loads(isolated_store.read_text(encoding="utf-8"))
    assert on_disk["default"] == "архив"
    assert "name" not in on_disk["profiles"]["архив"]


def test_builtins_cannot_be_overwritten_or_deleted():
    store = AudioFormatStore.load()
    with pytest.raises(ValueError, match="built-in"):
        store.put(AudioFormat(name="flac"))
    with pytest.raises(ValueError, match="built-in"):
        store.remove("wav")


def test_removing_the_default_profile_falls_back_to_wav():
    store = AudioFormatStore.load()
    store.put(AudioFormat(name="mine", codec="flac", extension="flac"))
    store.set_default("mine")
    store.remove("mine")
    assert store.default_name == "wav"


def test_default_pointing_at_a_missing_profile_falls_back_to_wav(isolated_store):
    isolated_store.write_text(json.dumps({"default": "gone", "profiles": {}}), encoding="utf-8")
    assert AudioFormatStore.load().default() == WAV_FORMAT


def test_set_default_rejects_unknown_names():
    with pytest.raises(ValueError, match="Unknown audio format 'nope'"):
        AudioFormatStore.load().set_default("nope")


@pytest.mark.parametrize(
    "content, message",
    [
        ("{not json", "Cannot read"),
        ("[]", "must be an object"),
        (json.dumps({"profiles": {"mp3-q5": {}}}), "built-in"),
        (json.dumps({"profiles": {"x": {"channels": 99}}}), "channels"),
        (json.dumps({"profiles": {"x": {"bitrat": "32k"}}}), "bitrat"),
    ],
)
def test_malformed_store_fails_loudly(isolated_store, content, message):
    isolated_store.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        AudioFormatStore.load()


# --- resolve_audio_format ---------------------------------------------------


def test_builtin_names_resolve_without_reading_a_broken_store(isolated_store):
    isolated_store.write_text("{broken", encoding="utf-8")
    assert resolve_audio_format("mp3-q5") == BUILTIN_FORMATS["mp3-q5"]


def test_none_resolves_to_the_store_default(isolated_store):
    isolated_store.write_text(json.dumps({"default": "opus-24k"}), encoding="utf-8")
    assert resolve_audio_format(None) == BUILTIN_FORMATS["opus-24k"]


def test_custom_names_resolve_from_the_store():
    store = AudioFormatStore.load()
    store.put(AudioFormat(name="tiny", codec="libopus", extension="opus", bitrate="12k"))
    store.save()
    assert resolve_audio_format("tiny").bitrate == "12k"


def test_unknown_name_lists_the_available_ones():
    with pytest.raises(ValueError, match="Available: wav, flac, mp3-q5, opus-24k"):
        resolve_audio_format("mp3-q9")


def test_keep_wav_override():
    assert resolve_audio_format("mp3-q5", keep_wav=True).keep_wav is True
    assert resolve_audio_format("mp3-q5", keep_wav=None).keep_wav is False
    fmt = BUILTIN_FORMATS["flac"]
    assert resolve_audio_format(fmt) is fmt


# --- find_job_audio ---------------------------------------------------------


def test_find_job_audio_prefers_the_working_wav(tmp_path):
    (tmp_path / "audio").mkdir()
    (tmp_path / "audio" / "normalized.wav").write_bytes(b"wav")
    archive_path(tmp_path, "mp3").write_bytes(b"mp3")
    assert find_job_audio(tmp_path) == tmp_path / "audio" / "normalized.wav"


def test_find_job_audio_falls_back_to_the_archive(tmp_path):
    (tmp_path / "audio").mkdir()
    archive_path(tmp_path, "mp3").write_bytes(b"mp3")
    archive_path(tmp_path, "mp3", "_left").write_bytes(b"left")  # never picked
    assert find_job_audio(tmp_path) == tmp_path / "audio" / "archive.mp3"


def test_find_job_audio_is_none_without_audio(tmp_path):
    assert find_job_audio(tmp_path) is None


def test_default_store_lives_at_the_repo_root(monkeypatch):
    monkeypatch.delenv(ENV_AUDIO_FORMATS_FILE)
    assert audio_formats.store_path() == audio_formats.REPO_ROOT / "audio_formats.json"


def test_example_profiles_file_is_a_valid_store():
    """audio_formats.example.json is what the README tells people to copy."""
    store = AudioFormatStore.load(audio_formats.REPO_ROOT / "audio_formats.example.json")

    assert store.default().name == "call-archive"
    assert store.get("call-archive").encoder_args() == [
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "libmp3lame", "-q:a", "5",
    ]
