"""Independent-channel processing, overlap, retention and model-free one-on-one calls."""

import json
import math
import struct
import subprocess
import wave
from pathlib import Path

import pytest

from diarrhizer.adapters.ffmpeg import FFmpegAdapter, resolve_ffmpeg_path
from diarrhizer.audio_formats import ENV_AUDIO_FORMATS_FILE, find_job_audio
from diarrhizer.pipeline.runner import run_pipeline
from diarrhizer.pipeline.settings import read_job_settings
from diarrhizer_gui.job_scan import job_mode, rediarize_plan, resume_plan, restart_plan, stage_done


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


class ASR:
    calls = []

    def __init__(self, **kwargs):
        pass

    def transcribe(self, audio_path, language=None):
        self.calls.append(Path(audio_path))
        left = "_left" in Path(audio_path).stem
        segments = ([{"start": 2., "end": 4., "text": "Мой ответ"}] if left else
                    [{"start": 0., "end": 3., "text": "Первый голос"},
                     {"start": 4., "end": 5., "text": "Второй голос"}])
        words = []
        for i, seg in enumerate(segments):
            middle = (seg["start"] + seg["end"]) / 2
            for text, start, end in zip(seg["text"].split(), [seg["start"], middle], [middle, seg["end"]]):
                words.append({"word": text, "start": start, "end": end, "segment_index": i})
        return {"segments": segments, "words": words, "language": "ru", "alignment_status": "aligned"}

    def unload(self):
        pass


class Diar:
    calls = []

    def __init__(self, **kwargs):
        self.options = kwargs

    def diarize(self, audio_path):
        self.calls.append((Path(audio_path), self.options))
        return {"segments": [{"start": 0., "end": 3., "speaker": "Speaker_00"},
                             {"start": 3., "end": 6., "speaker": "Speaker_02"}],
                "num_speakers": 2, "speakers": ["Speaker_00", "Speaker_02"]}

    def unload(self):
        pass


@pytest.fixture
def adapters(tmp_path, monkeypatch):
    ASR.calls = []
    Diar.calls = []
    monkeypatch.setenv(ENV_AUDIO_FORMATS_FILE, str(tmp_path / "formats.json"))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr("diarrhizer.pipeline.stages.transcribe.WhisperXAdapter", ASR)
    monkeypatch.setattr("diarrhizer.pipeline.stages.diarize.WhisperXDiarizeAdapter", Diar)


@pytest.fixture
def fake_media(adapters, monkeypatch):
    def convert(self, input_path, output_path, audio_profile="raw"):
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"mix")
        if audio_profile == "split-stereo":
            for channel in ("left", "right"):
                output.with_stem(output.stem + "_" + channel).write_bytes(channel.encode())

    def encode(self, input_path, output_path, encoder_args, audio_profile="raw", channel=None):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(str(channel).encode())

    monkeypatch.setattr(FFmpegAdapter, "convert_to_wav", convert)
    monkeypatch.setattr(FFmpegAdapter, "encode", encode)
    monkeypatch.setattr(FFmpegAdapter, "validate_split_input", lambda *_: None)
    monkeypatch.setattr("diarrhizer.adapters.ffmpeg.resolve_ffmpeg_path", lambda explicit=None: "ffmpeg")


def start(tmp_path, **kwargs):
    source = tmp_path / "call.wav"
    source.write_bytes(b"synthetic input")
    return run_pipeline(input_path=source, out_dir=tmp_path / "out", channel_mode="me-others",
                        device="cpu", **kwargs)


@pytest.mark.parametrize("my_channel", ["left", "right"])
def test_one_on_one_keeps_overlap_and_never_loads_diarization(tmp_path, fake_media, monkeypatch, my_channel):
    def forbidden(**kwargs):
        pytest.fail("One-on-one must not create a diarization adapter")
    monkeypatch.setattr("diarrhizer.pipeline.stages.diarize.WhisperXDiarizeAdapter", forbidden)
    result = start(tmp_path, my_channel=my_channel, one_on_one=True)
    job = Path(result["job_dir"])
    assert len(ASR.calls) == 2
    assert {p.stem for p in ASR.calls} == {"normalized_left", "normalized_right"}
    segments = read(job / "merged/segments.json")["segments"]
    assert [s["start"] for s in segments] == [0, 2, 4]
    assert segments[0]["end"] > segments[1]["start"]  # Both sides of the interruption survive.
    assert [s["text"] for s in segments] == ["Первый голос", "Мой ответ", "Второй голос"]
    for seg in segments:
        expected = "Speaker_00" if seg["channel"] == my_channel else "Speaker_01"
        assert seg["speaker_id"] == expected
        assert all(w["speaker_id"] == expected for w in seg["words"])
    exported = read(job / "export/result.json")
    assert exported["metadata"]["speakers"] == {"Speaker_00": "Я", "Speaker_01": "Собеседник"}
    assert read(job / "diar/diarization.json")["metadata"]["model_free"] is True
    assert read_job_settings(job)["my_channel"] == my_channel
    resumed = run_pipeline(job_dir=job, out_dir=tmp_path / "out")
    assert all(s["status"] == "cached" for s in resumed["stages"])
    assert len(ASR.calls) == 2
    assert job_mode(read(job / "meta/run.json")) == "me_others"
    assert resume_plan(job) is None


@pytest.mark.parametrize("my_channel", ["left", "right"])
def test_group_diarizes_only_others_and_offsets_their_ids(tmp_path, fake_media, monkeypatch, my_channel):
    monkeypatch.setenv("HF_TOKEN", "test-only")
    result = start(tmp_path, my_channel=my_channel, min_speakers=2, max_speakers=2)
    job = Path(result["job_dir"])
    assert len(Diar.calls) == 1
    source, options = Diar.calls[0]
    assert ("_right" if my_channel == "left" else "_left") in source.stem
    assert options["min_speakers"] == options["max_speakers"] == 2
    segments = read(job / "merged/segments.json")["segments"]
    assert {s["speaker_id"] for s in segments if s["channel"] != my_channel} == {"Speaker_01", "Speaker_02"}
    assert {s["speaker_id"] for s in segments if s["channel"] == my_channel} == {"Speaker_00"}
    rerun = run_pipeline(job_dir=job, out_dir=tmp_path / "out")
    assert all(s["status"] == "cached" for s in rerun["stages"])


def test_one_on_one_interleaves_replies_inside_long_asr_chunks(tmp_path, fake_media, monkeypatch):
    def transcribe(self, audio_path, language=None):
        entries = ([('Привет!', 0.2, 0.5), ('Отлично.', 2.2, 2.6)] if '_left' in Path(audio_path).stem
                   else [('Как дела?', 1.0, 1.6), ('Пока!', 3.1, 3.4)])
        words = []
        for text, start_time, end_time in entries:
            tokens = text.split()
            step = (end_time - start_time) / len(tokens)
            words.extend(dict(word=token, start=start_time + index * step,
                              end=start_time + (index + 1) * step, segment_index=0)
                         for index, token in enumerate(tokens))
        return {"segments": [{"start": 0, "end": 5, "text": " ".join(entry[0] for entry in entries)}],
                "words": words, "language": "ru"}

    monkeypatch.setattr(ASR, 'transcribe', transcribe)
    job = Path(start(tmp_path, one_on_one=True)["job_dir"])
    segments = read(job / 'export/result.json')['segments']
    assert [(seg['start'], seg['speaker_id'], seg['text']) for seg in segments] == [
        (0.2, 'Speaker_00', 'Привет!'), (1.0, 'Speaker_01', 'Как дела?'),
        (2.2, 'Speaker_00', 'Отлично.'), (3.1, 'Speaker_01', 'Пока!'),
    ]
    assert not Diar.calls
    (job / 'speakers.json').write_text(json.dumps({'Speaker_00': 'Мария', 'Speaker_01': 'Иван'}), encoding='utf-8')
    _, kwargs = restart_plan(job, 'diarize')
    run_pipeline(**kwargs)
    assert read(job / 'export/result.json')['metadata']['speakers'] == {'Speaker_00': 'Мария', 'Speaker_01': 'Иван'}
    assert not (job / 'speakers.before-rediarize.json').exists()
    assert not Diar.calls


@pytest.mark.parametrize("portable", [False, True])
def test_archives_resume_each_channel_after_working_wav_cleanup(tmp_path, fake_media, portable):
    options = {"project": "Calls", "storage": {"working": "discard", "archive": "template"}} if portable else {}
    result = start(tmp_path, one_on_one=True, audio_format="flac", keep_wav=False, **options)
    job = Path(result["job_dir"])
    assert not list((job / "audio").rglob("*.wav"))
    sources = [find_job_audio(job, "_left"), find_job_audio(job, "_right")]
    assert all(p.suffix == ".flac" for p in sources)
    result = run_pipeline(job_dir=job, out_dir=tmp_path / "out", from_stage="transcribe", force_stage="transcribe")
    assert ASR.calls[-2:] == sources
    assert result["stages"][-1]["status"] == "completed"


def test_channel_changes_require_conversion_and_cached_asr_survives_one_on_one_switch(tmp_path, fake_media, monkeypatch):
    result = start(tmp_path, one_on_one=True)
    job = Path(result["job_dir"])
    before = (job / "asr/transcript.json").read_bytes()
    with pytest.raises(ValueError, match="restarting from convert"):
        run_pipeline(job_dir=job, out_dir=tmp_path / "out", from_stage="diarize", my_channel="right")
    with pytest.raises(ValueError, match="restarting from diarize"):
        run_pipeline(job_dir=job, out_dir=tmp_path / "out", from_stage="export", one_on_one=False)
    monkeypatch.setenv("HF_TOKEN", "test-only")
    mode, kwargs = rediarize_plan(job, 2, 2)
    assert mode == "me_others" and kwargs["one_on_one"] is False
    run_pipeline(**kwargs)
    assert len(ASR.calls) == 2 and len(Diar.calls) == 1
    assert (job / "asr/transcript.json").read_bytes() == before
    assert read_job_settings(job)["one_on_one"] is False
    plan = restart_plan(job, "merge")
    assert plan[0] == "me_others" and plan[1]["my_channel"] == "left"


def test_selecting_the_other_local_channel_updates_recorded_layout(tmp_path, fake_media):
    first = start(tmp_path, one_on_one=True)
    job = Path(first["job_dir"])
    result = run_pipeline(job_dir=job, out_dir=tmp_path / "out", my_channel="right")
    assert len(ASR.calls) == 2  # Reuse independent channel transcripts, only assignments change.
    assert read_job_settings(job)["my_channel"] == "right"
    assert read(job / "merged/segments.json")["segments"][0]["speaker_id"] == "Speaker_00"
    assert result["stages"][-1]["status"] == "completed"


def test_group_without_token_fails_before_any_asr(tmp_path, fake_media):
    with pytest.raises(RuntimeError, match="HuggingFace token"):
        start(tmp_path, one_on_one=False)
    assert not ASR.calls and not Diar.calls


def test_missing_channel_never_falls_back_to_mix(tmp_path, fake_media):
    result = start(tmp_path, one_on_one=True)
    job = Path(result["job_dir"])
    (job / "audio/normalized_right.wav").unlink()
    with pytest.raises(FileNotFoundError, match="right channel audio"):
        run_pipeline(job_dir=job, out_dir=tmp_path / "out", from_stage="transcribe", force_stage="transcribe")


def test_full_resume_rebuilds_a_missing_channel(tmp_path, fake_media):
    result = start(tmp_path, one_on_one=True)
    job = Path(result["job_dir"])
    (job / "audio/normalized_right.wav").unlink()
    result = run_pipeline(job_dir=job, out_dir=tmp_path / "out")
    assert result["stages"][0]["status"] == "completed"
    assert (job / "audio/normalized_right.wav").exists()


def test_real_ffmpeg_preserves_stereo_channel_samples_and_rejects_mono(tmp_path, adapters):
    if not resolve_ffmpeg_path():
        pytest.skip("FFmpeg not installed")
    source = tmp_path / "stereo.wav"
    with wave.open(str(source), "wb") as wav:
        wav.setparams((2, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(b"".join(struct.pack("<hh", int(10000 * math.sin(i / 10)), 0)
                                 for i in range(16000)))
    adapter = FFmpegAdapter()
    adapter.validate_split_input(source)
    result = run_pipeline(input_path=source, out_dir=tmp_path / "out", channel_mode="me-others", one_on_one=True,
                          device="cpu", audio_profile="denoise-light")
    job = Path(result["job_dir"])
    for channel in ("left", "right"):
        with wave.open(str(find_job_audio(job, "_" + channel)), "rb") as wav:
            assert wav.getnchannels() == 1 and wav.getnframes() == 16000
            samples = struct.unpack("<" + "h" * 16000, wav.readframes(16000))
            assert (max(map(abs, samples)) == 0) == (channel == "right")
    mono = job / "audio/normalized_left.wav"
    with pytest.raises(ValueError, match="stereo"):
        adapter.validate_split_input(mono)
    with pytest.raises(ValueError, match="stereo"):
        run_pipeline(input_path=mono, out_dir=tmp_path / "rejected", channel_mode="me-others", one_on_one=True, device="cpu")
    assert not list((tmp_path / "rejected").glob("*/meta/run.json"))
    assert not list((tmp_path / "rejected").glob("*/asr/transcript.json"))


@pytest.fixture
def multitrack_media(tmp_path):
    ffmpeg = resolve_ffmpeg_path()
    if not ffmpeg:
        pytest.skip("FFmpeg not installed")
    source = tmp_path / "tracks.mkv"
    subprocess.run([
        ffmpeg, "-y", "-f", "lavfi", "-i", "color=c=black:s=16x16:r=1:d=2",
        "-f", "lavfi", "-i", "sine=frequency=300:sample_rate=16000:duration=2",
        "-itsoffset", "0.5", "-f", "lavfi", "-i", "sine=frequency=600:sample_rate=16000:duration=1.5",
        "-f", "lavfi", "-i", "sine=frequency=900:sample_rate=16000:duration=2",
        "-map", "0:v", "-map", "1:a", "-map", "2:a", "-map", "3:a",
        "-filter:a:0", "pan=stereo|c0=c0|c1=c0", "-c:v", "ffv1", "-c:a", "pcm_s16le", str(source),
    ], check=True, capture_output=True, timeout=30)
    return source


def samples(path):
    with wave.open(str(path), "rb") as wav:
        assert wav.getnchannels() == 1 and wav.getframerate() == 16000
        return struct.unpack('<' + 'h' * wav.getnframes(), wav.readframes(wav.getnframes()))


def amplitude(signal, frequency):
    return abs(sum(value * math.sin(2 * math.pi * frequency * i / 16000)
                   for i, value in enumerate(signal)))


@pytest.mark.parametrize("mix_track", [0, 3])
@pytest.mark.parametrize("my_channel", ["left", "right"])
def test_multitrack_extracts_selected_voices_and_preserves_delay(multitrack_media, tmp_path, adapters, mix_track, my_channel):
    result = run_pipeline(input_path=multitrack_media, out_dir=tmp_path / 'out', device='cpu',
                          channel_mode='me-others', channel_source='tracks', my_track=1, others_track=2,
                          mix_track=mix_track, my_channel=my_channel, one_on_one=True)
    job = Path(result['job_dir'])
    own = samples(find_job_audio(job, '_' + my_channel))
    other_channel = 'right' if my_channel == 'left' else 'left'
    remote = samples(find_job_audio(job, '_' + other_channel))
    assert amplitude(own, 300) > 20 * amplitude(own, 900)
    assert amplitude(remote, 600) > 20 * amplitude(remote, 900)
    first_sound = next(i for i, value in enumerate(remote) if abs(value) > 4)
    assert first_sound / 16000 == pytest.approx(.5, abs=.02)
    assert len(own) / 16000 == pytest.approx(2, abs=.02)
    assert len(remote) / 16000 == pytest.approx(2, abs=.02)
    playback = samples(find_job_audio(job))
    if mix_track:
        assert amplitude(playback, 900) > 20 * amplitude(playback, 300)
    else:
        assert amplitude(playback, 300) > 20 * amplitude(playback, 900)
        assert amplitude(playback, 600) > 20 * amplitude(playback, 900)
    assert len(ASR.calls) == 2 and not Diar.calls
    exported = read(job / 'export/result.json')
    for segment in exported['segments']:
        assert segment['source_track'] == (1 if segment['speaker_id'] == 'Speaker_00' else 2)
    assert {seg['source_track'] for seg in read(job / 'asr/transcript.json')['segments']} == {1, 2}
    assert read_job_settings(job)['mix_track'] == mix_track
    rerun = run_pipeline(job_dir=job, out_dir=tmp_path / 'out')
    assert all(stage['status'] == 'cached' for stage in rerun['stages'])


def test_track_archives_and_changed_selection(multitrack_media, tmp_path, adapters, monkeypatch):
    monkeypatch.setenv('HF_TOKEN', 'test-only')
    result = run_pipeline(input_path=multitrack_media, out_dir=tmp_path / 'out', device='cpu',
                          channel_mode='me-others', channel_source='tracks', mix_track=3,
                          audio_format='flac', keep_wav=False)
    job = Path(result['job_dir'])
    assert Diar.calls[0][0].stem == 'normalized_right'
    assert not list((job / 'audio').glob('*.wav'))
    assert find_job_audio(job, '_left').suffix == '.flac'
    run_pipeline(job_dir=job, out_dir=tmp_path / 'out', from_stage='transcribe', force_stage='transcribe')
    assert ASR.calls[-1].name == 'archive_right.flac'
    with pytest.raises(ValueError, match='restarting from convert'):
        run_pipeline(job_dir=job, out_dir=tmp_path / 'out', from_stage='transcribe', my_track=3)
    run_pipeline(job_dir=job, out_dir=tmp_path / 'out', my_track=2, others_track=1)
    assert read_job_settings(job)['my_track'] == 2
    segments = read(job / 'merged/segments.json')['segments']
    assert {s['source_track'] for s in segments if s['speaker_id'] == 'Speaker_00'} == {2}


def test_track_slot_change_rebuilds_wavs_instead_of_relabelling_old_audio(multitrack_media, tmp_path, adapters):
    result = run_pipeline(input_path=multitrack_media, out_dir=tmp_path / 'out', device='cpu',
                          channel_mode='me-others', channel_source='tracks', one_on_one=True, keep_wav=True)
    job = Path(result['job_dir'])
    left_wav = job / 'audio/normalized_left.wav'
    right_wav = job / 'audio/normalized_right.wav'
    original_left, original_right = samples(left_wav), samples(right_wav)
    assert amplitude(original_left, 300) > 20 * amplitude(original_left, 600)
    assert amplitude(original_right, 600) > 20 * amplitude(original_right, 300)

    result = run_pipeline(job_dir=job, out_dir=tmp_path / 'out', my_channel='right')
    assert [stage['status'] for stage in result['stages'][:2]] == ['completed', 'completed']
    assert samples(left_wav) == original_right
    assert samples(right_wav) == original_left
    assert read(job / 'asr/transcript.json')['channels']['right']['source_track'] == 1


@pytest.mark.parametrize("source_kind", ["stereo", "tracks"])
def test_only_two_flacs_remain_and_resume_without_mix(source_kind, request, tmp_path, adapters):
    if source_kind == "tracks":
        source = request.getfixturevalue("multitrack_media")
        options = {"channel_source": "tracks", "mix_track": 99}
    else:
        if not resolve_ffmpeg_path():
            pytest.skip("FFmpeg not installed")
        source = tmp_path / "stereo.wav"
        with wave.open(str(source), "wb") as wav:
            wav.setparams((2, 2, 16000, 0, "NONE", "not compressed"))
            wav.writeframes(struct.pack("<hh", 1000, -1000) * 16000)
        options = {"audio_profile": "split-stereo"}

    output = tmp_path / "out"
    first = run_pipeline(input_path=source, out_dir=output, device="cpu", channel_mode="me-others",
                         one_on_one=True, save_mix=False, audio_format="flac", to_stage="convert",
                         storage={"original": "discard", "working": "discard", "archive": "template"},
                         **options)
    job = Path(first["job_dir"])
    assert find_job_audio(job) is None
    assert stage_done(job)["convert"]
    _, resume = resume_plan(job)
    assert resume["from_stage"] == "transcribe" and resume["save_mix"] is False
    run_pipeline(**resume)

    voices = {find_job_audio(job, suffix) for suffix in ("_left", "_right")}
    assert all(path is not None and path.suffix == ".flac" for path in voices)
    assert {path for path in (job / "audio").rglob("*") if path.is_file()} == voices
    assert find_job_audio(job) is None
    assert not Diar.calls
    cached = run_pipeline(job_dir=job, out_dir=output)
    assert all(stage["status"] == "cached" for stage in cached["stages"])

    _, restart = restart_plan(job, "transcribe")
    run_pipeline(**restart)
    assert set(ASR.calls[-2:]) == voices


def test_conversion_restart_removes_previous_mix_and_keeps_voice_archives(tmp_path, fake_media):
    job = Path(start(tmp_path, one_on_one=True, audio_format="flac")["job_dir"])
    previous_mix = find_job_audio(job)
    assert previous_mix is not None

    _, restart = restart_plan(job, "convert", {"save_mix": False})
    run_pipeline(**restart)

    assert not previous_mix.exists()
    assert {path.name for path in (job / "audio").iterdir()} == {"archive_left.flac", "archive_right.flac"}


@pytest.mark.parametrize('options', [dict(my_track=0), dict(others_track=1), dict(my_track=4), dict(mix_track=4)])
def test_invalid_track_selection_fails_before_asr(multitrack_media, tmp_path, adapters, options):
    with pytest.raises(ValueError):
        run_pipeline(input_path=multitrack_media, out_dir=tmp_path / 'out', device='cpu',
                     channel_mode='me-others', channel_source='tracks', one_on_one=True, **options)
    assert not ASR.calls and not Diar.calls
