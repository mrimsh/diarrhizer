"""Real FFmpeg smoke with synthetic media; no models, accounts or recordings."""

import math
import shutil
import struct
import wave
from pathlib import Path

import pytest

from diarrhizer.pipeline.runner import run_pipeline
from diarrhizer.pipeline.stages.convert import ConvertStage
from diarrhizer.pipeline.stages.export import ExportStage
from diarrhizer.pipeline.stages.merge import MergeStage
from diarrhizer.projects import export_path, media_path, move_jobs
from diarrhizer.summary.service import read_transcript
from diarrhizer.utils import write_json_atomic


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg is not on PATH")
def test_real_ffmpeg_project_smoke(tmp_path):
    source = tmp_path / "synthetic.wav"
    with wave.open(str(source), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"".join(struct.pack("<h", int(1000 * math.sin(2 * math.pi * 440 * i / 16000)))
                                   for i in range(16000)))
    result = run_pipeline(input_path=source, out_dir=tmp_path / "out", stages=[ConvertStage()],
                          project="Smoke", team="Coders", audio_format="flac",
                          recorded_at="2021-02-03T04:05:06", naming_template="PROJECT-DD.MM.YYYY_HH-mm-SS-TEAM",
                          storage={"original": "template", "working": "discard", "archive": "template"})
    job = Path(result["job_dir"])
    assert media_path(job, "original", "wav").read_bytes() == source.read_bytes()
    assert media_path(job, "archive", "flac").read_bytes().startswith(b"fLaC")
    assert not media_path(job, "working", "wav").exists()
    write_json_atomic(job / "asr" / "transcript.json", {
        "segments": [{"start": 0, "end": 1, "text": "Synthetic smoke test"}], "words": [],
    })
    run_pipeline(job_dir=job, out_dir=tmp_path / "out", stages=[MergeStage(), ExportStage()],
                 speakers={"Speaker_00": "Alice"})
    assert export_path(job, "json").is_file()
    assert "Synthetic smoke test" in read_transcript(job)
    moved = move_jobs(tmp_path / "out", [job], project="Moved", team="Designers", rename=True,
                      recorded_at="2022-03-04T05:06:07", naming_template="PROJECT-DD.MM.YY_HH-mm-SS-TEAM")[job]
    assert "04.03.22_05-06-07" in moved.name
    assert media_path(moved, "archive", "flac").read_bytes().startswith(b"fLaC")
    assert media_path(moved, "original", "wav").read_bytes() == source.read_bytes()
    assert "Synthetic smoke test" in read_transcript(moved)
    assert export_path(moved, "json").is_file()
