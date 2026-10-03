"""Tests for diarrhizer.export.turns (EXPORT:SPEAKER_TURNS).

Merged segments are one Whisper phrase each; the readable exports join
consecutive phrases of the same person into one turn.
"""

from diarrhizer.export.markdown_export import export_to_markdown
from diarrhizer.export.text_export import export_to_text
from diarrhizer.export.turns import (
    TURN_GAP_SECONDS,
    TURN_SOFT_LIMIT,
    format_timestamp,
    format_turn_line,
    group_turns,
    speaking_time,
)
from diarrhizer.pipeline.runner import PipelineConfig


def _seg(start, end, speaker, text):
    return {"start": start, "end": end, "speaker_id": speaker, "text": text}


def _config(**overrides) -> PipelineConfig:
    return PipelineConfig(job_id="job", input_file="input.wav", language="ru", device="cpu", **overrides)


def test_consecutive_segments_of_one_speaker_become_one_turn():
    turns = group_turns([
        _seg(0, 2, "Speaker_00", "Привет."),
        _seg(2.5, 4, "Speaker_00", "Как дела?"),
        _seg(4, 5, "Speaker_01", "Хорошо."),
    ])
    assert [(t["speaker_name"], t["text"], t["start"], t["end"]) for t in turns] == [
        ("Speaker_00", "Привет. Как дела?", 0.0, 4.0),
        ("Speaker_01", "Хорошо.", 4.0, 5.0),
    ]


def test_alternating_speakers_stay_separate():
    turns = group_turns([
        _seg(0, 1, "Speaker_00", "a"),
        _seg(1, 2, "Speaker_01", "b"),
        _seg(2, 3, "Speaker_00", "c"),
    ])
    assert [t["text"] for t in turns] == ["a", "b", "c"]


def test_long_pause_starts_a_new_turn():
    turns = group_turns([
        _seg(0, 1, "Speaker_00", "a"),
        _seg(1 + TURN_GAP_SECONDS, 5, "Speaker_00", "b"),
    ])
    assert [t["text"] for t in turns] == ["a", "b"]


def test_soft_limit_closes_a_turn_at_a_sentence_end():
    long_sentence = "слово " * (TURN_SOFT_LIMIT // 6) + "конец."
    turns = group_turns([
        _seg(0, 10, "Speaker_00", long_sentence),
        _seg(10, 11, "Speaker_00", "Новая мысль."),
    ])
    assert len(turns) == 2


def test_soft_limit_waits_for_the_sentence_to_end():
    long_text = "слово " * (TURN_SOFT_LIMIT // 6) + "и"
    turns = group_turns([
        _seg(0, 10, "Speaker_00", long_text),
        _seg(10, 11, "Speaker_00", "продолжение."),
    ])
    assert len(turns) == 1


def test_two_ids_mapped_to_one_name_merge():
    """Diarization split one person in two; naming both the same joins them."""
    turns = group_turns(
        [_seg(0, 1, "Speaker_00", "a"), _seg(1, 2, "Speaker_02", "b")],
        {"Speaker_00": "Иван", "Speaker_02": "Иван"},
    )
    assert len(turns) == 1
    assert turns[0]["speaker_name"] == "Иван"
    assert turns[0]["speaker_ids"] == ["Speaker_00", "Speaker_02"]


def test_pyannote_ids_are_normalized():
    turns = group_turns([_seg(0, 1, "SPEAKER_01", "a")])
    assert turns[0]["speaker_id"] == "Speaker_01"


def test_missing_fields_do_not_raise():
    assert group_turns([{}]) == [
        {
            "start": 0.0,
            "end": 0.0,
            "speaker_id": "Speaker_00",
            "speaker_ids": ["Speaker_00"],
            "speaker_name": "Speaker_00",
            "text": "",
        }
    ]


def test_format_turn_line():
    turn = group_turns([_seg(61, 125, "Speaker_00", "текст")], {"Speaker_00": "Мария"})[0]
    assert format_turn_line(turn) == "[00:01:01 → 00:02:05] Мария: текст"


def test_format_timestamp_handles_none():
    assert format_timestamp(None) == "00:00:00"


def test_speaking_time_per_name_in_first_appearance_order():
    segments = [
        _seg(0, 10, "Speaker_01", "a"),
        _seg(10, 12, "Speaker_00", "b"),
        _seg(12, 20, "Speaker_01", "c"),
    ]
    assert speaking_time(segments) == {"Speaker_01": 18.0, "Speaker_00": 2.0}
    assert speaking_time(segments, {"Speaker_00": "Иван", "Speaker_01": "Иван"}) == {"Иван": 20.0}


def test_text_export_joins_a_speakers_phrases():
    segments = [
        _seg(0, 2, "Speaker_00", "Привет."),
        _seg(2, 4, "Speaker_00", "Как дела?"),
        _seg(4, 5, "Speaker_01", "Хорошо."),
    ]
    body = [line for line in export_to_text(segments, _config(), "in.wav").splitlines() if line.startswith("[")]
    assert body == [
        "[00:00:00 → 00:00:04] Speaker_00: Привет. Как дела?",
        "[00:00:04 → 00:00:05] Speaker_01: Хорошо.",
    ]


def test_markdown_lists_speaking_time_largest_first():
    segments = [
        _seg(0, 30, "Speaker_00", "a"),
        _seg(30, 120, "Speaker_01", "b"),
    ]
    md = export_to_markdown(segments, _config(speakers={"Speaker_01": "Мария"}), "in.wav")
    assert "**Speakers:**" in md
    lines = md.splitlines()
    first = lines.index("- Мария — 00:01:30 (75%)")
    second = lines.index("- Speaker_00 — 00:00:30 (25%)")
    assert first < second


def test_markdown_without_segments_has_no_speaker_summary():
    md = export_to_markdown([], _config(), "in.wav")
    assert "**Speakers:**" not in md
