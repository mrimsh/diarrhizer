"""Speaker turns: consecutive same-speaker segments read as one reply."""

from typing import Any

from diarrhizer.export.plain_transcript import SENTENCE_END
from diarrhizer.export.speakers import normalize_speaker_id, resolve_speaker_name

# A pause at least this long starts a new turn even when the same person keeps
# talking, so a long silence still shows up as a fresh timestamp.
TURN_GAP_SECONDS = 3.0
# Past this many characters a turn is closed at the next sentence end, so a
# monologue - or a whole "recognition only" job, where everyone is Speaker_00 -
# does not collapse into one wall of text under a single timestamp.
TURN_SOFT_LIMIT = 600


def format_timestamp(seconds: float | None) -> str:
    """Format seconds as HH:MM:SS (fractions are truncated)."""
    seconds = seconds or 0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


# [SEMANTIC-BEGIN] EXPORT:SPEAKER_TURNS
# @purpose: Group merged segments into speaker turns for the human-readable exports
# @description: The merge stage emits one segment per Whisper phrase (split further at speaker
#   changes), so one reply is often spread over several lines with the same name. group_turns()
#   joins consecutive segments whose speakers resolve to the same display name - which also merges
#   two diarization ids mapped to one person - and starts a new turn after a pause of
#   TURN_GAP_SECONDS or, past TURN_SOFT_LIMIT characters, at the next sentence end.
#   speaking_time() sums segment durations per display name for speaker summaries.
#   result.json keeps the fine-grained segments; only md/txt and the GUI view use turns.
# @inputs: merged segments (start, end, speaker_id, text), optional speakers mapping
# @outputs: list of turn dicts (start, end, speaker_id, speaker_ids, speaker_name, text)
# @sideEffects: None (pure functions)
# @errors: None
# @see: STAGE:MERGE, EXPORT:MARKDOWN, EXPORT:TEXT, EXPORT:PLAIN_TRANSCRIPT
def group_turns(
    segments: list[dict[str, Any]], speakers: dict | None = None
) -> list[dict[str, Any]]:
    """Join consecutive same-speaker segments into turns, in input order."""
    turns: list[dict[str, Any]] = []
    length = 0
    for seg in segments:
        speaker_id = normalize_speaker_id(seg.get("speaker_id") or "Speaker_00")
        name = resolve_speaker_name(speaker_id, speakers)
        start = float(seg.get("start") or 0.0)
        end = float(seg.get("end") or start)
        text = (seg.get("text") or "").strip()

        last = turns[-1] if turns else None
        continues = (
            last is not None
            and last["speaker_name"] == name
            and start - last["end"] < TURN_GAP_SECONDS
            and not (length >= TURN_SOFT_LIMIT and last["text"].endswith(SENTENCE_END))
        )
        if continues:
            if text:
                last["text"] = f"{last['text']} {text}" if last["text"] else text
                length += len(text) + 1
            last["end"] = max(last["end"], end)
            if speaker_id not in last["speaker_ids"]:
                last["speaker_ids"].append(speaker_id)
            continue

        turns.append(
            {
                "start": start,
                "end": end,
                "speaker_id": speaker_id,
                "speaker_ids": [speaker_id],
                "speaker_name": name,
                "text": text,
            }
        )
        length = len(text)
    return turns


def format_turn_line(turn: dict[str, Any]) -> str:
    """One plain-text line: "[HH:MM:SS → HH:MM:SS] Name: text"."""
    start = format_timestamp(turn.get("start"))
    end = format_timestamp(turn.get("end"))
    return f"[{start} → {end}] {turn.get('speaker_name', 'Speaker_00')}: {turn.get('text', '')}"


def speaking_time(
    segments: list[dict[str, Any]], speakers: dict | None = None
) -> dict[str, float]:
    """Seconds of speech per display name (the normalized id when unmapped),
    in order of first appearance."""
    totals: dict[str, float] = {}
    for seg in segments:
        name = resolve_speaker_name(seg.get("speaker_id") or "Speaker_00", speakers)
        start = float(seg.get("start") or 0.0)
        end = float(seg.get("end") or start)
        totals[name] = totals.get(name, 0.0) + max(0.0, end - start)
    return totals
# [SEMANTIC-END] EXPORT:SPEAKER_TURNS
