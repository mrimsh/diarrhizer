"""Small, shared ASR profiles and validation; safe to import without ML libraries."""

import math

ASR_PROFILES = ("default", "ru-call")
VAD_ONSET = 0.5
VAD_OFFSET = 0.363
CHUNK_SIZE = 30
RU_CALL_PROMPT = (
    "Обсуждаем рабочие вопросы на русском языке. "
    "Что уже готово? Проверим задачи, сроки и результаты. "
    "Да, договорились. Нет, этот вопрос ещё нужно уточнить."
)


# [SEMANTIC-BEGIN] CONFIG:ASR_PROFILE
# @purpose: Compose reproducible ASR hints and validate supported WhisperX VAD controls
# @description: An opt-in Russian-call style sample precedes the user's prompt and glossary.
#   The glossary is last because Whisper keeps the tail when its prompt token budget is exceeded.
#   Profiles never override the separately selected language or model. No text post-processing.
# @inputs: profile name, optional user prompt/glossary, VAD onset/offset, chunk size in seconds
# @outputs: composed prompt or None; validated VAD controls
# @sideEffects: None
# @errors: ValueError for unknown profiles, invalid text or non-finite/out-of-range controls
# @see: CONFIG:PIPELINE, STAGE:TRANSCRIBE, ADAPTER:WHISPERX_ASR
def compose_asr_prompt(profile: str = "default", initial_prompt: str | None = None,
                       glossary: str | None = None) -> str | None:
    """Combine profile sample, custom prompt and relevant terms without editing ASR output."""
    if profile not in ASR_PROFILES:
        raise ValueError(f"Unknown ASR profile: {profile!r}; choose {', '.join(ASR_PROFILES)}")
    for name, value in (("initial_prompt", initial_prompt), ("glossary", glossary)):
        if value is not None and not isinstance(value, str):
            raise ValueError(f"ASR {name} must be text or None")
    # Preserve the existing custom-prompt contract when no profile/glossary is used.
    if profile == "default" and not (glossary or "").strip():
        return initial_prompt
    parts = [RU_CALL_PROMPT] if profile == "ru-call" else []
    if initial_prompt and initial_prompt.strip():
        parts.append(initial_prompt.strip())
    if glossary and glossary.strip():
        parts.append("Термины и имена: " + ", ".join(line.strip() for line in glossary.splitlines() if line.strip()))
    return "\n\n".join(parts) or None


def validate_vad_options(onset: float = VAD_ONSET, offset: float = VAD_OFFSET,
                         chunk_size: int = CHUNK_SIZE) -> None:
    """Reject unsupported thresholds/chunk durations before models or job artifacts are touched."""
    for name, value in (("asr_vad_onset", onset), ("asr_vad_offset", offset)):
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value < 1:
            raise ValueError(f"{name} must be a finite number between 0 and 1 (exclusive)")
    if offset > onset:
        raise ValueError("asr_vad_offset must be less than or equal to asr_vad_onset")
    if type(chunk_size) is not int or not 1 <= chunk_size <= 30:
        raise ValueError("asr_chunk_size must be an integer from 1 to 30 seconds")
# [SEMANTIC-END] CONFIG:ASR_PROFILE
