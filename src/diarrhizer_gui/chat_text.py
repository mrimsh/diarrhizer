"""Russian words for what the dialogue screen shows: the context, what a turn left out, the totals, the options.

Pure functions over the dicts of diarrhizer.summary.chat (what a stored turn holds), so the screen and its tests
say the same thing the same way.
"""

from typing import Optional

from diarrhizer.adapters.llm import CONTEXT_FULL_HISTORY, CONTEXT_SESSION
from diarrhizer.export.turns import format_timestamp
from diarrhizer.summary import chat
from diarrhizer.summary.providers import KIND_ANTHROPIC_API, KIND_CLAUDE_CLI, KIND_CODEX_CLI, KIND_OPENAI_API, Provider


def number(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def how_context_works(provider: Optional[Provider], keep_session: bool) -> str:
    """How the chosen provider handles the dialogue's context: its cache, a CLI session, or the whole history."""
    if provider is None:
        return ""
    if provider.kind == KIND_ANTHROPIC_API:
        return "кэш провайдера: инструкция и транскрипт помечены точкой кэша и повторно читаются из него"
    if provider.kind == KIND_OPENAI_API:
        return "кэш провайдера: автоматический, по одинаковому началу запроса"
    if provider.kind == KIND_CODEX_CLI:
        return (
            "сессия Codex: после первого хода отправляется только новое сообщение"
            if keep_session else "полная история: весь диалог отправляется заново на каждом ходу"
        )
    if provider.kind == KIND_CLAUDE_CLI:
        return "полная история: весь диалог отправляется заново на каждом ходу (сессии Claude не используются)"
    return ""


def context_mode(mode: str) -> str:
    if mode == CONTEXT_SESSION:
        return "сессия CLI (отправлено только новое сообщение)"
    if mode == CONTEXT_FULL_HISTORY:
        return "полная история"
    return ""


def truncation(notes: list) -> str:
    """What a turn's request left out, in words; "" when nothing."""
    parts = []
    for note in notes or []:
        if note.get("kind") == "omitted_turns":
            parts.append(f"самые старые ходы ({note.get('count')} из {note.get('of')}) не отправлялись")
        elif note.get("kind") == "window":
            parts.append(
                f"транскрипт ограничен окном {format_timestamp(note.get('start'))}–{format_timestamp(note.get('end'))} "
                f"({note.get('kept')} из {note.get('of')} реплик)"
            )
    return "В запрос не вошло: " + "; ".join(parts) + "." if parts else ""


def totals(data: dict) -> str:
    """The dialogue's total, from chat.conversation_totals()."""
    if not data.get("turns"):
        return "Ходов пока нет."
    parts = [
        f"ходов {data['turns']}",
        f"вход {number(data['input_tokens'])}",
        f"выход {number(data['output_tokens'])}",
    ]
    if data.get("cached_tokens"):
        parts.append(f"из кэша {number(data['cached_tokens'])}")
    if data.get("cache_write_tokens"):
        parts.append(f"записано в кэш {number(data['cache_write_tokens'])}")
    if data.get("cost_usd") is not None:
        parts.append(f"стоимость ≈ ${data['cost_usd']:.4f}")
    return "Итого по диалогу: " + ", ".join(parts)


def too_long(check: chat.ContextCheck, provider_name: str) -> str:
    return (
        f"Контекст ≈ {number(check.estimated_tokens)} токенов, а предел для «{provider_name}» — "
        f"{number(check.limit)}. Ничего не отправлено. Что сделать?"
    )


def option_label(option: chat.ContextOption) -> str:
    if option.kind == "drop_turns":
        return f"Убрать {option.data['count']} самых старых ходов (транскрипт остаётся)"
    if option.kind == "window":
        data = option.data
        return (
            f"Ограничить транскрипт: {format_timestamp(data['start'])}–{format_timestamp(data['end'])} "
            f"({data['kept']} из {data['of']} реплик)"
        )
    return "Сменить модель…"


def warning(text: str) -> str:
    """A warning of chat.create_conversation, in Russian where it is a known one."""
    if "prompt cache works only within this dialogue" in text:
        return "Заметка стоит перед транскриптом: кэш эффективен только в рамках этого диалога."
    return text
