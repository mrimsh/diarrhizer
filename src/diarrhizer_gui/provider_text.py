"""Words for what a provider reports: the account, a request's tokens and cost, the plan's limits.

Pure functions over the to_dict() shapes of diarrhizer.summary.provider_info (that is what crosses the
child-process queue and what a summary's .json holds), so the account panel and the summary dialog say the
same thing the same way. Only what the provider reported is described - an absent field gives no text.
"""

from datetime import datetime
from typing import Optional

from diarrhizer.adapters.llm import CLAUDE_COST_NOTE, PRICE_COST_NOTE, PROVIDER_COST_NOTE

COST_NOTES = {
    CLAUDE_COST_NOTE: "оценка Claude Code по тарифам API; подписка по токенам не списывается",
    PROVIDER_COST_NOTE: "по данным провайдера",
    PRICE_COST_NOTE: "по ценам, заданным для провайдера",
}

# The providers' own plan names, as people write them.
PLAN_NAMES = {
    "prolite": "Pro Lite",
    "plus": "Plus",
    "pro": "Pro",
    "max": "Max",
    "team": "Team",
    "business": "Business",
    "enterprise": "Enterprise",
    "edu": "Edu",
    "free": "Free",
    "go": "Go",
}


def plan_name(plan: str) -> str:
    return PLAN_NAMES.get(plan.lower(), plan.replace("_", " ").title())


def account_text(account: Optional[dict]) -> str:
    """"Вход выполнен: me@example.com · ChatGPT · тариф Plus", or that nobody is signed in."""
    if not account or not account.get("signed_in"):
        return "Вход не выполнен"
    parts = [account.get("email") or "аккаунт без e-mail"]
    if account.get("method"):
        parts.append(account["method"])
    if account.get("plan"):
        parts.append(f"тариф {plan_name(account['plan'])}")
    if account.get("organization"):
        parts.append(account["organization"])
    return "Вход выполнен: " + " · ".join(parts)


def number(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def usage_text(usage: Optional[dict]) -> str:
    if not usage or not (usage.get("input_tokens") or usage.get("output_tokens")):
        return ""
    cached = f" (из кэша {number(usage['cached_tokens'])})" if usage.get("cached_tokens") else ""
    text = f"токены: вход {number(usage.get('input_tokens', 0))}{cached}, выход {number(usage.get('output_tokens', 0))}"
    if usage.get("reasoning_tokens"):
        text += f" (из них рассуждение {number(usage['reasoning_tokens'])})"
    return text


def cost_text(cost_usd: Optional[float], note: str = "") -> str:
    if not isinstance(cost_usd, (int, float)):
        return ""
    text = f"стоимость ≈ ${cost_usd:.4f}"
    return f"{text} ({COST_NOTES.get(note, note)})" if note else text


def window_length(minutes: Optional[int]) -> str:
    if not minutes:
        return ""
    if minutes % 1440 == 0:
        return f"{minutes // 1440} дн."
    if minutes % 60 == 0:
        return f"{minutes // 60} ч"
    return f"{minutes} мин"


def window_name(window: dict) -> str:
    """"7 дн.", "GPT-Spark, 5 ч". A raw machine label such as five_hour is dropped when the length is known."""
    label = window.get("label") or ""
    length = window_length(window.get("window_minutes"))
    if "_" in label and length:
        label = ""
    return ", ".join(part for part in (label, length) if part) or "окно лимита"


def reset_text(resets_at: Optional[float]) -> str:
    if not isinstance(resets_at, (int, float)):
        return ""
    return "сброс " + datetime.fromtimestamp(resets_at).strftime("%d.%m в %H:%M")


def window_text(window: dict) -> str:
    """"7 дн.: использовано 18%, сброс 28.10 в 14:49"."""
    used = window.get("used_percent")
    parts = [f"использовано {used:.0f}%" if isinstance(used, (int, float)) else (window.get("status") or "без данных")]
    if isinstance(used, (int, float)) and window.get("status") not in (None, "", "allowed"):
        parts.append(window["status"])
    reset = reset_text(window.get("resets_at"))
    if reset:
        parts.append(reset)
    return f"{window_name(window)}: " + ", ".join(parts)


def credits_text(credits: str) -> str:
    if not credits:
        return ""
    return {"none": "кредитов нет", "unlimited": "кредиты без ограничений"}.get(credits, f"кредиты: баланс {credits}")


def limits_lines(limits: Optional[dict], window_prefix: str = "") -> list[str]:
    """One line per usage window (after `window_prefix`), then credits and the "exhausted" warning."""
    if not limits:
        return []
    lines = [window_prefix + window_text(window) for window in limits.get("windows") or []]
    credits = credits_text(limits.get("credits") or "")
    if credits:
        lines.append(credits[:1].upper() + credits[1:])
    if limits.get("allowed") is False:
        lines.append("Лимит тарифа исчерпан")
    if limits.get("note"):
        lines.append(str(limits["note"]))
    return lines


def run_lines(meta: dict) -> list[str]:
    """What one request used, for a summary's or a check's metadata: model, tokens, cost, limits."""
    lines = []
    head = [f"модель {meta['model']}"] if meta.get("model") else []
    for part in (usage_text(meta.get("usage")), cost_text(meta.get("cost_usd"), meta.get("cost_note") or "")):
        if part:
            head.append(part)
    if head:
        lines.append(" · ".join(head))
    lines.extend(limits_lines(meta.get("limits"), window_prefix="Лимит "))
    return lines
