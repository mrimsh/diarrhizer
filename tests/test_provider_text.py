"""Tests for diarrhizer_gui.provider_text: the words for what a provider reports. Only what the provider
reported is described - an absent field gives no text, and nothing is invented to fill the gap.
"""

import re

import pytest

pytest.importorskip("PySide6", reason="GUI extra not installed")  # the package imports Qt-free, but lives next to it

from diarrhizer.adapters.llm import CLAUDE_COST_NOTE, PRICE_COST_NOTE, PROVIDER_COST_NOTE  # noqa: E402
from diarrhizer_gui import provider_text  # noqa: E402


def test_account_text_names_who_is_signed_in_and_on_which_plan():
    assert provider_text.account_text(None) == "Вход не выполнен"
    assert provider_text.account_text({"signed_in": False}) == "Вход не выполнен"
    assert provider_text.account_text(
        {"signed_in": True, "email": "me@example.com", "method": "ChatGPT", "plan": "prolite"}
    ) == "Вход выполнен: me@example.com · ChatGPT · тариф Pro Lite"
    assert provider_text.account_text(
        {"signed_in": True, "email": "", "method": "claude.ai", "plan": "max", "organization": "Example Org"}
    ) == "Вход выполнен: аккаунт без e-mail · claude.ai · тариф Max · Example Org"


def test_unknown_plans_are_shown_as_the_provider_wrote_them():
    assert provider_text.plan_name("enterprise_cbp_usage_based") == "Enterprise Cbp Usage Based"
    assert provider_text.plan_name("PLUS") == "Plus"


def test_usage_text_counts_in_thousands_and_leaves_out_what_is_zero():
    assert provider_text.usage_text(None) == "" and provider_text.usage_text({"input_tokens": 0}) == ""
    assert provider_text.usage_text(
        {"input_tokens": 13712, "output_tokens": 5, "cached_tokens": 1000, "reasoning_tokens": 2}
    ) == "токены: вход 13 712 (из кэша 1 000), выход 5 (из них рассуждение 2)"
    assert provider_text.usage_text({"input_tokens": 10, "output_tokens": 1}) == "токены: вход 10, выход 1"


def test_a_cost_says_where_it_comes_from_and_a_zero_cost_is_still_a_cost():
    assert provider_text.cost_text(None) == "" and provider_text.cost_text("n/a") == ""
    assert provider_text.cost_text(0.0123) == "стоимость ≈ $0.0123"
    assert provider_text.cost_text(0.0, PROVIDER_COST_NOTE) == "стоимость ≈ $0.0000 (по данным провайдера)"
    assert "по ценам, заданным для провайдера" in provider_text.cost_text(1.0, PRICE_COST_NOTE)
    text = provider_text.cost_text(0.5, CLAUDE_COST_NOTE)
    assert "оценка Claude Code" in text and "подписка по токенам не списывается" in text
    assert provider_text.cost_text(0.5, "something new") == "стоимость ≈ $0.5000 (something new)"  # kept as is


@pytest.mark.parametrize(
    "minutes, expected",
    [(300, "5 ч"), (10080, "7 дн."), (1440, "1 дн."), (45, "45 мин"), (90, "90 мин"), (None, ""), (0, "")],
)
def test_window_lengths(minutes, expected):
    assert provider_text.window_length(minutes) == expected


def test_window_names_drop_raw_machine_labels_when_the_length_is_known():
    assert provider_text.window_name({"label": "", "window_minutes": 10080}) == "7 дн."
    assert provider_text.window_name({"label": "GPT-Spark", "window_minutes": 300}) == "GPT-Spark, 5 ч"
    assert provider_text.window_name({"label": "five_hour", "window_minutes": 300}) == "5 ч"
    assert provider_text.window_name({"label": "five_hour", "window_minutes": None}) == "five_hour"
    assert provider_text.window_name({}) == "окно лимита"


def test_a_window_line_has_percent_status_and_reset_time():
    line = provider_text.window_text(
        {"label": "", "window_minutes": 10080, "used_percent": 18.4, "resets_at": 1793293782.0, "status": ""}
    )
    assert re.fullmatch(r"7 дн\.: использовано 18%, сброс \d\d\.\d\d в \d\d:\d\d", line)
    assert provider_text.window_text({"window_minutes": 300, "used_percent": None, "status": "allowed"}) == "5 ч: allowed"
    assert provider_text.window_text({"window_minutes": 300}) == "5 ч: без данных"
    warned = provider_text.window_text({"window_minutes": 300, "used_percent": 82.0, "status": "allowed_warning"})
    assert warned == "5 ч: использовано 82%, allowed_warning"


def test_credits_are_described_without_inventing_a_balance():
    assert provider_text.credits_text("") == ""
    assert provider_text.credits_text("none") == "кредитов нет"
    assert provider_text.credits_text("unlimited") == "кредиты без ограничений"
    assert provider_text.credits_text("12.50") == "кредиты: баланс 12.50"


def test_limits_lines_end_with_the_credits_and_the_exhausted_warning():
    assert provider_text.limits_lines(None) == [] and provider_text.limits_lines({}) == []
    lines = provider_text.limits_lines(
        {"windows": [{"window_minutes": 10080, "used_percent": 100.0}], "credits": "none", "allowed": False,
         "note": "Something to know"},
        window_prefix="Лимит ",
    )
    assert lines == ["Лимит 7 дн.: использовано 100%", "Кредитов нет", "Лимит тарифа исчерпан", "Something to know"]


def test_run_lines_say_only_what_the_request_reported():
    assert provider_text.run_lines({}) == []
    assert provider_text.run_lines({"model": None, "usage": None, "cost_usd": None, "limits": None}) == []
    lines = provider_text.run_lines({
        "model": "gpt-6-astra",
        "usage": {"input_tokens": 13712, "output_tokens": 5},
        "cost_usd": None,
        "limits": {"windows": [{"window_minutes": 10080, "used_percent": 18.0}], "credits": "", "allowed": True},
    })
    assert lines == [
        "модель gpt-6-astra · токены: вход 13 712, выход 5",
        "Лимит 7 дн.: использовано 18%",
    ]
