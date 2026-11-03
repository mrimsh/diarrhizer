"""One conservative size estimate for summary and chat requests; no tokenizer dependency."""

import math
from dataclasses import asdict, dataclass, field

from diarrhizer.summary.providers import KIND_ANTHROPIC_API, Provider

CHARS_PER_TOKEN = 3.0
MESSAGE_OVERHEAD = 16
REQUEST_OVERHEAD = 32
DEFAULT_RESPONSE_TOKENS = 1024
ANTHROPIC_RESPONSE_TOKENS = 8192  # Must match adapters.llm's default request max_tokens.


# [SEMANTIC-BEGIN] LLM:CONTEXT_BUDGET
# @purpose: Check the entire request plus a response reserve before any paid/provider call
# @description: Estimates every rendered message (template, note, transcript, history and new question),
#   adds framing overhead and an explicit max_tokens reserve or a documented backend default.
#   Unknown context limits remain advisory. This is an estimate, not a tokenizer guarantee; never truncates.
# @inputs: complete message texts, Provider and optional context limit override
# @outputs: shared ContextCheck for CLI, GUI previews, summary and chat; ContextTooLong on refusal
# @sideEffects: None
# @errors: ValueError for invalid limit override
# @see: SUMMARY:JOB, SUMMARY:CHAT, ADAPTER:LLM
def estimate_tokens(chars: int) -> int:
    return math.ceil(chars / CHARS_PER_TOKEN)


@dataclass
class ContextOption:
    kind: str
    label: str
    data: dict = field(default_factory=dict)


@dataclass
class ContextCheck:
    estimated_tokens: int  # Total, including framing and reserved output.
    limit: int | None
    options: list = field(default_factory=list)
    input_tokens: int = 0
    overhead_tokens: int = 0
    reserved_output_tokens: int = 0

    @property
    def exceeds(self) -> bool:
        return self.limit is not None and self.estimated_tokens > self.limit

    def to_dict(self) -> dict:
        return asdict(self)


class ContextTooLong(ValueError):
    def __init__(self, check: ContextCheck, provider_name: str):
        self.check = check
        super().__init__(
            f"The context is about {check.estimated_tokens:,} tokens, over the {check.limit:,} set for "
            f"'{provider_name}' (input {check.input_tokens:,}, framing {check.overhead_tokens:,}, "
            f"response reserve {check.reserved_output_tokens:,}). Nothing was sent. Options: "
            + "; ".join(option.label for option in check.options)
        )


def check_request(texts, provider: Provider, context_limit: int | None = None) -> ContextCheck:
    texts = list(texts)
    limit = provider.context_tokens if context_limit is None else context_limit
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("Context limit must be a positive integer")
    reserve = provider.max_tokens or (
        ANTHROPIC_RESPONSE_TOKENS if provider.kind == KIND_ANTHROPIC_API else DEFAULT_RESPONSE_TOKENS
    )
    input_tokens = estimate_tokens(sum(len(text) for text in texts))
    overhead = REQUEST_OVERHEAD + MESSAGE_OVERHEAD * len(texts)
    return ContextCheck(input_tokens + overhead + reserve, limit,
                        input_tokens=input_tokens, overhead_tokens=overhead, reserved_output_tokens=reserve)


def describe(check: ContextCheck) -> str:
    limit = f" / {check.limit:,}" if check.limit is not None else " (предел не задан)"
    return (f"Бюджет ≈ {check.estimated_tokens:,}{limit} токенов: запрос {check.input_tokens:,}, "
            f"оформление {check.overhead_tokens:,}, резерв ответа {check.reserved_output_tokens:,}."
            + (" Лимит превышен; запрос не будет отправлен." if check.exceeds else ""))
# [SEMANTIC-END] LLM:CONTEXT_BUDGET
