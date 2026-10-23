"""What a provider reports about itself and about a request.

Plain data, stdlib only, with a to_dict()/from_dict() pair each so the values
can cross the GUI's child-process queue and be stored in a summary's .json.
The adapters (adapters/llm.py, adapters/llm_accounts.py) fill them in from
whatever the provider's own tool or API offers; a field the provider does not
report stays empty rather than being guessed.
"""

import dataclasses
from dataclasses import dataclass
from typing import Any, Mapping, Optional


def _pick(cls, data: Mapping[str, Any]) -> dict:
    """Only the keys `cls` knows - a stored record from another version must not crash the loader."""
    known = {f.name for f in dataclasses.fields(cls)}
    return {key: value for key, value in data.items() if key in known}


# [SEMANTIC-BEGIN] CONFIG:PROVIDER_INFO
# @purpose: The shapes of what a provider tells us: who is signed in, which models, what a request used, what is left
# @description: Account (signed-in state, e-mail, plan), ModelOption (an id to pass to the provider, a display
#   name, the provider's own description - which for Claude includes its price), Usage (token counts of one
#   request), Limits (usage windows of the plan with percent used and reset time, plus credits). Nothing here
#   is computed or defaulted on the provider's behalf: a cost, a percentage or a plan name is present only
#   when the provider reported it (or, for cost, when the user set prices - see Provider.price_input).
# @inputs: values parsed by the adapters
# @outputs: dataclasses with to_dict()/from_dict()
# @sideEffects: None
# @errors: None (from_dict ignores unknown keys)
# @see: ADAPTER:LLM, ADAPTER:LLM_ACCOUNTS, SUMMARY:JOB, GUI:ACCOUNT_PANEL
@dataclass(frozen=True)
class Account:
    """Who a subscription provider is signed in as."""

    signed_in: bool
    email: str = ""
    plan: str = ""  # the provider's own plan name, e.g. "plus", "max"
    method: str = ""  # how the sign-in was made, e.g. "ChatGPT", "claude.ai", "API key"
    organization: str = ""

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Account":
        return cls(**_pick(cls, data))


@dataclass(frozen=True)
class ModelOption:
    """One model a provider offers."""

    id: str  # what goes to the provider's --model / "model" parameter; "" = the provider's own default
    name: str = ""
    description: str = ""
    default: bool = False
    resolved: str = ""  # the concrete model behind an alias such as "sonnet", when the provider says

    @property
    def label(self) -> str:
        return self.name or self.id

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelOption":
        return cls(**_pick(cls, data))


@dataclass(frozen=True)
class Usage:
    """Token counts of one request."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0  # input tokens served from the provider's prompt cache
    reasoning_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Usage":
        return cls(**_pick(cls, data))


@dataclass(frozen=True)
class LimitWindow:
    """One usage window of a plan, e.g. "7 days: 18 % used, resets at ..."."""

    label: str = ""
    used_percent: Optional[float] = None
    window_minutes: Optional[int] = None
    resets_at: Optional[float] = None  # unix seconds
    status: str = ""  # the provider's own word, e.g. "allowed", "allowed_warning", "rejected"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LimitWindow":
        return cls(**_pick(cls, data))


@dataclass(frozen=True)
class Limits:
    """The plan's usage windows and what else the provider says about what is left."""

    windows: tuple = ()  # of LimitWindow
    credits: str = ""  # e.g. "balance 0", "unlimited"
    allowed: Optional[bool] = None  # False when the plan's usage is exhausted
    note: str = ""  # something the user should know, e.g. why there are no windows

    def to_dict(self) -> dict:
        return {
            "windows": [window.to_dict() for window in self.windows],
            "credits": self.credits,
            "allowed": self.allowed,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Limits":
        windows = tuple(LimitWindow.from_dict(w) for w in data.get("windows") or [])
        return cls(
            windows=windows,
            credits=data.get("credits", ""),
            allowed=data.get("allowed"),
            note=data.get("note", ""),
        )
# [SEMANTIC-END] CONFIG:PROVIDER_INFO
