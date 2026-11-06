"""A provider-independent interface to a language model.

Add a provider by writing one class with `provider`, `model` and an async `complete` method that
returns a `Completion` or raises `LLMError`. Retries belong to the provider client, never to the
code calling `complete`, so that two retry loops never stack.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class StopReason(StrEnum):
    COMPLETE = "complete"
    MAX_TOKENS = "max_tokens"
    REFUSAL = "refusal"
    OTHER = "other"


class TokenUsage(_Frozen):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cache_read_input_tokens: int = Field(default=0, ge=0)
    cache_creation_input_tokens: int = Field(default=0, ge=0)

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens + other.cache_read_input_tokens,
            cache_creation_input_tokens=(
                self.cache_creation_input_tokens + other.cache_creation_input_tokens
            ),
        )


class LLMRequest(_Frozen):
    """What every provider needs. Sampling, effort and similar knobs differ by provider and model,
    so they belong to each provider's config and client, not here."""

    system: str
    prompt: str
    max_tokens: int = Field(ge=1)
    json_schema: dict[str, Any] | None = None
    # Not sent to the provider. Part of the cache key, so a reworded prompt never reuses old
    # answers.
    prompt_version: str | None = None


# api: a real, billed call. cache: an earlier response replayed; its usage is not new spend.
# fake: a scripted test response; it is never billed.
Source = Literal["api", "cache", "fake"]
# Where a response was first produced. A cache never produces responses itself.
Origin = Literal["api", "fake"]


class Completion(_Frozen):
    """One model response. Usage is what the provider reported, for failed answers too.

    For a cache hit, `source` is "cache", `usage` is what the original call used, and
    `original_source` says whether that call was real or fake.
    """

    text: str
    stop_reason: StopReason
    raw_stop_reason: str | None = None
    usage: TokenUsage
    model: str
    source: Source
    original_source: Origin | None = None

    @model_validator(mode="after")
    def _origin_only_for_cache(self) -> Self:
        if (self.source == "cache") != (self.original_source is not None):
            raise ValueError("original_source is required for cache hits and only for them")
        return self


class LLMErrorKind(StrEnum):
    AUTH = "auth"
    BILLING = "billing"
    PERMISSION = "permission"
    NOT_FOUND = "not_found"
    BAD_REQUEST = "bad_request"
    TOO_LARGE = "too_large"
    RATE_LIMIT = "rate_limit"
    OVERLOADED = "overloaded"
    SERVER = "server"
    TIMEOUT = "timeout"
    CONNECTION = "connection"


# These fail every request the same way, so running the remaining questions only wastes time.
FATAL_KINDS = frozenset(
    {LLMErrorKind.AUTH, LLMErrorKind.BILLING, LLMErrorKind.PERMISSION, LLMErrorKind.NOT_FOUND}
)


class LLMError(Exception):
    """A request that failed after the provider client's own retries.

    `run_wide` is for errors the provider identifies, by a structured code, as affecting every
    request (an account on hold, a spend limit). Such errors are fatal whatever their kind.

    `repeat_key` is for errors that may come from the run's settings rather than the question,
    when the provider gives no code to tell which. Two consecutive questions failing with the
    same key stop the run; see `generation.pipeline`.
    """

    def __init__(
        self,
        kind: LLMErrorKind,
        message: str,
        status_code: int | None = None,
        *,
        error_code: str | None = None,
        run_wide: bool = False,
        repeat_key: str | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.status_code = status_code
        self.error_code = error_code
        self.run_wide = run_wide
        self.repeat_key = repeat_key

    @property
    def fatal(self) -> bool:
        return self.run_wide or self.kind in FATAL_KINDS

    def __str__(self) -> str:
        status = f" (HTTP {self.status_code})" if self.status_code is not None else ""
        return f"{self.kind}{status}: {self.args[0]}"


class LLMClient(Protocol):
    provider: str
    model: str

    async def complete(self, request: LLMRequest) -> Completion: ...

    def response_settings(self) -> dict[str, Any]:
        """Client settings that can change the response (not credentials, retries or timeouts).

        A cache uses them in its key, so a response made under other settings is never reused.
        """
        ...

    async def aclose(self) -> None:
        """Release connections. Safe to call more than once."""
        ...
