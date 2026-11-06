"""Cost estimates from provider-reported usage. Prices are configured, never built in."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ragtriage.generation.llm import Completion


class Price(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    input_per_mtok: float = Field(ge=0)
    output_per_mtok: float = Field(ge=0)


def estimate_cost(price: Price | None, completion: Completion) -> float | None:
    """New USD for one response: 0.0 for a cache hit of a real call, None when unknown or fake.

    An estimate: retries the SDK made before this response are not visible.
    """
    if price is None or "fake" in (completion.source, completion.original_source):
        return None
    if completion.source == "cache":
        return 0.0  # no new spend; the original call was paid for when it was made
    usage = completion.usage
    return (
        usage.input_tokens * price.input_per_mtok + usage.output_tokens * price.output_per_mtok
    ) / 1_000_000
