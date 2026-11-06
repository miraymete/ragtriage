"""The result of one measurement: a score, a reason it does not apply, or a failure.

Keeping these apart is what stops an API error from looking like a low score. Averages use only
`Scored` values, and reports count the other two separately.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Outcome(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Scored(_Outcome):
    status: Literal["scored"] = "scored"
    value: float


class NotApplicable(_Outcome):
    status: Literal["not_applicable"] = "not_applicable"
    reason: str


class Failed(_Outcome):
    status: Literal["failed"] = "failed"
    reason: str


Outcome = Annotated[Scored | NotApplicable | Failed, Field(discriminator="status")]
