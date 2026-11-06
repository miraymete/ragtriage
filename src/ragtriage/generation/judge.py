"""Faithfulness: are the claims in an answer supported by the passages the generator saw?

Two calls per answer. The first splits the answer into atomic claims; the second gives each claim
a verdict against the passages. Every response is validated in code before it is used or cached.

This measures support by the given context only. It does not measure correctness, completeness or
relevance, and the judge itself has not been validated against human labels.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum
from html import unescape
from time import perf_counter
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict

from ragtriage.generation.cache import RemembersValidated
from ragtriage.generation.costs import Price, estimate_cost
from ragtriage.generation.judge_prompts import (
    CLAIMS_PROMPT_VERSION,
    CLAIMS_SCHEMA,
    CLAIMS_SYSTEM,
    VERDICTS_PROMPT_VERSION,
    VERDICTS_SCHEMA,
    VERDICTS_SYSTEM,
    render_claims_prompt,
    render_verdicts_prompt,
)
from ragtriage.generation.llm import (
    LLMClient,
    LLMError,
    LLMRequest,
    Origin,
    Source,
    StopReason,
    TokenUsage,
)
from ragtriage.models import Chunk
from ragtriage.outcomes import Failed, NotApplicable, Outcome, Scored

T = TypeVar("T")


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Verdict(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNCERTAIN = "uncertain"


class Claim(_Frozen):
    """One atomic claim. `id` is assigned by code from the claim's position, not by the model."""

    id: str
    text: str
    verdict: Verdict | None = None
    # Only for supported claims: the passage cited and a quote found verbatim in it. A quote
    # that is present does not prove the passage supports the claim's meaning.
    passage_id: str | None = None
    quote: str | None = None


class JudgeCall(_Frozen):
    stage: Literal["claims", "verdicts", "questions"]
    prompt_version: str
    provider: str
    model: str
    status: Literal["ok", "failed"]
    problem: str | None = None
    # None means unknown (an API error), never zero.
    usage: TokenUsage | None = None
    source: Source | None = None
    original_source: Origin | None = None
    cost_usd: float | None = None
    latency_s: float


class FaithfulnessResult(_Frozen):
    """`outcome` is Scored only when every claim has a definite verdict."""

    outcome: Outcome
    claims: tuple[Claim, ...] = ()
    n_supported: int = 0
    n_unsupported: int = 0
    n_uncertain: int = 0
    calls: tuple[JudgeCall, ...] = ()


class JudgeOutputError(ValueError):
    pass


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _load_object(text: str, keys: set[str]) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise JudgeOutputError(f"response is not JSON: {error.msg}") from error
    if not isinstance(data, dict) or set(data) != keys:
        raise JudgeOutputError(f"expected an object with keys {sorted(keys)}")
    return data


def parse_claims(text: str) -> list[str]:
    claims = _load_object(text, {"claims"})["claims"]
    if not isinstance(claims, list) or not all(isinstance(c, str) for c in claims):
        raise JudgeOutputError("claims must be a list of strings")
    cleaned = [claim.strip() for claim in claims]
    if any(not claim for claim in cleaned):
        raise JudgeOutputError("a claim is empty")
    normalized = [_normalize(claim) for claim in cleaned]
    if len(set(normalized)) != len(normalized):
        raise JudgeOutputError("the same claim appears twice")
    return cleaned


def parse_verdicts(text: str, claims: Sequence[Claim], passages: Mapping[str, str]) -> list[Claim]:
    """Check every verdict, and for supported claims check the cited passage and quote."""
    items = _load_object(text, {"verdicts"})["verdicts"]
    if not isinstance(items, list):
        raise JudgeOutputError("verdicts must be a list")
    fields = {"claim_id", "verdict", "passage_id", "quote"}
    by_id: dict[str, dict[str, str]] = {}
    for item in items:
        if not isinstance(item, dict) or set(item) != fields:
            raise JudgeOutputError(f"each verdict needs exactly the keys {sorted(fields)}")
        if not all(isinstance(item[key], str) for key in fields):
            raise JudgeOutputError("verdict fields must be strings")
        claim_id = item["claim_id"]
        if claim_id in by_id:
            raise JudgeOutputError(f"claim {claim_id!r} has more than one verdict")
        by_id[claim_id] = item

    expected = [claim.id for claim in claims]
    missing = [cid for cid in expected if cid not in by_id]
    unknown = sorted(set(by_id) - set(expected))
    if missing or unknown:
        raise JudgeOutputError(
            f"verdicts do not match the claims: missing {missing}, unknown {unknown}"
        )

    judged = []
    for claim in claims:
        item = by_id[claim.id]
        try:
            verdict = Verdict(item["verdict"])  # exact casing only
        except ValueError as error:
            raise JudgeOutputError(f"unknown verdict {item['verdict']!r}") from error
        if verdict is not Verdict.SUPPORTED:
            judged.append(claim.model_copy(update={"verdict": verdict}))
            continue
        passage_id, quote = item["passage_id"], item["quote"]
        if passage_id not in passages:
            raise JudgeOutputError(f"claim {claim.id} cites unknown passage {passage_id!r}")
        if not quote.strip():
            raise JudgeOutputError(f"claim {claim.id} is supported without a quote")
        # The model saw escaped passages, so a quote may contain &amp; for &. Whitespace may also
        # differ (passages can contain line breaks); every other character must match.
        if _normalize(unescape(quote)) not in _normalize(passages[passage_id]):
            raise JudgeOutputError(f"the quote for claim {claim.id} is not in passage {passage_id}")
        judged.append(
            claim.model_copy(update={"verdict": verdict, "passage_id": passage_id, "quote": quote})
        )
    return judged


Stage = Literal["claims", "verdicts", "questions"]


async def run_judge_stage(
    client: LLMClient,
    price: Price | None,
    stage: Stage,
    prompt_version: str,
    request: LLMRequest,
    parse: Callable[[str], T],
) -> tuple[JudgeCall, T | None]:
    """One judge call: complete, check the stop reason, parse, and only then cache.

    Shared by every judge-based metric. Raises only fatal errors; everything else becomes a
    failed `JudgeCall` with whatever usage the provider reported. There is no automatic retry.
    """
    base: dict[str, Any] = {
        "stage": stage,
        "prompt_version": prompt_version,
        "provider": client.provider,
        "model": client.model,
    }
    started = perf_counter()
    try:
        completion = await client.complete(request)
    except LLMError as error:
        if error.fatal:
            raise
        call = JudgeCall(
            **base, status="failed", problem=str(error), latency_s=perf_counter() - started
        )
        return call, None
    base |= {
        "model": completion.model,
        "usage": completion.usage,
        "source": completion.source,
        "original_source": completion.original_source,
        "cost_usd": estimate_cost(price, completion),
        "latency_s": perf_counter() - started,
    }
    if completion.stop_reason is not StopReason.COMPLETE:
        reason = completion.raw_stop_reason or str(completion.stop_reason)
        return JudgeCall(**base, status="failed", problem=f"stop reason: {reason}"), None
    try:
        parsed = parse(completion.text)
    except JudgeOutputError as error:
        return JudgeCall(**base, status="failed", problem=str(error)), None
    if isinstance(client, RemembersValidated):
        await client.remember(request, completion)
    return JudgeCall(**base, status="ok"), parsed


class FaithfulnessJudge:
    """Judges one answer at a time with any `LLMClient`, optionally wrapped in a cache."""

    def __init__(self, client: LLMClient, max_tokens: int = 4096, price: Price | None = None):
        self._client = client
        self._max_tokens = max_tokens
        self._price = price

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _stage(
        self,
        stage: Stage,
        prompt_version: str,
        request: LLMRequest,
        parse: Callable[[str], T],
    ) -> tuple[JudgeCall, T | None]:
        return await run_judge_stage(
            self._client, self._price, stage, prompt_version, request, parse
        )

    async def evaluate(
        self, question: str, answer: str, abstained: bool, chunks: Sequence[Chunk]
    ) -> FaithfulnessResult:
        """Judge a valid answer against exactly the chunks the generator was given.

        Labels, evidence and reference answers are deliberately not inputs.
        """
        if abstained:
            return FaithfulnessResult(outcome=NotApplicable(reason="abstained"))
        if not answer.strip():
            return FaithfulnessResult(outcome=NotApplicable(reason="empty_answer"))

        claims_call, texts = await self._stage(
            "claims",
            CLAIMS_PROMPT_VERSION,
            LLMRequest(
                system=CLAIMS_SYSTEM,
                prompt=render_claims_prompt(question, answer),
                max_tokens=self._max_tokens,
                json_schema=CLAIMS_SCHEMA,
                prompt_version=CLAIMS_PROMPT_VERSION,
            ),
            parse_claims,
        )
        if texts is None:
            reason = f"claims stage failed: {claims_call.problem}"
            return FaithfulnessResult(outcome=Failed(reason=reason), calls=(claims_call,))
        claims = [Claim(id=f"c{i}", text=text) for i, text in enumerate(texts, start=1)]
        if not claims:
            return FaithfulnessResult(
                outcome=NotApplicable(reason="no_claims"), calls=(claims_call,)
            )

        passages = {f"p{rank}": chunk.text for rank, chunk in enumerate(chunks, start=1)}
        verdicts_call, judged = await self._stage(
            "verdicts",
            VERDICTS_PROMPT_VERSION,
            LLMRequest(
                system=VERDICTS_SYSTEM,
                prompt=render_verdicts_prompt(
                    [(f"p{rank}", c.doc_id, c.text) for rank, c in enumerate(chunks, start=1)],
                    [(claim.id, claim.text) for claim in claims],
                ),
                max_tokens=self._max_tokens,
                json_schema=VERDICTS_SCHEMA,
                prompt_version=VERDICTS_PROMPT_VERSION,
            ),
            lambda text: parse_verdicts(text, claims, passages),
        )
        calls = (claims_call, verdicts_call)
        if judged is None:
            # No partial score: claims are kept for inspection, without verdicts.
            reason = f"verdicts stage failed: {verdicts_call.problem}"
            return FaithfulnessResult(
                outcome=Failed(reason=reason), claims=tuple(claims), calls=calls
            )

        counts = {verdict: sum(c.verdict is verdict for c in judged) for verdict in Verdict}
        outcome: Outcome
        if counts[Verdict.UNCERTAIN]:
            outcome = NotApplicable(reason="uncertain_claims")
        else:
            outcome = Scored(value=counts[Verdict.SUPPORTED] / len(judged))
        return FaithfulnessResult(
            outcome=outcome,
            claims=tuple(judged),
            n_supported=counts[Verdict.SUPPORTED],
            n_unsupported=counts[Verdict.UNSUPPORTED],
            n_uncertain=counts[Verdict.UNCERTAIN],
            calls=calls,
        )
