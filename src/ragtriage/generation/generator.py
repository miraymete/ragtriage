"""Generate one answer per question from the top-k retrieved chunks, and record what happened.

A generation either produces a valid answer (`ok`) or fails in a named way. Failures are recorded
and never scored, so a refusal or an API error cannot pass for a wrong answer or a right one.
"""

from __future__ import annotations

import json
from enum import StrEnum
from time import perf_counter
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ragtriage.generation.cache import RemembersValidated
from ragtriage.generation.costs import Price, estimate_cost
from ragtriage.generation.judge import FaithfulnessResult
from ragtriage.generation.llm import (
    Completion,
    LLMClient,
    LLMError,
    LLMErrorKind,
    LLMRequest,
    Origin,
    Source,
    StopReason,
    TokenUsage,
)
from ragtriage.generation.prompts import (
    ANSWER_SCHEMA,
    GENERATE_PROMPT_VERSION,
    GENERATE_SYSTEM,
    render_generation_prompt,
)
from ragtriage.generation.relevance import RelevanceResult
from ragtriage.matching import match_chunks
from ragtriage.models import QAExample, RetrievalRun, RetrievedChunk
from ragtriage.outcomes import Failed, Outcome, Scored


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class GenerationStatus(StrEnum):
    OK = "ok"
    INVALID_OUTPUT = "invalid_output"  # not the JSON we asked for, or an answer that says nothing
    REFUSED = "refused"
    TRUNCATED = "truncated"  # stopped at max_tokens, so the JSON may be incomplete
    API_ERROR = "api_error"


class ContextSpan(_Frozen):
    rank: int = Field(ge=1)
    doc_id: str
    start: int
    end: int


class GenerationRecord(_Frozen):
    """Everything about one answer. Nothing here says the answer is correct."""

    example_id: str
    # False means the test set gives no evidence: labeled unanswerable from this corpus.
    answerable: bool
    status: GenerationStatus
    abstained: bool | None = None
    answer: str | None = None
    problem: str | None = None
    # For API errors: the error kind, and the provider's key for errors that may come from the
    # settings rather than this question (see LLMError.repeat_key).
    error_kind: LLMErrorKind | None = None
    error_repeat_key: str | None = None
    raw_output: str | None = None
    contexts: tuple[ContextSpan, ...]
    # Whether gold evidence was among the top-k chunks actually sent to the model. None for
    # unanswerable questions. It says what the model could see, not that its answer is right.
    evidence_in_context: bool | None
    provider: str
    model: str
    prompt_version: str
    # Provider-reported usage. None means unknown (an API error), never zero. For a cache hit it is
    # the original call's usage, which this run did not spend again.
    usage: TokenUsage | None = None
    source: Source | None = None
    # For a cache hit, whether the original response was real ("api") or scripted ("fake").
    original_source: Origin | None = None
    # New USD spent by this run on this answer: 0.0 for a cache hit of a real call. None when no
    # price is set, the usage is unknown, or the response is fake. An estimate: SDK retries
    # before a final response are not visible here.
    cost_usd: float | None = None
    latency_s: float
    # Scored 1.0 when the model abstained exactly when the label says it should.
    abstention: Outcome
    # Set when a judge ran. Absent when no judge is configured.
    faithfulness: FaithfulnessResult | None = None
    # Set when answer relevance ran; independent of faithfulness.
    relevance: RelevanceResult | None = None


class InvalidOutput(ValueError):
    pass


def parse_answer(text: str) -> tuple[bool, str]:
    """Check the response against the schema and against what the fields mean.

    Structured outputs make valid JSON likely, not certain, so this never trusts the response.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise InvalidOutput(f"response is not JSON: {error.msg}") from error
    if not isinstance(data, dict):
        raise InvalidOutput("response is not a JSON object")
    if set(data) != {"abstained", "answer"}:
        raise InvalidOutput(f"expected keys abstained and answer, got {sorted(data)}")
    abstained, answer = data["abstained"], data["answer"]
    if not isinstance(abstained, bool) or not isinstance(answer, str):
        raise InvalidOutput("abstained must be a boolean and answer a string")
    if not abstained and not answer.strip():
        raise InvalidOutput("answer is empty but the model did not abstain")
    return abstained, answer


class Generator:
    def __init__(
        self,
        client: LLMClient,
        k: int,
        max_tokens: int = 16000,
        price: Price | None = None,
        min_overlap: float = 0.5,
    ) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self._client = client
        self._k = k
        self._max_tokens = max_tokens
        self._price = price
        self._min_overlap = min_overlap

    async def aclose(self) -> None:
        await self._client.aclose()

    @property
    def client(self) -> LLMClient:
        """The client this generator calls, so a stopped run can say which provider failed."""
        return self._client

    def context(self, run: RetrievalRun) -> tuple[RetrievedChunk, ...]:
        """The chunks the model sees: the top k. Deeper results are for retrieval triage."""
        return run.chunks[: self._k]

    async def generate(self, example: QAExample, run: RetrievalRun) -> GenerationRecord:
        """Answer one question. Raises only for errors that would fail every question."""
        chunks = self.context(run)
        base: dict[str, Any] = {
            "example_id": example.id,
            "answerable": example.answerable,
            "contexts": tuple(
                ContextSpan(rank=rank, doc_id=chunk.doc_id, start=chunk.start, end=chunk.end)
                for rank, chunk in enumerate(chunks, start=1)
            ),
            "evidence_in_context": (
                any(match_chunks(example.evidence, chunks, self._min_overlap))
                if example.answerable
                else None
            ),
            "provider": self._client.provider,
            "model": self._client.model,
            "prompt_version": GENERATE_PROMPT_VERSION,
        }
        request = LLMRequest(
            system=GENERATE_SYSTEM,
            prompt=render_generation_prompt(example.question, chunks),
            max_tokens=self._max_tokens,
            json_schema=ANSWER_SCHEMA,
            prompt_version=GENERATE_PROMPT_VERSION,
        )

        started = perf_counter()
        try:
            completion = await self._client.complete(request)
        except LLMError as error:
            if error.fatal:
                raise
            return GenerationRecord(
                **base,
                status=GenerationStatus.API_ERROR,
                problem=str(error),
                error_kind=error.kind,
                error_repeat_key=error.repeat_key,
                latency_s=perf_counter() - started,
                abstention=Failed(reason=GenerationStatus.API_ERROR),
            )
        base |= {
            "model": completion.model,
            "usage": completion.usage,
            "source": completion.source,
            "original_source": completion.original_source,
            "cost_usd": estimate_cost(self._price, completion),
            "latency_s": perf_counter() - started,
        }

        failure = _stop_failure(completion)
        if failure is None:
            try:
                abstained, answer = parse_answer(completion.text)
            except InvalidOutput as error:
                failure = (GenerationStatus.INVALID_OUTPUT, str(error))
        if failure is not None:
            status, problem = failure
            return GenerationRecord(
                **base,
                status=status,
                problem=problem,
                raw_output=completion.text or None,
                abstention=Failed(reason=status),
            )
        # Only now, after validation, may the response be cached.
        if isinstance(self._client, RemembersValidated):
            await self._client.remember(request, completion)
        return GenerationRecord(
            **base,
            status=GenerationStatus.OK,
            abstained=abstained,
            answer=answer,
            abstention=Scored(value=float(abstained != example.answerable)),
        )


def _stop_failure(completion: Completion) -> tuple[GenerationStatus, str] | None:
    if completion.stop_reason is StopReason.REFUSAL:
        return GenerationStatus.REFUSED, "the model declined to answer"
    if completion.stop_reason is StopReason.MAX_TOKENS:
        return GenerationStatus.TRUNCATED, "the response stopped at max_tokens"
    if completion.stop_reason is StopReason.OTHER:
        return (
            GenerationStatus.INVALID_OUTPUT,
            f"unexpected stop reason: {completion.raw_stop_reason}",
        )
    return None
