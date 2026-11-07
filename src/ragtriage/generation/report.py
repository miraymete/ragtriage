"""Summarize generation records for one experiment."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from statistics import fmean
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, computed_field

from ragtriage.generation.generator import GenerationRecord, GenerationStatus
from ragtriage.generation.llm import LLMErrorKind, Origin, Source, TokenUsage
from ragtriage.outcomes import Failed, NotApplicable, Scored


class Rate(BaseModel):
    model_config = ConfigDict(frozen=True)

    numerator: int
    denominator: int

    @computed_field  # type: ignore[prop-decorator]
    @property
    def value(self) -> float | None:
        return self.numerator / self.denominator if self.denominator else None


class UsageSummary(BaseModel):
    """Usage of one role (generator or judge), kept apart from every other role."""

    model_config = ConfigDict(frozen=True)

    # New, real API calls made by this run. Cache hits and fake responses are excluded.
    billed_usage: TokenUsage
    # What reused cache entries used when they were first made: avoided, not spent.
    cached_original_usage: TokenUsage
    n_api_calls: int
    n_cached: int
    n_fake: int
    # Calls that failed without reported usage: tokens and cost unknown, not zero.
    n_usage_unknown: int
    # Estimated new USD. None when no price is set or no call had a known cost.
    cost_usd: float | None


class FaithfulnessSummary(BaseModel):
    """Faithfulness for one experiment. The mean uses only Scored answers."""

    model_config = ConfigDict(frozen=True)

    mean: float | None
    n_scored: int
    # Not applicable, by reason (abstained, no_claims, uncertain_claims, no_valid_answer, ...).
    not_applicable: dict[str, int]
    n_failed: int
    # Claim verdicts over answers whose judging completed, including uncertain ones.
    n_supported: int
    n_unsupported: int
    n_uncertain: int
    judge_usage: UsageSummary


class RelevanceSummary(BaseModel):
    """Answer relevance for one experiment: the mean of Scored values, each in [-1, 1]."""

    model_config = ConfigDict(frozen=True)

    mean: float | None
    n_scored: int
    not_applicable: dict[str, int]
    n_failed: int
    judge_usage: UsageSummary


class StopAttempt(BaseModel):
    """The call whose fatal error stopped the run.

    It has no GenerationRecord or JudgeCall, because its result never completed. It is kept here
    so the attempt is not lost from the call count.
    """

    model_config = ConfigDict(frozen=True)

    role: Literal["generator", "faithfulness_judge", "relevance_judge"]
    example_id: str
    provider: str
    model: str
    error_kind: LLMErrorKind
    status_code: int | None
    message: str
    # True only when the provider answered with an HTTP status, so a request reached the API.
    # False for a scripted fake client and for errors raised before anything was sent, such as
    # missing credentials.
    sent_to_api: bool
    # The provider reports no usage for a failed request: unknown, never zero.
    usage: TokenUsage | None = None
    cost_usd: float | None = None


class GenerationReport(BaseModel):
    """Counts and rates. Failed generations are counted, never folded into a rate.

    Abstention rates use the test set's labels. "Unanswerable" means no evidence was labeled in
    this corpus. An answerable question still counts as answerable when retrieval missed its
    evidence, so the answerable rate is split by whether evidence reached the model.
    """

    model_config = ConfigDict(frozen=True)

    n_examples: int
    n_attempted: int
    n_evaluated: int
    failures: dict[GenerationStatus, int]
    # Set when an error that would fail every request stopped the run early.
    stopped: str | None = None
    # Set when a fatal provider error stopped the run. Not set for cache errors or for the
    # repeated-error stop, whose calls all have records.
    stop_attempt: StopAttempt | None = None
    # Abstained / evaluated, for questions labeled unanswerable. Higher is better.
    abstained_when_unanswerable: Rate
    # Abstained / evaluated, for questions labeled answerable: false abstentions at corpus level.
    abstained_when_answerable: Rate
    # The same, split by whether gold evidence was in the top-k chunks sent to the model.
    # Abstaining without the evidence is a retrieval problem more than a generation one.
    abstained_when_answerable_evidence_present: Rate
    abstained_when_answerable_evidence_missing: Rate
    # Usage of new, real API calls made by this run. Cache hits and fake responses are excluded.
    billed_usage: TokenUsage
    # What the reused cache entries used when they were first made: avoided, not spent.
    cached_original_usage: TokenUsage
    n_api_calls: int
    n_cached: int
    # Fake responses, including cache hits that were fake when first made.
    n_fake: int
    # API errors, where the provider reported no usage. Tokens and cost for these are unknown.
    n_usage_unknown: int
    # Estimated USD for the known calls. None when no price is set or no call had a known cost.
    # Excludes calls with unknown usage and any retries the SDK made before a final response.
    cost_usd: float | None
    mean_latency_s: float | None
    # Present only when a judge ran. Its usage and cost are separate from the generator's above.
    faithfulness: FaithfulnessSummary | None = None
    # Present only when answer relevance ran. Its judge usage is counted here and nowhere else.
    relevance: RelevanceSummary | None = None
    records: tuple[GenerationRecord, ...]


def _abstention_rate(records: Sequence[GenerationRecord]) -> Rate:
    return Rate(numerator=sum(bool(r.abstained) for r in records), denominator=len(records))


class _Usage(Protocol):
    @property
    def usage(self) -> TokenUsage | None: ...
    @property
    def source(self) -> Source | None: ...
    @property
    def original_source(self) -> Origin | None: ...
    @property
    def cost_usd(self) -> float | None: ...


def summarize_usage(items: Sequence[_Usage]) -> UsageSummary:
    billed = TokenUsage()
    cached = TokenUsage()
    for item in items:
        if item.usage is None:
            continue
        if item.source == "api":
            billed = billed + item.usage
        elif item.source == "cache" and item.original_source == "api":
            cached = cached + item.usage
    known_costs = [i.cost_usd for i in items if i.cost_usd is not None]
    return UsageSummary(
        billed_usage=billed,
        cached_original_usage=cached,
        n_api_calls=sum(i.source == "api" for i in items),
        n_cached=sum(i.source == "cache" for i in items),
        n_fake=sum("fake" in (i.source, i.original_source) for i in items),
        n_usage_unknown=sum(i.usage is None for i in items),
        cost_usd=sum(known_costs) if known_costs else None,
    )


def summarize_faithfulness(records: Sequence[GenerationRecord]) -> FaithfulnessSummary | None:
    results = [r.faithfulness for r in records if r.faithfulness is not None]
    if not results:
        return None
    scores = [r.outcome.value for r in results if isinstance(r.outcome, Scored)]
    return FaithfulnessSummary(
        mean=fmean(scores) if scores else None,
        n_scored=len(scores),
        not_applicable=dict(
            Counter(r.outcome.reason for r in results if isinstance(r.outcome, NotApplicable))
        ),
        n_failed=sum(isinstance(r.outcome, Failed) for r in results),
        n_supported=sum(r.n_supported for r in results),
        n_unsupported=sum(r.n_unsupported for r in results),
        n_uncertain=sum(r.n_uncertain for r in results),
        judge_usage=summarize_usage([call for r in results for call in r.calls]),
    )


def summarize_relevance(records: Sequence[GenerationRecord]) -> RelevanceSummary | None:
    results = [r.relevance for r in records if r.relevance is not None]
    if not results:
        return None
    scores = [r.outcome.value for r in results if isinstance(r.outcome, Scored)]
    return RelevanceSummary(
        mean=fmean(scores) if scores else None,
        n_scored=len(scores),
        not_applicable=dict(
            Counter(r.outcome.reason for r in results if isinstance(r.outcome, NotApplicable))
        ),
        n_failed=sum(isinstance(r.outcome, Failed) for r in results),
        judge_usage=summarize_usage([r.call for r in results if r.call is not None]),
    )


def summarize(
    records: Sequence[GenerationRecord],
    n_examples: int | None = None,
    stopped: str | None = None,
    stop_attempt: StopAttempt | None = None,
) -> GenerationReport:
    evaluated = [r for r in records if r.status is GenerationStatus.OK]
    answerable = [r for r in evaluated if r.answerable]
    billed = TokenUsage()
    cached = TokenUsage()
    for record in records:
        if record.usage is None:
            continue
        if record.source == "api":
            billed = billed + record.usage
        elif record.source == "cache" and record.original_source == "api":
            cached = cached + record.usage
    known_costs = [r.cost_usd for r in records if r.cost_usd is not None]
    return GenerationReport(
        n_examples=len(records) if n_examples is None else n_examples,
        n_attempted=len(records),
        n_evaluated=len(evaluated),
        failures=dict(Counter(r.status for r in records if r.status is not GenerationStatus.OK)),
        stopped=stopped,
        stop_attempt=stop_attempt,
        abstained_when_unanswerable=_abstention_rate([r for r in evaluated if not r.answerable]),
        abstained_when_answerable=_abstention_rate(answerable),
        abstained_when_answerable_evidence_present=_abstention_rate(
            [r for r in answerable if r.evidence_in_context]
        ),
        abstained_when_answerable_evidence_missing=_abstention_rate(
            [r for r in answerable if not r.evidence_in_context]
        ),
        billed_usage=billed,
        cached_original_usage=cached,
        n_api_calls=sum(r.source == "api" for r in records),
        n_cached=sum(r.source == "cache" for r in records),
        n_fake=sum("fake" in (r.source, r.original_source) for r in records),
        n_usage_unknown=sum(r.usage is None for r in records),
        cost_usd=sum(known_costs) if known_costs else None,
        mean_latency_s=fmean(r.latency_s for r in records) if records else None,
        faithfulness=summarize_faithfulness(records),
        relevance=summarize_relevance(records),
        records=tuple(records),
    )
