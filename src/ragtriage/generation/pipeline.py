"""Run generation over every experiment's retrieval results."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from ragtriage.generation.cache import CacheError
from ragtriage.generation.generator import GenerationRecord, GenerationStatus, Generator
from ragtriage.generation.judge import FaithfulnessJudge, FaithfulnessResult
from ragtriage.generation.llm import LLMClient, LLMError
from ragtriage.generation.relevance import RelevanceJudge, RelevanceResult
from ragtriage.generation.report import GenerationReport, StopAttempt, summarize
from ragtriage.models import QAExample, RetrievalRun
from ragtriage.outcomes import Failed, NotApplicable

# How many consecutive questions must fail with the same repeat key before the run stops.
REPEATED_ERROR_LIMIT = 2


async def generate_reports(
    generator: Generator,
    examples: Sequence[QAExample],
    runs_per_experiment: Sequence[Sequence[RetrievalRun]],
    judge: FaithfulnessJudge | None = None,
    relevance: RelevanceJudge | None = None,
) -> list[GenerationReport]:
    """One report per experiment, in order. Questions run one at a time for now.

    A failed question is recorded and the run moves on. The run stops, keeping the answers
    already produced, in two cases:

    - an error that fails every request (credentials, billing, permission, unknown model, or a
      run-wide code from the provider);
    - the same possibly-settings-related error (`repeat_key`) on two consecutive questions, such
      as an unsupported option, which would otherwise be retried on every question.

    The last report then has `stopped` set, and later experiments are left out. Other exceptions
    are bugs and propagate unchanged.

    With a `judge`, each valid answer is also judged for faithfulness against the same top-k
    chunks the generator saw, whether or not retrieval found the gold evidence. With `relevance`,
    it is also scored for answer relevance. The two metrics are independent: one failing for an
    answer does not skip the other. Only a fatal error stops both, and the whole run.
    """
    reports = []
    for runs in runs_per_experiment:
        by_id = {run.example_id: run for run in runs}
        records: list[GenerationRecord] = []
        # Which call is in flight, so a fatal error can be attributed to it.
        current: tuple[Role, LLMClient, str] | None = None
        try:
            for example in examples:
                run = by_id[example.id]
                current = ("generator", generator.client, example.id)
                record = await generator.generate(example, run)
                try:
                    if judge is not None:
                        current = ("faithfulness_judge", judge.client, example.id)
                        result = await _judge(judge, generator, example, run, record)
                        record = record.model_copy(update={"faithfulness": result})
                    if relevance is not None:
                        current = ("relevance_judge", relevance.client, example.id)
                        scored = await _relevance(relevance, example, record)
                        record = record.model_copy(update={"relevance": scored})
                except (LLMError, CacheError) as error:
                    records.append(_stopped(record, judge, relevance, error))
                    raise
                records.append(record)
                repeated = _repeated_error(records)
                if repeated:
                    reports.append(summarize(records, n_examples=len(examples), stopped=repeated))
                    return reports
        except LLMError as error:
            if not error.fatal:
                raise  # the generator records these itself; reaching here would be a bug
            attempt = _stop_attempt(current, error) if current is not None else None
            reports.append(
                summarize(
                    records, n_examples=len(examples), stopped=str(error), stop_attempt=attempt
                )
            )
            return reports
        except CacheError as error:
            # A cache that cannot be trusted stops the run instead of falling back to paid calls.
            stopped = f"cache error: {error}"
            reports.append(summarize(records, n_examples=len(examples), stopped=stopped))
            return reports
        reports.append(summarize(records))
    return reports


Role = Literal["generator", "faithfulness_judge", "relevance_judge"]


def _stop_attempt(current: tuple[Role, LLMClient, str], error: LLMError) -> StopAttempt:
    role, client, example_id = current
    return StopAttempt(
        role=role,
        example_id=example_id,
        provider=client.provider,
        model=client.model,
        error_kind=error.kind,
        status_code=error.status_code,
        message=str(error),
        # An HTTP status means the request reached the provider. The scripted fake never does.
        sent_to_api=client.provider != "fake" and error.status_code is not None,
    )


async def _judge(
    judge: FaithfulnessJudge,
    generator: Generator,
    example: QAExample,
    run: RetrievalRun,
    record: GenerationRecord,
) -> FaithfulnessResult:
    if record.status is not GenerationStatus.OK or record.answer is None:
        return FaithfulnessResult(outcome=NotApplicable(reason="no_valid_answer"))
    return await judge.evaluate(
        example.question, record.answer, bool(record.abstained), generator.context(run)
    )


async def _relevance(
    relevance: RelevanceJudge, example: QAExample, record: GenerationRecord
) -> RelevanceResult:
    if record.status is not GenerationStatus.OK or record.answer is None:
        return RelevanceResult(outcome=NotApplicable(reason="no_valid_answer"))
    return await relevance.evaluate(example.question, record.answer, bool(record.abstained))


def _stopped(
    record: GenerationRecord,
    judge: FaithfulnessJudge | None,
    relevance: RelevanceJudge | None,
    error: Exception,
) -> GenerationRecord:
    """Keep the answer, and mark the metrics that did not finish as stopped."""
    failed = Failed(reason=f"judge stopped: {error}")
    update: dict[str, object] = {}
    if judge is not None and record.faithfulness is None:
        update["faithfulness"] = FaithfulnessResult(outcome=failed)
    if relevance is not None and record.relevance is None:
        update["relevance"] = RelevanceResult(outcome=failed)
    return record.model_copy(update=update)


def _repeated_error(records: Sequence[GenerationRecord]) -> str | None:
    if len(records) < REPEATED_ERROR_LIMIT:
        return None
    recent = records[-REPEATED_ERROR_LIMIT:]
    keys = {r.error_repeat_key for r in recent}
    if len(keys) != 1 or None in keys:
        return None
    # The cause is unknown: a setting is a likely explanation, but this is not proof of one.
    return (
        f"stopped early after an unclassified invalid-request error repeated on "
        f"{REPEATED_ERROR_LIMIT} consecutive questions (cause unknown; a setting may be "
        f"responsible): {recent[-1].problem}"
    )


async def run_generation(
    generator: Generator,
    examples: Sequence[QAExample],
    runs_per_experiment: Sequence[Sequence[RetrievalRun]],
    judge: FaithfulnessJudge | None = None,
    relevance: RelevanceJudge | None = None,
) -> list[GenerationReport]:
    """`generate_reports`, then close every client whether it succeeded or raised."""
    try:
        return await generate_reports(generator, examples, runs_per_experiment, judge, relevance)
    finally:
        try:
            await generator.aclose()
        finally:
            try:
                if judge is not None:
                    await judge.aclose()
            finally:
                if relevance is not None:
                    await relevance.aclose()
