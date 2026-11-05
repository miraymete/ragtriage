"""Score retrieval runs against a test set."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from statistics import fmean

from pydantic import BaseModel, ConfigDict

from ragtriage.matching import match_chunks
from ragtriage.metrics.retrieval import (
    Failure,
    classify_failure,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from ragtriage.models import QAExample, RetrievalRun


class ExampleScore(BaseModel):
    model_config = ConfigDict(frozen=True)

    example_id: str
    scores: dict[str, float]
    failure: Failure


class RetrievalReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    mean: dict[str, float]
    per_example: tuple[ExampleScore, ...]
    failures: dict[Failure, int]
    triage_k: int
    n_unanswerable_skipped: int


def evaluate_retrieval(
    examples: Iterable[QAExample],
    runs: Iterable[RetrievalRun],
    ks: Sequence[int] = (1, 3, 5, 10),
    triage_k: int | None = None,
    min_overlap: float = 0.5,
) -> RetrievalReport:
    """Score each answerable example and average the results.

    Unanswerable questions are skipped: retrieval metrics are undefined for them, and scoring them
    as zero would drag every configuration down. They are counted in the report instead.

    Every example must have exactly one run, because a missing run would silently change the mean.
    """
    examples = list(examples)
    runs_by_id: dict[str, RetrievalRun] = {}
    for run in runs:
        if run.example_id in runs_by_id:
            raise ValueError(f"duplicate run for example {run.example_id!r}")
        runs_by_id[run.example_id] = run
    example_ids = {example.id for example in examples}
    if len(example_ids) != len(examples):
        raise ValueError("example ids must be unique")
    if example_ids != runs_by_id.keys():
        raise ValueError(
            "examples and runs do not line up: "
            f"no run for {sorted(example_ids - runs_by_id.keys())}, "
            f"no example for {sorted(runs_by_id.keys() - example_ids)}"
        )
    answerable = [example for example in examples if example.answerable]
    if not answerable:
        raise ValueError("no answerable examples to score")
    if not ks:
        raise ValueError("ks must not be empty")
    cutoff = max(ks) if triage_k is None else triage_k

    per_example = []
    for example in answerable:
        matches = match_chunks(example.evidence, runs_by_id[example.id].chunks, min_overlap)
        gold = range(len(example.evidence))
        grades = {i: float(item.grade) for i, item in enumerate(example.evidence)}
        scores = {"mrr": reciprocal_rank(matches)}
        for k in ks:
            scores[f"hit_rate@{k}"] = hit_rate_at_k(matches, k)
            scores[f"recall@{k}"] = recall_at_k(matches, gold, k)
            scores[f"precision@{k}"] = precision_at_k(matches, k)
            scores[f"ndcg@{k}"] = ndcg_at_k(matches, grades, k)
        per_example.append(
            ExampleScore(
                example_id=example.id,
                scores=scores,
                failure=classify_failure(matches, gold, cutoff),
            )
        )

    return RetrievalReport(
        mean={
            name: fmean(score.scores[name] for score in per_example)
            for name in per_example[0].scores
        },
        per_example=tuple(per_example),
        failures=dict(Counter(score.failure for score in per_example)),
        triage_k=cutoff,
        n_unanswerable_skipped=len(examples) - len(answerable),
    )
