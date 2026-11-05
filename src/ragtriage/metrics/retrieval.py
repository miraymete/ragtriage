"""Deterministic retrieval metrics. No LLM, no I/O.

Every function takes `matches`: per rank, the set of gold items that result covers. An empty set is
an irrelevant result. See `ragtriage.matching.match_chunks`.
"""

from __future__ import annotations

from collections.abc import Collection, Hashable, Mapping, Sequence
from enum import StrEnum
from math import log2
from typing import TypeVar

Matches = Sequence[Collection[Hashable]]
T = TypeVar("T", bound=Hashable)


class Failure(StrEnum):
    """Where a question went wrong at cutoff k."""

    OK = "ok"  # all gold evidence is in the top k
    RANKING = "ranking"  # missing evidence was retrieved, but below k
    RETRIEVAL = "retrieval"  # missing evidence was never retrieved


def _check_k(k: int) -> None:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")


def _covered(matches: Matches) -> set[Hashable]:
    return {item for match in matches for item in match}


def hit_rate_at_k(matches: Matches, k: int) -> float:
    """1.0 if any of the top k results is relevant, else 0.0."""
    _check_k(k)
    return float(any(matches[:k]))


def recall_at_k(matches: Matches, gold: Collection[Hashable], k: int) -> float:
    """Share of the gold items covered by the top k."""
    _check_k(k)
    if not gold:
        raise ValueError("recall is undefined without gold items")
    return len(_covered(matches[:k]) & set(gold)) / len(gold)


def precision_at_k(matches: Matches, k: int) -> float:
    """Relevant results in the top k, divided by k even when fewer than k came back."""
    _check_k(k)
    return sum(1 for match in matches[:k] if match) / k


def reciprocal_rank(matches: Matches) -> float:
    """1 / rank of the first relevant result, or 0.0 if there is none."""
    for rank, match in enumerate(matches, start=1):
        if match:
            return 1 / rank
    return 0.0


def ndcg_at_k(matches: Sequence[Collection[T]], grades: Mapping[T, float], k: int) -> float:
    """Normalized DCG with linear gain, crediting each gold item once at its best rank.

    One chunk can cover several gold items at a single rank, which can beat the one-item-per-rank
    ideal, so the result is capped at 1.0.
    """
    _check_k(k)
    if not grades:
        raise ValueError("nDCG is undefined without gold items")
    dcg = 0.0
    seen: set[T] = set()
    for rank, match in enumerate(matches[:k], start=1):
        new = set(match) - seen
        dcg += sum(grades[item] for item in new) / log2(rank + 1)
        seen |= new
    best = sorted(grades.values(), reverse=True)[:k]
    ideal = sum(grade / log2(rank + 1) for rank, grade in enumerate(best, start=1))
    return min(1.0, dcg / ideal)


def classify_failure(matches: Matches, gold: Collection[Hashable], k: int) -> Failure:
    """Separate "fix the ranking" from "fix the retrieval".

    The run must go deeper than k. If it stops at k, nothing can sit below the cutoff and every
    failure reads as RETRIEVAL.
    """
    _check_k(k)
    missing = set(gold) - _covered(matches[:k])
    if not missing:
        return Failure.OK
    return Failure.RANKING if missing & _covered(matches[k:]) else Failure.RETRIEVAL
