"""Uncertainty for per-question retrieval scores: bootstrap intervals and paired tests.

Every result here assumes the questions are independent draws from the questions the system will
really face. Questions written about the same document are not independent, and a small,
self-written test set is not a sample of real use. No method in this module fixes either problem.

All p-values are unadjusted. Looking at several metrics or several pairs of experiments raises the
chance that one of them looks significant by luck.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from math import isfinite
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from ragtriage.evaluate import RetrievalReport

# 2**20 sign patterns (about a million) are enumerated in chunks; more than that is sampled.
MAX_EXACT_NONZERO = 20
# Rows of work generated at once, so no step holds every resample or sign pattern in memory.
_CHUNK_CELLS = 1_000_000

Floats = NDArray[np.float64]


def statistic_metrics(k: int) -> tuple[str, ...]:
    """Metrics with intervals: recall@k, reciprocal rank (its mean is MRR) and nDCG@k."""
    return (f"recall@{k}", "mrr", f"ndcg@{k}")


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class BootstrapSettings(_Model):
    method: Literal["percentile"] = "percentile"
    confidence: float = Field(default=0.95, gt=0, lt=1)
    n_resamples: int = Field(default=10_000, ge=1)
    seed: int = 0


class PermutationSettings(_Model):
    # Used only when there are more than MAX_EXACT_NONZERO nonzero differences.
    n_draws: int = Field(default=100_000, ge=1)
    seed: int = 0
    max_exact_nonzero: int = MAX_EXACT_NONZERO


class MeanInterval(_Model):
    metric: str
    n: int
    mean: float
    # None when there is one question: one value says nothing about spread.
    ci_low: float | None
    ci_high: float | None
    # "single_question": no interval. "constant": every value is the same, so the interval has
    # zero width. That reflects the sample, not certainty about new questions.
    note: Literal["single_question", "constant"] | None = None


class ExperimentStatistics(_Model):
    experiment: str
    n_questions: int
    n_unanswerable_excluded: int
    bootstrap: BootstrapSettings
    metrics: tuple[MeanInterval, ...]


class PermutationTest(_Model):
    """Two-sided paired sign-flip test of the mean difference. Unadjusted for multiple tests."""

    method: Literal["exact", "monte_carlo"]
    p_value: float
    adjusted: Literal[False] = False
    n_nonzero: int
    # Exact: every one of 2**n_nonzero sign patterns. Monte Carlo: random patterns drawn.
    n_patterns: int
    # Patterns at least as extreme as the observed difference, ties included.
    n_extreme: int
    seed: int | None = None


class PairedDifference(_Model):
    metric: str
    n: int
    mean_baseline: float
    mean_candidate: float
    # Always candidate - baseline.
    mean_difference: float
    ci_low: float | None
    ci_high: float | None
    n_candidate_higher: int
    n_candidate_lower: int
    n_tied: int
    permutation: PermutationTest | None
    note: Literal["single_question", "all_zero", "constant_difference"] | None = None


class Comparison(_Model):
    baseline: str
    candidate: str
    direction: Literal["candidate - baseline"] = "candidate - baseline"
    n_questions: int
    n_unanswerable_excluded: int
    bootstrap: BootstrapSettings
    permutation: PermutationSettings
    metrics: tuple[PairedDifference, ...]


def _check_values(values: Sequence[float], what: str) -> Floats:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{what}: expected a flat list of values")
    if array.size == 0:
        raise ValueError(f"{what}: no values, so there is nothing to estimate")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{what}: values must be finite")
    return array


def _resampled_means(values: Floats, settings: BootstrapSettings) -> Floats:
    """Means of `n_resamples` bootstrap samples, drawn in chunks."""
    rng = np.random.default_rng(settings.seed)
    n = values.size
    rows = max(1, _CHUNK_CELLS // n)
    means = np.empty(settings.n_resamples)
    for start in range(0, settings.n_resamples, rows):
        stop = min(start + rows, settings.n_resamples)
        indices = rng.integers(0, n, size=(stop - start, n))
        means[start:stop] = values[indices].mean(axis=1)
    return means


def _percentile_interval(means: Floats, confidence: float) -> tuple[float, float]:
    alpha = 1 - confidence
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2], method="linear")
    return float(low), float(high)


def bootstrap_mean(
    values: Sequence[float], metric: str, settings: BootstrapSettings
) -> MeanInterval:
    """Percentile bootstrap interval for the mean of per-question values."""
    array = _check_values(values, metric)
    mean = float(array.mean())
    if array.size == 1:
        return MeanInterval(
            metric=metric, n=1, mean=mean, ci_low=None, ci_high=None, note="single_question"
        )
    if np.all(array == array[0]):
        return MeanInterval(
            metric=metric, n=array.size, mean=mean, ci_low=mean, ci_high=mean, note="constant"
        )
    low, high = _percentile_interval(_resampled_means(array, settings), settings.confidence)
    return MeanInterval(metric=metric, n=array.size, mean=mean, ci_low=low, ci_high=high)


def _is_extreme(sums: Floats, observed: float) -> NDArray[np.bool_]:
    # Ties count as extreme. The tolerance absorbs rounding, so a pattern whose sum equals the
    # observed one in exact arithmetic is never missed.
    return np.abs(sums) >= abs(observed) - 1e-9 * max(1.0, abs(observed))


def _sign_chunks_exact(m: int) -> Iterator[NDArray[np.int8]]:
    """All 2**m sign patterns, a chunk at a time."""
    total = 1 << m
    rows = max(1, _CHUNK_CELLS // max(m, 1))
    bits = np.arange(m, dtype=np.int64)
    for start in range(0, total, rows):
        codes = np.arange(start, min(start + rows, total), dtype=np.int64)
        yield (((codes[:, None] >> bits) & 1) * 2 - 1).astype(np.int8)


def sign_flip_test(differences: Sequence[float], settings: PermutationSettings) -> PermutationTest:
    """Two-sided paired permutation test: flip the sign of each per-question difference.

    Under the null hypothesis each difference is as likely to be positive as negative. Zero
    differences do not change when flipped, so only nonzero ones are permuted. Up to
    `max_exact_nonzero` of them, every pattern is enumerated and p = extreme / 2**m. Beyond that,
    random patterns are drawn and p = (extreme + 1) / (draws + 1), which never reports zero.
    """
    array = _check_values(differences, "differences")
    nonzero = array[array != 0]
    m = nonzero.size
    observed = float(nonzero.sum())
    if m == 0:
        # The only pattern is the observed one: there is no evidence of a difference.
        return PermutationTest(method="exact", p_value=1.0, n_nonzero=0, n_patterns=1, n_extreme=1)
    if m <= settings.max_exact_nonzero:
        extreme = sum(
            int(_is_extreme(signs @ nonzero, observed).sum()) for signs in _sign_chunks_exact(m)
        )
        total = 1 << m
        return PermutationTest(
            method="exact",
            p_value=extreme / total,
            n_nonzero=m,
            n_patterns=total,
            n_extreme=extreme,
        )
    rng = np.random.default_rng(settings.seed)
    rows = max(1, _CHUNK_CELLS // m)
    extreme = 0
    for start in range(0, settings.n_draws, rows):
        count = min(rows, settings.n_draws - start)
        signs = rng.integers(0, 2, size=(count, m), dtype=np.int8) * 2 - 1
        extreme += int(_is_extreme(signs @ nonzero, observed).sum())
    return PermutationTest(
        method="monte_carlo",
        p_value=(extreme + 1) / (settings.n_draws + 1),
        n_nonzero=m,
        n_patterns=settings.n_draws,
        n_extreme=extreme,
        seed=settings.seed,
    )


def _by_id(pairs: Sequence[tuple[str, float]], side: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for example_id, value in pairs:
        if example_id in values:
            raise ValueError(f"{side}: duplicate question id {example_id!r}")
        if not isfinite(value):
            raise ValueError(f"{side}: value for {example_id!r} is not finite")
        values[example_id] = value
    return values


def paired_difference(
    baseline: Sequence[tuple[str, float]],
    candidate: Sequence[tuple[str, float]],
    metric: str,
    bootstrap: BootstrapSettings,
    permutation: PermutationSettings,
) -> PairedDifference:
    """Compare two experiments question by question. Differences are candidate - baseline.

    Both sides must score exactly the same question ids. Anything else is an error: dropping the
    unmatched questions would quietly change what is being compared.
    """
    a = _by_id(baseline, "baseline")
    b = _by_id(candidate, "candidate")
    if a.keys() != b.keys():
        raise ValueError(
            f"{metric}: the experiments scored different questions: "
            f"only in baseline {sorted(a.keys() - b.keys())}, "
            f"only in candidate {sorted(b.keys() - a.keys())}"
        )
    if not a:
        raise ValueError(f"{metric}: no questions to compare")
    ids = sorted(a)
    va = np.array([a[i] for i in ids])
    vb = np.array([b[i] for i in ids])
    diff = vb - va
    common = {
        "metric": metric,
        "n": len(ids),
        "mean_baseline": float(va.mean()),
        "mean_candidate": float(vb.mean()),
        "mean_difference": float(diff.mean()),
        "n_candidate_higher": int((diff > 0).sum()),
        "n_candidate_lower": int((diff < 0).sum()),
        "n_tied": int((diff == 0).sum()),
    }
    if len(ids) == 1:
        return PairedDifference(
            **common, ci_low=None, ci_high=None, permutation=None, note="single_question"
        )
    test = sign_flip_test(diff, permutation)
    if np.all(diff == diff[0]):
        mean = float(diff[0])
        note: Literal["all_zero", "constant_difference"] = (
            "all_zero" if mean == 0 else "constant_difference"
        )
        return PairedDifference(**common, ci_low=mean, ci_high=mean, permutation=test, note=note)
    # Resampling the differences resamples the same question indices on both sides.
    low, high = _percentile_interval(_resampled_means(diff, bootstrap), bootstrap.confidence)
    return PairedDifference(**common, ci_low=low, ci_high=high, permutation=test)


def metric_values(report: RetrievalReport, metric: str) -> list[tuple[str, float]]:
    """(question id, score) for every scored question. Unanswerable ones are not in the report."""
    try:
        return [(score.example_id, score.scores[metric]) for score in report.per_example]
    except KeyError:
        raise ValueError(f"metric {metric!r} is not in the report") from None


def experiment_statistics(
    name: str, report: RetrievalReport, metrics: Sequence[str], settings: BootstrapSettings
) -> ExperimentStatistics:
    return ExperimentStatistics(
        experiment=name,
        n_questions=len(report.per_example),
        n_unanswerable_excluded=report.n_unanswerable_skipped,
        bootstrap=settings,
        metrics=tuple(
            bootstrap_mean([v for _, v in metric_values(report, m)], m, settings) for m in metrics
        ),
    )


def compare_experiments(
    baseline: tuple[str, RetrievalReport],
    candidate: tuple[str, RetrievalReport],
    metrics: Sequence[str],
    bootstrap: BootstrapSettings,
    permutation: PermutationSettings,
) -> Comparison:
    (name_a, report_a), (name_b, report_b) = baseline, candidate
    if name_a == name_b:
        raise ValueError("baseline and candidate must be different experiments")
    if report_a.n_unanswerable_skipped != report_b.n_unanswerable_skipped:
        raise ValueError("the experiments excluded different numbers of unanswerable questions")
    return Comparison(
        baseline=name_a,
        candidate=name_b,
        n_questions=len(report_a.per_example),
        n_unanswerable_excluded=report_a.n_unanswerable_skipped,
        bootstrap=bootstrap,
        permutation=permutation,
        metrics=tuple(
            paired_difference(
                metric_values(report_a, m), metric_values(report_b, m), m, bootstrap, permutation
            )
            for m in metrics
        ),
    )


# --- group-level (cluster) bootstrap ----------------------------------------------------------
#
# Questions from the same article share paragraphs, wording and difficulty, so they are not
# independent draws. This bootstrap resamples whole groups with replacement and keeps every
# question of each drawn group, as many times as the group is drawn. The estimate is the
# question-weighted mean, the same quantity as the metric tables: sum of values / number of
# questions over the drawn groups. It is not the mean of per-group means.
#
# Resampling groups does not fix a small number of groups or groups that do not represent real
# use. There is no group-level p-value here.


class GroupMeanInterval(_Model):
    metric: str
    n_groups: int
    n_questions: int
    mean: float
    ci_low: float | None
    ci_high: float | None
    note: Literal["single_group", "constant"] | None = None


class GroupStatistics(_Model):
    experiment: str
    method: Literal["group percentile bootstrap"] = "group percentile bootstrap"
    weighting: Literal["question"] = "question"
    n_groups: int
    n_questions: int
    n_unanswerable_excluded: int
    bootstrap: BootstrapSettings
    metrics: tuple[GroupMeanInterval, ...]


class GroupDifference(_Model):
    metric: str
    n_groups: int
    n_questions: int
    mean_baseline: float
    mean_candidate: float
    # Always candidate - baseline.
    mean_difference: float
    ci_low: float | None
    ci_high: float | None
    note: Literal["single_group", "all_zero", "constant_difference"] | None = None


class GroupComparison(_Model):
    baseline: str
    candidate: str
    direction: Literal["candidate - baseline"] = "candidate - baseline"
    method: Literal["paired group percentile bootstrap"] = "paired group percentile bootstrap"
    weighting: Literal["question"] = "question"
    n_groups: int
    n_questions: int
    n_unanswerable_excluded: int
    bootstrap: BootstrapSettings
    metrics: tuple[GroupDifference, ...]


def _group_arrays(
    values: Sequence[tuple[str, float]], groups: Mapping[str, str], what: str
) -> tuple[list[str], Floats, Floats]:
    """Per group, in name order: names, sum of values, and number of questions."""
    by_id = _by_id(values, what)
    if not by_id:
        raise ValueError(f"{what}: no questions")
    missing = sorted(i for i in by_id if i not in groups)
    if missing:
        raise ValueError(f"{what}: no group for {missing[:5]}")
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for qid, value in by_id.items():
        group = groups[qid]
        sums[group] = sums.get(group, 0.0) + value
        counts[group] = counts.get(group, 0) + 1
    names = sorted(sums)
    return (
        names,
        np.array([sums[g] for g in names], dtype=np.float64),
        np.array([counts[g] for g in names], dtype=np.float64),
    )


def resample_groups(sums: Floats, counts: Floats, settings: BootstrapSettings) -> Floats:
    """Question-weighted means of `n_resamples` group resamples, drawn in chunks.

    Each resample draws as many group indices as there are groups, with replacement. A group
    drawn twice counts twice, with all its questions.
    """
    rng = np.random.default_rng(settings.seed)
    g = sums.size
    rows = max(1, _CHUNK_CELLS // g)
    means = np.empty(settings.n_resamples)
    for start in range(0, settings.n_resamples, rows):
        stop = min(start + rows, settings.n_resamples)
        drawn = rng.integers(0, g, size=(stop - start, g))
        means[start:stop] = sums[drawn].sum(axis=1) / counts[drawn].sum(axis=1)
    return means


def group_bootstrap_mean(
    values: Sequence[tuple[str, float]],
    groups: Mapping[str, str],
    metric: str,
    settings: BootstrapSettings,
) -> GroupMeanInterval:
    names, sums, counts = _group_arrays(values, groups, metric)
    mean = float(sums.sum() / counts.sum())
    n_groups, n_questions = len(names), int(counts.sum())
    if n_groups == 1:
        return GroupMeanInterval(
            metric=metric,
            n_groups=1,
            n_questions=n_questions,
            mean=mean,
            ci_low=None,
            ci_high=None,
            note="single_group",
        )
    flat = np.array([v for _, v in values], dtype=np.float64)
    if np.all(flat == flat[0]):
        return GroupMeanInterval(
            metric=metric,
            n_groups=n_groups,
            n_questions=n_questions,
            mean=mean,
            ci_low=mean,
            ci_high=mean,
            note="constant",
        )
    low, high = _percentile_interval(resample_groups(sums, counts, settings), settings.confidence)
    return GroupMeanInterval(
        metric=metric,
        n_groups=n_groups,
        n_questions=n_questions,
        mean=mean,
        ci_low=low,
        ci_high=high,
    )


def group_paired_difference(
    baseline: Sequence[tuple[str, float]],
    candidate: Sequence[tuple[str, float]],
    groups: Mapping[str, str],
    metric: str,
    settings: BootstrapSettings,
) -> GroupDifference:
    """candidate - baseline, with both experiments resampled by the same group draws."""
    a = _by_id(baseline, "baseline")
    b = _by_id(candidate, "candidate")
    if a.keys() != b.keys():
        raise ValueError(
            f"{metric}: the experiments scored different questions: "
            f"only in baseline {sorted(a.keys() - b.keys())}, "
            f"only in candidate {sorted(b.keys() - a.keys())}"
        )
    diffs = [(qid, b[qid] - a[qid]) for qid in sorted(a)]
    names, sums, counts = _group_arrays(diffs, groups, metric)
    total = counts.sum()
    mean_a = sum(a.values()) / total
    mean_b = sum(b.values()) / total
    mean_diff = float(sums.sum() / total)
    n_groups, n_questions = len(names), int(total)

    def result(low: float | None, high: float | None, note: str | None) -> GroupDifference:
        return GroupDifference.model_validate(
            {
                "metric": metric,
                "n_groups": n_groups,
                "n_questions": n_questions,
                "mean_baseline": mean_a,
                "mean_candidate": mean_b,
                "mean_difference": mean_diff,
                "ci_low": low,
                "ci_high": high,
                "note": note,
            }
        )

    if n_groups == 1:
        return result(None, None, "single_group")
    flat = np.array([d for _, d in diffs], dtype=np.float64)
    if np.all(flat == flat[0]):
        value = float(flat[0])
        return result(value, value, "all_zero" if value == 0 else "constant_difference")
    # The per-question differences are resampled by group, so one draw of groups serves both
    # experiments.
    low, high = _percentile_interval(resample_groups(sums, counts, settings), settings.confidence)
    return result(low, high, None)


def _groups_for(report: RetrievalReport, groups: Mapping[str, str | None]) -> dict[str, str]:
    """The group of every scored question. A missing group is an error, never dropped."""
    result: dict[str, str] = {}
    missing = []
    for score in report.per_example:
        group = groups.get(score.example_id)
        if group is None:
            missing.append(score.example_id)
        else:
            result[score.example_id] = group
    if missing:
        raise ValueError(
            "group statistics need a group for every scored question; "
            f"{len(missing)} have none, e.g. {missing[:3]}"
        )
    return result


def group_statistics(
    name: str,
    report: RetrievalReport,
    groups: Mapping[str, str | None],
    metrics: Sequence[str],
    settings: BootstrapSettings,
) -> GroupStatistics:
    by_id = _groups_for(report, groups)
    return GroupStatistics(
        experiment=name,
        n_groups=len(set(by_id.values())),
        n_questions=len(by_id),
        n_unanswerable_excluded=report.n_unanswerable_skipped,
        bootstrap=settings,
        metrics=tuple(
            group_bootstrap_mean(metric_values(report, m), by_id, m, settings) for m in metrics
        ),
    )


def compare_groups(
    baseline: tuple[str, RetrievalReport],
    candidate: tuple[str, RetrievalReport],
    groups: Mapping[str, str | None],
    metrics: Sequence[str],
    settings: BootstrapSettings,
) -> GroupComparison:
    """Paired group bootstrap of candidate - baseline.

    Both experiments are grouped through the one `groups` mapping, and they must score exactly the
    same question ids. So a question always falls in the same group on both sides; there is no
    separate grouping per experiment that could disagree.
    """
    (name_a, report_a), (name_b, report_b) = baseline, candidate
    if name_a == name_b:
        raise ValueError("baseline and candidate must be different experiments")
    if report_a.n_unanswerable_skipped != report_b.n_unanswerable_skipped:
        raise ValueError("the experiments excluded different numbers of unanswerable questions")
    by_id = _groups_for(report_a, groups)
    return GroupComparison(
        baseline=name_a,
        candidate=name_b,
        n_groups=len(set(by_id.values())),
        n_questions=len(by_id),
        n_unanswerable_excluded=report_a.n_unanswerable_skipped,
        bootstrap=settings,
        metrics=tuple(
            group_paired_difference(
                metric_values(report_a, m), metric_values(report_b, m), by_id, m, settings
            )
            for m in metrics
        ),
    )


def format_p(p: float) -> str:
    """Four decimals, but never round a nonzero p-value down to "0.0000"."""
    return f"{p:.4f}" if p >= 0.0001 or p == 0 else f"{p:.1e}"
