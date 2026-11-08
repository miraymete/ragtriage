"""A single-file HTML report built from the results of one run. Nothing is re-run.

Every piece of dynamic text goes through `_e` (html.escape). Highlighting escapes each segment on
its own and only then wraps it in <mark>, so a highlight can never open a tag. Character offsets are
Python string indices (Unicode code points), the same as every span in ragtriage.

The default report is compact: gold evidence and retrieved passages are written only for questions
with a problem. Answers, metric outcomes and the context sent to the model are always kept. Content
left out is really left out of the file; a closed <details> would still cost its full size.
ReportOptions changes the page only, never a result.
"""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from ragtriage.evaluate import ExampleScore
from ragtriage.generation.generator import GenerationRecord, GenerationStatus
from ragtriage.generation.judge import FaithfulnessResult, JudgeCall
from ragtriage.generation.relevance import RelevanceResult
from ragtriage.generation.report import GenerationReport, Rate, StopAttempt, UsageSummary
from ragtriage.matching import covers
from ragtriage.metrics.retrieval import Failure
from ragtriage.metrics.statistics import (
    Comparison,
    ExperimentStatistics,
    GroupComparison,
    GroupStatistics,
    MeanInterval,
    PairedDifference,
    format_p,
)
from ragtriage.models import Document, Evidence, QAExample, RetrievalRun, RetrievedChunk
from ragtriage.outcomes import Failed, NotApplicable, Outcome, Scored
from ragtriage.runner import ExperimentResult


@dataclass(frozen=True)
class ReportData:
    source: str  # the config file the run used
    k: int
    depth: int
    min_overlap: float
    documents: Mapping[str, Document]
    examples: Sequence[QAExample]
    experiments: Sequence[tuple[ExperimentResult, Sequence[RetrievalRun]]]
    statistics: Sequence[ExperimentStatistics] | None = None
    comparison: Comparison | None = None
    # None: generation was not configured. Shorter than `experiments`: the run stopped.
    generation: Sequence[GenerationReport] | None = None
    group_statistics: Sequence[GroupStatistics] | None = None
    group_comparison: GroupComparison | None = None


class HtmlDetail(StrEnum):
    COMPACT = "compact"
    FULL = "full"


@dataclass(frozen=True)
class ReportOptions:
    detail: HtmlDetail = HtmlDetail.COMPACT
    # The last rank written for a question's retrieved results. None: every retrieved result.
    max_rank: int | None = None


def check_report_options(options: ReportOptions, depth: int) -> None:
    """A limit outside 1..depth could only hide everything or do nothing, so it is an error."""
    if options.max_rank is not None and not 1 <= options.max_rank <= depth:
        raise ValueError(
            f"the HTML display depth must be between 1 and the run's depth ({depth}), "
            f"got {options.max_rank}"
        )


def _e(value: object) -> str:
    return html.escape(str(value), quote=True)


def _num(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _cost(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _badge(text: str, kind: str) -> str:
    return f'<span class="badge {_e(kind)}">{_e(text)}</span>'


def _table(head: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Cells are HTML already; callers escape their text."""
    header = "".join(f"<th>{cell}</th>" for cell in head)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    table = f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"
    return f'<div class="scroll">{table}</div>'


# --- highlighting -----------------------------------------------------------------------------


def highlight(text: str, doc_id: str, start: int, evidence: Sequence[Evidence]) -> str:
    """Escape `text`, the document's characters from `start`, and mark overlaps with gold spans."""
    end = start + len(text)
    ranges = sorted(
        (max(gold.start, start) - start, min(gold.end, end) - start)
        for gold in evidence
        if gold.doc_id == doc_id and gold.start < end and gold.end > start
    )
    merged: list[list[int]] = []
    for low, high in ranges:
        if merged and low <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], high)
        else:
            merged.append([low, high])
    parts = []
    position = 0
    for low, high in merged:
        parts.append(_e(text[position:low]))
        parts.append(f"<mark>{_e(text[low:high])}</mark>")
        position = high
    parts.append(_e(text[position:]))
    return "".join(parts)


# --- summaries --------------------------------------------------------------------------------


def _retrieval_summary(data: ReportData) -> str:
    k = data.k
    first = data.experiments[0][0].report
    rows = []
    for result, _ in data.experiments:
        mean, failures = result.report.mean, result.report.failures
        rows.append(
            [
                _e(result.name),
                str(result.n_chunks),
                str(len(result.report.per_example)),
                _num(mean[f"recall@{k}"]),
                _num(mean[f"precision@{k}"]),
                _num(mean["mrr"]),
                _num(mean[f"ndcg@{k}"]),
                f"{result.context_chars:.0f}",
                str(failures.get(Failure.OK, 0)),
                str(failures.get(Failure.RANKING, 0)),
                str(failures.get(Failure.RETRIEVAL, 0)),
            ]
        )
    head = [
        "Experiment",
        "Chunks",
        "Scored questions",
        f"Recall@{k}",
        f"Precision@{k}",
        "MRR",
        f"nDCG@{k}",
        f"Context chars@{k}",
        "OK",
        "Ranking",
        "Retrieval",
    ]
    return (
        "<section><h2>Retrieval</h2>"
        f"<p>{len(first.per_example)} questions scored; {first.n_unanswerable_skipped} labeled "
        "unanswerable from this corpus and left out of every retrieval metric. "
        f"k = {k}, depth = {data.depth}, min_overlap = {data.min_overlap}.</p>"
        + _table(head, rows)
        + "</section>"
    )


def _interval(item: MeanInterval | PairedDifference) -> str:
    if item.ci_low is None or item.ci_high is None:
        return "no interval (n=1)"
    return f"[{item.ci_low:.3f}, {item.ci_high:.3f}]"


_NOTES = {
    "constant": "all values equal: zero width describes this sample, not new questions",
    "single_question": "one question: no uncertainty estimate",
    "all_zero": "every question scored the same in both experiments",
    "constant_difference": "same difference on every question: zero width describes this "
    "sample, not new questions",
}


def _statistics(data: ReportData) -> str:
    parts = []
    if data.statistics:
        settings = data.statistics[0].bootstrap
        metrics = [m.metric for m in data.statistics[0].metrics]
        rows = []
        for stat in data.statistics:
            cells = []
            for m in stat.metrics:
                note = f'<div class="muted">{_e(_NOTES[m.note])}</div>' if m.note else ""
                cells.append(f"{m.mean:.3f} {_e(_interval(m))}{note}")
            rows.append([_e(stat.experiment), str(stat.n_questions), *cells])
        parts.append(
            f"<h3>Mean with {settings.confidence:.0%} interval</h3>"
            f"<p>Percentile bootstrap over questions: {settings.n_resamples} resamples, seed "
            f"{settings.seed}. {data.statistics[0].n_unanswerable_excluded} unanswerable "
            "questions excluded.</p>"
            + _table(["Experiment", "Questions", *(_e(m) for m in metrics)], rows)
        )
    comparison = data.comparison
    if comparison is not None:
        rows = []
        for d in comparison.metrics:
            test = d.permutation
            if test is None:
                p = "n/a (n=1)"
            elif test.method == "exact":
                p = f"{format_p(test.p_value)} (exact, {test.n_extreme}/{test.n_patterns} patterns)"
            else:
                p = (
                    f"{format_p(test.p_value)} (Monte Carlo, {test.n_patterns} draws, "
                    f"seed {test.seed}, p = (extreme + 1) / (draws + 1))"
                )
            note = f'<div class="muted">{_e(_NOTES[d.note])}</div>' if d.note else ""
            rows.append(
                [
                    _e(d.metric),
                    f"{d.mean_baseline:.3f}",
                    f"{d.mean_candidate:.3f}",
                    f"{d.mean_difference:+.3f}",
                    _e(_interval(d)) + note,
                    f"{d.n_candidate_higher} / {d.n_candidate_lower} / {d.n_tied}",
                    _e(p),
                ]
            )
        b = comparison.bootstrap
        parts.append(
            f"<h3>Paired comparison: {_e(comparison.candidate)} − {_e(comparison.baseline)}</h3>"
            f"<p>Difference is candidate − baseline, matched by question id over "
            f"{comparison.n_questions} questions ({comparison.n_unanswerable_excluded} "
            f"unanswerable excluded). Interval: paired percentile bootstrap, {b.confidence:.0%}, "
            f"{b.n_resamples} resamples, seed {b.seed}. p-value: two-sided sign-flip permutation "
            "test, <strong>unadjusted</strong>.</p>"
            + _table(
                [
                    "Metric",
                    "Baseline mean",
                    "Candidate mean",
                    "Difference",
                    f"{b.confidence:.0%} CI",
                    "Higher / lower / tied",
                    "Unadjusted two-sided p",
                ],
                rows,
            )
        )
    if not parts:
        return ""
    return (
        "<section><h2>Question-level uncertainty</h2>"
        + "".join(parts)
        + '<div class="note"><p>The interval and the p-value come from different methods. The '
        "bootstrap resamples questions; the permutation test flips the sign of each question's "
        "difference. With few questions, or few questions that differ, they can disagree: an "
        "interval can exclude zero while p stays large, because an exact test on m nonzero "
        "differences cannot go below 2 / 2<sup>m</sup>. Percentile intervals also tend to be "
        "too narrow on small samples.</p>"
        "<p>p-values are not adjusted for multiple comparisons. Looking at several metrics or "
        "choosing one pair of experiments out of many raises the chance that a difference is "
        "only luck. No result here names a winner or shows that two experiments are equivalent."
        "</p><p>Everything assumes the questions are independent and represent the questions "
        "the system will really face. Questions about the same document are not independent, "
        "and a small self-written test set is not a sample of real use. Resampling fixes "
        "neither.</p></div></section>"
    )


def _group_interval(low: float | None, high: float | None) -> str:
    return "no interval (1 group)" if low is None or high is None else f"[{low:.3f}, {high:.3f}]"


def _group_section(data: ReportData) -> str:
    stats, comparison = data.group_statistics, data.group_comparison
    if not stats:
        return ""
    first = stats[0]
    b = first.bootstrap
    metrics = [m.metric for m in first.metrics]
    parts = [
        "<section><h2>Group-level uncertainty</h2>",
        f"<p>Group percentile bootstrap: whole groups are resampled with replacement, with all "
        f"their questions. The mean stays question-weighted, as in the tables above. "
        f"{first.n_groups} groups, {first.n_questions} questions, "
        f"{first.n_unanswerable_excluded} unanswerable excluded; {b.confidence:.0%}, "
        f"{b.n_resamples} resamples, seed {b.seed}.</p>",
        _table(
            ["Experiment", "Groups", "Questions", *(_e(m) for m in metrics)],
            [
                [
                    _e(s.experiment),
                    str(s.n_groups),
                    str(s.n_questions),
                    *(
                        f"{m.mean:.3f} {_e(_group_interval(m.ci_low, m.ci_high))}"
                        for m in s.metrics
                    ),
                ]
                for s in stats
            ],
        ),
    ]
    if comparison is not None:
        parts.append(
            f"<h3>Paired group comparison: {_e(comparison.candidate)} − "
            f"{_e(comparison.baseline)}</h3><p>Difference is candidate − baseline. Both "
            "experiments use the same group draws. There is no p-value for this comparison.</p>"
        )
        parts.append(
            _table(
                [
                    "Metric",
                    "Baseline mean",
                    "Candidate mean",
                    "Difference",
                    f"{b.confidence:.0%} CI",
                ],
                [
                    [
                        _e(d.metric),
                        f"{d.mean_baseline:.3f}",
                        f"{d.mean_candidate:.3f}",
                        f"{d.mean_difference:+.3f}",
                        _e(_group_interval(d.ci_low, d.ci_high)),
                    ]
                    for d in comparison.metrics
                ],
            )
        )
    parts.append(
        '<div class="note"><p>This allows for questions in one group not being independent. It '
        "does not make a few groups many, or make them represent real use. Any question-level "
        "p-value elsewhere on this page comes from a different method and is not part of this "
        "interval.</p></div></section>"
    )
    return "".join(parts)


def _rate(rate: Rate) -> str:
    return "n/a" if rate.value is None else f"{rate.numerator}/{rate.denominator}"


def _usage_row(
    role: str,
    usage: UsageSummary | GenerationReport,
) -> list[str]:
    billed, cached = usage.billed_usage, usage.cached_original_usage
    return [
        _e(role),
        str(usage.n_api_calls),
        f"{billed.input_tokens} / {billed.output_tokens}",
        str(usage.n_cached),
        f"{cached.input_tokens} / {cached.output_tokens}",
        str(usage.n_fake),
        str(usage.n_usage_unknown),
        _cost(usage.cost_usd),
    ]


def _counts(counts: Mapping[str, int]) -> str:
    if not counts:
        return "0"
    total = sum(counts.values())
    return f"{total} (" + ", ".join(f"{_e(k)} {v}" for k, v in sorted(counts.items())) + ")"


def _generation_summary(data: ReportData) -> str:
    reports = data.generation
    if reports is None:
        return ""
    parts = ["<section><h2>Generation</h2>"]
    parts.append(
        f"<p>The generator saw the top {data.k} chunks of each question. Answers are not checked "
        "for correctness, and the judges are not validated against human labels.</p>"
    )
    rows = []
    usage_rows = []
    for index, (result, _) in enumerate(data.experiments):
        if index >= len(reports):
            rows.append(
                [_e(result.name), _badge("not started", "warn")] + ["—"] * 6,
            )
            continue
        r = reports[index]
        status = _badge("stopped", "bad") if r.stopped else _badge("finished", "ok")
        failed = {str(k): v for k, v in r.failures.items()}
        rows.append(
            [
                _e(result.name),
                status,
                f"{r.n_evaluated} evaluated / {r.n_attempted} attempted / {r.n_examples} total",
                _counts(failed),
                _rate(r.abstained_when_unanswerable),
                _rate(r.abstained_when_answerable),
                _rate(r.abstained_when_answerable_evidence_present),
                _rate(r.abstained_when_answerable_evidence_missing),
            ]
        )
        usage_rows.append(_usage_row(f"{result.name}: generator", r))
        stop = r.stop_attempt
        if stop is not None and stop.sent_to_api:
            role = stop.role.replace("_", " ")
            usage_rows.append(
                [
                    _e(f"{result.name}: {role}, call that stopped the run ({stop.example_id})"),
                    "0 (no response)",
                    "unknown",
                    "0",
                    "—",
                    "0",
                    "1",
                    "unknown",
                ]
            )
        if r.faithfulness is not None:
            usage_rows.append(
                _usage_row(f"{result.name}: faithfulness judge", r.faithfulness.judge_usage)
            )
        if r.relevance is not None:
            usage_rows.append(
                _usage_row(f"{result.name}: relevance judge", r.relevance.judge_usage)
            )
    parts.append(
        _table(
            [
                "Experiment",
                "Run",
                "Questions",
                "Failed generations",
                "Abstained, unanswerable",
                "Abstained, answerable",
                "… evidence in context",
                "… evidence missing",
            ],
            rows,
        )
    )
    for index, r in enumerate(reports):
        if r.stopped:
            name = data.experiments[index][0].name
            parts.append(
                f'<div class="note bad"><strong>Stopped in {_e(name)}:</strong> {_e(r.stopped)}. '
                f"{r.n_attempted} of {r.n_examples} questions have a result; the rest were not "
                "started. Later experiments were not started either.</div>"
            )
    faith = [(i, r.faithfulness) for i, r in enumerate(reports) if r.faithfulness is not None]
    if faith:
        parts.append(
            "<h3>Faithfulness</h3><p>Share of an answer's claims supported by the chunks the "
            "generator saw. It does not measure correctness, completeness or relevance.</p>"
        )
        parts.append(
            _table(
                [
                    "Experiment",
                    "Mean of scored",
                    "Scored",
                    "Not applicable",
                    "Failed",
                    "Claims supported / unsupported / uncertain",
                ],
                [
                    [
                        _e(data.experiments[i][0].name),
                        _num(f.mean),
                        str(f.n_scored),
                        _counts(f.not_applicable),
                        str(f.n_failed),
                        f"{f.n_supported} / {f.n_unsupported} / {f.n_uncertain}",
                    ]
                    for i, f in faith
                ],
            )
        )
    rel = [(i, r.relevance) for i, r in enumerate(reports) if r.relevance is not None]
    if rel:
        parts.append(
            "<h3>Answer relevance</h3><p>Mean cosine similarity, in [−1, 1], between the "
            "question and three questions written from the answer alone. A semantic-similarity "
            "proxy, not a check of correctness or completeness.</p>"
        )
        parts.append(
            _table(
                ["Experiment", "Mean of scored", "Scored", "Not applicable", "Failed"],
                [
                    [
                        _e(data.experiments[i][0].name),
                        _num(v.mean),
                        str(v.n_scored),
                        _counts(v.not_applicable),
                        str(v.n_failed),
                    ]
                    for i, v in rel
                ],
            )
        )
    parts.append(
        "<h3>Usage and cost</h3><p>Each role is counted once, on its own row. Billed means new "
        "API calls in this run. Cache hits add no new tokens; their original usage is shown "
        "apart. Fake responses come from a scripted test model and are never billed. Cost is "
        "an estimate and is n/a without a configured price.</p>"
    )
    parts.append(
        _table(
            [
                "Role",
                "API calls with a response",
                "Billed input / output tokens",
                "Cache hits",
                "Cached original input / output",
                "Fake",
                "Usage unknown",
                "Est. cost (USD)",
            ],
            usage_rows,
        )
    )
    parts.append("</section>")
    return "".join(parts)


# --- which questions get passages -------------------------------------------------------------

# Why a judge did not score an answer. These are different things and are shown apart:
# the model abstained; generation failed, so there was nothing to judge; or the answer existed and
# could not be evaluated (every other reason, including one added later).
_ABSTAINED = "abstained"
_GENERATION_FAILED = "no_valid_answer"


@dataclass(frozen=True)
class Issue:
    kind: str  # retrieval, ranking, generation or judge
    label: str


def _judge_issue(name: str, outcome: Outcome) -> Issue | None:
    if isinstance(outcome, Failed):
        return Issue("judge", f"{name} failed")
    if isinstance(outcome, NotApplicable) and outcome.reason not in (
        _ABSTAINED,
        _GENERATION_FAILED,  # already reported as the failed generation itself
    ):
        return Issue("judge", f"{name} not evaluated: {outcome.reason}")
    return None


def question_issues(
    score: ExampleScore | None,
    record: GenerationRecord | None,
    stop: StopAttempt | None = None,
) -> tuple[Issue, ...]:
    """Why a compact report writes this question's gold evidence and retrieved passages."""
    issues: list[Issue | None] = []
    if score is not None and score.failure is not Failure.OK:
        issues.append(Issue(score.failure.value, score.failure.value))
    if stop is not None:
        kind = "generation" if stop.role == "generator" else "judge"
        issues.append(Issue(kind, "stopped the run here"))
    if record is not None:
        if record.status is not GenerationStatus.OK:
            issues.append(Issue("generation", f"generation {record.status.value}"))
        elif isinstance(record.abstention, Scored) and record.abstention.value < 1:
            wrong = "abstained on answerable" if record.answerable else "answered unanswerable"
            issues.append(Issue("generation", wrong))
        if record.faithfulness is not None:
            outcome = record.faithfulness.outcome
            if isinstance(outcome, Scored) and outcome.value < 1:
                issues.append(Issue("generation", "unsupported claims"))
            issues.append(_judge_issue("faithfulness", outcome))
        if record.relevance is not None:
            issues.append(_judge_issue("relevance", record.relevance.outcome))
    return tuple(issue for issue in issues if issue is not None)


@dataclass(frozen=True)
class _Plan:
    state: str  # off, on or not_started
    records: Mapping[str, GenerationRecord]
    issues: Mapping[str, tuple[Issue, ...]]
    detailed: frozenset[str]


def _plan(data: ReportData, index: int, options: ReportOptions) -> _Plan:
    result, _ = data.experiments[index]
    reports = data.generation
    state = "off" if reports is None else "on" if index < len(reports) else "not_started"
    records: dict[str, GenerationRecord] = {}
    stop: StopAttempt | None = None
    if reports is not None and state == "on":
        records = {r.example_id: r for r in reports[index].records}
        stop = reports[index].stop_attempt
    scores = {s.example_id: s for s in result.report.per_example}
    issues = {}
    for example in data.examples:
        here = stop if stop is not None and stop.example_id == example.id else None
        found = question_issues(scores.get(example.id), records.get(example.id), here)
        if found:
            issues[example.id] = found
    full = options.detail is HtmlDetail.FULL
    detailed = frozenset(e.id for e in data.examples if full or e.id in issues)
    return _Plan(state, records, issues, detailed)


def _judge_not_applicable(records: Mapping[str, GenerationRecord]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {"abstained": {}, "failed": {}, "not_evaluated": {}}
    for record in records.values():
        for name, item in (("faithfulness", record.faithfulness), ("relevance", record.relevance)):
            outcome = None if item is None else item.outcome
            if not isinstance(outcome, NotApplicable):
                continue
            group = {_ABSTAINED: "abstained", _GENERATION_FAILED: "failed"}.get(
                outcome.reason, "not_evaluated"
            )
            key = f"{name}: {outcome.reason}"
            counts[group][key] = counts[group].get(key, 0) + 1
    return counts


def _detail_section(data: ReportData, options: ReportOptions) -> str:
    rows = []
    not_applicable_rows = []
    for index, (result, runs) in enumerate(data.experiments):
        plan = _plan(data, index, options)
        by_kind = {
            kind: sum(any(i.kind == kind for i in found) for found in plan.issues.values())
            for kind in ("retrieval", "ranking", "generation", "judge")
        }
        written = hidden = omitted = 0
        for run in runs:
            if run.example_id not in plan.detailed:
                omitted += len(run.chunks)
                continue
            shown = _n_shown(run, options.max_rank)
            written += shown
            hidden += len(run.chunks) - shown
        rows.append(
            [
                _e(result.name),
                str(len(data.examples)),
                str(len(plan.detailed)),
                str(len(data.examples) - len(plan.detailed)),
                *(str(count) for count in by_kind.values()),
                str(written),
                str(hidden),
                str(omitted),
            ]
        )
        groups = _judge_not_applicable(plan.records)
        not_applicable_rows.append(
            [
                _e(result.name),
                _counts(groups["abstained"]),
                _counts(groups["failed"]),
                _counts(groups["not_evaluated"]),
            ]
        )
    head = [
        "Experiment",
        "Questions",
        "With retrieval passages",
        "Without retrieval passages",
        "Retrieval problems",
        "Ranking problems",
        "Generation problems",
        "Judge problems",
        "Retrieved results written",
        "Hidden by display depth",
        "Left out of questions without passages",
    ]
    if options.detail is HtmlDetail.FULL:
        rule = (
            "<p><strong>Full detail.</strong> Every question has its passages, gold evidence and "
            "any generation result.</p>"
        )
    else:
        rule = (
            "<p><strong>Compact report.</strong> Every question stays on this page with its "
            "scores. When generation ran, every question also keeps its answer or abstention, "
            "its metric outcomes, its judge details and the complete context sent to the model. "
            "Only gold evidence and retrieved passages are limited. They are written for a "
            "question with at least one of: a retrieval or ranking failure; a failed generation; "
            "an abstention that disagrees with the label; unsupported claims; a failed judge; a "
            "judge result that could not be evaluated; the call that stopped the run. The summary "
            "tables above use every question.</p>"
            "<p>Labeled unanswerable questions are not retrieval failures. Answer relevance has "
            "no threshold, so a low score alone selects nothing. For every question's passages, "
            "use <code>--html-detail full</code>.</p>"
        )
    if options.max_rank is not None:
        rule += (
            f"<p>Display depth: at most {options.max_rank} retrieved results are written per "
            f"question. Evaluation still used k = {data.k} and depth = {data.depth}, and the JSON "
            "results are unchanged. The context actually sent to the generator is never cut by "
            "this limit.</p>"
        )
    section = f'<h3>What this page includes</h3><div class="note">{rule}</div>' + _table(head, rows)
    if data.generation is not None:
        section += (
            "<h4>Why a judge gave no score</h4>"
            '<p class="muted">Counted per judge result. The first column is an abstention, the '
            "second follows a failed generation, and the third means an answer existed but could "
            "not be evaluated.</p>"
            + _table(
                [
                    "Experiment",
                    "Model abstained",
                    "Generation failed: nothing to judge",
                    "Could not be evaluated",
                ],
                not_applicable_rows,
            )
        )
    return section


# --- per question -----------------------------------------------------------------------------


def _evidence_block(example: QAExample, documents: Mapping[str, Document]) -> str:
    items = []
    for number, gold in enumerate(example.evidence, start=1):
        text = documents[gold.doc_id].text[gold.start : gold.end]
        items.append(
            f'<li><span class="span">#{number} {_e(gold.doc_id)} [{gold.start}, {gold.end}) '
            f"grade {gold.grade}</span><pre>{_e(text)}</pre></li>"
        )
    return '<h4>Gold evidence</h4><ol class="plain">' + "".join(items) + "</ol>"


def _chunks(
    ranked: Sequence[tuple[int, RetrievedChunk]],
    example: QAExample,
    min_overlap: float,
    seen: dict[tuple[str, int, int], int],
    k: int,
) -> str:
    rows = []
    for rank, chunk in ranked:
        key = (chunk.doc_id, chunk.start, chunk.end)
        first = seen.setdefault(key, rank)
        covered = [
            f"#{i}"
            for i, gold in enumerate(example.evidence, start=1)
            if covers(chunk, gold, min_overlap)
        ]
        tags = []
        if covered:
            tags.append(_badge("covers " + ", ".join(covered), "ok"))
        if first != rank:
            tags.append(_badge(f"duplicate of rank {first}, ignored by metrics", "warn"))
            if first <= k < rank:
                tags.append(
                    _badge(f"same chunk is in the generation context at rank {first}", "na")
                )
        rows.append(
            f'<li><span class="span">rank {rank} · {_e(chunk.doc_id)} '
            f"[{chunk.start}, {chunk.end}) · score {chunk.score:.4f}</span> {''.join(tags)}"
            f"<pre>{highlight(chunk.text, chunk.doc_id, chunk.start, example.evidence)}</pre></li>"
        )
    return '<ol class="plain">' + "".join(rows) + "</ol>"


def _n_shown(run: RetrievalRun, max_rank: int | None) -> int:
    return len(run.chunks) if max_rank is None else min(len(run.chunks), max_rank)


def _retrieved(
    run: RetrievalRun, example: QAExample, k: int, min_overlap: float, max_rank: int | None
) -> str:
    seen: dict[tuple[str, int, int], int] = {}
    n_shown = _n_shown(run, max_rank)
    ranked = list(enumerate(run.chunks[:n_shown], start=1))
    top, deep = ranked[:k], ranked[k:]
    parts = [
        f"<h4>Top {k} (what the generator is given)</h4>",
        _chunks(top, example, min_overlap, seen, k) if top else "<p>Nothing retrieved.</p>",
    ]
    if deep:
        parts.append(
            f"<details><summary>Deeper results, ranks {k + 1}–{len(run.chunks)} "
            "(retrieval triage only: results at these ranks were not added to the generation "
            "context)</summary>" + _chunks(deep, example, min_overlap, seen, k) + "</details>"
        )
    hidden = len(run.chunks) - n_shown
    if hidden:
        parts.append(
            f'<p class="muted">{hidden} more retrieved results not written here (ranks '
            f"{n_shown + 1}–{len(run.chunks)}): the display depth is {max_rank}. Metrics still "
            "use them.</p>"
        )
    return "".join(parts)


def _outcome(outcome: Outcome) -> str:
    if isinstance(outcome, Scored):
        return _badge(f"scored {outcome.value:.3f}", "ok")
    if isinstance(outcome, NotApplicable):
        if outcome.reason == _ABSTAINED:
            return _badge(f"N/A: {outcome.reason}", "na")
        if outcome.reason == _GENERATION_FAILED:
            return _badge(f"not run: {outcome.reason}", "bad")
        return _badge(f"not evaluated: {outcome.reason}", "warn")
    assert isinstance(outcome, Failed)
    return _badge("failed", "bad") + f" <span>{_e(outcome.reason)}</span>"


def _origin(source: str | None, original: str | None) -> str:
    if source is None:
        return _badge("no response", "na")
    if source == "fake" or original == "fake":
        label = "FAKE: scripted test model" + (" (from cache)" if source == "cache" else "")
        return _badge(label, "fake")
    if source == "cache":
        return _badge("cache (originally a real API call)", "na")
    return _badge("real API call", "na")


def _call(call: JudgeCall) -> str:
    usage = call.usage
    tokens = "unknown" if usage is None else f"{usage.input_tokens} in / {usage.output_tokens} out"
    problem = f" · {_e(call.problem)}" if call.problem else ""
    return (
        f"<li>{_e(call.stage)} · {_e(call.provider)}/{_e(call.model)} · "
        f"{_e(call.prompt_version)} · {_e(call.status)}{problem} · tokens {_e(tokens)} · "
        f"cost {_cost(call.cost_usd)} {_origin(call.source, call.original_source)}</li>"
    )


def _faithfulness(result: FaithfulnessResult) -> str:
    claims = "".join(
        f"<li><strong>{_e(c.id)}</strong> {_e(c.text)} → "
        f"{_e(c.verdict.value if c.verdict else 'no verdict')}"
        + (f" · passage {_e(c.passage_id)}" if c.passage_id else "")
        + (f"<pre>{_e(c.quote)}</pre>" if c.quote else "")
        + "</li>"
        for c in result.claims
    )
    calls = "".join(_call(c) for c in result.calls)
    return (
        f"<h4>Faithfulness {_outcome(result.outcome)}</h4>"
        + (f"<ul>{claims}</ul>" if claims else "")
        + (f'<ul class="muted">{calls}</ul>' if calls else "")
    )


def _relevance(result: RelevanceResult) -> str:
    sims = result.similarities
    pairs = "".join(
        f"<li>{_e(q)}" + (f" · cosine {sims[i]:.3f}" if i < len(sims) else "") + "</li>"
        for i, q in enumerate(result.questions)
    )
    embedder = f'<p class="muted">Embedder: {_e(result.embedder)}</p>' if result.embedder else ""
    call = f'<ul class="muted">{_call(result.call)}</ul>' if result.call else ""
    return (
        f"<h4>Answer relevance {_outcome(result.outcome)}</h4>"
        + (f"<ol>{pairs}</ol>" if pairs else "")
        + embedder
        + call
    )


def _generation(
    record: GenerationRecord | None,
    example: QAExample,
    documents: Mapping[str, Document],
    state: str,
    inline: bool = False,
) -> str:
    """With `inline`, the answer is already on the page and the context needs no second click."""
    if state == "off":
        return ""
    if record is None:
        return (
            "<h4>Generation</h4><p>"
            + _badge("no result", "warn")
            + " The run stopped before this question finished, so it has no answer.</p>"
        )
    status = (
        _badge("ok", "ok")
        if record.status is GenerationStatus.OK
        else _badge(f"failed: {record.status.value}", "bad")
    )
    lines = [
        f"<h4>Generation {status} {_origin(record.source, record.original_source)}</h4>",
        f'<p class="muted">{_e(record.provider)}/{_e(record.model)} · {_e(record.prompt_version)}'
        f" · latency {record.latency_s:.3f}s · cost {_cost(record.cost_usd)}</p>",
    ]
    if record.status is GenerationStatus.OK:
        abstained = "abstained" if record.abstained else "answered"
        lines.append(f"<p>{_badge(abstained, 'na')} Abstention: {_outcome(record.abstention)}</p>")
        if record.answer and not inline:
            lines.append(f"<pre>{_e(record.answer)}</pre>")
    else:
        lines.append(f"<p>Problem: {_e(record.problem or 'unknown')}</p>")
        if record.raw_output:
            lines.append(
                f"<details><summary>Raw model output</summary><pre>{_e(record.raw_output)}</pre>"
                "</details>"
            )
    if record.evidence_in_context is not None:
        lines.append(
            f"<p>Gold evidence in the context sent: {'yes' if record.evidence_in_context else 'no'}"
            "</p>"
        )
    context = "".join(
        f'<li><span class="span">rank {c.rank} · {_e(c.doc_id)} [{c.start}, {c.end})</span><pre>'
        + highlight(documents[c.doc_id].text[c.start : c.end], c.doc_id, c.start, example.evidence)
        + "</pre></li>"
        for c in record.contexts
    )
    heading = f"Context actually sent to the model ({len(record.contexts)} chunks)"
    listed = f'<ol class="plain">{context}</ol>'
    lines.append(
        f"<h4>{heading}</h4>{listed}"
        if inline
        else f"<details><summary>{heading}</summary>{listed}</details>"
    )
    if record.faithfulness is not None:
        lines.append(_faithfulness(record.faithfulness))
    if record.relevance is not None:
        lines.append(_relevance(record.relevance))
    return "".join(lines)


def _generation_brief(record: GenerationRecord | None, state: str) -> str:
    """Status, abstention and every metric outcome, with no passage text."""
    if state == "not_started":
        return _badge("not started", "warn")
    if record is None:
        return _badge("no generation result", "warn")
    parts = []
    if record.status is GenerationStatus.OK:
        parts.append(_badge("generation ok", "ok"))
        parts.append(_badge("abstained" if record.abstained else "answered", "na"))
    if record.faithfulness is not None:
        parts.append(f"faithfulness {_outcome(record.faithfulness.outcome)}")
    if record.relevance is not None:
        parts.append(f"relevance {_outcome(record.relevance.outcome)}")
    if "fake" in (record.source, record.original_source):
        parts.append(_badge("FAKE", "fake"))
    return " ".join(parts)


def _question_line(
    example: QAExample,
    score: ExampleScore | None,
    record: GenerationRecord | None,
    state: str,
    k: int,
    issues: Sequence[Issue],
) -> str:
    if score is not None:
        failure = score.failure
        kind = {"ok": "ok", "ranking": "warn", "retrieval": "bad"}[failure.value]
        line = (
            f"{_badge(failure.value, kind)} <strong>{_e(example.id)}</strong> · recall@{k} "
            f"{score.scores[f'recall@{k}']:.3f} · RR {score.scores['mrr']:.3f}"
        )
    else:
        line = f"{_badge('labeled unanswerable', 'na')} <strong>{_e(example.id)}</strong>"
    if state == "on":
        line += " · " + _generation_brief(record, state)
    # Retrieval and ranking already have their badge at the front of the line.
    return line + "".join(
        " " + _badge(i.label, "bad") for i in issues if i.kind in ("generation", "judge")
    )


def _gold_block(example: QAExample, documents: Mapping[str, Document]) -> str:
    if example.answerable:
        return _evidence_block(example, documents)
    return (
        "<p>No evidence is labeled for this question: it is labeled unanswerable from this "
        "corpus. It is not a retrieval failure and is not in any retrieval metric.</p>"
    )


def _reference(example: QAExample) -> str:
    if not example.reference_answer:
        return ""
    return f"<h4>Reference answer</h4><pre>{_e(example.reference_answer)}</pre>"


def _question(
    example: QAExample,
    score: ExampleScore | None,
    run: RetrievalRun,
    record: GenerationRecord | None,
    state: str,
    data: ReportData,
    issues: Sequence[Issue],
    max_rank: int | None,
) -> str:
    """A full block: everything about the question, collapsed behind one summary line."""
    summary = _question_line(example, score, record, state, data.k, issues)
    body = [
        f'<p class="question">{_e(example.question)}</p>',
        _gold_block(example, data.documents),
        _reference(example),
        _retrieved(run, example, data.k, data.min_overlap, max_rank),
        _generation(record, example, data.documents, state),
    ]
    return f'<details class="q"><summary>{summary}</summary>{"".join(body)}</details>'


def _visible_answer(record: GenerationRecord | None) -> str:
    if record is None:
        return ""
    if record.status is not GenerationStatus.OK:
        return f"<p>No valid answer: {_e(record.problem or record.status.value)}</p>"
    if record.answer:
        return f"<pre>{_e(record.answer)}</pre>"
    return (
        "<p>No answer text: the model abstained.</p>" if record.abstained else "<p>No answer.</p>"
    )


def _answer_block(
    example: QAExample,
    score: ExampleScore | None,
    run: RetrievalRun,
    record: GenerationRecord | None,
    data: ReportData,
    issues: Sequence[Issue],
    max_rank: int | None,
    with_passages: bool,
) -> str:
    """A compact block when generation ran: the outcome and answer stay visible without a click."""
    line = _question_line(example, score, record, "on", data.k, issues)
    head = f'<div class="q"><p>{line}</p><p class="question">{_e(example.question)}</p>'
    head += _visible_answer(record)
    inner = [_reference(example)]
    label = "Context sent to the model and generation details"
    if with_passages:
        label = "Gold evidence, retrieved passages, " + label[0].lower() + label[1:]
        inner.insert(0, _gold_block(example, data.documents))
        inner.append(_retrieved(run, example, data.k, data.min_overlap, max_rank))
    if record is not None or with_passages:
        inner.append(_generation(record, example, data.documents, "on", inline=True))
        head += f"<details><summary>{label}</summary>{''.join(inner)}</details>"
    return head + "</div>"


def _summary_rows(
    examples: Sequence[QAExample], scores: Mapping[str, ExampleScore], plan: _Plan, k: int
) -> str:
    has_generation = plan.state != "off"
    rows = []
    for example in examples:
        score = scores.get(example.id)
        if score is None:
            cells = [_badge("labeled unanswerable", "na"), _e(example.id), "—", "—"]
        else:
            cells = [
                _badge(score.failure.value, "ok"),
                _e(example.id),
                f"{score.scores[f'recall@{k}']:.3f}",
                f"{score.scores['mrr']:.3f}",
            ]
        if has_generation:
            cells.append(_generation_brief(plan.records.get(example.id), plan.state))
        cells.append(_e(example.question))
        rows.append(cells)
    head = ["Result", "Question id", f"Recall@{k}", "RR"]
    head += ["Generation", "Question"] if has_generation else ["Question"]
    return _table(head, rows)


def _question_blocks(
    examples: Sequence[QAExample],
    scores: Mapping[str, ExampleScore],
    runs_by_id: Mapping[str, RetrievalRun],
    plan: _Plan,
    data: ReportData,
    options: ReportOptions,
) -> list[str]:
    def block(example: QAExample) -> str:
        args = (
            example,
            scores.get(example.id),
            runs_by_id[example.id],
            plan.records.get(example.id),
        )
        issues = plan.issues.get(example.id, ())
        if options.detail is HtmlDetail.COMPACT and plan.state == "on":
            detailed = example.id in plan.detailed
            return _answer_block(*args, data, issues, options.max_rank, detailed)
        return _question(*args, plan.state, data, issues, options.max_rank)

    detailed = [e for e in examples if e.id in plan.detailed]
    rest = [e for e in examples if e.id not in plan.detailed]
    blocks = [block(e) for e in detailed]
    if rest:
        blocks.append(f"<h4>Without retrieval passages ({len(rest)})</h4>")
        if plan.state == "on":
            blocks.extend(block(e) for e in rest)
        else:
            blocks.append(_summary_rows(rest, scores, plan, data.k))
    return blocks


def _experiment_questions(data: ReportData, index: int, options: ReportOptions) -> str:
    result, runs = data.experiments[index]
    runs_by_id = {run.example_id: run for run in runs}
    scores = {s.example_id: s for s in result.report.per_example}
    plan = _plan(data, index, options)
    k = data.k
    answerable = sorted(
        (e for e in data.examples if e.id in scores),
        key=lambda e: (scores[e.id].scores[f"recall@{k}"], scores[e.id].scores["mrr"], e.id),
    )
    unanswerable = sorted((e for e in data.examples if not e.answerable), key=lambda e: e.id)
    parts = [f'<details class="exp"><summary><strong>{_e(result.name)}</strong></summary>']
    if plan.state == "not_started":
        parts.append(
            '<div class="note bad">Generation was not started for this experiment: the run '
            "stopped in an earlier one.</div>"
        )
    parts.append(
        f'<h3>Answerable questions ({len(answerable)})</h3><p class="muted">Sorted by '
        f"recall@{k} (lowest first), then reciprocal rank (lowest first), then question id.</p>"
    )
    parts.extend(_question_blocks(answerable, scores, runs_by_id, plan, data, options))
    if unanswerable:
        parts.append(
            f'<h3>Labeled unanswerable ({len(unanswerable)})</h3><p class="muted">No evidence '
            "is labeled in this corpus. These are excluded from retrieval metrics. Abstaining "
            "on them is the expected behavior. Sorted by question id.</p>"
        )
        parts.extend(_question_blocks(unanswerable, scores, runs_by_id, plan, data, options))
    parts.append("</details>")
    return "".join(parts)


# --- page -------------------------------------------------------------------------------------


_CSS = """
:root{--bg:#fbfbfa;--fg:#1d1d1b;--muted:#5f5f5a;--line:#d9d8d2;--card:#ffffff;
--ok:#1f6f43;--ok-bg:#e3f2e8;--warn:#7a5200;--warn-bg:#fbefd5;--bad:#9b1c1c;--bad-bg:#fbe3e3;
--na:#3d4a5c;--na-bg:#e8ecf2;--fake:#6b21a8;--fake-bg:#f1e5fb;--mark:#ffe58a;--code:#f3f2ee}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#e9e8e3;--muted:#a3a29b;
--line:#3a3935;--card:#1f1f1d;--ok:#8fd6a8;--ok-bg:#1d3326;--warn:#f1c76b;--warn-bg:#3a2f14;
--bad:#f3a1a1;--bad-bg:#3d1d1d;--na:#b9c4d4;--na-bg:#262d38;--fake:#d8b4fe;--fake-bg:#33204a;
--mark:#6b5a12;--code:#262624}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,
"Segoe UI",Roboto,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:1.6rem;margin:0 0 4px}h2{margin-top:2rem;border-bottom:1px solid var(--line);
padding-bottom:4px}h3{margin-top:1.4rem}h4{margin:1rem 0 .3rem}
.muted{color:var(--muted)}.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:.9rem;background:var(--card)}
th,td{border:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
th{background:var(--code)}
pre{white-space:pre-wrap;word-break:break-word;background:var(--code);padding:8px;
border-radius:6px;margin:4px 0;font:13px/1.45 ui-monospace,Consolas,monospace}
mark{background:var(--mark);color:inherit;border-radius:2px}
.badge{display:inline-block;padding:1px 7px;border-radius:999px;font-size:.78rem;
font-weight:600;margin-right:4px}
.ok{color:var(--ok);background:var(--ok-bg)}.warn{color:var(--warn);background:var(--warn-bg)}
.bad{color:var(--bad);background:var(--bad-bg)}.na{color:var(--na);background:var(--na-bg)}
.fake{color:var(--fake);background:var(--fake-bg)}
.note{border:1px solid var(--line);border-radius:8px;padding:10px 14px;margin:12px 0;
background:var(--card)}.note.bad{border-color:var(--bad)}.note.fake{border-color:var(--fake)}
details{margin:6px 0}summary{cursor:pointer}
details.exp{border:1px solid var(--line);border-radius:8px;padding:8px 12px;background:var(--card)}
details.exp>summary{font-size:1.05rem}
.q{border-top:1px solid var(--line);padding:6px 0}
.question{font-size:1.05rem;font-weight:600}
.span{font:12px ui-monospace,Consolas,monospace;color:var(--muted)}
ol.plain{padding-left:1.2rem}
"""


def _banner(data: ReportData) -> str:
    notes = [
        "<p>Built from the results of one run of "
        f"<code>{_e(data.source)}</code>. Nothing was re-run to make this page.</p>",
        "<p><strong>This file stores questions, document passages and answers as plain "
        "text.</strong> Treat it like the documents themselves.</p>",
        f"<p>Highlighted text marks characters that overlap a gold evidence span. A chunk "
        f"<em>covers</em> evidence only when the overlap reaches min_overlap = "
        f"{data.min_overlap} of the evidence or of the chunk. Offsets are character indices "
        "(Unicode code points), end exclusive.</p>",
    ]
    reports = data.generation
    if reports is not None:
        records = [r for report in reports for r in report.records]
        fake = sum("fake" in (r.source, r.original_source) for r in records)
        calls: list[JudgeCall] = []
        for r in records:
            if r.faithfulness is not None:
                calls.extend(r.faithfulness.calls)
            if r.relevance is not None and r.relevance.call is not None:
                calls.append(r.relevance.call)
        fake += sum("fake" in (c.source, c.original_source) for c in calls)
        attempts: list[GenerationRecord | JudgeCall] = [*records, *calls]
        with_response = sum(a.source == "api" for a in attempts)
        # No response means an API error. FakeLLM's scripted errors have provider "fake".
        errors = sum(a.source is None and a.provider != "fake" for a in attempts)
        cached = sum(a.source == "cache" for a in attempts)
        scripted = sum(
            a.source == "fake" or (a.source is None and a.provider == "fake") for a in attempts
        )
        if fake:
            notes.append(
                f'<div class="note fake"><strong>{fake} responses come from a scripted fake '
                "model.</strong> They are test data, not model output, and say nothing about "
                "any real model.</div>"
            )
        stops = [r.stop_attempt for r in reports if r.stop_attempt is not None]
        sent_stops = sum(s.sent_to_api for s in stops)
        errors += sent_stops
        stop_note = ""
        if sent_stops:
            stop_note = (
                f" The errors include {sent_stops} call that stopped the run: it has no result, "
                "and its tokens and cost are unknown."
            )
        if len(stops) > sent_stops:
            stop_note += (
                " The error that stopped the run was not counted: it came from the scripted fake "
                "model or was raised before any request was sent."
            )
        notes.append(
            f"<p>Logical API calls attempted, generator and judges, each counted once: "
            f"{with_response + errors} ({with_response} returned a response, including refusals, "
            f"truncated and invalid output; {errors} ended in an API error without one). "
            "These are not HTTP request counts: retries the SDK makes before a final response "
            f"are not visible here.{stop_note} Not included: {cached} cache hits and {scripted} "
            "scripted fake calls.</p>"
            "<p>Answers are not checked for correctness, and the judges have not been validated "
            "against human labels.</p>"
        )
    return '<div class="note">' + "".join(notes) + "</div>"


def render_report(data: ReportData, options: ReportOptions | None = None) -> str:
    if not data.experiments:
        raise ValueError("no experiments to report")
    options = options or ReportOptions()
    check_report_options(options, data.depth)
    body = [
        "<h1>ragtriage report</h1>",
        _banner(data),
        _retrieval_summary(data),
        _statistics(data),
        _group_section(data),
        _generation_summary(data),
        '<section><h2>Questions</h2><p class="muted">One block per experiment. Open a '
        "question to see its gold evidence, the retrieved chunks and any generation result.</p>",
        _detail_section(data, options),
        *(_experiment_questions(data, i, options) for i in range(len(data.experiments))),
        "</section>",
    ]
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>ragtriage report</title>"
        f"<style>{_CSS}</style></head><body><main>{''.join(body)}</main></body></html>\n"
    )
