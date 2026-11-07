"""Command line interface: `ragtriage run config.yaml`."""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml

from ragtriage.config import EvalConfig, load_config
from ragtriage.generation.cache import CacheError
from ragtriage.generation.generator import GenerationStatus, Generator
from ragtriage.generation.judge import FaithfulnessJudge
from ragtriage.generation.pipeline import run_generation
from ragtriage.generation.relevance import RelevanceJudge
from ragtriage.generation.report import GenerationReport, Rate
from ragtriage.html_report import ReportData, render_report
from ragtriage.metrics.retrieval import Failure
from ragtriage.metrics.statistics import (
    BootstrapSettings,
    Comparison,
    ExperimentStatistics,
    MeanInterval,
    PairedDifference,
    PermutationSettings,
    compare_experiments,
    experiment_statistics,
    statistic_metrics,
)
from ragtriage.models import QAExample, RetrievalRun
from ragtriage.runner import (
    ExperimentResult,
    build_generator,
    build_judge,
    build_relevance,
    run_retrieval,
)

app = typer.Typer(add_completion=False, help="Measure RAG pipelines and compare configurations.")


@app.callback()
def _main() -> None:
    """Keep `run` a named subcommand so more commands can be added later."""


def comparison_table(results: Sequence[ExperimentResult], k: int) -> str:
    rows = [
        f"| Experiment | Chunks | Recall@{k} | Precision@{k} | MRR | nDCG@{k} "
        f"| Context chars@{k} | Query ms | OK | Ranking | Retrieval |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for result in results:
        mean, failures = result.report.mean, result.report.failures
        rows.append(
            f"| {result.name} | {result.n_chunks} "
            f"| {mean[f'recall@{k}']:.3f} | {mean[f'precision@{k}']:.3f} | {mean['mrr']:.3f} "
            f"| {mean[f'ndcg@{k}']:.3f} | {result.context_chars:.0f} "
            f"| {result.retrieval_ms:.2f} | {failures.get(Failure.OK, 0)} "
            f"| {failures.get(Failure.RANKING, 0)} | {failures.get(Failure.RETRIEVAL, 0)} |"
        )
    return "\n".join(rows)


def failure_list(results: Sequence[ExperimentResult]) -> str:
    lines = []
    for result in results:
        failed = [s for s in result.report.per_example if s.failure is not Failure.OK]
        lines.append(f"\n{result.name}: {len(failed)} failed")
        lines.extend(f"- [{score.failure}] {score.example_id}" for score in failed)
    return "\n".join(lines)


def _interval(item: MeanInterval | PairedDifference) -> str:
    if item.ci_low is None or item.ci_high is None:
        return "no interval (n=1)"
    return f"[{item.ci_low:.3f}, {item.ci_high:.3f}]"


def _confidence(item: Comparison | ExperimentStatistics) -> str:
    return f"{item.bootstrap.confidence:.0%}"


def statistics_table(stats: Sequence[ExperimentStatistics]) -> str:
    metrics = [m.metric for m in stats[0].metrics]
    rows = [
        "| Experiment | Questions | " + " | ".join(f"{m} mean [CI]" for m in metrics) + " |",
        "|---|---|" + "---|" * len(metrics),
    ]
    for item in stats:
        cells = " | ".join(
            f"{m.mean:.3f} {_interval(m)}{' all equal' if m.note == 'constant' else ''}"
            for m in item.metrics
        )
        rows.append(f"| {item.experiment} | {item.n_questions} | {cells} |")
    return "\n".join(rows)


def _p_value(item: PairedDifference) -> str:
    test = item.permutation
    if test is None:
        return "n/a (n=1)"
    if test.method == "exact":
        return f"{test.p_value:.4f} (exact, {test.n_extreme}/{test.n_patterns})"
    return f"{test.p_value:.4f} (Monte Carlo, {test.n_patterns} draws)"


_DIFFERENCE_NOTES = {
    "all_zero": "every question scored the same in both experiments",
    "constant_difference": "every question has the same difference, so the interval has zero "
    "width; that describes this sample, not new questions",
}


def comparison_report(comparison: Comparison) -> str:
    rows = [
        f"| Metric | Baseline mean | Candidate mean | Difference | {_confidence(comparison)} CI "
        "| Higher / lower / tied | Unadjusted two-sided p |",
        "|---|---|---|---|---|---|---|",
    ]
    for item in comparison.metrics:
        rows.append(
            f"| {item.metric} | {item.mean_baseline:.3f} | {item.mean_candidate:.3f} "
            f"| {item.mean_difference:+.3f} | {_interval(item)} "
            f"| {item.n_candidate_higher} / {item.n_candidate_lower} / {item.n_tied} "
            f"| {_p_value(item)} |"
        )
    for item in comparison.metrics:
        if item.note in _DIFFERENCE_NOTES:
            rows.append(f"{item.metric}: {_DIFFERENCE_NOTES[item.note]}.")
    return "\n".join(rows)


STATISTICS_NOTE = (
    "Intervals and p-values assume the questions are independent and represent the questions the "
    "system will really face. Questions about the same document are not independent, and a small "
    "self-written set is not a sample of real use. With few questions, bootstrap intervals tend "
    "to be too narrow. p-values are unadjusted: checking several metrics or pairs raises the "
    "chance of a difference that is only luck. None of this names a winner."
)


def _rate(rate: Rate) -> str:
    return "n/a" if rate.value is None else f"{rate.numerator}/{rate.denominator}"


def generation_table(
    results: Sequence[ExperimentResult], reports: Sequence[GenerationReport]
) -> str:
    """One row per experiment that has a report. A stopped run has fewer reports than results."""
    rows = [
        "| Experiment | Evaluated | Failed | Abstained, unanswerable "
        "| Abstained, answerable | ... evidence in context | ... evidence missing "
        "| Billed input tokens | Billed output tokens | Est. cost (USD) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for result, report in zip(results, reports, strict=False):
        failed = ", ".join(f"{status} {n}" for status, n in sorted(report.failures.items()))
        cost = "n/a" if report.cost_usd is None else f"{report.cost_usd:.4f}"
        rows.append(
            f"| {result.name} | {report.n_evaluated}/{report.n_examples} "
            f"| {sum(report.failures.values())}{f' ({failed})' if failed else ''} "
            f"| {_rate(report.abstained_when_unanswerable)} "
            f"| {_rate(report.abstained_when_answerable)} "
            f"| {_rate(report.abstained_when_answerable_evidence_present)} "
            f"| {_rate(report.abstained_when_answerable_evidence_missing)} "
            f"| {report.billed_usage.input_tokens} | {report.billed_usage.output_tokens} | {cost} |"
        )
    return "\n".join(rows)


def generation_notes(reports: Sequence[GenerationReport]) -> str:
    lines = ["", "Answers are not checked for correctness yet."]
    cached = sum(r.n_cached for r in reports)
    if cached:
        lines.append(f"{cached} answers came from the cache; they add no new tokens or cost.")
    if any(r.n_fake for r in reports):
        lines.append("Some responses came from a scripted fake model; they are not billed usage.")
    unknown = sum(r.n_usage_unknown for r in reports)
    unknown += sum(bool(r.stop_attempt and r.stop_attempt.sent_to_api) for r in reports)
    if unknown:
        lines.append(f"{unknown} calls failed without reported usage; their tokens are unknown.")
    if any(r.cost_usd is not None for r in reports):
        lines.append("Cost is an estimate: it leaves out SDK retries and calls with unknown usage.")
    return "\n".join(lines)


def generation_failure_list(
    results: Sequence[ExperimentResult], reports: Sequence[GenerationReport]
) -> str:
    lines = []
    for result, report in zip(results, reports, strict=False):
        failed = [r for r in report.records if r.status is not GenerationStatus.OK]
        lines.append(f"\n{result.name}: {len(failed)} generations failed")
        lines.extend(f"- [{r.status}] {r.example_id}: {r.problem}" for r in failed)
    return "\n".join(lines)


def _generate(
    config: EvalConfig, examples: list[QAExample], runs: list[list[RetrievalRun]]
) -> list[GenerationReport]:
    assert config.generation is not None
    generator = build_generator(config.generation, config.k, config.min_overlap)
    judge = relevance = None
    try:
        judge = build_judge(config.generation)
        relevance = build_relevance(config.generation)
    except BaseException:
        # A judge could not start: release whatever was already opened.
        asyncio.run(_close(generator, judge))
        raise
    return asyncio.run(run_generation(generator, examples, runs, judge, relevance))


async def _close(*clients: Generator | FaithfulnessJudge | RelevanceJudge | None) -> None:
    for client in clients:
        if client is not None:
            await client.aclose()


def same_model_warnings(config: EvalConfig) -> list[str]:
    """Warn when a judge uses the generator's provider and model."""
    generation = config.generation
    if generation is None:
        return []
    generator = (generation.generator.provider, generation.generator.model)
    judges = {"faithfulness": generation.judge}
    if generation.relevance is not None:
        judges["relevance"] = generation.relevance.judge
    return [
        f"Warning: the {name} judge uses the same model as the generator "
        f"({generator[0]}/{generator[1]}). A judge may favor its own model's answers. "
        "A different model may reduce this bias but does not remove it."
        for name, judge in judges.items()
        if judge is not None and (judge.provider, judge.model) == generator
    ]


def relevance_table(
    results: Sequence[ExperimentResult], reports: Sequence[GenerationReport]
) -> str:
    rows = [
        "| Experiment | Relevance (mean cosine, -1..1) | Scored | N/A | Failed "
        "| Judge billed input tokens | Judge billed output tokens | Judge est. cost (USD) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for result, report in zip(results, reports, strict=False):
        summary = report.relevance
        if summary is None:
            continue
        mean = "n/a" if summary.mean is None else f"{summary.mean:.3f}"
        na = sum(summary.not_applicable.values())
        reasons = ", ".join(f"{k} {v}" for k, v in sorted(summary.not_applicable.items()))
        usage = summary.judge_usage
        cost = "n/a" if usage.cost_usd is None else f"{usage.cost_usd:.4f}"
        rows.append(
            f"| {result.name} | {mean} | {summary.n_scored} "
            f"| {na}{f' ({reasons})' if reasons else ''} | {summary.n_failed} "
            f"| {usage.billed_usage.input_tokens} | {usage.billed_usage.output_tokens} | {cost} |"
        )
    return "\n".join(rows)


def faithfulness_table(
    results: Sequence[ExperimentResult], reports: Sequence[GenerationReport]
) -> str:
    rows = [
        "| Experiment | Faithfulness (mean of scored) | Scored | N/A | Failed "
        "| Claims supported / unsupported / uncertain "
        "| Judge billed input tokens | Judge billed output tokens | Judge est. cost (USD) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for result, report in zip(results, reports, strict=False):
        summary = report.faithfulness
        if summary is None:
            continue
        mean = "n/a" if summary.mean is None else f"{summary.mean:.3f}"
        na = sum(summary.not_applicable.values())
        reasons = ", ".join(f"{k} {v}" for k, v in sorted(summary.not_applicable.items()))
        usage = summary.judge_usage
        cost = "n/a" if usage.cost_usd is None else f"{usage.cost_usd:.4f}"
        rows.append(
            f"| {result.name} | {mean} | {summary.n_scored} "
            f"| {na}{f' ({reasons})' if reasons else ''} | {summary.n_failed} "
            f"| {summary.n_supported} / {summary.n_unsupported} / {summary.n_uncertain} "
            f"| {usage.billed_usage.input_tokens} | {usage.billed_usage.output_tokens} | {cost} |"
        )
    return "\n".join(rows)


def _check_pair(config: EvalConfig, baseline: str, candidate: str) -> None:
    """Fail before any work if the chosen experiments do not exist."""
    names = [experiment.name for experiment in config.experiments]
    unknown = [name for name in (baseline, candidate) if name not in names]
    if unknown:
        raise ValueError(f"unknown experiment {unknown[0]!r}; choose from {names}")
    if baseline == candidate:
        raise ValueError("--baseline and --candidate must name different experiments")


@app.command()
def run(
    config_path: Annotated[Path, typer.Argument(help="YAML file that defines the experiments.")],
    out: Annotated[Path | None, typer.Option(help="Also write full results to this JSON.")] = None,
    failures: Annotated[bool, typer.Option(help="List the failed questions.")] = False,
    stats: Annotated[
        bool, typer.Option(help="Add bootstrap intervals for recall@k, MRR and nDCG@k.")
    ] = False,
    baseline: Annotated[
        str | None, typer.Option(help="Experiment A for a paired comparison (B - A).")
    ] = None,
    candidate: Annotated[
        str | None, typer.Option(help="Experiment B for a paired comparison (B - A).")
    ] = None,
    confidence: Annotated[float, typer.Option(help="Bootstrap confidence level.")] = 0.95,
    resamples: Annotated[int, typer.Option(help="Bootstrap resamples.")] = 10_000,
    permutations: Annotated[
        int, typer.Option(help="Random sign patterns when an exact test is too large.")
    ] = 100_000,
    seed: Annotated[int, typer.Option(help="Seed for resampling and random sign patterns.")] = 0,
    html: Annotated[
        Path | None, typer.Option(help="Also write a single-file HTML report of this run.")
    ] = None,
) -> None:
    """Run every experiment in a config and print a comparison table."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to a legacy codepage.
    if (baseline is None) != (candidate is None):
        typer.echo("error: --baseline and --candidate must be given together", err=True)
        raise typer.Exit(code=2)
    try:
        bootstrap = BootstrapSettings(confidence=confidence, n_resamples=resamples, seed=seed)
        permutation = PermutationSettings(n_draws=permutations, seed=seed)
        config = load_config(config_path)
        if baseline is not None and candidate is not None:
            _check_pair(config, baseline, candidate)
        retrieval = run_retrieval(config)
    except (ValueError, ImportError, OSError, yaml.YAMLError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=2) from error
    examples, outputs = retrieval.examples, retrieval.experiments
    results = [result for result, _ in outputs]

    typer.echo(comparison_table(results, config.k))
    if failures:
        typer.echo(failure_list(results))
    skipped = results[0].report.n_unanswerable_skipped
    scored = len(results[0].report.per_example)
    typer.echo(f"\nScored {scored} questions. Skipped {skipped} unanswerable.")

    metrics = statistic_metrics(config.k)
    experiment_stats: list[ExperimentStatistics] | None = None
    if stats:
        experiment_stats = [
            experiment_statistics(r.name, r.report, metrics, bootstrap) for r in results
        ]
        typer.echo(
            f"\nRetrieval statistics ({scored} questions, {skipped} unanswerable excluded; "
            f"percentile bootstrap, {_confidence(experiment_stats[0])} CI, "
            f"{resamples} resamples, seed {seed})\n"
        )
        typer.echo(statistics_table(experiment_stats))
    comparison: Comparison | None = None
    if baseline is not None and candidate is not None:
        by_name = {r.name: r.report for r in results}
        comparison = compare_experiments(
            (baseline, by_name[baseline]),
            (candidate, by_name[candidate]),
            metrics,
            bootstrap,
            permutation,
        )
        typer.echo(
            f"\nPaired comparison: candidate {candidate!r} - baseline {baseline!r} "
            f"({comparison.n_questions} questions, {skipped} unanswerable excluded; "
            f"paired percentile bootstrap, {resamples} resamples, seed {seed}; "
            "sign-flip permutation test)\n"
        )
        typer.echo(comparison_report(comparison))
    if stats or comparison is not None:
        if experiment_stats is not None and any(
            m.note == "constant" for s in experiment_stats for m in s.metrics
        ):
            typer.echo(
                "\n'all equal': every question has the same score, so the interval has zero "
                "width. That describes this sample, not certainty about new questions."
            )
        typer.echo("\n" + STATISTICS_NOTE)

    reports: list[GenerationReport] | None = None
    if config.generation is not None:
        try:
            reports = _generate(config, examples, [runs for _, runs in outputs])
        except (ImportError, CacheError) as error:  # raised before any request is sent
            typer.echo(f"error: generation not started: {error}", err=True)
            raise typer.Exit(code=2) from error
        typer.echo(f"\nGeneration (top {config.k} chunks per question)\n")
        typer.echo(generation_table(results, reports))
        if failures:
            typer.echo(generation_failure_list(results, reports))
        if any(r.faithfulness is not None for r in reports):
            typer.echo("\nFaithfulness (claims supported by the chunks the generator saw)\n")
            typer.echo(faithfulness_table(results, reports))
            typer.echo(
                "Faithfulness does not measure correctness, completeness or relevance, and the "
                "judge is not validated against human labels."
            )
        if any(r.relevance is not None for r in reports):
            typer.echo("\nAnswer relevance (questions written from the answer alone)\n")
            typer.echo(relevance_table(results, reports))
            typer.echo(
                "Relevance is a semantic-similarity proxy. It does not measure correctness or "
                "completeness."
            )
        for warning in same_model_warnings(config):
            typer.echo(warning)
        typer.echo(generation_notes(reports))

    if out:
        payload: list[dict[str, Any]] = [result.model_dump(mode="json") for result in results]
        if experiment_stats is not None:
            for item, stat in zip(payload, experiment_stats, strict=True):
                item["statistics"] = stat.model_dump(mode="json")
        if comparison is not None:
            # Stored on the candidate's entry, so the top level stays a list of experiments.
            for item in payload:
                if item["name"] == comparison.candidate:
                    item["comparison"] = comparison.model_dump(mode="json")
        if reports is not None:
            for item, report in zip(payload, reports, strict=False):
                item["generation"] = report.model_dump(mode="json")
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        typer.echo(f"Wrote {out}")

    if html:
        page = render_report(
            ReportData(
                source=str(config_path),
                k=config.k,
                depth=config.depth,
                min_overlap=config.min_overlap,
                documents=retrieval.documents,
                examples=examples,
                experiments=outputs,
                statistics=experiment_stats,
                comparison=comparison,
                generation=reports,
            )
        )
        html.write_text(page, encoding="utf-8")
        typer.echo(f"Wrote {html}")

    stopped = next((r.stopped for r in reports or [] if r.stopped), None)
    if stopped:
        # Results so far are printed and saved above; the exit code still marks the run as failed.
        typer.echo(f"error: generation stopped early: {stopped}", err=True)
        raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
