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
from ragtriage.metrics.retrieval import Failure
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


@app.command()
def run(
    config_path: Annotated[Path, typer.Argument(help="YAML file that defines the experiments.")],
    out: Annotated[Path | None, typer.Option(help="Also write full results to this JSON.")] = None,
    failures: Annotated[bool, typer.Option(help="List the failed questions.")] = False,
) -> None:
    """Run every experiment in a config and print a comparison table."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to a legacy codepage.
    try:
        config = load_config(config_path)
        examples, outputs = run_retrieval(config)
    except (ValueError, ImportError, OSError, yaml.YAMLError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=2) from error
    results = [result for result, _ in outputs]

    typer.echo(comparison_table(results, config.k))
    if failures:
        typer.echo(failure_list(results))
    skipped = results[0].report.n_unanswerable_skipped
    scored = len(results[0].report.per_example)
    typer.echo(f"\nScored {scored} questions. Skipped {skipped} unanswerable.")

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
        if reports is not None:
            for item, report in zip(payload, reports, strict=False):
                item["generation"] = report.model_dump(mode="json")
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        typer.echo(f"Wrote {out}")

    stopped = next((r.stopped for r in reports or [] if r.stopped), None)
    if stopped:
        # Results so far are printed and saved above; the exit code still marks the run as failed.
        typer.echo(f"error: generation stopped early: {stopped}", err=True)
        raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
