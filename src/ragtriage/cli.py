"""Command line interface: `ragtriage run config.yaml`."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import typer
import yaml

from ragtriage.config import load_config
from ragtriage.metrics.retrieval import Failure
from ragtriage.runner import ExperimentResult, run_experiments

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
        results = run_experiments(config)
    except (ValueError, ImportError, OSError, yaml.YAMLError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=2) from error

    typer.echo(comparison_table(results, config.k))
    if failures:
        typer.echo(failure_list(results))
    skipped = results[0].report.n_unanswerable_skipped
    scored = len(results[0].report.per_example)
    typer.echo(f"\nScored {scored} questions. Skipped {skipped} unanswerable.")
    if out:
        payload = [result.model_dump(mode="json") for result in results]
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        typer.echo(f"Wrote {out}")


if __name__ == "__main__":
    app()
