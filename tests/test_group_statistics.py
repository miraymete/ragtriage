"""Group-level (cluster) bootstrap. Expected values are worked out by hand."""

import json
import textwrap

import numpy as np
import pytest
from typer.testing import CliRunner

from ragtriage.cli import app
from ragtriage.evaluate import ExampleScore, RetrievalReport
from ragtriage.metrics.retrieval import Failure
from ragtriage.metrics.statistics import (
    BootstrapSettings,
    compare_groups,
    group_bootstrap_mean,
    group_paired_difference,
    group_statistics,
    resample_groups,
)
from ragtriage.models import QAExample

SETTINGS = BootstrapSettings(n_resamples=20_000, seed=3)

# Article A has three questions scoring 1; article B has one scoring 0.
VALUES = [("a1", 1.0), ("a2", 1.0), ("a3", 1.0), ("b1", 0.0)]
GROUPS = {"a1": "A", "a2": "A", "a3": "A", "b1": "B"}


def test_the_mean_is_question_weighted_not_a_mean_of_group_means():
    result = group_bootstrap_mean(VALUES, GROUPS, "mrr", SETTINGS)
    # 3 of 4 questions score 1. The mean of the two article means would be 0.5.
    assert result.mean == 0.75
    assert (result.n_groups, result.n_questions) == (2, 4)


def test_groups_are_resampled_whole_and_with_multiplicity():
    sums, counts = np.array([3.0, 0.0]), np.array([3.0, 1.0])
    means = resample_groups(sums, counts, SETTINGS)
    # Two draws from {A, B}: AA gives 3/6 = 1, AB or BA gives 3/4 = 0.75, BB gives 0/2 = 0.
    # Drawing questions one by one would also give 0.25 and 0.5.
    assert set(np.unique(means)) <= {0.0, 0.75, 1.0}
    share = {v: float(np.mean(means == v)) for v in (0.0, 0.75, 1.0)}
    assert share[1.0] == pytest.approx(0.25, abs=0.02)
    assert share[0.75] == pytest.approx(0.5, abs=0.02)
    assert share[0.0] == pytest.approx(0.25, abs=0.02)


def test_percentile_interval_over_group_draws():
    result = group_bootstrap_mean(VALUES, GROUPS, "mrr", SETTINGS)
    # 25% of draws give 0 and 25% give 1, so the 2.5% and 97.5% percentiles are 0 and 1.
    assert (result.ci_low, result.ci_high) == (0.0, 1.0)


def test_paired_difference_is_candidate_minus_baseline_on_shared_draws():
    baseline = [("a1", 0.0), ("a2", 0.0), ("a3", 0.0), ("b1", 1.0)]
    candidate = [("b1", 1.0), ("a1", 1.0), ("a2", 1.0), ("a3", 1.0)]  # order differs
    result = group_paired_difference(baseline, candidate, GROUPS, "mrr", SETTINGS)
    assert (result.mean_baseline, result.mean_candidate) == (0.25, 1.0)
    assert result.mean_difference == 0.75
    # Differences by group are A: 3 over 3 questions, B: 0 over 1, as in the case above.
    assert (result.ci_low, result.ci_high) == (0.0, 1.0)


def test_a_constant_difference_has_zero_width_because_draws_are_shared():
    # The candidate adds 0.25 to every question. With separate draws for each side the
    # difference would vary; with shared draws it cannot.
    baseline = [("a1", 0.0), ("a2", 0.5), ("a3", 0.25), ("b1", 0.75)]
    candidate = [(q, v + 0.25) for q, v in baseline]
    result = group_paired_difference(baseline, candidate, GROUPS, "mrr", SETTINGS)
    assert result.note == "constant_difference"
    assert (result.ci_low, result.ci_high) == (0.25, 0.25)
    same = group_paired_difference(baseline, baseline, GROUPS, "mrr", SETTINGS)
    assert (same.note, same.ci_low, same.ci_high) == ("all_zero", 0.0, 0.0)


def test_mismatched_questions_or_groups_are_errors():
    with pytest.raises(ValueError, match="no group for"):
        group_bootstrap_mean(VALUES, {"a1": "A"}, "mrr", SETTINGS)
    with pytest.raises(ValueError, match=r"only in baseline \['b1'\]"):
        group_paired_difference(VALUES, VALUES[:3], GROUPS, "mrr", SETTINGS)
    with pytest.raises(ValueError, match="duplicate question id"):
        group_bootstrap_mean([*VALUES, ("a1", 0.0)], GROUPS, "mrr", SETTINGS)


def test_one_group_gives_no_interval_and_equal_values_are_labeled():
    single = group_bootstrap_mean(VALUES[:3], GROUPS, "mrr", SETTINGS)
    assert (single.ci_low, single.ci_high, single.note) == (None, None, "single_group")
    flat = group_bootstrap_mean([(q, 0.5) for q, _ in VALUES], GROUPS, "mrr", SETTINGS)
    assert (flat.ci_low, flat.ci_high, flat.note) == (0.5, 0.5, "constant")


def test_seed_reproducibility():
    rng = np.random.default_rng(0)
    values = [(f"q{i}", float(rng.random())) for i in range(60)]
    groups = {q: f"g{i % 7}" for i, (q, _) in enumerate(values)}
    first = group_bootstrap_mean(values, groups, "mrr", SETTINGS)
    assert group_bootstrap_mean(values, groups, "mrr", SETTINGS) == first
    other = group_bootstrap_mean(values, groups, "mrr", SETTINGS.model_copy(update={"seed": 4}))
    assert (other.ci_low, other.ci_high) != (first.ci_low, first.ci_high)


def _report(scores):
    return RetrievalReport(
        mean={"mrr": sum(scores.values()) / len(scores)},
        per_example=tuple(
            ExampleScore(example_id=q, scores={"mrr": v}, failure=Failure.OK)
            for q, v in scores.items()
        ),
        failures={Failure.OK: len(scores)},
        triage_k=1,
        n_unanswerable_skipped=0,
    )


def test_reports_need_a_group_for_every_scored_question():
    report = _report(dict(VALUES))
    stats = group_statistics("x", report, GROUPS, ["mrr"], SETTINGS)
    assert (stats.n_groups, stats.n_questions, stats.weighting) == (2, 4, "question")
    with pytest.raises(ValueError, match="1 have none"):
        group_statistics("x", report, {**GROUPS, "b1": None}, ["mrr"], SETTINGS)
    with pytest.raises(ValueError, match="different experiments"):
        compare_groups(("x", report), ("x", report), GROUPS, ["mrr"], SETTINGS)


def test_group_field_is_optional_and_round_trips():
    assert QAExample(id="q", question="?").group is None
    example = QAExample.model_validate({"id": "q", "question": "?", "group": "Article"})
    assert QAExample.model_validate(example.model_dump()).group == "Article"


# --- CLI --------------------------------------------------------------------------------------

runner = CliRunner()


def _workspace(tmp_path, with_groups=True):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("Ankara başkenttir. İzmir bir liman kentidir.", "utf-8")
    (docs / "b.txt").write_text("Bursa yeşildir. Konya düzdür.", "utf-8")
    rows = [
        ("q1", "İzmir nasıl?", "a", "İzmir bir liman kentidir.", "A"),
        ("q2", "Ankara ne?", "a", "Ankara başkenttir.", "A"),
        ("q3", "Bursa nasıl?", "b", "Bursa yeşildir.", "B"),
    ]
    lines = []
    for qid, question, doc, quote, group in rows:
        row = {"id": qid, "question": question, "evidence": [{"doc_id": doc, "quote": quote}]}
        if with_groups:
            row["group"] = group
        lines.append(json.dumps(row, ensure_ascii=False))
    (tmp_path / "qa.jsonl").write_text("\n".join(lines) + "\n", "utf-8")
    path = tmp_path / "config.yaml"
    path.write_text(
        textwrap.dedent(
            """\
            documents: docs
            dataset: qa.jsonl
            k: 1
            depth: 5
            experiments:
              - name: naive
                chunker: {type: fixed, size: 4000}
                retriever: {type: bm25, tokenizer: naive}
              - name: prefix
                chunker: {type: fixed, size: 4000}
                retriever: {type: bm25, tokenizer: turkish_prefix}
            """
        ),
        "utf-8",
    )
    return path


def test_cli_group_stats_are_separate_from_question_level_stats(tmp_path):
    path = _workspace(tmp_path)
    out, html = tmp_path / "out.json", tmp_path / "report.html"
    args = ["run", str(path), "--group-stats", "--resamples", "200", "--seed", "1"]
    args += ["--baseline", "naive", "--candidate", "prefix", "--out", str(out), "--html", str(html)]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "Group-level statistics (2 groups, 3 questions" in result.output
    assert "question-weighted" in result.output
    assert "no p-value" in result.output
    naive, prefix = json.loads(out.read_text("utf-8"))
    assert naive["group_statistics"]["method"] == "group percentile bootstrap"
    assert naive["group_statistics"]["bootstrap"] == {
        "method": "percentile",
        "confidence": 0.95,
        "n_resamples": 200,
        "seed": 1,
    }
    assert prefix["group_comparison"]["direction"] == "candidate - baseline"
    assert "p_value" not in json.dumps(prefix["group_comparison"])
    page = html.read_text("utf-8")
    assert "<h2>Group-level uncertainty</h2>" in page
    assert "There is no p-value for this comparison" in page


def test_cli_group_stats_without_groups_is_an_error(tmp_path):
    result = runner.invoke(app, ["run", str(_workspace(tmp_path, False)), "--group-stats"])
    assert result.exit_code == 2
    assert "need a group for every scored question" in result.output


def test_default_run_has_no_group_section(tmp_path):
    result = runner.invoke(app, ["run", str(_workspace(tmp_path))])
    assert result.exit_code == 0, result.output
    assert "Group-level" not in result.output


def test_a_question_is_in_the_same_group_for_both_experiments():
    baseline = _report({"a1": 0.0, "a2": 0.0, "b1": 1.0})
    candidate = _report({"a1": 1.0, "a2": 1.0, "b1": 1.0})
    groups = {"a1": "A", "a2": "A", "b1": "B"}
    result = compare_groups(("base", baseline), ("cand", candidate), groups, ["mrr"], SETTINGS)
    # One mapping groups both sides: A holds the +1 differences, B the 0. Grouping the candidate
    # differently is not possible through this API.
    assert (result.n_groups, result.n_questions) == (2, 3)
    assert result.metrics[0].mean_difference == pytest.approx(2 / 3)
    # A candidate that scored a question the baseline did not is rejected, not regrouped.
    extra = _report({"a1": 1.0, "a2": 1.0, "b1": 1.0, "c1": 1.0})
    with pytest.raises(ValueError, match=r"only in candidate \['c1'\]"):
        compare_groups(
            ("base", baseline), ("cand", extra), {**groups, "c1": "C"}, ["mrr"], SETTINGS
        )
    # A question the baseline scored but the mapping does not know is an error too.
    with pytest.raises(ValueError, match="1 have none"):
        compare_groups(
            ("base", baseline), ("cand", candidate), {"a1": "A", "a2": "A"}, ["mrr"], SETTINGS
        )
