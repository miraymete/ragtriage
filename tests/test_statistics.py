"""Expected values below are worked out by hand from the sign patterns, not by a second copy of
the algorithm."""

import json
import textwrap

import pytest
from typer.testing import CliRunner

from ragtriage.cli import app
from ragtriage.evaluate import ExampleScore, RetrievalReport
from ragtriage.metrics import statistics as stats_module
from ragtriage.metrics.retrieval import Failure
from ragtriage.metrics.statistics import (
    BootstrapSettings,
    PermutationSettings,
    bootstrap_mean,
    compare_experiments,
    experiment_statistics,
    paired_difference,
    sign_flip_test,
)

BOOT = BootstrapSettings(n_resamples=2000, seed=7)
PERM = PermutationSettings(n_draws=20_000, seed=7)


# --- exact sign-flip test ---------------------------------------------------------------------


def test_exact_test_on_three_differences():
    # Sums of +-1 +-2 +-3: 6, 4, 2, 0, 0, -2, -4, -6. |sum| >= 6 for 2 of 8 patterns.
    test = sign_flip_test([1, 2, 3], PERM)
    assert (test.method, test.n_nonzero, test.n_patterns, test.n_extreme) == ("exact", 3, 8, 2)
    assert test.p_value == 0.25
    assert test.adjusted is False


def test_ties_with_the_observed_value_count_as_extreme():
    # Observed sum 2. Sums of +-1 +-1 +-2: 4, 0, 2, -2, 2, -2, 0, -4. |sum| >= 2 for 6 of 8.
    test = sign_flip_test([1, -1, 2], PERM)
    assert (test.n_extreme, test.p_value) == (6, 0.75)
    # Observed sum 2. Sums of +-1 +-1: 2, 0, 0, -2. Both ends count: 2 of 4.
    assert sign_flip_test([1, 1], PERM).p_value == 0.5


def test_ties_survive_floating_point_rounding():
    # 0.1 + 0.2 - 0.3 is 0 in exact arithmetic, so every one of the 8 patterns is a tie.
    test = sign_flip_test([0.1, 0.2, -0.3], PERM)
    assert (test.n_extreme, test.p_value) == (8, 1.0)


def test_zero_differences_are_not_permuted():
    test = sign_flip_test([0, 1, 0, 2, 3, 0], PERM)
    assert (test.n_nonzero, test.n_patterns, test.p_value) == (3, 8, 0.25)


def test_all_zero_differences_give_p_one():
    test = sign_flip_test([0.0, 0.0, 0.0], PERM)
    assert (test.method, test.p_value, test.n_nonzero) == ("exact", 1.0, 0)


def test_exact_test_at_the_largest_exact_size():
    # Twenty equal differences: only all-plus and all-minus reach |sum| = 20.
    test = sign_flip_test([1.0] * 20, PERM)
    assert test.method == "exact"
    assert test.p_value == 2 / 2**20


def test_exact_patterns_are_generated_in_bounded_chunks():
    chunks = stats_module._sign_chunks_exact(20)
    sizes = [chunk.shape for chunk in chunks]
    assert sum(rows for rows, _ in sizes) == 2**20
    assert max(rows * cols for rows, cols in sizes) <= stats_module._CHUNK_CELLS
    assert len(sizes) > 1


def test_bad_differences_are_rejected():
    with pytest.raises(ValueError, match="no values"):
        sign_flip_test([], PERM)
    with pytest.raises(ValueError, match="finite"):
        sign_flip_test([1.0, float("nan")], PERM)


# --- Monte Carlo sign-flip test ---------------------------------------------------------------


def test_monte_carlo_is_used_above_the_exact_limit_and_is_reproducible():
    settings = PermutationSettings(n_draws=20_000, seed=3, max_exact_nonzero=2)
    test = sign_flip_test([1, 2, 3], settings)
    assert test.method == "monte_carlo"
    assert test.seed == 3
    assert test.n_patterns == 20_000
    assert test.p_value == (test.n_extreme + 1) / (20_000 + 1)
    # The exact answer is 0.25; 20,000 draws land well within 0.02 of it.
    assert abs(test.p_value - 0.25) < 0.02
    assert sign_flip_test([1, 2, 3], settings) == test
    other = sign_flip_test([1, 2, 3], settings.model_copy(update={"seed": 4}))
    assert other.n_extreme != test.n_extreme


def test_monte_carlo_never_reports_zero():
    # 40 equal differences: the chance of drawing an extreme pattern is 2 / 2**40.
    test = sign_flip_test([1.0] * 40, PermutationSettings(n_draws=1000, seed=0))
    assert test.method == "monte_carlo"
    assert test.n_extreme == 0
    assert test.p_value == 1 / 1001


# --- bootstrap intervals ----------------------------------------------------------------------


def test_bootstrap_interval_for_two_values():
    # Resampled means of [0, 1] are 0, 0.5 and 1 with probabilities 1/4, 1/2, 1/4, so the
    # 2.5% and 97.5% percentiles are 0 and 1.
    interval = bootstrap_mean([0.0, 1.0], "recall@3", BOOT)
    assert (interval.mean, interval.ci_low, interval.ci_high) == (0.5, 0.0, 1.0)
    assert interval.note is None


def test_bootstrap_is_reproducible_and_depends_on_the_seed():
    values = [0.0, 0.2, 0.5, 0.5, 1.0, 1.0, 0.0, 0.33]
    first = bootstrap_mean(values, "mrr", BOOT)
    assert bootstrap_mean(values, "mrr", BOOT) == first
    other = bootstrap_mean(values, "mrr", BOOT.model_copy(update={"seed": 8}))
    assert (other.ci_low, other.ci_high) != (first.ci_low, first.ci_high)
    assert first.ci_low is not None and first.ci_high is not None
    assert min(values) <= first.ci_low <= first.mean <= first.ci_high <= max(values)


def test_bootstrap_with_one_value_or_equal_values():
    one = bootstrap_mean([0.7], "mrr", BOOT)
    assert (one.n, one.mean, one.ci_low, one.ci_high, one.note) == (
        1,
        0.7,
        None,
        None,
        "single_question",
    )
    same = bootstrap_mean([1.0, 1.0, 1.0], "mrr", BOOT)
    assert (same.ci_low, same.ci_high, same.note) == (1.0, 1.0, "constant")


def test_bootstrap_rejects_empty_and_non_finite_values():
    with pytest.raises(ValueError, match="nothing to estimate"):
        bootstrap_mean([], "mrr", BOOT)
    with pytest.raises(ValueError, match="finite"):
        bootstrap_mean([0.5, float("inf")], "mrr", BOOT)


def test_settings_are_validated():
    with pytest.raises(ValueError):
        BootstrapSettings(confidence=1.0)
    with pytest.raises(ValueError):
        BootstrapSettings(n_resamples=0)
    with pytest.raises(ValueError):
        PermutationSettings(n_draws=0)


# --- paired comparison ------------------------------------------------------------------------


def test_pairs_are_matched_by_question_id_and_the_difference_is_candidate_minus_baseline():
    baseline = [("q1", 0.0), ("q2", 1.0), ("q3", 0.5)]
    # Same questions, different order. Matching by position would pair q1 with q3.
    candidate = [("q3", 0.5), ("q1", 1.0), ("q2", 0.0)]
    result = paired_difference(baseline, candidate, "recall@3", BOOT, PERM)
    # Differences by id: q1 +1, q2 -1, q3 0.
    assert result.mean_difference == 0.0
    assert (result.n_candidate_higher, result.n_candidate_lower, result.n_tied) == (1, 1, 1)
    assert result.permutation is not None
    # Nonzero differences +1, -1: sums 0, 2, -2, 0 against an observed 0, so all 4 are extreme.
    assert result.permutation.p_value == 1.0

    better = paired_difference(
        [("q1", 0.0), ("q2", 0.0)], [("q1", 1.0), ("q2", 0.5)], "mrr", BOOT, PERM
    )
    assert better.mean_difference == 0.75
    assert (better.mean_baseline, better.mean_candidate) == (0.0, 0.75)
    assert better.ci_low is not None and better.ci_low > 0


def test_mismatched_or_duplicate_questions_are_errors():
    with pytest.raises(ValueError, match=r"only in baseline \['q2'\].*only in candidate \['q3'\]"):
        paired_difference([("q1", 1.0), ("q2", 1.0)], [("q1", 1.0), ("q3", 1.0)], "mrr", BOOT, PERM)
    with pytest.raises(ValueError, match="duplicate question id 'q1'"):
        paired_difference([("q1", 1.0), ("q1", 0.0)], [("q1", 1.0), ("q2", 1.0)], "mrr", BOOT, PERM)
    with pytest.raises(ValueError, match="no questions"):
        paired_difference([], [], "mrr", BOOT, PERM)


def test_one_question_gives_no_uncertainty_estimate():
    result = paired_difference([("q1", 0.0)], [("q1", 1.0)], "mrr", BOOT, PERM)
    assert result.mean_difference == 1.0
    assert (result.ci_low, result.ci_high, result.permutation) == (None, None, None)
    assert result.note == "single_question"


def test_all_zero_and_constant_differences_are_labeled():
    same = paired_difference(
        [("q1", 0.5), ("q2", 1.0)], [("q1", 0.5), ("q2", 1.0)], "mrr", BOOT, PERM
    )
    assert (same.note, same.ci_low, same.ci_high) == ("all_zero", 0.0, 0.0)
    assert same.permutation is not None and same.permutation.p_value == 1.0

    shifted = paired_difference(
        [("q1", 0.0), ("q2", 0.25), ("q3", 0.5)],
        [("q1", 0.5), ("q2", 0.75), ("q3", 1.0)],
        "mrr",
        BOOT,
        PERM,
    )
    assert (shifted.note, shifted.ci_low, shifted.ci_high) == ("constant_difference", 0.5, 0.5)
    # Three equal differences: 2 of 8 patterns reach |sum| = 1.5.
    assert shifted.permutation is not None and shifted.permutation.p_value == 0.25


def _report(scores: dict[str, float], skipped: int = 1) -> RetrievalReport:
    return RetrievalReport(
        mean={"mrr": sum(scores.values()) / len(scores)},
        per_example=tuple(
            ExampleScore(example_id=i, scores={"mrr": v}, failure=Failure.OK)
            for i, v in scores.items()
        ),
        failures={Failure.OK: len(scores)},
        triage_k=1,
        n_unanswerable_skipped=skipped,
    )


def test_experiment_statistics_and_comparison_from_reports():
    a = _report({"q1": 0.0, "q2": 0.5, "q3": 1.0})
    b = _report({"q1": 1.0, "q2": 0.5, "q3": 1.0})
    stat = experiment_statistics("a", a, ["mrr"], BOOT)
    assert (stat.n_questions, stat.n_unanswerable_excluded) == (3, 1)
    assert stat.bootstrap.method == "percentile"

    comparison = compare_experiments(("a", a), ("b", b), ["mrr"], BOOT, PERM)
    assert comparison.direction == "candidate - baseline"
    [metric] = comparison.metrics
    assert metric.mean_difference == pytest.approx(1 / 3)
    # One nonzero difference: patterns +1 and -1, both extreme.
    assert metric.permutation is not None and metric.permutation.p_value == 1.0

    with pytest.raises(ValueError, match="different experiments"):
        compare_experiments(("a", a), ("a", b), ["mrr"], BOOT, PERM)
    with pytest.raises(ValueError, match="unanswerable"):
        compare_experiments(("a", a), ("b", _report({"q1": 1.0}, skipped=2)), ["mrr"], BOOT, PERM)
    with pytest.raises(ValueError, match="not in the report"):
        experiment_statistics("a", a, ["recall@3"], BOOT)


# --- CLI --------------------------------------------------------------------------------------

runner = CliRunner()


@pytest.fixture
def config_path(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text(
        "Ankara başkenttir. İzmir bir liman kentidir. Bursa yeşildir. Konya düzdür.", "utf-8"
    )
    (tmp_path / "qa.jsonl").write_text(
        '{"id": "q1", "question": "İzmir nasıl bir kent?", '
        '"evidence": [{"doc_id": "a", "quote": "İzmir bir liman kentidir."}]}\n'
        '{"id": "q2", "question": "Bursa nasıl?", '
        '"evidence": [{"doc_id": "a", "quote": "Bursa yeşildir."}]}\n'
        '{"id": "q3", "question": "Efes nerede?"}\n',
        "utf-8",
    )
    path = tmp_path / "config.yaml"
    path.write_text(
        textwrap.dedent(
            """\
            documents: docs
            dataset: qa.jsonl
            k: 1
            depth: 5
            experiments:
              - name: small
                chunker: {type: fixed, size: 20}
                retriever: {type: bm25}
              - name: large
                chunker: {type: fixed, size: 60}
                retriever: {type: bm25}
            """
        ),
        "utf-8",
    )
    return path


def test_default_run_has_no_statistics(config_path, tmp_path):
    out = tmp_path / "out.json"
    result = runner.invoke(app, ["run", str(config_path), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert "statistics" not in result.output.lower()
    assert all("statistics" not in item and "comparison" not in item for item in _load(out))


def test_stats_and_comparison_are_printed_and_saved(config_path, tmp_path):
    out = tmp_path / "out.json"
    args = ["run", str(config_path), "--out", str(out), "--stats", "--seed", "5"]
    args += ["--baseline", "small", "--candidate", "large", "--resamples", "500"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "2 questions, 1 unanswerable excluded" in result.output
    assert "candidate 'large' - baseline 'small'" in result.output
    assert "Unadjusted two-sided p" in result.output
    assert "None of this names a winner." in result.output

    small, large = _load(out)
    assert small["statistics"]["bootstrap"] == {
        "method": "percentile",
        "confidence": 0.95,
        "n_resamples": 500,
        "seed": 5,
    }
    assert [m["metric"] for m in small["statistics"]["metrics"]] == ["recall@1", "mrr", "ndcg@1"]
    assert "comparison" not in small
    comparison = large["comparison"]
    assert (comparison["baseline"], comparison["candidate"]) == ("small", "large")
    assert comparison["direction"] == "candidate - baseline"
    assert comparison["n_unanswerable_excluded"] == 1
    assert all(m["permutation"]["adjusted"] is False for m in comparison["metrics"])
    # Same seed, same numbers.
    again = tmp_path / "again.json"
    assert runner.invoke(app, [*args[:3], str(again), *args[4:]]).exit_code == 0
    assert _load(again)[1]["comparison"] == comparison


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--baseline", "small"], "must be given together"),
        (["--baseline", "small", "--candidate", "nope"], "unknown experiment 'nope'"),
        (["--baseline", "small", "--candidate", "small"], "different experiments"),
        (["--stats", "--confidence", "1.5"], "confidence"),
    ],
)
def test_bad_statistics_options_fail_before_running(config_path, extra, message):
    result = runner.invoke(app, ["run", str(config_path), *extra])
    assert result.exit_code == 2
    assert message in result.output
    assert "| Experiment |" not in result.output


def _load(path):
    return json.loads(path.read_text("utf-8"))


def test_a_small_monte_carlo_p_value_is_never_shown_as_zero():
    from ragtriage.metrics.statistics import format_p

    assert format_p(1 / 100_001) == "1.0e-05"
    assert format_p(0.03125) == "0.0312"
    assert format_p(1.0) == "1.0000"
