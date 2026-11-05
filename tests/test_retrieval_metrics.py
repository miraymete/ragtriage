"""Expected values are worked out by hand in the comments, not copied from the code's output."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ragtriage.metrics.retrieval import (
    Failure,
    classify_failure,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

# Rank 2 finds "a", rank 4 finds "b", and "c" is never found.
MATCHES = [set(), {"a"}, set(), {"b"}, set()]
GOLD = {"a", "b", "c"}
GRADES = {"a": 1.0, "b": 1.0, "c": 1.0}


@pytest.mark.parametrize(("k", "expected"), [(1, 0.0), (2, 1.0), (5, 1.0)])
def test_hit_rate(k, expected):
    assert hit_rate_at_k(MATCHES, k) == expected


@pytest.mark.parametrize(("k", "expected"), [(1, 0.0), (2, 1 / 3), (5, 2 / 3)])
def test_recall(k, expected):
    assert recall_at_k(MATCHES, GOLD, k) == pytest.approx(expected)


@pytest.mark.parametrize(("k", "expected"), [(1, 0.0), (2, 1 / 2), (5, 2 / 5)])
def test_precision(k, expected):
    assert precision_at_k(MATCHES, k) == pytest.approx(expected)


def test_precision_divides_by_k_when_fewer_results_come_back():
    assert precision_at_k([{"a"}], 5) == pytest.approx(1 / 5)


def test_reciprocal_rank():
    assert reciprocal_rank(MATCHES) == pytest.approx(1 / 2)
    assert reciprocal_rank([set(), set()]) == 0.0
    assert reciprocal_rank([]) == 0.0


def test_ndcg_binary():
    # DCG = 1/log2(3) + 1/log2(5) = 1.06161, ideal = 1 + 1/log2(3) + 1/log2(4) = 2.13093
    assert ndcg_at_k(MATCHES, GRADES, 5) == pytest.approx(1.06161 / 2.13093, abs=1e-4)


def test_ndcg_graded_penalizes_putting_the_better_item_second():
    # DCG = 1 + 3/log2(3) = 2.89279, ideal = 3 + 1/log2(3) = 3.63093
    assert ndcg_at_k([{"b"}, {"a"}], {"a": 3.0, "b": 1.0}, 2) == pytest.approx(
        2.89279 / 3.63093, abs=1e-4
    )


def test_ndcg_credits_each_gold_item_once():
    # Overlapping chunks can cover the same passage twice. The repeat must earn nothing.
    assert ndcg_at_k([{"a"}, {"a"}], {"a": 1.0, "b": 1.0}, 2) == pytest.approx(
        1 / (1 + 1 / 1.58496), abs=1e-4
    )


def test_ndcg_is_capped_when_one_chunk_covers_all_the_gold():
    assert ndcg_at_k([{"a", "b"}], {"a": 1.0, "b": 1.0}, 2) == 1.0


def test_overlapping_chunks_count_for_precision_but_not_twice_for_recall():
    matches = [{"a"}, {"a"}]
    assert precision_at_k(matches, 2) == 1.0
    assert recall_at_k(matches, {"a", "b"}, 2) == 0.5


@pytest.mark.parametrize(
    ("matches", "k", "expected"),
    [
        ([{"a"}, {"b"}, set()], 2, Failure.OK),
        ([{"a"}, set(), {"b"}], 2, Failure.RANKING),
        ([{"a"}, set(), set()], 2, Failure.RETRIEVAL),
        ([set(), set(), set()], 2, Failure.RETRIEVAL),
        # One gold item is below the cutoff and one is gone: ranking alone would still help.
        ([set(), set(), {"a"}], 2, Failure.RANKING),
    ],
)
def test_classify_failure(matches, k, expected):
    assert classify_failure(matches, {"a", "b"}, k) is expected


@pytest.mark.parametrize(
    "call",
    [
        lambda: hit_rate_at_k([], 0),
        lambda: recall_at_k([], {"a"}, 0),
        lambda: precision_at_k([], -1),
        lambda: ndcg_at_k([], {"a": 1.0}, 0),
        lambda: classify_failure([], {"a"}, 0),
        lambda: recall_at_k([], set(), 1),
        lambda: ndcg_at_k([], {}, 1),
    ],
)
def test_invalid_input_raises(call):
    with pytest.raises(ValueError):
        call()


GOLD_ITEMS = ["a", "b", "c", "d"]
rankings = st.lists(st.frozensets(st.sampled_from(GOLD_ITEMS)), max_size=12)
cutoffs = st.integers(min_value=1, max_value=15)
gold_grades = st.dictionaries(
    st.sampled_from(GOLD_ITEMS), st.integers(min_value=1, max_value=3), min_size=1
)


@given(rankings, gold_grades, cutoffs)
def test_all_metrics_stay_in_unit_range(matches, grades, k):
    matches = [match & grades.keys() for match in matches]
    for value in (
        hit_rate_at_k(matches, k),
        recall_at_k(matches, grades.keys(), k),
        precision_at_k(matches, k),
        reciprocal_rank(matches),
        ndcg_at_k(matches, grades, k),
    ):
        assert 0.0 <= value <= 1.0


@given(rankings, gold_grades, cutoffs)
def test_recall_and_hit_rate_never_drop_as_k_grows(matches, grades, k):
    matches = [match & grades.keys() for match in matches]
    assert recall_at_k(matches, grades.keys(), k) <= recall_at_k(matches, grades.keys(), k + 1)
    assert hit_rate_at_k(matches, k) <= hit_rate_at_k(matches, k + 1)


@given(gold_grades)
def test_ideal_ranking_scores_one(grades):
    ordered = sorted(grades, key=lambda item: -grades[item])
    matches = [{item} for item in ordered]
    k = len(matches)
    assert recall_at_k(matches, grades.keys(), k) == 1.0
    assert precision_at_k(matches, k) == 1.0
    assert reciprocal_rank(matches) == 1.0
    assert ndcg_at_k(matches, grades, k) == pytest.approx(1.0)
    assert classify_failure(matches, grades.keys(), k) is Failure.OK


@given(rankings, gold_grades, cutoffs)
def test_failure_type_agrees_with_recall(matches, grades, k):
    matches = [match & grades.keys() for match in matches]
    failure = classify_failure(matches, grades.keys(), k)
    assert (failure is Failure.OK) == (recall_at_k(matches, grades.keys(), k) == 1.0)
