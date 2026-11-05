import pytest

from ragtriage.matching import covers, match_chunks
from ragtriage.models import Chunk, Evidence


def chunk(start, end, doc_id="d"):
    return Chunk(doc_id=doc_id, start=start, end=end, text="x" * (end - start))


GOLD = Evidence(doc_id="d", start=100, end=200)


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        (chunk(0, 1000), True),  # large chunk holds the whole passage
        (chunk(120, 140), True),  # small chunk sits inside the passage
        (chunk(50, 150), True),  # half of the passage, exactly at the threshold
        (chunk(0, 110), False),  # clips the edge of the passage
        (chunk(200, 300), False),  # touching is not overlapping
        (chunk(100, 200, doc_id="other"), False),  # same offsets, different document
    ],
)
def test_covers(candidate, expected):
    assert covers(candidate, GOLD) is expected


@pytest.mark.parametrize("value", [0, -0.1, 1.1])
def test_covers_rejects_bad_threshold(value):
    with pytest.raises(ValueError):
        covers(chunk(0, 10), GOLD, min_overlap=value)


def test_match_chunks_reports_gold_per_rank():
    gold = [GOLD, Evidence(doc_id="d", start=500, end=600)]
    assert match_chunks(gold, [chunk(0, 50), chunk(500, 600), chunk(100, 600)]) == [
        frozenset(),
        frozenset({1}),
        frozenset({0, 1}),
    ]


def test_duplicate_chunks_keep_only_their_best_rank():
    assert match_chunks([GOLD], [chunk(100, 200), chunk(100, 200), chunk(0, 10)]) == [
        frozenset({0}),
        frozenset(),
    ]
