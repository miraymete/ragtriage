import pytest

from ragtriage import (
    Document,
    Evidence,
    Failure,
    QAExample,
    RetrievalRun,
    RetrievedChunk,
    evaluate_retrieval,
)
from ragtriage.pipelines.chunking import chunk_fixed


def retrieved(start, end):
    return RetrievedChunk(doc_id="d", start=start, end=end, text="x" * (end - start), score=1.0)


def example(example_id, *spans):
    evidence = tuple(Evidence(doc_id="d", start=s, end=e) for s, e in spans)
    return QAExample(id=example_id, question="?", evidence=evidence)


def test_means_and_failures_across_examples():
    examples = [example("hit", (0, 10)), example("ranked_out", (100, 110)), example("gone", (5, 9))]
    runs = [
        RetrievalRun(example_id="hit", chunks=(retrieved(0, 10), retrieved(50, 60))),
        RetrievalRun(example_id="ranked_out", chunks=(retrieved(50, 60), retrieved(100, 110))),
        RetrievalRun(example_id="gone", chunks=(retrieved(50, 60),)),
    ]
    report = evaluate_retrieval(examples, runs, ks=(1, 2), triage_k=1)
    assert report.mean["recall@1"] == pytest.approx(1 / 3)
    assert report.mean["recall@2"] == pytest.approx(2 / 3)
    assert report.mean["mrr"] == pytest.approx((1 + 1 / 2 + 0) / 3)
    assert report.failures == {Failure.OK: 1, Failure.RANKING: 1, Failure.RETRIEVAL: 1}
    assert report.triage_k == 1


def test_unanswerable_questions_are_counted_but_do_not_lower_the_mean():
    examples = [example("hit", (0, 10)), example("no_answer")]
    runs = [
        RetrievalRun(example_id="hit", chunks=(retrieved(0, 10),)),
        RetrievalRun(example_id="no_answer", chunks=(retrieved(50, 60),)),
    ]
    report = evaluate_retrieval(examples, runs, ks=(1,))
    assert report.mean["recall@1"] == 1.0
    assert report.n_unanswerable_skipped == 1
    assert [score.example_id for score in report.per_example] == ["hit"]


def test_gold_labels_stay_valid_across_chunk_sizes():
    """The reason evidence is a span: one label set must score any chunking."""
    document = Document(id="d", text="abcdefghij" * 100)
    examples = [example("q", (420, 460))]
    for size in (64, 128, 512):
        chunks = [c for c in chunk_fixed(document, size) if c.start <= 430 < c.end]
        run = RetrievalRun(
            example_id="q",
            chunks=tuple(RetrievedChunk(**c.model_dump(), score=1.0) for c in chunks),
        )
        assert evaluate_retrieval(examples, [run], ks=(1,)).mean["recall@1"] == 1.0


@pytest.mark.parametrize(
    ("examples", "runs"),
    [
        ([example("a", (0, 1))], []),
        ([], [RetrievalRun(example_id="a", chunks=())]),
        (
            [example("a", (0, 1))],
            [RetrievalRun(example_id="a", chunks=()), RetrievalRun(example_id="a", chunks=())],
        ),
        ([example("a", (0, 1)), example("a", (0, 1))], [RetrievalRun(example_id="a", chunks=())]),
        ([example("a")], [RetrievalRun(example_id="a", chunks=())]),
    ],
    ids=["missing_run", "extra_run", "duplicate_run", "duplicate_example", "nothing_answerable"],
)
def test_misaligned_input_raises(examples, runs):
    with pytest.raises(ValueError):
        evaluate_retrieval(examples, runs)


def test_empty_run_scores_zero_instead_of_crashing():
    report = evaluate_retrieval(
        [example("a", (0, 10))], [RetrievalRun(example_id="a", chunks=())], ks=(3,)
    )
    assert report.mean == {
        "mrr": 0.0,
        "hit_rate@3": 0.0,
        "recall@3": 0.0,
        "precision@3": 0.0,
        "ndcg@3": 0.0,
    }
    assert report.failures == {Failure.RETRIEVAL: 1}
