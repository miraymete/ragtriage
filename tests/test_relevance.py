"""Answer relevance with a scripted judge and a table embedder whose cosines are known exactly.

These tests check validation, arithmetic and bookkeeping. They say nothing about the quality of a
real judge or a real embedding model.
"""

import asyncio
import math

import numpy as np
import pytest

from ragtriage import Evidence, QAExample, RetrievalRun, RetrievedChunk
from ragtriage.generation.cache import CachedLLM, ResponseCache
from ragtriage.generation.costs import Price
from ragtriage.generation.generator import GenerationStatus, Generator
from ragtriage.generation.judge import FaithfulnessJudge
from ragtriage.generation.llm import LLMError, LLMErrorKind, StopReason
from ragtriage.generation.pipeline import run_generation
from ragtriage.generation.relevance import (
    QUESTIONS_PROMPT_VERSION,
    RelevanceJudge,
    cosine_similarities,
    parse_questions,
)
from ragtriage.generation.testing import FakeLLM, fake_completion
from ragtriage.outcomes import Failed, NotApplicable, Scored

QUESTION = "İznik çinisinin hamurunda ne var?"
ANSWER = "Hamurunda yüksek oranda kuvars bulunur."
GENERATED = ["Hamurda ne bulunur?", "Çini neden serttir?", "Kuvars nerede kullanılır?"]


class TableEmbedder:
    """Returns hand-picked vectors for `embed_query`; `embed_documents` must not be used."""

    def __init__(self, table):
        self.table = table
        self.queries = []

    def embed_query(self, text):
        self.queries.append(text)
        return np.array(self.table[text], dtype=np.float32)

    def embed_documents(self, texts):
        raise AssertionError("relevance must embed every question as a query")


def table(original, *generated):
    return {QUESTION: original, **dict(zip(GENERATED, generated, strict=True))}


def questions(*items):
    return fake_completion({"questions": list(items)})


def assert_scored(outcome, expected):
    assert isinstance(outcome, Scored)
    assert outcome.value == pytest.approx(expected)


SETTINGS = {"type": "stub", "dim": 2}


def evaluate(script, vectors, answer=ANSWER, abstained=False, price=None):
    llm = FakeLLM(script, model="judge-model")
    embedder = TableEmbedder(vectors)
    judge = RelevanceJudge(llm, embedder, SETTINGS, price=price)
    result = asyncio.run(judge.evaluate(QUESTION, answer, abstained))
    return result, llm, embedder


def test_score_is_the_mean_cosine_worked_out_by_hand():
    # cosines: 1.0, 0.6, 0.6 -> mean 2.2 / 3
    vectors = table([1, 0], [2, 0], [0.6, 0.8], [3, -4])
    result, _, embedder = evaluate([questions(*GENERATED)], vectors)
    assert result.similarities == pytest.approx((1.0, 0.6, 0.6))
    assert_scored(result.outcome, 2.2 / 3)
    assert result.questions == tuple(GENERATED)
    assert result.call.prompt_version == QUESTIONS_PROMPT_VERSION
    assert result.call.model == "fake-model", "the model the response reports"
    assert result.embedder == SETTINGS
    assert embedder.queries == [QUESTION, *GENERATED], "one embed_query path for all four"


def test_negative_similarity_is_kept_not_clipped():
    vectors = table([1, 0], [-1, 0], [0, 1], [-0.6, -0.8])
    result, _, _ = evaluate([questions(*GENERATED)], vectors)
    assert result.similarities == pytest.approx((-1.0, 0.0, -0.6))
    assert result.outcome.value == pytest.approx(-1.6 / 3)


def test_all_opposite_questions_score_minus_one():
    vectors = table([0, 2], [0, -1], [0, -3], [0, -0.5])
    result, _, _ = evaluate([questions(*GENERATED)], vectors)
    assert_scored(result.outcome, -1.0)


@pytest.mark.parametrize(
    ("vectors", "problem"),
    [
        (table([0, 0], [1, 0], [1, 0], [1, 0]), "zero norm"),
        (table([1, 0], [1, 0], [0, 0], [1, 0]), "zero norm"),
        (table([1, 0], [math.nan, 0], [1, 0], [1, 0]), "NaN or infinity"),
        (table([1, 0], [1, 0], [math.inf, 1], [1, 0]), "NaN or infinity"),
        (table([1, 0], [1, 0, 0], [1, 0], [1, 0]), "dimensions"),
        (table([[1, 0]], [1, 0], [1, 0], [1, 0]), "expected a vector"),
    ],
    ids=["zero_original", "zero_generated", "nan", "inf", "dim_mismatch", "matrix"],
)
def test_bad_embeddings_fail_instead_of_scoring(vectors, problem):
    result, _, _ = evaluate([questions(*GENERATED)], vectors)
    assert isinstance(result.outcome, Failed)
    assert result.outcome.reason.startswith("embedding failed:")
    assert problem in result.outcome.reason
    assert result.similarities == ()
    assert result.questions == tuple(GENERATED), "the judge output is still kept"


def test_the_judge_sees_only_the_answer():
    vectors = table([1, 0], [1, 0], [1, 0], [1, 0])
    _, llm, _ = evaluate([questions(*GENERATED)], vectors)
    [request] = llm.requests
    assert ANSWER in request.prompt
    assert QUESTION not in request.prompt + request.system
    assert "<question>" not in request.prompt and "<passage" not in request.prompt
    assert "güvenilmeyen veri" in request.system


@pytest.mark.parametrize(
    "response",
    [
        questions("Bir?", "İki?"),
        questions("Bir?", "İki?", "Üç?", "Dört?"),
        questions("Bir?", "  ", "Üç?"),
        questions("Hamurda ne var?", "hamurda  NE var", "Üç?"),
        questions("KIRMIZI mı?", "kırmızı mı", "Üç?"),
        fake_completion({"questions": "tek dize"}),
        fake_completion({"questions": [], "extra": 1}),
        fake_completion("not json"),
        questions(*GENERATED).model_copy(update={"stop_reason": StopReason.MAX_TOKENS}),
        questions(*GENERATED).model_copy(update={"stop_reason": StopReason.REFUSAL}),
    ],
    ids=[
        "two",
        "four",
        "empty",
        "duplicate_after_normalization",
        "turkish_casing_duplicate",
        "wrong_type",
        "extra_key",
        "not_json",
        "truncated",
        "refused",
    ],
)
def test_invalid_questions_fail_with_no_extra_call(response):
    result, llm, embedder = evaluate([response], table([1, 0], [1, 0], [1, 0], [1, 0]))
    assert isinstance(result.outcome, Failed)
    assert result.outcome.reason.startswith("questions stage failed:")
    assert len(llm.requests) == 1, "no automatic retry"
    assert embedder.queries == [], "nothing is embedded after a failed stage"


def test_parse_questions_returns_stripped_questions():
    assert parse_questions('{"questions": [" Bir? ", "İki?", "Üç?"]}') == ["Bir?", "İki?", "Üç?"]


@pytest.mark.parametrize(
    ("answer", "abstained", "reason"),
    [("Bağlamda yok.", True, "abstained"), ("  ", False, "empty_answer")],
)
def test_abstained_and_empty_answers_are_not_applicable(answer, abstained, reason):
    result, llm, _ = evaluate([], {}, answer=answer, abstained=abstained)
    assert result.outcome == NotApplicable(reason=reason)
    assert llm.requests == []


def test_a_non_fatal_api_error_fails_and_a_fatal_one_propagates():
    error = LLMError(LLMErrorKind.SERVER, "boom", status_code=500)
    result, _, _ = evaluate([error], {})
    assert isinstance(result.outcome, Failed) and "server" in result.outcome.reason
    with pytest.raises(LLMError):
        evaluate([LLMError(LLMErrorKind.AUTH, "no key")], {})


def test_cosine_helper_rejects_mismatched_query_dimensions():
    embedder = TableEmbedder({"a": [1, 0], "b": [1, 0, 0]})
    with pytest.raises(ValueError, match="dimensions"):
        cosine_similarities(embedder, "a", ["b"])


# Pipeline integration

P1 = "İznik çinisi yüksek oranda kuvars içeren hamuru sayesinde serttir."
CHUNKS = (RetrievedChunk(doc_id="iznik", start=0, end=len(P1), text=P1, score=1.0),)
EXAMPLE = QAExample(
    id="q",
    question=QUESTION,
    evidence=(Evidence(doc_id="iznik", start=0, end=10),),
    reference_answer="GİZLİ REFERANS",
)
VECTORS = table([1, 0], [1, 0], [0.6, 0.8], [0, 1])


def gen_answer(text=ANSWER, abstained=False, **kwargs):
    return fake_completion({"abstained": abstained, "answer": text}, **kwargs)


def pipeline(gen_script, relevance_script, faith_script=None, cache_path=None, price=None):
    gen_llm = FakeLLM(gen_script)
    rel_llm = FakeLLM(relevance_script, model="relevance-model")
    faith_llm = FakeLLM(faith_script or [], model="faith-model")
    clients = [gen_llm, rel_llm, faith_llm]
    if cache_path is not None:
        clients = [CachedLLM(c, ResponseCache(cache_path)) for c in clients]
    embedder = TableEmbedder(VECTORS)
    generator = Generator(clients[0], k=1, price=price)
    relevance = RelevanceJudge(clients[1], embedder, SETTINGS, price=price)
    judge = FaithfulnessJudge(clients[2], price=price) if faith_script is not None else None
    runs = [[RetrievalRun(example_id=EXAMPLE.id, chunks=CHUNKS)]]
    [report] = asyncio.run(run_generation(generator, [EXAMPLE], runs, judge, relevance))
    return report, gen_llm, rel_llm, faith_llm


def test_relevance_runs_without_faithfulness_and_never_sees_labels_or_context():
    report, _, rel_llm, _ = pipeline([gen_answer()], [questions(*GENERATED)])
    [record] = report.records
    assert record.faithfulness is None
    assert_scored(record.relevance.outcome, 1.6 / 3)
    prompt = rel_llm.requests[0].prompt
    assert "GİZLİ REFERANS" not in prompt and P1 not in prompt and QUESTION not in prompt
    assert report.faithfulness is None
    assert report.relevance.n_scored == 1


def test_a_faithfulness_failure_does_not_block_relevance():
    report, _, _, _ = pipeline(
        [gen_answer()], [questions(*GENERATED)], faith_script=[fake_completion("bozuk")]
    )
    [record] = report.records
    assert isinstance(record.faithfulness.outcome, Failed)
    assert isinstance(record.relevance.outcome, Scored)


def test_a_relevance_failure_does_not_block_faithfulness():
    faith = [
        fake_completion({"claims": ["Hamurunda kuvars bulunur."]}),
        fake_completion(
            {
                "verdicts": [
                    {
                        "claim_id": "c1",
                        "verdict": "supported",
                        "passage_id": "p1",
                        "quote": "kuvars içeren hamuru",
                    }
                ]
            }
        ),
    ]
    report, _, _, _ = pipeline([gen_answer()], [fake_completion("bozuk")], faith_script=faith)
    [record] = report.records
    assert record.faithfulness.outcome == Scored(value=1.0)
    assert isinstance(record.relevance.outcome, Failed)
    assert (report.relevance.n_failed, report.faithfulness.n_scored) == (1, 1)


def test_failed_generation_and_abstention_are_not_applicable():
    report, _, rel_llm, _ = pipeline([fake_completion("bozuk")], [])
    assert report.records[0].status is GenerationStatus.INVALID_OUTPUT
    assert report.records[0].relevance.outcome == NotApplicable(reason="no_valid_answer")
    report, _, rel_llm, _ = pipeline([gen_answer("Yok.", abstained=True)], [])
    assert report.records[0].relevance.outcome == NotApplicable(reason="abstained")
    assert rel_llm.requests == []
    assert report.relevance.not_applicable == {"abstained": 1}


def test_usage_is_separate_for_generator_faithfulness_and_relevance():
    price = Price(input_per_mtok=1.0, output_per_mtok=0.0)
    api = {"source": "api"}
    faith = [
        fake_completion({"claims": []}, input_tokens=30).model_copy(update=api),
    ]
    report, _, _, _ = pipeline(
        [gen_answer(input_tokens=10, source="api")],
        [questions(*GENERATED).model_copy(update={**api, "usage": gen_answer().usage})],
        faith_script=faith,
        price=price,
    )
    assert report.billed_usage.input_tokens == 10
    assert report.faithfulness.judge_usage.billed_usage.input_tokens == 30
    assert report.relevance.judge_usage.billed_usage.input_tokens == 100
    assert report.cost_usd == pytest.approx(10 / 1_000_000)
    assert report.faithfulness.judge_usage.cost_usd == pytest.approx(30 / 1_000_000)
    assert report.relevance.judge_usage.cost_usd == pytest.approx(100 / 1_000_000)


def test_relevance_questions_are_cached_after_validation_and_stay_fake(tmp_path):
    path = tmp_path / "cache.sqlite"
    pipeline([gen_answer()], [questions(*GENERATED)], cache_path=path)

    report, gen_llm, rel_llm, _ = pipeline([], [], cache_path=path)
    assert gen_llm.requests == [] and rel_llm.requests == []
    [record] = report.records
    assert (record.relevance.call.source, record.relevance.call.original_source) == (
        "cache",
        "fake",
    )
    assert_scored(record.relevance.outcome, 1.6 / 3)
    usage = report.relevance.judge_usage
    assert (usage.n_cached, usage.n_api_calls, usage.billed_usage.input_tokens) == (1, 0, 0)


def test_invalid_questions_are_not_cached(tmp_path):
    path = tmp_path / "cache.sqlite"
    pipeline([gen_answer()], [questions("Bir?", "İki?")], cache_path=path)
    _, _, rel_llm, _ = pipeline([], [questions(*GENERATED)], cache_path=path)
    assert len(rel_llm.requests) == 1, "the invalid response was not reused"


def test_a_fatal_relevance_error_keeps_the_record_and_stops():
    report, _, _, _ = pipeline([gen_answer()], [LLMError(LLMErrorKind.NOT_FOUND, "no model")])
    assert report.stopped.startswith("not_found")
    [record] = report.records
    assert record.relevance.outcome.reason.startswith("judge stopped:")
