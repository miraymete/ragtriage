"""The SQLite response cache. Every test uses a temporary file and a scripted model."""

import asyncio
import json
import sqlite3

import pytest

from ragtriage import Evidence, QAExample, RetrievalRun, RetrievedChunk
from ragtriage.generation.cache import (
    CACHE_SCHEMA_VERSION,
    CachedLLM,
    CacheError,
    ResponseCache,
    cache_key,
)
from ragtriage.generation.generator import GenerationStatus, Generator, Price
from ragtriage.generation.llm import LLMError, LLMErrorKind, LLMRequest, StopReason
from ragtriage.generation.pipeline import generate_reports, run_generation
from ragtriage.generation.testing import FakeLLM, fake_completion

GOLD = Evidence(doc_id="d", start=0, end=10)
EXAMPLE = QAExample(id="a", question="Soru nedir?", evidence=(GOLD,))
OTHER = QAExample(id="b", question="Başka soru?", evidence=(GOLD,))
CHUNKS = (RetrievedChunk(doc_id="d", start=0, end=11, text="altın parça", score=1.0),)
PRICE = Price(input_per_mtok=1.0, output_per_mtok=5.0)


def answer(**kwargs):
    return fake_completion({"abstained": False, "answer": "Cevap."}, **kwargs)


def runs(*examples):
    return [[RetrievalRun(example_id=e.id, chunks=CHUNKS) for e in examples]]


class SettingsLLM(FakeLLM):
    """A scripted client whose settings change the response, like a provider's effort."""

    def __init__(self, script, settings, **kwargs):
        super().__init__(script, **kwargs)
        self._settings = settings

    def response_settings(self):
        return self._settings


def run_once(path, inner, examples=(EXAMPLE,), price=None, max_tokens=100):
    cached = CachedLLM(inner, ResponseCache(path))
    generator = Generator(cached, k=1, max_tokens=max_tokens, price=price)
    [report] = asyncio.run(run_generation(generator, list(examples), runs(*examples)))
    return report


def test_a_repeated_request_is_served_from_the_cache(tmp_path):
    path = tmp_path / "cache.sqlite"
    first_llm = FakeLLM([answer(), answer()])
    first = run_once(path, first_llm, examples=(EXAMPLE, EXAMPLE))

    assert len(first_llm.requests) == 1, "the second identical request never reached the model"
    assert [r.source for r in first.records] == ["fake", "cache"]
    assert first.records[1].answer == "Cevap."
    assert first.records[1].status is GenerationStatus.OK


def test_an_entry_survives_a_new_client_and_connection(tmp_path):
    path = tmp_path / "cache.sqlite"
    run_once(path, FakeLLM([answer()]))

    second_llm = FakeLLM([])  # any request would fail the test
    report = run_once(path, second_llm)
    assert second_llm.requests == []
    assert report.records[0].source == "cache"


def test_fake_responses_stay_fake_when_replayed(tmp_path):
    path = tmp_path / "cache.sqlite"
    run_once(path, FakeLLM([answer()]), price=PRICE)
    report = run_once(path, FakeLLM([]), price=PRICE)

    [record] = report.records
    assert (record.source, record.original_source) == ("cache", "fake")
    assert record.cost_usd is None, "a fake response never has a real cost"
    assert report.n_fake == 1
    assert report.cached_original_usage.input_tokens == 0, "fake usage is not real usage"
    assert report.billed_usage.input_tokens == 0


def test_a_cache_hit_adds_no_new_usage_or_cost(tmp_path):
    path = tmp_path / "cache.sqlite"
    first = run_once(path, FakeLLM([answer(source="api", input_tokens=100)]), price=PRICE)
    assert first.billed_usage.input_tokens == 100
    assert first.cost_usd == pytest.approx((100 + 20 * 5.0) / 1_000_000)

    second = run_once(path, FakeLLM([]), price=PRICE)
    [record] = second.records
    assert (record.source, record.original_source) == ("cache", "api")
    assert record.usage.input_tokens == 100, "the original usage is kept for reference"
    assert record.cost_usd == 0.0
    assert second.billed_usage.input_tokens == 0
    assert second.cached_original_usage.input_tokens == 100
    assert (second.n_api_calls, second.n_cached) == (0, 1)
    assert second.cost_usd == 0.0


@pytest.mark.parametrize(
    "failure",
    [
        answer(stop_reason=StopReason.REFUSAL),
        answer(stop_reason=StopReason.MAX_TOKENS),
        fake_completion("not json"),
        fake_completion({"abstained": False, "answer": " "}),
        LLMError(LLMErrorKind.SERVER, "boom", status_code=500),
    ],
    ids=["refusal", "truncated", "invalid_json", "empty_answer", "api_error"],
)
def test_failed_responses_are_never_cached(tmp_path, failure):
    path = tmp_path / "cache.sqlite"
    run_once(path, FakeLLM([failure]))

    retry = FakeLLM([answer()])
    report = run_once(path, retry)
    assert len(retry.requests) == 1, "nothing was cached, so the request is made again"
    assert report.records[0].source == "fake"


def request(**changes):
    fields = {
        "system": "sys",
        "prompt": "soru",
        "max_tokens": 100,
        "json_schema": {"type": "object"},
        "prompt_version": "v1",
    }
    return LLMRequest(**(fields | changes))


BASE_KEY = cache_key("fake", "m", {"effort": "low"}, request())


@pytest.mark.parametrize(
    "key",
    [
        cache_key("other", "m", {"effort": "low"}, request()),
        cache_key("fake", "m2", {"effort": "low"}, request()),
        cache_key("fake", "m", {"effort": "high"}, request()),
        cache_key("fake", "m", {"effort": "low"}, request(system="sys2")),
        cache_key("fake", "m", {"effort": "low"}, request(prompt="soru2")),
        cache_key("fake", "m", {"effort": "low"}, request(max_tokens=101)),
        cache_key("fake", "m", {"effort": "low"}, request(json_schema={"type": "array"})),
        cache_key("fake", "m", {"effort": "low"}, request(prompt_version="v2")),
    ],
    ids=["provider", "model", "setting", "system", "prompt", "max_tokens", "schema", "version"],
)
def test_any_change_that_can_alter_the_response_changes_the_key(key):
    assert key != BASE_KEY


def test_the_key_is_deterministic():
    reordered = request(json_schema={"b": 1, "a": 2})
    assert cache_key("p", "m", {"y": 1, "x": 2}, reordered) == cache_key(
        "p", "m", {"x": 2, "y": 1}, request(json_schema={"a": 2, "b": 1})
    )


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ({"model": "m1"}, {"model": "m2"}),
        ({"settings": {"effort": "low"}}, {"settings": {"effort": "high"}}),
        ({"max_tokens": 100}, {"max_tokens": 200}),
    ],
    ids=["model", "setting", "max_tokens"],
)
def test_a_changed_model_or_setting_misses_end_to_end(tmp_path, first, second):
    path = tmp_path / "cache.sqlite"

    def make(options, script):
        llm = SettingsLLM(script, options.get("settings", {}), model=options.get("model", "m"))
        return llm, options.get("max_tokens", 100)

    llm, max_tokens = make(first, [answer()])
    run_once(path, llm, max_tokens=max_tokens)
    llm, max_tokens = make(second, [answer()])
    report = run_once(path, llm, max_tokens=max_tokens)
    assert len(llm.requests) == 1, "a cache miss reached the model"
    assert report.records[0].source == "fake"


def test_a_changed_question_misses(tmp_path):
    path = tmp_path / "cache.sqlite"
    run_once(path, FakeLLM([answer()]))
    llm = FakeLLM([answer()])
    run_once(path, llm, examples=(OTHER,))
    assert len(llm.requests) == 1


def test_remember_ignores_cache_hits_and_unfinished_responses(tmp_path):
    cache = ResponseCache(tmp_path / "cache.sqlite")
    cached = CachedLLM(FakeLLM([]), cache)
    req = request()
    asyncio.run(cached.remember(req, answer(stop_reason=StopReason.MAX_TOKENS)))
    asyncio.run(cached.remember(req, answer(source="cache")))
    assert cache.get(cache_key("fake", "fake-model", {}, req), "fake", "fake-model") is None
    cache.close()


def test_a_corrupt_entry_raises_instead_of_being_used(tmp_path):
    path = tmp_path / "cache.sqlite"
    run_once(path, FakeLLM([answer()]))
    with sqlite3.connect(path) as db:
        db.execute("UPDATE responses SET payload = ?", (json.dumps({"text": 1}),))
    db.close()

    llm = FakeLLM([answer()])
    cached = CachedLLM(llm, ResponseCache(path))
    generator = Generator(cached, k=1, max_tokens=100)
    [report] = asyncio.run(run_generation(generator, [EXAMPLE], runs(EXAMPLE)))
    assert report.stopped.startswith("cache error:")
    assert "corrupt" in report.stopped
    assert llm.requests == [], "no silent fall back to a (paid) call"
    assert path.exists(), "nothing is deleted"


def test_an_entry_for_another_model_under_the_same_key_is_rejected(tmp_path):
    cache = ResponseCache(tmp_path / "cache.sqlite")
    cached = CachedLLM(FakeLLM([]), cache)
    asyncio.run(cached.remember(request(), answer()))
    key = cache_key("fake", "fake-model", {}, request())
    with pytest.raises(CacheError, match="belongs to"):
        cache.get(key, "fake", "another-model")
    cache.close()


def test_a_file_that_is_not_sqlite_is_rejected_and_kept(tmp_path):
    path = tmp_path / "cache.sqlite"
    path.write_text("not a database at all, just text that is long enough", "utf-8")
    with pytest.raises(CacheError, match="not a readable SQLite cache"):
        ResponseCache(path)
    assert path.read_text("utf-8").startswith("not a database")


def test_a_cache_of_another_version_is_rejected(tmp_path):
    path = tmp_path / "cache.sqlite"
    ResponseCache(path).close()
    with sqlite3.connect(path) as db:
        db.execute(f"PRAGMA user_version = {CACHE_SCHEMA_VERSION + 1}")
    db.close()
    with pytest.raises(CacheError, match="version"):
        ResponseCache(path)


def test_the_connection_is_closed_on_success_and_on_failure(tmp_path):
    cache = ResponseCache(tmp_path / "ok.sqlite")
    inner = FakeLLM([answer()])
    generator = Generator(CachedLLM(inner, cache), k=1)
    asyncio.run(run_generation(generator, [EXAMPLE], runs(EXAMPLE)))
    assert cache.closed and inner.closed

    def bug(request):
        raise RuntimeError("a programming error")

    cache = ResponseCache(tmp_path / "bug.sqlite")
    inner = FakeLLM(bug)
    generator = Generator(CachedLLM(inner, cache), k=1)
    with pytest.raises(RuntimeError):
        asyncio.run(run_generation(generator, [EXAMPLE], runs(EXAMPLE)))
    assert cache.closed and inner.closed


def test_without_a_cache_nothing_is_stored(tmp_path):
    llm = FakeLLM([answer(), answer()])
    generator = Generator(llm, k=1)
    asyncio.run(generate_reports(generator, [EXAMPLE, EXAMPLE], runs(EXAMPLE, EXAMPLE)))
    assert len(llm.requests) == 2
    assert list(tmp_path.iterdir()) == []
