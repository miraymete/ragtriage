import json
import sys
import textwrap

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from ragtriage import cli
from ragtriage.config import load_config
from ragtriage.generation.generator import Generator
from ragtriage.generation.llm import LLMError, LLMErrorKind
from ragtriage.generation.testing import FakeLLM, fake_completion

runner = CliRunner()

BASE = """\
documents: docs
dataset: qa.jsonl
k: 1
depth: 5
experiments:
  - name: one
    chunker: {type: fixed, size: 50}
    retriever: {type: bm25}
"""
GENERATION = """\
generation:
  generator:
    provider: anthropic
    model: some-model
"""


@pytest.fixture
def workspace(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("Ankara başkenttir. İzmir bir liman kentidir.", "utf-8")
    (tmp_path / "qa.jsonl").write_text(
        '{"id": "q1", "question": "İzmir nasıl bir kent?", '
        '"evidence": [{"doc_id": "a", "quote": "İzmir bir liman kentidir."}]}\n'
        '{"id": "q2", "question": "Efes nerede?"}\n',
        "utf-8",
    )
    return tmp_path


def write(directory, text):
    path = directory / "config.yaml"
    path.write_text(textwrap.dedent(text), "utf-8")
    return path


def test_generation_block_is_optional(workspace):
    assert load_config(write(workspace, BASE)).generation is None


def test_generation_block_loads_with_defaults_and_price(workspace):
    config = load_config(
        write(
            workspace,
            BASE
            + GENERATION
            + "    effort: low\n    price_per_mtok: {input_per_mtok: 1, output_per_mtok: 5}\n",
        )
    )
    generator = config.generation.generator
    assert (generator.model, generator.effort, generator.max_retries) == ("some-model", "low", 2)
    assert generator.price_per_mtok.output_per_mtok == 5


@pytest.mark.parametrize(
    "extra",
    ["    effort: extreme\n", "    temperature: 0\n", "    max_tokens: 0\n"],
    ids=["unknown_effort", "temperature_rejected", "zero_max_tokens"],
)
def test_bad_generation_settings_fail_at_load(workspace, extra):
    with pytest.raises(ValidationError):
        load_config(write(workspace, BASE + GENERATION + extra))


def test_unknown_provider_fails_at_load(workspace):
    text = BASE + GENERATION.replace("anthropic", "magic")
    with pytest.raises(ValidationError):
        load_config(write(workspace, text))


def fake_generator(script):
    def build(spec, k, min_overlap):
        return Generator(FakeLLM(script), k=k, min_overlap=min_overlap)

    return build


def answer(abstained=False, text="Liman kenti."):
    return fake_completion({"abstained": abstained, "answer": text})


def test_cli_runs_generation_and_writes_records(workspace, monkeypatch):
    monkeypatch.setattr(cli, "build_generator", fake_generator([answer(), answer(True, "")]))
    out = workspace / "results.json"
    result = runner.invoke(
        cli.app, ["run", str(write(workspace, BASE + GENERATION)), "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert "| Experiment | Chunks |" in result.output
    assert "Generation (top 1 chunks per question)" in result.output
    assert "| one | 2/2 | 0 | 1/1 | 0/1 |" in result.output
    assert "not checked for correctness" in result.output

    [saved] = json.loads(out.read_text("utf-8"))
    generation = saved["generation"]
    assert generation["n_evaluated"] == 2
    first = generation["records"][0]
    assert first["answer"] == "Liman kenti."
    assert first["contexts"][0]["rank"] == 1
    assert first["prompt_version"]
    assert first["usage"]["input_tokens"] == 100
    assert first["cost_usd"] is None


def test_cli_lists_failed_generations(workspace, monkeypatch):
    script = [fake_completion("broken"), LLMError(LLMErrorKind.SERVER, "boom", status_code=500)]
    monkeypatch.setattr(cli, "build_generator", fake_generator(script))
    result = runner.invoke(cli.app, ["run", str(write(workspace, BASE + GENERATION)), "--failures"])
    assert result.exit_code == 0, result.output
    assert "| one | 0/2 | 2 (api_error 1, invalid_output 1) | n/a | n/a |" in result.output
    assert "- [invalid_output] q1: response is not JSON" in result.output
    assert "- [api_error] q2: server (HTTP 500): boom" in result.output


def test_cli_stops_cleanly_on_a_fatal_error(workspace, monkeypatch):
    error = LLMError(LLMErrorKind.AUTH, "no Anthropic credentials found")
    monkeypatch.setattr(cli, "build_generator", fake_generator([error]))
    result = runner.invoke(cli.app, ["run", str(write(workspace, BASE + GENERATION))])
    assert result.exit_code == 2
    assert "| Experiment | Chunks |" in result.output, "retrieval results are still shown"
    assert "generation stopped early: auth: no Anthropic credentials found" in result.output
    assert "Traceback" not in result.output


def test_cli_explains_a_missing_sdk_without_calling_anything(workspace, monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)
    result = runner.invoke(cli.app, ["run", str(write(workspace, BASE + GENERATION))])
    assert result.exit_code == 2
    assert "ragtriage[anthropic]" in result.output


def test_output_without_generation_is_unchanged(workspace):
    out = workspace / "results.json"
    result = runner.invoke(cli.app, ["run", str(write(workspace, BASE)), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert "Generation" not in result.output
    [saved] = json.loads(out.read_text("utf-8"))
    assert set(saved) == {
        "name",
        "n_chunks",
        "report",
        "context_chars",
        "index_seconds",
        "retrieval_ms",
    }


def test_a_stopped_run_still_prints_and_saves_completed_answers(workspace, monkeypatch):
    script = [answer(), LLMError(LLMErrorKind.NOT_FOUND, "no such model", status_code=404)]
    monkeypatch.setattr(cli, "build_generator", fake_generator(script))
    out = workspace / "results.json"
    result = runner.invoke(
        cli.app, ["run", str(write(workspace, BASE + GENERATION)), "--out", str(out)]
    )
    assert result.exit_code == 2
    assert "| one | 1/2 |" in result.output
    assert "generation stopped early: not_found (HTTP 404): no such model" in result.output
    [saved] = json.loads(out.read_text("utf-8"))
    assert saved["generation"]["stopped"].startswith("not_found")
    assert [r["example_id"] for r in saved["generation"]["records"]] == ["q1"]


def test_fake_responses_are_labeled_in_the_output(workspace, monkeypatch):
    monkeypatch.setattr(cli, "build_generator", fake_generator([answer(), answer(True, "")]))
    result = runner.invoke(cli.app, ["run", str(write(workspace, BASE + GENERATION))])
    assert "scripted fake model" in result.output
    assert "| 0 | 0 | n/a |" in result.output, "no billed tokens, no cost"


def test_cache_path_is_resolved_against_the_config_file(workspace):
    config = load_config(write(workspace, BASE + GENERATION + "  cache: {path: c/r.sqlite}\n"))
    assert config.generation.cache.path == workspace / "c" / "r.sqlite"
    assert config.generation.cache.enabled is True


def test_a_second_cli_run_is_served_from_the_cache(workspace, monkeypatch):
    from ragtriage import runner as runner_module

    calls = []

    def build_llm(spec):
        llm = FakeLLM([answer(), answer(True, "")], model=spec.model)
        calls.append(llm)
        return llm

    monkeypatch.setattr(runner_module, "build_llm", build_llm)
    config = write(workspace, BASE + GENERATION + "  cache: {path: cache.sqlite}\n")
    out = workspace / "results.json"
    for _ in range(2):
        result = runner.invoke(cli.app, ["run", str(config), "--out", str(out)])
        assert result.exit_code == 0, result.output

    assert [len(llm.requests) for llm in calls] == [2, 0]
    assert "2 answers came from the cache" in result.output
    [saved] = json.loads(out.read_text("utf-8"))
    assert [r["source"] for r in saved["generation"]["records"]] == ["cache", "cache"]
    assert [r["original_source"] for r in saved["generation"]["records"]] == ["fake", "fake"]
    assert all(llm.closed for llm in calls)


def test_disabled_cache_makes_every_call(workspace, monkeypatch):
    from ragtriage import runner as runner_module

    calls = []

    def build_llm(spec):
        llm = FakeLLM([answer(), answer(True, "")])
        calls.append(llm)
        return llm

    monkeypatch.setattr(runner_module, "build_llm", build_llm)
    config = write(workspace, BASE + GENERATION + "  cache: {enabled: false, path: c.sqlite}\n")
    for _ in range(2):
        assert runner.invoke(cli.app, ["run", str(config)]).exit_code == 0
    assert [len(llm.requests) for llm in calls] == [2, 2]
    assert not (workspace / "c.sqlite").exists()


def test_an_unusable_cache_file_stops_before_any_call(workspace, monkeypatch):
    from ragtriage import runner as runner_module

    built = []
    monkeypatch.setattr(runner_module, "build_llm", lambda spec: built.append(1) or FakeLLM([]))
    (workspace / "bad.sqlite").write_text("not a database, only some plain text", "utf-8")
    config = write(workspace, BASE + GENERATION + "  cache: {path: bad.sqlite}\n")
    result = runner.invoke(cli.app, ["run", str(config)])
    assert result.exit_code == 2
    assert "generation not started" in result.output
    assert "not a readable SQLite cache" in result.output
    assert built == [], "no client was created"
    assert (workspace / "bad.sqlite").exists()


def test_cli_shows_faithfulness_counts_and_separate_judge_usage(workspace, monkeypatch):
    from ragtriage.generation.judge import FaithfulnessJudge

    monkeypatch.setattr(cli, "build_generator", fake_generator([answer(), answer(True, "")]))
    judge_script = [
        fake_completion({"claims": ["İzmir bir liman kentidir."]}),
        fake_completion(
            {
                "verdicts": [
                    {
                        "claim_id": "c1",
                        "verdict": "supported",
                        "passage_id": "p1",
                        "quote": "liman kentidir",
                    }
                ]
            }
        ),
    ]
    judges = []

    def build_judge(spec):
        judge_llm = FakeLLM(judge_script, model="judge-model")
        judges.append(judge_llm)
        return FaithfulnessJudge(judge_llm)

    monkeypatch.setattr(cli, "build_judge", build_judge)
    out = workspace / "results.json"
    config = BASE + GENERATION + "  judge: {provider: anthropic, model: judge-model}\n"
    result = runner.invoke(cli.app, ["run", str(write(workspace, config)), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert "Faithfulness (claims supported" in result.output
    assert "| one | 1.000 | 1 | 1 (abstained 1) | 0 | 1 / 0 / 0 |" in result.output
    assert "does not measure correctness" in result.output
    assert judges[0].closed

    [saved] = json.loads(out.read_text("utf-8"))
    record = saved["generation"]["records"][0]
    faith = record["faithfulness"]
    assert faith["outcome"] == {"status": "scored", "value": 1.0}
    assert faith["claims"][0]["quote"] == "liman kentidir"
    assert faith["calls"][0]["prompt_version"].startswith("faithfulness-claims")
    assert saved["generation"]["faithfulness"]["judge_usage"]["n_fake"] == 2


def test_judge_is_optional_and_loads(workspace):
    plain = load_config(write(workspace, BASE + GENERATION))
    assert plain.generation.judge is None
    judged = load_config(
        write(workspace, BASE + GENERATION + "  judge: {provider: anthropic, model: j}\n")
    )
    assert judged.generation.judge.model == "j"


RELEVANCE = """\
  relevance:
    judge: {provider: anthropic, model: relevance-model}
    embedder: {type: hashing, dim: 64}
"""


def test_relevance_needs_an_explicit_embedder(workspace):
    text = BASE + GENERATION + "  relevance:\n    judge: {provider: anthropic, model: r}\n"
    with pytest.raises(ValidationError, match="embedder"):
        load_config(write(workspace, text))


def test_cli_shows_relevance_and_stores_questions_and_embedder(workspace, monkeypatch):
    import numpy as np

    from ragtriage.generation.relevance import RelevanceJudge

    class Stub:
        def embed_query(self, text):
            return np.array([1.0, 0.0])

        def embed_documents(self, texts):
            raise AssertionError

    monkeypatch.setattr(cli, "build_generator", fake_generator([answer(), answer(True, "")]))
    script = [fake_completion({"questions": ["Bir?", "İki?", "Üç?"]})]
    monkeypatch.setattr(
        cli,
        "build_relevance",
        lambda spec: RelevanceJudge(FakeLLM(script), Stub(), {"type": "stub"}),
    )
    out = workspace / "results.json"
    result = runner.invoke(
        cli.app, ["run", str(write(workspace, BASE + GENERATION + RELEVANCE)), "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert "Answer relevance (questions written from the answer alone)" in result.output
    assert "| one | 1.000 | 1 | 1 (abstained 1) | 0 |" in result.output
    assert "semantic-similarity proxy" in result.output
    assert "Faithfulness (" not in result.output, "faithfulness was not configured"
    [saved] = json.loads(out.read_text("utf-8"))
    rel = saved["generation"]["records"][0]["relevance"]
    assert rel["questions"] == ["Bir?", "İki?", "Üç?"]
    assert rel["similarities"] == [1.0, 1.0, 1.0]
    assert rel["embedder"] == {"type": "stub"}
    assert rel["call"]["prompt_version"].startswith("relevance-questions")


def test_same_model_judges_trigger_a_warning(workspace):
    same = (
        BASE
        + GENERATION
        + "  judge: {provider: anthropic, model: some-model}\n"
        + RELEVANCE.replace("relevance-model", "some-model")
    )
    warnings = cli.same_model_warnings(load_config(write(workspace, same)))
    assert len(warnings) == 2
    assert "faithfulness judge uses the same model" in warnings[0]
    assert "does not remove it" in warnings[0]
    different = BASE + GENERATION + RELEVANCE
    assert cli.same_model_warnings(load_config(write(workspace, different))) == []
