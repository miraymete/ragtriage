import json
import textwrap

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from ragtriage.cli import app
from ragtriage.config import (
    FixedChunker,
    RecursiveChunker,
    SentenceChunker,
    VectorSpec,
    load_config,
)
from ragtriage.runner import run_experiments

runner = CliRunner()

EXPERIMENT = """
  - name: {name}
    chunker: {{type: fixed, size: 50}}
    retriever: {{type: bm25}}
"""


@pytest.fixture
def workspace(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("Ankara başkenttir. " * 3 + "İzmir bir liman kentidir.", "utf-8")
    (tmp_path / "qa.jsonl").write_text(
        '{"id": "q1", "question": "İzmir nasıl bir kent?", '
        '"evidence": [{"doc_id": "a", "quote": "İzmir bir liman kentidir."}]}\n'
        '{"id": "q2", "question": "Efes nerede?"}\n',
        "utf-8",
    )
    return tmp_path


def write_config(directory, body=None):
    path = directory / "config.yaml"
    default = "documents: docs\ndataset: qa.jsonl\nk: 1\ndepth: 5\nexperiments:" + (
        EXPERIMENT.format(name="one")
    )
    path.write_text(textwrap.dedent(body) if body else default, "utf-8")
    return path


def test_load_config_builds_typed_specs_and_resolves_paths_from_the_file(workspace):
    path = write_config(
        workspace,
        """\
        documents: docs
        dataset: qa.jsonl
        experiments:
          - name: f
            chunker: {type: fixed, size: 10}
            retriever: {type: bm25}
          - name: r
            chunker: {type: recursive, size: 10, overlap: 2}
            retriever: {type: bm25, tokenizer: naive}
          - name: s
            chunker: {type: sentence, max_chars: 10}
            retriever: {type: vector, embedder: {type: hashing, dim: 8}}
        """,
    )
    config = load_config(path)
    assert config.documents == workspace / "docs"
    assert config.dataset == workspace / "qa.jsonl"
    f, r, s = config.experiments
    assert isinstance(f.chunker, FixedChunker)
    assert isinstance(r.chunker, RecursiveChunker)
    assert isinstance(s.chunker, SentenceChunker)
    assert isinstance(s.retriever, VectorSpec)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("chunker: {type: fixed, size: 10, overlp: 2}", "overlp"),
        ("chunker: {type: fixed, size: 10, overlap: 10}", "overlap"),
        ("chunker: {type: magic, size: 10}", "magic"),
        ("chunker: {type: recursive, size: 0}", "size"),
    ],
)
def test_bad_chunker_settings_fail_at_load(workspace, change, message):
    path = write_config(
        workspace,
        f"""\
        documents: docs
        dataset: qa.jsonl
        experiments:
          - name: x
            {change}
            retriever: {{type: bm25}}
        """,
    )
    with pytest.raises(ValidationError, match=message):
        load_config(path)


def test_depth_must_exceed_k(workspace):
    path = write_config(
        workspace,
        "documents: docs\ndataset: qa.jsonl\nk: 5\ndepth: 5\nexperiments:"
        + EXPERIMENT.format(name="one"),
    )
    with pytest.raises(ValidationError, match="must exceed k"):
        load_config(path)


def test_experiment_names_must_be_unique(workspace):
    path = write_config(
        workspace,
        "documents: docs\ndataset: qa.jsonl\nexperiments:"
        + EXPERIMENT.format(name="same")
        + EXPERIMENT.format(name="same"),
    )
    with pytest.raises(ValidationError, match="duplicate"):
        load_config(path)


@pytest.mark.parametrize("body", ["- a\n- b\n", "just text\n"])
def test_top_level_must_be_a_mapping(workspace, body):
    with pytest.raises(ValueError, match="mapping"):
        load_config(write_config(workspace, body))


def test_run_experiments_scores_and_reuses_one_embedder(workspace, monkeypatch):
    built = []

    from ragtriage import runner as runner_module
    from ragtriage.pipelines.embeddings import HashingEmbedder

    def counting(spec):
        built.append(spec)
        return HashingEmbedder(spec.dim)

    monkeypatch.setattr(runner_module, "build_embedder", counting)
    path = write_config(
        workspace,
        """\
        documents: docs
        dataset: qa.jsonl
        k: 1
        depth: 5
        experiments:
          - name: a
            chunker: {type: sentence, max_chars: 40}
            retriever: {type: vector, embedder: {type: hashing, dim: 512}}
          - name: b
            chunker: {type: fixed, size: 40}
            retriever: {type: vector, embedder: {type: hashing, dim: 512}}
          - name: c
            chunker: {type: sentence, max_chars: 40}
            retriever: {type: bm25, tokenizer: turkish_prefix}
        """,
    )
    a, b, c = run_experiments(load_config(path))
    assert len(built) == 1
    assert (a.name, b.name, c.name) == ("a", "b", "c")
    assert a.report.mean["recall@1"] == 1.0
    assert c.report.mean["recall@1"] == 1.0
    assert a.report.n_unanswerable_skipped == 1
    assert a.context_chars > 0
    assert a.retrieval_ms > 0


def test_cli_prints_table_and_writes_json(workspace):
    out = workspace / "results.json"
    result = runner.invoke(
        app, ["run", str(write_config(workspace)), "--out", str(out), "--failures"]
    )
    assert result.exit_code == 0, result.output
    assert "| Experiment | Chunks | Recall@1 |" in result.output
    assert "| one |" in result.output
    assert "one: 0 failed" in result.output
    assert "Scored 1 questions. Skipped 1 unanswerable." in result.output
    [saved] = json.loads(out.read_text("utf-8"))
    assert saved["name"] == "one"
    assert saved["report"]["mean"]["recall@1"] == 1.0
    assert saved["report"]["per_example"][0]["example_id"] == "q1"


def test_cli_failures_flag_names_the_failed_question(workspace):
    (workspace / "qa.jsonl").write_text(
        '{"id": "q9", "question": "zürafa", '
        '"evidence": [{"doc_id": "a", "quote": "İzmir bir liman kentidir."}]}\n',
        "utf-8",
    )
    result = runner.invoke(app, ["run", str(write_config(workspace)), "--failures"])
    assert result.exit_code == 0, result.output
    assert "- [retrieval] q9" in result.output


@pytest.mark.parametrize(
    "body",
    ["experiments: []\n", "not: [valid\n"],
    ids=["invalid_config", "invalid_yaml"],
)
def test_cli_reports_config_errors_without_a_traceback(workspace, body):
    result = runner.invoke(app, ["run", str(write_config(workspace, body))])
    assert result.exit_code == 2
    assert "error:" in result.output
    assert "Traceback" not in result.output


def test_cli_reports_a_missing_file(workspace):
    result = runner.invoke(app, ["run", str(workspace / "nope.yaml")])
    assert result.exit_code == 2
    assert "error:" in result.output
