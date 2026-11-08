import subprocess
import sys
from pathlib import Path

import pytest

from ragtriage import load_documents, load_qa_examples
from ragtriage.config import load_config

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_CONFIGS = sorted((ROOT / "examples").glob("**/*.yaml"))

# The only examples whose data is downloaded and git-ignored: config -> (documents, dataset).
# Being under data/ is not enough. A new config has to be listed here to skip the data check.
_XQUAD = "data/xquad_tr"
DOWNLOADED_EXAMPLES = {
    "examples/xquad_tr/pilot_tokenizers.yaml": (
        f"{_XQUAD}/dev/docs",
        f"{_XQUAD}/dev/pilot_qa.jsonl",
    ),
    "examples/xquad_tr/pilot_chunking.yaml": (f"{_XQUAD}/dev/docs", f"{_XQUAD}/dev/pilot_qa.jsonl"),
    "examples/xquad_tr/dev_stage1.yaml": (f"{_XQUAD}/dev/docs", f"{_XQUAD}/dev/qa.jsonl"),
    "examples/xquad_tr/dev_stage2.yaml": (f"{_XQUAD}/dev/docs", f"{_XQUAD}/dev/qa.jsonl"),
    "examples/xquad_tr/dev_locked.yaml": (f"{_XQUAD}/dev/docs", f"{_XQUAD}/dev/qa.jsonl"),
    "examples/xquad_tr/test_locked.yaml": (f"{_XQUAD}/test/docs", f"{_XQUAD}/test/qa.jsonl"),
}


def validate_example(path, key):
    """Check a config and, unless it is a listed downloaded example, load its data.

    The choice depends only on `key` and the paths in the config, never on whether the files
    exist, so a local run and CI behave the same. CI downloads nothing.
    """
    config = load_config(path)
    expected = DOWNLOADED_EXAMPLES.get(key)
    if expected is not None:
        assert config.documents.resolve() == (ROOT / expected[0]).resolve()
        assert config.dataset.resolve() == (ROOT / expected[1]).resolve()
        return
    examples = load_qa_examples(config.dataset, load_documents(config.documents))
    assert examples


@pytest.mark.parametrize("path", EXAMPLE_CONFIGS, ids=lambda p: p.name)
def test_every_example_config_and_dataset_is_valid(path):
    """Configs that need a model download never run in CI, so at least check they would load."""
    validate_example(path, path.relative_to(ROOT).as_posix())


def _config(directory, documents, dataset):
    path = directory / "config.yaml"
    path.write_text(
        f"documents: {Path(documents).as_posix()}\n"
        f"dataset: {Path(dataset).as_posix()}\n"
        "experiments:\n"
        "  - name: one\n"
        "    chunker: {type: fixed, size: 100}\n"
        "    retriever: {type: bm25}\n",
        "utf-8",
    )
    return path


def test_every_listed_downloaded_example_exists():
    found = {path.relative_to(ROOT).as_posix() for path in EXAMPLE_CONFIGS}
    assert set(DOWNLOADED_EXAMPLES) <= found


def test_listed_examples_never_load_data(monkeypatch):
    def fail(*_):
        raise AssertionError("listed examples must not need the downloaded data")

    monkeypatch.setitem(globals(), "load_documents", fail)
    for key in DOWNLOADED_EXAMPLES:
        validate_example(ROOT / key, key)


@pytest.mark.parametrize(
    "documents",
    [f"{_XQUAD}/dev/docs", "data/not_listed/docs"],
    ids=["real_xquad_path", "other_path_under_data"],
)
def test_an_unlisted_config_under_data_still_loads_its_data(tmp_path, monkeypatch, documents):
    class Loaded(Exception):
        pass

    def loaded(*_):
        raise Loaded

    # Stubbed so the result is the same whether or not data/ exists on this machine.
    monkeypatch.setitem(globals(), "load_documents", loaded)
    path = _config(tmp_path, ROOT / documents, ROOT / f"{_XQUAD}/dev/qa.jsonl")
    with pytest.raises(Loaded):
        validate_example(path, "examples/xquad_tr/not_listed.yaml")


@pytest.mark.parametrize(
    ("documents", "dataset"),
    [
        (f"{_XQUAD}/dev/docs", f"{_XQUAD}/test/qa.jsonl"),
        (f"{_XQUAD}/test/docs", f"{_XQUAD}/dev/qa.jsonl"),
        ("data/other/docs", f"{_XQUAD}/test/qa.jsonl"),
    ],
    ids=["wrong_dataset", "wrong_documents", "other_directory"],
)
def test_a_listed_config_with_the_wrong_paths_fails(tmp_path, documents, dataset):
    path = _config(tmp_path, ROOT / documents, ROOT / dataset)
    with pytest.raises(AssertionError):
        validate_example(path, "examples/xquad_tr/test_locked.yaml")


def test_example_configs_were_found():
    assert len(EXAMPLE_CONFIGS) >= 3


def test_quickstart_runs_through_the_cli_and_prints_a_comparison():
    """A real subprocess, because the console encoding on Windows only shows up here."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ragtriage.cli",
            "run",
            str(ROOT / "examples" / "turkish_quickstart" / "config.yaml"),
        ],
        capture_output=True,
        encoding="utf-8",
        check=True,
    )
    assert "| Experiment |" in result.stdout
    assert "sentence 300, hashing" in result.stdout
    assert "Scored 21 questions. Skipped 2 unanswerable." in result.stdout
