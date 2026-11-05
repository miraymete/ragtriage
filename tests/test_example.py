import subprocess
import sys
from pathlib import Path

import pytest

from ragtriage import load_documents, load_qa_examples
from ragtriage.config import load_config

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_CONFIGS = sorted((ROOT / "examples").glob("**/*.yaml"))


@pytest.mark.parametrize("path", EXAMPLE_CONFIGS, ids=lambda p: p.name)
def test_every_example_config_and_dataset_is_valid(path):
    """Configs that need a model download never run in CI, so at least check they would load."""
    config = load_config(path)
    examples = load_qa_examples(config.dataset, load_documents(config.documents))
    assert examples


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
