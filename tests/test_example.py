import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


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
