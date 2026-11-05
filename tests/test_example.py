import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_quickstart_runs_and_prints_a_comparison():
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "turkish_quickstart" / "run.py")],
        capture_output=True,
        encoding="utf-8",
        check=True,
    )
    assert "| Experiment |" in result.stdout
    assert "turkish+prefix5" in result.stdout
