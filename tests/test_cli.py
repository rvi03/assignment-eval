import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_module_help_prints_usage():
    result = subprocess.run(
        [sys.executable, "-m", "evaluator", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "usage: python -m evaluator" in result.stdout
