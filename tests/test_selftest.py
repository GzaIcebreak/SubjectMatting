import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_selftest_mode_imports_everything_and_exits_zero():
    env = {**os.environ, "SUBJECTMATTING_SELFTEST": "1", "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run([sys.executable, str(ROOT / "app.py")], cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "selftest ok" in r.stdout
    assert "torch=" in r.stdout and "gradio=" in r.stdout
