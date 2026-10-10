import json
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def start_worker(tmp_path, **env):
    env = {**os.environ, "SUBJECTMATTING_LOG": str(tmp_path / "worker.log"), "PYTHONIOENCODING": "utf-8", **env}
    return subprocess.Popen(
        [sys.executable, str(ROOT / "worker.py")],
        cwd=ROOT,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )


def send(p, **msg):
    p.stdin.write(json.dumps(msg) + "\n")
    p.stdin.flush()


def read_until(p, type_):
    """Events up to and including the first one of type_ (a short protocol, so no timeout handling)."""
    events = []
    for line in p.stdout:
        events.append(json.loads(line))
        if events[-1]["type"] == type_:
            return events
    raise AssertionError(f"worker exited before {type_!r}: {events}")


def test_selftest_mode_imports_everything_and_exits_zero():
    env = {**os.environ, "SUBJECTMATTING_SELFTEST": "1", "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run([sys.executable, str(ROOT / "worker.py")], cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "selftest ok" in r.stdout and "torch=" in r.stdout


def test_plan_and_run_round_trip_with_fake_pipeline(tmp_path):
    src = tmp_path / "photos"
    src.mkdir()
    Image.new("RGB", (32, 24), (200, 50, 50)).save(src / "a.jpg")
    (src / "broken.png").write_bytes(b"not an image")

    p = start_worker(tmp_path, SUBJECTMATTING_FAKE_PIPELINE="1")
    try:
        send(p, id=1, cmd="plan", inputs=[str(src), str(tmp_path / "nope.jpg")], output_dir=None)
        plan = read_until(p, "plan")[-1]
        assert plan["id"] == 1 and plan["error"] is None
        out = tmp_path / "photos_matting"
        assert [(Path(j["src"]).name, Path(j["dst"])) for j in plan["jobs"]] == [
            ("a.jpg", out / "a.png"),
            ("broken.png", out / "broken.png"),
        ]
        assert plan["skipped"] == [{"path": str(tmp_path / "nope.jpg"), "reason": "文件不存在"}]

        for i, job in enumerate(plan["jobs"], 2):
            send(p, id=i, cmd="run", **job)
        ok = read_until(p, "result")[-1]
        assert ok["id"] == 2 and Image.open(ok["dst"]).mode == "RGBA"
        bad = read_until(p, "error")[-1]
        assert bad["id"] == 3 and "broken.png" in bad["message"]
    finally:
        p.stdin.close()
        assert p.wait(timeout=60) == 0  # exits on stdin EOF
    # Library chatter goes to the log, never the protocol stream.
    assert "--- worker start" in (tmp_path / "worker.log").read_text(encoding="utf-8")


def test_ready_is_sent_before_results(tmp_path):
    p = start_worker(tmp_path, SUBJECTMATTING_FAKE_PIPELINE="1")
    try:
        assert read_until(p, "ready")[-1] == {"type": "ready"}
    finally:
        p.stdin.close()
        p.wait(timeout=60)
