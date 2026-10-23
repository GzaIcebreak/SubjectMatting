"""The hard cases (eval/hard_cases.json) through the app's own worker, end to end over its JSON-lines protocol.

    .venv/bin/python scripts/eval_worker.py ["dist/抠图demo.app"] [--out outputs/eval_worker]

The API key comes from $DASHSCOPE_API_KEY or .env (as for matting/qwen.py); the worker's log goes to <out>/worker.log.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from matting.qwen import load_api_key  # noqa: E402
from scripts.eval_hard import BG_MAX, FG_MIN, point_alpha  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("app", nargs="?", default="dist/抠图demo.app")
    ap.add_argument("--cases", default="eval/hard_cases.json")
    ap.add_argument("--images", default="test_images/hard")
    ap.add_argument("--out", default="outputs/eval_worker")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "DASHSCOPE_API_KEY": load_api_key(), "SUBJECTMATTING_LOG": str(out / "worker.log")}
    worker = subprocess.Popen([str(Path(args.app) / "Contents/MacOS/matting-worker-q")], stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, text=True, env=env)  # fmt: skip

    def read():
        return json.loads(worker.stdout.readline())

    while (m := read())["type"] != "ready":
        if m["type"] == "fatal":
            print("fatal:", m["message"])
            return 1
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))["cases"]
    failed = 0
    for i, c in enumerate(cases):
        instr = c.get("instruction")
        name = Path(c["image"]).stem + (f"_{instr}" if instr else "")
        dst = out / f"{name}.png"
        msg = {"cmd": "run", "id": i, "src": str((Path(args.images) / c["image"]).resolve()), "dst": str(dst)}
        if instr:
            msg["instruction"] = instr
        t = time.perf_counter()
        worker.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
        worker.stdin.flush()
        r = read()
        secs = time.perf_counter() - t
        if r["type"] != "result":
            failed += 1
            print(f"FAIL {name:22s} {secs:5.1f}s  {r.get('message')}")
            continue
        a = np.asarray(Image.open(dst).convert("RGBA"))[..., 3]
        bad = [f"fg{p}={v:.2f}" for p in c["fg"] if (v := point_alpha(a, *p)) <= FG_MIN]
        bad += [f"bg{p}={v:.2f}" for p in c["bg"] if (v := point_alpha(a, *p)) >= BG_MAX]
        failed += bool(bad)
        stages = " ".join(f"{k} {v:.2f}" for k, v in r["stages"].items())
        print(f"{'FAIL' if bad else 'PASS'} {name:22s} {secs:5.1f}s  {' '.join(bad)} [{stages}]", flush=True)
    worker.stdin.close()
    worker.wait(timeout=30)
    print(f"{failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
