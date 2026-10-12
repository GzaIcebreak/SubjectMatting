"""The C++ CLIP tokenizer (smq_cli --tokenize) against the upstream Python one, on Qwen-style labels and edge cases.

    PYTHONPATH=. .venv/bin/python cpp/tests/tokenizer_test.py --cli cpp/build/smq_cli --models onnx_models_q
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from matting.pipeline_vlm_onnx import CONTEXT, clip_tokenizer  # noqa: E402

LABELS = [
    "cat", "dog", "woman", "man", "batter", "catcher", "umpire", "owl", "zebra", "macaw", "car",
    "teddy bear", "shopping bag", "grocery bags", "parrot beak", "bald eagle", "king penguin", "brown bear",
    "stop sign", "white pickup truck", "silver suv", "tabby cat", "calico cat",
    "Red Car", "  spaced   out  ", "cat's toy", "rock'n'roll", "3d printer", "t-shirt", "50/50 bag", "e-bike!!",
    "café", "naïve art", "a very long label that certainly does not fit into sixteen tokens of context at all",
]  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True)
    ap.add_argument("--models", default="onnx_models_q")
    args = ap.parse_args()
    tok = clip_tokenizer()
    failed = 0
    for label in LABELS:
        want = tok([label], context_length=CONTEXT)[0].tolist()
        out = subprocess.run([args.cli, "--models", args.models, "--tokenize", label], capture_output=True, text=True, check=True)
        got = [int(x) for x in out.stdout.split()]
        ok = got == want
        failed += not ok
        if not ok:
            print(f"FAIL {label!r}\n  want {want}\n  got  {got}")
    print(f"{len(LABELS) - failed}/{len(LABELS)} labels tokenized identically")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
