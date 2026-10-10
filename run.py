"""CLI: python run.py --input <file|dir> [--output <dir>]

For a folder, --output defaults to a sibling folder: D:/photos -> D:/photos_matting.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from matting.batch import run_batch
from matting.common import load_rgb
from matting.pipeline_a import PipelineA


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    src = Path(args.input)

    t0 = time.perf_counter()
    pipe = PipelineA()
    print(f"models loaded in {time.perf_counter() - t0:.1f}s")

    if src.is_file():
        out = Path(args.output or "outputs")
        out.mkdir(parents=True, exist_ok=True)
        rgba, info = pipe.run(load_rgb(src))
        rgba.save(out / f"{src.stem}.png")
        print(f"{src.name}: total={info['total']:.2f}s -> {out / f'{src.stem}.png'}")
        return

    failed = 0
    for ev in run_batch(pipe, src, args.output):
        if ev.error:
            failed += 1
            print(f"[{ev.index}/{ev.total}] {ev.src.name}: FAILED {ev.error}")
        else:
            print(f"[{ev.index}/{ev.total}] {ev.src.name}: {ev.info['total']:.2f}s -> {ev.dst}")
    print(f"done in {time.perf_counter() - t0:.1f}s, failed: {failed}")


if __name__ == "__main__":
    main()
