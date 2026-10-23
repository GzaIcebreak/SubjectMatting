"""Pipeline Q on the hard cases in eval/hard_cases.json: point checks, timings and a contact sheet.

    PYTHONPATH=.:third_party/efficientsam3/sam3 .venv/bin/python scripts/eval_hard.py [--out outputs/eval_hard] [--only castle3]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from matting.common import load_alpha, load_rgb  # noqa: E402
from matting.pipeline_vlm_onnx import PipelineVlmOnnx  # noqa: E402

FG_MIN, BG_MAX = 0.8, 0.2


def point_alpha(alpha: np.ndarray, x: float, y: float) -> float:
    h, w = alpha.shape
    r = max(2, round(0.01 * max(w, h)))
    cx, cy = round(x * (w - 1)), round(y * (h - 1))
    return float(alpha[max(0, cy - r): cy + r + 1, max(0, cx - r): cx + r + 1].mean()) / 255


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="eval/hard_cases.json")
    ap.add_argument("--images", default="test_images/hard")
    ap.add_argument("--out", default="outputs/eval_hard")
    ap.add_argument("--only", nargs="*", default=[])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pipe = PipelineVlmOnnx()
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))["cases"]
    tiles, failed = [], 0
    for c in cases:
        if args.only and not any(o in c["image"] for o in args.only):
            continue
        instr = c.get("instruction")
        name = Path(c["image"]).stem + (f"_{instr}" if instr else "")
        t = time.perf_counter()
        try:
            src = Path(args.images) / c["image"]
            rgba, info = pipe.run(load_rgb(src), instr, load_alpha(src))
            err = None
        except Exception as e:  # noqa: BLE001 - report and go on
            rgba, info, err = None, {}, f"{type(e).__name__}: {e}"
        secs = time.perf_counter() - t
        bad = []
        if rgba is not None:
            a = np.asarray(rgba)[..., 3]
            bad += [f"fg{p}={v:.2f}" for p in c["fg"] if (v := point_alpha(a, *p)) <= FG_MIN]
            bad += [f"bg{p}={v:.2f}" for p in c["bg"] if (v := point_alpha(a, *p)) >= BG_MAX]
            rgba.save(out / f"{name}.png")
        ok = err is None and not bad
        failed += not ok
        what = " ".join(f"{k} {v:.2f}" for k, v in info.get("stages", {}).items())
        print(f"{'PASS' if ok else 'FAIL'} {name:22s} {secs:5.1f}s  {err or ' '.join(bad)}  [{what}] {info.get('summary', '')}", flush=True)
        if rgba is not None:
            bg = Image.new("RGBA", rgba.size, (40, 140, 220, 255))
            bg.alpha_composite(rgba)
            tile = bg.convert("RGB")
            tile = tile.resize((round(tile.width * 240 / tile.height), 240))
            d = ImageDraw.Draw(tile)
            for pts, col in ((c["fg"], (0, 255, 0)), (c["bg"], (255, 0, 0))):
                for x, y in pts:
                    d.ellipse((x * tile.width - 4, y * tile.height - 4, x * tile.width + 4, y * tile.height + 4), outline=col, width=2)
            d.text((4, 4), ("OK " if ok else "FAIL ") + name, fill=(255, 255, 255))
            tiles.append(tile)
    if tiles:
        cols = 5
        rows = (len(tiles) + cols - 1) // cols
        cw = max(t.width for t in tiles)
        sheet = Image.new("RGB", (cols * (cw + 4), rows * 244), "white")
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % cols) * (cw + 4), (i // cols) * 244))
        sheet.save(out / "sheet.jpg", quality=85)
    print(f"{failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
