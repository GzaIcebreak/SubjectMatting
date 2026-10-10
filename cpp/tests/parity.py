"""End-to-end check: sm_cli (C++) against matting/pipeline_onnx.py on the same decoded pixels.

    .venv/bin/python cpp/tests/parity.py --cli cpp/build/sm_cli --models onnx_models [images...]

Each image is decoded by PIL and handed to both sides as PNG (JPEG decoders differ slightly, which is not what
this test is about). With no images, a few synthetic ones are generated. Fails when an alpha differs by more
than --max-mean on average or the alpha>0.5 masks overlap less than --min-iou.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from matting.common import load_rgb  # noqa: E402
from matting.pipeline_onnx import PipelineOnnx  # noqa: E402


def synthetic(tmp: Path) -> list[Path]:
    out = []
    for i, (w, h) in enumerate([(640, 480), (1300, 900), (500, 800)]):
        im = Image.new("RGB", (w, h), (235, 235, 230))
        d = ImageDraw.Draw(im)
        d.ellipse((w * 0.25, h * 0.2, w * 0.7, h * 0.85), fill=(200, 40, 40))
        d.rectangle((w * 0.6, h * 0.5, w * 0.8, h * 0.9), fill=(30, 90, 180))
        d.line((0, h * 0.1 * (i + 1), w, h * 0.05), fill=(90, 90, 90), width=3)
        p = tmp / f"synthetic_{i}.png"
        im.save(p)
        out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True)
    ap.add_argument("--models", default="onnx_models")
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--max-mean", type=float, default=0.5, help="max mean |alpha diff| in 1/255 units")
    ap.add_argument("--min-iou", type=float, default=0.99)
    ap.add_argument("images", nargs="*")
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp())
    images = [Path(p) for p in args.images] or synthetic(tmp)
    pngs = []
    for p in images:
        q = tmp / f"in_{p.stem}.png"
        load_rgb(p).save(q)
        pngs.append(q)
    cmd = [args.cli, "--models", args.models, "--precision", args.precision]
    for q in pngs:
        cmd += [str(q), str(tmp / f"out_{q.name}")]
    subprocess.run(cmd, check=True)

    ref = PipelineOnnx(args.models, args.precision)
    failed = 0
    for q in pngs:
        a = np.asarray(ref.run(Image.open(q).convert("RGB"))[0])[..., 3].astype(np.float64)
        b = np.asarray(Image.open(tmp / f"out_{q.name}"))[..., 3].astype(np.float64)
        mean = np.abs(a - b).mean()
        ma, mb = a > 127.5, b > 127.5
        iou = (ma & mb).sum() / max((ma | mb).sum(), 1)
        ok = mean <= args.max_mean and iou >= args.min_iou
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'} {q.name[3:]:28s} mean|diff| {mean:.4f}/255  IoU {iou:.5f}  max {np.abs(a - b).max():.0f}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
