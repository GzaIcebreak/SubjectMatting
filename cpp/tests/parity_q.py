"""End-to-end check: smq_cli (C++ pipeline Q) against matting/pipeline_vlm_onnx.py on identical pixels and subjects.

    PYTHONPATH=.:third_party/efficientsam3/sam3 .venv/bin/python cpp/tests/parity_q.py --cli cpp/build/smq_cli \
        [--cache outputs/bench_vlm/qwen_cache.json]

Subjects come from bench_vlm.py's Qwen cache when it exists (no API calls), else from a few fixed boxes on
synthetic images. Fails when an alpha differs by more than --max-mean (1/255 units) on average or IoU < --min-iou.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from matting.common import load_rgb, resize_long_side  # noqa: E402
from matting.pipeline_vlm_onnx import PipelineVlmOnnx  # noqa: E402
from matting.qwen import Detection, Subject  # noqa: E402


class FixedDetector:
    def __init__(self):
        self.subjects: list[Subject] = []

    def detect(self, img, instruction=None):
        return Detection(self.subjects, "fixed", 0.0)


def cases(cache: Path, tmp: Path):
    """(png path, subjects in work-image pixels)"""
    if cache.exists():
        for key, c in json.loads(cache.read_text(encoding="utf-8")).items():
            stem = key.split("|")[0]
            src = Path("test_images") / f"{stem}.jpg"
            if stem and src.exists() and c["subjects"]:
                png = tmp / f"{stem}_{abs(hash(key)) % 10000}.png"
                load_rgb(src).save(png)
                yield png, [Subject(s["label"], tuple(s["box"])) for s in c["subjects"]]
        return
    for i, (w, h) in enumerate([(640, 480), (900, 1200)]):
        im = Image.new("RGB", (w, h), (235, 235, 230))
        ImageDraw.Draw(im).ellipse((w * 0.25, h * 0.2, w * 0.7, h * 0.85), fill=(200, 40, 40))
        png = tmp / f"synthetic_{i}.png"
        im.save(png)
        ww, wh = resize_long_side(im).size
        yield png, [Subject("red ball", (ww * 0.22, wh * 0.17, ww * 0.73, wh * 0.88))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True)
    ap.add_argument("--models", default="onnx_models_q")
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--cache", default="outputs/bench_vlm/qwen_cache.json")
    ap.add_argument("--max-mean", type=float, default=0.5)
    ap.add_argument("--min-iou", type=float, default=0.99)
    ap.add_argument("--plain-only", action="store_true", help="skip the cached subject cases")
    ap.add_argument("--plain", nargs="*", default=[], help="solid-background images for the fast path (default: synthetic)")
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp())
    det = FixedDetector()
    ref = PipelineVlmOnnx(args.models, args.precision, detector=det)
    failed = total = 0
    for png, subjects in (cases(Path(args.cache), tmp) if not args.plain_only else []):
        img = Image.open(png).convert("RGB")
        work = resize_long_side(img)
        det.subjects = subjects
        a = np.asarray(ref.run(img)[0])[..., 3].astype(np.float64)
        sx, sy = img.width / work.width, img.height / work.height  # work pixels -> input pixels for the C API
        specs = [f"{s.label}:{s.box[0] * sx},{s.box[1] * sy},{s.box[2] * sx},{s.box[3] * sy}" for s in subjects]
        out = tmp / f"out_{png.name}"
        r = subprocess.run([args.cli, "--models", args.models, "--precision", args.precision, str(png), str(out), *specs],
                           capture_output=True, text=True)  # fmt: skip
        total += 1
        if r.returncode != 0:
            failed += 1
            print(f"FAIL {png.name}: smq_cli exit {r.returncode}: {r.stderr.strip()[:200]}")
            continue
        b = np.asarray(Image.open(out))[..., 3].astype(np.float64)
        mean = np.abs(a - b).mean()
        ma, mb = a > 127.5, b > 127.5
        iou = (ma & mb).sum() / max((ma | mb).sum(), 1)
        ok = mean <= args.max_mean and iou >= args.min_iou
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'} {png.stem:22s} mean|diff| {mean:.4f}/255  IoU {iou:.5f}  max {np.abs(a - b).max():.0f}  {r.stdout.split('  ')[-1].strip()}")
    # Plain-background fast path (auto mode, no subjects): solid-background images, synthetic if none are given.
    plain = [Path(p) for p in args.plain] or []
    if not plain:
        for i, (w, h) in enumerate([(700, 500), (900, 900)]):
            im = Image.new("RGB", (w, h), (252, 252, 250))
            d = ImageDraw.Draw(im)
            d.ellipse((w * 0.3, h * 0.15, w * 0.7, h * 0.7), fill=(170, 140, 90))
            d.rectangle((w * 0.2, h * 0.68, w * 0.8, h * 0.8), fill=(90, 50, 30))
            plain.append(tmp / f"plain_{i}.png")
            im.save(plain[-1])
    for src in plain:
        png = tmp / f"plain_in_{src.stem}.png"
        load_rgb(src).save(png)
        img = Image.open(png).convert("RGB")
        a = np.asarray(ref.run(img)[0])[..., 3].astype(np.float64)  # auto mode -> plain path
        out = tmp / f"plain_out_{src.stem}.png"
        r = subprocess.run([args.cli, "--models", args.models, "--precision", args.precision, "--plain", str(png), str(out)],
                           capture_output=True, text=True)  # fmt: skip
        total += 1
        if r.returncode != 0:
            failed += 1
            print(f"FAIL plain {src.stem}: smq_cli exit {r.returncode}: {r.stderr.strip()[:200]}")
            continue
        b = np.asarray(Image.open(out))[..., 3].astype(np.float64)
        mean = np.abs(a - b).mean()
        ok = mean <= args.max_mean
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'} plain {src.stem[:16]:16s} mean|diff| {mean:.4f}/255  max {np.abs(a - b).max():.0f}")
    print(f"{total - failed}/{total} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
