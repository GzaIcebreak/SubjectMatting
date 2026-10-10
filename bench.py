"""Benchmark pipeline A on test_images/: CPU timings, previews, results.json and report.html."""

from __future__ import annotations

import argparse
import html
import json
import platform
import time
from pathlib import Path

import torch
from PIL import Image

from matting.batch import list_images
from matting.common import checkerboard_preview, load_rgb
from matting.pipeline_a import PipelineA

PREVIEW_H = 360


def thumb(img: Image.Image) -> Image.Image:
    w, h = img.size
    return img.resize((round(w * PREVIEW_H / h), PREVIEW_H), Image.LANCZOS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="test_images")
    ap.add_argument("--output", default="outputs/bench")
    args = ap.parse_args()
    out = Path(args.output)
    (out / "img").mkdir(parents=True, exist_ok=True)
    files = list_images(Path(args.input))

    t = time.perf_counter()
    pipe = PipelineA()
    load_s = time.perf_counter() - t

    # Warm-up so the first image's numbers aren't dominated by lazy init / allocator growth.
    pipe.run(load_rgb(files[0]))

    rows = []
    for f in files:
        img = load_rgb(f)
        thumb(img).save(out / "img" / f"{f.stem}_orig.jpg", quality=90)
        rgba, info = pipe.run(img)
        rgba.save(out / "img" / f"{f.stem}.png")
        thumb(checkerboard_preview(rgba)).save(out / "img" / f"{f.stem}_prev.jpg", quality=90)
        rows.append({"file": f.name, "size": img.size, "total": info["total"], "stages": info["stages"]})
        print(f"{f.name}: {info['total']:.2f}s {info['stages']}")

    env = {
        "cpu": platform.processor(),
        "threads": torch.get_num_threads(),
        "torch": torch.__version__,
        "model_load_s": load_s,
    }
    (out / "results.json").write_text(json.dumps({"env": env, "rows": rows}, indent=2), encoding="utf-8")
    write_report(out, env, rows)
    print(f"report: {out / 'report.html'}")


def write_report(out: Path, env: dict, rows: list[dict]) -> None:
    avg = sum(r["total"] for r in rows) / len(rows)
    cards = []
    for r in rows:
        stem = Path(r["file"]).stem
        stages = " · ".join(f"{k} {v:.2f}s" for k, v in r["stages"].items())
        cards.append(
            f"""<section><h3>{html.escape(r['file'])} <small>{r['size'][0]}×{r['size'][1]}</small></h3>
<div class="grid">
<figure><img src="img/{stem}_orig.jpg"><figcaption>原图</figcaption></figure>
<figure><img src="img/{stem}_prev.jpg"><figcaption>{r['total']:.2f}s<br>{stages}</figcaption></figure>
</div></section>"""
        )
    doc = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Subject Matting Bench</title><style>
:root{{--bg:#fafafa;--fg:#1a1a1a;--muted:#666;--card:#fff;--line:#e3e3e3}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151515;--fg:#eee;--muted:#999;--card:#1f1f1f;--line:#333}}}}
body{{background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif;margin:0 auto;max-width:1200px;padding:16px}}
section{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;margin:16px 0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}}
figure{{margin:0}}img{{width:100%;height:auto;border-radius:4px}}figcaption,small{{color:var(--muted)}}
</style></head><body>
<h1>自动主体抠图：BiRefNet_lite + SAM2.1-tiny + ViTMatte（CPU）</h1>
<p>CPU {html.escape(env['cpu'])} · {env['threads']} 线程 · torch {env['torch']} ·
平均单张 <b>{avg:.2f}s</b>（稳态）· 模型加载 {env['model_load_s']:.1f}s</p>
{''.join(cards)}
</body></html>"""
    (out / "report.html").write_text(doc, encoding="utf-8")


if __name__ == "__main__":
    main()
