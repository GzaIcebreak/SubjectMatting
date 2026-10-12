"""Evaluate pipeline Q (Qwen-VL + EfficientSAM3) against pipeline A on test_images/.

    .venv/bin/python bench_vlm.py [--variants efficientvit repvit] [--output outputs/bench_vlm]

Runs auto mode on every image (next to pipeline A's result) and the instruction cases below, then writes
results.json and report.html. Qwen replies are cached in <output>/qwen_cache.json, so re-runs and extra SAM3
variants don't call the API again (delete the file to re-query).
"""

from __future__ import annotations

import argparse
import html
import json
import os
import time
import zlib
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from PIL import Image, ImageDraw  # noqa: E402

from matting.batch import list_images  # noqa: E402
from matting.common import checkerboard_preview, load_rgb, resize_long_side  # noqa: E402
from matting.pipeline_vlm import NoSubjectError, PipelineVlm  # noqa: E402
from matting.qwen import Detection, QwenDetector, Subject  # noqa: E402

PREVIEW_H = 300

# (image stem, instruction, expectation in words for the reader of the report)
INSTRUCTION_CASES = [
    ("two_cats", "只要右边那只猫", "只有右边的猫"),
    ("two_cats", "只要左边那只猫", "只有左边的猫"),
    ("dog_person", "只要狗", "只有狗"),
    ("dog_person", "只要人", "只有女人"),
    ("groceries", "购物袋", "只有购物袋"),
    ("groceries", "整辆车", "整个车尾"),
    ("teddy_bears", "最右边的泰迪熊", "最右边那只熊"),
    ("baseball", "击球手", "拿球棒的人"),
    ("mac_Parrot", "鹦鹉的嘴", "只有鸟喙（部件级，SAM3 可能做不到）"),
    ("two_cats", "狗", "应报错：图中没有狗"),
]


class CachedDetector:
    """QwenDetector with a JSON cache keyed by image + instruction."""

    def __init__(self, path: Path, model: str):
        self.path = path
        self.inner = QwenDetector(model)
        self.cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        self.key = ""

    def detect(self, img, instruction=None) -> Detection:
        key = f"{self.key}|{instruction or ''}"
        if key in self.cache:
            c = self.cache[key]
            return Detection([Subject(s["label"], tuple(s["box"])) for s in c["subjects"]], c["model"], c["seconds"], c["usage"], c["raw"])
        d = self.inner.detect(img, instruction)
        self.cache[key] = {
            "subjects": [{"label": s.label, "box": list(s.box)} for s in d.subjects],
            "model": d.model, "seconds": d.seconds, "usage": d.usage, "raw": d.raw,
        }  # fmt: skip
        self.path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=1), encoding="utf-8")
        return d


def thumb(img: Image.Image) -> Image.Image:
    return img.resize((round(img.width * PREVIEW_H / img.height), PREVIEW_H), Image.LANCZOS)


def boxes_preview(img: Image.Image, det: Detection) -> Image.Image:
    work = resize_long_side(img).copy()
    d = ImageDraw.Draw(work)
    for s in det.subjects:
        d.rectangle(s.box, outline=(255, 40, 40), width=max(2, work.width // 200))
        d.text((s.box[0] + 6, s.box[1] + 4), s.label, fill=(255, 40, 40))
    return work


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="test_images")
    ap.add_argument("--output", default="outputs/bench_vlm")
    ap.add_argument("--qwen-model", default="qwen3-vl-plus")
    ap.add_argument("--variants", nargs="+", default=["efficientvit"])
    ap.add_argument("--skip-a", action="store_true", help="don't run pipeline A")
    args = ap.parse_args()
    out = Path(args.output)
    (out / "img").mkdir(parents=True, exist_ok=True)
    files = list_images(Path(args.input))
    by_stem = {f.stem: f for f in files}
    detector = CachedDetector(out / "qwen_cache.json", args.qwen_model)
    results: dict = {"qwen_model": args.qwen_model, "auto": [], "instruction": [], "variants": {}}

    for f in files:
        thumb(load_rgb(f)).save(out / "img" / f"{f.stem}_orig.jpg")

    if not args.skip_a:
        from matting.pipeline_a import PipelineA

        pa = PipelineA()
        pa.run(load_rgb(files[0]))
        for f in files:
            rgba, info = pa.run(load_rgb(f))
            thumb(checkerboard_preview(rgba)).save(out / "img" / f"{f.stem}_A.jpg")
            results.setdefault("a_seconds", {})[f.stem] = info["total"]
        del pa

    for variant in args.variants:
        t = time.perf_counter()
        pq = PipelineVlm(sam_variant=variant, detector=detector)
        load_s = time.perf_counter() - t
        rows = []

        def run(stem, instruction):
            img = load_rgb(by_stem[stem])
            detector.key = stem
            tag = f"{stem}_{variant}" + ("" if instruction is None else f"_i{zlib.crc32(instruction.encode()) % 10**6}")
            row = {"image": stem, "instruction": instruction, "tag": tag}
            try:
                rgba, info = pq.run(img, instruction)
                thumb(checkerboard_preview(rgba)).save(out / "img" / f"{tag}_Q.jpg")
                thumb(boxes_preview(img, info["detection"])).save(out / "img" / f"{tag}_boxes.jpg")
                row.update(
                    ok=True, stages=info["stages"], total=info["total"],
                    subjects=[{"label": c.subject.label, "box": [round(v) for v in c.subject.box], "kept": c.kept,
                               "scores": [round(s, 3) for s in c.scores]} for c in info["choices"]],
                    usage=info["detection"].usage,
                )  # fmt: skip
            except NoSubjectError as e:
                row.update(ok=False, error=str(e))
            rows.append(row)
            status = f"{row.get('total', 0):.1f}s " + ", ".join(s["label"] for s in row.get("subjects", [])) if row["ok"] else row["error"]
            print(f"[{variant}] {stem:14s} {instruction or 'auto':10s} {status}")
            return row

        pq.run(load_rgb(files[0]))  # warm-up (uses the cache after the first call)
        results["auto"] += [run(f.stem, None) for f in files]
        results["instruction"] += [
            {**run(stem, instr), "expect": expect} for stem, instr, expect in INSTRUCTION_CASES if stem in by_stem
        ]
        results["variants"][variant] = {"load_seconds": load_s}
        del pq

    (out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    write_report(out, results, files, args)
    print(f"report: {out / 'report.html'}")


def write_report(out: Path, results: dict, files, args) -> None:
    def img(name, title=""):
        p = out / "img" / name
        return f'<figure><img src="img/{name}"><figcaption>{html.escape(title)}</figcaption></figure>' if p.exists() else "<figure>—</figure>"

    def stats(rows):
        ok = [r for r in rows if r["ok"]]
        if not ok:
            return "无"
        avg = {k: sum(r["stages"][k] for r in ok) / len(ok) for k in ok[0]["stages"]}
        toks = [r["usage"].get("total_tokens", 0) for r in ok if r.get("usage")]
        return (
            f"成功 {len(ok)}/{len(rows)}，平均 {sum(r['total'] for r in ok) / len(ok):.2f}s/张（"
            + " · ".join(f"{k} {v:.2f}s" for k, v in avg.items())
            + (f"），Qwen 平均 {sum(toks) / len(toks):.0f} tokens/次" if toks else "）")
        )

    parts = [
        "<!doctype html><meta charset=utf-8><title>Qwen-VL + EfficientSAM3 评测</title><style>",
        "body{font-family:-apple-system,sans-serif;margin:24px;color:#222} table{border-collapse:collapse} td{border-top:1px solid #ddd;padding:8px;vertical-align:top}",
        "figure{margin:0 6px 0 0;display:inline-block} img{height:200px} figcaption{font-size:12px;color:#666} .err{color:#c00} code{font-size:12px}",
        "</style>",
        f"<h1>Qwen-VL + EfficientSAM3 评测</h1><p>Qwen 模型：<code>{html.escape(results['qwen_model'])}</code>；SAM3 变体：{', '.join(args.variants)}</p>",
    ]
    for variant in args.variants:
        auto = [r for r in results["auto"] if r["tag"].startswith(f"{r['image']}_{variant}")]
        instr = [r for r in results["instruction"] if r["tag"].startswith(f"{r['image']}_{variant}")]
        a_avg = sum(results.get("a_seconds", {}).values()) / max(len(results.get("a_seconds", {})), 1)
        parts.append(f"<h2>{variant}：全自动（左：原图；中：方案 A / SAM2，{a_avg:.2f}s/张；右：Qwen 框 + 方案 Q）</h2><p>{stats(auto)}</p><table>")
        for r in auto:
            cells = img(f"{r['image']}_orig.jpg", r["image"]) + img(f"{r['image']}_A.jpg", "A (SAM2)")
            if r["ok"]:
                labels = "、".join(f"{s['label']}" for s in r["subjects"])
                cells += img(f"{r['tag']}_boxes.jpg", f"Qwen：{labels}") + img(f"{r['tag']}_Q.jpg", f"Q {r['total']:.1f}s")
            else:
                cells += f"<span class=err>{html.escape(r['error'])}</span>"
            parts.append(f"<tr><td>{cells}</td></tr>")
        parts.append(f"</table><h2>{variant}：指定模式</h2><p>{stats(instr)}</p><table>")
        for r in instr:
            head = f"<b>{html.escape(r['instruction'])}</b><br>期望：{html.escape(r['expect'])}<br>"
            cells = img(f"{r['image']}_orig.jpg", r["image"])
            if r["ok"]:
                labels = "、".join(s["label"] for s in r["subjects"])
                cells += img(f"{r['tag']}_boxes.jpg", f"Qwen：{labels}") + img(f"{r['tag']}_Q.jpg", f"Q {r['total']:.1f}s")
            else:
                cells += f"<span class=err>报错：{html.escape(r['error'])}</span>"
            parts.append(f"<tr><td>{head}{cells}</td></tr>")
        parts.append("</table>")
    (out / "report.html").write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    main()
