"""Local web UI: python app.py  ->  http://127.0.0.1:7860

Env vars:
  SUBJECTMATTING_SELFTEST=1  import every dependency, print versions, exit 0 (no model download). Used by CI
                             to prove a PyInstaller bundle can start.
  PORT                       server port (default 7860).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

FROZEN = bool(getattr(sys, "frozen", False))  # running inside a PyInstaller bundle

if FROZEN and not os.environ.get("SUBJECTMATTING_SELFTEST"):
    # A double-clicked .app has no console: keep stdout/stderr in a log file so failures are diagnosable.
    _log_dir = Path.home() / ("Library/Logs" if sys.platform == "darwin" else ".subjectmatting")
    _log_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(_log_dir / "SubjectMatting.log", "a", buffering=1, encoding="utf-8")
    print(f"--- start {time.strftime('%Y-%m-%d %H:%M:%S')} ---")

import gradio as gr  # noqa: E402

from matting.batch import default_output_dir, run_batch  # noqa: E402
from matting.common import checkerboard_preview  # noqa: E402
from matting.pipeline_a import PipelineA  # noqa: E402


def selftest() -> None:
    """Import everything the pipelines need at runtime (incl. what BiRefNet's remote code imports)."""
    import cv2  # noqa: F401
    import einops  # noqa: F401
    import huggingface_hub
    import kornia  # noqa: F401
    import timm  # noqa: F401
    import torch
    import torchvision  # noqa: F401
    import transformers
    from transformers import AutoModelForImageSegmentation, Sam2Model, VitMatteForImageMatting  # noqa: F401

    print(
        "selftest ok",
        f"python={sys.version.split()[0]} torch={torch.__version__} transformers={transformers.__version__}",
        f"gradio={gr.__version__} huggingface_hub={huggingface_hub.__version__} frozen={FROZEN}",
    )


if os.environ.get("SUBJECTMATTING_SELFTEST"):
    selftest()
    sys.exit(0)

pipe = PipelineA()
GALLERY_MAX = 60  # keep the batch gallery light; all results are on disk anyway


def process(img):
    if img is None:
        return None, None, ""
    rgba, info = pipe.run(img.convert("RGB"))
    log = f"{info['total']:.2f}s  " + " ".join(f"{k}={v:.2f}s" for k, v in info["stages"].items())
    return checkerboard_preview(rgba), rgba, log


def clean_path(s: str) -> str:
    # Windows "Copy as path" wraps the path in quotes.
    return (s or "").strip().strip('"').strip("'").strip()


def batch_process(folder: str, out_dir: str):
    folder, out_dir = clean_path(folder), clean_path(out_dir)
    if not folder:
        yield "请输入文件夹路径", []
        return
    out = Path(out_dir) if out_dir else default_output_dir(Path(folder))
    t0, lines, gallery, failed = time.perf_counter(), [], [], []
    try:
        for ev in run_batch(pipe, folder, out):
            if ev.error:
                failed.append(ev.src.name)
                lines.append(f"[{ev.index}/{ev.total}] {ev.src.name}  失败：{ev.error}")
            else:
                lines.append(f"[{ev.index}/{ev.total}] {ev.src.name}  {ev.info['total']:.2f}s → {ev.dst.name}")
                if len(gallery) < GALLERY_MAX:
                    preview = checkerboard_preview(ev.rgba)
                    preview.thumbnail((320, 320))
                    gallery.append((preview, ev.src.name))
            head = f"输出目录：{out}\n进度：{ev.index}/{ev.total}　已用 {time.perf_counter() - t0:.1f}s\n"
            yield head + "\n".join(lines[-200:]), gallery
    except (FileNotFoundError, ValueError, PermissionError) as e:
        yield f"错误：{e}", []
        return
    if not lines:
        yield f"文件夹里没有图片（支持 jpg / jpeg / png / webp / bmp，不含子文件夹）：{folder}", []
        return
    summary = f"完成：成功 {len(lines) - len(failed)} 张，失败 {len(failed)} 张，总耗时 {time.perf_counter() - t0:.1f}s"
    if failed:
        summary += "\n失败：" + "、".join(failed)
    yield f"输出目录：{out}\n{summary}\n\n" + "\n".join(lines[-200:]), gallery


with gr.Blocks(title="Subject Matting") as demo:
    gr.Markdown("## 自动主体抠图（BiRefNet_lite + SAM2.1-tiny + ViTMatte，CPU）")
    with gr.Tab("单张"):
        with gr.Row():
            inp = gr.Image(type="pil", label="输入图片")
            with gr.Column():
                btn = gr.Button("抠图", variant="primary")
                log = gr.Textbox(label="耗时", lines=2)
        with gr.Row():
            prev = gr.Image(label="预览")
            out = gr.Image(type="pil", format="png", label="RGBA（可下载）")
    with gr.Tab("批量"):
        with gr.Row():
            with gr.Column():
                folder_in = gr.Textbox(label="输入文件夹路径", placeholder=r"例如 D:\photos")
                folder_out = gr.Textbox(label="输出文件夹（留空 = 同级的 <文件夹名>_matting）", placeholder=r"例如 D:\photos_matting")
                batch_btn = gr.Button("开始批量抠图", variant="primary")
            batch_log = gr.Textbox(label="进度", lines=12, max_lines=20)
        batch_gallery = gr.Gallery(label=f"结果预览（最多显示 {GALLERY_MAX} 张，全部结果见输出目录）", columns=6, height="auto")

    # Single and batch share one model instance: run them through one queue so they never compete for the CPU.
    btn.click(process, [inp], [prev, out, log], concurrency_id="model")
    batch_btn.click(batch_process, [folder_in, folder_out], [batch_log, batch_gallery], concurrency_id="model")

if __name__ == "__main__":
    # In the packaged app nobody sees the URL in a console, so open the browser for them.
    demo.launch(server_name="127.0.0.1", server_port=int(os.environ.get("PORT", 7860)), inbrowser=FROZEN)
