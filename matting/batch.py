"""Folder batch processing: every image in a folder (non-recursive) -> <name>.png RGBA in an output folder."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from PIL import Image

from .common import load_rgb

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class BatchEvent:
    index: int  # 1-based
    total: int
    src: Path
    dst: Path | None  # None when the image failed
    rgba: Image.Image | None
    info: dict | None
    error: str | None


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def default_output_dir(folder: Path) -> Path:
    """D:/photos -> D:/photos_matting (sibling, so outputs never end up inside the input folder)."""
    folder = folder.resolve()
    return folder.parent / f"{folder.name or 'images'}_matting"


def output_names(files: list[Path]) -> list[str]:
    """<stem>.png, or <stem>_<ext>.png when several inputs share a stem (a.jpg + a.png)."""
    stems = Counter(f.stem for f in files)
    return [f"{f.stem}_{f.suffix[1:].lower()}.png" if stems[f.stem] > 1 else f"{f.stem}.png" for f in files]


def run_batch(pipe, folder: str | Path, out_dir: str | Path | None = None) -> Iterator[BatchEvent]:
    """Run pipe.run() on each image, save RGBA PNGs, yield one event per image. A failing image is reported
    via BatchEvent.error and does not stop the batch."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"文件夹不存在：{folder}")
    out = Path(out_dir) if out_dir else default_output_dir(folder)
    if out.resolve() == folder.resolve():
        raise ValueError("输出文件夹不能和输入文件夹相同（会覆盖同名 .png 原图）")
    files = list_images(folder)
    if not files:
        return
    out.mkdir(parents=True, exist_ok=True)
    for i, (src, name) in enumerate(zip(files, output_names(files)), 1):
        try:
            rgba, info = pipe.run(load_rgb(src))
            dst = out / name
            rgba.save(dst)
            yield BatchEvent(i, len(files), src, dst, rgba, info, None)
        except Exception as e:  # noqa: BLE001 — one bad image must not abort the batch
            yield BatchEvent(i, len(files), src, None, None, None, f"{type(e).__name__}: {e}")
