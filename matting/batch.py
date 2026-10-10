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


@dataclass
class Job:
    src: Path
    dst: Path


def plan_jobs(inputs: list[str | Path], out_dir: str | Path | None = None) -> tuple[list[Job], list[tuple[Path, str]]]:
    """Dropped files/folders -> (jobs, skipped), with run_batch's naming rules.

    A folder contributes its images (non-recursive); a loose image is grouped with its parent folder. Each group
    goes to out_dir, or to default_output_dir(group folder) when out_dir is None. Duplicates are dropped, and
    names that would still collide in a shared out_dir get _2, _3, ... appended.
    """
    out = Path(out_dir) if out_dir else None
    groups: dict[Path, list[Path]] = {}
    skipped: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            folder, files = p, list_images(p)
            if not files:
                skipped.append((p, "文件夹里没有图片"))
        elif p.is_file():
            if p.suffix.lower() not in IMAGE_EXTS:
                skipped.append((p, "不支持的格式"))
                continue
            folder, files = p.parent, [p]
        else:
            skipped.append((p, "文件不存在"))
            continue
        for f in files:
            if f.resolve() not in seen:
                seen.add(f.resolve())
                groups.setdefault(folder.resolve(), []).append(f)

    jobs: list[Job] = []
    used: set[Path] = set()
    for folder, files in groups.items():
        dst_dir = out or default_output_dir(folder)
        if dst_dir.resolve() == folder:
            skipped += [(f, "输出文件夹不能和输入文件夹相同") for f in files]
            continue
        for f, name in zip(files, output_names(files)):
            dst, n = dst_dir / name, 1
            while dst.resolve() in used:
                n += 1
                dst = dst_dir / f"{Path(name).stem}_{n}.png"
            used.add(dst.resolve())
            jobs.append(Job(f, dst))
    return jobs, skipped


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
