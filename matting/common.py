"""Shared utilities: image I/O, resizing, trimap, ViTMatte wrapper, timing."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

DEVICE = torch.device("cpu")
WORK_LONG_SIDE = 1024

torch.set_num_threads(max(1, os.cpu_count() or 1))


class Timer:
    """Collects named stage durations (seconds)."""

    def __init__(self):
        self.stages: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = self.stages.get(name, 0.0) + time.perf_counter() - t0

    @property
    def total(self) -> float:
        return sum(self.stages.values())


def load_rgb(path: str | Path) -> Image.Image:
    # Phone photos store their rotation in EXIF; apply it so the result is upright like in every viewer.
    img = ImageOps.exif_transpose(Image.open(path))
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    return img.convert("RGB")


def resize_long_side(img: Image.Image, long_side: int = WORK_LONG_SIDE) -> Image.Image:
    w, h = img.size
    scale = long_side / max(w, h)
    if scale >= 1.0:
        return img
    return img.resize((round(w * scale), round(h * scale)), Image.BICUBIC)


def make_trimap(mask: np.ndarray, k: int | None = None) -> np.ndarray:
    """Binary mask (H,W bool/uint8) -> trimap uint8 with 0=bg, 128=unknown, 255=fg."""
    m = (mask > 0).astype(np.uint8)
    if k is None:
        k = max(5, round(0.012 * max(m.shape)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    fg = cv2.erode(m, kernel)
    maybe = cv2.dilate(m, kernel)
    # Components thinner than the band (text strokes, wires) would vanish from fg entirely and become 100%
    # "unknown", which ViTMatte may erase. Give each such component its medial core as definite foreground.
    n, labels = cv2.connectedComponents(m, connectivity=8)
    if n > 1:
        dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
        has_fg = np.zeros(n, bool)
        has_fg[np.unique(labels[fg > 0])] = True
        peak = np.zeros(n, np.float32)
        np.maximum.at(peak, labels.ravel(), dist.ravel())
        thin = ~has_fg
        thin[0] = False
        core = thin[labels] & (dist >= 0.5 * peak[labels]) & (m > 0)
        fg[core] = 1
    trimap = np.zeros(m.shape, np.uint8)
    trimap[maybe > 0] = 128
    trimap[fg > 0] = 255
    return trimap


def largest_components(mask: np.ndarray, min_ratio: float = 0.1) -> list[np.ndarray]:
    """Connected components whose area is >= min_ratio * largest component, largest first."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return []
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = [i + 1 for i in np.argsort(-areas) if areas[i] >= min_ratio * areas.max()]
    return [labels == i for i in keep]


def compose_rgba(img: Image.Image, alpha: np.ndarray) -> Image.Image:
    """alpha float [0,1] at any size -> RGBA at img size."""
    a = cv2.resize(alpha.astype(np.float32), img.size, interpolation=cv2.INTER_LINEAR)
    a8 = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    rgba = np.dstack([np.asarray(img.convert("RGB")), a8])
    return Image.fromarray(rgba, "RGBA")


def checkerboard_preview(rgba: Image.Image, cell: int = 16) -> Image.Image:
    w, h = rgba.size
    yy, xx = np.mgrid[0:h, 0:w]
    board = np.where(((yy // cell + xx // cell) % 2)[..., None] == 0, 255, 200).astype(np.uint8)
    bg = Image.fromarray(np.repeat(board, 3, axis=2), "RGB").convert("RGBA")
    return Image.alpha_composite(bg, rgba).convert("RGB")


class ViTMatte:
    """Trimap-based matting with hustvl/vitmatte-small-composition-1k."""

    MODEL_ID = "hustvl/vitmatte-small-composition-1k"

    def __init__(self):
        from transformers import VitMatteForImageMatting, VitMatteImageProcessor

        self.processor = VitMatteImageProcessor.from_pretrained(self.MODEL_ID)
        self.model = VitMatteForImageMatting.from_pretrained(self.MODEL_ID).to(DEVICE).eval()

    @torch.inference_mode()
    def __call__(self, img: Image.Image, trimap: np.ndarray) -> np.ndarray:
        """img (work size) + trimap (same size) -> alpha float32 (H,W)."""
        w, h = img.size
        inputs = self.processor(images=img, trimaps=Image.fromarray(trimap, "L"), return_tensors="pt")
        alpha = self.model(**inputs).alphas[0, 0, :h, :w].float().numpy()
        # Trust the trimap's certain regions; ViTMatte only decides the unknown band.
        alpha[trimap == 255] = 1.0
        alpha[trimap == 0] = 0.0
        return alpha
