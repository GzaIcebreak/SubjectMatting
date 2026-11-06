"""Plain-background candidates preserving their measured holes and small components.
Colour alone cannot distinguish same-colour object parts from holes. No blanket fill is
applied; ambiguous material/background cases need additional evidence or user guidance.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .common import largest_components

RING_FRAC = 0.01  # border ring width, x the long side (at least 2 px)
UNIFORM_TOL = 16  # a ring pixel matches the background if every channel is within this
UNIFORM_SHARE = 0.95  # share of ring pixels that must match for the background to count as plain
KEY_TOL = 24  # foreground: some channel differs from the background by more than this
MIN_AREA = 0.0005  # foreground pieces smaller than this x the image are noise


def plain_background(work: Image.Image) -> np.ndarray | None:
    """Background colour (per-channel median of the border ring) if the border is uniform, else None."""
    a = np.asarray(work.convert("RGB"), np.int16)
    h, w, _ = a.shape
    m = max(2, round(RING_FRAC * max(w, h)))
    ring = np.concatenate([a[:m].reshape(-1, 3), a[-m:].reshape(-1, 3), a[m:-m, :m].reshape(-1, 3), a[m:-m, -m:].reshape(-1, 3)])
    bg = np.median(ring, axis=0)
    share = (np.abs(ring - bg).max(axis=1) <= UNIFORM_TOL).mean()
    return bg if share >= UNIFORM_SHARE else None


def key_mask(work: Image.Image, bg: np.ndarray) -> np.ndarray:
    """Pixels that differ from the background colour, without specks."""
    a = np.asarray(work.convert("RGB"), np.int16)
    fg = np.abs(a - bg).max(axis=2) > KEY_TOL
    keep = [c for c in largest_components(fg, min_ratio=0.0) if c.sum() >= 2]
    return np.logical_or.reduce(keep) if keep else np.zeros(fg.shape, bool)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """mask plus every region of ~mask not connected to the image border (4-connectivity)."""
    n, labels = cv2.connectedComponents((~mask).astype(np.uint8), connectivity=4)
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    outside = np.isin(labels, border[border > 0])
    return ~outside


def key_candidates(work: Image.Image, bg: np.ndarray) -> list[np.ndarray]:
    """The colour key's connected pieces, holes preserved, largest first."""
    fg = key_mask(work, bg)
    pieces = largest_components(fg, min_ratio=0.0)
    return [p for p in pieces if p.sum() >= 2]
