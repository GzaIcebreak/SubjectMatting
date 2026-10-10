"""Writes test cases for sm_ops_test: inputs plus the outputs PIL / OpenCV produce for them.

    .venv/bin/python cpp/tests/gen_ops_cases.py build/ops_cases && build/sm_ops_test build/ops_cases

manifest.txt has one case per line: <op> <params...> <input file> <expected file>. Files are raw bytes
(uint8, or float32 little endian for float data).
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
lines = []


def save(name: str, arr: np.ndarray) -> str:
    (out / name).write_bytes(np.ascontiguousarray(arr).tobytes())
    return name


def smooth_image(w, h, c):
    """Natural-ish image: blurred noise plus edges, so resampling is exercised on real gradients."""
    x = rng.integers(0, 256, (h, w, c), dtype=np.uint8)
    x = cv2.GaussianBlur(x, (0, 0), 3)
    x[h // 3 : h // 2, :] = 255 - x[h // 3 : h // 2, :]
    return x if c > 1 else x[..., None]


def blobs(w, h, n=6):
    m = np.zeros((h, w), np.uint8)
    for _ in range(n):
        cx, cy = rng.integers(0, w), rng.integers(0, h)
        ax, ay = rng.integers(3, w // 4), rng.integers(3, h // 4)
        cv2.ellipse(m, (int(cx), int(cy)), (int(ax), int(ay)), float(rng.integers(0, 180)), 0, 360, 1, -1)
    m[h // 2, :] = 1  # a 1px line: thin structures matter for the trimap
    m[5:9, w - 3 :] = 1  # touching the border
    return m


# PIL resize, RGB and L, up and down, both filters
for i, (sw, sh, dw, dh, c, f) in enumerate(
    [
        (300, 200, 123, 457, 3, "bilinear"),
        (1500, 1000, 1024, 683, 3, "bicubic"),
        (200, 150, 768, 768, 3, "bilinear"),
        (1024, 683, 1024, 1024, 3, "bilinear"),
        (768, 768, 1024, 683, 1, "bilinear"),
        (1024, 1024, 641, 1024, 1, "bilinear"),
        (333, 777, 600, 1024, 3, "bicubic"),
    ]
):
    src = smooth_image(sw, sh, c)
    mode = "RGB" if c == 3 else "L"
    img = Image.fromarray(src if c == 3 else src[..., 0], mode)
    exp = np.asarray(img.resize((dw, dh), Image.BILINEAR if f == "bilinear" else Image.BICUBIC))
    lines.append(f"pil_resize {sw} {sh} {c} {dw} {dh} {f} {save(f'r{i}_in', src)} {save(f'r{i}_exp', exp)}")

# cv2 INTER_LINEAR on float logits
for i, (sw, sh, dw, dh) in enumerate([(256, 256, 1024, 683), (256, 256, 700, 1024), (64, 48, 64, 48), (100, 80, 333, 251)]):
    src = rng.normal(0, 4, (sh, sw)).astype(np.float32)
    exp = cv2.resize(src, (dw, dh), interpolation=cv2.INTER_LINEAR)
    lines.append(f"resize_linear {sw} {sh} {dw} {dh} {save(f'l{i}_in', src)} {save(f'l{i}_exp', exp)}")

# Morphology, connected components and distance transforms on blob masks
for i, (w, h) in enumerate([(1024, 683), (640, 1024), (97, 61)]):
    m = blobs(w, h)
    mi = save(f"m{i}", m)
    for r in (1, 5, 12):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        lines.append(f"erode {w} {h} {r} {mi} {save(f'm{i}_e{r}', cv2.erode(m, k))}")
        lines.append(f"dilate {w} {h} {r} {mi} {save(f'm{i}_d{r}', cv2.dilate(m, k))}")
    _, labels = cv2.connectedComponents(m, connectivity=8)
    lines.append(f"components {w} {h} {mi} {save(f'm{i}_cc', labels.astype(np.int32))}")
    for ms in (3, 5):
        d = cv2.distanceTransform(m, cv2.DIST_L2, ms)
        lines.append(f"distance {w} {h} {ms} {mi} {save(f'm{i}_dt{ms}', d.astype(np.float32))}")

(out / "manifest.txt").write_text("\n".join(lines) + "\n")
print(f"{len(lines)} cases in {out}")
