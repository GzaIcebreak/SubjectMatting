"""Pipeline A: saliency (BiRefNet_lite) -> point prompts -> SAM2.1-hiera-tiny object completion -> trimap -> ViTMatte.

Division of labour: SAM decides where an object begins and ends; saliency decides whether that object is
part of the subject. Saliency alone stops short of low-contrast parts (e.g. the white underside of a plate
below its rim), and a box built from it would truncate SAM there, so no box prompt is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PIL import Image

from .common import DEVICE, Timer, ViTMatte, compose_rgba, largest_components, make_trimap, resize_long_side
from .saliency import Saliency

MAX_POINTS = 25
# A SAM object counts as subject when >= MIN_INSIDE of it lies in the salient region AND its mask is stable
# (SAM's stability score: IoU of the masks at logit thresholds +1 and -1; soft gradients such as a glow halo
# around a logo score 0.08-0.34, real object parts saliency missed - a plate underside, the top of a gift box -
# score 0.46-0.95). Inside ratio alone cannot separate them (glow 0.64-0.67 vs box top 0.71-0.74).
MIN_INSIDE = 0.6
MIN_STABILITY = 0.4
SAFETY_PAD = 0.15  # final clip: saliency bbox padded by this x long side


@dataclass
class Prompts:
    box: list[float]  # x1, y1, x2, y2 — tight bbox of the salient region
    points: list[list[float]]  # [[x, y], ...] positive points, one SAM object each
    sal_mask: np.ndarray  # bool (H, W)


def touches_border(mask: np.ndarray) -> bool:
    return bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())


def sample_points(mask: np.ndarray, max_points: int = MAX_POINTS, erode_frac: float = 0.015, grid_frac: float = 0.1):
    """Positive prompt points inside `mask`: one distance-transform peak per component (so thin parts are
    covered) plus a coarse grid over the eroded core (so big objects get several object-level guesses).
    Components come largest first, so when the budget is tight the biggest ones keep their peaks."""
    h, w = mask.shape
    long = max(h, w)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    order = sorted(range(1, n), key=lambda i: -stats[i, cv2.CC_STAT_AREA])
    peaks = []
    for i in order:
        dist = cv2.distanceTransform((labels == i).astype(np.uint8), cv2.DIST_L2, 5)
        y, x = np.unravel_index(np.argmax(dist), dist.shape)
        peaks.append([float(x), float(y)])
    k = max(1, round(erode_frac * long))
    core = cv2.erode(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    step = max(8, round(grid_frac * long))
    grid = [[float(x), float(y)] for y in range(step // 2, h, step) for x in range(step // 2, w, step) if core[y, x]]
    budget = max_points - len(peaks)
    if budget <= 0:
        return peaks[:max_points]
    if len(grid) > budget:
        idx = np.linspace(0, len(grid) - 1, budget).round().astype(int)
        grid = [grid[i] for i in idx]
    return peaks + grid


def prompts_from_saliency(sal: np.ndarray, thr: float = 0.5, max_points: int = MAX_POINTS) -> Prompts:
    """Secondary components cut by the image border are dropped: they are almost always scenery peeking into
    the frame (e.g. a leaf sprig in a corner), not a second subject. The largest component is always kept."""
    h, w = sal.shape
    comps = largest_components(sal > thr)
    comps = comps[:1] + [c for c in comps[1:] if not touches_border(c)]
    if not comps:
        # Nothing salient: fall back to the central 80% of the image.
        box = [0.1 * w, 0.1 * h, 0.9 * w, 0.9 * h]
        return Prompts(box, [[w / 2, h / 2]], np.zeros((h, w), bool))
    sal_mask = np.logical_or.reduce(comps)
    ys, xs = np.nonzero(sal_mask)
    box = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
    return Prompts(box, sample_points(sal_mask, max_points), sal_mask)


def complete_mask(
    cands: np.ndarray,
    sal: np.ndarray,
    stability: np.ndarray | None = None,
    min_inside: float = MIN_INSIDE,
    min_stability: float = MIN_STABILITY,
) -> np.ndarray:
    """cands: bool (N points, 3 levels, H, W); stability: (N, 3) or None. For each point keep the LARGEST level
    whose mask lies at least `min_inside` inside the salient region (object level rather than part level, but
    not the whole scene) and is stable (not a soft-edged guess). Union of the kept masks."""
    if cands.size == 0:
        return np.zeros_like(sal)
    areas = cands.sum(axis=(2, 3))
    inside = (cands & sal).sum(axis=(2, 3)) / np.maximum(areas, 1)
    ok = (areas > 0) & (inside >= min_inside)
    if stability is not None:
        ok &= stability >= min_stability
    best = np.where(ok, areas, -1).argmax(axis=1)
    out = np.zeros_like(sal)
    for i in np.nonzero(ok.any(axis=1))[0]:
        out |= cands[i, best[i]]
    return out


def pad_box(box: list[float], pad: int, w: int, h: int) -> list[float]:
    x1, y1, x2, y2 = box
    return [max(0, x1 - pad), max(0, y1 - pad), min(w - 1, x2 + pad), min(h - 1, y2 + pad)]


def fuse_masks(sam: np.ndarray, sal: np.ndarray, box: list[float]) -> np.ndarray:
    """Saliency mask + the SAM components that touch it, clipped to the box.

    SAM refines boundaries and extends into parts saliency missed; saliency recovers thin parts SAM drops.
    SAM components that don't touch the salient region are background specks (they'd become floating
    debris after matting), so they are discarded.
    """
    n, labels = cv2.connectedComponents(sam.astype(np.uint8), connectivity=8)
    touching = np.unique(labels[sal & sam])
    sam_kept = np.isin(labels, touching[touching > 0])
    x1, y1, x2, y2 = (int(v) for v in box)
    box_mask = np.zeros_like(sal)
    box_mask[y1 : y2 + 1, x1 : x2 + 1] = True
    return (sam_kept | sal) & box_mask


class Sam2Tiny:
    MODEL_ID = "facebook/sam2.1-hiera-tiny"

    def __init__(self):
        from transformers import Sam2Model, Sam2Processor

        self.processor = Sam2Processor.from_pretrained(self.MODEL_ID)
        self.model = Sam2Model.from_pretrained(self.MODEL_ID).to(DEVICE).eval()

    @torch.inference_mode()
    def set_image(self, img: Image.Image) -> tuple:
        """Encode the image once (the expensive part); the predict methods then only run the decoder."""
        inputs = self.processor(images=img, return_tensors="pt").to(DEVICE)
        return self.model.get_image_embeddings(inputs["pixel_values"]), inputs["original_sizes"]

    @torch.inference_mode()
    def _decode(self, state: tuple, **prompt_kwargs) -> tuple[np.ndarray, np.ndarray]:
        embeddings, original_sizes = state
        inputs = self.processor(original_sizes=original_sizes, return_tensors="pt", **prompt_kwargs).to(DEVICE)
        inputs.pop("original_sizes")
        out = self.model(image_embeddings=embeddings, **inputs, multimask_output=True)
        logits = self.processor.post_process_masks(out.pred_masks.cpu(), original_sizes, binarize=False)[0]
        # SAM's stability score: how much the mask changes when the logit threshold moves by +-1.
        stability = (logits > 1).sum(dim=(-2, -1)) / (logits > -1).sum(dim=(-2, -1)).clamp(min=1)
        return (logits > 0).numpy(), out.iou_scores[0].float().numpy(), stability.float().numpy()

    def predict_points(self, state: tuple, points: list[list[float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Each point prompts its own object. -> (masks bool (N, 3, H, W), scores (N, 3), stability (N, 3))."""
        return self._decode(state, input_points=[[[p] for p in points]], input_labels=[[[1] for _ in points]])

    def predict_box(self, state: tuple, box: list[float], points: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
        """One object from a box + points. -> (masks bool (3, H, W), scores (3,))."""
        masks, scores, _ = self._decode(
            state, input_boxes=[[box]], input_points=[[points]], input_labels=[[[1] * len(points)]]
        )
        return masks[0], scores[0]


class PipelineA:
    name = "A: BiRefNet_lite + SAM2.1-tiny + ViTMatte"

    def __init__(self):
        self.saliency = Saliency()
        self.sam = Sam2Tiny()
        self.matte = ViTMatte()

    def run(self, img: Image.Image) -> tuple[Image.Image, dict]:
        timer = Timer()
        work = resize_long_side(img)
        w, h = work.size
        with timer.stage("saliency"):
            sal = self.saliency(work)
            prompts = prompts_from_saliency(sal)
        with timer.stage("sam2"):
            state = self.sam.set_image(work)
            if prompts.sal_mask.any():
                cands, _, stability = self.sam.predict_points(state, prompts.points)
                completed = complete_mask(cands, prompts.sal_mask, stability)
                safety = pad_box(prompts.box, round(SAFETY_PAD * max(w, h)), w, h)
                mask = fuse_masks(completed, prompts.sal_mask, safety)
            else:
                masks, scores = self.sam.predict_box(state, prompts.box, prompts.points)
                mask = masks[int(np.argmax(scores))]
        with timer.stage("matting"):
            trimap = make_trimap(mask)
            alpha = self.matte(work, trimap)
        info = {
            "stages": timer.stages,
            "total": timer.total,
            "n_points": len(prompts.points),
            "prompts": {"box": prompts.box, "points": prompts.points},
            "mask": mask,
            "trimap": trimap,
        }
        return compose_rgba(img, alpha), info
