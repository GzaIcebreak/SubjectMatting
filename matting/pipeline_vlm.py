"""Q pipeline: semantic localization, topology-preserving candidates, conservative matting.
Colour-key components preserve holes and small details. Overlapping complementary SAM3
candidates are merged per subject; only disjoint alternatives need semantic judging.
Semantic remove boxes are recorded but not subtracted: colour agreement is insufficient
background evidence. Q-only trimap constraints preserve narrow branches and known holes.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from .common import ViTMatte, compose_rgba, largest_components, make_trimap, resize_long_side
from .pipeline_a import SAFETY_PAD, fuse_masks, pad_box
from .plain_background import key_candidates, plain_background, fill_holes
from .qwen import Detection, QwenDetector, QwenError, QwenJudge, Subject
from .sam3 import EfficientSam3, Instances

MIN_INSIDE = 0.5  # share of an instance's mask that must fall inside a (slightly enlarged) Qwen box
BOX_MARGIN = 0.05  # each side of a Qwen box is enlarged by this x its size before measuring
FALLBACK_IOU = 0.3  # a box with no instance inside takes its best-overlapping instance if IoU >= this
CORE_PROB = 0.5  # SAM3's own mask threshold: decides which object is the subject
EXTEND_PROB = 0.3  # lower threshold: completes parts SAM3 is unsure about (a white eagle head, a parrot's beak)
CORE_MIN_RATIO = 0.05  # core pieces smaller than this x the largest one are specks
DUPLICATE_SHARE = 0.98  # an instance this much inside a higher-scored kept one is a duplicate (often object + background)
MATTE_MARGIN = 64  # ViTMatte runs on the trimap's non-background box plus this margin (pixels), not the whole image
REMOVE_MIN_COVER = 0.1  # a remove region is segmented only if the subject mask covers this much of its box
REMOVE_MAX_AREA = 0.25  # holes and gaps are small: a bigger remove box is ordinary background (wall, grass), ignored
GROW_QUANT = 8  # remove regions grow over pixels whose colour (quantized to this step, +-1 step) occurs in them
GROW_MIN_COUNT = 5  # ... at least this many times
GROW_REACH = 1.0  # ... within the remove box enlarged by this x its size on each side


class NoSubjectError(RuntimeError):
    pass


@dataclass
class Choice:
    """What was kept for one Qwen subject, for reports and debugging."""

    subject: Subject
    kept: list[int]  # indices into that label's Instances
    scores: list[float]


def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def inside_fraction(mask: np.ndarray, box, margin: float = BOX_MARGIN) -> float:
    """Share of the mask's pixels inside `box` enlarged by `margin` of its size on every side."""
    area = mask.sum()
    if area == 0:
        return 0.0
    h, w = mask.shape
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    xa, ya = max(0, int(x1 - mx)), max(0, int(y1 - my))
    xb, yb = min(w, int(np.ceil(x2 + mx))), min(h, int(np.ceil(y2 + my)))
    return float(mask[ya:yb, xa:xb].sum()) / float(area)


def complete_instance(prob: np.ndarray, box, core_prob: float | None = None, extend_prob: float | None = None) -> np.ndarray:
    """One SAM3 instance -> its subject mask, cleaned the way pipeline A cleans SAM2's output (fuse_masks).

    In pipeline A, SAM decides where an object ends and the saliency map decides whether it belongs to the
    subject. Here the confident core (prob > core_prob, specks dropped) plays the saliency map's part and the
    low-threshold mask plays SAM's: pieces of it that touch the core complete the object, pieces that don't
    are discarded, and the result is clipped to the Qwen box padded by SAFETY_PAD of the long side.
    """
    core_prob = CORE_PROB if core_prob is None else core_prob
    extend_prob = EXTEND_PROB if extend_prob is None else extend_prob
    comps = largest_components(prob > core_prob, CORE_MIN_RATIO)
    if not comps:
        return np.zeros(prob.shape, bool)
    core = np.logical_or.reduce(comps)
    h, w = prob.shape
    safety = pad_box(list(box), round(SAFETY_PAD * max(w, h)), w, h)
    return fuse_masks(prob > extend_prob, core, safety)


def select_instances(inst: Instances, boxes: list, min_inside: float = MIN_INSIDE, fallback_iou: float = FALLBACK_IOU) -> list[int]:
    """Instances that lie mostly inside one of the Qwen boxes. A box that keeps nothing falls back to the
    instance overlapping it best (when the overlap is meaningful), so a slightly-off box still gets its object."""
    kept: set[int] = set()
    for box in boxes:
        hits = [i for i in range(len(inst.scores)) if inside_fraction(inst.masks[i], box) >= min_inside]
        if not hits and len(inst.scores):
            ious = [box_iou(inst.boxes[i], box) for i in range(len(inst.scores))]
            best = int(np.argmax(ious))
            if ious[best] >= fallback_iou:
                hits = [best]
        kept.update(hits)
    return sorted(kept)


def matte_box(trimap: np.ndarray, margin: int = MATTE_MARGIN) -> tuple[int, int, int, int]:
    """x0, y0, x1, y1: the trimap's non-background area plus `margin`, clipped to the image."""
    h, w = trimap.shape
    ys, xs = np.nonzero(trimap)
    return (max(0, int(xs.min()) - margin), max(0, int(ys.min()) - margin),
            min(w, int(xs.max()) + 1 + margin), min(h, int(ys.max()) + 1 + margin))  # fmt: skip


def matte_cropped(matte, work: Image.Image, trimap: np.ndarray) -> np.ndarray:
    """ViTMatte on matte_box only; certain background everywhere else."""
    x0, y0, x1, y1 = matte_box(trimap)
    alpha = np.zeros(trimap.shape, np.float32)
    alpha[y0:y1, x0:x1] = matte(work.crop((x0, y0, x1, y1)), trimap[y0:y1, x0:x1])
    return alpha


def drop_duplicates(inst: Instances, idx: list[int], share: float = DUPLICATE_SHARE) -> list[int]:
    """Mask NMS by containment: highest score first, skip an instance when `share` of its mask is already covered
    by a kept one. SAM3 returns many such near-duplicates; at a lower threshold their extra pixels are background."""
    kept: list[int] = []
    for i in sorted(idx, key=lambda i: -inst.scores[i]):
        area = inst.masks[i].sum()
        if area and not any((inst.masks[i] & inst.masks[j]).sum() >= share * area for j in kept):
            kept.append(i)
    return sorted(kept)


def make_trimap_q(mask):
    trimap = make_trimap(mask)
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    padded = np.pad(dist, 1)
    l, r = padded[1:-1, :-2], padded[1:-1, 2:]
    u, d = padded[:-2, 1:-1], padded[2:, 1:-1]
    ridge = ((dist > l) & (dist >= r)) | ((dist > u) & (dist >= d))
    trimap[mask & ridge] = 255
    trimap[fill_holes(mask) & ~mask] = 0
    return trimap


def needs_judge(owners: list[int | None]) -> bool:
    """owners: the subject index of each candidate (None for colour-key pieces). Unambiguous when every subject has
    at most one candidate (and there is at most one key piece)."""
    return any(owners.count(o) > 1 for o in set(owners))


def grow_like(seed: np.ndarray, within: np.ndarray, rgb: np.ndarray, box, q: int = GROW_QUANT,
              min_count: int = GROW_MIN_COUNT, reach: float = GROW_REACH) -> np.ndarray:
    """seed plus the pixels of `within` connected to it whose colour also occurs in it, near the box.

    SAM3 finds the core of a see-through region (a checkerboard in a notch) but stops short where it fades out; the
    rest looks the same, so it is taken by appearance. Pixels that look alike but are not connected through
    look-alike pixels (white teeth vs a white checkerboard cell) stay."""
    if not seed.any():
        return seed
    n = 256 // q
    qi = (rgb // q).astype(np.int64)
    key = (qi[..., 0] * n + qi[..., 1]) * n + qi[..., 2]
    vals, counts = np.unique(key[seed], return_counts=True)
    palette = set()
    for v in vals[counts >= min_count]:
        r, g, b = v // (n * n), (v // n) % n, v % n
        for dr, dg, db in itertools.product((-1, 0, 1), repeat=3):
            palette.add(((r + dr) * n + g + dg) * n + b + db)
    h, w = seed.shape
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) * reach, (y2 - y1) * reach
    near = np.zeros(seed.shape, bool)
    near[max(0, int(y1 - my)): min(h, int(np.ceil(y2 + my))), max(0, int(x1 - mx)): min(w, int(np.ceil(x2 + mx)))] = True
    alike = np.isin(key, np.fromiter(palette, np.int64)) & within & near
    _, labels = cv2.connectedComponents((alike | seed).astype(np.uint8), connectivity=8)
    ids = np.unique(labels[seed])
    return np.isin(labels, ids[ids > 0])


def box_cover(mask: np.ndarray, box) -> float:
    """Share of the box covered by the mask."""
    h, w = mask.shape
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    area = max(1, (x2 - x1) * (y2 - y1))
    return float(mask[y1:y2, x1:x2].sum()) / area


class PipelineVlm:
    name = "Q: Qwen-VL + EfficientSAM3 + ViTMatte"

    def __init__(self, qwen_model: str = "qwen3-vl-plus", sam_variant: str = "efficientvit", detector=None, judge=None):
        self.detector = detector or QwenDetector(qwen_model)
        self.judge = judge if judge is not None else QwenJudge(detector=self.detector)
        self.sam = EfficientSam3(sam_variant)
        self.matte = ViTMatte()

    def run(self, img: Image.Image, instruction: str | None = None, alpha_in: np.ndarray | None = None) -> tuple[Image.Image, dict]:
        """alpha_in: the input's own alpha channel (H x W uint8, at img size), if it has one: what is transparent
        there stays transparent."""
        stages: dict[str, float] = {}
        work = resize_long_side(img)

        t = time.perf_counter()
        det: Detection = self.detector.detect(work, instruction)
        stages["qwen"] = time.perf_counter() - t
        if not det.subjects:
            raise NoSubjectError(f"图中没找到：{instruction}" if instruction else "没有识别到主体")

        # Candidates: the colour key on a solid background (auto mode), else SAM3 per subject.
        t = time.perf_counter()
        bg = plain_background(work) if instruction is None else None
        state = None
        candidates: list[np.ndarray] = []
        owners: list[int | None] = []
        choices: list[Choice] = []
        if bg is not None:
            candidates = key_candidates(work, bg)
            owners = [None] * len(candidates)
            stages["key"] = time.perf_counter() - t
        if not candidates:
            bg = None
            state = self.sam.set_image(work)
            # One decoder pass per subject, its box as the only exemplar (the ONNX decoder takes exactly one box).
            for k, s in enumerate(det.subjects):
                inst = self.sam.find(state, s.label, [s.box])
                kept = drop_duplicates(inst, select_instances(inst, [s.box]))
                choices.append(Choice(s, kept, [float(inst.scores[i]) for i in kept]))
                groups = []
                for i in kept:
                    m = complete_instance(inst.probs[i], s.box)
                    if not m.any(): continue
                    g = 0
                    while g < len(groups):
                        if np.any(m & groups[g]):
                            m |= groups.pop(g)
                            g = 0
                        else: g += 1
                    groups.append(m)
                candidates.extend(groups)
                owners.extend([k] * len(groups))
            stages["sam3"] = time.perf_counter() - t
        if not candidates:
            raise NoSubjectError("SAM3 没有在 Qwen 给出的位置分割出对象")

        keep = list(range(len(candidates)))
        if bg is None and self.judge and needs_judge(owners):
            t = time.perf_counter()
            try:
                keep = self.judge.judge(work, det.subjects, candidates, instruction) or keep
            except QwenError:
                pass  # no second opinion: keep every candidate, as before
            stages["judge"] = time.perf_counter() - t
        mask = np.logical_or.reduce([candidates[i] for i in keep])

        # Semantic boxes/colour similarity are not pixel evidence for subtraction.
        removed = []
        if not mask.any():
            raise NoSubjectError("SAM3 没有在 Qwen 给出的位置分割出对象")

        summary = f"{[s.label for s in det.subjects]} {'key' if bg is not None else 'sam3'} {len(candidates)} cand keep {[i + 1 for i in keep]}"
        if removed:
            summary += f" removed {removed}"
        rgba, info = self._matte(img, work, mask, stages, {"detection": det, "choices": choices, "background": bg, "summary": summary})
        if alpha_in is not None:
            a = np.asarray(rgba).copy()
            a[..., 3] = np.minimum(a[..., 3], alpha_in)
            rgba = Image.fromarray(a, "RGBA")
        return rgba, info

    def _matte(self, img, work, mask, stages, extra) -> tuple[Image.Image, dict]:
        t = time.perf_counter()
        trimap = make_trimap_q(mask)
        alpha = matte_cropped(self.matte, work, trimap)
        stages["matting"] = time.perf_counter() - t
        info = {"stages": stages, "total": sum(stages.values()), "mask": mask, "trimap": trimap, **extra}
        return compose_rgba(img, alpha), info
