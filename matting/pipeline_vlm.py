"""Pipeline Q: Qwen-VL finds the subject(s) -> EfficientSAM3 segments them -> trimap -> ViTMatte.

Qwen returns each subject as an English noun phrase plus a box. For each subject SAM3 is prompted with the phrase
and the box (as an exemplar) and returns every instance of the concept; the ones mostly inside the box are kept.
That way SAM3, not the box, decides where an object ends (a tight box doesn't cut it off), while the box decides
which instance is meant ("the cat on the right"). Each kept instance is then completed and cleaned with pipeline
A's fuse_masks (see complete_instance), so edges go through the same steps as the SAM2 route: fuse -> trimap ->
ViTMatte. Lives next to pipeline_a.py and only calls its helpers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .common import ViTMatte, compose_rgba, largest_components, make_trimap, resize_long_side
from .pipeline_a import SAFETY_PAD, fuse_masks, pad_box
from .plain_background import key_mask, plain_background
from .qwen import Detection, QwenDetector, Subject
from .sam3 import EfficientSam3, Instances

MIN_INSIDE = 0.5  # share of an instance's mask that must fall inside a (slightly enlarged) Qwen box
BOX_MARGIN = 0.05  # each side of a Qwen box is enlarged by this x its size before measuring
FALLBACK_IOU = 0.3  # a box with no instance inside takes its best-overlapping instance if IoU >= this
CORE_PROB = 0.5  # SAM3's own mask threshold: decides which object is the subject
EXTEND_PROB = 0.3  # lower threshold: completes parts SAM3 is unsure about (a white eagle head, a parrot's beak)
CORE_MIN_RATIO = 0.05  # core pieces smaller than this x the largest one are specks
DUPLICATE_SHARE = 0.5  # an instance this much inside a higher-scored kept one is a duplicate (often object + background)


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


def drop_duplicates(inst: Instances, idx: list[int], share: float = DUPLICATE_SHARE) -> list[int]:
    """Mask NMS by containment: highest score first, skip an instance when `share` of its mask is already covered
    by a kept one. SAM3 returns many such near-duplicates; at a lower threshold their extra pixels are background."""
    kept: list[int] = []
    for i in sorted(idx, key=lambda i: -inst.scores[i]):
        area = inst.masks[i].sum()
        if area and not any((inst.masks[i] & inst.masks[j]).sum() >= share * area for j in kept):
            kept.append(i)
    return sorted(kept)


class PipelineVlm:
    name = "Q: Qwen-VL + EfficientSAM3 + ViTMatte"

    def __init__(self, qwen_model: str = "qwen3-vl-plus", sam_variant: str = "efficientvit", detector=None):
        self.detector = detector or QwenDetector(qwen_model)
        self.sam = EfficientSam3(sam_variant)
        self.matte = ViTMatte()

    def run(self, img: Image.Image, instruction: str | None = None) -> tuple[Image.Image, dict]:
        stages: dict[str, float] = {}
        work = resize_long_side(img)

        if instruction is None:
            # Solid background: the subject is everything that isn't background; no Qwen, no SAM3.
            t = time.perf_counter()
            bg = plain_background(work)
            mask = key_mask(work, bg) if bg is not None else None
            stages["key"] = time.perf_counter() - t
            if mask is not None and mask.any():
                return self._matte(img, work, mask, stages, {"detection": None, "choices": [], "background": bg})
            del stages["key"]

        t = time.perf_counter()
        det: Detection = self.detector.detect(work, instruction)
        stages["qwen"] = time.perf_counter() - t
        if not det.subjects:
            raise NoSubjectError(f"图中没找到：{instruction}" if instruction else "没有识别到主体")

        t = time.perf_counter()
        state = self.sam.set_image(work)
        mask = np.zeros((work.height, work.width), bool)
        choices: list[Choice] = []
        # One decoder pass per subject, its box as the only exemplar (the ONNX decoder takes exactly one box).
        for s in det.subjects:
            inst = self.sam.find(state, s.label, [s.box])
            kept = drop_duplicates(inst, select_instances(inst, [s.box]))
            choices.append(Choice(s, kept, [float(inst.scores[i]) for i in kept]))
            for i in kept:
                mask |= complete_instance(inst.probs[i], s.box)
        stages["sam3"] = time.perf_counter() - t
        if not mask.any():
            raise NoSubjectError("SAM3 没有在 Qwen 给出的位置分割出对象")

        return self._matte(img, work, mask, stages, {"detection": det, "choices": choices})

    def _matte(self, img, work, mask, stages, extra) -> tuple[Image.Image, dict]:
        t = time.perf_counter()
        trimap = make_trimap(mask)
        alpha = self.matte(work, trimap)
        stages["matting"] = time.perf_counter() - t
        info = {"stages": stages, "total": sum(stages.values()), "mask": mask, "trimap": trimap, **extra}
        return compose_rgba(img, alpha), info
