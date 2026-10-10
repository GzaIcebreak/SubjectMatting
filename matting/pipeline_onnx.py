"""Pipeline A with every model on ONNX Runtime (models from scripts/export_onnx.py).

This is the reference the C++ library (cpp/) is checked against, so the pre/post-processing is spelled out with
plain PIL/numpy operations instead of the Hugging Face processors. The mask logic is shared with pipeline_a.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

# Official onnxruntime builds start Microsoft telemetry unless this is set before the first session.
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

import onnxruntime as ort  # noqa: E402
from PIL import Image

from .common import compose_rgba, make_trimap, resize_long_side
from .pipeline_a import SAFETY_PAD, complete_mask, fuse_masks, pad_box, prompts_from_saliency

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
SAM_SIZE = 1024
SAM_MASK_SIZE = 256


def session(path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def imagenet_tensor(img: Image.Image, w: int, h: int) -> np.ndarray:
    """RGB image -> (1,3,h,w) float32, resized with PIL bilinear and ImageNet-normalized."""
    x = np.asarray(img.convert("RGB").resize((w, h), Image.BILINEAR), np.float32) / 255.0
    return ((x - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)[None]


def upsample_bilinear(x: np.ndarray, w: int, h: int) -> np.ndarray:
    """float (..., H0, W0) -> (..., h, w), half-pixel centers (= torch interpolate bilinear, align_corners=False)."""
    import cv2

    lead = x.shape[:-2]
    flat = x.reshape(-1, *x.shape[-2:])
    out = np.stack([cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR) for m in flat])
    return out.reshape(*lead, h, w)


class OnnxSaliency:
    """BiRefNet_lite at 768, retried at 1024 when the map comes out (almost) empty."""

    def __init__(self, path: Path, size: int = 768, retry_size: int = 1024):
        self.sess = session(path)
        self.size = size
        self.retry_size = retry_size

    def __call__(self, img: Image.Image) -> np.ndarray:
        sal = self._predict(img, self.size)
        if (sal > 0.5).mean() < 0.005 and self.retry_size:
            sal = self._predict(img, self.retry_size)
        return sal

    def _predict(self, img: Image.Image, size: int) -> np.ndarray:
        pred = self.sess.run(None, {"pixel_values": imagenet_tensor(img, size, size)})[0][0, 0]
        small = Image.fromarray((pred * 255).astype(np.uint8))
        return np.asarray(small.resize(img.size, Image.BILINEAR)) / 255.0


class OnnxSam2:
    def __init__(self, encoder: Path, decoder: Path):
        self.encoder = session(encoder)
        self.decoder = session(decoder)

    def set_image(self, img: Image.Image) -> tuple:
        feats = self.encoder.run(None, {"pixel_values": imagenet_tensor(img, SAM_SIZE, SAM_SIZE)})
        return feats, img.size

    def _decode(self, state: tuple, points: np.ndarray, labels: np.ndarray):
        """points (N, P, 2) in image pixels, labels (N, P) -> masks bool (N,3,H,W), scores (N,3), stability (N,3)."""
        (feat0, feat1, embed), (w, h) = state
        scaled = points.astype(np.float32) * np.array([SAM_SIZE / w, SAM_SIZE / h], np.float32)
        logits, scores = self.decoder.run(
            None,
            {
                "high_res_feat0": feat0,
                "high_res_feat1": feat1,
                "image_embed": embed,
                "points": scaled[None],
                "labels": labels.astype(np.int64)[None],
            },
        )
        logits = upsample_bilinear(logits[0], w, h)
        stability = (logits > 1).sum(axis=(-2, -1)) / np.maximum((logits > -1).sum(axis=(-2, -1)), 1)
        return logits > 0, scores[0], stability.astype(np.float32)

    def predict_points(self, state: tuple, points: list[list[float]]):
        """Each point prompts its own object."""
        pts = np.array(points, np.float32)[:, None, :]
        return self._decode(state, pts, np.ones((len(points), 1), np.int64))

    def predict_box(self, state: tuple, box: list[float], points: list[list[float]]):
        """One object from points + a box (its corners as labels 2 and 3, after the points)."""
        pts = np.array(points + [box[:2], box[2:]], np.float32)[None]
        labels = np.array([[1] * len(points) + [2, 3]], np.int64)
        masks, scores, _ = self._decode(state, pts, labels)
        return masks[0], scores[0]


class OnnxViTMatte:
    def __init__(self, path: Path):
        self.sess = session(path)

    def __call__(self, img: Image.Image, trimap: np.ndarray) -> np.ndarray:
        w, h = img.size
        rgb = np.asarray(img.convert("RGB"), np.float32) / 255.0 * 2.0 - 1.0
        x = np.concatenate([rgb.transpose(2, 0, 1), (trimap.astype(np.float32) / 255.0)[None]])
        ph, pw = -h % 32, -w % 32
        x = np.pad(x, ((0, 0), (0, ph), (0, pw)))
        alpha = self.sess.run(None, {"pixel_values": x[None]})[0][0, 0, :h, :w].astype(np.float32)
        alpha[trimap == 255] = 1.0
        alpha[trimap == 0] = 0.0
        return alpha


class PipelineOnnx:
    name = "A (ONNX)"

    def __init__(self, models_dir: str | Path = "onnx_models", precision: str = "fp16"):
        d = Path(models_dir)
        self.saliency = OnnxSaliency(d / f"birefnet_lite.{precision}.onnx")
        self.sam = OnnxSam2(d / f"sam2_encoder.{precision}.onnx", d / f"sam2_decoder.{precision}.onnx")
        self.matte = OnnxViTMatte(d / f"vitmatte.{precision}.onnx")

    def run(self, img: Image.Image) -> tuple[Image.Image, dict]:
        stages: dict[str, float] = {}
        t = time.perf_counter()
        work = resize_long_side(img)
        w, h = work.size
        sal = self.saliency(work)
        prompts = prompts_from_saliency(sal)
        stages["saliency"] = time.perf_counter() - t

        t = time.perf_counter()
        state = self.sam.set_image(work)
        if prompts.sal_mask.any():
            cands, _, stability = self.sam.predict_points(state, prompts.points)
            completed = complete_mask(cands, prompts.sal_mask, stability)
            safety = pad_box(prompts.box, round(SAFETY_PAD * max(w, h)), w, h)
            mask = fuse_masks(completed, prompts.sal_mask, safety)
        else:
            masks, scores = self.sam.predict_box(state, prompts.box, prompts.points)
            mask = masks[int(np.argmax(scores))]
        stages["sam2"] = time.perf_counter() - t

        t = time.perf_counter()
        trimap = make_trimap(mask)
        alpha = self.matte(work, trimap)
        stages["matting"] = time.perf_counter() - t
        info = {"stages": stages, "total": sum(stages.values()), "mask": mask, "trimap": trimap}
        return compose_rgba(img, alpha), info
