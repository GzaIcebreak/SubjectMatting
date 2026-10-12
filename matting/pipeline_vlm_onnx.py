"""Pipeline Q with EfficientSAM3 and ViTMatte on ONNX Runtime (models from scripts/export_sam3_onnx.py).

The reference for the C++ core (cpp/src/pipeline_q.cpp): same selection and cleanup as pipeline_vlm.py, with the
SAM3 pre/post-processing spelled out in reproducible operations (PIL bilinear resize to 1008, OpenCV bilinear
upsampling of the mask logits). Tokenization uses the upstream CLIP tokenizer; the C++ port is tested against it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

import onnxruntime as ort  # noqa: E402

from .pipeline_onnx import OnnxViTMatte, session, upsample_bilinear  # noqa: E402
from .pipeline_vlm import PipelineVlm  # noqa: E402
from .sam3 import UPSTREAM, Instances  # noqa: E402

SAM3_SIZE = 1008
CONTEXT = 16
CONFIDENCE = 0.3  # same as matting/sam3.py


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def clip_tokenizer():
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    from sam3.model.tokenizer_ve import SimpleTokenizer

    gz = next(UPSTREAM.parent.rglob("bpe_simple_vocab_16e6.txt.gz"))
    return SimpleTokenizer(bpe_path=str(gz))


class OnnxSam3:
    """Drop-in for matting.sam3.EfficientSam3 (set_image / find) on the exported graphs."""

    def __init__(self, models_dir: str | Path = "onnx_models_q", precision: str = "fp16", confidence: float = CONFIDENCE):
        d = Path(models_dir)
        self.image_encoder = session(d / f"sam3_image_encoder.{precision}.onnx")
        self.text_encoder = session(d / f"sam3_text_encoder.{precision}.onnx")
        self.decoder = session(d / f"sam3_decoder.{precision}.onnx")
        self.decoder_inputs = {i.name for i in self.decoder.get_inputs()}
        self.tokenizer = clip_tokenizer()
        self.confidence = confidence
        self._text_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def set_image(self, img: Image.Image) -> dict:
        x = np.asarray(img.convert("RGB").resize((SAM3_SIZE, SAM3_SIZE), Image.BILINEAR), np.float32) / 255.0
        x = ((x - 0.5) / 0.5).transpose(2, 0, 1)[None]
        fpn0, fpn1, fpn2, pos2 = self.image_encoder.run(None, {"image": x})
        return {"feats": {"fpn0": fpn0, "fpn1": fpn1, "fpn2": fpn2, "pos2": pos2}, "size": img.size}

    def _text(self, label: str) -> tuple[np.ndarray, np.ndarray]:
        if label not in self._text_cache:
            tokens = self.tokenizer([label], context_length=CONTEXT).numpy()
            (text,) = self.text_encoder.run(None, {"tokens": tokens})
            self._text_cache[label] = (text, tokens == 0)
        return self._text_cache[label]

    def find(self, state: dict, label: str, boxes) -> Instances:
        (box,) = boxes  # the exported decoder takes exactly one exemplar
        w, h = state["size"]
        text, text_mask = self._text(label)
        x1, y1, x2, y2 = box
        cxcywh = np.array([[[(x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h]]], np.float32)
        feed = {**state["feats"], "text": text, "text_mask": text_mask, "boxes": cxcywh, "box_labels": np.ones((1, 1), bool)}
        logits, pred_boxes, masks, presence = self.decoder.run(None, {k: v for k, v in feed.items() if k in self.decoder_inputs})
        scores = sigmoid(logits[0, :, 0]) * sigmoid(presence[0, 0])
        keep = np.nonzero(scores > self.confidence)[0]
        cx, cy, bw, bh = pred_boxes[0, keep].T
        xyxy = np.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], 1)
        probs = sigmoid(upsample_bilinear(masks[0, keep], w, h)).astype(np.float32) if len(keep) else np.zeros((0, h, w), np.float32)
        return Instances(probs > 0.5, probs, xyxy.reshape(-1, 4), scores[keep])


class PipelineVlmOnnx(PipelineVlm):
    name = "Q (ONNX): Qwen-VL + EfficientSAM3 + ViTMatte"

    def __init__(self, models_dir: str | Path = "onnx_models_q", precision: str = "fp16", qwen_model: str = "qwen3-vl-plus", detector=None):
        from .qwen import QwenDetector

        self.detector = detector or QwenDetector(qwen_model)
        self.sam = OnnxSam3(models_dir, precision)
        self.matte = OnnxViTMatte(Path(models_dir) / "vitmatte.fp16.onnx")


__all__ = ["OnnxSam3", "PipelineVlmOnnx", "ort"]
