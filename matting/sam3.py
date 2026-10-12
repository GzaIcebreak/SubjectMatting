"""EfficientSAM3 (distilled SAM3: EfficientViT-B1 image encoder + MobileCLIP-S0 text encoder) on CPU.

Upstream code lives in third_party/efficientsam3 (scripts/fetch_efficientsam3.sh) and the weights in
third_party/weights. Given a noun phrase and an example box, SAM3 returns every instance of that concept in the
image; the box is an exemplar ("things like this"), not a crop, so callers pick instances themselves.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
UPSTREAM = ROOT / "third_party/efficientsam3/sam3"
WEIGHTS = ROOT / "third_party/weights"

# (backbone_type, model_name) per checkpoint, read off the state dict shapes.
VARIANTS = {
    "efficientvit": ("efficientvit", "b1"),
    "repvit": ("repvit", "m1.1"),
    "tinyvit": ("tinyvit", "11m"),
}


@dataclass
class Instances:
    masks: np.ndarray  # bool (N, H, W): probs > 0.5
    probs: np.ndarray  # float32 (N, H, W): per-pixel mask probability
    boxes: np.ndarray  # (N, 4) x1, y1, x2, y2 pixels
    scores: np.ndarray  # (N,)


class EfficientSam3:
    def __init__(self, variant: str = "efficientvit", confidence: float = 0.3):
        if str(UPSTREAM) not in sys.path:
            sys.path.insert(0, str(UPSTREAM))
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_efficientsam3_image_model

        backbone, name = VARIANTS[variant]
        self.model = build_efficientsam3_image_model(
            checkpoint_path=str(WEIGHTS / f"efficientsam3_{variant}.pt"),
            device="cpu",
            backbone_type=backbone,
            model_name=name,
            text_encoder_type="MobileCLIP-S0",
            text_encoder_context_length=16,
            load_from_HF=False,
        )
        self.processor = Sam3Processor(self.model, device="cpu", confidence_threshold=confidence)

    def set_image(self, img: Image.Image) -> dict:
        """Encodes the image once; reuse the state for several prompts."""
        return self.processor.set_image(img.convert("RGB"))

    @torch.inference_mode()
    def find(self, state: dict, label: str, boxes: list[tuple[float, float, float, float]] = ()) -> Instances:
        """All instances of `label`, with `boxes` (pixels) as positive exemplars. One decoder pass."""
        p = self.processor
        p.reset_all_prompts(state)
        state["backbone_out"].update(self.model.backbone.forward_text([label], device="cpu"))
        state["geometric_prompt"] = self.model._get_dummy_prompt()
        w, h = state["original_width"], state["original_height"]
        for x1, y1, x2, y2 in boxes:
            cxcywh = torch.tensor([(x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h], dtype=torch.float32)
            state["geometric_prompt"].append_boxes(cxcywh.view(1, 1, 4), torch.tensor([True]).view(1, 1))
        state = p._forward_grounding(state)
        return Instances(
            masks=state["masks"].squeeze(1).cpu().numpy().astype(bool),
            probs=state["masks_logits"].squeeze(1).cpu().numpy().astype(np.float32),
            boxes=state["boxes"].cpu().numpy().reshape(-1, 4),
            scores=state["scores"].cpu().numpy().reshape(-1),
        )
