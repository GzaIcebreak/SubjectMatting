"""Salient-object map with ZhengPeng7/BiRefNet_lite.

BiRefNet_lite is trained at 1024 but that costs ~9.5s on CPU; 768 costs ~2.6s. At 512 it sometimes outputs an
all-zero map (e.g. test_images/cat_whiskers.jpg), so 768 is the default and 1024 is the retry for empty maps.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from .common import DEVICE

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class Saliency:
    MODEL_ID = "ZhengPeng7/BiRefNet_lite"

    def __init__(self, size: int = 768, retry_size: int = 1024):
        from transformers import AutoModelForImageSegmentation

        self.size = size
        self.retry_size = retry_size
        self.model = (
            AutoModelForImageSegmentation.from_pretrained(self.MODEL_ID, trust_remote_code=True)
            .to(DEVICE)
            .eval()
            .float()
        )

    def __call__(self, img: Image.Image) -> np.ndarray:
        """-> saliency float32 in [0,1] at img size."""
        sal = self._predict(img, self.size)
        if (sal > 0.5).mean() < 0.005 and self.retry_size:
            sal = self._predict(img, self.retry_size)
        return sal

    @torch.inference_mode()
    def _predict(self, img: Image.Image, size: int) -> np.ndarray:
        x = np.asarray(img.convert("RGB").resize((size, size), Image.BILINEAR), np.float32) / 255.0
        x = torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
        pred = self.model(x)[-1].sigmoid()[0, 0].cpu().numpy()
        return np.asarray(Image.fromarray((pred * 255).astype(np.uint8)).resize(img.size, Image.BILINEAR)) / 255.0
