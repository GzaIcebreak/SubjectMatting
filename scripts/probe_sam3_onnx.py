"""Phase-1 check: can EfficientSAM3 be exported to ONNX in three parts, and do they match PyTorch?

    PYTHONPATH=. .venv/bin/python scripts/probe_sam3_onnx.py [--variant efficientvit] [--out onnx_models_sam3]

  sam3_image_encoder   image (1,3,1008,1008) -> FPN features + position encodings
  sam3_text_encoder    token ids (1,16) int64 (CLIP BPE, done in host code) -> text features (16,1,256)
  sam3_decoder         features + text features/mask + exemplar boxes (1,N,4 cxcywh, normalized) -> logits,
                       boxes, mask logits, presence logit

Prints the max abs difference against PyTorch for each part and for the end-to-end masks on a real image.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402

from matting.common import load_rgb, resize_long_side  # noqa: E402
from matting.sam3 import EfficientSam3  # noqa: E402


class ImageEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        out = self.model.backbone.forward_image(image)
        return tuple(out["backbone_fpn"]) + tuple(out["vision_pos_enc"])


class TextEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.lb = model.backbone.language_backbone

    def forward(self, tokens):
        emb = self.lb.encoder.forward_embedding(tokens)
        mem = self.lb.encoder(emb, return_all_tokens=True, input_is_embeddings=True)
        return self.lb.projector(mem).transpose(0, 1)


class Decoder(torch.nn.Module):
    def __init__(self, model, n_levels):
        super().__init__()
        self.model = model
        self.n = n_levels

    def forward(self, *args):
        fpn, pos = list(args[: self.n]), list(args[self.n : 2 * self.n])
        text, text_mask, boxes, box_labels = args[2 * self.n :]
        from sam3.model.data_misc import FindStage

        backbone_out = {
            "backbone_fpn": fpn,
            "vision_pos_enc": pos,
            "vision_features": fpn[-1],
            "language_features": text,
            "language_mask": text_mask,
        }
        prompt = self.model._get_dummy_prompt()
        prompt.append_boxes(boxes.transpose(0, 1), box_labels.transpose(0, 1))
        find = FindStage(
            img_ids=torch.tensor([0], dtype=torch.long), text_ids=torch.tensor([0], dtype=torch.long),
            input_boxes=None, input_boxes_mask=None, input_boxes_label=None, input_points=None, input_points_mask=None,
        )  # fmt: skip
        out = self.model.forward_grounding(backbone_out=backbone_out, find_input=find, geometric_prompt=prompt, find_target=None)
        return out["pred_logits"], out["pred_boxes"], out["pred_masks"], out["presence_logit_dec"]


def export(module, args, path, input_names, output_names, dynamic_axes=None):
    t = time.perf_counter()
    torch.onnx.export(module.eval(), args, str(path), input_names=input_names, output_names=output_names,
                      dynamic_axes=dynamic_axes, opset_version=19, dynamo=False, do_constant_folding=True)  # fmt: skip
    print(f"  exported {path.name}: {path.stat().st_size / 1e6:.1f} MB in {time.perf_counter() - t:.0f}s")


def maxdiff(a, b) -> float:
    return float(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).max())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="efficientvit")
    ap.add_argument("--out", default="onnx_models_sam3")
    ap.add_argument("--image", default="test_images/two_cats.jpg")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    sam = EfficientSam3(args.variant)
    model = sam.model.eval()
    sess = lambda p: ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])  # noqa: E731

    # Real inputs, prepared exactly as Sam3Processor does.
    work = resize_long_side(load_rgb(args.image))
    state = sam.set_image(work)
    image = sam.processor.transform(torch.from_numpy(np.asarray(work)).permute(2, 0, 1)).unsqueeze(0)
    tokens = model.backbone.language_backbone.tokenizer(["cat"], context_length=16)

    results = {}
    print("image encoder:")
    enc = ImageEncoder(model)
    with torch.inference_mode():
        ref_img = enc(image)
    n = len(ref_img) // 2
    names = [f"fpn{i}" for i in range(n)] + [f"pos{i}" for i in range(n)]
    try:
        export(enc, (image,), out / "sam3_image_encoder.onnx", ["image"], names)
        got = sess(out / "sam3_image_encoder.onnx").run(None, {"image": image.numpy()})
        results["image_encoder"] = max(maxdiff(g, r) for g, r in zip(got, ref_img))
    except Exception as e:  # noqa: BLE001
        results["image_encoder"] = f"FAILED {type(e).__name__}: {str(e)[:300]}"
    print("  ", results["image_encoder"], [tuple(t.shape) for t in ref_img])

    print("text encoder:")
    te = TextEncoder(model)
    with torch.inference_mode():
        ref_text = te(tokens)
    try:
        export(te, (tokens,), out / "sam3_text_encoder.onnx", ["tokens"], ["text"])
        got = sess(out / "sam3_text_encoder.onnx").run(None, {"tokens": tokens.numpy()})
        results["text_encoder"] = maxdiff(got[0], ref_text)
    except Exception as e:  # noqa: BLE001
        results["text_encoder"] = f"FAILED {type(e).__name__}: {str(e)[:300]}"
    print("  ", results["text_encoder"], tuple(ref_text.shape))

    print("decoder:")
    dec = Decoder(model, n)
    text_mask = tokens == 0
    boxes = torch.tensor([[[0.75, 0.4, 0.45, 0.75]]], dtype=torch.float32)  # one exemplar, cxcywh normalized
    labels = torch.ones(1, 1, dtype=torch.bool)
    dec_args = (*ref_img, ref_text, text_mask, boxes, labels)
    with torch.inference_mode():
        ref_dec = dec(*dec_args)
    dnames = names + ["text", "text_mask", "boxes", "box_labels"]
    try:
        export(dec, dec_args, out / "sam3_decoder.onnx", dnames, ["logits", "pred_boxes", "masks", "presence"],
               {"boxes": {1: "n"}, "box_labels": {1: "n"}})  # fmt: skip
        s = sess(out / "sam3_decoder.onnx")
        used = {i.name for i in s.get_inputs()}  # inputs the graph doesn't read are pruned by the exporter
        feed = {k: v.numpy() for k, v in zip(dnames, dec_args) if k in used}
        print("  decoder inputs:", sorted(used))
        got = s.run(None, feed)
        results["decoder"] = max(maxdiff(g, r) for g, r in zip(got, ref_dec))
        # Two exemplar boxes: is the box count really dynamic?
        boxes2 = torch.tensor([[[0.75, 0.4, 0.45, 0.75], [0.25, 0.5, 0.45, 0.8]]], dtype=torch.float32)
        labels2 = torch.ones(1, 2, dtype=torch.bool)
        with torch.inference_mode():
            ref2 = dec(*ref_img, ref_text, text_mask, boxes2, labels2)
        got2 = s.run(None, {**feed, "boxes": boxes2.numpy(), "box_labels": labels2.numpy()})
        results["decoder_2_boxes"] = max(maxdiff(g, r) for g, r in zip(got2, ref2))
    except Exception as e:  # noqa: BLE001
        results["decoder"] = f"FAILED {type(e).__name__}: {str(e)[:300]}"
    print("  ", results.get("decoder"), results.get("decoder_2_boxes"))

    print("\nsummary:")
    for k, v in results.items():
        print(f"  {k:16s} {v}")
    sizes = {p.name: round(p.stat().st_size / 1e6, 1) for p in out.glob("*.onnx")}
    print("  sizes (MB, fp32):", sizes, "total", round(sum(sizes.values()), 1))


if __name__ == "__main__":
    main()
