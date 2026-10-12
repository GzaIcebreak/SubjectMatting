"""Export EfficientSAM3 to ONNX for pipeline Q's C++ core (cpp/src/pipeline_q.cpp) and matting/pipeline_vlm_onnx.py.

    PYTHONPATH=.:third_party/efficientsam3/sam3 .venv/bin/python scripts/export_sam3_onnx.py [--out onnx_models_q]

Writes <out>/{sam3_image_encoder,sam3_text_encoder,sam3_decoder}.{fp32,fp16}.onnx, clip_bpe_merges.txt for the
tokenizer, and copies vitmatte.fp16.onnx from onnx_models/ (scripts/export_onnx.py).

  sam3_image_encoder  image (1,3,1008,1008): RGB/255, (x-0.5)/0.5 -> fpn0 (1,256,288,288), fpn1 (1,256,144,144),
                      fpn2 (1,256,72,72), pos2 (1,256,72,72)
  sam3_text_encoder   tokens (1,16) int64, CLIP BPE with <start>/<end>, zero padded -> text (16,1,256)
  sam3_decoder        fpn0..2, pos2, text, text_mask (1,16) bool = tokens==0, boxes (1,1,4) cx,cy,w,h in [0,1],
                      box_labels (1,1) bool -> logits (1,200,1), pred_boxes (1,200,4) cx,cy,w,h in [0,1],
                      masks (1,200,288,288) mask logits, presence (1,1)
                      score = sigmoid(logits) * sigmoid(presence)

The decoder takes exactly one exemplar box: upstream merges prompt sequences with data-dependent indexing
(concat_padded_sequences) that the tracer bakes in. Our prompts never contain padding, so for the export it is
replaced by a plain concatenation (identical results, checked below); with that, one box exports and runs, while
the attention reshapes still pin the box count. Callers run the decoder once per box.
"""

from __future__ import annotations

import argparse
import gzip
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

import numpy as np  # noqa: E402
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "third_party/efficientsam3/sam3")]

from matting.common import load_rgb, resize_long_side  # noqa: E402
from matting.sam3 import EfficientSam3  # noqa: E402
from scripts.export_onnx import to_fp16_weights  # noqa: E402

SIZE = 1008
CONTEXT = 16


class ImageEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        out = self.model.backbone.forward_image(image)
        fpn, pos = out["backbone_fpn"], out["vision_pos_enc"]
        return fpn[0], fpn[1], fpn[2], pos[2]


class TextEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.lb = model.backbone.language_backbone

    def forward(self, tokens):
        emb = self.lb.encoder.forward_embedding(tokens)
        mem = self.lb.encoder(emb, return_all_tokens=True, input_is_embeddings=True)
        return self.lb.projector(mem).transpose(0, 1)


class Decoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, fpn0, fpn1, fpn2, pos2, text, text_mask, boxes, box_labels):
        from sam3.model.data_misc import FindStage

        # Only the last level's position encoding is read; the others just need the right spatial sizes.
        pos = [torch.zeros_like(fpn0[:, :1]).expand_as(fpn0), torch.zeros_like(fpn1[:, :1]).expand_as(fpn1), pos2]
        backbone_out = {
            "backbone_fpn": [fpn0, fpn1, fpn2],
            "vision_pos_enc": pos,
            "vision_features": fpn2,
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


def concat_unpadded(seq1, mask1, seq2, mask2, return_index=False):
    """concat_padded_sequences for sequences without padding: a plain concatenation."""
    out = (torch.cat([seq1, seq2], 0), torch.cat([mask1, mask2], 1))
    if return_index:
        idx = (torch.arange(seq2.shape[0]) + seq1.shape[0])[:, None].expand(-1, seq2.shape[1])
        return (*out, idx)
    return out


def export(module, args, path: Path, inputs, outputs) -> None:
    torch.onnx.export(module.eval(), args, str(path), input_names=inputs, output_names=outputs, opset_version=19,
                      dynamo=False, do_constant_folding=True)  # fmt: skip
    m = onnx.load(str(path), load_external_data=True)
    onnx.save(m, str(path))
    for extra in path.parent.glob(f"{path.name}.data"):
        extra.unlink()
    print(f"  {path.name}: {path.stat().st_size / 1e6:.1f} MB")


def maxdiff(got, ref) -> float:
    return max(float(np.abs(np.asarray(g, np.float64) - r.numpy().astype(np.float64)).max()) for g, r in zip(got, ref))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="onnx_models_q")
    ap.add_argument("--variant", default="efficientvit")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    sam = EfficientSam3(args.variant)
    model = sam.model.eval()
    work = resize_long_side(load_rgb(ROOT / "test_images/two_cats.jpg"))
    image = sam.processor.transform(torch.from_numpy(np.asarray(work)).permute(2, 0, 1)).unsqueeze(0)
    tokens = model.backbone.language_backbone.tokenizer(["cat"], context_length=CONTEXT)
    box = torch.tensor([[[0.75, 0.4, 0.45, 0.75]]])
    label = torch.ones(1, 1, dtype=torch.bool)

    enc, te, dec = ImageEncoder(model), TextEncoder(model), Decoder(model)
    with torch.inference_mode():
        feats = enc(image)
        text = te(tokens)
        ref_dec = dec(*feats, text, tokens == 0, box, label)

    import sam3.model.geometry_encoders as ge

    upstream = ge.concat_padded_sequences
    ge.concat_padded_sequences = concat_unpadded
    try:
        with torch.inference_mode():
            assert maxdiff(dec(*feats, text, tokens == 0, box, label), ref_dec) == 0.0
        print("fp32:")
        export(enc, (image,), out / "sam3_image_encoder.fp32.onnx", ["image"], ["fpn0", "fpn1", "fpn2", "pos2"])
        export(te, (tokens,), out / "sam3_text_encoder.fp32.onnx", ["tokens"], ["text"])
        export(dec, (*feats, text, tokens == 0, box, label), out / "sam3_decoder.fp32.onnx",
               ["fpn0", "fpn1", "fpn2", "pos2", "text", "text_mask", "boxes", "box_labels"],
               ["logits", "pred_boxes", "masks", "presence"])  # fmt: skip
    finally:
        ge.concat_padded_sequences = upstream

    print("fp16 (weights only):")
    for n in ["sam3_image_encoder", "sam3_text_encoder", "sam3_decoder"]:
        to_fp16_weights(out / f"{n}.fp32.onnx", out / f"{n}.fp16.onnx")

    print("check against PyTorch:")
    for prec in ["fp32", "fp16"]:
        s = lambda n: ort.InferenceSession(str(out / f"{n}.{prec}.onnx"), providers=["CPUExecutionProvider"])  # noqa: E731
        f = s("sam3_image_encoder").run(None, {"image": image.numpy()})
        t = s("sam3_text_encoder").run(None, {"tokens": tokens.numpy()})
        d = s("sam3_decoder")
        names = {i.name for i in d.get_inputs()}
        feed_dec = dict(zip(["fpn0", "fpn1", "fpn2", "pos2"], f)) | {"text": t[0], "text_mask": (tokens == 0).numpy(),
                                                                    "boxes": box.numpy(), "box_labels": label.numpy()}  # fmt: skip
        got = d.run(None, {k: v for k, v in feed_dec.items() if k in names})
        pm, pr = 1 / (1 + np.exp(-got[2])), torch.sigmoid(ref_dec[2]).numpy()
        print(f"  {prec}: image {maxdiff(f, feats):.1e}  text {maxdiff(t, [text]):.1e}  "
              f"mask prob {float(np.abs(pm - pr).max()):.4f}  mask pixels flipped {int(((pm > .5) != (pr > .5)).sum())}")  # fmt: skip

    # CLIP BPE merges for the C++ tokenizer (the vocabulary is rebuilt from them, as in SimpleTokenizer).
    gz = next((ROOT / "third_party/efficientsam3/sam3").rglob("bpe_simple_vocab_16e6.txt.gz"))
    merges = gzip.open(gz).read().decode("utf-8").split("\n")[1 : 49152 - 256 - 2 + 1]
    (out / "clip_bpe_merges.txt").write_text("\n".join(merges) + "\n", encoding="utf-8")
    print(f"  clip_bpe_merges.txt: {len(merges)} merges")

    vitmatte = ROOT / "onnx_models/vitmatte.fp16.onnx"
    if vitmatte.exists():
        shutil.copy(vitmatte, out / "vitmatte.fp16.onnx")
        print("  copied vitmatte.fp16.onnx")
    else:
        print("  vitmatte.fp16.onnx missing: run scripts/export_onnx.py first")


if __name__ == "__main__":
    main()
