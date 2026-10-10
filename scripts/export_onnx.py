"""Export the three pipeline models to ONNX for the C++ library (cpp/) and matting/pipeline_onnx.py.

    .venv/bin/python scripts/export_onnx.py [--out onnx_models]

Writes, for each precision in fp32 / fp16 / int8:
    birefnet_lite.<p>.onnx   pixel_values (1,3,H,W) ImageNet-normalized  -> saliency (1,1,H,W) in [0,1]
    sam2_encoder.<p>.onnx    pixel_values (1,3,1024,1024) ImageNet-normalized
                             -> high_res_feat0 (1,32,256,256), high_res_feat1 (1,64,128,128), image_embed (1,256,64,64)
    sam2_decoder.<p>.onnx    the three features + points (1,N,P,2) in 1024x1024 space, labels (1,N,P) int64
                             (1 = positive, 0 = negative, 2/3 = box top-left/bottom-right, -1 = padding)
                             -> masks (1,N,3,256,256) logits, iou_scores (1,N,3)
    vitmatte.<p>.onnx        pixel_values (1,4,H,W): RGB*2-1 and trimap/255, H and W multiples of 32 -> alphas (1,1,H,W)

fp16 stores weights as float16 and casts them back to float32 when the session loads (CPU kernels are float32),
so it halves the files without changing the arithmetic. int8 is ONNX Runtime dynamic quantization (weights int8).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")  # onnxruntime is used for int8 quantization

import numpy as np  # noqa: E402
import onnx  # noqa: E402
import torch  # noqa: E402
from onnx import TensorProto, helper, numpy_helper  # noqa: E402
from torch.onnx import register_custom_op_symbolic  # noqa: E402
from torch.onnx import symbolic_helper as sh  # noqa: E402

OPSET = 19  # first opset with DeformConv (BiRefNet's decoder uses deformable convolutions)


@sh.parse_args("v", "v", "v", "v", "v", "i", "i", "i", "i", "i", "i", "i", "i", "b")
def _deform_conv2d(g, inp, weight, offset, mask, bias, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, n_wg, n_og, use_mask):
    args = [inp, weight, offset, bias] + ([mask] if use_mask else [])
    return g.op(
        "DeformConv",
        *args,
        dilations_i=[dil_h, dil_w],
        group_i=n_wg,
        kernel_shape_i=sh._get_tensor_sizes(weight)[2:],
        offset_group_i=n_og,
        pads_i=[pad_h, pad_w, pad_h, pad_w],
        strides_i=[stride_h, stride_w],
    )


register_custom_op_symbolic("torchvision::deform_conv2d", _deform_conv2d, OPSET)


def export(module, args, path: Path, input_names, output_names, dynamic_axes) -> None:
    torch.onnx.export(
        module.eval(),
        args,
        str(path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=OPSET,
        dynamo=False,
        do_constant_folding=True,
    )
    # Keep everything in one file (no external data) so the C++ side loads a single path per model.
    model = onnx.load(str(path), load_external_data=True)
    onnx.save(model, str(path))
    for extra in path.parent.glob(f"{path.name}.data"):
        extra.unlink()
    print(f"  {path.name}: {path.stat().st_size / 1e6:.1f} MB")


class BiRefNetWrap(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model(pixel_values)[-1].sigmoid()


class Sam2EncoderWrap(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        feat0, feat1, embed = self.model.get_image_embeddings(pixel_values)
        return feat0, feat1, embed


class Sam2DecoderWrap(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, feat0, feat1, embed, points, labels):
        out = self.model(image_embeddings=[feat0, feat1, embed], input_points=points, input_labels=labels, multimask_output=True)
        return out.pred_masks, out.iou_scores


class VitMatteWrap(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values).alphas


def export_fp32(out: Path) -> None:
    from matting.common import ViTMatte
    from matting.pipeline_a import Sam2Tiny
    from matting.saliency import Saliency

    print("fp32:")
    birefnet = Saliency().model
    export(
        BiRefNetWrap(birefnet), (torch.randn(1, 3, 768, 768),), out / "birefnet_lite.fp32.onnx",
        ["pixel_values"], ["saliency"], {"pixel_values": {2: "h", 3: "w"}, "saliency": {2: "h", 3: "w"}},
    )  # fmt: skip

    sam = Sam2Tiny().model
    with torch.inference_mode():
        feats = sam.get_image_embeddings(torch.randn(1, 3, 1024, 1024))
    export(
        Sam2EncoderWrap(sam), (torch.randn(1, 3, 1024, 1024),), out / "sam2_encoder.fp32.onnx",
        ["pixel_values"], ["high_res_feat0", "high_res_feat1", "image_embed"], None,
    )  # fmt: skip
    points = torch.rand(1, 3, 2, 2) * 1024
    labels = torch.ones(1, 3, 2, dtype=torch.int64)
    export(
        Sam2DecoderWrap(sam), (*feats, points, labels), out / "sam2_decoder.fp32.onnx",
        ["high_res_feat0", "high_res_feat1", "image_embed", "points", "labels"], ["masks", "iou_scores"],
        {"points": {1: "n", 2: "p"}, "labels": {1: "n", 2: "p"}, "masks": {1: "n"}, "iou_scores": {1: "n"}},
    )  # fmt: skip

    vitmatte = ViTMatte().model
    export(
        VitMatteWrap(vitmatte), (torch.randn(1, 4, 640, 768),), out / "vitmatte.fp32.onnx",
        ["pixel_values"], ["alphas"], {"pixel_values": {2: "h", 3: "w"}, "alphas": {2: "h", 3: "w"}},
    )  # fmt: skip


def to_fp16_weights(src: Path, dst: Path, min_elements: int = 1024) -> None:
    """Store large float32 initializers as float16 followed by a Cast back to float32."""
    model = onnx.load(str(src))
    graph = model.graph
    casts = []
    for init in graph.initializer:
        if init.data_type != TensorProto.FLOAT or np.prod(init.dims) < min_elements:
            continue
        arr = numpy_helper.to_array(init).astype(np.float16)
        name = init.name
        init.CopyFrom(numpy_helper.from_array(arr, f"{name}__fp16"))
        casts.append(helper.make_node("Cast", [f"{name}__fp16"], [name], to=TensorProto.FLOAT, name=f"{name}__cast"))
    nodes = list(graph.node)
    del graph.node[:]
    graph.node.extend(casts + nodes)
    onnx.save(model, str(dst))
    print(f"  {dst.name}: {dst.stat().st_size / 1e6:.1f} MB")


def to_int8(src: Path, dst: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    # Conv stays float: dynamic ConvInteger is slow on CPU and hurts the fine edge maps most.
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"])
    print(f"  {dst.name}: {dst.stat().st_size / 1e6:.1f} MB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="onnx_models")
    ap.add_argument("--skip-fp32", action="store_true", help="reuse existing *.fp32.onnx")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not args.skip_fp32:
        export_fp32(out)
    names = ["birefnet_lite", "sam2_encoder", "sam2_decoder", "vitmatte"]
    print("fp16 (weights only):")
    for n in names:
        to_fp16_weights(out / f"{n}.fp32.onnx", out / f"{n}.fp16.onnx")
    print("int8 (dynamic, MatMul/Gemm):")
    for n in names:
        to_int8(out / f"{n}.fp32.onnx", out / f"{n}.int8.onnx")


if __name__ == "__main__":
    main()
