"""Bounded Q detail checks (no inference by default).

  .venv/bin/python cpp/tests/detail_parity.py --smoke
  .venv/bin/python cpp/tests/detail_parity.py --image IMAGE --reference-dir FROZEN_CASE

--smoke uses tiny ONNX fixtures to check lazy loading, recovery, stride and
composition. --image runs ONE real C++ image; with no --reference-dir it also
runs the Python reference, in a separate process after C++ has exited. Coordinate
real-model runs on memory-constrained hosts. Never imports the 47-case runner.
"""
from __future__ import annotations

import argparse
import ctypes as C
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
U8P = C.POINTER(C.c_uint8)


class Timings(C.Structure):
    _fields_ = [(name, C.c_double) for name in ("sam3", "matting", "total")]


def library(path):
    lib = C.CDLL(str(Path(path).resolve()))
    assert hasattr(lib, "smq_detail"), "Q library must export smq_detail"
    lib.smq_detail.argtypes = [C.c_void_p, U8P, C.c_int, C.c_int, C.c_int,
                              U8P, U8P, C.POINTER(Timings), C.c_void_p, C.c_size_t]
    lib.smq_detail.restype = C.c_int
    lib.smq_create.argtypes = [C.c_char_p, C.c_char_p, C.c_int, C.c_void_p, C.c_size_t]
    lib.smq_create.restype = C.c_void_p
    lib.smq_destroy.argtypes = [C.c_void_p]
    return lib


def invalid_inputs(lib):
    # Every case must be rejected before dereferencing this non-null sentinel.
    pixel = (C.c_uint8 * 4)(11, 22, 33, 44)
    base = [C.c_void_p(1), pixel, 1, 1, 3, pixel, pixel]
    cases = [(0, None), (1, None), (2, 0), (2, -1), (3, 0), (3, -1),
             (4, 2), (4, -1), (5, None), (6, None), (2, 2**30)]
    for index, value in cases:
        args = base.copy()
        args[index] = value
        for error_size in (0, 1, 64):
            err = C.create_string_buffer(b"sentinel", 64)
            timing = Timings(-1, -1, -1)
            rc = lib.smq_detail(*args, C.byref(timing), err if error_size else None, error_size)
            assert rc == 1, (index, value, rc)
            assert list(pixel) == [11, 22, 33, 44], "invalid call wrote output"
            assert timing.total == -1, "invalid call wrote timings"
            if error_size:
                assert err.raw[error_size - 1] == 0
                if error_size > 1:
                    assert err.value, "missing diagnostic"
    print("PASS exported interface and invalid inputs", flush=True)


def create(lib, models):
    err = C.create_string_buffer(2048)
    ctx = lib.smq_create(os.fsencode(models), b"fp16", 4, err, len(err))
    assert ctx, err.value.decode()
    return ctx


def detail(lib, ctx, rgb):
    import numpy as np
    h, w, _ = rgb.shape
    # Non-packed input catches accidental stride assumptions.
    padded = np.full((h, w * 3 + 7), 199, np.uint8)
    padded[:, :w * 3] = rgb.reshape(h, w * 3)
    direct = np.full((h, w, 4), 17, np.uint8)
    refined = direct.copy()
    timings = Timings()
    err = C.create_string_buffer(2048)
    rc = lib.smq_detail(ctx, padded.ctypes.data_as(U8P), w, h, padded.strides[0],
                        direct.ctypes.data_as(U8P), refined.ctypes.data_as(U8P),
                        C.byref(timings), err, len(err))
    return rc, err.value.decode(), direct, refined, timings


def smoke(lib):
    import numpy as np
    import onnx
    from onnx import TensorProto as T, helper as H

    def model(path, channels, output):
        # Real, tiny ORT graphs: constant probability 0.25, dynamic spatial size.
        nodes = [H.make_node("ReduceMean", ["pixel_values"], ["mean"], axes=[1], keepdims=1),
                 H.make_node("Mul", ["mean", "zero"], ["zeros"]),
                 H.make_node("Add", ["zeros", "quarter"], [output])]
        graph = H.make_graph(nodes, "fixture", [H.make_tensor_value_info("pixel_values", T.FLOAT, [1, channels, "h", "w"])],
                             [H.make_tensor_value_info(output, T.FLOAT, [1, 1, "h", "w"])],
                             [H.make_tensor("zero", T.FLOAT, [], [0]), H.make_tensor("quarter", T.FLOAT, [], [.25])])
        m = H.make_model(graph, opset_imports=[H.make_opsetid("", 13)])
        m.ir_version = 9
        onnx.save(m, path)

    with tempfile.TemporaryDirectory(prefix="smq-detail-fixture-") as tmp:
        d = Path(tmp)
        for name in ("sam3_image_encoder", "sam3_text_encoder", "sam3_decoder"):
            model(d / f"{name}.fp16.onnx", 3, "unused")
        model(d / "vitmatte.fp16.onnx", 4, "alphas")
        (d / "clip_bpe_merges.txt").symlink_to(ROOT / "onnx_models_q/clip_bpe_merges.txt")
        ctx = create(lib, d)  # BiRefNet deliberately absent: creation must succeed.
        try:
            rgb = np.full((33, 65, 3), [12, 34, 56], np.uint8)
            for _ in range(2):
                rc, err, direct, refined, _ = detail(lib, ctx, rgb)
                assert rc == 2 and "birefnet_lite.fp16.onnx" in err, (rc, err)
                assert (direct == 17).all() and (refined == 17).all()
            model(d / "birefnet_lite.fp16.onnx", 3, "probability")
            # The SAME context must retry loading; an empty binary mask is valid.
            for h, w in ((33, 65), (17, 2050), (1, 2050)):
                rgb = np.full((h, w, 3), [12, 34, 56], np.uint8)
                rc, err, direct, refined, timing = detail(lib, ctx, rgb)
                assert rc == 0, err
                assert np.array_equal(direct[..., :3], rgb)
                assert np.array_equal(refined[..., :3], rgb)
                assert (direct[..., 3] == 63).all(), "probability must not be sigmoid/quantized before resize"
                assert (refined[..., 3] == 0).all(), "empty foreground must remain background"
                assert timing.sam3 >= 0 and timing.matting >= 0 and timing.total > 0
        finally:
            lib.smq_destroy(ctx)
    print("PASS lazy missing-model recovery, probabilities, padded rows, source RGBA and bounded resize", flush=True)


def worker(args):
    import numpy as np
    from PIL import Image
    from matting.common import compose_rgba, make_trimap, resize_long_side
    rgb = Image.open(args.image).convert("RGB")
    out = Path(args.output)
    if args.worker == "cpp":
        lib = library(args.lib)
        with tempfile.TemporaryDirectory(prefix="smq-detail-models-") as tmp:
            d = Path(tmp)
            for p in Path(args.models).resolve().iterdir():
                if p.is_file() and p.name != "birefnet_lite.fp16.onnx":
                    (d / p.name).symlink_to(p)
            (d / "birefnet_lite.fp16.onnx").symlink_to(Path(args.birefnet).resolve())
            ctx = create(lib, d)
            try:
                rc, err, direct, refined, timing = detail(lib, ctx, np.asarray(rgb))
                assert rc == 0, err
                Image.fromarray(direct).save(out / "cpp_direct.png")
                Image.fromarray(refined).save(out / "cpp_binary.png")
                print(f"C++ seconds: detail={timing.sam3:.3f}, matte={timing.matting:.3f}, total={timing.total:.3f}", flush=True)
            finally:
                lib.smq_destroy(ctx)
    else:
        import cv2
        import onnxruntime as ort
        from matting.pipeline_onnx import imagenet_tensor, OnnxViTMatte
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 1
        work = resize_long_side(rgb)
        sal = ort.InferenceSession(str(args.birefnet), opts, providers=["CPUExecutionProvider"])
        p = sal.run(None, {"pixel_values": imagenet_tensor(work, 1024, 1024)})[0][0, 0]
        p = cv2.resize(p, work.size, interpolation=cv2.INTER_LINEAR)
        del sal
        compose_rgba(rgb, p).save(out / "direct.png")
        mat = object.__new__(OnnxViTMatte)
        mat.sess = ort.InferenceSession(str(Path(args.models) / "vitmatte.fp16.onnx"), opts, providers=["CPUExecutionProvider"])
        compose_rgba(rgb, mat(work, make_trimap(p > .5))).save(out / "binary.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lib", default=str(ROOT / "cpp/build/libsubject_matting_q.dylib"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--image", type=Path)
    ap.add_argument("--models", type=Path, default=ROOT / "onnx_models_q")
    ap.add_argument("--birefnet", type=Path, default=ROOT / "onnx_models/birefnet_lite.fp16.onnx")
    ap.add_argument("--reference-dir", type=Path)
    ap.add_argument("--max-mean", type=float, default=.5)
    ap.add_argument("--min-iou", type=float, default=.99)
    ap.add_argument("--worker", choices=["cpp", "python"], help=argparse.SUPPRESS)
    ap.add_argument("--output", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        worker(args)
        return
    lib = library(args.lib)
    invalid_inputs(lib)
    if args.smoke:
        smoke(lib)
    if args.image:
        import numpy as np
        from PIL import Image
        with tempfile.TemporaryDirectory(prefix="smq-detail-parity-") as tmp:
            command = [sys.executable, str(Path(__file__).resolve()), "--lib", args.lib,
                       "--image", str(args.image), "--models", str(args.models),
                       "--birefnet", str(args.birefnet), "--output", tmp]
            subprocess.run(command + ["--worker", "cpp"], check=True)
            if not args.reference_dir:
                subprocess.run(command + ["--worker", "python"], check=True)
            reference = args.reference_dir or Path(tmp)
            for name in ("direct", "binary"):
                a = np.asarray(Image.open(reference / f"{name}.png"))
                b = np.asarray(Image.open(Path(tmp) / f"cpp_{name}.png"))
                assert a.shape == b.shape and np.array_equal(a[..., :3], b[..., :3])
                delta = np.abs(a[..., 3].astype(float) - b[..., 3])
                ma, mb = a[..., 3] > 127, b[..., 3] > 127
                union = (ma | mb).sum()
                iou = float((ma & mb).sum() / union) if union else 1.0
                print(f"{name}: mean={delta.mean():.6f}/255 max={delta.max():.0f} IoU={iou:.6f}", flush=True)
                assert delta.mean() <= args.max_mean and iou >= args.min_iou


if __name__ == "__main__":
    main()
