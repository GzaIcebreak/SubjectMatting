#!/usr/bin/env bash
# Builds "dist/抠图demo.app" (SubjectMatting Q): the SwiftUI front end and matting-worker-q (macos/), libsubject_matting_q with
# ONNX Runtime (cpp/), the EfficientSAM3 + ViTMatte fp16 models and the CLIP BPE merges, ad-hoc signed.
# Prerequisites:
#   PYTHONPATH=.:third_party/efficientsam3/sam3 .venv/bin/python scripts/export_sam3_onnx.py  -> onnx_models_q/
#   cmake -S cpp -B cpp/build -DONNXRUNTIME_DIR=... && cmake --build cpp/build
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="$ROOT/dist/抠图demo.app"
CPP_BUILD="$ROOT/cpp/build"
MODELS="$ROOT/onnx_models_q"

for f in sam3_image_encoder.fp16.onnx sam3_text_encoder.fp16.onnx sam3_decoder.fp16.onnx vitmatte.fp16.onnx clip_bpe_merges.txt; do
  [ -f "$MODELS/$f" ] || { echo "missing $MODELS/$f: run scripts/export_sam3_onnx.py first" >&2; exit 1; }
done
[ -f "$CPP_BUILD/build.ninja" ] || [ -f "$CPP_BUILD/Makefile" ] || { echo "missing $CPP_BUILD: build cpp/ first" >&2; exit 1; }

[ -f "$ROOT/onnx_models/birefnet_lite.fp16.onnx" ] || { echo "missing BiRefNet detail weights" >&2; exit 1; }

STAGE="$(mktemp -d)"
cmake --install "$CPP_BUILD" --prefix "$STAGE" >/dev/null

swift build -c release --package-path "$ROOT/macos" --product SubjectMattingQ
swift build -c release --package-path "$ROOT/macos" --product matting-worker-q
BIN="$(swift build -c release --package-path "$ROOT/macos" --show-bin-path)"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Frameworks" "$APP/Contents/Resources/models"
cp "$ROOT/packaging/InfoQ.plist" "$APP/Contents/Info.plist"
cp "$ROOT/packaging/AppIconQ.icns" "$APP/Contents/Resources/AppIconQ.icns"
cp "$BIN/SubjectMattingQ" "$BIN/matting-worker-q" "$APP/Contents/MacOS/"
cp -L "$STAGE/lib/libsubject_matting_q.0.dylib" "$APP/Contents/Frameworks/"
cp -L "$STAGE/lib/libonnxruntime.1.dylib" "$APP/Contents/Frameworks/"
for f in sam3_image_encoder.fp16.onnx sam3_text_encoder.fp16.onnx sam3_decoder.fp16.onnx vitmatte.fp16.onnx clip_bpe_merges.txt; do
  cp "$MODELS/$f" "$APP/Contents/Resources/models/"
done
cp "$ROOT/onnx_models/birefnet_lite.fp16.onnx" "$APP/Contents/Resources/models/"
rm -rf "$STAGE"

codesign --force --sign - "$APP"/Contents/Frameworks/*.dylib "$APP/Contents/MacOS/matting-worker-q"
codesign --force --sign - "$APP"
codesign --verify --deep --strict --verbose=2 "$APP"
du -sh "$APP"
