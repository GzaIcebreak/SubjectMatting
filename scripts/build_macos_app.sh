#!/usr/bin/env bash
# Builds dist/SubjectMatting.app: the SwiftUI front end and matting-worker (macos/), the C++ matting library with
# ONNX Runtime (cpp/), and the fp16 models, ad-hoc signed. Prerequisites (see README):
#   PYTHONPATH=. .venv/bin/python scripts/export_onnx.py             -> onnx_models/*.fp16.onnx
#   cmake -S cpp -B cpp/build -DONNXRUNTIME_DIR=... && cmake --build cpp/build
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="$ROOT/dist/SubjectMatting.app"
CPP_BUILD="$ROOT/cpp/build"

for f in "$ROOT"/onnx_models/{birefnet_lite,sam2_encoder,sam2_decoder,vitmatte}.fp16.onnx; do
  [ -f "$f" ] || { echo "missing $f: run scripts/export_onnx.py first" >&2; exit 1; }
done
[ -f "$CPP_BUILD/build.ninja" ] || [ -f "$CPP_BUILD/Makefile" ] || { echo "missing $CPP_BUILD: build cpp/ first" >&2; exit 1; }

# The C++ library with install rpaths (@loader_path), staged so its build-tree rpaths don't leak into the app.
STAGE="$(mktemp -d)"
cmake --install "$CPP_BUILD" --prefix "$STAGE" >/dev/null

swift build -c release --package-path "$ROOT/macos"
BIN="$(swift build -c release --package-path "$ROOT/macos" --show-bin-path)"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Frameworks" "$APP/Contents/Resources/models"
cp "$ROOT/packaging/Info.plist" "$APP/Contents/Info.plist"
cp "$ROOT/packaging/AppIcon.icns" "$APP/Contents/Resources/AppIcon.icns"
cp "$BIN/SubjectMatting" "$BIN/matting-worker" "$APP/Contents/MacOS/"
# Copy the real files under the names the loader asks for (@rpath/libsubject_matting.0.dylib, libonnxruntime.1.dylib).
cp -L "$STAGE/lib/libsubject_matting.0.dylib" "$APP/Contents/Frameworks/"
cp -L "$STAGE/lib/libonnxruntime.1.dylib" "$APP/Contents/Frameworks/"
cp "$ROOT"/onnx_models/*.fp16.onnx "$APP/Contents/Resources/models/"
rm -rf "$STAGE"

# No Apple Developer ID: ad-hoc signatures, inside out.
codesign --force --sign - "$APP"/Contents/Frameworks/*.dylib "$APP/Contents/MacOS/matting-worker"
codesign --force --sign - "$APP"
codesign --verify --deep --strict --verbose=2 "$APP"
du -sh "$APP"
