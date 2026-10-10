#!/usr/bin/env bash
# Packs dist/SubjectMatting-core-<os>-<arch>.zip: the C header, libsubject_matting + libonnxruntime, the fp16
# models, sm_cli and the README. Run after building cpp/build and exporting onnx_models/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD="$ROOT/cpp/build"
NAME="SubjectMatting-core-$(uname -s | tr 'A-Z' 'a-z')-$(uname -m)"
OUT="$ROOT/dist/$NAME"
rm -rf "$OUT" "$OUT.zip"
# cmake --install rewrites the rpaths to relative ones (lib: @loader_path, sm_cli: ../lib).
cmake --install "$BUILD" --prefix "$OUT" >/dev/null
mkdir -p "$OUT/models"
cp "$ROOT"/onnx_models/*.fp16.onnx "$OUT/models/"
cp "$ROOT/cpp/README.md" "$OUT/"
if [ "$(uname -s)" = Darwin ]; then
  # No Developer ID: ad-hoc signatures (install_name_tool invalidated the linker's).
  codesign --force --sign - "$OUT"/lib/*.dylib "$OUT/bin/sm_cli"
fi
(cd "$ROOT/dist" && ditto -c -k --keepParent "$NAME" "$NAME.zip" 2>/dev/null || zip -qry "$NAME.zip" "$NAME")
du -sh "$OUT" "$OUT.zip"
