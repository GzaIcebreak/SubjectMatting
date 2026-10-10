#!/usr/bin/env bash
# Puts an ONNX Runtime release (include/ + lib/) in cpp/third_party/onnxruntime for -DONNXRUNTIME_DIR.
#   cpp/scripts/fetch_onnxruntime.sh            official release tarball for this OS/arch (GitHub)
#   cpp/scripts/fetch_onnxruntime.sh --wheel    headers from GitHub + the dylib from the onnxruntime pip wheel in
#                                               .venv (much smaller download when GitHub is slow; macOS only)
set -euo pipefail
VERSION=1.31.0
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/third_party/onnxruntime"
rm -rf "$DEST"
mkdir -p "$DEST/include" "$DEST/lib"

if [ "${1:-}" = "--wheel" ]; then
  RAW="https://raw.githubusercontent.com/microsoft/onnxruntime/v$VERSION/include/onnxruntime/core/session"
  for h in onnxruntime_c_api.h onnxruntime_cxx_api.h onnxruntime_cxx_inline.h onnxruntime_float16.h \
           onnxruntime_ep_c_api.h onnxruntime_error_code.h onnxruntime_session_options_config_keys.h \
           onnxruntime_run_options_config_keys.h onnxruntime_lite_custom_op.h; do
    curl -fsSL --retry 3 -o "$DEST/include/$h" "$RAW/$h"
  done
  LIB="$(ls "$ROOT"/../.venv/lib/python3.*/site-packages/onnxruntime/capi/libonnxruntime.$VERSION.dylib)"
  cp "$LIB" "$DEST/lib/"
  (cd "$DEST/lib" && ln -sf "libonnxruntime.$VERSION.dylib" libonnxruntime.1.dylib && ln -sf libonnxruntime.1.dylib libonnxruntime.dylib)
else
  case "$(uname -s)-$(uname -m)" in
    Darwin-arm64) PKG="onnxruntime-osx-arm64-$VERSION" ;;
    Darwin-x86_64) PKG="onnxruntime-osx-x86_64-$VERSION" ;;
    Linux-x86_64) PKG="onnxruntime-linux-x64-$VERSION" ;;
    Linux-aarch64) PKG="onnxruntime-linux-aarch64-$VERSION" ;;
    *) echo "unsupported platform; download a release from https://github.com/microsoft/onnxruntime/releases" >&2; exit 1 ;;
  esac
  TMP="$(mktemp -d)"
  curl -fsSL --retry 3 -o "$TMP/ort.tgz" "https://github.com/microsoft/onnxruntime/releases/download/v$VERSION/$PKG.tgz"
  tar xzf "$TMP/ort.tgz" -C "$TMP"
  cp -R "$TMP/$PKG/include/." "$DEST/include/"
  cp -a "$TMP/$PKG/lib/." "$DEST/lib/"
  rm -rf "$TMP"
fi
ls -la "$DEST/lib"
