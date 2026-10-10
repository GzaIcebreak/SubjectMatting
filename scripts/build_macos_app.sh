#!/usr/bin/env bash
# Builds dist/SubjectMatting.app: the SwiftUI front end (macos/) with the PyInstaller worker inside, ad-hoc signed.
# Run from anywhere after building the worker:  pyinstaller --noconfirm --clean packaging/worker.spec
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKER="$ROOT/dist/matting-worker"
APP="$ROOT/dist/SubjectMatting.app"

if [ ! -x "$WORKER/matting-worker" ]; then
  echo "missing $WORKER/matting-worker: run 'pyinstaller --noconfirm --clean packaging/worker.spec' first" >&2
  exit 1
fi

swift build -c release --package-path "$ROOT/macos"
BIN="$(swift build -c release --package-path "$ROOT/macos" --show-bin-path)/SubjectMatting"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$ROOT/packaging/Info.plist" "$APP/Contents/Info.plist"
cp "$BIN" "$APP/Contents/MacOS/SubjectMatting"
cp "$ROOT/packaging/AppIcon.icns" "$APP/Contents/Resources/AppIcon.icns"
cp -R "$WORKER" "$APP/Contents/Resources/worker"

# No Apple Developer ID: ad-hoc signature. PyInstaller has already ad-hoc signed the worker's binaries.
codesign --force --deep --sign - "$APP"
codesign --verify --deep --strict --verbose=2 "$APP"
du -sh "$APP"
