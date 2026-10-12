#!/usr/bin/env bash
# Fetches the EfficientSAM3 upstream code (pinned) and fine-tuned weights into third_party/ (not committed).
#   scripts/fetch_efficientsam3.sh [efficientvit|repvit|tinyvit ...]   default: efficientvit
# HF_ENDPOINT=https://hf-mirror.com speeds up the weight download in mainland China.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMMIT=bd0936c788fed8d51fa799437f05abd97b401b06
DEST="$ROOT/third_party/efficientsam3"
if [ ! -d "$DEST/.git" ]; then
  git clone -q https://github.com/SimonZeng7108/efficientsam3 "$DEST"
fi
git -C "$DEST" fetch -q --depth 1 origin "$COMMIT" 2>/dev/null || true
git -C "$DEST" checkout -q "$COMMIT"

HF="${HF_ENDPOINT:-https://huggingface.co}"
mkdir -p "$ROOT/third_party/weights"
for v in "${@:-efficientvit}"; do
  f="efficientsam3_$v.pt"
  curl -fL --retry 5 -C - -o "$ROOT/third_party/weights/$f" "$HF/Simon7108528/EfficientSAM3/resolve/main/efficientsam3_ft/$f"
done
ls -la "$ROOT/third_party/weights"
