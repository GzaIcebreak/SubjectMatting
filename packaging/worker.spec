# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the matting worker (onedir). scripts/build_macos_app.sh puts it inside SubjectMatting.app.

Build from the repo root:  pyinstaller --noconfirm --clean packaging/worker.spec
Models are NOT bundled; the worker downloads them to ~/.cache/huggingface on first run (~440MB).
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_all

ROOT = Path(SPECPATH).resolve().parent  # noqa: F821 — SPECPATH is injected by PyInstaller

datas, binaries, hiddenimports = [], [], []
# Packages with data files / dynamically imported submodules PyInstaller's static analysis misses:
# transformers resolves model classes by name; timm/kornia/einops are imported at runtime by BiRefNet's remote
# code, which is not visible to the analysis at all.
for pkg in ["transformers", "tokenizers", "huggingface_hub", "safetensors", "timm", "kornia", "einops", "torchvision"]:
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h
hiddenimports += ["matting", "matting.common", "matting.saliency", "matting.pipeline_a", "matting.batch", "cv2"]

a = Analysis(
    [str(ROOT / "worker.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["gradio", "tkinter", "matplotlib", "pytest", "IPython", "jupyter"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="matting-worker",
    console=True,  # talks JSON over stdin/stdout with the SwiftUI app
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="matting-worker")
