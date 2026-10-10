# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec: onedir bundle, wrapped in SubjectMatting.app on macOS.

Build from the repo root:  pyinstaller --noconfirm --clean packaging/SubjectMatting.spec
Models are NOT bundled; they are downloaded to ~/.cache/huggingface on first run (~440MB).
"""

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

ROOT = Path(SPECPATH).resolve().parent  # noqa: F821 — SPECPATH is injected by PyInstaller

datas, binaries, hiddenimports = [], [], []
# Packages with data files / dynamically imported submodules PyInstaller's static analysis misses:
# gradio ships its frontend as data; transformers resolves model classes by name; timm/kornia/einops are
# imported at runtime by BiRefNet's remote code, which is not visible to the analysis at all.
for pkg in [
    "gradio", "gradio_client", "safehttpx", "groovy",
    "transformers", "tokenizers", "huggingface_hub", "safetensors",
    "timm", "kornia", "einops", "torchvision",
]:  # fmt: skip
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h
hiddenimports += ["matting", "matting.common", "matting.saliency", "matting.pipeline_a", "matting.batch", "cv2"]

a = Analysis(
    [str(ROOT / "app.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    # gradio reads its own source at runtime (inspect); keep it as .py instead of compiling into the archive.
    module_collection_mode={"gradio": "py"},
    excludes=["tkinter", "matplotlib", "pytest", "IPython", "jupyter"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SubjectMatting",
    console=sys.platform != "darwin",  # .app on macOS; a console exe elsewhere (used for local dry runs)
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="SubjectMatting")

# BUNDLE is a no-op on non-macOS platforms.
app = BUNDLE(
    coll,
    name="SubjectMatting.app",
    icon=None,
    bundle_identifier="com.gzaicebreak.subjectmatting",
    info_plist={
        "CFBundleName": "SubjectMatting",
        "CFBundleDisplayName": "SubjectMatting",
        "CFBundleShortVersionString": "0.1.0",
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": "14.0",
    },
)
