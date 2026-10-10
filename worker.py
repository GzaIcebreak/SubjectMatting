"""Matting worker for the native macOS app: one JSON object per line on stdin (commands) and stdout (events).

Commands:  {"id": 1, "cmd": "plan", "inputs": ["/a.jpg", "/photos"], "output_dir": null}
           {"id": 2, "cmd": "run", "src": "/a.jpg", "dst": "/a_matting/a.png"}
Events:    download {done, total} | loading | ready | fatal {message}
           plan {id, jobs: [{src, dst}], skipped: [{path, reason}], error}
           result {id, dst, total, stages} | error {id, message}
`plan` is answered as soon as it arrives, even while models are still downloading; `run` is processed in order
once the models are loaded. The worker exits when stdin closes.

Env vars:
  SUBJECTMATTING_SELFTEST=1        import every dependency, print versions, exit 0 (no model download). Used by CI.
  SUBJECTMATTING_FAKE_PIPELINE=1   skip the models; `run` writes the input with an opaque alpha. Used by tests.
  SUBJECTMATTING_LOG               log file (default ~/Library/Logs/SubjectMatting.log).
"""

from __future__ import annotations

import multiprocessing
import os
import sys

if __name__ == "__main__":
    # Must run before anything else: in the PyInstaller bundle multiprocessing helpers re-execute this binary,
    # and without this each helper would run the whole worker again.
    multiprocessing.freeze_support()

import json  # noqa: E402
import queue  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

# Every file the pipeline loads. Listing them lets a fully cached install start offline without network checks.
MODEL_FILES = {
    "ZhengPeng7/BiRefNet_lite": ["config.json", "BiRefNet_config.py", "birefnet.py", "model.safetensors"],
    "facebook/sam2.1-hiera-tiny": [
        "config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "model.safetensors",
    ],
    "hustvl/vitmatte-small-composition-1k": ["config.json", "preprocessor_config.json", "model.safetensors"],
}


class Protocol:
    """Writes events to the original stdout. Must be created before redirect_output_to_log()."""

    def __init__(self):
        self._out = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()

    def send(self, type_: str, **fields) -> None:
        line = json.dumps({"type": type_, **fields}, ensure_ascii=False)
        with self._lock:
            self._out.write(line + "\n")
            self._out.flush()


def redirect_output_to_log() -> None:
    """Point fds 1 and 2 at the log file so library prints, progress bars and C-level output can't corrupt the
    protocol stream."""
    log = Path(os.environ.get("SUBJECTMATTING_LOG") or Path.home() / "Library/Logs/SubjectMatting.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    f = open(log, "a", buffering=1, encoding="utf-8")
    os.dup2(f.fileno(), 1)
    os.dup2(f.fileno(), 2)
    sys.stdout = sys.stderr = f
    print(f"--- worker start {time.strftime('%Y-%m-%d %H:%M:%S')} ---")


def selftest() -> None:
    """Import everything the pipeline needs at runtime (incl. what BiRefNet's remote code imports)."""
    import cv2  # noqa: F401
    import einops  # noqa: F401
    import huggingface_hub
    import kornia  # noqa: F401
    import timm  # noqa: F401
    import torch
    import torchvision  # noqa: F401
    import transformers
    from transformers import AutoModelForImageSegmentation, Sam2Model, VitMatteForImageMatting  # noqa: F401

    import matting.pipeline_a  # noqa: F401

    print(
        "selftest ok",
        f"python={sys.version.split()[0]} torch={torch.__version__} transformers={transformers.__version__}",
        f"huggingface_hub={huggingface_hub.__version__} frozen={bool(getattr(sys, 'frozen', False))}",
    )


def missing_model_files() -> list[tuple[str, str]]:
    from huggingface_hub import try_to_load_from_cache

    return [(r, f) for r, files in MODEL_FILES.items() for f in files if not isinstance(try_to_load_from_cache(r, f), str)]


def download_models(proto: Protocol, missing: list[tuple[str, str]]) -> None:
    """Download the missing files, reporting overall bytes at most a few times per second."""
    from huggingface_hub import HfApi, hf_hub_download
    from tqdm.auto import tqdm

    sizes: dict[tuple[str, str], int] = {}
    api = HfApi()
    for repo in {r for r, _ in missing}:
        for info in api.get_paths_info(repo, [f for r, f in missing if r == repo]):
            sizes[(repo, info.path)] = info.size or 0
    total = sum(sizes.values())
    progress = dict.fromkeys(missing, 0)
    last = [0.0]

    def report(force: bool = False) -> None:
        if force or time.monotonic() - last[0] > 0.25:
            last[0] = time.monotonic()
            proto.send("download", done=min(sum(progress.values()), total), total=total)

    for key in missing:

        class Bar(tqdm):
            def update(self, n=1):
                r = super().update(n)
                progress[key] = self.n
                report()
                return r

        report(force=True)
        hf_hub_download(key[0], key[1], tqdm_class=Bar)
        progress[key] = sizes.get(key, 0)
    report(force=True)


class FakePipeline:
    def run(self, img):
        return img.convert("RGBA"), {"total": 0.0, "stages": {}}


def load_pipeline(proto: Protocol):
    if os.environ.get("SUBJECTMATTING_FAKE_PIPELINE"):
        return FakePipeline()
    missing = missing_model_files()
    if missing:
        try:
            download_models(proto, missing)
        except Exception as e:
            raise RuntimeError(f"模型下载失败，请检查网络后重试（{type(e).__name__}: {e}）") from e
    # Everything is cached now: skip the per-file "is there a newer revision" requests.
    os.environ["HF_HUB_OFFLINE"] = "1"
    proto.send("loading")
    from matting.pipeline_a import PipelineA

    return PipelineA()


def handle_plan(proto: Protocol, msg: dict) -> None:
    from matting.batch import plan_jobs

    try:
        jobs, skipped = plan_jobs(msg.get("inputs") or [], msg.get("output_dir"))
        proto.send(
            "plan",
            id=msg.get("id"),
            jobs=[{"src": str(j.src), "dst": str(j.dst)} for j in jobs],
            skipped=[{"path": str(p), "reason": r} for p, r in skipped],
            error=None,
        )
    except Exception as e:  # noqa: BLE001 — report instead of killing the worker
        proto.send("plan", id=msg.get("id"), jobs=[], skipped=[], error=f"{type(e).__name__}: {e}")


def handle_run(proto: Protocol, pipe, msg: dict) -> None:
    from matting.common import load_rgb

    try:
        dst = Path(msg["dst"])
        rgba, info = pipe.run(load_rgb(msg["src"]))
        dst.parent.mkdir(parents=True, exist_ok=True)
        rgba.save(dst)
        proto.send("result", id=msg.get("id"), dst=str(dst), total=info["total"], stages=info["stages"])
    except Exception as e:  # noqa: BLE001 — one bad image must not stop the worker
        proto.send("error", id=msg.get("id"), message=f"{type(e).__name__}: {e}")


def read_commands(proto: Protocol, runs: queue.Queue) -> None:
    """stdin reader thread: answers `plan` directly, queues `run` for the main thread, None on EOF."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            print(f"bad command: {line[:200]}")
            continue
        if msg.get("cmd") == "plan":
            handle_plan(proto, msg)
        elif msg.get("cmd") == "run":
            runs.put(msg)
        else:
            print(f"unknown command: {msg.get('cmd')}")
    runs.put(None)


def main() -> None:
    if os.environ.get("SUBJECTMATTING_SELFTEST"):
        selftest()
        return
    proto = Protocol()
    redirect_output_to_log()
    runs: queue.Queue = queue.Queue()
    threading.Thread(target=read_commands, args=(proto, runs), daemon=True).start()
    try:
        pipe = load_pipeline(proto)
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        proto.send("fatal", message=str(e) if isinstance(e, RuntimeError) else f"{type(e).__name__}: {e}")
        sys.exit(1)
    proto.send("ready")
    while (msg := runs.get()) is not None:
        handle_run(proto, pipe, msg)


if __name__ == "__main__":
    main()
