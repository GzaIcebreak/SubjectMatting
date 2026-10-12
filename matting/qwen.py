"""Subject detection with a Qwen vision model over DashScope's OpenAI-compatible API.

Qwen looks at the image and returns the subjects to cut out, each as a short English noun phrase (what SAM3's
text encoder understands) and a bounding box. Two modes:
  auto         the subject(s) a person would want to cut out
  instruction  whatever the user asked for, e.g. "只要左边那只猫"

The API key comes from $DASHSCOPE_API_KEY or a .env file in the repo root (never from the code).
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3-vl-plus"
UPLOAD_LONG_SIDE = 1024  # enough for boxes; keeps uploads small and cheap

AUTO_PROMPT = """找出这张图里，一般人想抠出来的前景主体。可以有多个（比如人和他牵着的狗），但背景、地面、桌面、墙、天空不算主体。
只输出 JSON，不要任何解释：
{"subjects": [{"label": "...", "box_2d": [x1, y1, x2, y2]}]}
- label：主体的英文名词短语，1~3 个词，单数，例如 "cat"、"red car"、"teddy bear"。
- box_2d：主体的外接框，坐标是相对图片宽高的 0~1000 整数，左上角 (x1, y1)，右下角 (x2, y2)。框要完整包住整个主体。"""

INSTRUCTION_PROMPT = """用户想从这张图里抠出：「{instruction}」。
找出图中符合这个描述的对象（可能有多个，也可能没有）。只输出 JSON，不要任何解释：
{{"subjects": [{{"label": "...", "box_2d": [x1, y1, x2, y2]}}]}}
- label：对象的英文名词短语，1~3 个词，单数，例如 "cat"、"shopping bag"。不要把位置写进 label（"左边的猫" 写成 "cat"，靠 box_2d 区分是哪一只）。
- box_2d：对象的外接框，坐标是相对图片宽高的 0~1000 整数，左上角 (x1, y1)，右下角 (x2, y2)。框要完整包住整个对象。
- 图中没有符合描述的对象时输出 {{"subjects": []}}。"""


@dataclass
class Subject:
    label: str
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels of the original image


@dataclass
class Detection:
    subjects: list[Subject]
    model: str
    seconds: float
    usage: dict = field(default_factory=dict)  # token counts as reported by the API
    raw: str = ""


class QwenError(RuntimeError):
    pass


def load_api_key() -> str:
    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        env = Path(__file__).resolve().parent.parent / ".env"
        if env.exists():
            for line in env.read_text(encoding="utf-8").splitlines():
                name, _, value = line.partition("=")
                if name.strip() == "DASHSCOPE_API_KEY":
                    key = value.strip().strip('"').strip("'")
    if not key:
        raise QwenError("没有找到 DASHSCOPE_API_KEY（环境变量或仓库根目录的 .env）")
    return key


def encode_image(img: Image.Image, long_side: int = UPLOAD_LONG_SIDE) -> str:
    img = img.convert("RGB")
    scale = long_side / max(img.size)
    if scale < 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def parse_subjects(text: str, width: int, height: int) -> list[Subject]:
    """The model's reply -> subjects with pixel boxes. Accepts {"subjects": [...]} as asked, or the bare
    [{"bbox_2d": ..., "label": ...}] list some Qwen models fall back to; ```json fences and surrounding text are ok."""
    m = re.search(r"[\[{]", text)
    if not m:
        raise QwenError(f"模型没有返回 JSON：{text[:200]}")
    try:
        data, _ = json.JSONDecoder().raw_decode(text[m.start() :])
    except json.JSONDecodeError as e:
        raise QwenError(f"模型返回的 JSON 无法解析：{e}") from e
    items = data if isinstance(data, list) else data.get("subjects", []) if isinstance(data, dict) else []
    subjects = []
    for s in items:
        if not isinstance(s, dict):
            continue
        label = str(s.get("label", "")).strip().lower()
        box = s.get("box_2d") or s.get("bbox_2d") or s.get("box")
        if not label or not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        x1, y1, x2, y2 = (float(v) for v in box)
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        clamp = lambda v: min(max(v, 0.0), 1000.0)  # noqa: E731
        px = (clamp(x1) / 1000 * width, clamp(y1) / 1000 * height, clamp(x2) / 1000 * width, clamp(y2) / 1000 * height)
        if px[2] - px[0] >= 2 and px[3] - px[1] >= 2:
            subjects.append(Subject(label, px))
    return subjects


class QwenDetector:
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None, base_url: str = BASE_URL, timeout: float = 60):
        self.model = model
        self.api_key = api_key or load_api_key()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def detect(self, img: Image.Image, instruction: str | None = None) -> Detection:
        prompt = INSTRUCTION_PROMPT.format(instruction=instruction) if instruction else AUTO_PROMPT
        body = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": encode_image(img)}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "temperature": 0,
            # Hybrid-thinking models would otherwise spend seconds reasoning about a bounding box.
            "enable_thinking": False,
        }
        t = time.perf_counter()
        text, usage = self._post(body)
        return Detection(parse_subjects(text, *img.size), self.model, time.perf_counter() - t, usage, text)

    def _post(self, body: dict) -> tuple[str, dict]:
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.load(r)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            raise QwenError(f"Qwen 接口返回 {e.code}：{detail}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            raise QwenError(f"无法连接 Qwen 接口：{e}") from e
        try:
            return data["choices"][0]["message"]["content"], data.get("usage", {})
        except (KeyError, IndexError) as e:
            raise QwenError(f"Qwen 接口返回格式异常：{str(data)[:300]}") from e
