"""Subject detection with a Qwen vision model over DashScope's OpenAI-compatible API.

Qwen looks at the image and returns the subjects to cut out, each as a short English noun phrase (what SAM3's
text encoder understands) and a bounding box, plus the regions inside the subjects' outline that must become
transparent (background seen through a ring or between an arm and the body, a fake transparency checkerboard
painted into the picture). Two modes:
  auto         the subject(s) a person would want to cut out
  instruction  whatever the user asked for, e.g. "只要左边那只猫"

QwenJudge is the second opinion: shown the candidate masks the local models proposed, it says which belong to the
subjects (a loose box lets SAM3 propose the chair next to the cat; the colour key proposes stray specks).

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
DEFAULT_MODEL = "qwen3-vl-plus"  # best boxes of the Qwen models tried (qwen3.6-plus misplaced some)
JUDGE_MODEL = "qwen3.6-plus"  # judged candidates as well as qwen3.8-max (the strongest) at a third of the time
UPLOAD_LONG_SIDE = 1024  # enough for boxes; keeps uploads small and cheap

AUTO_PROMPT = """这张图要做抠图：去掉背景，只留下前景主体。请找出前景主体。
规则：
1. 一件物品、模型、摆件、产品，连同它的底座、支架、配件，或放在同一个底座/盘子/托盘上的一组东西，算一个整体：只输出一个主体，框住整体，不要拆成部件。
2. 前景里有几个各自独立的个体时（几个人、几只动物、几个玩偶，即使挨在一起），每个单独输出一个框，最多 4 个。
3. 背景、地面、桌面、墙、天空、房间不算主体。
4. remove：落在主体轮廓里面或凹口里、但不属于主体、抠图时要变透明的区域，例如环形/字母/数字中间的洞或凹口里透出来的背景，手臂和身体之间、两腿之间露出的背景，画在图里的假透明棋盘格。每一块分开的区域单独输出一个框，框要包住那一整块。牙齿、眼白、衣服上的白字、主体身上的花纹都属于主体，不算。只写图里确实看得到的，没有就输出空列表。
只输出 JSON，不要任何解释：
{"subjects": [{"label": "...", "box_2d": [x1, y1, x2, y2]}], "remove": [{"label": "...", "box_2d": [x1, y1, x2, y2]}]}
- label：主体的英文名词，1~3 个词，单数，只写一种东西的名字，不要数量、不要 and/with（三只熊写 "teddy bear"，城堡模型连火车写 "castle model"）。remove 的 label 写那块区域里看到的东西（"checkerboard"、"sky"、"wall"）。
- box_2d：外接框，坐标是相对图片宽高的 0~1000 整数，左上角 (x1, y1)，右下角 (x2, y2)。主体的框要完整包住整个主体，remove 的框只框住那块区域。"""

INSTRUCTION_PROMPT = """用户想从这张图里抠出：「{instruction}」。
找出图中符合这个描述的对象（可能有多个，也可能没有），以及落在这些对象轮廓里面、但不属于对象、抠图时要变透明的区域（remove：环形/字母/数字中间的洞或凹口里透出来的背景，手臂和身体之间露出的背景，画在图里的假透明棋盘格；每一块分开的区域单独一个框，框要包住那一整块；牙齿、眼白、白字、花纹属于对象，不算；只写图里确实看得到的，没有就输出空列表）。只输出 JSON，不要任何解释：
{{"subjects": [{{"label": "...", "box_2d": [x1, y1, x2, y2]}}], "remove": [{{"label": "...", "box_2d": [x1, y1, x2, y2]}}]}}
- label：对象的英文名词短语，1~3 个词，单数，例如 "cat"、"shopping bag"。不要把位置写进 label（"左边的猫" 写成 "cat"，靠 box_2d 区分是哪一只）。
- box_2d：对象的外接框，坐标是相对图片宽高的 0~1000 整数，左上角 (x1, y1)，右下角 (x2, y2)。框要完整包住整个对象。
- remove 的 label 写那块区域里看到的东西（"checkerboard"、"sky"），框只框住那块区域。
- 图中没有符合描述的对象时输出 {{"subjects": [], "remove": []}}。"""

JUDGE_PROMPT = """第一张是原图。要抠出的主体是：{subjects}。
第二张是 {n} 个编号候选区域（每格里亮的部分是候选，暗的部分不是）。
逐个判断：候选亮的部分是否基本都落在上面这些主体身上（只是主体的一部分也算）？如果亮的部分明显包含主体以外的东西（家具、墙、地面、别的物体），就不要。
只输出 JSON：{{"keep": [编号...]}}"""


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
    remove: list[Subject] = field(default_factory=list)  # regions inside the subjects that must become transparent


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


def first_json(text: str):
    """The first JSON object/array in a model reply: ```json fences, text around it and trailing commas are ok."""
    m = re.search(r"[\[{]", text)
    if not m:
        raise QwenError(f"模型没有返回 JSON：{text[:200]}")
    tail = text[m.start() :]
    # Seen in replies: trailing commas, and a box written without its brackets ("box_2d": 37, 421, 251, 969).
    fixed = re.sub(r",(\s*[\]}])", r"\1", tail)
    fixed = re.sub(r'("b(?:b)?ox(?:_2d)?"\s*:\s*)(-?[\d.]+\s*,\s*-?[\d.]+\s*,\s*-?[\d.]+\s*,\s*-?[\d.]+)', r"\1[\2]", fixed)
    for candidate in (tail, fixed):
        try:
            return json.JSONDecoder().raw_decode(candidate)[0]
        except json.JSONDecodeError as e:
            error = e
    raise QwenError(f"模型返回的 JSON 无法解析：{error}")


def parse_detection(text: str, width: int, height: int) -> tuple[list[Subject], list[Subject]]:
    """The model's reply -> (subjects, remove regions) with pixel boxes. Accepts {"subjects": [...], "remove": [...]}
    as asked, or the bare [{"bbox_2d": ..., "label": ...}] list of subjects some Qwen models fall back to."""
    data = first_json(text)
    if isinstance(data, list):
        return boxes_to_pixels(data, width, height), []
    if isinstance(data, dict):
        return boxes_to_pixels(data.get("subjects", []), width, height), boxes_to_pixels(data.get("remove", []), width, height)
    return [], []


def parse_subjects(text: str, width: int, height: int) -> list[Subject]:
    return parse_detection(text, width, height)[0]


def boxes_to_pixels(items, width: int, height: int) -> list[Subject]:
    """[{"label", "box_2d" in 0~1000}] -> Subjects in pixels; entries without a label or a usable box are dropped."""
    if not isinstance(items, list):
        return []
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
        subjects, remove = parse_detection(text, *img.size)
        return Detection(subjects, self.model, time.perf_counter() - t, usage, text, remove)

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


def parse_keep(text: str, n: int) -> list[int]:
    """The judge's reply -> 0-based candidate indices. Takes the last {"keep": [...]} (some models think aloud first)."""
    found = re.findall(r'"keep"\s*:\s*\[([^\]]*)\]', text)
    if not found:
        raise QwenError(f"判断模型没有返回 keep：{text[-200:]}")
    return sorted({int(v) - 1 for v in re.findall(r"\d+", found[-1]) if 1 <= int(v) <= n})


def candidate_sheet(img: Image.Image, masks: list, tile_long_side: int = 384) -> Image.Image:
    """One tile per candidate: the image, dimmed outside the candidate, numbered 1..n in the corner; 3 per row."""
    import numpy as np
    from PIL import ImageDraw, ImageFont

    w, h = img.size
    s = tile_long_side / max(w, h)
    tw, th = max(1, round(w * s)), max(1, round(h * s))
    small = np.asarray(img.convert("RGB").resize((tw, th), Image.BILINEAR), np.float32)
    dim = small * 0.15 + np.array([0, 0, 12], np.float32)
    cols = min(len(masks), 3)
    rows = (len(masks) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * th))
    font = ImageFont.load_default(size=max(16, th // 10))
    for k, m in enumerate(masks):
        ms = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((tw, th), Image.BILINEAR)) > 127
        tile = Image.fromarray(np.where(ms[..., None], small, dim).astype(np.uint8))
        ImageDraw.Draw(tile).text((6, 2), str(k + 1), fill=(255, 40, 40), font=font, stroke_width=3, stroke_fill=(255, 255, 255))
        sheet.paste(tile, ((k % cols) * tw, (k // cols) * th))
    return sheet


class QwenJudge:
    """Asks a Qwen model which candidate masks belong to the subjects."""

    def __init__(self, model: str = JUDGE_MODEL, detector: QwenDetector | None = None):
        self.model = model
        self.detector = detector or QwenDetector()

    def judge(self, img: Image.Image, subjects: list[Subject], masks: list, instruction: str | None = None) -> list[int]:
        names = "、".join(f"{s.label}（框 {[round(v) for v in s.box]}）" for s in subjects)
        if instruction:
            names += f"（用户要求：{instruction}）"
        body = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": encode_image(img)}},
                        {"type": "image_url", "image_url": {"url": encode_image(candidate_sheet(img, masks), long_side=1152)}},
                        {"type": "text", "text": JUDGE_PROMPT.format(subjects=names, n=len(masks))},
                    ],
                }
            ],
            "temperature": 0,
            "enable_thinking": False,
        }
        text, _ = self.detector._post(body)
        return parse_keep(text, len(masks))
