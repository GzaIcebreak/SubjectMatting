# Qwen-VL + EfficientSAM3 抠图（分支 `feat/qwen-sam3`）

日期：2026-10-12

## 目标

用视觉语言模型识别主体，再交给 SAM3 分割、ViTMatte 抠图：

- **全自动**：Qwen 判断"一般人想抠出来的主体"（可以多个）。
- **指定**：用户用一句话说要抠什么（"只要左边那只猫""购物袋"），Qwen 理解后给出对象。

保持轻量：大模型走云端 API，本地只多一个蒸馏版 SAM3。

## 边界

- `main` 和已发布的 v0.2.0（SAM2 版）不动；所有工作在 `feat/qwen-sam3`。
- 分支里现有 SAM2 代码（`matting/pipeline_a.py`、`worker.py`、`cpp/`、`macos/`）不修改、不删除。
  新东西只放新文件；trimap / ViTMatte / 合成等共用函数只调用不改。

## 决定

| 问题 | 结论 |
|---|---|
| 解决什么 | 全自动 + 用户指定，两者都要 |
| 大模型在哪跑 | 云端 API（不做本地模型、不做离线回退） |
| 识别结果怎么交给分割 | 方案 3：Qwen 给"英文短名词 + 框"，EfficientSAM3 用词 + 框（框作示例）分割，再按框挑个体 |
| 用哪家 | 只接 Qwen（阿里云百炼 OpenAI 兼容接口），默认 `qwen3-vl-plus` |
| 做到哪一步 | 分阶段：①Python 原型 + 评测 → 用户看报告决定 → ②ONNX 导出 + C++ → ③app 集成 |

## 流程

```
图片 ─长边≤1024─▶ Qwen-VL：{"subjects":[{"label":"cat","box_2d":[x1,y1,x2,y2]}]}（0~1000 相对坐标）
                          │ label：英文 1~3 词（SAM3 文本编码器只认英文）；指定模式下位置信息只体现在框里
                          ▼
   EfficientSAM3：图像编码一次；同一 label 的所有框一起作为正示例，一次解码 → 该概念的全部个体
                          ▼
   挑个体：mask 至少 50% 落在（四边各外扩 5% 的）Qwen 框内的保留；某个框一个都没挑到时，
          退回到与它框 IoU 最大的个体（IoU ≥ 0.3）。全部保留个体取并集。
                          ▼
   trimap → ViTMatte → RGBA（沿用 matting/common.py）
```

SAM3 的框提示是"找和框里相似的东西"，不是裁剪：所以由 SAM3 决定物体边界（框画小了不会截断物体），
由 Qwen 的框决定要哪一个（"右边那只猫"）。

## 文件

| 文件 | 作用 |
|---|---|
| `matting/qwen.py` | Qwen 客户端：缩图上传、提示词（全自动 / 指定）、容错解析（`{"subjects":…}` 或裸列表 `bbox_2d`）、0~1000 → 像素 |
| `matting/sam3.py` | EfficientSAM3 封装（CPU）：`set_image` 一次，`find(label, boxes)` 一次解码 |
| `matting/pipeline_vlm.py` | 方案 Q：Qwen → SAM3 → 挑个体 → ViTMatte；`select_instances` 等纯函数 |
| `bench_vlm.py` | 评测：15 张图全自动（与方案 A 并排）+ 10 条指定指令（含一条应报错的），Qwen 结果缓存 |
| `scripts/fetch_efficientsam3.sh` | 下载上游代码（固定 commit）和权重到 `third_party/`（不入库） |
| `requirements-vlm.txt` | 上游代码额外需要的包 |
| `tests/test_vlm.py` | 解析和挑个体的单元测试（不调 API、不加载模型） |

API Key：环境变量 `DASHSCOPE_API_KEY` 或仓库根目录 `.env`（已在 `.gitignore`），不进代码。

## 出错处理

- 网络 / 接口错误、超时（60s）、返回不是 JSON：这张图报错（`QwenError`），不静默回退到 SAM2。
- Qwen 没找到对象：全自动报"没有识别到主体"，指定模式报"图中没找到：<指令>"。
- SAM3 在 Qwen 的位置没分出东西：报错，不输出空图。

## 第一阶段交付（本文档的范围）

1. 原型跑通（上面的文件）。
2. 评测报告：质量对比方案 A、各阶段耗时、Qwen token 用量；三个 SAM3 变体（EfficientViT / RepViT / TinyViT）选一个。
3. ONNX 可导出性验证：图像编码器、文本编码器、解码器能否导出并与 PyTorch 对上。

用户看过报告后再决定是否进入第二、三阶段（C++ 核心与 app 集成，单独设计）。

## 隐私与费用

全自动和指定模式都要把图片（缩到长边 1024 的 JPEG）上传到阿里云。到 app 阶段需在设置里明确告知。
费用按 token 计，评测报告里给出每张图的实际 token 数。
