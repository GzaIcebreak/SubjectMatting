# 第一阶段报告：Qwen-VL + EfficientSAM3（方案 Q）

日期：2026-10-12 · 分支 `feat/qwen-sam3` · 设计见 [specs/2026-10-12-qwen-sam3-design.md](superpowers/specs/2026-10-12-qwen-sam3-design.md)

本机（Apple Silicon，CPU 推理）。完整对比图：`bench_vlm.py` 生成的 `outputs/bench_vlm/report.html`（不入库）。

## 结论

- **指定模式是方案 A 做不到的新能力，而且基本可用**：10 条指令 9 条符合预期（左/右边的猫、只要狗、只要人、
  购物袋、整辆车、最右边的泰迪熊、击球手；图中没有的"狗"正确报错）。只有部件级的"鹦鹉的嘴"只抠出一半。
- **全自动模式质量接近方案 A**：15 张全部成功。加了与方案 A 相同的 mask 清理后（见下），大部分图与 A 持平；
  剩下的差距：击球手的球棒没抠到，cat_whiskers 带一条笔记本边，two_cats 带进遥控器。
  另有两张是"主体判断"不同而非错误：stop_sign 只要牌不要杆，groceries 只要购物袋不要车尾。
- **速度**：本地部分约 3.4s/张 + Qwen 网络约 2.1s ≈ **5.5s/张**；同一轮里方案 A 6.6s/张。
- **Qwen 用量**：每张图一次调用，平均约 540 输入 token + 50 输出 token。`qwen3-vl-plus` 的框最准
  （试过的 qwen3.6-plus 框偏，qwen3.8-flash 不按格式输出）。
- **体积**：EfficientSAM3 三个 ONNX 合计 385MB（fp32），fp16 约 190MB；加 ViTMatte 52MB 和 ONNX Runtime 33MB，
  第二阶段做成 app 估计约 **280MB**（SAM2 版 232MB）。
- **ONNX 导出**：图像编码器、文本编码器导出后与 PyTorch 误差约 1e-5；**解码器还导不出来**（旧导出器把
  示例框编码里的形状写死，新导出器卡在上游 `decoder.py` 的数据相关分支）。需要在第二阶段改写这部分前向，
  有工作量，但不是死路。另外 C++ 端需要实现 CLIP BPE 分词器（约 200 行 + 1.3MB 词表）。

## 方法

1. Qwen（`qwen3-vl-plus`）看缩到长边 1024 的图，返回 `{"subjects":[{"label","box_2d"}]}`：英文短名词 + 0~1000 坐标框。
2. EfficientSAM3 图像编码一次；同一 label 的所有框作为正示例，一次解码，得到该概念的全部个体。
3. 选个体：mask ≥50% 落在（外扩 5% 的）Qwen 框内；按分数去重（≥50% 被更高分个体覆盖的丢掉）。
4. **mask 清理，与方案 A 的 `fuse_masks` 同一套**：个体概率 >0.5 的部分作"核心"（去掉小于最大块 5% 的碎点），
   概率 >0.3 的部分只保留与核心相连的块，并入后按 Qwen 框外扩 15% 安全裁剪。直接调用 `pipeline_a` 的
   `fuse_masks` / `pad_box` / `largest_components`，不改 SAM2 代码。
5. trimap → ViTMatte → RGBA，与方案 A 完全相同（`make_trimap`、`ViTMatte`、`compose_rgba`）。

### 第 4 步为什么需要

蒸馏版 SAM3 对一些部位"认得出但没把握"：白头鹰的头、鹦鹉的喙概率只有 0.4~0.5，按 0.5 切就丢了
（TinyViT 版甚至丢掉整个鹰头）。固定阈值 0.2 会把背景（粉色毯子）整块带进来；按物体自适应阈值对鹦鹉无效
（身体 0.93、喙 0.4）。最终方案是"0.3 + 只要和核心相连"，再加去重——渗进来的背景大多来自
"物体 + 一点背景"的低分重复个体，去重后就没了。

## 三个 SAM3 变体

EfficientViT / RepViT / TinyViT 三个 `efficientsam3_ft` 权重，加清理后质量接近、速度接近（本地 3.2~3.4s/张）。
清理前 TinyViT 丢鹰头、EfficientViT 丢鹦鹉喙，RepViT 最稳。第二阶段建议用 **EfficientViT**（权重最小，468MB fp32），
如果导出受阻再换 RepViT。

## 原始数据

| | 全自动 15 张 | 指定 10 条 |
|---|---|---|
| 成功 | 15/15 | 9/10 符合预期（1 条正确报错，1 条部件只抠一半） |
| 方案 Q 本地耗时 | 3.4s（SAM3 2.4 + 抠图 1.0） | 3.4s |
| Qwen 耗时 | 约 2.1s | 约 2.0s |
| 方案 A（SAM2） | 6.6s | 不支持 |

## 复现

```bash
uv pip install -p .venv -r requirements-vlm.txt
```
```bash
HF_ENDPOINT=https://hf-mirror.com scripts/fetch_efficientsam3.sh efficientvit repvit tinyvit
```
```bash
.venv/bin/python bench_vlm.py --variants efficientvit repvit tinyvit
```
```bash
PYTHONPATH=. .venv/bin/python scripts/probe_sam3_onnx.py
```

需要 `DASHSCOPE_API_KEY`（环境变量或仓库根目录 `.env`）。
