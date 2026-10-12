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

## 第二、三阶段：C++ 核心与 SubjectMatting Q.app（2026-10-12）

独立的新 app（名字 SubjectMatting Q，Bundle ID `com.gzaicebreak.subjectmatting.q`），可以和 SAM2 版同时安装。
SAM2 版的界面、worker、C++ 流水线和 Python 代码都没改；共用的 `MattingCore` 只做了向后兼容的扩展
（任务可带"指令"、worker 可指定可执行文件名和额外环境变量），SAM2 版行为和测试不变。

**ONNX 导出**（`scripts/export_sam3_onnx.py`）：图像编码器、文本编码器、解码器三个模型。解码器导出的关键是把上游
`concat_padded_sequences`（数据相关的索引）在导出时换成普通拼接——我们的提示没有填充，结果完全一致；
解码器一次只接受一个示例框，所以流水线改成"每个主体解码一次"（PyTorch 版同步改了）。
fp16 与 PyTorch 在 25 个评测用例上 IoU ≥ 0.9993。

**C++ 核心**（`libsubject_matting_q`，头文件 `subject_matting_q.h`，与 SAM2 的库分开）：调用方传入 Qwen 给的
主体（英文词 + 框），库内分割、挑个体、清理、trimap、ViTMatte，不联网。CLIP 分词器移植到 C++，34 个标签与 Python
完全一致；整条流水线与 Python ONNX 参考版在 24 个用例上逐像素一致（`cpp/tests/parity_q.py`）。

**app**（`scripts/build_macos_app_q.sh` → `dist/SubjectMatting Q.app`，269MB：模型 236MB、库 32MB）：
- 右侧顶部"要抠什么？"输入框：留空 = 自动识别主体；输入后新加的图都按这个要求抠；回车或「重新抠图」用新要求重做选中的图，
  右键菜单也有。列表里每张图下面显示它用的要求。
- 「设置」（⌘,）里填阿里云百炼 API Key，存进钥匙串，保存后后台进程自动重启生效；没填 Key 时界面提示并给出设置入口。
- 后台进程 `matting-worker-q`：每张图先调 Qwen（缩到长边 1024 的 JPEG），再调 C++ 核心；网络在后台进程里，界面不卡。
- 本机实测：单主体每张 5.4~7.4 秒（Qwen 1.3~2.9 秒 + SAM3 约 4.5 秒 + 边缘 0.5~1 秒），两个主体的图约 15 秒。

**已知问题**：SAM3 解码器在 ONNX Runtime 上比 PyTorch 慢约 2.5 倍（对 72×72 个图像 token 的注意力，PyTorch 有优化算子），
每多一个主体多约 4 秒；CoreML 执行器因个别张量形状不支持而无法使用。

## 修订：纯色背景快速通道 + 整体主体提示词（2026-10-12）

用户用纯色背景的城堡模型（`~/Documents/打印/`）和一张办公室场景图测试，发现两个问题：

1. **主体被拆碎**：旧的自动提示词让 Qwen 把一个城堡模型拆成 castle / mountain / train / base 四个主体，每个单独跑 SAM3，
   拼起来的岩石有洞；办公室图拆成 6 个物件，每个约 4 秒，整张 30 多秒。
2. **SAM3 认不全"整体"**：即使给一个"castle model"整体框，蒸馏版 SAM3 也常常漏掉木质底座（取决于它是否认为底座属于
   "城堡模型"）。SAM2 版（BiRefNet 显著性）同样会漏掉底座或岩石。通用词 `object` / 纯框提示 `visual` 更差。

改动：

- **纯色背景快速通道**（自动模式，`matting/plain_background.py`，C++ `smq_matte_plain`）：图像边缘一圈 ≥95% 的像素与
  背景色（逐通道中位数）相差 ≤16 时判定为纯色背景，主体 = 与背景色相差 >24 的所有像素（去掉小于 0.05% 的碎点），
  再走同样的 trimap + ViTMatte。**不调 Qwen、不跑 SAM3**。4 张城堡图全部完整（含底座），每张 2.4~2.9 秒（原来 30 秒以上
  且有缺块）。15 张测试图里没有一张被误判为纯色背景（最接近的是黑底白头鹰，边缘一致率 0.73）。C++ 与 Python 逐像素一致。
- **自动提示词**：一件物品连同底座、配件算一个整体；几个独立个体（人、动物、玩偶，即使挨在一起）分别输出；label 只写一种东西的名字
  （不要数量和 and/with）。15 张测试图结果与之前持平（泰迪熊、狗和人、两只猫等）。
- 指定模式不变（总是 Qwen + SAM3）。

剩余：非纯色背景、主体多的图（如办公室场景，Qwen 给出 4 个物件）仍是每个主体约 4 秒。

## 修订：速度（2026-10-12）

worker 日志现在每张图一行，列出各步耗时（`~/Library/Logs/SubjectMattingQ.log`）。按用户的几张图回放，瓶颈和处理：

| 瓶颈 | 处理 | 效果 |
|---|---|---|
| ViTMatte 总在整张 1024² 图上跑 | 只在 trimap 非背景区域外扩 64 像素的范围内跑（`matte_cropped`，C++ 同步） | 全部用例抠图总时间 33.5s → 18.9s；"train" 3.6s → 0.65s。结果与整图版有差异（ViTMatte 的位置编码随输入尺寸变），两个方向都有，平均 0.3/255 |
| 等 Qwen 网络 2~5 秒 | 批量时，处理当前图的同时读入下一张图并提前请求 Qwen；单张图时，等 Qwen 的同时先做 SAM3 图像编码（新 C 接口 `smq_prepare`、`smq_is_plain`） | 批量里 Qwen 等待基本为 0；单张少等约 1 秒 |
| 读大图慢（12MP 2.6s） | 直接解码再按 EXIF 方向绘制，不再用 ImageIO 的全尺寸缩略图接口（8 种方向都与旧方法逐像素一致） | |

回放 6 张（纯色、指定 train、办公室 4 主体、两只猫、12MP 大图、纯色城堡）：总计约 74s → 49s。
仍然最慢的是 SAM3：每个主体约 3~4 秒（试过 ONNX Runtime 的 fp16 计算，CPU 上没有对应内核，反而慢得不可用）。

另修一个会让 app"卡住"的问题：app 是 ad-hoc 签名，每次更新后 macOS 会重新询问是否允许读取钥匙串里的 API Key；
原来在主线程同步读，提示框被挡住时整个 app 冻结、一张图都不处理。现在钥匙串读写都在后台线程，读完再启动后台进程；
开发时设置了环境变量 `DASHSCOPE_API_KEY` 就完全不碰钥匙串。
