# 自动主体抠图（CPU）设计文档

日期：2026-10-09

> **2026-10-10 更新：只保留方案 A。** 方案 C（EfficientSAM3）的代码、上游仓库和权重都已删除（移到回收站）。
> 下文中方案 C 的部分仅作为选型记录保留；代码以 `matting/pipeline_a.py` 为准。

## 目标

在纯 CPU 本机上，对**任意通用图片**自动分离主体与背景，输出**发丝级 alpha**（RGBA PNG）。
实现两套方案并在同一批图片上对比质量与耗时。

## 约束

- 运行环境：纯 CPU（本机 Intel Core Ultra 7 265K / 20 线程 / 31GB；即使有 GPU 也强制 CPU）
- Python 3.12 venv（`.venv`），PyTorch CPU 版
- 无需人工提示：全自动；可选人工覆盖（文本提示）
- 质量优先：单张数秒可接受

## 共同结构

两条流水线都分为：**主体 mask** → **trimap** → **ViTMatte 抠图** → RGBA。
SAM 系列只给二值 mask，发丝细节由 ViTMatte-S（`hustvl/vitmatte-small-composition-1k`，~25M）补上。

```
pipeline_a:  显著性(BiRefNet_lite@512) ─► 框+点 ─► SAM2.1-hiera-tiny ─┐
                                                                      ├─► trimap ─► ViTMatte-S ─► alpha
pipeline_c:  EfficientSAM3 EV-M + 文本提示(自动词表) ─► 选实例 ───────┘
```

### 方案 A：显著性 → SAM2.1-tiny → ViTMatte

1. **显著性**：`ZhengPeng7/BiRefNet_lite`，输入缩放到 512×512 以加速，输出显著图 `S∈[0,1]`。
   （原计划用 U²-Netp，但 HF 上只有第三方上传版本；改用作者官方的 BiRefNet_lite，低分辨率下只用来定位主体。）
2. **提示生成**：`S>0.5` 二值化 → 保留面积 ≥ 最大连通域 10% 的连通域 → 外接框作为 box 提示；
   每个保留连通域的距离变换峰值作为正点。
3. **SAM2.1-hiera-tiny**（`facebook/sam2.1-hiera-tiny`，transformers `Sam2Model`）：box+正点 → 取 3 个多 mask 中
   与显著图 IoU 最高者（而非仅看 SAM 分数，避免选到部件/整图）。
4. **融合**：SAM mask 与显著 mask 的并集再与扩张后的框求交，作为最终主体 mask（SAM 补边界，显著图补 SAM 漏掉的细长部分）。

### 方案 C：EfficientSAM3（SAM3 蒸馏版）→ ViTMatte

1. 模型：EfficientSAM3 **EV-M**（EfficientViT 视觉 + MobileCLIP-S0 文本，89M 参数，官方 stage3 微调权重
   `efficientsam3_ft/efficientsam3_efficientvit.pt`），Apache-2.0。
2. **自动提示**：SAM3 需要名词概念，通用图片事先不知道主体类别。做法：
   - 图像编码一次，对一个宽泛词表（person, animal, dog, cat, bird, car, vehicle, product, bottle, plant, flower,
     furniture, toy, food, object）逐个做文本提示（文本编码和解码很便宜）。
   - 每个检出实例打分：`score_sam3 × center_prior × area_term`，取最高者；与其重叠 > 50% 的同提示实例合并。
   - 支持 `--prompt` 手动指定，跳过词表。
3. 主体 mask → 同样的 trimap + ViTMatte。

### trimap 与抠图（共用）

- 处理分辨率：长边缩放到 1024（ViTMatte 输入需 32 对齐，内部 pad）。
- trimap：mask 腐蚀 k 像素为确定前景，膨胀 k 像素外为确定背景，中间为未知；`k = max(5, 0.012 × 长边)`。
- ViTMatte 输出 alpha 后上采样回原图分辨率，与原图合成 RGBA。

## 文件结构

```
SubjectMatting/
  docs/                       本文档 + 测试报告
  matting/
    common.py                 读写图像、resize、trimap、ViTMatte 封装、计时
    saliency.py               BiRefNet_lite 显著性
    pipeline_a.py             方案 A
    pipeline_c.py             方案 C（依赖 third_party/efficientsam3）
  run.py                      CLI：python run.py --pipeline a|c|both --input dir --output dir
  bench.py                    对测试图批量跑两方案，输出耗时表 + 对比拼图 + HTML 报告
  app.py                      Gradio 本地 Web 界面（上传图片，左右对比 A/C）
  test_images/                测试图（多类别）
  weights/                    下载的权重（不入库）
  third_party/efficientsam3/  上游代码
```

## 测试

- **单元**：trimap 生成（形状、三值、腐蚀/膨胀方向）、提示生成（多连通域、空显著图回退到全图中心框）、
  实例选择打分（构造假实例）。
- **集成**：两条流水线在测试图上端到端跑通，输出 RGBA，alpha 非全 0/全 1。
- **基准**：每张图记录各阶段 CPU 耗时（首次加载与稳态分开），生成对比拼图（原图 / A / C，棋盘格背景）和报告。

## 错误处理

- 显著图为空 → 回退为图像中心 80% 框。
- SAM3 词表全部无检出 → 回退为 `"object"` 提示；仍无 → 输出全图 mask 并告警。
- 模型权重缺失 → 明确报错并给出下载命令。

## 实施记录（与上文设计的偏差）

实测后做了以下调整，代码以此为准：

1. **显著性分辨率 512 → 768（空图时重试 1024）**。BiRefNet_lite 按 1024 训练，在 512 输入下对部分图片
   直接输出全 0（cat_whiskers、groceries），导致方案 A 退化成中心框。CPU 耗时：512=1.8s，768=2.6s，1024=9.5s。
2. **方案 C 词表 15 个 → 6 个**（person / animal / vehicle / plant / food / object）。每个提示都要完整跑一遍
   SAM3 融合编码器 + 解码器（CPU 上约 0.65s/个），耗时与词表长度线性相关。
   还试过把所有提示**批量**送进一次前向：结果和逐个跑完全一致，但 CPU 上更慢（12.6s vs 9.8s），所以保留逐个运行。
3. **方案 C 实例选择新增"包含合并"规则**：任意提示检出的实例，只要 ≥80% 落在最佳实例外接框内、且与其 mask 相接，
   就并入。解决"车 object mask 把车内购物袋挖成洞"的问题。
4. EfficientSAM3 EV-M 对应的主干是 `efficientvit b1`（权重核对：0 个缺失键、0 个形状不匹配）。
5. 观察到蒸馏版文本编码器的概念区分很弱：熊被判成 `person`(0.95)，泰迪熊被判成 `food`。
   之所以还能自动抠对，主要靠"分数 × 中心先验 × 面积"的选择规则，而不是语义正确。

测试结果见 [test-report](2026-10-09-test-report.md)。

### 2026-10-10：碎片 / 非主体 / 细笔画修复（方案 A）

- **SAM 孤岛丢弃**（`fuse_masks`）：SAM mask 中不与显著区域相连的连通块视为背景碎片，不再并入。
  修复：亭子左侧的月洞门碎片、礼盒下方的笔碎点、20 风格化里的星星和爱心。
  代价：显著模型漏掉、只有 SAM 抓到的独立小元素也会被丢，例如 21 风格化卡片的"一"和小圆点。
  量过 SAM logit，没法区分这类合法元素和碎片（合法笔画 0.04–0.08，碎片 0.10–0.42）。
- **贴边的次级显著区域丢弃**（`prompts_from_saliency`）：最大连通域始终保留；其余连通域碰到图像边框就丢弃
  （通常是从画框边缘探进来的景物）。在 52 张图上只影响 11 风格化（左上角的树叶）。
- **细笔画保核心**（`make_trimap`）：整块被腐蚀掉的细连通域保留中轴作为确定前景，避免整段变成未知区后被 ViTMatte 抹掉。
- SAM 改为图像编码只算一次（`set_image` / `predict`），只重跑 decoder 约 41ms，输出与原实现逐像素一致。
- **试过但已撤回**：
  - 框整体外扩 5%：SAM 会把背景（椅子、窗户）一起抠进来。
  - "被框平切"检测：盘底的切口沿椭圆边缘，不是平直的，检测不到。
  - 单边外扩加守卫阈值：多出来的究竟是物体的延续还是背景，靠这些指标分不开。
  - 盘子底部被切的问题由下面的"点驱动物体补全"解决。

### 2026-10-10：框驱动 → 点驱动的物体补全（方案 A）

**问题根因**：旧流程用显著图的外接框当 SAM 的硬边界。显著图只覆盖"蛋挞 + 盘面"，盘子底边在它那里是 0，
框到不了那里，SAM 再准也被截断。单纯放大框会把椅子等背景带进来。

**新流程**（`pipeline_a.py`）：SAM 决定物体边界，显著图决定该物体算不算主体。
1. `sample_points`：在显著区域内取 ≤25 个提示点（每个连通域的距离变换峰值 + 腐蚀核心上的粗网格），不给框。
2. `Sam2Tiny.predict_points`：所有点一次解码，每点 3 个层级的候选 mask，同时算 SAM 稳定性分数
   （logit 阈值 ±1 两个 mask 的 IoU）。图像编码只算一次。
3. `complete_mask`：每个点取**通过规则的最大候选**，规则 = 包含率（mask 落在显著区域内的比例）≥ 0.6
   **且** 稳定性 ≥ 0.4；全部并集。
4. `fuse_masks`：并上显著图、丢孤岛，再按显著外接框外扩 15% 做安全裁剪。

**阈值依据**（52 张图实测）：
- 包含率：盘子 0.82–0.85，礼盒被显著图漏掉的上半部分 0.71–0.74，logo 周围的光晕 0.64–0.67。
  只用包含率时 0.6 会放进光晕，0.75 会切掉礼盒上半部分，中间只有 0.03 的空隙，分不开。
- 稳定性：光晕 0.08–0.34，礼盒 0.46/0.81，盘子 0.48–0.95。光晕是软渐变，SAM 对它的边界没有把握，
  这正是稳定性分数衡量的东西。0.4 两边各有 ≥0.06 的余量。
- 这些阈值是在 52 张图上调出来的，样本不大；换图集要复查。

**结果**：两张蛋挞的盘子完整；之前修好的碎片、树叶、毯子、星星没有复发；礼盒、光晕图不受影响。
耗时 10 张基准平均 3.78s → 4.33s（SAM 阶段 0.62 → 1.12s）。
**已知取舍**：15 写实的木槿花多带了一片相连的叶子（SAM 认为花 + 叶是一个物体，包含率 0.85）；
21 风格化卡片的"一"字仍缺（显著图漏掉、SAM 只给出孤岛，与碎片无法区分）。

## 非目标

- 视频、批量服务化、ONNX 导出、量化（可作为后续优化）。
