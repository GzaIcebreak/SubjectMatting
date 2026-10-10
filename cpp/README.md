# subject_matting — C++ 抠图核心库（ONNX Runtime）

把 SubjectMatting 的抠图流水线（BiRefNet_lite 显著性 → SAM2.1-tiny 物体补全 → ViTMatte 发丝级 alpha）做成一个
不依赖 Python / PyTorch 的 C++ 库，对外只有一个 C 头文件，方便嵌入任何 app（Swift、Kotlin/JNI、C#、Python ctypes…）。

## 体积

| 部分 | 大小 |
|---|---|
| `libsubject_matting`（本库） | 0.14 MB |
| `libonnxruntime`（官方完整版） | 33 MB |
| 模型 `*.fp16.onnx` ×4 | 208 MB |
| **合计** | **约 240 MB**（Python/PyTorch 版：程序 823 MB + 模型 418 MB） |

模型也可以首次使用时再下载，这样安装包只有约 33 MB。

## 质量

- 与 Python 参考实现 `matting/pipeline_onnx.py` 逐像素一致（15 张测试图里 13 张完全相同，其余只有零星像素差 1/255）。
  图像缩放、腐蚀/膨胀、连通域、距离变换都按 PIL / OpenCV 的算法实现，`tests/ops_test.cpp` 逐像素对照。
- fp16 模型与原 PyTorch 版相比：alpha 平均差 0.01/255，IoU ≥ 0.999。int8 不推荐（BiRefNet 量化后反而更大，质量下降，只快约 10%）。
- 注意：不同 JPEG 解码器解出的像素略有不同，个别临界图片的结果可能因此不同；这与本库无关。

## C 接口

```c
#include "subject_matting.h"

char err[256];
sm_context *ctx = sm_create("/path/to/models", "fp16", /*num_threads*/ 0, err, sizeof err);
if (!ctx) { fprintf(stderr, "%s\n", err); return; }

/* rgb: width*height RGB 字节，行距 stride；EXIF 方向请先转正。out: width*height*4 字节 RGBA。 */
sm_timings t;
if (sm_matte(ctx, rgb, width, height, width * 3, out_rgba, &t, err, sizeof err) != 0)
    fprintf(stderr, "%s\n", err);

sm_destroy(ctx);
```

- 一个 `sm_context` 加载约 200 MB 模型，不是线程安全的：每个线程一个，或者串行调用。
- 带透明通道的输入请先合成到白底（和 Python 版 `load_rgb` 一致）。

## 构建

依赖：CMake ≥ 3.18、C++17 编译器、ONNX Runtime 1.31。

```bash
cpp/scripts/fetch_onnxruntime.sh
```
```bash
cmake -S cpp -B cpp/build -DONNXRUNTIME_DIR=$PWD/cpp/third_party/onnxruntime
```
```bash
cmake --build cpp/build
```

GitHub 下载慢时，macOS 上可以用 `cpp/scripts/fetch_onnxruntime.sh --wheel`：头文件从 GitHub 取，库文件用 `.venv` 里
onnxruntime pip 包自带的 dylib。

其他平台用对应的 ONNX Runtime 预编译包（Windows / Linux / Android AAR / iOS xcframework 都有官方发布），
把 `ONNXRUNTIME_DIR` 指过去即可；本库除 ONNX Runtime 外没有其他依赖。

## 模型

```bash
PYTHONPATH=. .venv/bin/python scripts/export_onnx.py
```

在 `onnx_models/` 生成 fp32 / fp16 / int8 三套（需要 PyTorch 环境，只在导出时用）。模型的输入输出见脚本开头的说明。

## 命令行与测试

```bash
cpp/build/sm_cli --models onnx_models in.jpg out.png
```
```bash
.venv/bin/python cpp/tests/gen_ops_cases.py cpp/build/ops_cases
```
```bash
cpp/build/sm_ops_test cpp/build/ops_cases
```

打包成 SDK 压缩包（头文件 + 库 + fp16 模型 + sm_cli）：`cpp/scripts/package_sdk.sh` → `dist/SubjectMatting-core-<系统>-<架构>.zip`。
