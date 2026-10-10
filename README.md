# SubjectMatting — CPU 自动主体抠图（SAM2.1-tiny）

全自动流水线，输出带发丝级 alpha 的 RGBA PNG：

BiRefNet_lite 显著性 → 点提示 → SAM2.1-hiera-tiny 物体补全 → trimap → ViTMatte-S

CPU（Ultra 7 265K）上平均约 4.3s/张。设计见 [docs/2026-10-09-subject-matting-design.md](docs/2026-10-09-subject-matting-design.md)，
测试结果见 [docs/2026-10-09-test-report.md](docs/2026-10-09-test-report.md)。

## macOS 独立包（Apple Silicon）

GitHub Actions 的 `macos-latest` runner 用 PyInstaller 打成 `SubjectMatting.app`（见 `.github/workflows/build-macos.yml`
和 `packaging/SubjectMatting.spec`）。手动触发 workflow 可在 Actions 的 artifact 里拿到 `SubjectMatting-macos-arm64.zip`；
推 `v*` 标签会自动创建 Release。安装与首次运行说明见 [packaging/RELEASE_NOTES.md](packaging/RELEASE_NOTES.md)
（无开发者签名，首次需右键打开；模型首次运行时下载）。

## 环境

已建好：`.venv`（Python 3.12，torch CPU 版）。重建方法：

```bash
python -m uv venv .venv --python 3.12 --native-tls
python -m uv pip install --native-tls -p .venv torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m uv pip install --native-tls -p .venv transformers safetensors huggingface_hub opencv-python-headless timm einops kornia gradio pytest
```

模型首次运行时自动从 Hugging Face 下载到 `~/.cache/huggingface`：
`facebook/sam2.1-hiera-tiny`、`ZhengPeng7/BiRefNet_lite`、`hustvl/vitmatte-small-composition-1k`（共约 440MB）。

> 本机有 TLS 代理：pip 要加 `--use-feature=truststore`，uv 要加 `--native-tls`。

## 使用

```bash
.venv/Scripts/python app.py
```
浏览器打开 http://127.0.0.1:7860：
- **单张**：上传图片即可抠图，RGBA 结果可直接下载。
- **批量**：填入文件夹路径（可以直接粘贴带引号的路径）。处理该文件夹下的 jpg/jpeg/png/webp/bmp（不含子文件夹），
  每张输出 `<原文件名>.png`；输出目录留空时默认放在同级的 `<文件夹名>_matting`（例：`D:\photos` → `D:\photos_matting`）。
  单张出错会跳过并列在汇总里，不会中断整批；输出目录不能和输入目录相同。

命令行批量（规则相同）：

```bash
.venv/Scripts/python run.py --input D:/photos
```
```bash
.venv/Scripts/python run.py --input D:/photos --output D:/photos_out
```
```bash
.venv/Scripts/python bench.py
```
`bench.py` 会生成 `outputs/bench/report.html`。

```bash
.venv/Scripts/python -m pytest tests -q
```
