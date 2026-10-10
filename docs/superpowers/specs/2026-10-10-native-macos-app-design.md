# 原生 macOS 界面设计

> **2026-10-10 更新：** 抠图进程已从 PyInstaller 打包的 `worker.py` 换成 Swift 写的 `matting-worker`，
> 调用 C++ 核心库（`cpp/`，ONNX Runtime + fp16 模型，模型内置在 app 里，不再下载）。协议不变；
> 输出路径规则同时有 Python（`matting/batch.py`）和 Swift（`MattingCore/JobPlanner.swift`）两份，测试用例相同。
> 下文关于 PyInstaller 和模型下载的部分只对 `worker.py` 仍然成立。

日期：2026-10-10

## 目标

用 SwiftUI 原生窗口替换打包版里的 Gradio 网页界面。双击 app 就弹出窗口，不再依赖浏览器，也不会被系统标成"未响应"。
只支持 macOS 14+ / Apple Silicon，ad-hoc 签名（没有 Apple Developer 账号）。

## 功能

1. 拖入或打开单张图片、多张图片、文件夹（不含子文件夹），自动排队处理。不再分"单张/批量"两个页面。
2. 左侧列表：缩略图、文件名、状态（排队 / 处理中 / 完成+耗时 / 失败+原因 / 已停止）。底部显示总进度。
3. 右侧大图：只显示抠图结果，画在棋盘格上显示透明区域；还没处理完时显示淡化的原图和状态。下方显示输出路径和各阶段耗时。
   （最初设计是原图/结果对比滑块，试用后按用户要求改为只显示结果。）
4. 处理完自动保存 PNG，规则和命令行 / 原批量功能一致：
   - 文件夹 `~/photos` → `~/photos_matting/`
   - 单独的图片 → 原图所在目录旁边的 `<目录名>_matting/`
   - 工具栏可指定统一的输出目录（只对还没处理的图片生效）；输出目录不能等于输入目录
   - 同名不同扩展名 → `<名>_<扩展名>.png`；不同目录的同名文件输出到同一目录时加 `_2`、`_3`
5. 工具栏：添加、输出目录、停止（清空排队，当前这张做完）、保存、在访达中显示。
   保存：「另存为…」（⌘⇧S）把选中的结果复制到别处；「导出全部结果…」（⌘⇧E）复制全部已完成结果到一个文件夹，同名不覆盖。
   列表右键菜单也有另存为；结果图可以直接拖到访达或其他应用。
6. 首次启动下载约 440MB 模型时显示进度条；下载或加载失败时显示原因和"重试"按钮。
7. 可以把图片/文件夹拖到程序坞图标上，或在访达里右键「打开方式」选 SubjectMatting。

不做：复制到剪贴板、Windows 版、Core ML 加速。

## 结构

```
SubjectMatting.app/Contents/
├─ MacOS/SubjectMatting          SwiftUI 界面（macos/ 下的 Swift Package）
└─ Resources/worker/matting-worker   Python 抠图进程（worker.py，PyInstaller onedir）
```

界面启动 worker 子进程，两者通过 stdin/stdout 每行一条 JSON 通信。worker 常驻，模型只加载一次。
worker 崩溃时界面把当前图片标为失败，自动重启 worker，继续处理队列。

### 协议

界面 → worker：

| 命令 | 字段 | 说明 |
|---|---|---|
| `plan` | `id`, `inputs`（文件/文件夹路径）, `output_dir`（可为 null） | 展开文件夹、计算输出路径。模型没加载好也能立即回复 |
| `run` | `id`, `src`, `dst` | 抠一张图并保存到 `dst` |

worker → 界面：

| 事件 | 字段 |
|---|---|
| `download` | `done`, `total`（字节） |
| `loading` | — |
| `ready` | — |
| `fatal` | `message`（模型下载/加载失败，worker 随后退出） |
| `plan` | `id`, `jobs: [{src, dst}]`, `skipped: [{path, reason}]`, `error` |
| `result` | `id`, `dst`, `total`, `stages` |
| `error` | `id`, `message`（单张失败，不影响后续） |

stdin 关闭（界面退出）时 worker 退出。

### worker 细节

- 第一行调用 `multiprocessing.freeze_support()`（v0.1.1 的教训）。
- 启动时把真正的 stdout 复制出来专门用于协议，再把文件描述符 1/2 重定向到 `~/Library/Logs/SubjectMatting.log`，
  这样库打印的警告、进度条（包括 C 层输出）都不会混进协议。
- 模型文件清单固定（3 个仓库共 11 个文件）。全部已缓存时设 `HF_HUB_OFFLINE=1` 直接加载，离线可用、启动更快；
  有缺失时用 `hf_hub_download` 逐个下载，另开线程按缓存目录里已下载的字节数（含 `.incomplete`）上报进度。
- 读 stdin 的线程直接处理 `plan`，`run` 放进队列由主线程顺序执行。
- 输出路径规则在 `matting/batch.py` 的 `plan_jobs()` 里实现，复用 `default_output_dir`、`output_names`，命令行和 worker 共用。
- 测试用环境变量：`SUBJECTMATTING_SELFTEST=1`（只导入依赖后退出，CI 用）、`SUBJECTMATTING_FAKE_PIPELINE=1`（不加载模型，原图加不透明 alpha，用于协议测试）。

### 界面细节

- Swift Package `macos/`：`MattingCore` 库（协议编解码、`JobStore` 队列状态机、`WorkerClient` 进程管理）+ `SubjectMatting` 可执行目标（SwiftUI 视图）+ `MattingCoreTests`。
- `JobStore` 通过一个 `WorkerTransport` 协议跟 worker 交互，测试时换成假实现。
- 开发时可用环境变量 `SUBJECTMATTING_WORKER` 指定 worker 命令（例如 `.venv/bin/python worker.py`）。
- 关闭最后一个窗口即退出 app。

## 打包与 CI

- `packaging/worker.spec`：PyInstaller 打包 `worker.py`，不含 gradio。
- `scripts/build_macos_app.sh`：`swift build -c release` → 组装 `.app`（Info.plist、可执行文件、worker）→ `codesign --force --deep --sign -`。
- GitHub Actions：pytest → `swift test` → 打包 worker → worker 自检 + multiprocessing 子进程检查 → 组装 app → 压缩 → tag 时发 Release。
- `app.py`（Gradio）保留在仓库给开发/Windows 用，不再打进 Mac 包。旧的 `packaging/SubjectMatting.spec` 删除。
- 版本 0.2.0。

## 测试

- pytest：`plan_jobs` 规则（文件夹、散文件、同名、跨目录冲突、输出=输入报错、非图片跳过）；用 fake pipeline 起真实 worker 子进程跑 `plan`/`run` 往返；worker 自检。
- swift test：协议 JSON 编解码；`JobStore`：排队→逐张发送、结果/失败更新状态、停止、worker 重启后继续。
- 手动：本机打包后双击运行，拖入文件夹实测。
