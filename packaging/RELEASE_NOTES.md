## SubjectMatting macOS（Apple Silicon）

自动主体抠图：BiRefNet_lite 显著性 → SAM2.1-tiny 物体补全 → ViTMatte 发丝级 alpha。纯 CPU 运行。

### v0.2.0 更新

- 改成原生 macOS 窗口，不再通过浏览器使用。双击打开就是 app 窗口，关掉窗口即退出。
- 把图片、多张图片或文件夹拖进窗口（或拖到程序坞图标上）即可排队处理；左侧列表显示每张的状态和耗时，右侧显示抠图结果。
- 结果自动保存：文件夹 `photos` → 同级的 `photos_matting`；工具栏可以改输出位置、停止/继续、在访达中显示。
- 「另存为…」（⌘⇧S）和「导出全部结果…」（⌘⇧E）：把结果另存到任意位置；也可以直接把结果图拖到访达或其他应用。
- 首次启动下载模型时显示进度条；下载失败可以点「重试」。
- 手机照片按 EXIF 方向自动摆正。

### v0.1.1 更新

- 修复：打包后的 app 启动时不断重复启动自身（多进程辅助进程把整个应用又运行了一遍），导致一直卡在启动、页面打不开。v0.1.0 用户请升级。

### 安装与首次运行

1. 下载 `SubjectMatting-macos-arm64.zip`，解压得到 `SubjectMatting.app`，拖到「应用程序」或任意目录。
2. 这个包**没有 Apple 开发者签名**，首次打开会被 Gatekeeper 拦截。任选其一：
   - 先双击打开一次（会被拦截），再到「系统设置 → 隐私与安全性」底部点「仍要打开」（macOS 15 起右键「打开」已不能绕过）；
   - 或在终端执行：`xattr -dr com.apple.quarantine /path/to/SubjectMatting.app`
3. 首次启动会从 Hugging Face 下载约 440MB 模型到 `~/.cache/huggingface`，之后离线可用。
4. 下载完成后即可把图片或文件夹拖进窗口。

### 排错

- 日志：`~/Library/Logs/SubjectMatting.log`
- 仅支持 Apple Silicon（M1 及以后）与 macOS 14+。
