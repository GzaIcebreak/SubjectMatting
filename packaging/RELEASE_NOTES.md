## SubjectMatting macOS（Apple Silicon）

自动主体抠图：BiRefNet_lite 显著性 → SAM2.1-tiny 物体补全 → ViTMatte 发丝级 alpha。纯 CPU 运行。

### 安装与首次运行

1. 下载 `SubjectMatting-macos-arm64.zip`，解压得到 `SubjectMatting.app`，拖到「应用程序」或任意目录。
2. 这个包**没有 Apple 开发者签名**，首次打开会被 Gatekeeper 拦截。任选其一：
   - 右键 `SubjectMatting.app` → 打开 → 再点「打开」；
   - 或在终端执行：`xattr -dr com.apple.quarantine /path/to/SubjectMatting.app`
3. 首次启动会从 Hugging Face 下载约 440MB 模型到 `~/.cache/huggingface`，之后离线可用。
4. 启动后浏览器自动打开 http://127.0.0.1:7860。

### 排错

- 日志：`~/Library/Logs/SubjectMatting.log`
- 端口被占用：用 `PORT=7861 open -a SubjectMatting` 换端口。
- 仅支持 Apple Silicon（M1 及以后）与 macOS 14+。
