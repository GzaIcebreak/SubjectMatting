import MattingCore
import MattingQ
import SwiftUI

struct SettingsView: View {
    @Environment(JobStore.self) private var store
    @State private var key = QKey.shared.key ?? ""
    @State private var status: String?

    var body: some View {
        Form {
            Section {
                SecureField("API Key", text: $key, prompt: Text("sk-…"))
                HStack {
                    Button("保存") { save() }
                        .keyboardShortcut(.defaultAction)
                    Link("在阿里云百炼控制台获取 Key", destination: URL(string: "https://bailian.console.aliyun.com/")!)
                    Spacer()
                    if let status { Text(status).foregroundStyle(.secondary) }
                }
            } header: {
                Text("阿里云百炼（Qwen-VL）")
            } footer: {
                Text("Key 保存在本机的钥匙串里，只在调用阿里云接口时使用。")
            }
            Section("隐私与费用") {
                Text("每张图会缩小到长边 1024 像素后上传到阿里云百炼的 \(QwenClient.defaultModel) 模型，用来识别要抠的主体；分割和抠图都在本机完成。调用按阿里云的价格计费（每张图约 600 tokens）。")
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .formStyle(.grouped)
        .frame(width: 480)
    }

    private func save() {
        status = "正在保存…"
        let saving = key
        QKey.shared.save(saving) { ok in
            guard ok else {
                status = "保存到钥匙串失败"
                return
            }
            status = saving.trimmingCharacters(in: .whitespaces).isEmpty ? "已删除" : "已保存"
            store.restartWorker()
        }
    }
}
