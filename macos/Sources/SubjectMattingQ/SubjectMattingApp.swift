import AppKit
import MattingCore
import MattingQ
import SwiftUI

// SubjectMatting Q: copied from Sources/SubjectMatting and changed for Qwen-VL + SAM3 (instruction field, API key).

final class AppDelegate: NSObject, NSApplicationDelegate {
    // Called on every worker (re)start, so a key saved in Settings is picked up by restartWorker().
    let store = JobStore { WorkerProcess(extraEnvironment: QKey.shared.environment(), executableName: "matting-worker-q") }

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Also needed when the bare binary is run outside the .app during development.
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
        // The worker needs the API key; reading it may wait on a Keychain prompt, so not on the main thread.
        QKey.shared.load { [store] in store.start() }
    }

    /// Images or folders dropped on the Dock icon or opened with "Open With".
    func application(_ application: NSApplication, open urls: [URL]) {
        store.add(urls, instruction: QOptions.shared.instruction)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }

    func applicationWillTerminate(_ notification: Notification) {
        store.shutdown()
    }
}

@main
struct SubjectMattingQApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    private var store: JobStore { delegate.store }

    var body: some Scene {
        Window("SubjectMatting Q", id: "main") {
            ContentView()
                .environment(store)
                .frame(minWidth: 820, minHeight: 520)
        }
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("添加图片或文件夹…") { Panels.addFiles(to: store) }
                    .keyboardShortcut("o")
            }
            CommandGroup(replacing: .saveItem) {
                Button("另存为…") { Panels.saveSelected(of: store) }
                    .keyboardShortcut("s", modifiers: [.command, .shift])
                    .disabled(store.selectedResult == nil)
                Button("导出全部结果…") { Panels.exportAll(of: store) }
                    .keyboardShortcut("e", modifiers: [.command, .shift])
                    .disabled(store.doneCount == 0)
            }
        }
        Settings {
            SettingsView().environment(store)
        }
    }
}

enum Panels {
    static func addFiles(to store: JobStore) {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = true
        panel.allowedContentTypes = [.image, .folder]
        panel.prompt = "添加"
        if panel.runModal() == .OK {
            store.add(panel.urls, instruction: QOptions.shared.instruction)
        }
    }

    /// Results are already auto-saved to job.dst; this copies the selected one somewhere else.
    static func saveSelected(of store: JobStore) {
        guard let result = store.selectedResult else { return }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.png]
        panel.nameFieldStringValue = result.lastPathComponent
        panel.directoryURL = result.deletingLastPathComponent()
        panel.canCreateDirectories = true
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            try ResultExport.save(result, to: url)
        } catch {
            store.notice = "另存失败：\(error.localizedDescription)"
        }
    }

    static func exportAll(of store: JobStore) {
        let results = ResultExport.finishedResults(store.jobs)
        guard !results.isEmpty else { return }
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.canCreateDirectories = true
        panel.prompt = "导出到这里"
        panel.message = "把 \(results.count) 张抠图结果复制到这个文件夹（同名文件不会被覆盖）"
        guard panel.runModal() == .OK, let folder = panel.url else { return }
        let r = ResultExport.exportAll(results, to: folder)
        store.notice = r.errors.isEmpty
            ? "已导出 \(r.copied) 张到 \(folder.lastPathComponent)"
            : "已导出 \(r.copied) 张，失败 \(r.errors.count) 张：\(r.errors.prefix(2).joined(separator: "；"))"
        if r.copied > 0 {
            NSWorkspace.shared.activateFileViewerSelecting([folder])
        }
    }

    static func chooseOutputDir(for store: JobStore) {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.canCreateDirectories = true
        panel.prompt = "选择"
        panel.message = "抠图结果保存到这个文件夹（只影响还没处理的图片）"
        if panel.runModal() == .OK, let url = panel.url {
            store.setOutputDir(url)
        }
    }
}
