import AppKit
import MattingCore
import SwiftUI

final class AppDelegate: NSObject, NSApplicationDelegate {
    let store = JobStore { WorkerProcess() }

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Also needed when the bare binary is run outside the .app during development.
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
        store.start()
    }

    /// Images or folders dropped on the Dock icon or opened with "Open With".
    func application(_ application: NSApplication, open urls: [URL]) {
        store.add(urls)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }

    func applicationWillTerminate(_ notification: Notification) {
        store.shutdown()
    }
}

@main
struct SubjectMattingApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    private var store: JobStore { delegate.store }

    var body: some Scene {
        Window("SubjectMatting", id: "main") {
            ContentView()
                .environment(store)
                .frame(minWidth: 820, minHeight: 520)
        }
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("添加图片或文件夹…") { Panels.addFiles(to: store) }
                    .keyboardShortcut("o")
            }
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
            store.add(panel.urls)
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
