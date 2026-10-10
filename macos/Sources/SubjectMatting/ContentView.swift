import AppKit
import MattingCore
import SwiftUI

struct ContentView: View {
    @Environment(JobStore.self) private var store
    @State private var dropTargeted = false

    var body: some View {
        @Bindable var store = store
        NavigationSplitView {
            List(selection: $store.selection) {
                ForEach(store.jobs) { job in
                    JobRow(job: job).tag(job.id)
                }
            }
            .safeAreaInset(edge: .bottom) { StatusFooter().padding(10) }
            .navigationSplitViewColumnWidth(min: 220, ideal: 270)
        } detail: {
            if let id = store.selection, let job = store.jobs.first(where: { $0.id == id }) {
                JobDetail(job: job)
            } else {
                EmptyState()
            }
        }
        .toolbar { Toolbar() }
        .dropDestination(for: URL.self) { urls, _ in
            store.add(urls)
            return true
        } isTargeted: { dropTargeted = $0 }
        .overlay {
            if dropTargeted {
                RoundedRectangle(cornerRadius: 12)
                    .strokeBorder(Color.accentColor, style: StrokeStyle(lineWidth: 3, dash: [8]))
                    .padding(6)
                    .allowsHitTesting(false)
            }
        }
        .overlay(alignment: .top) {
            if let notice = store.notice {
                NoticeBanner(text: notice) { store.notice = nil }
                    .padding()
            }
        }
    }
}

// MARK: Toolbar

struct Toolbar: ToolbarContent {
    @Environment(JobStore.self) private var store

    private var selectedJob: Job? { store.jobs.first { $0.id == store.selection } }

    var body: some ToolbarContent {
        ToolbarItemGroup(placement: .primaryAction) {
            Button { Panels.addFiles(to: store) } label: { Label("添加", systemImage: "plus") }
                .help("添加图片或文件夹（也可以直接拖进窗口）")

            Menu {
                Button {
                    store.setOutputDir(nil)
                } label: {
                    if store.outputDir == nil { Image(systemName: "checkmark") }
                    Text("原文件夹旁的「<文件夹名>_matting」")
                }
                Button("选择文件夹…") { Panels.chooseOutputDir(for: store) }
                if let dir = store.outputDir {
                    Divider()
                    Text("当前：\(dir.path)")
                }
            } label: {
                Label(store.outputDir.map { "输出：\($0.lastPathComponent)" } ?? "输出：默认", systemImage: "folder")
            }
            .help("抠图结果保存的位置")

            if store.isProcessing {
                Button { store.stopQueue() } label: { Label("停止", systemImage: "stop.fill") }
                    .help("清空排队的图片，正在处理的这张会做完")
            } else if store.canResume {
                Button { store.resumeQueue() } label: { Label("继续", systemImage: "arrow.clockwise") }
                    .help("重新处理已停止和失败的图片")
            }

            Button {
                if let job = selectedJob { revealInFinder(job) }
            } label: { Label("在访达中显示", systemImage: "magnifyingglass") }
                .disabled(selectedJob == nil)
                .help("在访达中显示选中图片的结果")

            Button { store.removeFinished() } label: { Label("清除已完成", systemImage: "checklist.checked") }
                .disabled(store.doneCount == 0)
                .help("从列表里移除已完成的图片（文件不会删除）")
        }
    }

    private func revealInFinder(_ job: Job) {
        if case .done = job.status, FileManager.default.fileExists(atPath: job.dst.path) {
            NSWorkspace.shared.activateFileViewerSelecting([job.dst])
        } else {
            NSWorkspace.shared.activateFileViewerSelecting([job.src])
        }
    }
}

// MARK: Sidebar

struct JobRow: View {
    let job: Job

    var body: some View {
        HStack(spacing: 10) {
            Thumbnail(url: thumbnailURL, key: statusKey(job.status))
                .frame(width: 40, height: 40)
            VStack(alignment: .leading, spacing: 2) {
                Text(job.src.lastPathComponent).lineLimit(1).truncationMode(.middle)
                statusText.font(.caption)
            }
        }
        .padding(.vertical, 2)
    }

    private var thumbnailURL: URL {
        if case .done = job.status { return job.dst }
        return job.src
    }

    @ViewBuilder private var statusText: some View {
        switch job.status {
        case .queued: Text("排队中").foregroundStyle(.secondary)
        case .processing:
            HStack(spacing: 4) {
                ProgressView().controlSize(.mini)
                Text("处理中…")
            }.foregroundStyle(Color.accentColor)
        case let .done(total, _): Text("完成 · \(total, specifier: "%.1f") 秒").foregroundStyle(.secondary)
        case let .failed(message): Text("失败：\(message)").foregroundStyle(.red).lineLimit(1)
        case .stopped: Text("已停止").foregroundStyle(.orange)
        }
    }
}

struct StatusFooter: View {
    @Environment(JobStore.self) private var store

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            // With no jobs the empty state already shows the worker status in the middle of the window.
            if !store.jobs.isEmpty {
                if store.workerState != .ready {
                    WorkerStatusView(compact: true)
                }
                let finished = store.doneCount + store.failedCount
                ProgressView(value: Double(finished), total: Double(store.jobs.count))
                HStack {
                    Text("完成 \(store.doneCount) / 共 \(store.jobs.count)")
                    if store.failedCount > 0 {
                        Text("· 失败 \(store.failedCount)").foregroundStyle(.red)
                    }
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// Model download / loading / failure. Shown in the empty state, and in the sidebar footer once there are jobs.
struct WorkerStatusView: View {
    @Environment(JobStore.self) private var store
    var compact = false

    var body: some View {
        switch store.workerState {
        case .starting:
            HStack { ProgressView().controlSize(.small); Text("正在启动…") }
                .font(compact ? .caption : .body)
        case let .downloading(done, total):
            VStack(alignment: compact ? .leading : .center, spacing: 6) {
                Text("首次使用，正在下载模型").font(compact ? .caption : .headline)
                ProgressView(value: Double(done), total: Double(max(total, 1)))
                    .frame(maxWidth: compact ? .infinity : 320)
                Text("\(mb(done)) / \(mb(total)) MB").font(.caption).foregroundStyle(.secondary).monospacedDigit()
            }
        case .loading:
            HStack { ProgressView().controlSize(.small); Text("正在加载模型…") }
                .font(compact ? .caption : .body)
        case .ready:
            EmptyView()
        case let .failed(message):
            VStack(alignment: compact ? .leading : .center, spacing: 6) {
                Label("抠图引擎启动失败", systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.red)
                    .font(compact ? .caption : .headline)
                Text(message).font(.caption).foregroundStyle(.secondary)
                    .multilineTextAlignment(compact ? .leading : .center)
                    .textSelection(.enabled)
                Button("重试") { store.retryWorker() }
            }
        }
    }

    private func mb(_ bytes: Int64) -> String { String(format: "%.0f", Double(bytes) / 1_048_576) }
}

// MARK: Detail

struct EmptyState: View {
    @Environment(JobStore.self) private var store

    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "photo.on.rectangle.angled")
                .font(.system(size: 56, weight: .light))
                .foregroundStyle(.secondary)
            Text("把图片或文件夹拖到这里").font(.title2)
            Text("支持 jpg / png / webp / bmp，文件夹只处理第一层的图片。\n结果自动保存为透明背景的 PNG。")
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
            Button("选择图片或文件夹…") { Panels.addFiles(to: store) }
                .controlSize(.large)
                .keyboardShortcut(.defaultAction)
            if store.workerState != .ready {
                WorkerStatusView().padding(.top, 12)
            }
        }
        .padding(40)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

struct JobDetail: View {
    let job: Job

    var body: some View {
        VStack(spacing: 0) {
            ResultView(original: job.src, result: resultURL, resultKey: statusKey(job.status))
                .overlay {
                    switch job.status {
                    case .processing:
                        overlayBadge { ProgressView().controlSize(.small); Text("正在抠图…") }
                    case .queued:
                        overlayBadge { Text("排队中") }
                    case .stopped:
                        overlayBadge { Text("已停止，点工具栏「继续」重新处理") }
                    case let .failed(message):
                        overlayBadge { Label(message, systemImage: "exclamationmark.triangle.fill").foregroundStyle(.red) }
                    case .done:
                        EmptyView()
                    }
                }
            Divider()
            InfoBar(job: job)
        }
    }

    private var resultURL: URL? {
        if case .done = job.status { return job.dst }
        return nil
    }

    private func overlayBadge<C: View>(@ViewBuilder _ content: () -> C) -> some View {
        HStack(spacing: 6, content: content)
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .background(.regularMaterial, in: Capsule())
    }
}

struct InfoBar: View {
    let job: Job

    var body: some View {
        HStack(spacing: 16) {
            VStack(alignment: .leading, spacing: 2) {
                Text(job.src.lastPathComponent).font(.headline).lineLimit(1)
                Text("→ \(job.dst.path)").font(.caption).foregroundStyle(.secondary)
                    .lineLimit(1).truncationMode(.middle).textSelection(.enabled)
            }
            Spacer()
            if case let .done(total, stages) = job.status {
                VStack(alignment: .trailing, spacing: 2) {
                    Text("\(total, specifier: "%.1f") 秒").font(.headline).monospacedDigit()
                    Text(stageText(stages)).font(.caption).foregroundStyle(.secondary).monospacedDigit()
                }
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
    }

    private func stageText(_ stages: [String: Double]) -> String {
        let names = ["saliency": "识别主体", "sam2": "补全", "matting": "边缘"]
        return ["saliency", "sam2", "matting"].compactMap { key in
            stages[key].map { "\(names[key]!) \(String(format: "%.1f", $0))s" }
        }.joined(separator: " · ")
    }
}

struct NoticeBanner: View {
    let text: String
    let dismiss: () -> Void

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: "info.circle.fill").foregroundStyle(.orange)
            Text(text).lineLimit(2)
            Button { dismiss() } label: { Image(systemName: "xmark") }.buttonStyle(.borderless)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 10))
        .shadow(radius: 4, y: 2)
        .task(id: text) {
            try? await Task.sleep(for: .seconds(8))
            dismiss()
        }
    }
}

/// Changes whenever a job's output may have changed on disk, so cached images are reloaded.
func statusKey(_ status: JobStatus) -> String {
    switch status {
    case let .done(total, _): "done-\(total)"
    default: "\(status)"
    }
}
