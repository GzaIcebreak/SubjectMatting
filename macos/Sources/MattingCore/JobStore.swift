import Foundation
import Observation

public enum JobStatus: Equatable, Sendable {
    case queued
    case processing
    case done(total: Double, stages: [String: Double])
    case failed(String)
    /// Was queued when the user pressed stop.
    case stopped
}

public struct Job: Identifiable, Equatable, Sendable {
    public let id: UUID
    public let src: URL
    public var dst: URL
    public var status: JobStatus

    var isWaiting: Bool { status == .queued || status == .stopped }
}

public enum WorkerState: Equatable, Sendable {
    case starting
    case downloading(done: Int64, total: Int64)
    case loading
    case ready
    case failed(String)
}

/// The image queue. Sends one `run` at a time to the worker, keeps every job's status, and restarts the worker
/// when it dies. Must be used from the main thread (WorkerTransport delivers its callbacks there).
@Observable
public final class JobStore {
    public private(set) var jobs: [Job] = []
    public var selection: Job.ID?
    public private(set) var workerState: WorkerState = .starting
    /// nil: every input goes to <its folder>_matting next to its folder.
    public private(set) var outputDir: URL?
    /// Latest message about dropped items that were not added; the UI clears it when shown.
    public var notice: String?

    public static let maxConsecutiveCrashes = 3

    private enum PlanPurpose { case add, replan }

    private let makeTransport: () -> WorkerTransport
    private var transport: WorkerTransport?
    private var lastCommandID = 0
    private var pendingPlans: [Int: PlanPurpose] = [:]
    /// Plans requested while no worker is running (before start(), or while it restarts after a crash).
    private var unsentPlans: [(PlanPurpose, [String])] = []
    private var running: (commandID: Int, jobID: UUID)?
    private var consecutiveCrashes = 0

    public init(makeTransport: @escaping () -> WorkerTransport) {
        self.makeTransport = makeTransport
    }

    // MARK: Worker lifecycle

    public func start() {
        let t = makeTransport()
        t.onEvent = { [weak self] in self?.handle($0) }
        t.onExit = { [weak self] in self?.workerExited($0) }
        transport = t
        workerState = .starting
        do {
            try t.start()
        } catch {
            transport = nil
            workerState = .failed("无法启动抠图进程：\(error.localizedDescription)")
            return
        }
        let unsent = unsentPlans
        unsentPlans.removeAll()
        for (purpose, inputs) in unsent {
            sendPlan(purpose, inputs: inputs)
        }
    }

    /// After a fatal error (e.g. the model download failed): try again from scratch.
    public func retryWorker() {
        guard case .failed = workerState else { return }
        consecutiveCrashes = 0
        start()
    }

    public func shutdown() {
        transport?.onExit = nil
        transport?.stop()
        transport = nil
    }

    // MARK: User actions

    public func add(_ urls: [URL]) {
        guard !urls.isEmpty else { return }
        sendPlan(.add, inputs: urls.map(\.path))
    }

    /// Applies to jobs that have not been processed yet; finished results stay where they are.
    public func setOutputDir(_ url: URL?) {
        outputDir = url
        let waiting = jobs.filter(\.isWaiting)
        if !waiting.isEmpty {
            sendPlan(.replan, inputs: waiting.map(\.src.path))
        }
    }

    /// Queued jobs become `stopped`; the image being processed finishes.
    public func stopQueue() {
        for i in jobs.indices where jobs[i].status == .queued {
            jobs[i].status = .stopped
        }
    }

    /// Requeues stopped and failed jobs.
    public func resumeQueue() {
        for i in jobs.indices {
            switch jobs[i].status {
            case .stopped, .failed: jobs[i].status = .queued
            default: break
            }
        }
        pump()
    }

    public func removeFinished() {
        jobs.removeAll {
            if case .done = $0.status { return true }
            return false
        }
        if let s = selection, !jobs.contains(where: { $0.id == s }) {
            selection = jobs.first?.id
        }
    }

    // MARK: Derived state

    public var isProcessing: Bool { jobs.contains { $0.status == .queued || $0.status == .processing } }
    public var canResume: Bool {
        jobs.contains {
            switch $0.status {
            case .stopped, .failed: return true
            default: return false
            }
        }
    }

    public var doneCount: Int { jobs.filter { if case .done = $0.status { return true } else { return false } }.count }
    public var failedCount: Int { jobs.filter { if case .failed = $0.status { return true } else { return false } }.count }

    // MARK: Worker events

    private func nextCommandID() -> Int {
        lastCommandID += 1
        return lastCommandID
    }

    private func sendPlan(_ purpose: PlanPurpose, inputs: [String]) {
        guard let transport else {
            if case .failed = workerState {
                notice = "抠图引擎没有运行，请先点「重试」"
            } else {
                unsentPlans.append((purpose, inputs))
            }
            return
        }
        let id = nextCommandID()
        pendingPlans[id] = purpose
        transport.send(.plan(id: id, inputs: inputs, outputDir: outputDir?.path))
    }

    func handle(_ event: WorkerEvent) {
        switch event {
        case let .download(done, total):
            workerState = .downloading(done: done, total: total)
        case .loading:
            workerState = .loading
        case .ready:
            workerState = .ready
            pump()
        case let .fatal(message):
            // The worker exits right after this; workerExited() then leaves the state alone.
            workerState = .failed(message)
        case let .plan(id, planned, skipped, error):
            guard let purpose = pendingPlans.removeValue(forKey: id) else { return }
            if let error {
                notice = "无法添加：\(error)"
                return
            }
            switch purpose {
            case .add: addPlanned(planned)
            case .replan: applyReplan(planned)
            }
            if !skipped.isEmpty {
                notice = Self.describe(skipped)
            }
            pump()
        case let .result(id, _, total, stages):
            consecutiveCrashes = 0
            finish(id, .done(total: total, stages: stages))
        case let .error(id, message):
            consecutiveCrashes = 0
            finish(id, .failed(message))
        }
    }

    private func addPlanned(_ planned: [PlannedJob]) {
        var firstNew: UUID?
        for p in planned {
            let src = URL(fileURLWithPath: p.src).standardizedFileURL
            let dst = URL(fileURLWithPath: p.dst).standardizedFileURL
            if let i = jobs.firstIndex(where: { $0.src == src }) {
                // Added again: run it again unless it is still waiting or running.
                if jobs[i].status == .queued || jobs[i].status == .processing { continue }
                jobs[i].dst = uniqueDestination(dst, excluding: jobs[i].id)
                jobs[i].status = .queued
                firstNew = firstNew ?? jobs[i].id
            } else {
                let id = UUID()
                jobs.append(Job(id: id, src: src, dst: uniqueDestination(dst, excluding: id), status: .queued))
                firstNew = firstNew ?? id
            }
        }
        if selection == nil {
            selection = firstNew
        }
    }

    private func applyReplan(_ planned: [PlannedJob]) {
        for p in planned {
            let src = URL(fileURLWithPath: p.src).standardizedFileURL
            if let i = jobs.firstIndex(where: { $0.src == src && $0.isWaiting }) {
                jobs[i].dst = uniqueDestination(URL(fileURLWithPath: p.dst).standardizedFileURL, excluding: jobs[i].id)
            }
        }
    }

    /// The worker only sees one drop at a time, so a.jpg and a.png dropped separately would both become a.png.
    private func uniqueDestination(_ dst: URL, excluding id: UUID) -> URL {
        let taken = Set(jobs.filter { $0.id != id }.map(\.dst))
        var candidate = dst
        var n = 1
        while taken.contains(candidate) {
            n += 1
            let stem = dst.deletingPathExtension().lastPathComponent
            candidate = dst.deletingLastPathComponent().appendingPathComponent("\(stem)_\(n).png")
        }
        return candidate
    }

    private func pump() {
        guard workerState == .ready, running == nil, let transport,
              let i = jobs.firstIndex(where: { $0.status == .queued })
        else { return }
        jobs[i].status = .processing
        let id = nextCommandID()
        running = (id, jobs[i].id)
        transport.send(.run(id: id, src: jobs[i].src.path, dst: jobs[i].dst.path))
    }

    private func finish(_ commandID: Int, _ status: JobStatus) {
        guard let r = running, r.commandID == commandID else { return }
        running = nil
        if let i = jobs.firstIndex(where: { $0.id == r.jobID }) {
            jobs[i].status = status
        }
        pump()
    }

    private func workerExited(_ status: Int32) {
        transport = nil
        pendingPlans.removeAll()
        if let r = running {
            running = nil
            if let i = jobs.firstIndex(where: { $0.id == r.jobID }) {
                jobs[i].status = .failed("抠图进程意外退出（状态 \(status)）")
            }
        }
        if case .failed = workerState { return }
        consecutiveCrashes += 1
        if consecutiveCrashes >= Self.maxConsecutiveCrashes {
            workerState = .failed("抠图进程连续退出 \(consecutiveCrashes) 次，详情见 ~/Library/Logs/SubjectMatting.log")
            return
        }
        start()
    }

    static func describe(_ skipped: [SkippedInput]) -> String {
        let shown = skipped.prefix(3).map { "\(URL(fileURLWithPath: $0.path).lastPathComponent)（\($0.reason)）" }
        let more = skipped.count > 3 ? " 等 \(skipped.count) 项" : ""
        return "未添加：" + shown.joined(separator: "、") + more
    }
}
