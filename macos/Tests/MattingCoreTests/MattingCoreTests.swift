import Foundation
import XCTest

@testable import MattingCore

final class FakeTransport: WorkerTransport {
    var onEvent: ((WorkerEvent) -> Void)?
    var onExit: ((Int32) -> Void)?
    var sent: [WorkerCommand] = []
    var started = 0
    var stopped = false

    func start() throws { started += 1 }
    func send(_ command: WorkerCommand) { sent.append(command) }
    func stop() { stopped = true }

    func emit(_ event: WorkerEvent) { onEvent?(event) }
    func exit(_ status: Int32) { onExit?(status) }

    var lastRun: (id: Int, src: String, dst: String)? {
        for c in sent.reversed() { if case let .run(id, src, dst) = c { return (id, src, dst) } }
        return nil
    }

    var lastPlanID: Int? {
        for c in sent.reversed() { if case let .plan(id, _, _) = c { return id } }
        return nil
    }
}

final class ProtocolTests: XCTestCase {
    func testDecodesEveryEventType() throws {
        func d(_ s: String) throws -> WorkerEvent { try WorkerEvent.decode(Data(s.utf8)) }
        XCTAssertEqual(try d(#"{"type":"download","done":5,"total":10}"#), .download(done: 5, total: 10))
        XCTAssertEqual(try d(#"{"type":"ready"}"#), .ready)
        XCTAssertEqual(try d(#"{"type":"fatal","message":"no net"}"#), .fatal(message: "no net"))
        XCTAssertEqual(
            try d(#"{"type":"plan","id":1,"jobs":[{"src":"/a.jpg","dst":"/m/a.png"}],"skipped":[{"path":"/x.txt","reason":"不支持的格式"}],"error":null}"#),
            .plan(id: 1, jobs: [PlannedJob(src: "/a.jpg", dst: "/m/a.png")], skipped: [SkippedInput(path: "/x.txt", reason: "不支持的格式")], error: nil)
        )
        XCTAssertEqual(
            try d(#"{"type":"result","id":2,"dst":"/m/a.png","total":1.5,"stages":{"sam2":0.5}}"#),
            .result(id: 2, dst: "/m/a.png", total: 1.5, stages: ["sam2": 0.5])
        )
        XCTAssertEqual(try d(#"{"type":"error","id":3,"message":"boom"}"#), .error(id: 3, message: "boom"))
        XCTAssertThrowsError(try d(#"{"type":"surprise"}"#))
    }

    func testEncodesCommandsAsOneLine() throws {
        let line = WorkerCommand.plan(id: 7, inputs: ["/照片"], outputDir: nil).encodedLine()
        XCTAssertEqual(line.last, 0x0A)
        let obj = try JSONSerialization.jsonObject(with: line.dropLast()) as! [String: Any]
        XCTAssertEqual(obj["cmd"] as? String, "plan")
        XCTAssertEqual(obj["inputs"] as? [String], ["/照片"])
        XCTAssertTrue(obj["output_dir"] is NSNull)
    }
}

final class JobStoreTests: XCTestCase {
    var transports: [FakeTransport] = []
    var store: JobStore!
    var t: FakeTransport { transports.last! }

    override func setUp() {
        transports = []
        store = JobStore { [unowned self] in
            let t = FakeTransport()
            self.transports.append(t)
            return t
        }
        store.start()
    }

    func addPlanned(_ names: [String]) {
        store.add(names.map { URL(fileURLWithPath: "/in/\($0)") })
        t.emit(.plan(id: t.lastPlanID!, jobs: names.map { PlannedJob(src: "/in/\($0)", dst: "/in_matting/\(($0 as NSString).deletingPathExtension).png") }, skipped: [], error: nil))
    }

    func testRunsOneJobAtATimeOnlyAfterReady() {
        addPlanned(["a.jpg", "b.jpg"])
        XCTAssertEqual(store.jobs.map(\.status), [.queued, .queued])
        XCTAssertNil(t.lastRun, "must not run before the models are loaded")
        XCTAssertEqual(store.selection, store.jobs[0].id)

        t.emit(.ready)
        XCTAssertEqual(t.lastRun?.src, "/in/a.jpg")
        XCTAssertEqual(store.jobs.map(\.status), [.processing, .queued])

        t.emit(.result(id: t.lastRun!.id, dst: "/in_matting/a.png", total: 2, stages: ["x": 2]))
        XCTAssertEqual(store.jobs[0].status, .done(total: 2, stages: ["x": 2]))
        XCTAssertEqual(t.lastRun?.src, "/in/b.jpg")

        t.emit(.error(id: t.lastRun!.id, message: "bad"))
        XCTAssertEqual(store.jobs[1].status, .failed("bad"))
        XCTAssertFalse(store.isProcessing)
        XCTAssertEqual(store.doneCount, 1)
        XCTAssertEqual(store.failedCount, 1)
    }

    func testSeparateDropsDoNotShareAnOutputFile() {
        t.emit(.ready)
        store.add([URL(fileURLWithPath: "/in/a.jpg")])
        t.emit(.plan(id: t.lastPlanID!, jobs: [PlannedJob(src: "/in/a.jpg", dst: "/out/a.png")], skipped: [], error: nil))
        store.add([URL(fileURLWithPath: "/in/a.png")])
        t.emit(.plan(id: t.lastPlanID!, jobs: [PlannedJob(src: "/in/a.png", dst: "/out/a.png")], skipped: [], error: nil))
        XCTAssertEqual(store.jobs.map(\.dst.path), ["/out/a.png", "/out/a_2.png"])
    }

    func testFilesAddedBeforeStartAreSentOnceTheWorkerRuns() {
        // Finder's "Open With" delivers files before applicationDidFinishLaunching starts the worker.
        let early = JobStore { [unowned self] in
            let t = FakeTransport()
            self.transports.append(t)
            return t
        }
        early.add([URL(fileURLWithPath: "/in/a.jpg")])
        XCTAssertNil(early.notice)
        early.start()
        guard case let .plan(_, inputs, _) = t.sent.first else { return XCTFail("expected the early plan") }
        XCTAssertEqual(inputs, ["/in/a.jpg"])
    }

    func testSkippedInputsBecomeANotice() {
        store.add([URL(fileURLWithPath: "/in/x.txt")])
        t.emit(.plan(id: t.lastPlanID!, jobs: [], skipped: [SkippedInput(path: "/in/x.txt", reason: "不支持的格式")], error: nil))
        XCTAssertEqual(store.notice, "未添加：x.txt（不支持的格式）")
        XCTAssertTrue(store.jobs.isEmpty)
    }

    func testStopAndResume() {
        addPlanned(["a.jpg", "b.jpg", "c.jpg"])
        t.emit(.ready)
        store.stopQueue()
        XCTAssertEqual(store.jobs.map(\.status), [.processing, .stopped, .stopped])
        t.emit(.result(id: t.lastRun!.id, dst: "", total: 1, stages: [:]))
        XCTAssertEqual(t.lastRun?.src, "/in/a.jpg", "nothing new is started after stop")
        store.resumeQueue()
        XCTAssertEqual(t.lastRun?.src, "/in/b.jpg")
    }

    func testOutputDirChangeReplansOnlyWaitingJobs() {
        addPlanned(["a.jpg", "b.jpg"])
        t.emit(.ready)
        store.setOutputDir(URL(fileURLWithPath: "/custom"))
        guard case let .plan(id, inputs, outputDir) = t.sent.last! else { return XCTFail("expected a plan") }
        XCTAssertEqual(inputs, ["/in/b.jpg"])
        XCTAssertEqual(outputDir, "/custom")
        t.emit(.plan(id: id, jobs: [PlannedJob(src: "/in/b.jpg", dst: "/custom/b.png")], skipped: [], error: nil))
        XCTAssertEqual(store.jobs.map(\.dst.path), ["/in_matting/a.png", "/custom/b.png"])
    }

    func testCrashFailsCurrentJobRestartsWorkerAndContinues() {
        addPlanned(["a.jpg", "b.jpg"])
        t.emit(.ready)
        t.exit(9)
        XCTAssertEqual(store.jobs[0].status, .failed("抠图进程意外退出（状态 9）"))
        XCTAssertEqual(transports.count, 2, "worker restarted")
        XCTAssertEqual(store.workerState, .starting)
        t.emit(.ready)
        XCTAssertEqual(t.lastRun?.src, "/in/b.jpg")
    }

    func testRepeatedCrashesGiveUp() {
        for _ in 0..<JobStore.maxConsecutiveCrashes { t.exit(1) }
        guard case .failed = store.workerState else { return XCTFail("expected failed, got \(store.workerState)") }
        XCTAssertEqual(transports.count, JobStore.maxConsecutiveCrashes)
        store.retryWorker()
        XCTAssertEqual(store.workerState, .starting)
    }

    func testFatalIsNotTreatedAsACrash() {
        t.emit(.fatal(message: "download failed"))
        t.exit(1)
        XCTAssertEqual(store.workerState, .failed("download failed"))
        XCTAssertEqual(transports.count, 1)
    }
}
