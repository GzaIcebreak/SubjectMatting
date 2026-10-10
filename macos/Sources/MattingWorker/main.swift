// matting-worker: the app's matting process. Same JSON-lines protocol as worker.py (see its docstring), backed by
// the C++ library instead of PyTorch. The models ship inside the app, so it never sends `download`.
//
//   matting-worker [--lib path/to/libsubject_matting.dylib] [--models dir] [--precision fp16]
// Defaults: ../Frameworks/libsubject_matting.0.dylib and ../Resources/models relative to the executable.
// SUBJECTMATTING_LOG overrides the log file (default ~/Library/Logs/SubjectMatting.log).
import Foundation
import MattingCore

// MARK: Output

/// The protocol owns stdout; fd 2 (ONNX Runtime messages, crashes) goes to the log file.
let protocolOut = FileHandle.standardOutput
let outLock = NSLock()

func send(_ fields: [String: Any]) {
    guard var data = try? JSONSerialization.data(withJSONObject: fields, options: [.sortedKeys]) else { return }
    data.append(0x0A)
    outLock.lock()
    protocolOut.write(data)
    outLock.unlock()
}

func redirectStderrToLog() {
    let env = ProcessInfo.processInfo.environment["SUBJECTMATTING_LOG"]
    let log = env.map { URL(fileURLWithPath: $0) }
        ?? FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Logs/SubjectMatting.log")
    try? FileManager.default.createDirectory(at: log.deletingLastPathComponent(), withIntermediateDirectories: true)
    let fd = open(log.path, O_WRONLY | O_APPEND | O_CREAT, 0o644)
    if fd >= 0 {
        dup2(fd, 2)
        close(fd)
    }
    let stamp = ISO8601DateFormatter().string(from: Date())
    FileHandle.standardError.write("--- matting-worker start \(stamp) ---\n".data(using: .utf8)!)
}

// MARK: Arguments

let exeDir = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath().deletingLastPathComponent()
var libraryURL = exeDir.appendingPathComponent("../Frameworks/libsubject_matting.0.dylib").standardizedFileURL
var modelsURL = exeDir.appendingPathComponent("../Resources/models").standardizedFileURL
var precision = "fp16"
var args = CommandLine.arguments.dropFirst().makeIterator()
while let a = args.next() {
    switch a {
    case "--lib": libraryURL = URL(fileURLWithPath: args.next() ?? "")
    case "--models": modelsURL = URL(fileURLWithPath: args.next() ?? "")
    case "--precision": precision = args.next() ?? precision
    default:
        FileHandle.standardError.write("unknown argument \(a)\n".data(using: .utf8)!)
        exit(2)
    }
}

redirectStderrToLog()
signal(SIGPIPE, SIG_IGN)  // the app went away: writes fail instead of killing us; stdin EOF ends the loop

// MARK: Commands

/// `run` commands for the main thread, in order; nil = stdin closed.
final class RunQueue {
    private var items: [[String: Any]?] = []
    private let cond = NSCondition()

    func put(_ item: [String: Any]?) {
        cond.lock()
        items.append(item)
        cond.signal()
        cond.unlock()
    }

    func take() -> [String: Any]? {
        cond.lock()
        while items.isEmpty { cond.wait() }
        let item = items.removeFirst()
        cond.unlock()
        return item
    }
}

let runs = RunQueue()

func handlePlan(_ msg: [String: Any]) {
    let inputs = msg["inputs"] as? [String] ?? []
    let outputDir = msg["output_dir"] as? String
    let (jobs, skipped) = JobPlanner.plan(inputs, outputDir: outputDir)
    send([
        "type": "plan",
        "id": msg["id"] ?? NSNull(),
        "jobs": jobs.map { ["src": $0.src, "dst": $0.dst] },
        "skipped": skipped.map { ["path": $0.path, "reason": $0.reason] },
        "error": NSNull(),
    ])
}

/// stdin reader: answers `plan` right away (even while models load), queues `run` for the main thread.
Thread {
    while let line = readLine(strippingNewline: true) {
        guard !line.isEmpty, let data = line.data(using: .utf8),
              let msg = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { continue }
        switch msg["cmd"] as? String {
        case "plan": handlePlan(msg)
        case "run": runs.put(msg)
        default: FileHandle.standardError.write("unknown command: \(line.prefix(200))\n".data(using: .utf8)!)
        }
    }
    runs.put(nil)
}.start()

// MARK: Main

send(["type": "loading"])
let engine: MattingEngine
do {
    engine = try MattingEngine(library: libraryURL, models: modelsURL, precision: precision)
} catch {
    send(["type": "fatal", "message": "\(error)"])
    exit(1)
}
send(["type": "ready"])

while let msg = runs.take() {
    let id = msg["id"] ?? NSNull()
    do {
        guard let src = msg["src"] as? String, let dst = msg["dst"] as? String else {
            throw ImageFiles.Failure(description: "run 命令缺少 src/dst")
        }
        let img = try ImageFiles.loadRGB(URL(fileURLWithPath: src))
        let (rgba, t) = try engine.matte(rgb: img.pixels, width: img.width, height: img.height)
        let dstURL = URL(fileURLWithPath: dst)
        try FileManager.default.createDirectory(at: dstURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try ImageFiles.writePNG(rgba: rgba, width: img.width, height: img.height, colorSpace: img.colorSpace, to: dstURL)
        send([
            "type": "result", "id": id, "dst": dst, "total": t.total,
            "stages": ["saliency": t.saliency, "sam2": t.sam2, "matting": t.matting],
        ])
    } catch {
        send(["type": "error", "id": id, "message": "\(error)"])
    }
}
