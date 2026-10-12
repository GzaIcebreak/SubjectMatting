// matting-worker-q: SubjectMatting Q's matting process. Same JSON-lines protocol as matting-worker; each `run`
// (optionally with "instruction") asks Qwen-VL what to cut out, then libsubject_matting_q segments and mattes it.
//
//   matting-worker-q [--lib libsubject_matting_q.dylib] [--models dir] [--precision fp16]
// Defaults: ../Frameworks/libsubject_matting_q.0.dylib and ../Resources/models relative to the executable.
// Environment: DASHSCOPE_API_KEY (required; the app passes it from the Keychain), SUBJECTMATTING_QWEN_MODEL,
// DASHSCOPE_BASE_URL, SUBJECTMATTING_LOG (default ~/Library/Logs/SubjectMattingQ.log).
import Foundation
import MattingCore
import MattingQ

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
        ?? FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Logs/SubjectMattingQ.log")
    try? FileManager.default.createDirectory(at: log.deletingLastPathComponent(), withIntermediateDirectories: true)
    let fd = open(log.path, O_WRONLY | O_APPEND | O_CREAT, 0o644)
    if fd >= 0 {
        dup2(fd, 2)
        close(fd)
    }
    let stamp = ISO8601DateFormatter().string(from: Date())
    FileHandle.standardError.write("--- matting-worker-q start \(stamp) ---\n".data(using: .utf8)!)
}

let exeDir = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath().deletingLastPathComponent()
var libraryURL = exeDir.appendingPathComponent("../Frameworks/libsubject_matting_q.0.dylib").standardizedFileURL
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
signal(SIGPIPE, SIG_IGN)

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

Thread {
    while let line = readLine(strippingNewline: true) {
        guard !line.isEmpty, let data = line.data(using: .utf8),
              let msg = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { continue }
        switch msg["cmd"] as? String {
        case "plan":
            let (jobs, skipped) = JobPlanner.plan(msg["inputs"] as? [String] ?? [], outputDir: msg["output_dir"] as? String)
            send([
                "type": "plan", "id": msg["id"] ?? NSNull(),
                "jobs": jobs.map { ["src": $0.src, "dst": $0.dst] },
                "skipped": skipped.map { ["path": $0.path, "reason": $0.reason] },
                "error": NSNull(),
            ])
        case "run": runs.put(msg)
        default: FileHandle.standardError.write("unknown command: \(line.prefix(200))\n".data(using: .utf8)!)
        }
    }
    runs.put(nil)
}.start()

send(["type": "loading"])
let env = ProcessInfo.processInfo.environment
guard let apiKey = env["DASHSCOPE_API_KEY"], !apiKey.isEmpty else {
    send(["type": "fatal", "message": "还没有设置阿里云百炼 API Key：请在「设置」里填写后点「重试」"])
    exit(1)
}
let qwen = QwenClient(
    apiKey: apiKey,
    model: env["SUBJECTMATTING_QWEN_MODEL"] ?? QwenClient.defaultModel,
    baseURL: env["DASHSCOPE_BASE_URL"].flatMap(URL.init(string:)) ?? QwenClient.defaultBaseURL
)
let engine: MattingEngineQ
do {
    engine = try MattingEngineQ(library: libraryURL, models: modelsURL, precision: precision)
} catch {
    send(["type": "fatal", "message": "\(error)"])
    exit(1)
}
send(["type": "ready"])

while let msg = runs.take() {
    let id = msg["id"] ?? NSNull()
    let instruction = (msg["instruction"] as? String).flatMap { $0.isEmpty ? nil : $0 }
    do {
        guard let src = msg["src"] as? String, let dst = msg["dst"] as? String else {
            throw ImageFiles.Failure(description: "run 命令缺少 src/dst")
        }
        let img = try ImageFiles.loadRGB(URL(fileURLWithPath: src))
        let t0 = Date()
        let subjects = try qwen.detect(jpeg: try ImageFiles.jpegForUpload(img), width: img.width, height: img.height, instruction: instruction)
        let qwenSeconds = Date().timeIntervalSince(t0)
        FileHandle.standardError.write("\(URL(fileURLWithPath: src).lastPathComponent): \(subjects.map(\.label)) in \(String(format: "%.1f", qwenSeconds))s\n".data(using: .utf8)!)
        if subjects.isEmpty {
            throw ImageFiles.Failure(description: instruction.map { "图中没找到：\($0)" } ?? "没有识别到主体")
        }
        let (rgba, t) = try engine.matte(rgb: img.pixels, width: img.width, height: img.height, subjects: subjects)
        let dstURL = URL(fileURLWithPath: dst)
        try FileManager.default.createDirectory(at: dstURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try ImageFiles.writePNG(rgba: rgba, width: img.width, height: img.height, colorSpace: img.colorSpace, to: dstURL)
        send([
            "type": "result", "id": id, "dst": dst, "total": qwenSeconds + t.total,
            "stages": ["qwen": qwenSeconds, "sam3": t.sam3, "matting": t.matting],
        ])
    } catch let MattingEngineQ.Failure.nothingSegmented(message) {
        send(["type": "error", "id": id, "message": "没有在 Qwen 给出的位置分割出对象（\(message)）"])
    } catch {
        send(["type": "error", "id": id, "message": "\(error)"])
    }
}
