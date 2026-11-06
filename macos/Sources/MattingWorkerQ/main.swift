// matting-worker-q: SubjectMatting Q's matting process. Same JSON-lines protocol as matting-worker; each `run`
// (optionally with "instruction") follows matting/pipeline_vlm.py: Qwen-VL names the subjects and the see-through
// regions, libsubject_matting_q proposes candidates (colour key or SAM3), a second Qwen model picks among them when
// that is ambiguous, the see-through regions are cut out, and ViTMatte mattes the result.
//
//   matting-worker-q [--lib libsubject_matting_q.dylib] [--models dir] [--precision fp16]
// Defaults: ../Frameworks/libsubject_matting_q.0.dylib and ../Resources/models relative to the executable.
// Environment: DASHSCOPE_API_KEY (required; the app passes it from the Keychain), SUBJECTMATTING_QWEN_MODEL,
// SUBJECTMATTING_QWEN_JUDGE_MODEL, DASHSCOPE_BASE_URL, SUBJECTMATTING_LOG (default ~/Library/Logs/SubjectMattingQ.log).
import Foundation
import CryptoKit
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

    /// The next item without removing it (nil when the queue is empty or closed).
    func peek() -> [String: Any]? {
        cond.lock()
        defer { cond.unlock() }
        return items.first ?? nil
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

/// One line per image in the log: what was done and how long each step took.
func log(_ name: String, _ img: ImageFiles.RGB, _ what: String, _ stages: [String: Double]) {
    let order = ["load", "prepare", "qwen", "key", "sam3", "judge", "remove", "matting", "detail", "review", "write"]
    let parts = order.compactMap { k in stages[k].map { "\(k) \(String(format: "%.2f", $0))" } }
    let total = stages.values.reduce(0, +)
    let line = "\(name) \(img.width)x\(img.height): \(what) | \(parts.joined(separator: " ")) | total \(String(format: "%.2f", total))s\n"
    FileHandle.standardError.write(line.data(using: .utf8)!)
}

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
    judgeModel: env["SUBJECTMATTING_QWEN_JUDGE_MODEL"] ?? QwenClient.defaultJudgeModel,
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

/// Work done ahead for a queued image while the current one is being matted: load it and ask Qwen. Network time
/// then overlaps with local compute.
final class Prefetch {
    final class Item {
        let loaded = DispatchSemaphore(value: 0)  // image loaded
        let done = DispatchSemaphore(value: 0)  // Qwen answered
        var img: Result<ImageFiles.RGB, Error> = .failure(ImageFiles.Failure(description: "图片没有读取"))
        var loadSeconds = 0.0
        var plain = false  // solid background in auto mode: the colour key proposes, SAM3 may not be needed
        var jpeg = Data()  // what Qwen saw; the judge sees it too
        var detection: Result<QwenDetection, Error> = .success(QwenDetection(subjects: []))
        var qwenSeconds = 0.0
        private let lock = NSLock()
        private var finished = false

        var isFinished: Bool {
            lock.lock()
            defer { lock.unlock() }
            return finished
        }

        func finish() {
            lock.lock()
            finished = true
            lock.unlock()
            done.signal()
        }
    }

    private var items: [String: Item] = [:]
    private let lock = NSLock()

    static func key(_ msg: [String: Any]) -> String { "\(msg["id"] ?? "")|\(msg["src"] ?? "")" }

    func start(_ msg: [String: Any]) {
        let key = Self.key(msg)
        lock.lock()
        if items[key] != nil {
            lock.unlock()
            return
        }
        let item = Item()
        items[key] = item
        lock.unlock()
        let instruction = (msg["instruction"] as? String).flatMap { $0.isEmpty ? nil : $0 }
        DispatchQueue.global(qos: .userInitiated).async {
            var signaledLoaded = false
            defer {
                if !signaledLoaded { item.loaded.signal() }
                item.finish()
            }
            guard let src = msg["src"] as? String else { return }
            let t0 = Date()
            do {
                let img = try ImageFiles.loadRGB(URL(fileURLWithPath: src))
                item.img = .success(img)
                item.loadSeconds = Date().timeIntervalSince(t0)
                item.plain = instruction == nil && engine.isPlain(rgb: img.pixels, width: img.width, height: img.height)
                item.loaded.signal()
                signaledLoaded = true
                let t1 = Date()
                item.detection = Result {
                    item.jpeg = try ImageFiles.jpegForUpload(img)
                    return try qwen.detect(jpeg: item.jpeg, width: img.width, height: img.height, instruction: instruction)
                }
                item.qwenSeconds = Date().timeIntervalSince(t1)
            } catch {
                item.img = .failure(error)
            }
        }
    }

    /// The work for msg, started now if it wasn't prefetched; returns once the image is loaded.
    func take(_ msg: [String: Any]) -> Item {
        start(msg)
        lock.lock()
        let item = items.removeValue(forKey: Self.key(msg))!
        lock.unlock()
        item.loaded.wait()
        return item
    }
}

let prefetch = Prefetch()

/// Ambiguous when some subject got several candidates, or the colour key found several pieces (matting/pipeline_vlm.py).
func needsJudge(_ owners: [Int]) -> Bool {
    Dictionary(grouping: owners, by: { $0 }).values.contains { $0.count > 1 }
}

while let msg = runs.take() {
    let id = msg["id"] ?? NSNull()
    let instruction = (msg["instruction"] as? String).flatMap { $0.isEmpty ? nil : $0 }
    do {
        guard let src = msg["src"] as? String, let dst = msg["dst"] as? String else {
            throw ImageFiles.Failure(description: "run 命令缺少 src/dst")
        }
        let name = URL(fileURLWithPath: src).lastPathComponent
        // Start on the next queued image (load + Qwen) while this one is matted.
        if let next = runs.peek() { prefetch.start(next) }
        // This image: prefetched if it was queued behind the previous one; otherwise start it now, so Qwen's
        // round trip overlaps with the SAM3 image encoding below.
        let item = prefetch.take(msg)
        let img = try item.img.get()
        let dstURL = URL(fileURLWithPath: dst)
        try FileManager.default.createDirectory(at: dstURL.deletingLastPathComponent(), withIntermediateDirectories: true)

        // While Qwen answers (unless it already has), encode the image for SAM3 - not needed on a solid background.
        let tPrepare = Date()
        if !item.plain && !item.isFinished { try engine.prepare(rgb: img.pixels, width: img.width, height: img.height) }
        let prepareSeconds = Date().timeIntervalSince(tPrepare)
        let tWait = Date()
        item.done.wait()
        let qwenWaitSeconds = Date().timeIntervalSince(tWait)
        let detection = try item.detection.get()
        if detection.subjects.isEmpty {
            throw ImageFiles.Failure(description: instruction.map { "图中没找到：\($0)" } ?? "没有识别到主体")
        }

        // Candidates, then Qwen's pick when ambiguous.
        let tSegment = Date()
        let cands = try engine.candidates(rgb: img.pixels, width: img.width, height: img.height, subjects: detection.subjects,
                                          autoKey: instruction == nil)
        let segmentSeconds = Date().timeIntervalSince(tSegment)
        if cands.masks.isEmpty { throw MattingEngineQ.Failure.nothingSegmented("SAM3 没有分割出对象") }
        var keep = Array(cands.masks.indices)
        var judgeSeconds: Double?
        var judgeNote = ""
        if !cands.usedKey && needsJudge(cands.owners) {
            let tJudge = Date()
            do {
                let sheet = try ImageFiles.candidateSheet(img, masks: cands.masks, maskWidth: cands.width, maskHeight: cands.height)
                let scale = min(1.0, 1024.0 / Double(max(img.width, img.height)))  // boxes in the uploaded image's pixels
                let shown = detection.subjects.map { QwenSubject(label: $0.label, box: $0.box.map { $0 * scale }) }
                let picked = try qwen.judge(jpeg: item.jpeg, sheet: try ImageFiles.jpegForUpload(sheet, longSide: 1152), subjects: shown,
                                            count: cands.masks.count, instruction: instruction)
                if !picked.isEmpty { keep = picked }
            } catch {
                judgeNote = " judge failed: \(error)"  // no second opinion: keep every candidate
            }
            judgeSeconds = Date().timeIntervalSince(tJudge)
        }
        var mask = [UInt8](repeating: 0, count: cands.width * cands.height)
        for i in keep {
            for (k, v) in cands.masks[i].enumerated() where v != 0 { mask[k] = 1 }
        }

        // See-through regions inside the subjects.
        let tRemove = Date()
        // Semantic boxes and colour similarity cannot prove background membership.
        // Preserve the mask until a pixel-evidence-based interior matte is available.
        let removeSeconds = Date().timeIntervalSince(tRemove)

        var (rgba, t) = try engine.matteMask(rgb: img.pixels, width: img.width, height: img.height, mask: mask)
        if let alpha = img.alpha {  // transparent in the file stays transparent
            for i in 0..<(img.width * img.height) { rgba[i * 4 + 3] = min(rgba[i * 4 + 3], alpha[i]) }
        }
        let originalRGBA = rgba
        var hybridDecision = HybridSelection.Decision(useDetail: false, reason: "组合流程关闭")
        var hybridReview: HybridReview?
        var hybridMetrics: HybridSelection.Metrics?
        var detailRGBA: [UInt8]?
        var directRGBA: [UInt8]?
        var rawDetailRGBA: [UInt8]?
        var detailSeconds = 0.0, reviewSeconds = 0.0
        var hybridError: String?
        if env["SUBJECTMATTING_HYBRID"] != "0" {
            let started = Date()
            do {
                let d = try engine.detail(rgb: img.pixels, width: img.width, height: img.height)
                if msg["audit_dir"] is String {directRGBA = d.direct; rawDetailRGBA = d.refined}
                var detail = HybridSelection.protectThinLines(direct: d.direct, refined: d.refined, width: img.width, height: img.height)
                if let alpha = img.alpha { for i in alpha.indices { detail[i*4+3] = min(detail[i*4+3], alpha[i]) } }
                detailRGBA = detail
                detailSeconds = Date().timeIntervalSince(started)
                let metrics = HybridSelection.metrics(original: originalRGBA, detail: detail, width: img.width, height: img.height, subjects: detection.subjects)
                hybridMetrics = metrics
                if metrics.severeLoss || metrics.targetExpansion {
                    hybridDecision = HybridSelection.choose(review: nil, severeLoss: metrics.severeLoss, targetExpansion: metrics.targetExpansion)
                } else if metrics.disagreement < 0.005 {
                    hybridDecision = HybridSelection.Decision(useDetail: false, reason: "候选几乎一致，保留原流程")
                } else {
                    let reviewStart = Date()
                    do {
                        hybridReview = try qwen.reviewHybrid(jpeg: item.jpeg, img: img, original: originalRGBA, detail: detail,
                                                            subjects: detection.subjects, instruction: instruction)
                    } catch { hybridError = "组合判断失败：\(error)" }
                    reviewSeconds = Date().timeIntervalSince(reviewStart)
                    hybridDecision = HybridSelection.choose(review: hybridReview, severeLoss: false)
                }
                if hybridDecision.useDetail { rgba = detail }
            } catch {
                detailSeconds = Date().timeIntervalSince(started)
                hybridError = "细节候选失败：\(error)"
                hybridDecision = HybridSelection.choose(review: nil, severeLoss: false)
            }
        }
        let tWrite = Date()
        try ImageFiles.writePNG(rgba: rgba, width: img.width, height: img.height, colorSpace: img.colorSpace, to: dstURL)
        // Opt-in audit for reproducible tests; ordinary app jobs create only the requested output PNG.
        if let auditPath = msg["audit_dir"] as? String {
            do {
                let runID = UUID().uuidString
                var dir = URL(fileURLWithPath: auditPath)
                if let files=try? FileManager.default.contentsOfDirectory(atPath:dir.path), files.contains("decision.json") || files.contains("original.png") {dir=dir.appendingPathComponent(runID)}
                try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
                for (file, pixels) in [("original.png", Optional(originalRGBA)), ("detail.png", detailRGBA),
                                       ("direct.png", directRGBA), ("detail_raw.png", rawDetailRGBA), ("final.png", Optional(rgba))] {
                    if let pixels { try ImageFiles.writePNG(rgba: pixels, width: img.width, height: img.height, colorSpace: img.colorSpace, to: dir.appendingPathComponent(file)) }
                }
                if let detailRGBA {
                    try ImageFiles.jpegForUpload(ImageFiles.compositeForReview(img,rgba:originalRGBA)).write(to:dir.appendingPathComponent("review_original.jpg"))
                    try ImageFiles.jpegForUpload(ImageFiles.compositeForReview(img,rgba:detailRGBA)).write(to:dir.appendingPathComponent("review_detail.jpg"))
                    for (index,jpeg) in try ImageFiles.hybridEvidence(img,original:originalRGBA,detail:detailRGBA).enumerated() {try jpeg.write(to:dir.appendingPathComponent("review_crop_\(index).jpg"))}
                }
                func object<T: Encodable>(_ value: T) throws -> Any { try JSONSerialization.jsonObject(with: JSONEncoder().encode(value)) }
                var record: [String: Any] = ["run_id":runID, "delivered":true, "dst":dst, "sha256":SHA256.hash(data:try Data(contentsOf:dstURL)).map{String(format:"%02x",$0)}.joined(), "decision": try object(hybridDecision), "subjects": detection.subjects.map { ["label": $0.label, "box": $0.box] as [String: Any] },
                                            "used_key": cands.usedKey, "kept_candidates": keep, "detail_seconds": detailSeconds, "review_seconds": reviewSeconds]
                if let hybridMetrics { record["metrics"] = try object(hybridMetrics) }
                if let hybridReview { record["review"] = try object(hybridReview) }
                if let hybridError { record["error"] = hybridError }
                try JSONSerialization.data(withJSONObject: record, options: [.sortedKeys, .prettyPrinted]).write(to: dir.appendingPathComponent("decision.json"))
            } catch { FileHandle.standardError.write("audit failed: \(error)\n".data(using: .utf8)!) }
        }
        let segmentKey = cands.usedKey ? "key" : "sam3"
        var stages: [String: Double] = ["load": item.loadSeconds, "prepare": prepareSeconds, "qwen": qwenWaitSeconds, segmentKey: segmentSeconds,
                                        "remove": removeSeconds, "matting": t.matting, "detail": detailSeconds, "review": reviewSeconds, "write": Date().timeIntervalSince(tWrite)]
        if let judgeSeconds { stages["judge"] = judgeSeconds }
        var what = "\(instruction ?? "auto") \(detection.subjects.map(\.label)) \(segmentKey) \(cands.masks.count) cand"
        what += " keep \(keep.map { $0 + 1 })"
        what += " hybrid=\(hybridDecision.useDetail ? "detail" : "original") \(hybridDecision.reason)"
        what += " \(hybridError ?? "") \(judgeNote)"
        what += " (Qwen \(String(format: "%.2f", item.qwenSeconds))s)"
        log(name, img, what, stages)
        // The app shows: 识别 (Qwen), 纯色背景 or 分割 (incl. SAM3's image encoding and the cut-outs), 判断, 边缘.
        var shown: [String: Double] = ["qwen": qwenWaitSeconds, segmentKey: prepareSeconds + segmentSeconds + removeSeconds, "matting": t.matting, "detail": detailSeconds, "review": reviewSeconds]
        if let judgeSeconds { shown["judge"] = judgeSeconds }
        send(["type": "result", "id": id, "dst": dst, "total": shown.values.reduce(0, +), "stages": shown, "hybrid": hybridDecision.useDetail ? "detail" : "original", "hybrid_reason": hybridDecision.reason])
    } catch let MattingEngineQ.Failure.nothingSegmented(message) {
        send(["type": "error", "id": id, "message": "没有在 Qwen 给出的位置分割出对象（\(message)）"])
    } catch {
        send(["type": "error", "id": id, "message": "\(error)"])
    }
}
