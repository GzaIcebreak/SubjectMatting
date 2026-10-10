import Foundation

// JSON-lines protocol spoken with worker.py; see the docstring there for the full description.

public struct PlannedJob: Codable, Equatable, Sendable {
    public var src: String
    public var dst: String

    public init(src: String, dst: String) {
        self.src = src
        self.dst = dst
    }
}

public struct SkippedInput: Codable, Equatable, Sendable {
    public var path: String
    public var reason: String

    public init(path: String, reason: String) {
        self.path = path
        self.reason = reason
    }
}

public enum WorkerEvent: Equatable, Sendable {
    case download(done: Int64, total: Int64)
    case loading
    case ready
    case fatal(message: String)
    case plan(id: Int, jobs: [PlannedJob], skipped: [SkippedInput], error: String?)
    case result(id: Int, dst: String, total: Double, stages: [String: Double])
    case error(id: Int, message: String)

    private struct Raw: Decodable {
        var type: String
        var id: Int?
        var done: Int64?
        var total: Double?
        var message: String?
        var jobs: [PlannedJob]?
        var skipped: [SkippedInput]?
        var error: String?
        var dst: String?
        var stages: [String: Double]?
    }

    public struct DecodingFailure: Error, Equatable {
        public var line: String
    }

    /// Decodes one line of worker output. Unknown event types are an error so protocol drift shows up in the log.
    public static func decode(_ line: Data) throws -> WorkerEvent {
        let raw = try JSONDecoder().decode(Raw.self, from: line)
        let failure = DecodingFailure(line: String(decoding: line, as: UTF8.self))
        switch raw.type {
        case "download":
            return .download(done: raw.done ?? 0, total: Int64(raw.total ?? 0))
        case "loading":
            return .loading
        case "ready":
            return .ready
        case "fatal":
            return .fatal(message: raw.message ?? "")
        case "plan":
            guard let id = raw.id else { throw failure }
            return .plan(id: id, jobs: raw.jobs ?? [], skipped: raw.skipped ?? [], error: raw.error)
        case "result":
            guard let id = raw.id, let dst = raw.dst else { throw failure }
            return .result(id: id, dst: dst, total: raw.total ?? 0, stages: raw.stages ?? [:])
        case "error":
            guard let id = raw.id else { throw failure }
            return .error(id: id, message: raw.message ?? "")
        default:
            throw failure
        }
    }
}

public enum WorkerCommand: Equatable, Sendable {
    case plan(id: Int, inputs: [String], outputDir: String?)
    case run(id: Int, src: String, dst: String)

    /// One JSON object terminated by a newline, ready to write to the worker's stdin.
    public func encodedLine() -> Data {
        var obj: [String: Any]
        switch self {
        case let .plan(id, inputs, outputDir):
            obj = ["id": id, "cmd": "plan", "inputs": inputs, "output_dir": outputDir ?? NSNull()]
        case let .run(id, src, dst):
            obj = ["id": id, "cmd": "run", "src": src, "dst": dst]
        }
        var data = try! JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys])
        data.append(0x0A)
        return data
    }
}
