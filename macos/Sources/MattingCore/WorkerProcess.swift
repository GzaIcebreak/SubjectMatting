import Foundation

/// How JobStore talks to the worker. Callbacks are delivered on the main queue.
public protocol WorkerTransport: AnyObject {
    var onEvent: ((WorkerEvent) -> Void)? { get set }
    /// Called once when the worker process has exited, with its exit status.
    var onExit: ((Int32) -> Void)? { get set }
    func start() throws
    func send(_ command: WorkerCommand)
    /// Closes the worker's stdin and terminates it.
    func stop()
}

/// Runs the bundled matting-worker (or $SUBJECTMATTING_WORKER during development) as a child process.
public final class WorkerProcess: WorkerTransport {
    public var onEvent: ((WorkerEvent) -> Void)?
    public var onExit: ((Int32) -> Void)?

    private let process = Process()
    private let stdin = Pipe()
    private let stdout = Pipe()
    private var buffer = Data()

    public init() {}

    /// $SUBJECTMATTING_WORKER is a command line such as ".venv/bin/python worker.py", run from the current
    /// directory; otherwise the PyInstaller build inside the app bundle.
    static func command() -> (URL, [String]) {
        if let dev = ProcessInfo.processInfo.environment["SUBJECTMATTING_WORKER"], !dev.isEmpty {
            let parts = dev.split(separator: " ").map(String.init)
            let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            return (URL(fileURLWithPath: parts[0], relativeTo: cwd).absoluteURL, Array(parts.dropFirst()))
        }
        let exe = Bundle.main.resourceURL!.appendingPathComponent("worker/matting-worker")
        return (exe, [])
    }

    public func start() throws {
        // Writing to a worker that just died must fail with EPIPE, not kill the app with SIGPIPE.
        signal(SIGPIPE, SIG_IGN)
        let (exe, args) = Self.command()
        process.executableURL = exe
        process.arguments = args
        process.standardInput = stdin
        process.standardOutput = stdout
        // stderr is not read: the worker sends its own output to the log file. Inherit it for development.
        process.terminationHandler = { [weak self] p in
            let status = p.terminationStatus
            DispatchQueue.main.async {
                self?.stdout.fileHandleForReading.readabilityHandler = nil
                self?.onExit?(status)
            }
        }
        stdout.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let chunk = handle.availableData
            DispatchQueue.main.async { self?.receive(chunk) }
        }
        try process.run()
    }

    private func receive(_ chunk: Data) {
        buffer.append(chunk)
        while let nl = buffer.firstIndex(of: 0x0A) {
            let line = buffer[buffer.startIndex..<nl]
            buffer.removeSubrange(buffer.startIndex...nl)
            guard !line.isEmpty else { continue }
            do {
                onEvent?(try WorkerEvent.decode(Data(line)))
            } catch {
                NSLog("SubjectMatting: unreadable worker output: %@", String(decoding: line, as: UTF8.self))
            }
        }
    }

    public func send(_ command: WorkerCommand) {
        guard process.isRunning else { return }
        try? stdin.fileHandleForWriting.write(contentsOf: command.encodedLine())
    }

    public func stop() {
        try? stdin.fileHandleForWriting.close()
        if process.isRunning { process.terminate() }
    }
}
