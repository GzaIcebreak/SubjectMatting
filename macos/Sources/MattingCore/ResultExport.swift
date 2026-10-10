import Foundation

/// Copies finished results somewhere else. Results are always auto-saved to job.dst first; this is the
/// "Save As…" / "Export All…" on top of that.
public enum ResultExport {
    public static func finishedResults(_ jobs: [Job]) -> [URL] {
        jobs.compactMap { job in
            if case .done = job.status { return job.dst }
            return nil
        }
    }

    /// Copies one result to `destination`, replacing an existing file there (the save panel already asked).
    public static func save(_ result: URL, to destination: URL) throws {
        let fm = FileManager.default
        guard result.standardizedFileURL != destination.standardizedFileURL else { return }
        if fm.fileExists(atPath: destination.path) {
            try fm.removeItem(at: destination)
        }
        try fm.copyItem(at: result, to: destination)
    }

    /// Copies every result into `folder` without overwriting anything: name.png, name_2.png, ...
    /// Returns how many were copied and the errors for the rest.
    @discardableResult
    public static func exportAll(_ results: [URL], to folder: URL) -> (copied: Int, errors: [String]) {
        let fm = FileManager.default
        var copied = 0
        var errors: [String] = []
        for result in results {
            let stem = result.deletingPathExtension().lastPathComponent
            var target = folder.appendingPathComponent(result.lastPathComponent)
            var n = 1
            while fm.fileExists(atPath: target.path) {
                n += 1
                target = folder.appendingPathComponent("\(stem)_\(n).png")
            }
            do {
                try fm.copyItem(at: result, to: target)
                copied += 1
            } catch {
                errors.append("\(result.lastPathComponent)：\(error.localizedDescription)")
            }
        }
        return (copied, errors)
    }
}
