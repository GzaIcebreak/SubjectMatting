import Foundation

/// Dropped files/folders -> (source, destination) pairs. Same rules as matting.batch.plan_jobs in Python
/// (the CLI uses that one): a folder contributes its images (non-recursive), a loose image is grouped with its
/// parent folder, each group goes to `outputDir` or to "<folder>_matting" next to the folder, same-stem inputs
/// become <stem>_<ext>.png, and names that still collide in a shared output folder get _2, _3, ...
public enum JobPlanner {
    public static let imageExtensions: Set<String> = ["jpg", "jpeg", "png", "webp", "bmp"]

    static func isImage(_ url: URL) -> Bool { imageExtensions.contains(url.pathExtension.lowercased()) }

    /// Like Python's Path.resolve(): symlinks in the existing part of the path resolved (realpath), the rest kept.
    /// (Foundation's resolvingSymlinksInPath also strips /private, which Python doesn't.)
    static func resolved(_ url: URL) -> URL {
        let u = url.standardizedFileURL
        if let r = realpath(u.path, nil) {
            defer { free(r) }
            return URL(fileURLWithPath: String(cString: r))
        }
        let parent = u.deletingLastPathComponent()
        return parent.path == u.path ? u : resolved(parent).appendingPathComponent(u.lastPathComponent)
    }

    /// Images directly inside `folder`, sorted by path.
    public static func listImages(_ folder: URL) -> [URL] {
        let names = (try? FileManager.default.contentsOfDirectory(atPath: folder.path)) ?? []
        return names.sorted { $0.unicodeScalars.lexicographicallyPrecedes($1.unicodeScalars) }
            .map { folder.appendingPathComponent($0) }
            .filter { url in
                var isDir: ObjCBool = false
                return FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir) && !isDir.boolValue && isImage(url)
            }
    }

    /// /photos -> /photos_matting (a sibling, so results never land inside the input folder).
    public static func defaultOutputDir(_ folder: URL) -> URL {
        let f = resolved(folder)
        let name = f.lastPathComponent.isEmpty || f.path == "/" ? "images" : f.lastPathComponent
        return f.deletingLastPathComponent().appendingPathComponent("\(name)_matting")
    }

    /// <stem>.png, or <stem>_<ext>.png when several inputs share a stem (a.jpg + a.png).
    public static func outputNames(_ files: [URL]) -> [String] {
        var stems: [String: Int] = [:]
        for f in files { stems[f.deletingPathExtension().lastPathComponent, default: 0] += 1 }
        return files.map { f in
            let stem = f.deletingPathExtension().lastPathComponent
            return stems[stem]! > 1 ? "\(stem)_\(f.pathExtension.lowercased()).png" : "\(stem).png"
        }
    }

    public static func plan(_ inputs: [String], outputDir: String?) -> (jobs: [PlannedJob], skipped: [SkippedInput]) {
        let fm = FileManager.default
        let out = outputDir.map { URL(fileURLWithPath: $0) }
        var groupOrder: [URL] = []
        var groups: [URL: [URL]] = [:]
        var skipped: [SkippedInput] = []
        var seen = Set<URL>()
        for raw in inputs {
            let p = URL(fileURLWithPath: raw)
            var isDir: ObjCBool = false
            let folder: URL
            let files: [URL]
            if fm.fileExists(atPath: raw, isDirectory: &isDir), isDir.boolValue {
                folder = p
                files = listImages(p)
                if files.isEmpty { skipped.append(SkippedInput(path: raw, reason: "文件夹里没有图片")) }
            } else if fm.fileExists(atPath: raw) {
                guard isImage(p) else {
                    skipped.append(SkippedInput(path: raw, reason: "不支持的格式"))
                    continue
                }
                folder = p.deletingLastPathComponent()
                files = [p]
            } else {
                skipped.append(SkippedInput(path: raw, reason: "文件不存在"))
                continue
            }
            for f in files where seen.insert(resolved(f)).inserted {
                let key = resolved(folder)
                if groups[key] == nil { groupOrder.append(key) }
                groups[key, default: []].append(f)
            }
        }

        var jobs: [PlannedJob] = []
        var used = Set<URL>()
        for folder in groupOrder {
            let files = groups[folder]!
            let dstDir = out ?? defaultOutputDir(folder)
            if resolved(dstDir) == folder {
                skipped += files.map { SkippedInput(path: $0.path, reason: "输出文件夹不能和输入文件夹相同") }
                continue
            }
            for (f, name) in zip(files, outputNames(files)) {
                var dst = dstDir.appendingPathComponent(name)
                var n = 1
                while used.contains(resolved(dst)) {
                    n += 1
                    dst = dstDir.appendingPathComponent("\((name as NSString).deletingPathExtension)_\(n).png")
                }
                used.insert(resolved(dst))
                jobs.append(PlannedJob(src: f.path, dst: dst.path))
            }
        }
        return (jobs, skipped)
    }
}
