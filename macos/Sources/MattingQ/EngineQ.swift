import Foundation

/// libsubject_matting_q (cpp/, subject_matting_q.h), loaded with dlopen so the package builds without it.
public final class MattingEngineQ {
    public struct Timings {  // smq_timings
        public var sam3 = 0.0, matting = 0.0, total = 0.0
        public init() {}
    }

    public enum Failure: Error, CustomStringConvertible {
        case nothingSegmented(String)
        case other(String)

        public var description: String {
            switch self {
            case let .nothingSegmented(m), let .other(m): m
            }
        }
    }

    /// smq_subject: { const char *label; float x1, y1, x2, y2; }
    private struct CSubject {
        var label: UnsafePointer<CChar>?
        var x1: Float, y1: Float, x2: Float, y2: Float
    }

    private typealias Create = @convention(c) (
        UnsafePointer<CChar>?, UnsafePointer<CChar>?, Int32, UnsafeMutablePointer<CChar>?, Int
    ) -> OpaquePointer?
    private typealias Matte = @convention(c) (
        OpaquePointer?, UnsafePointer<UInt8>?, Int32, Int32, Int32, UnsafeRawPointer?, Int32, UnsafeMutablePointer<UInt8>?,
        UnsafeMutableRawPointer?, UnsafeMutablePointer<CChar>?, Int
    ) -> Int32
    private typealias MattePlain = @convention(c) (
        OpaquePointer?, UnsafePointer<UInt8>?, Int32, Int32, Int32, UnsafeMutablePointer<UInt8>?, UnsafeMutableRawPointer?,
        UnsafeMutablePointer<CChar>?, Int
    ) -> Int32
    private typealias Destroy = @convention(c) (OpaquePointer?) -> Void

    private let handle: UnsafeMutableRawPointer
    private let matteFn: Matte
    private let mattePlainFn: MattePlain
    private let destroyFn: Destroy
    private let ctx: OpaquePointer

    public init(library: URL, models: URL, precision: String = "fp16") throws {
        guard let h = dlopen(library.path, RTLD_NOW | RTLD_LOCAL) else {
            throw Failure.other("无法加载抠图库 \(library.path)：\(String(cString: dlerror()))")
        }
        func symbol<T>(_ name: String, as _: T.Type) throws -> T {
            guard let p = dlsym(h, name) else { throw Failure.other("抠图库缺少 \(name)") }
            return unsafeBitCast(p, to: T.self)
        }
        handle = h
        let create = try symbol("smq_create", as: Create.self)
        matteFn = try symbol("smq_matte", as: Matte.self)
        mattePlainFn = try symbol("smq_matte_plain", as: MattePlain.self)
        destroyFn = try symbol("smq_destroy", as: Destroy.self)
        var err = [CChar](repeating: 0, count: 512)
        let errLen = err.count
        guard let c = create(models.path, precision, 0, &err, errLen) else {
            throw Failure.other("模型加载失败：\(String(cString: err))")
        }
        ctx = c
    }

    deinit {
        destroyFn(ctx)
    }

    /// Auto mode on a solid background (no Qwen, no SAM3): nil when the background is not a plain colour.
    /// timings.sam3 holds the keying time.
    public func mattePlain(rgb: [UInt8], width: Int, height: Int) throws -> (rgba: [UInt8], timings: Timings)? {
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        var timings = Timings()
        var err = [CChar](repeating: 0, count: 512)
        let errLen = err.count
        let status = withUnsafeMutableBytes(of: &timings) { t in
            mattePlainFn(ctx, rgb, Int32(width), Int32(height), Int32(width * 3), &rgba, t.baseAddress, &err, errLen)
        }
        switch status {
        case 0: return (rgba, timings)
        case 4: return nil
        default: throw Failure.other(String(cString: err))
        }
    }

    /// subjects: label + box in pixels of the image. Returns width*height*4 RGBA bytes.
    public func matte(rgb: [UInt8], width: Int, height: Int, subjects: [QwenSubject]) throws -> (rgba: [UInt8], timings: Timings) {
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        var timings = Timings()
        var err = [CChar](repeating: 0, count: 512)
        let errLen = err.count
        let labels = subjects.map { strdup($0.label) }
        defer { labels.forEach { free($0) } }
        let cSubjects = zip(subjects, labels).map { s, label in
            CSubject(label: UnsafePointer(label), x1: Float(s.box[0]), y1: Float(s.box[1]), x2: Float(s.box[2]), y2: Float(s.box[3]))
        }
        let status = cSubjects.withUnsafeBytes { subs in
            withUnsafeMutableBytes(of: &timings) { t in
                matteFn(ctx, rgb, Int32(width), Int32(height), Int32(width * 3), subs.baseAddress, Int32(subjects.count), &rgba,
                        t.baseAddress, &err, errLen)
            }
        }
        switch status {
        case 0: return (rgba, timings)
        case 3: throw Failure.nothingSegmented(String(cString: err))
        default: throw Failure.other(String(cString: err))
        }
    }
}
