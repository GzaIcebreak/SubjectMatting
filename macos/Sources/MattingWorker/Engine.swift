import Foundation

/// libsubject_matting (cpp/), loaded with dlopen so the Swift package builds and tests without it.
/// The function types mirror cpp/include/subject_matting.h.
final class MattingEngine {
    struct Timings {  // sm_timings
        var saliency = 0.0, sam2 = 0.0, matting = 0.0, total = 0.0
    }

    struct Failure: Error, CustomStringConvertible {
        var description: String
    }

    private typealias Create = @convention(c) (
        UnsafePointer<CChar>?, UnsafePointer<CChar>?, Int32, UnsafeMutablePointer<CChar>?, Int
    ) -> OpaquePointer?
    private typealias Matte = @convention(c) (
        OpaquePointer?, UnsafePointer<UInt8>?, Int32, Int32, Int32, UnsafeMutablePointer<UInt8>?,
        UnsafeMutableRawPointer?, UnsafeMutablePointer<CChar>?, Int
    ) -> Int32
    private typealias Destroy = @convention(c) (OpaquePointer?) -> Void

    private let handle: UnsafeMutableRawPointer
    private let matteFn: Matte
    private let destroyFn: Destroy
    private let ctx: OpaquePointer

    init(library: URL, models: URL, precision: String = "fp16") throws {
        guard let h = dlopen(library.path, RTLD_NOW | RTLD_LOCAL) else {
            throw Failure(description: "无法加载抠图库 \(library.path)：\(String(cString: dlerror()))")
        }
        func symbol<T>(_ name: String, as _: T.Type) throws -> T {
            guard let p = dlsym(h, name) else { throw Failure(description: "抠图库缺少 \(name)") }
            return unsafeBitCast(p, to: T.self)
        }
        handle = h
        let create = try symbol("sm_create", as: Create.self)
        matteFn = try symbol("sm_matte", as: Matte.self)
        destroyFn = try symbol("sm_destroy", as: Destroy.self)
        var err = [CChar](repeating: 0, count: 512)
        let errLen = err.count
        guard let c = create(models.path, precision, 0, &err, errLen) else {
            throw Failure(description: "模型加载失败：\(String(cString: err))")
        }
        ctx = c
    }

    deinit {
        destroyFn(ctx)
    }

    /// rgb: width*height*3 bytes -> width*height*4 RGBA bytes.
    func matte(rgb: [UInt8], width: Int, height: Int) throws -> (rgba: [UInt8], timings: Timings) {
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        var timings = Timings()
        var err = [CChar](repeating: 0, count: 512)
        let errLen = err.count
        let status = withUnsafeMutableBytes(of: &timings) { t in
            matteFn(ctx, rgb, Int32(width), Int32(height), Int32(width * 3), &rgba, t.baseAddress, &err, errLen)
        }
        if status != 0 { throw Failure(description: String(cString: err)) }
        return (rgba, timings)
    }
}
