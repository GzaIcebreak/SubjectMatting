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
    private typealias Detail = @convention(c) (
        OpaquePointer?, UnsafePointer<UInt8>?, Int32, Int32, Int32, UnsafeMutablePointer<UInt8>?, UnsafeMutablePointer<UInt8>?,
        UnsafeMutableRawPointer?, UnsafeMutablePointer<CChar>?, Int
    ) -> Int32
    private typealias Destroy = @convention(c) (OpaquePointer?) -> Void
    private typealias Prepare = @convention(c) (
        OpaquePointer?, UnsafePointer<UInt8>?, Int32, Int32, Int32, UnsafeMutablePointer<CChar>?, Int
    ) -> Int32
    private typealias IsPlain = @convention(c) (UnsafePointer<UInt8>?, Int32, Int32, Int32) -> Int32
    /// smq_masks: { int count, width, height; uint8_t *data; int *owner; int used_key; }
    private struct CMasks {
        var count: Int32, width: Int32, height: Int32
        var data: UnsafeMutablePointer<UInt8>?
        var owner: UnsafeMutablePointer<Int32>?
        var usedKey: Int32
    }
    private typealias Candidates = @convention(c) (
        OpaquePointer?, UnsafePointer<UInt8>?, Int32, Int32, Int32, UnsafeRawPointer?, Int32, Int32,
        UnsafeMutablePointer<UnsafeMutableRawPointer?>?, UnsafeMutablePointer<CChar>?, Int
    ) -> Int32
    private typealias MasksFree = @convention(c) (UnsafeMutableRawPointer?) -> Void
    private typealias Remove = @convention(c) (
        OpaquePointer?, UnsafePointer<UInt8>?, Int32, Int32, Int32, UnsafeMutablePointer<UInt8>?, UnsafeRawPointer?, Int32,
        UnsafeMutablePointer<CChar>?, Int
    ) -> Int32
    private typealias MatteMask = @convention(c) (
        OpaquePointer?, UnsafePointer<UInt8>?, Int32, Int32, Int32, UnsafePointer<UInt8>?, UnsafeMutablePointer<UInt8>?,
        UnsafeMutableRawPointer?, UnsafeMutablePointer<CChar>?, Int
    ) -> Int32

    /// Candidate pieces of the subject (smq_candidates), at work size.
    public struct CandidateSet {
        public var masks: [[UInt8]]  // width*height each, 0/1
        public var owners: [Int]  // subject index per mask, -1 for colour-key pieces
        public var width: Int, height: Int
        public var usedKey: Bool
    }

    private let handle: UnsafeMutableRawPointer
    private let matteFn: Matte
    private let mattePlainFn: MattePlain
    private let prepareFn: Prepare
    private let isPlainFn: IsPlain
    private let destroyFn: Destroy
    private let candidatesFn: Candidates
    private let masksFreeFn: MasksFree
    private let removeFn: Remove
    private let matteMaskFn: MatteMask
    private let detailFn: Detail?
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
        prepareFn = try symbol("smq_prepare", as: Prepare.self)
        isPlainFn = try symbol("smq_is_plain", as: IsPlain.self)
        destroyFn = try symbol("smq_destroy", as: Destroy.self)
        candidatesFn = try symbol("smq_candidates", as: Candidates.self)
        masksFreeFn = try symbol("smq_masks_free", as: MasksFree.self)
        removeFn = try symbol("smq_remove", as: Remove.self)
        matteMaskFn = try symbol("smq_matte_mask", as: MatteMask.self)
        detailFn = dlsym(h, "smq_detail").map { unsafeBitCast($0, to: Detail.self) }
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

    /// Runs SAM3's image encoder now (while Qwen is being asked), so the next matte() on this image skips it.
    public func prepare(rgb: [UInt8], width: Int, height: Int) throws {
        var err = [CChar](repeating: 0, count: 512)
        let errLen = err.count
        if prepareFn(ctx, rgb, Int32(width), Int32(height), Int32(width * 3), &err, errLen) != 0 {
            throw Failure.other(String(cString: err))
        }
    }

    /// Whether mattePlain would take this image. Uses no models, so it can run on another thread.
    public func isPlain(rgb: [UInt8], width: Int, height: Int) -> Bool {
        isPlainFn(rgb, Int32(width), Int32(height), Int32(width * 3)) == 1
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

    private static func withCSubjects<R>(_ subjects: [QwenSubject], _ body: (UnsafeRawPointer?) -> R) -> R {
        let labels = subjects.map { strdup($0.label) }
        defer { labels.forEach { free($0) } }
        let c = zip(subjects, labels).map { s, label in
            CSubject(label: UnsafePointer(label), x1: Float(s.box[0]), y1: Float(s.box[1]), x2: Float(s.box[2]), y2: Float(s.box[3]))
        }
        return c.withUnsafeBytes { body($0.baseAddress) }
    }

    /// autoKey: on a solid background use the colour key's pieces instead of SAM3.
    public func candidates(rgb: [UInt8], width: Int, height: Int, subjects: [QwenSubject], autoKey: Bool) throws -> CandidateSet {
        var err = [CChar](repeating: 0, count: 512)
        let errLen = err.count
        var out: UnsafeMutableRawPointer?
        let status = Self.withCSubjects(subjects) { subs in
            candidatesFn(ctx, rgb, Int32(width), Int32(height), Int32(width * 3), subs, Int32(subjects.count), autoKey ? 1 : 0, &out, &err, errLen)
        }
        guard status == 0, let m = out else { throw Failure.other(String(cString: err)) }
        defer { masksFreeFn(m) }
        let c = m.load(as: CMasks.self)
        let plane = Int(c.width) * Int(c.height)
        var masks: [[UInt8]] = []
        var owners: [Int] = []
        for i in 0..<Int(c.count) {
            masks.append(Array(UnsafeBufferPointer(start: c.data! + i * plane, count: plane)))
            owners.append(Int(c.owner![i]))
        }
        return CandidateSet(masks: masks, owners: owners, width: Int(c.width), height: Int(c.height), usedKey: c.usedKey != 0)
    }

    /// Cuts the see-through regions out of a work-size mask.
    public func remove(rgb: [UInt8], width: Int, height: Int, mask: inout [UInt8], regions: [QwenSubject]) throws {
        if regions.isEmpty { return }
        var err = [CChar](repeating: 0, count: 512)
        let errLen = err.count
        let status = Self.withCSubjects(regions) { subs in
            removeFn(ctx, rgb, Int32(width), Int32(height), Int32(width * 3), &mask, subs, Int32(regions.count), &err, errLen)
        }
        if status != 0 { throw Failure.other(String(cString: err)) }
    }

    /// Trimap + ViTMatte for a work-size mask -> width*height*4 RGBA bytes.
    public func matteMask(rgb: [UInt8], width: Int, height: Int, mask: [UInt8]) throws -> (rgba: [UInt8], timings: Timings) {
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        var timings = Timings()
        var err = [CChar](repeating: 0, count: 512)
        let errLen = err.count
        let status = withUnsafeMutableBytes(of: &timings) { t in
            matteMaskFn(ctx, rgb, Int32(width), Int32(height), Int32(width * 3), mask, &rgba, t.baseAddress, &err, errLen)
        }
        switch status {
        case 0: return (rgba, timings)
        case 3: throw Failure.nothingSegmented(String(cString: err))
        default: throw Failure.other(String(cString: err))
        }
    }
    static func validateDetailInput(count:Int,width:Int,height:Int) throws {
        guard width>0,height>0,width<=Int(Int32.max)/3,height<=Int(Int32.max) else {throw Failure.other("无效图片尺寸")}
        let (pixels,o1)=width.multipliedReportingOverflow(by:height)
        let (_,o2)=pixels.multipliedReportingOverflow(by:4)
        let (bytes,o3)=pixels.multipliedReportingOverflow(by:3)
        guard !o1,!o2,!o3,count>=bytes else {throw Failure.other("RGB 数据不足或尺寸溢出")}
    }

    public func detail(rgb:[UInt8],width:Int,height:Int) throws -> (direct:[UInt8],refined:[UInt8],timings:Timings) {
        try Self.validateDetailInput(count:rgb.count,width:width,height:height)
        guard let fn=detailFn else {throw Failure.other("抠图库没有细节候选功能")}
        var direct=[UInt8](repeating:0,count:width*height*4), refined=direct
        var t=Timings(),err=[CChar](repeating:0,count:512)
        let errLen=err.count
        let rc=withUnsafeMutableBytes(of:&t) { ptr in
            fn(ctx,rgb,Int32(width),Int32(height),Int32(width*3),&direct,&refined,ptr.baseAddress,&err,errLen)
        }
        guard rc==0 else {throw Failure.other(String(cString:err))}
        return (direct,refined,t)
    }

}
