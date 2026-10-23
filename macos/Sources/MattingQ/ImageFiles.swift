import CoreGraphics
import CoreText
import Foundation
import ImageIO
import UniformTypeIdentifiers

/// Image file I/O with ImageIO: any format macOS reads (jpg, png, webp, bmp, heic, ...), EXIF orientation applied.
/// Same as the SAM2 worker's (Sources/MattingWorker/ImageFiles.swift), plus the scaled-down JPEG sent to Qwen.
public enum ImageFiles {
    public struct Failure: Error, CustomStringConvertible {
        public var description: String
        public init(description: String) { self.description = description }
    }

    public struct RGB {
        public var pixels: [UInt8]  // width*height*3
        public var width: Int
        public var height: Int
        public var colorSpace: CGColorSpace
        /// The file's own alpha (width*height, upright), nil when it has none or is fully opaque. What is
        /// transparent there stays transparent in the result.
        public var alpha: [UInt8]? = nil
    }

    /// Like matting.common.load_rgb: upright, transparent areas composited onto white. Pixels stay in the
    /// file's own RGB color space (no conversion), and the result PNG is tagged with that space.
    public static func loadRGB(_ url: URL) throws -> RGB {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil), CGImageSourceGetCount(source) > 0,
              let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let raw = CGImageSourceCreateImageAtIndex(source, 0, [kCGImageSourceShouldCacheImmediately: true] as CFDictionary)
        else { throw Failure(description: "无法读取图片（格式不支持或文件已损坏）") }
        // Decoding the image itself and applying the EXIF orientation while drawing is several times faster on large
        // photos than asking ImageIO for a full-size "thumbnail" with the transform (which resamples).
        let orientation = (props[kCGImagePropertyOrientation] as? NSNumber)?.intValue ?? 1
        let rw = raw.width, rh = raw.height
        let swaps = (5...8).contains(orientation)
        let w = swaps ? rh : rw, h = swaps ? rw : rh
        let space = raw.colorSpace.flatMap { $0.model == .rgb ? $0 : nil } ?? CGColorSpace(name: CGColorSpace.sRGB)!
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        let drawn = rgba.withUnsafeMutableBytes { buf -> Bool in
            guard let ctx = CGContext(data: buf.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                      space: space, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
            else { return false }
            ctx.setFillColor(red: 1, green: 1, blue: 1, alpha: 1)
            ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
            ctx.interpolationQuality = .none
            ctx.concatenate(orientationTransform(orientation, width: CGFloat(rw), height: CGFloat(rh)))
            ctx.draw(raw, in: CGRect(x: 0, y: 0, width: rw, height: rh))
            return true
        }
        guard drawn else { throw Failure(description: "无法解码图片") }
        var rgb = [UInt8](repeating: 0, count: w * h * 3)
        for i in 0..<(w * h) {
            rgb[i * 3] = rgba[i * 4]
            rgb[i * 3 + 1] = rgba[i * 4 + 1]
            rgb[i * 3 + 2] = rgba[i * 4 + 2]
        }
        return RGB(pixels: rgb, width: w, height: h, colorSpace: space, alpha: alphaChannel(raw, orientation: orientation))
    }

    static func alphaChannel(_ raw: CGImage, orientation: Int) -> [UInt8]? {
        switch raw.alphaInfo {
        case .none, .noneSkipFirst, .noneSkipLast: return nil
        default: break
        }
        let rw = raw.width, rh = raw.height
        let swaps = (5...8).contains(orientation)
        let w = swaps ? rh : rw, h = swaps ? rw : rh
        // Drawn onto transparent black; the alpha channel is the file's alpha.
        let space = CGColorSpace(name: CGColorSpace.sRGB)!
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        let drawn = rgba.withUnsafeMutableBytes { buf -> Bool in
            guard let ctx = CGContext(data: buf.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4, space: space,
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
            else { return false }
            ctx.interpolationQuality = .none
            ctx.concatenate(orientationTransform(orientation, width: CGFloat(rw), height: CGFloat(rh)))
            ctx.draw(raw, in: CGRect(x: 0, y: 0, width: rw, height: rh))
            return true
        }
        let alpha = stride(from: 3, to: rgba.count, by: 4).map { rgba[$0] }
        guard drawn, alpha.contains(where: { $0 < 255 }) else { return nil }
        return alpha
    }

    /// Maps the stored image (width x height, CoreGraphics' y-up space) to its display orientation (EXIF 1...8).
    static func orientationTransform(_ orientation: Int, width w: CGFloat, height h: CGFloat) -> CGAffineTransform {
        switch orientation {
        case 2: return CGAffineTransform(a: -1, b: 0, c: 0, d: 1, tx: w, ty: 0)  // mirror horizontally
        case 3: return CGAffineTransform(a: -1, b: 0, c: 0, d: -1, tx: w, ty: h)  // rotate 180
        case 4: return CGAffineTransform(a: 1, b: 0, c: 0, d: -1, tx: 0, ty: h)  // mirror vertically
        case 5: return CGAffineTransform(a: 0, b: -1, c: -1, d: 0, tx: h, ty: w)  // transpose
        case 6: return CGAffineTransform(a: 0, b: -1, c: 1, d: 0, tx: 0, ty: w)  // rotate 90 clockwise
        case 7: return CGAffineTransform(a: 0, b: 1, c: 1, d: 0, tx: 0, ty: 0)  // transverse
        case 8: return CGAffineTransform(a: 0, b: 1, c: -1, d: 0, tx: h, ty: 0)  // rotate 90 counter-clockwise
        default: return .identity
        }
    }

    /// The previous implementation (ImageIO full-size thumbnail with transform): the reference for the tests.
    static func loadRGBViaThumbnail(_ url: URL) throws -> RGB {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let pw = props[kCGImagePropertyPixelWidth] as? Int, let ph = props[kCGImagePropertyPixelHeight] as? Int,
              let image = CGImageSourceCreateThumbnailAtIndex(source, 0, [
                  kCGImageSourceCreateThumbnailFromImageAlways: true, kCGImageSourceCreateThumbnailWithTransform: true,
                  kCGImageSourceThumbnailMaxPixelSize: max(pw, ph),
              ] as CFDictionary)
        else { throw Failure(description: "thumbnail failed") }
        let w = image.width, h = image.height
        let space = CGColorSpace(name: CGColorSpace.sRGB)!
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        rgba.withUnsafeMutableBytes { buf in
            let ctx = CGContext(data: buf.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4, space: space,
                                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        }
        return RGB(pixels: stride(from: 0, to: rgba.count, by: 4).flatMap { rgba[$0..<($0 + 3)] }, width: w, height: h, colorSpace: space)
    }

    /// rgba: straight (non-premultiplied) alpha.
    public static func writePNG(rgba: [UInt8], width: Int, height: Int, colorSpace: CGColorSpace, to url: URL) throws {
        guard let provider = CGDataProvider(data: Data(rgba) as CFData),
              let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
                                  space: colorSpace, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                                  provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
              let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
        else { throw Failure(description: "无法创建 PNG") }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else { throw Failure(description: "无法写入 \(url.path)") }
    }

    /// The image for the Qwen request: long side at most `longSide`, JPEG quality 0.85 (like matting/qwen.py).
    public static func jpegForUpload(_ img: RGB, longSide: Int = 1024) throws -> Data {
        let scale = min(1.0, Double(longSide) / Double(max(img.width, img.height)))
        let w = max(1, Int((Double(img.width) * scale).rounded())), h = max(1, Int((Double(img.height) * scale).rounded()))
        var rgba = [UInt8](repeating: 255, count: img.width * img.height * 4)
        for i in 0..<(img.width * img.height) {
            rgba[i * 4] = img.pixels[i * 3]
            rgba[i * 4 + 1] = img.pixels[i * 3 + 1]
            rgba[i * 4 + 2] = img.pixels[i * 3 + 2]
        }
        guard let provider = CGDataProvider(data: Data(rgba) as CFData),
              let full = CGImage(width: img.width, height: img.height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: img.width * 4,
                                 space: img.colorSpace, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                                 provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent),
              let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0, space: img.colorSpace,
                                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { throw Failure(description: "无法缩放图片") }
        ctx.interpolationQuality = .high
        ctx.draw(full, in: CGRect(x: 0, y: 0, width: w, height: h))
        guard let small = ctx.makeImage() else { throw Failure(description: "无法缩放图片") }
        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(out, UTType.jpeg.identifier as CFString, 1, nil) else {
            throw Failure(description: "无法编码 JPEG")
        }
        CGImageDestinationAddImage(dest, small, [kCGImageDestinationLossyCompressionQuality: 0.85] as CFDictionary)
        guard CGImageDestinationFinalize(dest) else { throw Failure(description: "无法编码 JPEG") }
        return out as Data
    }

    /// RGB pixels resized to w x h (high-quality interpolation).
    static func resized(_ img: RGB, width w: Int, height h: Int) -> [UInt8]? {
        var rgba = [UInt8](repeating: 255, count: img.width * img.height * 4)
        for i in 0..<(img.width * img.height) {
            rgba[i * 4] = img.pixels[i * 3]
            rgba[i * 4 + 1] = img.pixels[i * 3 + 1]
            rgba[i * 4 + 2] = img.pixels[i * 3 + 2]
        }
        var out = [UInt8](repeating: 0, count: w * h * 4)
        guard let provider = CGDataProvider(data: Data(rgba) as CFData),
              let full = CGImage(width: img.width, height: img.height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: img.width * 4,
                                 space: img.colorSpace, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                                 provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent)
        else { return nil }
        let ok = out.withUnsafeMutableBytes { buf -> Bool in
            guard let ctx = CGContext(data: buf.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                      space: img.colorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
            else { return false }
            ctx.interpolationQuality = .high
            ctx.draw(full, in: CGRect(x: 0, y: 0, width: w, height: h))
            return true
        }
        guard ok else { return nil }
        return stride(from: 0, to: out.count, by: 4).flatMap { out[$0..<($0 + 3)] }
    }

    /// The judge's picture (like matting.qwen.candidate_sheet): one tile per candidate mask (maskWidth x maskHeight,
    /// one byte per pixel), the image dimmed outside the candidate, numbered 1...n in the corner, 3 tiles per row.
    public static func candidateSheet(_ img: RGB, masks: [[UInt8]], maskWidth: Int, maskHeight: Int, tileLongSide: Int = 384) throws -> RGB {
        let s = Double(tileLongSide) / Double(max(img.width, img.height))
        let tw = max(1, Int((Double(img.width) * s).rounded())), th = max(1, Int((Double(img.height) * s).rounded()))
        guard let small = resized(img, width: tw, height: th) else { throw Failure(description: "无法缩放图片") }
        let cols = min(masks.count, 3), rows = (masks.count + cols - 1) / cols
        let sw = cols * tw, sh = rows * th
        var sheet = [UInt8](repeating: 0, count: sw * sh * 4)
        for (k, m) in masks.enumerated() {
            let ox = (k % cols) * tw, oy = (k / cols) * th
            for y in 0..<th {
                let my = min(maskHeight - 1, Int((Double(y) + 0.5) * Double(maskHeight) / Double(th)))
                for x in 0..<tw {
                    let mx = min(maskWidth - 1, Int((Double(x) + 0.5) * Double(maskWidth) / Double(tw)))
                    let inside = m[my * maskWidth + mx] != 0
                    let src = (y * tw + x) * 3, dst = ((oy + y) * sw + ox + x) * 4
                    for c in 0..<3 {
                        let v = Double(small[src + c])
                        sheet[dst + c] = UInt8(min(255, inside ? v : v * 0.15 + (c == 2 ? 12 : 0)))
                    }
                    sheet[dst + 3] = 255
                }
            }
        }
        // Numbers: red with a white outline, top-left of each tile. CoreGraphics' origin is bottom-left.
        sheet.withUnsafeMutableBytes { buf in
            guard let ctx = CGContext(data: buf.baseAddress, width: sw, height: sh, bitsPerComponent: 8, bytesPerRow: sw * 4,
                                      space: img.colorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
            else { return }
            let size = CGFloat(max(16, th / 10))
            let font = CTFontCreateWithName("Helvetica-Bold" as CFString, size, nil)
            for k in masks.indices {
                let ox = CGFloat((k % cols) * tw), top = CGFloat(sh - (k / cols) * th)
                for (stroke, color) in [(CGFloat(6), CGColor(red: 1, green: 1, blue: 1, alpha: 1)), (CGFloat(0), CGColor(red: 1, green: 0.16, blue: 0.16, alpha: 1))] {
                    let attrs: [CFString: Any] = [kCTFontAttributeName: font, kCTForegroundColorAttributeName: color,
                                                  kCTStrokeColorAttributeName: color, kCTStrokeWidthAttributeName: stroke]
                    let line = CTLineCreateWithAttributedString(CFAttributedStringCreate(nil, "\(k + 1)" as CFString, attrs as CFDictionary))
                    ctx.textPosition = CGPoint(x: ox + 6, y: top - size - 2)
                    CTLineDraw(line, ctx)
                }
            }
        }
        let rgb = stride(from: 0, to: sheet.count, by: 4).flatMap { sheet[$0..<($0 + 3)] }
        return RGB(pixels: rgb, width: sw, height: sh, colorSpace: img.colorSpace)
    }
}
