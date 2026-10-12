import CoreGraphics
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
    }

    /// Like matting.common.load_rgb: upright, transparent areas composited onto white. Pixels stay in the
    /// file's own RGB color space (no conversion), and the result PNG is tagged with that space.
    public static func loadRGB(_ url: URL) throws -> RGB {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil), CGImageSourceGetCount(source) > 0,
              let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let pw = props[kCGImagePropertyPixelWidth] as? Int, let ph = props[kCGImagePropertyPixelHeight] as? Int
        else { throw Failure(description: "无法读取图片（格式不支持或文件已损坏）") }
        // A "thumbnail" at full size is the documented way to get the EXIF orientation applied.
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: max(pw, ph),
            kCGImageSourceShouldCacheImmediately: true,
        ]
        guard let image = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) else {
            throw Failure(description: "无法解码图片")
        }
        let w = image.width, h = image.height
        let space = image.colorSpace.flatMap { $0.model == .rgb ? $0 : nil } ?? CGColorSpace(name: CGColorSpace.sRGB)!
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        let drawn = rgba.withUnsafeMutableBytes { buf -> Bool in
            guard let ctx = CGContext(data: buf.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                      space: space, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
            else { return false }
            ctx.setFillColor(red: 1, green: 1, blue: 1, alpha: 1)
            ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
            ctx.interpolationQuality = .none
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
            return true
        }
        guard drawn else { throw Failure(description: "无法解码图片") }
        var rgb = [UInt8](repeating: 0, count: w * h * 3)
        for i in 0..<(w * h) {
            rgb[i * 3] = rgba[i * 4]
            rgb[i * 3 + 1] = rgba[i * 4 + 1]
            rgb[i * 3 + 2] = rgba[i * 4 + 2]
        }
        return RGB(pixels: rgb, width: w, height: h, colorSpace: space)
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
}
