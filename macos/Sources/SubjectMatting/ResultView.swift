import AppKit
import ImageIO
import SwiftUI

/// The matting result on a checkerboard. Until the result exists, the original is shown instead.
struct ResultView: View {
    let original: URL
    let result: URL?
    /// Changes when the result file is rewritten, to reload it.
    let resultKey: String

    @State private var image: NSImage?

    var body: some View {
        ZStack {
            if let image {
                if result != nil {
                    Image(nsImage: image).resizable().aspectRatio(contentMode: .fit).background(Checkerboard())
                } else {
                    Image(nsImage: image).resizable().aspectRatio(contentMode: .fit).opacity(0.35)
                }
            } else {
                ProgressView()
            }
        }
        .padding(20)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .task(id: "\(original.path)|\(result?.path ?? "")|\(resultKey)") {
            image = nil
            image = await ImageLoader.load(result ?? original, maxPixels: 2400)
        }
    }
}

struct Checkerboard: View {
    var cell: CGFloat = 10

    var body: some View {
        Canvas { ctx, size in
            ctx.fill(Path(CGRect(origin: .zero, size: size)), with: .color(Color(white: 0.96)))
            var dark = Path()
            for row in 0..<Int((size.height / cell).rounded(.up)) {
                for col in 0..<Int((size.width / cell).rounded(.up)) where (row + col).isMultiple(of: 2) {
                    dark.addRect(CGRect(x: CGFloat(col) * cell, y: CGFloat(row) * cell, width: cell, height: cell))
                }
            }
            ctx.fill(dark, with: .color(Color(white: 0.82)))
        }
    }
}

struct Thumbnail: View {
    let url: URL
    let key: String
    @State private var image: NSImage?

    var body: some View {
        ZStack {
            Checkerboard(cell: 5)
            if let image {
                Image(nsImage: image).resizable().aspectRatio(contentMode: .fit)
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 5))
        .task(id: "\(url.path)|\(key)") {
            image = await ImageLoader.load(url, maxPixels: 96)
        }
    }
}

enum ImageLoader {
    /// Decodes off the main thread, downsampled to at most maxPixels on the long side, EXIF orientation applied
    /// (the worker applies it too).
    static func load(_ url: URL, maxPixels: Int) async -> NSImage? {
        await Task.detached(priority: .userInitiated) {
            guard let source = CGImageSourceCreateWithURL(url as CFURL, [kCGImageSourceShouldCache: false] as CFDictionary)
            else { return nil }
            let options: [CFString: Any] = [
                kCGImageSourceCreateThumbnailFromImageAlways: true,
                kCGImageSourceCreateThumbnailWithTransform: true,
                kCGImageSourceThumbnailMaxPixelSize: maxPixels,
                kCGImageSourceShouldCacheImmediately: true,
            ]
            guard let cg = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) else { return nil }
            return NSImage(cgImage: cg, size: NSSize(width: cg.width, height: cg.height))
        }.value
    }
}
