import AppKit
import ImageIO
import SwiftUI

/// Switch between the original and the matting result without changing the saved output.
struct ResultView: View {
    let original: URL
    let result: URL?
    /// Changes when the result file is rewritten, to reload it.
    let resultKey: String

    @State private var image: NSImage?
    @State private var showOriginal = false
    @State private var loadFailed = false

    private var displaysOriginal: Bool { showOriginal || result == nil }
    private var displayedURL: URL { displaysOriginal ? original : (result ?? original) }

    var body: some View {
        VStack(spacing: 12) {
            Picker("预览", selection: Binding(
                get: { displaysOriginal },
                set: { showOriginal = $0 }
            )) {
                Text("原图").tag(true)
                Text("抠图结果").tag(false)
            }
            .pickerStyle(.segmented)
            .frame(width: 220)
            .disabled(result == nil)
            .help(result == nil ? "结果生成后即可切换查看" : "切换查看抠图前后的图片")

            ZStack {
                if let image {
                    if displaysOriginal {
                        Image(nsImage: image).resizable().aspectRatio(contentMode: .fit)
                            .background(Checkerboard())
                            .accessibilityLabel("原图预览")
                    } else {
                        Image(nsImage: image).resizable().aspectRatio(contentMode: .fit).background(Checkerboard())
                            .accessibilityLabel("抠图结果预览")
                            // Only the result preview exports the transparent PNG by dragging.
                            .onDrag { NSItemProvider(contentsOf: result) ?? NSItemProvider() }
                            .help("可以把结果直接拖到访达或其他应用")
                    }
                } else if loadFailed {
                    Label(displaysOriginal ? "无法读取原图" : "无法读取抠图结果", systemImage: "photo.badge.exclamationmark")
                        .foregroundStyle(.secondary)
                } else {
                    ProgressView()
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .padding(20)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .onChange(of: original) { showOriginal = false }
        .task(id: "\(displayedURL.path)|\(displaysOriginal)|\(resultKey)") {
            image = nil
            loadFailed = false
            let loaded = await ImageLoader.load(displayedURL, maxPixels: 2400)
            // A previous load may finish after the user has switched images or preview modes.
            guard !Task.isCancelled else { return }
            image = loaded
            loadFailed = loaded == nil
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
