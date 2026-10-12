import XCTest

@testable import MattingQ

/// The same cases as tests/test_vlm.py::test_parse_subjects_* for the Python parser.
final class QwenParseTests: XCTestCase {
    func testObjectFormScales0To1000ToPixels() throws {
        let s = try QwenClient.parseSubjects("```json\n{\"subjects\": [{\"label\": \"Cat\", \"box_2d\": [100, 200, 500, 1000]}]}\n```", width: 640, height: 480)
        XCTAssertEqual(s, [QwenSubject(label: "cat", box: [64, 96, 320, 480])])
    }

    func testBareListAndBbox2dKey() throws {
        let s = try QwenClient.parseSubjects(#"here: [{"bbox_2d": [13, 115, 501, 986], "label": "cat"}, {"bbox_2d": [542, 57, 999, 771], "label": "cat"}]"#, width: 1000, height: 1000)
        XCTAssertEqual(s.map(\.box), [[13, 115, 501, 986], [542, 57, 999, 771]])
    }

    func testDropsDegenerateAndFixesSwappedCorners() throws {
        let s = try QwenClient.parseSubjects(#"{"subjects": [{"label": "a", "box_2d": [500, 500, 500, 600]}, {"label": "b", "box_2d": [900, 900, 100, 100]}]}"#, width: 1000, height: 1000)
        XCTAssertEqual(s, [QwenSubject(label: "b", box: [100, 100, 900, 900])])
    }

    func testEmptyAndGarbage() throws {
        XCTAssertEqual(try QwenClient.parseSubjects(#"{"subjects": []}"#, width: 10, height: 10), [])
        XCTAssertThrowsError(try QwenClient.parseSubjects("sorry, I can't see a subject", width: 10, height: 10))
    }

    func testBracketsInsideStringsDontEndTheJSON() throws {
        let s = try QwenClient.parseSubjects(#"{"subjects": [{"label": "sign {stop]", "box_2d": [0, 0, 500, 500]}]} trailing"#, width: 100, height: 100)
        XCTAssertEqual(s.first?.label, "sign {stop]")
    }
}

import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

/// The fast loader must place every pixel where the ImageIO thumbnail path (with EXIF transform) puts it.
final class ImageOrientationTests: XCTestCase {
    func testAllEightExifOrientationsMatchTheThumbnailPath() throws {
        // An asymmetric 5x3 image with distinct pixels, saved as PNG with each orientation tag.
        let w = 5, h = 3
        var px = [UInt8](repeating: 255, count: w * h * 4)
        for i in 0..<(w * h) { px[i * 4] = UInt8(i * 16); px[i * 4 + 1] = UInt8(255 - i * 16); px[i * 4 + 2] = UInt8((i * 37) % 256) }
        let space = CGColorSpace(name: CGColorSpace.sRGB)!
        let provider = CGDataProvider(data: Data(px) as CFData)!
        let image = CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4, space: space,
                            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue), provider: provider,
                            decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        for o in 1...8 {
            let url = dir.appendingPathComponent("o\(o).png")
            let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)!
            CGImageDestinationAddImage(dest, image, [kCGImagePropertyOrientation: o] as CFDictionary)
            XCTAssertTrue(CGImageDestinationFinalize(dest))
            let fast = try ImageFiles.loadRGB(url), ref = try ImageFiles.loadRGBViaThumbnail(url)
            XCTAssertEqual([fast.width, fast.height], [ref.width, ref.height], "orientation \(o)")
            XCTAssertEqual(fast.pixels, ref.pixels, "orientation \(o)")
        }
    }
}
