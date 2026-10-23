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

/// tests/test_vlm.py's cases for remove regions, the judge's reply and the repairs.
final class QwenDetectionTests: XCTestCase {
    func testRemoveRegions() throws {
        let d = try QwenClient.parseDetection(#"{"subjects": [{"label": "girl", "box_2d": [100, 100, 500, 900]}], "remove": [{"label": "checkerboard", "box_2d": [600, 600, 700, 800]}]}"#, width: 1000, height: 1000)
        XCTAssertEqual(d.subjects.map(\.label), ["girl"])
        XCTAssertEqual(d.remove, [QwenSubject(label: "checkerboard", box: [600, 600, 700, 800])])
    }

    func testBoxWithoutBracketsAndTrailingCommas() throws {
        let d = try QwenClient.parseDetection(#"{"subjects": [{"label": "umpire", "box_2d": 37, 421, 251, 969},], "remove": []}"#, width: 1000, height: 1000)
        XCTAssertEqual(d.subjects, [QwenSubject(label: "umpire", box: [37, 421, 251, 969])])
        XCTAssertEqual(d.remove, [])
    }

    func testKeepTakesTheLastAnswer() throws {
        XCTAssertEqual(try QwenClient.parseKeep(#"candidate 1 looks like a chair... {"keep": [3, 1, 9]}"#, count: 3), [0, 2])
        XCTAssertThrowsError(try QwenClient.parseKeep("no idea", count: 3))
    }
}

final class ImageHelperTests: XCTestCase {
    func testCandidateSheetLayout() throws {
        let img = ImageFiles.RGB(pixels: [UInt8](repeating: 200, count: 40 * 20 * 3), width: 40, height: 20,
                                 colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!)
        var m = [UInt8](repeating: 0, count: 40 * 20)
        m[10 * 40 + 20] = 1
        let sheet = try ImageFiles.candidateSheet(img, masks: Array(repeating: m, count: 4), maskWidth: 40, maskHeight: 20, tileLongSide: 40)
        XCTAssertEqual([sheet.width, sheet.height], [120, 40])  // 3 per row, 2 rows
        XCTAssertLessThan(sheet.pixels[(19 * 120 + 39) * 3], 60)  // dimmed outside the candidate
    }

    func testAlphaOfATransparentPNGIsKept() throws {
        let w = 4, h = 2
        var px = [UInt8](repeating: 255, count: w * h * 4)
        px[3] = 0  // top-left transparent
        px[4 * 1 + 3] = 128
        let space = CGColorSpace(name: CGColorSpace.sRGB)!
        let image = CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4, space: space,
                            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue), provider: CGDataProvider(data: Data(px) as CFData)!,
                            decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".png")
        defer { try? FileManager.default.removeItem(at: url) }
        let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)!
        CGImageDestinationAddImage(dest, image, nil)
        XCTAssertTrue(CGImageDestinationFinalize(dest))
        let rgb = try ImageFiles.loadRGB(url)
        XCTAssertEqual(rgb.alpha?.prefix(3).map { Int($0) }, [0, 128, 255])
        XCTAssertEqual(Array(rgb.pixels.prefix(3)), [255, 255, 255])  // transparent composited onto white
    }
}
