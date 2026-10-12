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
