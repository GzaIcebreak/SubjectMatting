import XCTest
@testable import MattingQ

final class HybridSelectionTests: XCTestCase {
    func rgba(_ alpha: [UInt8]) -> [UInt8] { alpha.flatMap { [80, 100, 120, $0] } }
    func testUncertainAndIncompleteKeepOriginal() throws {
        let good = try HybridReview.parse(#"{"choice":"detail","complete":true,"same_target":true,"removal_supported":true,"appearance_sensitive":false,"material_ambiguous":false,"confidence":"high","reason":"网格背景去除"}"#)
        XCTAssertTrue(HybridSelection.choose(review: good, severeLoss: false).useDetail)
        XCTAssertFalse(HybridSelection.choose(review: good, severeLoss: true).useDetail)
        XCTAssertFalse(HybridSelection.choose(review: nil, severeLoss: false).useDetail)
        for field in ["complete", "same_target", "removal_supported"] {
            let bad = try HybridReview.parse(#"{"choice":"detail","complete":true,"same_target":true,"removal_supported":true,"appearance_sensitive":false,"material_ambiguous":false,"confidence":"high","reason":"test"}"#.replacingOccurrences(of: "\"\(field)\":true", with: "\"\(field)\":false"))
            XCTAssertFalse(HybridSelection.choose(review: bad, severeLoss: false).useDetail)
        }
        let transparent = try HybridReview.parse(#"{"choice":"detail","complete":true,"same_target":true,"removal_supported":true,"appearance_sensitive":false,"material_ambiguous":true,"confidence":"high","reason":"玻璃内部变化不确定"}"#)
        XCTAssertFalse(HybridSelection.choose(review: transparent, severeLoss: false).useDetail)
    }
    func testMalformedReviewDoesNotSilentlyApprove() {
        XCTAssertThrowsError(try HybridReview.parse(#"{"choice":"detail"}"#))
        XCTAssertThrowsError(try HybridReview.parse(#"{"choice":"detail","complete":"true"}"#))
    }
    func testThinLineProtectionDoesNotFillHoleOrChangeRGB() {
        var a = [UInt8](repeating:0,count:81)
        for y in 0..<9 { a[y*9+4]=255 }
        let direct = rgba(a), empty = rgba([UInt8](repeating:0,count:81))
        let kept = HybridSelection.protectThinLines(direct: direct, refined: empty, width:9,height:9)
        XCTAssertEqual(kept[4*9*4+4*4+3],255)
        XCTAssertEqual(kept[4*9*4+2*4+3],0)
        for i in stride(from:0,to:kept.count,by:4) { XCTAssertEqual(Array(kept[i..<i+3]),Array(empty[i..<i+3])) }
    }
    func testMissingTargetIsDifferentFromInternalHole() {
        let a = rgba([UInt8](repeating:255,count:100))
        var hole = [UInt8](repeating:255,count:100)
        for y in 2..<8 { for x in 2..<8 { hole[y*10+x]=0 } }
        let s=[QwenSubject(label:"object",box:[0,0,10,10])]
        XCTAssertFalse(HybridSelection.metrics(original:a,detail:rgba(hole),width:10,height:10,subjects:s).severeLoss)
        XCTAssertTrue(HybridSelection.metrics(original:a,detail:rgba([UInt8](repeating:0,count:100)),width:10,height:10,subjects:s).severeLoss)
    }
    func testAddedObjectOutsideTargetIsVetoed() throws {
        var a=[UInt8](repeating:0,count:400),b=a
        for y in 10..<20 {for x in 6..<14 {a[y*20+x]=255;b[y*20+x]=255}}
        for y in 0..<6 {for x in 6..<14 {b[y*20+x]=64}}
        let m=HybridSelection.metrics(original:rgba(a),detail:rgba(b),width:20,height:20,subjects:[QwenSubject(label:"object",box:[6,10,14,20])])
        XCTAssertTrue(m.targetExpansion)
        let r=try HybridReview.parse(#"{"choice":"detail","complete":true,"same_target":true,"removal_supported":true,"appearance_sensitive":false,"material_ambiguous":false,"confidence":"high","reason":"better"}"#)
        XCTAssertTrue(HybridSelection.choose(review:r,severeLoss:false,targetExpansion:false).useDetail)
        XCTAssertFalse(HybridSelection.choose(review:r,severeLoss:false,targetExpansion:true).useDetail)
    }

    func testOpticalAppearanceProtectedEvenWhenReviewerLikesDetail() throws {
        let r=try HybridReview.parse(#"{"choice":"detail","complete":true,"same_target":true,"removal_supported":true,"appearance_sensitive":true,"material_ambiguous":false,"confidence":"high","reason":"内部透光"}"#)
        XCTAssertFalse(HybridSelection.choose(review:r,severeLoss:false).useDetail)
    }
    func testInvalidDetailDimensionsRejectedWithoutAllocation() {
        for (w,h,n) in [(0,1,0),(-1,1,0),(Int.max,2,10),(3,3,1)] {
            XCTAssertThrowsError(try MattingEngineQ.validateDetailInput(count:n,width:w,height:h))
        }
        XCTAssertNoThrow(try MattingEngineQ.validateDetailInput(count:27,width:3,height:3))
    }

    func testDisconnectedRemnantsAreRejectedButPerforatedFrameIsNot() {
        let a=[UInt8](repeating:255,count:1600)
        var fragments=[UInt8](repeating:0,count:1600),frame=fragments
        for y in 0..<40 {for x in 0..<40 {
            if y<6 || y>32 {fragments[y*40+x]=255}
            if x<2 || x>37 || y<2 || y>37 {frame[y*40+x]=255}
        }}
        let subjects=[QwenSubject(label:"object",box:[0,0,40,40])]
        XCTAssertTrue(HybridSelection.metrics(original:rgba(a),detail:rgba(fragments),width:40,height:40,subjects:subjects).severeLoss)
        XCTAssertFalse(HybridSelection.metrics(original:rgba(a),detail:rgba(frame),width:40,height:40,subjects:subjects).severeLoss)
    }

}
