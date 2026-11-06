import XCTest
@testable import MattingQ

final class ExteriorGrowthTests: XCTestCase {
    private func rgba(_ mask:[UInt8])->[UInt8] {mask.flatMap{[90,110,130,$0]}}
    private func accepts(_ a:[UInt8],_ b:[UInt8],_ w:Int,_ h:Int) throws -> Bool {
        let r=try HybridReview.parse(#"{"choice":"detail","complete":true,"same_target":true,"removal_supported":true,"appearance_sensitive":false,"material_ambiguous":false,"confidence":"high","reason":"looks good"}"#)
        // A loose, correct target box must not authorize arbitrary foreground inside it.
        let m=HybridSelection.metrics(original:rgba(a),detail:rgba(b),width:w,height:h,subjects:[QwenSubject(label:"object",box:[0,0,Double(w),Double(h)])])
        return HybridSelection.choose(review:r,severeLoss:m.severeLoss,targetExpansion:m.targetExpansion).useDetail
    }
    func testAttachedExteriorBackgroundCannotBeApprovedByReviewer() throws {
        for scale in [1,3] {for transpose in [false,true] {
            let w=(transpose ? 80:160)*scale,h=(transpose ? 160:80)*scale
            var a=[UInt8](repeating:0,count:w*h),b=a
            for y in 0..<h {for x in 0..<w {
                let u=(transpose ? y:x)/scale,v=(transpose ? x:y)/scale
                if u>=50 && u<90 && v>=15 && v<45 {a[y*w+x]=255;b[y*w+x]=255}
                if u>=70 && u<110 && v>=40 && v<70 {b[y*w+x]=255}
            }}
            XCTAssertFalse(try accepts(a,b,w,h),"scale \(scale), transpose \(transpose)")
        }}
    }
    func testInteriorRestorationAndNarrowBoundaryCorrectionAreAllowed() throws {
        let w=120,h=100
        var a=[UInt8](repeating:0,count:w*h),b=a
        for y in 20..<80 {for x in 30..<90 {
            b[y*w+x]=255
            if x<40 || x>=80 || y<30 || y>=70 {a[y*w+x]=255}
        }}
        for y in 20..<80 {b[y*w+90]=255}
        XCTAssertTrue(try accepts(a,b,w,h))
    }
    func testRemovingInternalBackgroundRemainsAllowed() throws {
        let w=100,h=100
        var a=[UInt8](repeating:0,count:w*h),b=a
        for y in 10..<90 {for x in 10..<90 {
            a[y*w+x]=255
            if x<15 || x>=85 || y<15 || y>=85 {b[y*w+x]=255}
        }}
        XCTAssertTrue(try accepts(a,b,w,h))
    }
    func testWideCanvasCannotHideExteriorTranslucentGrowth() throws {
        let w=1000,h=100
        var a=[UInt8](repeating:0,count:w*h),b=a
        for y in 20..<50 {for x in 450..<500 {a[y*w+x]=255;b[y*w+x]=255}}
        for y in 50..<90 {for x in 450..<520 {b[y*w+x]=96}}
        XCTAssertFalse(try accepts(a,b,w,h))
    }
    func testLargeImageExpansionIsStillDetected() throws {
        let w=2048,h=512
        var a=[UInt8](repeating:0,count:w*h),b=a
        for y in 100..<300 {for x in 700..<1000 {a[y*w+x]=255;b[y*w+x]=255}}
        for y in 200..<450 {for x in 900..<1200 {b[y*w+x]=255}}
        XCTAssertFalse(try accepts(a,b,w,h))
    }
    func testPoolingDoesNotSealAnExteriorChannel() throws {
        let w=2048,h=512
        for offset in [0,1] {
            var a=[UInt8](repeating:0,count:w*h),b=a
            for y in 100..<400 {for x in 700..<1100 {
                b[y*w+x]=255
                if x<720 || x>=1080 || y<120 || y>=380 {a[y*w+x]=255}
            }}
            // A one-pixel channel connects the apparent hole to exterior background.
            for y in 100..<120 {a[y*w+800+offset]=0}
            XCTAssertFalse(try accepts(a,b,w,h),"sampling offset \(offset)")
        }
    }

}
