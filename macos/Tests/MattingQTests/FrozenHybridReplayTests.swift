import XCTest
@testable import MattingQ

/// Optional local artifact replay; no inference/network and no case IDs in production routing.
final class FrozenHybridReplayTests: XCTestCase {
    func testFrozenCandidatesWhenProvided() throws {
        let env=ProcessInfo.processInfo.environment
        guard let directory=env["SMQ_FROZEN_HYBRID"],let destination=env["SMQ_FROZEN_RESULT"] else {
            throw XCTSkip("Set SMQ_FROZEN_HYBRID and SMQ_FROZEN_RESULT for local frozen-candidate regression")
        }
        let expectedDetail: Set<Int> = [1,2,3,4,6,7,11,12,18,21,22,23,25,26,28,30,31,32,34,35,36,38,42,44,45,46]
        var results:[[String:Any]]=[]
        for i in 0..<47 {
            let folder=URL(fileURLWithPath:directory).appendingPathComponent(String(format:"%02d",i))
            let record=try JSONSerialization.jsonObject(with:Data(contentsOf:folder.appendingPathComponent("decision.json"))) as! [String:Any]
            let a=try ImageFiles.loadRGB(folder.appendingPathComponent("original.png")),b=try ImageFiles.loadRGB(folder.appendingPathComponent("detail.png"))
            let alphaA=a.alpha ?? [UInt8](repeating:255,count:a.width*a.height),alphaB=b.alpha ?? [UInt8](repeating:255,count:b.width*b.height)
            let subjects=(record["subjects"] as! [[String:Any]]).map{QwenSubject(label:$0["label"] as! String,box:$0["box"] as! [Double])}
            let metrics=HybridSelection.metrics(original:alphaA.flatMap{[0,0,0,$0]},detail:alphaB.flatMap{[0,0,0,$0]},width:a.width,height:a.height,subjects:subjects)
            let review=try record["review"].map{try JSONDecoder().decode(HybridReview.self,from:JSONSerialization.data(withJSONObject:$0))}
            let choice=HybridSelection.choose(review:review,severeLoss:metrics.severeLoss,targetExpansion:metrics.targetExpansion)
            let old=(record["decision"] as! [String:Any])["useDetail"] as! Bool
            results.append(["id":i,"old_detail":old,"detail":choice.useDetail,"changed":old != choice.useDetail,"exterior_added_share":metrics.exteriorAddedShare,"exterior_expansion":metrics.exteriorExpansion,"reason":choice.reason])
            XCTAssertEqual(choice.useDetail,expectedDetail.contains(i),"case \(i)")
        }
        try JSONSerialization.data(withJSONObject:results,options:[.prettyPrinted,.sortedKeys]).write(to:URL(fileURLWithPath:destination))
    }
}
