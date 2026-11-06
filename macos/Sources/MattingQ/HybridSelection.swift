import Foundation

/// Structured semantic opinion; confidence is a review category, not a calibrated probability.
public struct HybridReview: Codable {
    public var choice: String
    public var complete: Bool
    public var same_target: Bool
    public var removal_supported: Bool
    public var appearance_sensitive: Bool
    public var material_ambiguous: Bool
    public var confidence: String
    public var reason: String

    public static func parse(_ text: String) throws -> HybridReview {
        let object = try QwenClient.firstJSON(text)
        let data = try JSONSerialization.data(withJSONObject: object)
        let value = try JSONDecoder().decode(Self.self, from: data)
        guard ["original", "detail", "uncertain"].contains(value.choice), ["high", "medium", "low"].contains(value.confidence) else {
            throw QwenError("组合判断字段不合法")
        }
        return value
    }
}

public enum HybridSelection {
    public struct Decision: Codable {
        public var useDetail: Bool
        public var reason: String
        public init(useDetail:Bool,reason:String) {self.useDetail=useDetail;self.reason=reason}
    }
    public struct Metrics: Codable {
        public var disagreement: Double
        public var removedShare: Double
        public var addedShare: Double
        public var severeLoss: Bool
        public var targetExpansion: Bool
        public var exteriorExpansion: Bool = false
        public var exteriorAddedShare: Double = 0
        public var exteriorTolerancePixels: Int = 0
        public var detailOccupancy: Double
    }

    public static func choose(review: HybridReview?, severeLoss: Bool, targetExpansion: Bool = false) -> Decision {
        guard !targetExpansion else {return Decision(useDetail:false,reason:"细节候选在既有主体支持范围外新增了大块内容")}
        guard !severeLoss else { return Decision(useDetail:false,reason:"细节候选破坏目标连续性、丢失整块目标或近乎为空") }
        guard let r = review else { return Decision(useDetail:false,reason:"判断不可用，保留原流程") }
        let accepted = r.choice == "detail" && r.complete && r.same_target && r.removal_supported && !r.material_ambiguous && !r.appearance_sensitive && r.confidence == "high"
        return Decision(useDetail:accepted,reason:(accepted ? "采用细节候选：" : "保留原流程：") + r.reason)
    }

    /// Geometry only acts as a veto for gross loss; removing an internal hole is not an error by itself.
    public static func metrics(original: [UInt8], detail: [UInt8], width: Int, height: Int, subjects: [QwenSubject]) -> Metrics {
        let n=width*height
        guard n>0, original.count==n*4, detail.count==n*4 else {
            return Metrics(disagreement:1,removedShare:1,addedShare:0,severeLoss:true,targetExpansion:false,detailOccupancy:0)
        }
        var union=0, changed=0, removed=0, added=0, oldArea=0, newArea=0
        for i in 0..<n {
            let a=original[i*4+3], b=detail[i*4+3]
            if a>16 || b>16 { union += 1; if abs(Int(a)-Int(b))>48 { changed += 1 } }
            if a>=128 { oldArea += 1; if b<128 { removed += 1 } }
            if b>=128 { newArea += 1; if a<128 { added += 1 } }
        }
        let boxes=subjects.filter {$0.box.count==4 && $0.box.allSatisfy{$0.isFinite}}.map {s -> [Double] in
            let pad=0.05*Double(max(width,height))
            return [s.box[0]-pad,s.box[1]-pad,s.box[2]+pad,s.box[3]+pad]
        }
        var outsideAdded=0.0, detailMass=0.0
        if !boxes.isEmpty {
            for y in 0..<height {for x in 0..<width {
                let k=(y*width+x)*4+3
                detailMass += Double(detail[k])/255
                if detail[k]>16 && detail[k]>original[k] && !boxes.contains(where:{Double(x)>=$0[0] && Double(y)>=$0[1] && Double(x)<$0[2] && Double(y)<$0[3]}) {outsideAdded += Double(Int(detail[k])-Int(original[k]))/255}
            }}
        }
        let growth = ExteriorGrowth.measure(original: original, detail: detail, width: width, height: height)
        let boxExpansion=outsideAdded>max(4,Double(n)/1000) && outsideAdded>0.03*max(1,detailMass)
        var severe = newArea < max(1,oldArea/20)
        for subject in subjects where subject.box.count==4 && subject.box.allSatisfy({$0.isFinite}) {
            let x0=max(0,min(width,Int(subject.box[0]))),x1=max(0,min(width,Int(ceil(subject.box[2]))))
            let y0=max(0,min(height,Int(subject.box[1]))),y1=max(0,min(height,Int(ceil(subject.box[3]))))
            guard x1>x0,y1>y0 else {continue}
            var a=0,b=0
            var rowsA=[Int](repeating:0,count:y1-y0),rowsB=rowsA
            var colsA=[Int](repeating:0,count:x1-x0),colsB=colsA
            for y in y0..<y1 { for x in x0..<x1 {
                let i=(y*width+x)*4+3
                if original[i]>=128 {a += 1;rowsA[y-y0] += 1;colsA[x-x0] += 1}
                if detail[i]>=128 {b += 1}
                if detail[i]>32 {rowsB[y-y0] += 1;colsB[x-x0] += 1}
            }}
            // A long empty slab between two surviving pieces breaks a once-continuous object.
            // Unlike area loss, this does not reject a frame merely because its interior becomes transparent.
            func broken(_ old:[Int],_ new:[Int],cross:Int)->Bool {
                let threshold=max(2,Int(ceil(0.01*Double(cross))))
                guard let first=new.firstIndex(where:{$0>=threshold}),let last=new.lastIndex(where:{$0>=threshold}),last>first else {return false}
                var run=0
                let required=max(3,Int(ceil(0.12*Double(old.count))))
                for k in first...last {
                    if old[k]>=max(2,Int(ceil(0.02*Double(cross)))) && new[k]<threshold {run += 1} else {run=0}
                    if run>=required {return true}
                }
                return false
            }
            if broken(rowsA,rowsB,cross:x1-x0) || broken(colsA,colsB,cross:y1-y0) {severe=true}
            if a>max(4,(x1-x0)*(y1-y0)/100), b*10<a { severe=true }
        }
        return Metrics(disagreement:Double(changed)/Double(max(1,union)),removedShare:Double(removed)/Double(max(1,oldArea)),addedShare:Double(added)/Double(max(1,newArea)),severeLoss:severe,targetExpansion:boxExpansion || growth.veto,exteriorExpansion:growth.veto,exteriorAddedShare:growth.addedShare,exteriorTolerancePixels:growth.tolerancePixels,detailOccupancy:Double(newArea)/Double(n))
    }

    /// Only rescue confident BiRefNet thin cores when matting weakens them. Never union with SAM3,
    /// never change RGB, and never fill a pixel BiRefNet predicts as background.
    public static func protectThinLines(direct:[UInt8], refined:[UInt8], width:Int,height:Int) -> [UInt8] {
        guard width>2,height>2,direct.count==width*height*4,refined.count==direct.count else {return refined}
        var out=refined
        let radius=max(1,min(6,Int((Double(max(width,height))/512).rounded())))
        for y in 1..<(height-1) { for x in 1..<(width-1) {
            let k=(y*width+x)*4+3
            guard direct[k]>=220,Int(direct[k])-Int(refined[k])>32 else {continue}
            for r in 1...radius {
                let horizontal=x>=r && x+r<width && direct[k-r*4]<128 && direct[k+r*4]<128
                let vertical=y>=r && y+r<height && direct[k-r*width*4]<128 && direct[k+r*width*4]<128
                if horizontal || vertical { out[k]=direct[k]; break }
            }
        }}
        return out
    }
}
