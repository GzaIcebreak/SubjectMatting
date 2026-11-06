import Foundation

extension ImageFiles {
    public static func compositeForReview(_ img:RGB,rgba:[UInt8]) -> RGB {
        let scale=min(1.0,1024.0/Double(max(img.width,img.height)))
        let w=max(1,Int(Double(img.width)*scale)),h=max(1,Int(Double(img.height)*scale))
        var pixels=[UInt8](repeating:0,count:w*h*3)
        for y in 0..<h {for x in 0..<w {
            let sx=min(img.width-1,Int((Double(x)+0.5)/scale)),sy=min(img.height-1,Int((Double(y)+0.5)/scale)),i=sy*img.width+sx
            let a=Double(rgba[i*4+3])/255, bg:[Double]=((x/24+y/24)%2==0) ? [38,116,186] : [45,48,52]
            for c in 0..<3 {pixels[(y*w+x)*3+c]=UInt8((Double(rgba[i*4+c])*a+bg[c]*(1-a)).rounded())}
        }}
        return RGB(pixels:pixels,width:w,height:h,colorSpace:img.colorSpace)
    }

    /// Up to four nonoverlapping cells: cover both removed and added content, then remaining largest conflicts.
    /// Review composites are bounded before allocation, including for large input photographs.
    public static func hybridEvidence(_ img:RGB, original:[UInt8],detail:[UInt8]) throws -> [Data] {
        var cells:[(x0:Int,y0:Int,x1:Int,y1:Int,added:Double,removed:Double)]=[]
        for gy in 0..<4 {for gx in 0..<4 {
            let x0=gx*img.width/4,x1=(gx+1)*img.width/4,y0=gy*img.height/4,y1=(gy+1)*img.height/4
            guard x1>x0,y1>y0 else {continue}
            var added=0.0,removed=0.0
            for y in y0..<y1 {for x in x0..<x1 {let i=(y*img.width+x)*4+3;let delta=Double(Int(detail[i])-Int(original[i]));added += max(0,delta);removed += max(0,-delta)}}
            cells.append((x0,y0,x1,y1,added,removed))
        }}
        var indices:[Int]=[]
        for ranking in [cells.indices.sorted{cells[$0].removed>cells[$1].removed},cells.indices.sorted{cells[$0].added>cells[$1].added}] {
            if let i=ranking.first,!indices.contains(i),cells[i].added+cells[i].removed>0 {indices.append(i)}
        }
        for i in cells.indices.sorted(by:{cells[$0].added+cells[$0].removed>cells[$1].added+cells[$1].removed}) where indices.count<4 && !indices.contains(i) && cells[i].added+cells[i].removed>0 {indices.append(i)}
        return try indices.map {index in
            let c=cells[index],cw=c.x1-c.x0,ch=c.y1-c.y0
            let scale=min(1.0,384.0/Double(max(cw,ch))),w=max(1,Int(Double(cw)*scale)),h=max(1,Int(Double(ch)*scale))
            var pixels=[UInt8](repeating:0,count:w*3*h*3)
            for y in 0..<h {for x in 0..<w {
                let sx=c.x0+min(cw-1,Int((Double(x)+0.5)/scale)),sy=c.y0+min(ch-1,Int((Double(y)+0.5)/scale)),i=sy*img.width+sx
                let bg:[Double]=((x/16+y/16)%2==0) ? [38,116,186] : [45,48,52]
                for col in 0..<3 {for channel in 0..<3 {
                    let value:UInt8
                    if col==0 {value=img.pixels[i*3+channel]} else {
                        let rgba=col==1 ? original:detail,a=Double(rgba[i*4+3])/255
                        value=UInt8((Double(rgba[i*4+channel])*a+bg[channel]*(1-a)).rounded())
                    }
                    pixels[(y*w*3+col*w+x)*3+channel]=value
                }}
            }}
            return try jpegForUpload(RGB(pixels:pixels,width:w*3,height:h,colorSpace:img.colorSpace),longSide:1152)
        }
    }
}

extension QwenClient {
    public func reviewHybrid(jpeg:Data,img:ImageFiles.RGB,original:[UInt8],detail:[UInt8],subjects:[QwenSubject],instruction:String?) throws -> HybridReview {
        let target=subjects.map(\.label).joined(separator:", ")
        let prompt="""
        比较两个去背景结果，选择可以安全交付的一个。第一张是原图，第二张是A原流程，第三张是B细节流程。
        A/B放在相同蓝灰棋盘底上。后续图片是分歧最大的局部，每张从左到右为原图/A/B。棋盘底不属于图像内容。
        已有目标：\(target)。用户要求：\(instruction ?? "自动选择前景主体")。
        先检查整体目标，再比较孔洞、细杆、毛发和边缘；不要因为透明面积更多就认为更好。
        - A代表已有的语义选择。B必须保留同一目标以及必要的底座、附属部件、完整身体；不能漏掉整个个体或把整体切成不相连残片。
        - B可以恢复A误删的实体部位、清除A误留的背景；但不能擅自扩大目标范围或加入额外的对象、倒影、环境效果。
        - 内部孔洞要结合原图证明是透过主体看到的背景。实心表面、高光、花纹不能变透明。
        - 先独立看原图：若主体包含透明/半透明/折射/体积散射材质，appearance_sensitive=true（例如透过物体看到背景、玻璃、液体、烟雾），这是材质事实，与A/B哪个更好无关。普通不透明物体的表面亮斑不算透明材质。
        - 对 appearance_sensitive=true 的图，本轮要求保持原流程的光学外观，不允许自动改成“更透明”或增加环境效果；请选择original。
        - 反射/透射/半透明内容不能仅凭颜色删掉或变成实体；两结果的这类差异无法可靠判断时 material_ambiguous=true，保留A。
        - 细杆断裂、残留底座、明显抹除附属物都算缺陷。若B有明显改进且没有这些退步，选detail；两者近似、利弊并存、无法判断则选original或uncertain。
        complete: B完整保留目标与必要附属物；same_target: B没有增加额外目标且与既有语义选择一致；removal_supported: B相对A删除的区域有原图证据是背景，若没有新增删除则true。
        只输出JSON：{"choice":"original|detail|uncertain","complete":true,"same_target":true,"removal_supported":true,"appearance_sensitive":false,"material_ambiguous":false,"confidence":"high|medium|low","reason":"简短中文理由，指出具体保留或丢失的部分"}
        """
        let images = try [jpeg,ImageFiles.jpegForUpload(ImageFiles.compositeForReview(img,rgba:original)),ImageFiles.jpegForUpload(ImageFiles.compositeForReview(img,rgba:detail))] + ImageFiles.hybridEvidence(img,original:original,detail:detail)
        return try HybridReview.parse(chat(model:judgeModel,images:images,text:prompt))
    }
}
