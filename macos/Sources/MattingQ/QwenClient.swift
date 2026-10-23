import Foundation

/// Subject detection with Qwen-VL over DashScope's OpenAI-compatible API. Port of matting/qwen.py: same prompts,
/// same lenient parsing, boxes in Qwen's 0~1000 coordinates converted to pixels of the original image. detect()
/// also returns the see-through regions inside the subjects; judge() asks a second model which candidate masks
/// belong to the subjects.
public struct QwenSubject: Equatable, Sendable {
    public var label: String
    public var box: [Double]  // x1, y1, x2, y2 in pixels

    public init(label: String, box: [Double]) {
        self.label = label
        self.box = box
    }
}

public struct QwenDetection: Equatable, Sendable {
    public var subjects: [QwenSubject]
    public var remove: [QwenSubject]  // regions inside the subjects that must become transparent

    public init(subjects: [QwenSubject], remove: [QwenSubject] = []) {
        self.subjects = subjects
        self.remove = remove
    }
}

public struct QwenError: Error, Equatable, CustomStringConvertible {
    public var description: String
    public init(_ description: String) { self.description = description }
}

public final class QwenClient: @unchecked Sendable {
    public static let defaultBaseURL = URL(string: "https://dashscope.aliyuncs.com/compatible-mode/v1")!
    public static let defaultModel = "qwen3-vl-plus"  // best boxes of the Qwen models tried
    public static let defaultJudgeModel = "qwen3.6-plus"  // judged as well as qwen3.8-max at a third of the time

    static let autoPrompt = """
    这张图要做抠图：去掉背景，只留下前景主体。请找出前景主体。
    规则：
    1. 一件物品、模型、摆件、产品，连同它的底座、支架、配件，或放在同一个底座/盘子/托盘上的一组东西，算一个整体：只输出一个主体，框住整体，不要拆成部件。
    2. 前景里有几个各自独立的个体时（几个人、几只动物、几个玩偶，即使挨在一起），每个单独输出一个框，最多 4 个。
    3. 背景、地面、桌面、墙、天空、房间不算主体。
    4. remove：落在主体轮廓里面或凹口里、但不属于主体、抠图时要变透明的区域，例如环形/字母/数字中间的洞或凹口里透出来的背景，手臂和身体之间、两腿之间露出的背景，画在图里的假透明棋盘格。每一块分开的区域单独输出一个框，框要包住那一整块。主体外面的普通背景（墙、窗、地面、家具、天空等）本来就会去掉，不要列。牙齿、眼白、衣服上的白字、主体身上的花纹都属于主体，不算。大多数图没有这种区域，没有就输出空列表。
    只输出 JSON，不要任何解释：
    {"subjects": [{"label": "...", "box_2d": [x1, y1, x2, y2]}], "remove": [{"label": "...", "box_2d": [x1, y1, x2, y2]}]}
    - label：主体的英文名词，1~3 个词，单数，只写一种东西的名字，不要数量、不要 and/with（三只熊写 "teddy bear"，城堡模型连火车写 "castle model"）。remove 的 label 写那块区域里看到的东西（"checkerboard"、"sky"、"wall"）。
    - box_2d：外接框，坐标是相对图片宽高的 0~1000 整数，左上角 (x1, y1)，右下角 (x2, y2)。主体的框要完整包住整个主体，remove 的框只框住那块区域。
    """

    static func instructionPrompt(_ instruction: String) -> String {
        """
        用户想从这张图里抠出：「\(instruction)」。
        找出图中符合这个描述的对象（可能有多个，也可能没有），以及落在这些对象轮廓里面、但不属于对象、抠图时要变透明的区域（remove：环形/字母/数字中间的洞或凹口里透出来的背景，手臂和身体之间露出的背景，画在图里的假透明棋盘格；每一块分开的区域单独一个框，框要包住那一整块；对象外面的普通背景本来就会去掉，不要列；牙齿、眼白、白字、花纹属于对象，不算；大多数图没有这种区域，没有就输出空列表）。只输出 JSON，不要任何解释：
        {"subjects": [{"label": "...", "box_2d": [x1, y1, x2, y2]}], "remove": [{"label": "...", "box_2d": [x1, y1, x2, y2]}]}
        - label：对象的英文名词短语，1~3 个词，单数，例如 "cat"、"shopping bag"。不要把位置写进 label（"左边的猫" 写成 "cat"，靠 box_2d 区分是哪一只）。
        - box_2d：对象的外接框，坐标是相对图片宽高的 0~1000 整数，左上角 (x1, y1)，右下角 (x2, y2)。框要完整包住整个对象。
        - remove 的 label 写那块区域里看到的东西（"checkerboard"、"sky"），框只框住那块区域。
        - 图中没有符合描述的对象时输出 {"subjects": [], "remove": []}。
        """
    }

    static func judgePrompt(subjects: String, n: Int) -> String {
        """
        第一张是原图。要抠出的主体是：\(subjects)。
        第二张是 \(n) 个编号候选区域（每格里亮的部分是候选，暗的部分不是）。
        逐个判断：候选亮的部分是否基本都落在上面这些主体身上（只是主体的一部分也算）？如果亮的部分明显包含主体以外的东西（家具、墙、地面、别的物体），就不要。
        只输出 JSON：{"keep": [编号...]}
        """
    }

    let apiKey: String
    let model: String
    let judgeModel: String
    let baseURL: URL
    let timeout: TimeInterval

    public init(apiKey: String, model: String = defaultModel, judgeModel: String = defaultJudgeModel, baseURL: URL = defaultBaseURL,
                timeout: TimeInterval = 60) {
        self.apiKey = apiKey
        self.model = model
        self.judgeModel = judgeModel
        self.baseURL = baseURL
        self.timeout = timeout
    }

    /// jpeg: the image (already scaled down for upload). width/height: the original image, for the pixel boxes.
    /// Blocks until the reply arrives (the worker processes one image at a time).
    public func detect(jpeg: Data, width: Int, height: Int, instruction: String?) throws -> QwenDetection {
        let prompt = instruction.map(Self.instructionPrompt) ?? Self.autoPrompt
        let content = try chat(model: model, images: [jpeg], text: prompt)
        return try Self.parseDetection(content, width: width, height: height)
    }

    /// Which of the numbered candidates (sheet: one tile each, 1...n) belong to the subjects. 0-based indices.
    public func judge(jpeg: Data, sheet: Data, subjects: [QwenSubject], count: Int, instruction: String?) throws -> [Int] {
        var names = subjects.map { s in "\(s.label)（框 [\(s.box.map { String(Int($0.rounded())) }.joined(separator: ", "))]）" }
            .joined(separator: "、")
        if let instruction { names += "（用户要求：\(instruction)）" }
        let content = try chat(model: judgeModel, images: [jpeg, sheet], text: Self.judgePrompt(subjects: names, n: count))
        return try Self.parseKeep(content, count: count)
    }

    func chat(model: String, images: [Data], text: String) throws -> String {
        let parts: [[String: Any]] = images.map { ["type": "image_url", "image_url": ["url": "data:image/jpeg;base64," + $0.base64EncodedString()]] }
            + [["type": "text", "text": text]]
        let body: [String: Any] = [
            "model": model,
            "messages": [["role": "user", "content": parts]],
            "temperature": 0,
            // Hybrid-thinking models would otherwise spend seconds reasoning about a bounding box.
            "enable_thinking": false,
        ]
        var request = URLRequest(url: baseURL.appendingPathComponent("chat/completions"), timeoutInterval: timeout)
        request.httpMethod = "POST"
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, status) = try Self.send(request)
        guard status == 200 else { throw Self.httpError(status, data) }
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let choices = obj["choices"] as? [[String: Any]],
              let message = choices.first?["message"] as? [String: Any],
              let content = message["content"] as? String
        else { throw QwenError("Qwen 接口返回格式异常：\(String(decoding: data.prefix(300), as: UTF8.self))") }
        return content
    }

    static func send(_ request: URLRequest) throws -> (Data, Int) {
        let done = DispatchSemaphore(value: 0)
        var result: Result<(Data, Int), Error> = .failure(QwenError("无法连接阿里云百炼"))
        URLSession.shared.dataTask(with: request) { data, response, error in
            if let error {
                result = .failure(QwenError("无法连接阿里云百炼：\(error.localizedDescription)"))
            } else {
                result = .success((data ?? Data(), (response as? HTTPURLResponse)?.statusCode ?? 0))
            }
            done.signal()
        }.resume()
        done.wait()
        return try result.get()
    }

    static func httpError(_ status: Int, _ data: Data) -> QwenError {
        let detail = String(decoding: data.prefix(300), as: UTF8.self)
        switch status {
        case 401, 403: return QwenError("阿里云百炼 API Key 无效或没有权限（\(status)），请在「设置」里检查")
        case 429: return QwenError("阿里云百炼请求太频繁或额度不足（429）：\(detail)")
        default: return QwenError("Qwen 接口返回 \(status)：\(detail)")
        }
    }

    /// The model's reply -> subjects with pixel boxes (see parseDetection).
    public static func parseSubjects(_ text: String, width: Int, height: Int) throws -> [QwenSubject] {
        try parseDetection(text, width: width, height: height).subjects
    }

    /// The model's reply -> subjects and remove regions with pixel boxes. Accepts {"subjects": [...], "remove": [...]}
    /// as asked, or the bare [{"bbox_2d": ..., "label": ...}] list of subjects some Qwen models fall back to; ```json
    /// fences, surrounding text, trailing commas and a box written without its brackets are ok.
    public static func parseDetection(_ text: String, width: Int, height: Int) throws -> QwenDetection {
        let value = try firstJSON(text)
        if let list = value as? [Any] { return QwenDetection(subjects: pixelBoxes(list, width: width, height: height)) }
        guard let dict = value as? [String: Any] else { return QwenDetection(subjects: []) }
        return QwenDetection(subjects: pixelBoxes(dict["subjects"] as? [Any] ?? [], width: width, height: height),
                             remove: pixelBoxes(dict["remove"] as? [Any] ?? [], width: width, height: height))
    }

    /// The judge's reply -> 0-based indices < count. Takes the last {"keep": [...]} (some models think aloud first).
    public static func parseKeep(_ text: String, count: Int) throws -> [Int] {
        let re = try! NSRegularExpression(pattern: #""keep"\s*:\s*\[([^\]]*)\]"#)
        let ns = text as NSString
        guard let m = re.matches(in: text, range: NSRange(location: 0, length: ns.length)).last else {
            throw QwenError("判断模型没有返回 keep：\(text.suffix(200))")
        }
        let nums = ns.substring(with: m.range(at: 1)).split(whereSeparator: { !$0.isNumber }).compactMap { Int($0) }
        return Array(Set(nums.filter { 1 <= $0 && $0 <= count }.map { $0 - 1 })).sorted()
    }

    static func firstJSON(_ text: String) throws -> Any {
        guard let start = text.firstIndex(where: { $0 == "{" || $0 == "[" }) else {
            throw QwenError("模型没有返回 JSON：\(text.prefix(200))")
        }
        let tail = String(text[start...])
        // Seen in replies: trailing commas, and a box written without its brackets ("box_2d": 37, 421, 251, 969).
        var fixed = tail.replacingOccurrences(of: #",(\s*[\]}])"#, with: "$1", options: .regularExpression)
        fixed = fixed.replacingOccurrences(of: #"("b(?:b)?ox(?:_2d)?"\s*:\s*)(-?[\d.]+\s*,\s*-?[\d.]+\s*,\s*-?[\d.]+\s*,\s*-?[\d.]+)"#,
                                           with: "$1[$2]", options: .regularExpression)
        for candidate in [tail, fixed] {
            if let json = firstJSONValue(candidate), let data = json.data(using: .utf8),
               let value = try? JSONSerialization.jsonObject(with: data) {
                return value
            }
        }
        throw QwenError("模型返回的 JSON 无法解析：\(tail.prefix(200))")
    }

    /// [{"label", "box_2d" in 0~1000}] -> boxes in pixels; entries without a label or a usable box are dropped.
    static func pixelBoxes(_ items: [Any], width: Int, height: Int) -> [QwenSubject] {
        var subjects: [QwenSubject] = []
        for case let s as [String: Any] in items {
            let label = (s["label"] as? String ?? "").trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            let raw = (s["box_2d"] ?? s["bbox_2d"] ?? s["box"]) as? [Any] ?? []
            let nums = raw.compactMap { ($0 as? NSNumber)?.doubleValue }
            guard !label.isEmpty, nums.count == 4 else { continue }
            let clamp = { (v: Double) in min(max(v, 0), 1000) }
            let x1 = clamp(min(nums[0], nums[2])), x2 = clamp(max(nums[0], nums[2]))
            let y1 = clamp(min(nums[1], nums[3])), y2 = clamp(max(nums[1], nums[3]))
            let px = [x1 / 1000 * Double(width), y1 / 1000 * Double(height), x2 / 1000 * Double(width), y2 / 1000 * Double(height)]
            if px[2] - px[0] >= 2, px[3] - px[1] >= 2 {
                subjects.append(QwenSubject(label: label, box: px))
            }
        }
        return subjects
    }

    /// The first complete JSON object/array at the start of `s` (bracket matching that respects strings).
    static func firstJSONValue(_ s: String) -> String? {
        var depth = 0
        var inString = false
        var escaped = false
        for (i, c) in s.enumerated() {
            if inString {
                if escaped { escaped = false } else if c == "\\" { escaped = true } else if c == "\"" { inString = false }
                continue
            }
            switch c {
            case "\"": inString = true
            case "{", "[": depth += 1
            case "}", "]":
                depth -= 1
                if depth == 0 { return String(s.prefix(i + 1)) }
            default: break
            }
        }
        return nil
    }
}
